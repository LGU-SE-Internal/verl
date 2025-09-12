import pandas as pd
import argparse
from tqdm import tqdm
import sys
import os
import json
from copy import deepcopy
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from verl_utils.data.utils.api import get_chat_completion
from verl_utils.tool.edit_tool import EditTool
from verl_utils.tool.search_tool import SearchTool
from verl_utils.reward.extract_answer import extract_answer_naive

def get_tool_config(path: str):
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)['tools']
    tool_config = [tool_config['tool_schema'] for tool_config in config]
    return tool_config

def loop(model: str, messages: list, root:str, tool_config:dict, tools_kwargs:dict, max_rounds=20, max_tokens=163840):
    tokens = 0
    rounds = 0
    judge = None
    search_tool = SearchTool(root, tools_kwargs['instance_id'])
    # print(f"Processing instance: {tools_kwargs['instance_id']}") # for debug
    while tokens < max_tokens and rounds < max_rounds:
        messages[-1]['content'] += f'\n\n[ROOT]: You have to finish this task within {max_rounds} turns. This is {rounds+1} turn. Only {max_rounds-rounds-1} left!'
        messages, tokens, finish_reason = get_chat_completion(model, messages, tool_config)
        rounds += 1
        if finish_reason == 'stop' or finish_reason == 'length':
            judge = extract_answer_naive(messages[-1]['content'])
            return messages, tokens, finish_reason, judge
        elif finish_reason == 'tool_calls':
            tool_calls = messages[-1]['tool_calls']
            for tool_call in tool_calls:
                function_name = tool_call['function']['name']
                function_args = tool_call['function'].get('arguments', None)
                args_dict = json.loads(function_args) if function_args else None
                if function_name == 'search_tool': 
                    construct = args_dict.get("construct", None)
                    entity = args_dict.get("entity", None)
                    result = search_tool.execute(construct, entity)
                else:
                    print(f"TOOL CALL ERROR: {function_name} is not implemented.")
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call['id'],
                    'name': function_name,
                    'content': result
                })
                # print()
                # print(result) # for debug
                # print('-'*80)
        else:
            print(f"Unknown status: {finish_reason}")
            # exit()
            break
    return messages, tokens, finish_reason, judge

def ver(model, root, split, output_path, tool_config, total_parts=1, part_idx=0):
    if split == 'train':
        data = pd.read_parquet(f'{root}/data_train_ver.parquet')
        max_rounds = 10
    else:
        data = pd.read_parquet(f'{root}/data_test_ver.parquet')
        max_rounds = 20
    
    # Split data into parts if needed
    if total_parts > 1:
        part_size = len(data) // total_parts
        start_idx = part_idx * part_size
        end_idx = (part_idx + 1) * part_size if part_idx < total_parts - 1 else len(data)
        data = data.iloc[start_idx:end_idx]
        # Modify output path to include part info
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{part_idx}-{total_parts}{ext}"
    
    if os.path.exists(output_path):
        msg_df = pd.read_parquet(output_path)
        processed_idx = set(msg_df['idx'].tolist())
        msg_df['is_invalid'] = msg_df['label'].isna()
        invalid_idx = set(msg_df.loc[msg_df['is_invalid'], 'idx'])
        processed_idx = set(msg_df['idx'].tolist()) - invalid_idx
        msg_df = msg_df[~msg_df['is_invalid']].drop(columns=['is_invalid'])
        print(f"Remove {len(invalid_idx)} invalid lines.")
    else:
        msg_df = pd.DataFrame()
        processed_idx = set()
        
    for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Part {part_idx}/{total_parts}"):
        if idx in processed_idx:
            continue
        messages = row['prompt']
        tools_kwargs = row['extra_info']['tools_kwargs']
        original_messages = deepcopy(messages)
        original_messages[0]['content'] = original_messages[0]['content'].split('You have access to tools to assist with the issue resolving.')[0].strip() # remove tool instructions for rollouts
        messages, tokens, _, judge = loop(model, original_messages, root, tool_config, tools_kwargs, max_rounds)
        if isinstance(judge, str):
            label = (eval(judge) == row['reward_model']['ground_truth'])
        else:
            label = None
        new_row = pd.DataFrame([{
            "idx": idx,
            "id": row['extra_info']['instance_id'],
            "judge": judge,
            "label": label,
            "messages": messages,
            "total_tokens": tokens
        }])
        # print(messages) # for debug
        # print('-'*80)
        # print("### THE JUDGE IS SHOWN:")
        # print(judge)
        # print("### THE ORACLE IS SHOWN:")
        # print(row['reward_model']['ground_truth'])
        # print("### THE LABEL CORRECTNESS:")
        # print(new_row.iloc[0]['label'])
        # exit()
        msg_df = pd.concat([msg_df, new_row], ignore_index=True)
        if idx % 10 == 0: # save per 10 instance
            msg_df.to_parquet(output_path)
    msg_df.to_parquet(output_path)
    print(f"Rollout finished. Output path: {output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="claude37", help="model name")
    parser.add_argument("--root", type=str, default="data/datasets", help="data path")
    parser.add_argument("--split", type=str, default="test", help="data type")
    parser.add_argument("--tool_config_path", type=str, default="verl_utils/tool/config/tool_config/trav_tool_config.yaml", help="tool config path")
    parser.add_argument("--total_parts", type=int, default=1, help="Total number of parts to split the data into")
    parser.add_argument("--part_idx", type=int, default=0, help="Index of the part to process (0-based)")
    
    args = parser.parse_args()

    output_path = f'{args.root}/ver_{args.model}_{args.split}.parquet'
    tool_config = get_tool_config(args.tool_config_path)
    ver(args.model, args.root, args.split, output_path, tool_config, total_parts=args.total_parts, part_idx=args.part_idx)