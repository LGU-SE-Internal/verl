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
from verl_utils.eval.result_evaluator import evaluate_batch, evaluate_naive, extract_batch_combine
from verl_utils.tool.function.get_func_from_context import get_impl_and_deps

def get_tool_config(path: str):
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)['tools']
    tool_config = [tool_config['tool_schema'] for tool_config in config]
    return tool_config

def loop(model: str, messages: list, tools: list, tree: dict, max_rounds=25, max_tokens=32768):
    tokens = 0
    rounds = 0
    while tokens < max_tokens and rounds < max_rounds:
        messages, tokens, finish_reason = get_chat_completion(model, messages, tools)
        rounds += 1
        if finish_reason == 'stop' or finish_reason == 'length':
            return messages, tokens, finish_reason
        else:
            tool_calls = messages[-1]['tool_calls']
            for tool_call in tool_calls:
                function_name = tool_call['function']['name']
                function_arguments = json.loads(tool_call['function']['arguments'])
                if function_name == 'get_code_of_methods': # only implemented for get_code_of_methods
                    result, _, _ = get_impl_and_deps(**function_arguments, tree=tree)
                else:
                    print(f"TOOL CALL ERROR: {function_name} is not implemented.")
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call['id'],
                    'name': function_name,
                    'content': result
                })
    return messages, tokens, finish_reason

def rollout(model, data_path, output_path, tool_path=None, context_path=None, total_parts=1, part_idx=0):
    data = pd.read_parquet(data_path)
    
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
        msg_df['is_invalid'] = msg_df.apply(lambda x: extract_batch_combine(x['messages'][-1]['content']) is None, axis=1)
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
        label = row['reward_model']['ground_truth']
        original_messages = deepcopy(messages)
        if tool_path and context_path:
            tool = get_tool_config(tool_path)
            with open(f"{context_path}/{row['extra_info']['id']}.json", "r") as f:
                tree = json.load(f)
            messages, tokens, _ = loop(model, original_messages, tool, tree)
        elif not tool_path and not context_path:
            messages, tokens, _ = get_chat_completion(model, original_messages)
        else:
            raise ValueError('tool_path and context_path are not aligned.')
        new_row = pd.DataFrame([{
            "idx": idx,
            "id": row['extra_info']['id'],
            "messages": messages,
            "ground_truth": label,
            "total_tokens": tokens
        }])
        msg_df = pd.concat([msg_df, new_row], ignore_index=True)
        if idx % 10 == 0: # save per 10 instance
            msg_df.to_parquet(output_path)
    msg_df.to_parquet(output_path)
    print(f"Rollout finished. Output path: {output_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="claude37", help="model name")
    parser.add_argument("--data_path", type=str, default="data/data_test_batch_without_tool.parquet", help="data path")
    parser.add_argument("--context_path", type=str, default="data/context", help="context path")
    parser.add_argument("--tool_config_path", type=str, default="verl_utils/tool/config/tool_config/agentic_tool_config.yaml", help="tool config path")
    parser.add_argument("--tooluse", type=bool, default=False, help="output path")
    parser.add_argument("--naive", type=bool, default=False, help="use batch or naive")
    parser.add_argument("--distill", type=bool, default=False, help="for distillation")
    parser.add_argument("--total_parts", type=int, default=1, help="Total number of parts to split the data into")
    parser.add_argument("--part_idx", type=int, default=0, help="Index of the part to process (0-based)")
    
    args = parser.parse_args()

    if args.tooluse:
        output_path = f'data/test_{args.model}_{args.tag}.parquet'
        rollout(args.model, args.data_path, output_path, args.tool_config_path, args.context_path, args.total_parts, args.part_idx)
    elif args.naive:
        output_path = f'data/test_without_tool_{args.model}_naive.parquet'
        rollout(args.model, args.data_path, output_path, total_parts=args.total_parts, part_idx=args.part_idx)
    elif args.distill:
        output_path = f'data/distill_rollouts_{args.model}.parquet'
        rollout(args.model, args.data_path, output_path, total_parts=args.total_parts, part_idx=args.part_idx)
    else:
        output_path = f'data/test_without_tool_{args.model}_batch.parquet'
        rollout(args.model, args.data_path, output_path, total_parts=args.total_parts, part_idx=args.part_idx)
        
    if args.total_parts == 1:
        if args.naive:
            from verl_utils.reward.extract_answer import extract_answer_naive
            evaluate_naive(output_path, extract_answer_naive)
        else:
            evaluate_batch(output_path)