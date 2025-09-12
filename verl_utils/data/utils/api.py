import openai
import json

meta_config = json.load(open("config.json", "r")) # low effort
# meta_config = json.load(open("config_high.json", "r")) # high effort

def get_chat_completion(model: str, messages: list, tools: list = [], parallel_tool_calls: bool = False):
    try:
        config = meta_config[model]
        
        client = openai.AzureOpenAI(
            azure_endpoint=config['base_url'],
            api_version=config['api_version'],
            api_key=config['api_key']
        )
        
        completion = client.chat.completions.create(
            model=config['model'],
            temperature=config['temperature'],
            messages=messages,
            extra_headers={"X-TT-LOGID": ""},
            reasoning_effort=config.get("reasoning_effort", None),
            tools=tools,
            max_tokens=config['max_tokens'],
        )

        # print(completion.choices) # for debug

        response = completion.choices[-1].message.content
        tokens = completion.usage.total_tokens
        finish_reason = completion.choices[-1].finish_reason
        messages = list(messages)

        # for tool call
        if finish_reason == "tool_calls" and completion.choices[-1].message.tool_calls:
            tool_calls = []
            for tool_call in completion.choices[-1].message.tool_calls:
                tool_calls.append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    }
                })
            messages.append({
                "role": "assistant",
                "content": response,
                "tool_calls": tool_calls
            })
        
        # for normal completion 
        else:
            messages.append({
                "role": "assistant",
                "content": response,
                "tool_calls": None
            })
        
        return messages, tokens, finish_reason
        
    except Exception as e:
        print(f"API REQUEST ERROR: {str(e)}")
        return messages, 0, 'stop'

if __name__ == "__main__":
    message = [
        {"role": "user", "content": "123+321=?"},
    ]
    history, tokens, finish_reason = get_chat_completion("gpt4o", message)
    print(history)
    print(tokens)
    print(finish_reason)
    history, tokens, finish_reason  = get_chat_completion("o4mini", message)
    print(history)
    print(tokens)
    print(finish_reason)

    import yaml
    tool_config = [yaml.load(open("verl_utils/tool/config/tool_config/agentic_tool_config.yaml", 'r'), Loader=yaml.FullLoader)['tools'][0]['tool_schema']]
    print(json.dumps(tool_config, indent=4))
    import pandas as pd
    df = pd.read_parquet('data/data_oracle.parquet')
    prompt = df.iloc[0]['prompt']
    history, tokens, finish_reason = get_chat_completion("gpt4o", prompt, tool_config)
    print(history)
    print(tokens)
    print(finish_reason)
