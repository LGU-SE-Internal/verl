# for vllm tooluse
system_prompt = """
You are a software expert. You will be given a software issue and some patch candidates in user query. You need to judge which patch(es) can resolve the issue. You have access to code search tools to assist with the judgement. The function signatures are within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_code_of_methods", "description": "A tool that outputs the source code of the input method or function.", "parameters": {"type": "object", "properties": {"method_name": {"type": "string", "description": "Full qualified name of a method or function", "enum": null}}, "required": ["method_name"]}, "strict": false}}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
In your response, you need to first carefully review, critic, and compare the given candidates in the mind, and then conduct function calling to get the relevant code context if you feel current information is not sufficient. The results of previous function calls will be given back to you in the next round, and you can continue to call functions until you get enough information. Finally, put the ID(s) of correct patch candidates within \\boxed{}, e.g., \\boxed{1}, \\boxed{2, 4}, \\boxed{1, 2, 3, 4} (all correct), \\boxed{} (all wrong).
""".strip()

def process_fn(example, idx):

    alpha = 0.5
    id = example.pop("instance_id") # unique id
    repo = example.pop("repo") # repo name
    issue = example.pop("problem_statement") # original issue
    patch_list = example.pop("patch") # patches
    resolved = example.pop("resolved") # true or false
    sha = example.pop("base_commit") # sha of the base commit
    split = example.pop("split") # train, test
    loc = example.pop("oracle_location") # loc

    data = {
        "data_source": f'batch_{split}',
        "prompt": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Here is an issue and four patches trying to resolve the issue:\n"
                    f"<issue>\n{issue.strip()}\n</issue>\n"
                    f"<patch-1>\n{patch_list[0].strip()}\n</patch-1>\n"
                    f"<patch-2>\n{patch_list[1].strip()}\n</patch-2>\n"
                    f"<patch-3>\n{patch_list[2].strip()}\n</patch-3>\n"
                    f"<patch-4>\n{patch_list[3].strip()}\n</patch-4>\n"
                    "Which patch(es) can resolve the issue?"
                ),
            },
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": resolved},
        "extra_info": {
            "split": split,
            "index": idx,
            "id": id,
            "repo": repo,
            "sha": sha,
            "alpha": alpha,
            "loc": loc,
            "need_tools_kwargs": True, # for multi-turn
            "tools_kwargs": {
                "get_code_of_methods": {
                    "create_kwargs": {"issue_id": id},
                },
            },
        },
    }
    return data