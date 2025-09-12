system_prompt = """You are an expert AI software engineering agent. Your primary goal is to verify whether the given patch resolves the GitHub issue in the user message. Following this workflow methodically:

1.  Understand the problem:\n    - Thoroughly comprehend the issue description, identifying core components and expected behavior\n    - Determine reproduction steps and failure conditions
2.  Explore and Locate:\n    - Use `search_tool` to explore the relevant files, entities, and test cases related to the bug report\n    - Locate the exact root cause of the bug
3.  Review and Judge:\n    - Review the original reproduction steps to judge the patch effectiveness\n    - Review the relevant regression tests to check whether introducing any new bugs\n    - Iterate using `search_tool` for review until you confirm no edge cases are overlooked
4.  Stop the process:\n    - Finally, put the ID(s) of correct patch candidates within \\boxed{}, e.g., \\boxed{1}, \\boxed{2, 4}, \\boxed{1, 2, 3, 4} (all correct), \\boxed{} (all wrong)\n    - DO NOT make any judgement or assumption before you have gotten sufficient context information from the `search_tool` responses.
""".strip()

# system_prompt = """You are an expert AI software engineering agent. Your primary goal is to verify whether the given patch resolves the GitHub issue in the user message. Following this workflow methodically:

# 1.  Understand the problem:\n    - Thoroughly comprehend the issue description, identifying core components and expected behavior\n    - Determine reproduction steps and failure conditions
# 2.  Explore and Locate:\n    - Use `search_tool` to explore the relevant files, entities, and test cases related to the bug report\n    - Locate the exact root cause of the bug
# 3.  Review and Judge:\n    - Review the original reproduction steps to judge the patch effectiveness\n    - Review the relevant regression tests to check whether introducing any new bugs\n    - Iterate using `search_tool` for review until you confirm no edge cases are overlooked
# 4.  Stop the process:\n    - Finally, put the ID(s) of correct patch candidates within \\boxed{}, e.g., \\boxed{1}, \\boxed{2, 4}, \\boxed{1, 2, 3, 4} (all correct), \\boxed{} (all wrong)\n    - DO NOT make any judgement or assumption before you have gotten sufficient context information from the `search_tool` responses.

# You have access to tools to assist with the issue resolving. The tool configs are shown in JSON Schema format:

# ```json
# [{'type': 'function', 'function': {'name': 'search_tool', 'description': "A code graph tool to show the source code of a function/class/class_method.\\nNotes:\\n1. If multiple matches are found, all of them will be returned.\\n2. `search_tool` always reflects the **initial repository snapshot** (code state at the start of the conversation).\\n3. Only python entities are available.", 'parameters': {'type': 'object', 'properties': {'construct': {'type': 'string', 'description': 'Type of program construct', 'enum': ['function', 'class', 'class_method']}, 'entity': {'type': 'string', 'description': 'Simple entity name (e.g., `foo`)'}}, 'required': ['construct', 'entity']}}}]
# ```

# For each function call, you should generate a json object with function name and arguments within <tool_call></tool_call> XML tags:

# <tool_call>
# {"name": <function-name>, "arguments": <args-json-object>}
# </tool_call>

# The result of a function call will be given back to you as a new user message after execution, and you can continue to call function. Only **one** tool may be invoked at a time. Overlong tool response will be truncated.
# """.strip()

# ```json
# [{'type': 'function', 'function': {'name': 'search_tool', 'description': "A code graph tool to show either (1) the source code of a function/class/class_method, (2) a file's skeleton structure, or (3) a directory tree structure.\\nNotes:\\n1. If multiple matches are found, all of them will be returned.\\n2. `search_tool` always reflects the **initial repository snapshot** (code state before applying the patch).\\n3. Only python entities or '.py' files are available.", 'parameters': {'type': 'object', 'properties': {'construct': {'type': 'string', 'description': 'Type of program construct', 'enum': ['function', 'class', 'class_method', 'directory', 'file']}, 'entity': {'type': 'string', 'description': 'Simple entity name (e.g., `foo`) or relative path (e.g., `.`, `dir/sub_dir` or `dir/sub_dir/file.py`)'}}, 'required': ['construct', 'entity']}}}]
# ```

def process_fn(example, idx):

    id = example.pop("instance_id") # unique id
    issue = example.pop("problem_statement") # original issue
    patch = example.pop("patch") # user patch
    resolved = example.pop("resolved") # flag
    sha = example.pop("base_commit") # sha of the base commit
    split = example.pop("split") # train, test

    data = {
        "data_source": f'ver_{split}',
        "prompt": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    '[ORIGINAL ISSUE]\n'
                    f'{issue.strip()}\n\n'
                    '[CANDIDATE PATCH 1]\n'
                    f'{patch[0].strip()}\n'
                    '[CANDIDATE PATCH 2]\n'
                    f'{patch[1].strip()}\n'
                    '[CANDIDATE PATCH 3]\n'
                    f'{patch[2].strip()}\n'
                    '[CANDIDATE PATCH 4]\n'
                    f'{patch[3].strip()}\n'
                    'Please reason about the correctness of the patches above. Use the `search_tool` to get more context. Finally, put the number of correct patches within \\boxed{}.'
                )
            },
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": resolved},
        "extra_info": {
            "split": split,
            "index": idx,
            "instance_id": id,
            "issue": issue,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "search_tool": {
                    "create_kwargs": {
                        "id": id, 
                        "sha": sha,
                    },
                }
            }
            # "tools_kwargs": {
            #     "instance_id": id,
            #     "base_commit": sha,
            # },
        },
    }
    return data