system_prompt = """You are an expert AI software engineering agent. Your primary goal is to resolve a GitHub issue given in the user message. Following this workflow methodically:

1.  Understand the problem:\n    - Thoroughly comprehend the issue description, identifying core components and expected behavior\n    - Determine reproduction steps and failure conditions
2.  Explore and Locate:\n    - Use `search_tool` to explore the relevant files, entities, and test cases related to the bug report\n    - Locate the exact root cause of the bug
3.  Develop and Fix:\n    - Develop minimal, targeted code modifications to address the root cause\n    - Use `edit_tool` to apply surgical patch. Aim for minimal, clean changes
4.  Review and Revise:\n    - Review the original reproduction steps to ensure the fix effectiveness\n    - Review the relevant regression tests to avoid introducing any new bugs\n    - Iterate using `search_tool` for review and `edit_tool` for revise until you confirm no edge cases are overlooked
5.  Submit the patch:\n    - Call `patch_submission` tool to generate a unix diff patch and submit it to the user when confirming full resolution\n    - Ensure the final patch is non-empty before finishing this conversation\n    - All code changes persist throughout the conversation and will be included in the final patch
""".strip()

# system_prompt = """You are an expert AI software engineering agent. Your primary goal is to resolve a GitHub issue given in the user message. Following this workflow methodically:

# 1.  Understand the problem:\n    - Thoroughly comprehend the issue description, identifying core components and expected behavior\n    - Determine reproduction steps and failure conditions
# 2.  Explore and Locate:\n    - Use `search_tool` to explore the relevant files, entities, and test cases related to the bug report\n    - Locate the exact root cause of the bug
# 3.  Develop and Fix:\n    - Develop minimal, targeted code modifications to address the root cause\n    - Use `edit_tool` to apply surgical patch. Aim for minimal, clean changes
# 4.  Review and Revise:\n    - Review the original reproduction steps to ensure the fix effectiveness\n    - Review the relevant regression tests to avoid introducing any new bugs\n    - Iterate using `search_tool` for review and `edit_tool` for revise until you confirm no edge cases are overlooked
# 5.  Submit the patch:\n    - Call `patch_submission` tool to generate a unix diff patch and submit it to the user when confirming full resolution\n    - Ensure the final patch is non-empty before finishing this conversation\n    - All code changes persist throughout the conversation and will be included in the final patch

# You have access to tools to assist with the issue resolving. The tool configs are shown in JSON Schema format:

# ```json
# [{'type': 'function', 'function': {'name': 'edit_tool', 'description': 'A file edit tool to replace the content between `start_line` and `end_line` with a `new_str` in a file.\\nNotes:\\n1. The content at `start_line` and `end_line` is also included in the replacement.\\n2. `start_line` must not exceed `end_line`\\n3. `edit_tool` **permanently modifies the actual repository** (changes persist to the final outcome).', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': 'Relative file path e.g. `dir/file.py`'}, 'start_line': {'type': 'string', 'description': 'Start line number.'}, 'end_line': {'type': 'string', 'description': 'End line number'}, 'new_str': {'type': 'string', 'description': 'Replacement string'}}, 'required': ['path', 'start_line', 'end_line', 'new_str']}}}, {'type': 'function', 'function': {'name': 'search_tool', 'description': "A code graph tool to show the source code of a function/class/class_method.\\nNotes:\\n1. If multiple matches are found, all of them will be returned.\\n2. `search_tool` always reflects the **initial repository snapshot** (code state at the start of the conversation).\\n3. Only python entities are available.", 'parameters': {'type': 'object', 'properties': {'construct': {'type': 'string', 'description': 'Type of program construct', 'enum': ['function', 'class', 'class_method']}, 'entity': {'type': 'string', 'description': 'Simple entity name (e.g., `foo`)'}}, 'required': ['construct', 'entity']}}}, {'type': 'function', 'function': {'name': 'patch_submission', 'description': 'Automatically generate a diff patch for existing code changes and submit it to the user.', 'parameters': {'type': 'object', 'properties': {}}}}]
# ```

# For each function call, you should generate a json object with function name and arguments within <tool_call></tool_call> XML tags:

# <tool_call>
# {"name": <function-name>, "arguments": <args-json-object>}
# </tool_call>

# The result of a function call will be given back to you as a new user message after execution, and you can continue to call function. Only **one** tool may be invoked at a time. Overlong tool response will be truncated.
# """.strip()


# ```json
# [{'type': 'function', 'function': {'name': 'edit_tool', 'description': 'A file edit tool to replace the content between `start_line` and `end_line` with a `new_str` in a file.\\nNotes:\\n1. The content at `start_line` and `end_line` is also included in the replacement.\\n2. `start_line` must not exceed `end_line`\\n3. `edit_tool` **permanently modifies the actual repository** (changes persist to the final outcome).', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': 'Relative file path e.g. `dir/file.py`'}, 'start_line': {'type': 'string', 'description': 'Start line number.'}, 'end_line': {'type': 'string', 'description': 'End line number'}, 'new_str': {'type': 'string', 'description': 'Replacement string'}}, 'required': ['path', 'start_line', 'end_line', 'new_str']}}}, {'type': 'function', 'function': {'name': 'search_tool', 'description': "A code graph tool to show either (1) the source code of a function/class/class_method, (2) a file's skeleton structure, or (3) a directory tree structure.\\nNotes:\\n1. If multiple matches are found, all of them will be returned.\\n2. `search_tool` always reflects the **initial repository snapshot** (code state at the start of the conversation).\\n3. Only python entities or '.py' files are available.", 'parameters': {'type': 'object', 'properties': {'construct': {'type': 'string', 'description': 'Type of program construct', 'enum': ['function', 'class', 'class_method', 'directory', 'file']}, 'entity': {'type': 'string', 'description': 'Simple entity name (e.g., `foo`) or relative path (e.g., `.`, `dir/sub_dir` or `dir/sub_dir/file.py`)'}}, 'required': ['construct', 'entity']}}}, {'type': 'function', 'function': {'name': 'patch_submission', 'description': 'Automatically generate a diff patch for existing code changes and submit it to the user.', 'parameters': {'type': 'object', 'properties': {}}}}]
# ```

def process_fn(example, idx):

    id = example.pop("instance_id") # unique id
    issue = example.pop("problem_statement") # original issue
    sha = example.pop("base_commit") # sha of the base commit
    # split = example.pop("split") # train, test
    split = 'random'

    data = {
        "data_source": f'gen_{split}',
        "prompt": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": issue,
            },
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "split": split,
            "index": idx,
            "instance_id": id,
            "issue": issue,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "edit_tool": {
                    "create_kwargs": {
                        "id": id, 
                        "sha": sha,
                    }
                },
                "search_tool": {
                    "create_kwargs": {
                        "id": id, 
                        "sha": sha,
                    },
                },
                "patch_submission": {
                    "create_kwargs": {
                        "id": id, 
                        "sha": sha,
                    }
                }
            }
            # "tools_kwargs": {
            #     "instance_id": id,
            #     "base_commit": sha,
            # },
        },
    }
    return data