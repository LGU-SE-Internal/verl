### w/o tool
system_prompt = """ 
You are a software expert in patch review. You will be given a GitHub issue and a patch. You need to determine whether the patch truly resolves the issue. Carefully review, critic, and compare the given candidates. You need to first think about the reasoning process in the mind until you get the final answer. Finally, put the final answer ('True' or 'False') within \\boxed{}
""".strip()

def process_fn(example, idx):

    id = example.pop("instance_id") # unique id
    repo = example.pop("repo") # repo name
    issue = example.pop("problem_statement") # original issue
    patch = example.pop("patch") # patch with full method context, FQN, and import packages
    resolved = example.pop("resolved") # true or false
    sha = example.pop("base_commit") # sha of the base commit
    split = example.pop("split") # train, test
    gt_loc = example.pop("oracle_location") # ground truth location of the edit hunk
    # patch_import = example.pop('file_pkg') # import packages in the patch


    data = {
        "data_source": f'naive_{split}',
        "prompt": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"<issue>\n{issue.strip()}\n</issue>\n"
                    f"<patch>\n{patch.strip()}\n</patch>"
                    "Does the patch resolve the issue?"
                ),
            },
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": resolved},
        "extra_info": {
            "split": split,
            "index": idx,
            "id": id,
        }
    }
    return data