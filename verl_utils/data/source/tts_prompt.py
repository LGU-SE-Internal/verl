# w/o tool
system_prompt = """You are a software expert. You will be given a software issue and some patch candidates in user query. You need to judge which patch(es) can resolve the issue. Carefully review, critic, and compare the given candidates. You need to first think about the reasoning process in the mind until you get the final answer. Finally, place **only one** ID of the most likely correct patch candidate within \\boxed{}, i.e., \\boxed{1} or \\boxed{2} or \\boxed{3} or \\boxed{4}.
""".strip()

# Finally, place **only one** ID of the most likely correct patch candidate within \\boxed{}, i.e., \\boxed{1} or \\boxed{2} or \\boxed{3} or \\boxed{4}.
# Finally, put the ID(s) of correct patch candidates within \\boxed{}, e.g., \\boxed{1}, \\boxed{2, 4}, \\boxed{1, 2, 3, 4} (all correct), \\boxed{} (all wrong).

def process_fn(example, idx):

    id = example.pop("instance_id") # unique id
    issue = example.pop("problem_statement") # original issue
    patch_list = example.pop("patch") # patches
    resolved = example.pop("resolved") # true or false
    split = example.pop("split") # train, test

    plist = []
    rlist = []
    for i in range(len(patch_list)):
        if patch_list[i] != '# NO PATCH GENERATED!':
            plist.append(patch_list[i])
            rlist.append(bool(resolved[i]))
    if len(plist) < 4: # auto completion
        delta = 4 - len(plist)
        for i in range(delta):
            plist.append("# NO PATCH GENERATED!")
            rlist.append(False)

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
                    f"<issue>\n{issue.strip()}\n</issue>\n"
                    f"<patch-1>\n{plist[0].strip()}\n</patch-1>\n"
                    f"<patch-2>\n{plist[1].strip()}\n</patch-2>\n"
                    f"<patch-3>\n{plist[2].strip()}\n</patch-3>\n"
                    f"<patch-4>\n{plist[3].strip()}\n</patch-4>\n"
                ),
            },
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": rlist},
        "extra_info": {
            "split": split,
            "index": idx,
            "id": id,
        },
    }
    return data