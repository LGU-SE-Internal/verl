"""Lite scaffold prompt generator for verl.

Uses the same tool set as gen_prompt (edit_tool, search_tool, patch_submission)
for local execution, but reads from swe_dataset.py format (extra_info JSON string)
and includes docker_image in extra_info for ARL reward.
"""

import json

# Same system prompt as gen_prompt — references lite scaffold tools
system_prompt = """You are an expert AI software engineering agent. Your primary goal is to resolve a GitHub issue given in the user message. Following this workflow methodically:

1.  Understand the problem:\n    - Thoroughly comprehend the issue description, identifying core components and expected behavior\n    - Determine reproduction steps and failure conditions
2.  Explore and Locate:\n    - Use `search_tool` to explore the relevant files, entities, and test cases related to the bug report\n    - Locate the exact root cause of the bug
3.  Develop and Fix:\n    - Develop minimal, targeted code modifications to address the root cause\n    - Use `edit_tool` to apply surgical patch. Aim for minimal, clean changes
4.  Review and Revise:\n    - Review the original reproduction steps to ensure the fix effectiveness\n    - Review the relevant regression tests to avoid introducing any new bugs\n    - Iterate using `search_tool` for review and `edit_tool` for revise until you confirm no edge cases are overlooked
5.  Submit the patch:\n    - Call `patch_submission` tool to generate a unix diff patch and submit it to the user when confirming full resolution\n    - Ensure the final patch is non-empty before finishing this conversation\n    - All code changes persist throughout the conversation and will be included in the final patch
""".strip()


def _extract_fields(example):
    """Extract fields from either raw HuggingFace format or pre-processed (extra_info) format."""
    if "extra_info" in example and example["extra_info"] is not None:
        extra = json.loads(example["extra_info"]) if isinstance(example["extra_info"], str) else example["extra_info"]
    else:
        extra = dict(example)
    return extra


def process_fn(example, idx):
    """Convert a parquet row to verl format with lite scaffold.

    Handles both SWE-Bench and R2E-Gym data formats.
    Includes docker_image in extra_info for ARL reward.
    """
    extra = _extract_fields(example)

    # SWE-Bench: instance_id, base_commit, repo, problem_statement, docker_image
    # R2E-Gym: repo_name, commit_hash, problem_statement, docker_image
    instance_id = extra.get("instance_id") or ""
    problem_statement = extra.get("problem_statement", "")
    docker_image = extra.get("docker_image", "")

    # base_commit for SWE-Bench, commit_hash for R2E-Gym
    sha = extra.get("base_commit") or extra.get("commit_hash", "")
    repo_name = extra.get("repo") or extra.get("repo_name", "")

    # For R2E-Gym data without instance_id, construct one
    if not instance_id:
        instance_id = f"{repo_name}_{sha[:12]}" if repo_name else f"lite_{idx}"

    split = "train"

    # Preserve SWE-Bench harness fields if present (needed by swebench reward path)
    swebench_fields = {}
    for key in ["FAIL_TO_PASS", "PASS_TO_PASS", "repo", "version",
                "base_commit", "patch", "test_patch", "instance_id",
                "environment_setup_commit"]:
        if key in extra:
            swebench_fields[key] = extra[key]

    data = {
        "data_source": f"lite_{split}",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_statement},
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "split": split,
            "index": idx,
            "instance_id": instance_id,
            "issue": problem_statement,
            "need_tools_kwargs": True,
            # docker_image for ARL reward
            "docker_image": docker_image,
            "commit_hash": sha,
            "repo_name": repo_name,
            "expected_output_json": extra.get("expected_output_json", ""),
            # SWE-Bench harness fields
            **swebench_fields,
            # tools_kwargs for lite tools (edit_tool, search_tool, patch_submission)
            # NOTE: create_kwargs as JSON string to survive Arrow serialization
            "tools_kwargs": {
                "edit_tool": {
                    "create_kwargs": json.dumps({
                        "id": instance_id,
                        "sha": sha,
                    })
                },
                "search_tool": {
                    "create_kwargs": json.dumps({
                        "id": instance_id,
                        "sha": sha,
                    })
                },
                "patch_submission": {
                    "create_kwargs": json.dumps({
                        "id": instance_id,
                        "sha": sha,
                    })
                },
            },
        },
    }
    return data
