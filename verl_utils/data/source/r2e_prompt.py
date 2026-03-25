"""R2E-Gym prompt generator for verl.

Reads R2E-Gym parquet files (from rllm/trace_analyzer data format)
and produces verl-compatible training rows with tools_kwargs for
ARL-backed r2egym scaffold tools.
"""

import json

import pandas as pd

system_prompt = """You are an expert AI software engineering agent. Your primary goal is to resolve a GitHub issue given in the user message. Follow this workflow:

1.  **Understand the problem**: Thoroughly comprehend the issue description, identifying core components and expected behavior. Determine reproduction steps and failure conditions.
2.  **Explore and Locate**: Use `search` to find relevant files and code. Use `file_editor` with `view` command to read files. Use `execute_bash` to explore the repository structure. Locate the exact root cause of the bug.
3.  **Develop and Fix**: Develop minimal, targeted code modifications to address the root cause. Use `file_editor` with `str_replace` command to apply surgical patches. Aim for minimal, clean changes.
4.  **Review and Verify**: Use `execute_bash` to run relevant tests and verify your fix. Review regression tests to avoid introducing new bugs. Iterate until you confirm no edge cases are overlooked.
5.  **Submit**: Call `finish` to submit your solution when you are confident the issue is resolved.

Important notes:
- The repository is located at `/testbed` as the current working directory
- You have access to: `file_editor`, `search`, `execute_bash`, and `finish` tools
- Make minimal changes — do not refactor unrelated code
- Always verify your fix by running tests before finishing""".strip()


def _extract_fields(example):
    """Extract fields from either raw HuggingFace format or pre-processed (extra_info) format.

    Raw HuggingFace format (flat columns):
        repo, instance_id, base_commit, docker_image, problem_statement, ...
    Pre-processed format (rllm pipeline):
        extra_info (JSON string or dict) containing all fields
    """
    if "extra_info" in example and example["extra_info"] is not None:
        extra = json.loads(example["extra_info"]) if isinstance(example["extra_info"], str) else example["extra_info"]
    else:
        # Raw HuggingFace dataset — fields are top-level columns
        extra = dict(example)
    return extra


def process_fn(example, idx):
    """Convert an R2E-Gym or SWE-Bench parquet row to verl training format.

    Handles both data sources and both raw/pre-processed formats:
    - R2E-Gym: has repo_name, commit_hash, expected_output_json
    - SWE-Bench Verified: has repo, instance_id, base_commit, FAIL_TO_PASS, PASS_TO_PASS
    """
    extra = _extract_fields(example)

    docker_image = extra.get("docker_image", "")
    # R2E-Gym uses commit_hash, SWE-Bench uses base_commit
    commit_hash = extra.get("commit_hash") or extra.get("base_commit", "")
    # R2E-Gym uses repo_name, SWE-Bench uses repo
    repo_name = extra.get("repo_name") or extra.get("repo", "")
    problem_statement = extra.get("problem_statement", "")
    expected_output_json = extra.get("expected_output_json", "")

    # SWE-Bench has instance_id directly; R2E-Gym needs to construct one
    instance_id = extra.get("instance_id") or (
        f"{repo_name}_{commit_hash[:12]}" if repo_name else f"r2e_{idx}"
    )
    # Infer split from the data_source set during swe_dataset.py
    # r2e_train → "train", r2e_test / r2e_test_hard → "test"
    raw_data_source = example.get("data_source", "")
    split = "test" if "test" in raw_data_source else "train"

    # Preserve SWE-Bench harness fields if present (needed by swebench reward)
    swebench_fields = {}
    for key in ["FAIL_TO_PASS", "PASS_TO_PASS", "repo", "version",
                "base_commit", "patch", "test_patch", "instance_id",
                "environment_setup_commit"]:
        if key in extra:
            swebench_fields[key] = extra[key]

    data = {
        "data_source": f"r2e_{split}",
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
            # Fields needed by ARL reward and tool creation
            "docker_image": docker_image,
            "commit_hash": commit_hash,
            "repo_name": repo_name,
            "expected_output_json": expected_output_json,
            # SWE-Bench harness fields (for swebench reward path)
            **swebench_fields,
            # tools_kwargs: passed to BaseTool.create() via ToolAgentLoop
            # NOTE: create_kwargs stored as JSON string to avoid HuggingFace
            # datasets Arrow serialization dropping keys from nested dicts.
            "tools_kwargs": {
                "file_editor": {
                    "create_kwargs": json.dumps({
                        "docker_image": docker_image,
                        "commit_hash": commit_hash,
                    })
                },
            },
        },
    }
    return data


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Convert R2E-Gym parquet to verl format")
    parser.add_argument("input", help="Input R2E-Gym parquet file")
    parser.add_argument("output", help="Output verl-format parquet file")
    parser.add_argument("--split", default="train", help="Split name (train/test)")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    rows = []
    for idx in range(len(df)):
        row = df.iloc[idx].to_dict()
        converted = process_fn(row, idx)
        # Override split if specified
        converted["extra_info"]["split"] = args.split
        converted["data_source"] = f"r2e_{args.split}"
        rows.append(converted)

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")
