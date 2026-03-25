"""Download R2E-Gym and SWE-Bench Verified from HuggingFace, save as verl-compatible parquet.

Usage:
    python verl_utils/data/swe_dataset.py --local_dir data/r2e

Output:
    data/r2e/info_r2e_train.parquet       (R2E-Gym Subset, for training)
    data/r2e/info_r2e_test.parquet        (SWE-Bench Verified, for validation)
    data/r2e/info_r2e_test_hard.parquet   (SWE-Bench Verified hard subset, for validation)

Then convert to training format:
    python verl_utils/data/data_process.py --data_source r2e_train --file_path data/r2e/info_r2e_train.parquet
    python verl_utils/data/data_process.py --data_source r2e_test  --file_path data/r2e/info_r2e_test.parquet
    python verl_utils/data/data_process.py --data_source r2e_test  --file_path data/r2e/info_r2e_test_hard.parquet
"""

import argparse
import json
import os

import pandas as pd
from datasets import load_dataset


DATASETS = {
    "R2E-Gym/R2E-Gym-Subset": ("info_r2e_train.parquet", "r2e_train"),
    "R2E-Gym/SWE-Bench-Verified": ("info_r2e_test.parquet", "r2e_test"),
}

HARD_DIFFICULTIES = {"1-4 hours", ">4 hours"}


def _get_hard_instance_ids():
    """Get instance_ids of hard problems from the original SWE-bench/SWE-bench_Verified dataset."""
    print("Downloading SWE-bench/SWE-bench_Verified for difficulty labels ...")
    ds = load_dataset("SWE-bench/SWE-bench_Verified")
    split = ds["test"] if "test" in ds else ds["train"]
    hard_ids = set()
    for row in split:
        if row.get("difficulty") in HARD_DIFFICULTIES:
            hard_ids.add(row["instance_id"])
    print(f"Found {len(hard_ids)} hard instances from SWE-bench/SWE-bench_Verified")
    return hard_ids


def _to_rows(split_data, data_source="r2e"):
    """Convert HuggingFace dataset split to list of row dicts for parquet."""
    rows = []
    for row in split_data:
        row_dict = dict(row)
        problem_statement = row_dict.get("problem_statement", "")
        rows.append({
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": ""},  # placeholder, replaced by r2e_prompt.py
                {"role": "user", "content": problem_statement},
            ],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": json.dumps(row_dict),  # JSON string — survives Arrow serialization
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/r2e")
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    for dataset_name, (output_name, data_source) in DATASETS.items():
        output_path = os.path.join(local_dir, output_name)
        if os.path.exists(output_path):
            print(f"{output_path} already exists, skipping.")
            continue

        print(f"Downloading {dataset_name} ...")
        dataset_splits = load_dataset(dataset_name)

        if "train" in dataset_splits:
            split_data = dataset_splits["train"]
        elif "test" in dataset_splits:
            split_data = dataset_splits["test"]
        else:
            print(f"Skipping {dataset_name}: no train/test split found.")
            continue

        # Serialize each row as JSON string in extra_info,
        # same pattern as rllm's swe_dataset.py.
        # This avoids Arrow nested-dict serialization issues.
        rows = _to_rows(split_data, data_source=data_source)
        df = pd.DataFrame(rows)
        df.to_parquet(output_path)
        print(f"Saved {len(df)} rows to {output_path}")

    # Build hard subset for SWE-Bench-Verified test set
    hard_output_path = os.path.join(local_dir, "info_r2e_test_hard.parquet")
    if os.path.exists(hard_output_path):
        print(f"{hard_output_path} already exists, skipping.")
    else:
        # Load R2E version of SWE-Bench-Verified (need docker_image field for ARL pods)
        print("Downloading R2E-Gym/SWE-Bench-Verified for hard subset ...")
        dataset_splits = load_dataset("R2E-Gym/SWE-Bench-Verified")
        split_data = dataset_splits["test"] if "test" in dataset_splits else dataset_splits["train"]

        hard_ids = _get_hard_instance_ids()
        hard_rows = []
        for row in split_data:
            row_dict = dict(row)
            if row_dict.get("instance_id") in hard_ids:
                problem_statement = row_dict.get("problem_statement", "")
                hard_rows.append({
                    "data_source": "r2e_test",
                    "prompt": [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": problem_statement},
                    ],
                    "ability": "code",
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": json.dumps(row_dict),
                })
        hard_df = pd.DataFrame(hard_rows)
        hard_df.to_parquet(hard_output_path)
        print(f"Saved {len(hard_df)} hard rows to {hard_output_path}")


if __name__ == "__main__":
    main()
