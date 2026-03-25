"""Download R2E-Gym and SWE-Bench Verified from HuggingFace, save as verl-compatible parquet.

Usage:
    python verl_utils/data/swe_dataset.py --local_dir data/r2e

Output:
    data/r2e/info_r2e_train.parquet   (R2E-Gym Subset, for training)
    data/r2e/info_r2e_test.parquet    (SWE-Bench Verified, for validation)

Then convert to training format:
    python verl_utils/data/data_process.py --data_source r2e_train --file_path data/r2e/info_r2e_train.parquet
    python verl_utils/data/data_process.py --data_source r2e_test  --file_path data/r2e/info_r2e_test.parquet
"""

import argparse
import json
import os

import pandas as pd
from datasets import load_dataset


DATASETS = {
    "R2E-Gym/R2E-Gym-Subset": "info_r2e_train.parquet",
    "R2E-Gym/SWE-Bench-Verified": "info_r2e_test.parquet",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/r2e")
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    for dataset_name, output_name in DATASETS.items():
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
        rows = []
        for row in split_data:
            row_dict = dict(row)
            problem_statement = row_dict.get("problem_statement", "")
            rows.append({
                "data_source": "r2e",
                "prompt": [
                    {"role": "system", "content": ""},  # placeholder, replaced by r2e_prompt.py
                    {"role": "user", "content": problem_statement},
                ],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": json.dumps(row_dict),  # JSON string — survives Arrow serialization
            })

        df = pd.DataFrame(rows)
        df.to_parquet(output_path)
        print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
