import pandas as pd
import json
from pathlib import Path
import re

def remove_index_from_patch(patch_str):
    """Remove the 'index xxxx..xxxx' line from patch string using regex"""
    if not isinstance(patch_str, str):
        return patch_str
    
    pattern = r'\nindex [a-f0-9]+\.\.[a-f0-9]+(?: \d+)?\n'
    return re.sub(pattern, '\n', patch_str, flags=re.MULTILINE)

def get_pure_patch(patch_str):
    patches = re.split(r'(?=^diff --git a/)', patch_str, flags=re.MULTILINE)
    filtered_patches = []
    for patch in patches:
        m = re.match(r'diff --git a/(.*?) b/(.*?)\n', patch)
        if not m:
            continue
        origin_filename = m.group(1)
        filename_lower = origin_filename.lower()
        if ".py" not in filename_lower \
            or "test" in filename_lower and 'pytest' not in filename_lower\
            or "reproduce" in filename_lower \
            or origin_filename == '/dev/null' \
            or re.search(r'^new file mode \d{6}$', patch, flags=re.MULTILINE) \
            or re.search(r'^--- /dev/null$', patch, flags=re.MULTILINE):
            continue
        filtered_patches.append(remove_index_from_patch(patch))
    patch = '\n'.join(filtered_patches)
    return patch.strip()

# Base directory and file pattern
base_dir = Path("data/rollouts/deepswe/")
file_pattern = "r2egym-deepswe-64k-100-steps-run-{}.jsonl"
df_list = []

# Process files 0-15
for i in range(16):
    file_path = base_dir / file_pattern.format(i)
    
    # Read all lines at once and parse JSON
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # print(data[0]['output_patch'])
    # print(data[0]['reward'])
    # print(get_pure_patch(data[0]['output_patch']))
    # exit()
    
    # Create DataFrame directly from the list of dicts
    df = pd.DataFrame([{
        "instance_id": item["ds"]["instance_id"],
        "problem_statement": item["problem_statement"],
        "patch": get_pure_patch(item["output_patch"]),
        "resolved": item["reward"],
        "split": 'test'
    } for item in data])
    df = df[df["patch"] != ""]
    
    df_list.append(df)

# Concatenate all DataFrames first
combined_df = pd.concat(df_list, ignore_index=True)

# Group by instance_id and aggregate patch and resolved into lists
merged_df = combined_df.groupby('instance_id').agg({
    'patch': list,
    'problem_statement': 'first',
    'resolved': list,
    'split': 'first'
}).reset_index()

merged_df['patch_length'] = merged_df['patch'].apply(len)
merged_df['resolved_length'] = merged_df['resolved'].apply(len)

invalid_patch = merged_df[merged_df['patch_length'] != 16]
invalid_resolved = merged_df[merged_df['resolved_length'] != 16]

print("Invalid patch lengths:")
print(invalid_patch[['instance_id', 'patch_length']])

print("\nInvalid resolved lengths:")
print(invalid_resolved[['instance_id', 'resolved_length']])

for idx, row in merged_df.iterrows():
    len_patch = len(row['patch'])
    len_resolved = len(row['resolved'])
    assert len_patch == len_resolved
    if len_patch < 4:
        print(f"WARNING: instance id: {row['instance_id']}: only {len_patch} patches! This may affect the performance.")
    if len_patch < 16:
        print(f"instance id: {row['instance_id']}: only {len_patch} patches!")
        delta = 16 - len_patch
        for i in range(delta):
            merged_df.iloc[idx]['patch'].append("# NO PATCH GENERATED!")
            merged_df.iloc[idx]['resolved'].append(False)

print(merged_df)

# Verify that each list has exactly 16 elements (one from each run)
assert all(merged_df['patch'].apply(len) == 16)
assert all(merged_df['resolved'].apply(len) == 16)

merged_df.to_parquet("data/info_test_deepswe.parquet")
# merged_df.to_parquet("data/info_test_deepswe_alter.parquet")