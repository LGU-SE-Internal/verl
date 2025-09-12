import pandas as pd
import re

def is_only_comment_or_whitespace(patch_str):
    """
    Checks if a patch contains only changes to comments or whitespace.
    Returns True if only noise changes are present, False otherwise.
    """
    # Regex to find all added or removed lines, excluding the diff header lines --- and +++
    changed_lines = re.findall(r'^[+-](?![+-]{2} )(.+)', patch_str, flags=re.MULTILINE)

    if not changed_lines:
        # No lines were added or removed, so it's not a substantive change.
        return True

    for line in changed_lines:
        stripped_line = line.strip()
        # Check if the line is NOT noise. A line is considered noise if it's:
        # 1. Empty or just whitespace.
        # 2. A single-line comment.
        if stripped_line and not stripped_line.startswith('#') and not (stripped_line.startswith('"""') and stripped_line.endswith('"""')) and not (stripped_line.startswith("'''") and stripped_line.endswith("'''")):
            # If we find even one line that is actual code, the patch is valid.
            return False
            
    # If all changed lines were noise, return True to filter this patch out.
    return True

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
        if is_only_comment_or_whitespace(patch):
            continue
            
        filtered_patches.append(remove_index_from_patch(patch))
    patch = '\n'.join(filtered_patches)
    return patch.strip()

# Read all parquet files
df1 = pd.read_parquet('data/info_train_01.parquet')
df2 = pd.read_parquet('data/info_train_02.parquet')
df3 = pd.read_parquet('data/info_train_03.parquet')
df4 = pd.read_parquet('data/info_train_04.parquet')
df5 = pd.read_parquet('data/info_train_05.parquet')
df6 = pd.read_parquet('data/info_train_06.parquet')

# FULL version
df1 = pd.concat([df1, pd.read_parquet('data/info_sft_01.parquet')], ignore_index=True)
df2 = pd.concat([df2, pd.read_parquet('data/info_sft_02.parquet')], ignore_index=True)
df3 = pd.concat([df3, pd.read_parquet('data/info_sft_03.parquet')], ignore_index=True)
df4 = pd.concat([df4, pd.read_parquet('data/info_sft_04.parquet')], ignore_index=True)
df5 = pd.concat([df5, pd.read_parquet('data/info_sft_05.parquet')], ignore_index=True)
df6 = pd.concat([df6, pd.read_parquet('data/info_sft_06.parquet')], ignore_index=True)

# Concatenate all dataframes
df_merged = pd.concat([df1, df2, df3, df4, df5, df6])

print("Applying patch filters to all rows...")
df_merged['patch'] = df_merged['patch'].apply(get_pure_patch)

original_count = len(df_merged)
df_merged = df_merged[df_merged['patch'] != '']
print(f"Filtered out {original_count - len(df_merged)} rows with empty patches.")

# Count the number of True and False in resolved field
true_count = df_merged['resolved'].sum()
false_count = len(df_merged) - true_count

print(true_count, false_count)

# Determine which class is the minority
if true_count > false_count:
    majority_class = True
    minority_count = false_count
else:
    majority_class = False
    minority_count = true_count

# Split the dataframe into majority and minority classes
df_majority = df_merged[df_merged['resolved'] == majority_class]
df_minority = df_merged[df_merged['resolved'] != majority_class]

# Randomly sample from majority class to match minority count
df_majority_sampled = df_majority.sample(n=minority_count, random_state=42)

# Combine the sampled majority with the minority
df_balanced = pd.concat([df_majority_sampled, df_minority])

# Shuffle the dataframe to mix the classes
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

df_balanced['split'] = 'train'

true_count = df_balanced['resolved'].sum()
false_count = len(df_balanced) - true_count

print(true_count, false_count)

# Save the balanced dataframe
# df_balanced.to_parquet('data/info_train_naive_balanced.parquet', index=False)
# df_balanced.to_parquet('data/info_train_naive_full.parquet', index=False)
df_balanced.to_parquet('data/info_train_ver.parquet', index=False)
print(df_balanced)