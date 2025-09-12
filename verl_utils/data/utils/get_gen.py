import pandas as pd
import json

name = "data/datasets/filtered_issue_final.jsonl"
data = []  # List to store extracted data

# Open the file in read mode (added quotes around 'r')
with open(name, 'r') as f:
    lines = f.readlines()

for line in lines:
    obj = json.loads(line)
    # Extract only the required fields
    extracted_data = {
        'instance_id': obj.get('instance_id'),
        'base_commit': obj.get('base_commit'),
        'problem_statement': obj.get('problem_statement'),
        'split': 'train'
    }
    data.append(extracted_data)

# Convert to DataFrame and save as Parquet
df = pd.DataFrame(data)
df.to_parquet('data/datasets/swe-ossi.parquet', index=False)
df.to_parquet('data/info_train_gen.parquet', index=False)

name = "data/datasets/swe-bench-verified.parquet"
df = pd.read_parquet(name)
df['split'] = 'test'
df.to_parquet(name, index=False)
df.to_parquet('data/info_test_gen.parquet', index=False)