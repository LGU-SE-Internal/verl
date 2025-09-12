import pandas as pd
from tqdm import tqdm

# df1 = pd.read_parquet('data/info_train.parquet')
# df1 = pd.read_parquet('data/info_sft.parquet')
df1 = pd.read_parquet('data/info_train_naive_direct.parquet')
df2 = pd.read_parquet('data/info_oracle.parquet')

id2label = {}
num_true = 0
num_false = 0

print(len(df1))

for idx, row in df1.iterrows():
    if row['instance_id'] not in id2label:
        id2label[row['instance_id']] = [row['resolved']]
    else:
        id2label[row['instance_id']].append(row['resolved'])
    
    if row['resolved']:
        num_true += 1
    else:
        num_false += 1

delta = num_false - num_true
print(delta)

all_false_id = []
for key, value in id2label.items():
    if not (True in value):
        all_false_id.append(key)

print(len(all_false_id))

df1 = pd.concat([df1, df2[df2['instance_id'].isin(all_false_id[:delta])]])
print(len(df1[df1['resolved']==True]))
print(len(df1[df1['resolved']==False]))
# df1.to_parquet('data/info_sft_naive_balanced.parquet')
df1.to_parquet('data/info_train_naive_direct_balanced.parquet')



    
        
