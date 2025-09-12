from copy import deepcopy
import pandas as pd
import numpy as np
import re
np.random.seed(42)


df1 = pd.read_parquet('data/info_distill_01.parquet')
df2 = pd.read_parquet('data/info_distill_02.parquet')
df3 = pd.read_parquet('data/info_distill_03.parquet')
df4 = pd.read_parquet('data/info_distill_04.parquet')
df5 = pd.read_parquet('data/info_distill_05.parquet')
df6 = pd.read_parquet('data/info_distill_06.parquet')

# 把patch = ""的行去掉。
df1 = df1[df1['patch'].fillna('').str.strip() != '']
df2 = df2[df2['patch'].fillna('').str.strip() != '']
df3 = df3[df3['patch'].fillna('').str.strip() != '']
df4 = df4[df4['patch'].fillna('').str.strip() != '']
df5 = df5[df5['patch'].fillna('').str.strip() != '']
df6 = df6[df6['patch'].fillna('').str.strip() != '']

def remove_index_from_patch(patch_str):
    """Remove the 'index xxxx..xxxx' line from patch string using regex"""
    if not isinstance(patch_str, str):
        return patch_str
    
    pattern = r'\nindex [a-f0-9]+\.\.[a-f0-9]+(?: \d+)?\n'
    return re.sub(pattern, '\n', patch_str, flags=re.MULTILINE)

# 应用到每个 DataFrame 的 patch 列
df1['patch'] = df1['patch'].apply(remove_index_from_patch)
df2['patch'] = df2['patch'].apply(remove_index_from_patch)
df3['patch'] = df3['patch'].apply(remove_index_from_patch)
df4['patch'] = df4['patch'].apply(remove_index_from_patch)
df5['patch'] = df5['patch'].apply(remove_index_from_patch)
df6['patch'] = df6['patch'].apply(remove_index_from_patch)

df_lists = [df1, df2, df3, df4, df5, df6]


df1_instance_list = df1['instance_id'].tolist()
df2_instance_list = df2['instance_id'].tolist()
df3_instance_list = df3['instance_id'].tolist()
df4_instance_list = df4['instance_id'].tolist()
df5_instance_list = df5['instance_id'].tolist()
df6_instance_list = df6['instance_id'].tolist()

instance_lists = [df1_instance_list, df2_instance_list, df3_instance_list, df4_instance_list, df5_instance_list, df6_instance_list]

def get_4_cross_lists(lists):
    return set(lists[0]).intersection(set(lists[1])).intersection(set(lists[2])).intersection(set(lists[3]))

def get_4_merged_df(lists, cross_ids):
    df = deepcopy(lists[0])
    df = df[df['instance_id'].isin(cross_ids)]
    df = df.drop(columns=['patch', 'resolved'])
    df['patch'] = df.apply(lambda row: [lists[0][lists[0]['instance_id'] == row['instance_id']]['patch'].values[0], lists[1][lists[1]['instance_id'] == row['instance_id']]['patch'].values[0], lists[2][lists[2]['instance_id'] == row['instance_id']]['patch'].values[0], lists[3][lists[3]['instance_id'] == row['instance_id']]['patch'].values[0]], axis=1)
    df['resolved'] = df.apply(lambda row: [lists[0][lists[0]['instance_id'] == row['instance_id']]['resolved'].values[0], lists[1][lists[1]['instance_id'] == row['instance_id']]['resolved'].values[0], lists[2][lists[2]['instance_id'] == row['instance_id']]['resolved'].values[0], lists[3][lists[3]['instance_id'] == row['instance_id']]['resolved'].values[0]], axis=1)
    overlap_cnt = df[df['patch'].apply(lambda x: x[0] == x[1] or x[0] == x[2] or x[0] == x[3] or x[1] == x[2] or x[1] == x[3] or x[2] == x[3])].shape[0]
    return df, overlap_cnt

cross_id_lists = []
merged_df_lists = []
overlap_cnts = 0
for i in range(6):
    for j in range(i+1, 6):
        temp_cross_instance_list = []
        temp_df_lists = []
        for k in range(6):
            if k != i and k != j:
                temp_cross_instance_list.append(deepcopy(instance_lists[k]))
                temp_df_lists.append(deepcopy(df_lists[k]))
        cross_ids = get_4_cross_lists(temp_cross_instance_list)
        cross_id_lists.append(cross_ids)
        merged_df, overlap_cnt = get_4_merged_df(temp_df_lists, cross_ids)
        merged_df_lists.append(merged_df)
        overlap_cnts += overlap_cnt

print(overlap_cnts)

patch_instance_count_list = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
}
for df in merged_df_lists:
    patch_instance_count_list['0'] += df[df['resolved'].apply(lambda x: x.count(True) == 0)].shape[0]
    patch_instance_count_list['1'] += df[df['resolved'].apply(lambda x: x.count(True) == 1)].shape[0]
    patch_instance_count_list['2'] += df[df['resolved'].apply(lambda x: x.count(True) == 2)].shape[0]
    patch_instance_count_list['3'] += df[df['resolved'].apply(lambda x: x.count(True) == 3)].shape[0]
    patch_instance_count_list['4'] += df[df['resolved'].apply(lambda x: x.count(True) == 4)].shape[0]

print(patch_instance_count_list)

new_df = pd.DataFrame()
for df in merged_df_lists:
    df_mix = df[df['resolved'].apply(lambda x: x.count(True) > 0 and x.count(True) < 4)]
    len_df = df_mix.shape[0]
    df_all_correct = df[df['resolved'].apply(lambda x: x.count(True) == 4)]
    df_all_incorrect = df[df['resolved'].apply(lambda x: x.count(True) == 0)]
    if df_all_correct.shape[0] > len_df // 3:
        df_all_correct = df_all_correct.sample(n=len_df // 3, random_state=42)
    if df_all_incorrect.shape[0] > len_df // 3:
        df_all_incorrect = df_all_incorrect.sample(n=len_df // 3, random_state=42)
    new_df = pd.concat([new_df, df_mix, df_all_correct, df_all_incorrect], axis=0)
new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True) # row shuffle
print(new_df)


def shuffle_patch_resolved(row):
    perm = np.random.permutation(len(row['patch']))
    row['patch'] = [row['patch'][i] for i in perm]
    row['resolved'] = [row['resolved'][i] for i in perm]
    return row

new_df = new_df.apply(shuffle_patch_resolved, axis=1) # column shuffle

print(new_df)


patch_instance_count_list['0'] = new_df[new_df['resolved'].apply(lambda x: x.count(True) == 0)].shape[0]
patch_instance_count_list['1'] = new_df[new_df['resolved'].apply(lambda x: x.count(True) == 1)].shape[0]
patch_instance_count_list['2'] = new_df[new_df['resolved'].apply(lambda x: x.count(True) == 2)].shape[0]
patch_instance_count_list['3'] = new_df[new_df['resolved'].apply(lambda x: x.count(True) == 3)].shape[0]
patch_instance_count_list['4'] = new_df[new_df['resolved'].apply(lambda x: x.count(True) == 4)].shape[0]

print(patch_instance_count_list)

# 打印有多少patch完全一样的
print(new_df[new_df['patch'].apply(lambda x: x[0] == x[1] and x[1] == x[2] and x[2] == x[3])].shape[0])

# 保存为info_distill_batch.parquet
new_df.to_parquet('data/info_distill_batch.parquet')