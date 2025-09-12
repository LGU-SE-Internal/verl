import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from verl_utils.eval.result_evaluator import extract_batch_combine
from verl_utils.reward.reward_fn import get_batch_reward

path = 'data/distill_gemini25_new3.parquet'
msg_df = pd.read_parquet(path)
msg_df['answer'] = msg_df.apply(lambda x: extract_batch_combine(x['messages'][-1]['content']), axis=1)
msg_df['acc'] = msg_df.apply(lambda x: get_batch_reward(x['answer'], x['ground_truth'], 0.0) if x['answer'] else 0.0, axis=1)
msg_df['is_invalid'] = msg_df.apply(lambda x: x['acc'] != 1.0, axis=1)
invalid_idx = set(msg_df.loc[msg_df['is_invalid'], 'idx'])
processed_idx = set(msg_df['idx'].tolist()) - invalid_idx
msg_df = msg_df[~msg_df['is_invalid']].drop(columns=['is_invalid', 'acc']).reset_index(drop=True)

# 添加patch_count列用于分层抽样
msg_df['patch_count'] = msg_df['answer'].apply(np.sum)
msg_path = path.replace('data/', 'data/data_')
msg_df.to_parquet(msg_path)
print(f"Saved data to: {msg_path}")

# 按patch_count分组
groups = msg_df.groupby('patch_count')
group_0 = groups.get_group(0) if 0 in groups.groups else pd.DataFrame()
group_1 = groups.get_group(1) if 1 in groups.groups else pd.DataFrame()
group_2 = groups.get_group(2) if 2 in groups.groups else pd.DataFrame()
group_3 = groups.get_group(3) if 3 in groups.groups else pd.DataFrame()
group_4 = groups.get_group(4) if 4 in groups.groups else pd.DataFrame()

print(len(group_0))
print(len(group_1))
print(len(group_2))
print(len(group_3))
print(len(group_4))


# 平衡样本数量
min_0_4 = min(len(group_0), len(group_4))
min_1_3 = min(len(group_1), len(group_3))

# 分层随机抽样（设置随机种子保证可复现性）
balanced_groups = []
if min_0_4 > 0:
    balanced_groups.append(group_0.sample(min_0_4, random_state=42) if len(group_0) > min_0_4 else group_0)
    balanced_groups.append(group_4.sample(min_0_4, random_state=42) if len(group_4) > min_0_4 else group_4)
if min_1_3 > 0:
    balanced_groups.append(group_1.sample(min_1_3, random_state=42) if len(group_1) > min_1_3 else group_1)
    balanced_groups.append(group_3.sample(min_1_3, random_state=42) if len(group_3) > min_1_3 else group_3)
if not group_2.empty:
    balanced_groups.append(group_2)

# 合并平衡后的数据集
balanced_df = pd.concat(balanced_groups, ignore_index=True)
balanced_df['enable_thinking'] = False
print(balanced_df)

# 打印新的分布
new_patch_counts = balanced_df['patch_count'].value_counts().to_dict()
print("Balanced distribution:")
print(new_patch_counts)
print(f"Original size: {len(msg_df)}, Balanced size: {len(balanced_df)}")

# 保存平衡后的数据
balanced_path = path.replace('data/', 'data/data_balanced_')
balanced_df.to_parquet(balanced_path)
print(f"Saved balanced data to: {balanced_path}")