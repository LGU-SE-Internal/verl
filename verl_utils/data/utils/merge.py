import pandas as pd
import os

# get_naive_train and get_naive_test??

# df1 = pd.read_parquet('data/info_test_01.parquet')
# df2 = pd.read_parquet('data/info_test_02.parquet')
# df3 = pd.read_parquet('data/info_test_03.parquet')
# df4 = pd.read_parquet('data/info_test_04.parquet')

# df_merged = pd.concat([df1, df2, df3, df4])

# df_merged.to_parquet('data/info_test_naive.parquet', index=False)


# df1 = pd.read_parquet('data/info_train_01.parquet')
# df2 = pd.read_parquet('data/info_train_02.parquet')
# df3 = pd.read_parquet('data/info_train_03.parquet')
# df4 = pd.read_parquet('data/info_train_04.parquet')
# df5 = pd.read_parquet('data/info_train_05.parquet')
# df6 = pd.read_parquet('data/info_train_06.parquet')

# df_merged = pd.concat([df1, df2, df3, df4, df5, df6])

# df_merged.to_parquet('data/info_train_naive.parquet', index=False)


# df1 = pd.read_parquet('data/info_sft_01.parquet')
# df2 = pd.read_parquet('data/info_sft_02.parquet')
# df3 = pd.read_parquet('data/info_sft_03.parquet')
# df4 = pd.read_parquet('data/info_sft_04.parquet')
# df5 = pd.read_parquet('data/info_sft_05.parquet')
# df6 = pd.read_parquet('data/info_sft_06.parquet')

# df_merged = pd.concat([df1, df2, df3, df4, df5, df6])

# df_merged.to_parquet('data/info_sft_naive.parquet', index=False)

path = "data/rollouts/distill_gemini25"
df_list = []
for root, dirs, files in os.walk(path):
    for file in files:
        df = pd.read_parquet(os.path.join(root, file))
        df_list.append(df)

df_merged = pd.concat(df_list, ignore_index=True)

df_merged.to_parquet('data/distill_gemini25.parquet', index=False)