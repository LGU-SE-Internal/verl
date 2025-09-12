
import pandas as pd
import numpy as np
import json
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns


# import pandas as pd
# import json
# import sys
# import os

# sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
# from verl_utils.reward.extract_answer import extract_batch_combine

# index = 7
# path = 'output_VoteMasking_SB_DAPO_RL_32B_225.parquet'
# df = pd.read_parquet(path)
# case = df.iloc[index]['responses'].tolist()
# gt = df.iloc[index]['reward_model']['ground_truth']
# nums = [0] * 4
# rlist = []
# for resp in case:
#     r = extract_batch_combine(resp)
#     rlist.append(r)
#     for i in range(4):
#         if r[i]:
#             nums[i] +=1
# print(nums) # [4,4,0,0] 前两个是无效票，后两个视为正确预测
# print(gt) # [False, False, False, False]

# pred = []
# for num in nums:
#     if num >= 6:
#         pred.append(True)
#     elif num <= 2:
#         pred.append(False)
#     else:
#         pred.append(None)

# correct = 0
# total = 0
# ignore = 0
# for i in range(4):
#     if pred[i] is None:
#         ignore += 1
#     else:
#         total += 1
#         if pred[i] == gt[i]:
#             correct +=1

# acc = correct / total
# drop_rate = ignore / 4

# print(acc) # 1
# print(drop_rate) # 0.5

# --- Setup: Add path and create mock data ---

# Add the parent directory of 'verl_utils' to the system path
# This ensures that 'from verl_utils...' works correctly.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from verl_utils.reward.extract_answer import extract_batch_combine


def analyze_voting_confidence(file_path: str):
    """
    Performs a comprehensive analysis of voting confidence vs. accuracy.
    """
    # 1. Load Data
    df = pd.read_parquet(file_path)
    
    # 2. Process Data to get vote counts and predictions
    results = []
    empty_judge = 0
    for _, row in df.iterrows():
        case_responses = row['responses']
        ground_truth = row['reward_model']['ground_truth']
        
        vote_counts = np.zeros(4, dtype=int)
        for resp_str in case_responses:
            votes_in_round = extract_batch_combine(resp_str)
            print(votes_in_round)
            if votes_in_round is None:
                empty_judge += 1
                continue
            vote_counts += np.array(votes_in_round)
            
        for i in range(4):
            count = vote_counts[i]
            truth = ground_truth[i]
            
            prediction = None
            # if count >= 6:
            # if count == 8:
            if count == 32:
                prediction = True
            # elif count <= 2:
            elif count == 0:
                prediction = False
            
            is_correct = None
            if prediction is not None:
                is_correct = (prediction == truth)
                
            results.append({
                'vote_count': count,      # Number of positive votes (0-8)
                'ground_truth': truth,    # The real answer
                'prediction': prediction, # The model's prediction (True, False, or None)
                'is_correct': is_correct  # If the prediction was correct (True, False, or None)
            })

    analysis_df = pd.DataFrame(results)

    # 3. Calculate Overall Metrics
    total_judgements = len(analysis_df)
    valid_judgements_df = analysis_df.dropna(subset=['prediction'])
    
    num_valid = len(valid_judgements_df)
    num_ignored = total_judgements - num_valid
    
    drop_rate = num_ignored / total_judgements
    overall_accuracy = valid_judgements_df['is_correct'].mean()
    error_rate = empty_judge / total_judgements

    print("\n--- Overall Analysis Results ---")
    print(f"Total Judgements Processed: {total_judgements}")
    print(f"Valid Judgements (Votes <= 2 or >= 6): {num_valid}")
    print(f"Ignored Judgements (Votes between 3 and 5): {num_ignored}")
    print(f"Drop Rate: {drop_rate:.2%}")
    print(f"Error Rate: {error_rate:.2%}")
    print(f"Accuracy on Valid Judgements: {overall_accuracy:.2%}")
    print("--------------------------------\n")

    # 4. Visualization
    
    # Plot 1: Distribution of Vote Counts
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax = sns.countplot(x='vote_count', data=analysis_df, palette='viridis')
    plt.title('Distribution of Vote Counts', fontsize=16)
    plt.xlabel('Number of Positive Votes (out of 8)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Highlight the ignored "low confidence" zone
    plt.axvspan(2.5, 5.5, color='red', alpha=0.15, label='Ignored Zone (Low Confidence)')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        
    plt.legend()
    plt.show()

    # Plot 2: Accuracy vs. Vote Confidence
    # We only care about accuracy for the valid votes (0, 1, 2, 6, 7, 8)
    accuracy_by_vote = valid_judgements_df.groupby('vote_count')['is_correct'].mean().reset_index()
    accuracy_by_vote = accuracy_by_vote.rename(columns={'is_correct': 'accuracy'})
    
    plt.figure(figsize=(12, 6))
    bar_plot = sns.barplot(x='vote_count', y='accuracy', data=accuracy_by_vote, palette='coolwarm')
    
    plt.title('Accuracy vs. Vote Confidence', fontsize=16)
    plt.xlabel('Number of Positive Votes (out of 8)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.05) # Accuracy is between 0 and 1
    
    # Add accuracy labels on top of each bar
    for index, row in accuracy_by_vote.iterrows():
        bar_plot.text(index, row.accuracy + 0.02, f'{row.accuracy:.2%}', 
                      color='black', ha="center")
        
    plt.show()


if __name__ == '__main__':
    # parquet_path = 'output_VoteMasking_SB_DAPO_RL_32B_225.parquet'
    # parquet_path = 'output_VoteMaskingx32_SB_DAPO_RL_7B_FULL_MORE_410.parquet'
    parquet_path = 'output_VoteMaskingx32_SB_DAPO_RL_32B_225.parquet'
    
    # Create mock data if the file doesn't exist
    create_mock_data(parquet_path)
    
    # Run the analysis
    analyze_voting_confidence(parquet_path)