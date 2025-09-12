import pandas as pd
import random
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from verl_utils.reward.extract_answer import *

random.seed(42)

def get_batch_tp_tn_0(pred_list_batch, gt_list, type):
    tp = 0.0
    tn = 0.0
    pred_list = pred_list_batch[0]
    if not pred_list:
        return 0
    for i in range(4):
        if pred_list[i] and gt_list[i]:
            tp += 1.0
        if not pred_list[i] and not gt_list[i]:
            tn += 1.0
    if type == 'tp':
        return tp
    if type == 'tn':
        return tn

def select_1_entropy(pred_list_batch, gt_list, entropy_batch):
    pred_list_batch = [pred for pred in pred_list_batch if pred is not None]
    if not pred_list_batch:
        return False
    pred_list = pred_list_batch[0]
    entropy_low = entropy_batch[0]
    for i in range(1, len(pred_list_batch)):
        if entropy_batch[i] < entropy_low:
            entropy_low = entropy_batch[i]
            pred_list = pred_list_batch[i]

    true_indices = [i for i, pred in enumerate(pred_list) if pred]
    if true_indices == []:
        selected_idx = random.choice(range(4))
    else:
        selected_idx = random.choice(true_indices)
    return bool(gt_list[selected_idx])

def select_1_voting(pred_list_batch, gt_list):
    pred_list_batch = [pred for pred in pred_list_batch if pred is not None]
    if not pred_list_batch:
        return False
    vote = [0 for i in range(len(pred_list_batch[0]))]
    for pred in pred_list_batch:
        for idx in range(len(pred)):
            if pred[idx]:
                vote[idx] += 1
    max_vote = max(vote)
    max_indices = [i for i, x in enumerate(vote) if x == max_vote]

    selected_idx = random.choice(max_indices)
    return bool(gt_list[selected_idx])

def select_1_random(pred_list_batch, gt_list):
    pred_list_batch = [pred for pred in pred_list_batch if pred is not None]
    if not pred_list_batch:
        return False
    pred_list = random.choice(pred_list_batch)
    true_indices = [i for i, pred in enumerate(pred_list) if pred]
    if true_indices == []:
        selected_idx = random.choice(range(4))
    else:
        selected_idx = random.choice(true_indices)
    return bool(gt_list[selected_idx])

def select_1_baseline(gt_list):
    selected_idx_baseline = random.choice(range(4))
    return bool(gt_list[selected_idx_baseline])

def select_1_best(gt_list):
    if True in gt_list[0:4]:
        return True
    return False

def evaluate_batch(output_path):
    df = pd.read_parquet(output_path)
    df['answer'] = df.apply(lambda x: [extract_batch_combine(resp) for resp in x['responses'] if resp is not None], axis=1)
    df['ground_truth'] = df.apply(lambda x: x['reward_model']['ground_truth'], axis=1)
    print(df.iloc[0]['responses'])
    print(df.iloc[0]['answer'])
    print(df.iloc[0]['ground_truth'])
    df['tp'] = df.apply(lambda x: get_batch_tp_tn_0(x['answer'], x['ground_truth'], 'tp'), axis=1)
    df['tn'] = df.apply(lambda x: get_batch_tp_tn_0(x['answer'], x['ground_truth'], 'tn'), axis=1)
    df['#p'] = df.apply(lambda x: x['ground_truth'].sum(), axis=1)
    df['#n'] = df.apply(lambda x: (1-x['ground_truth']).sum(), axis=1)
    # df['selected_answer'] = df.apply(lambda x: select_1_random(x['answer'], x['ground_truth']), axis=1)
    # df['selected_answer'] = df.apply(lambda x: select_1_entropy(x['answer'], x['ground_truth'], x['entropy']), axis=1)
    df['selected_answer'] = df.apply(lambda x: select_1_voting(x['answer'], x['ground_truth']), axis=1)
    df['baseline_answer'] = df.apply(lambda x: select_1_baseline(x['ground_truth']), axis=1)
    df['best_answer'] = df.apply(lambda x: select_1_best(x['ground_truth']), axis=1)
    num_selected_correct = df[df['selected_answer'] == True].shape[0]
    num_baseline_correct = df[df['baseline_answer'] == True].shape[0]
    num_best_correct = df[df['best_answer'] == True].shape[0]
    invalid_num = df['answer'].apply(lambda x: all(item is None for item in x)).sum()
    total_num = df.shape[0]

    print('-'*80)
    print(f"Name:\t{output_path.split('verl/')[-1].split('/actor')[0]}")
    print('-'*80)
    print(f"TPR[0]:\t{df['tp'].sum()/df['#p'].sum()}")
    print(f"TNR[0]:\t{df['tn'].sum()/df['#n'].sum()}")
    print(f"Select:\t{num_selected_correct}")
    print(f"Random:\t{num_baseline_correct}")
    print(f"Best:\t{num_best_correct}")
    print(f"SR:\t{num_selected_correct/total_num}")
    print(f"RR:\t{num_baseline_correct/total_num}")
    print(f"BR:\t{num_best_correct/total_num}")
    print(f"Error:\t{invalid_num}")
    print('-'*80)
    print(f"Total:\t{total_num}")
    print('-'*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True, help='Path to input parquet dataset')
    args = parser.parse_args()
    evaluate_batch(args.file_path)