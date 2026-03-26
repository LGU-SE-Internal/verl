import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import requests
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from verl_utils.reward.extract_answer import (extract_patch,
                                              extract_think_format,
                                              extract_tool_format)

random.seed(42)
SERVER_URL = os.environ.get("RM_SERVER_URL", None)
ROOT_DIR = os.environ.get("ROOT_DIR", "")
RM_BATCH_SIZE = 4

# --- Pydantic Models ---
class BatchRequest(BaseModel):
    issue: str
    patch_list: List[str]

class BatchItem(BaseModel):
    batch_id: str
    data: BatchRequest

class MultiBatchRequest(BaseModel):
    batches: List[BatchItem]

def compute_score_remote_stage(data_sources, solution_strs, ground_truths, extra_infos):
    scores = compute_score_remote(data_sources, solution_strs, ground_truths, extra_infos)
    new_scores = []
    for score in scores:
        if score == -1.0:
            new_scores.append(0.0)
        elif score == 0.0:
            new_scores.append(0.1)
        else:
            new_scores.append(1.0)
    return new_scores

def compute_score_remote_clip(data_sources, solution_strs, ground_truths, extra_infos):
    scores = compute_score_remote(data_sources, solution_strs, ground_truths, extra_infos)
    scores = [0.0 if score == -1.0 else score for score in scores]
    return scores

def random_reward(data_sources, solution_strs, ground_truths, extra_infos):
    if 'test' in data_sources[0]:
        # return compute_score_bench(data_sources, solution_strs, ground_truths, extra_infos)
        return compute_score_record(data_sources, solution_strs, ground_truths, extra_infos)
    else:
        return compute_score_random(data_sources, solution_strs, ground_truths, extra_infos)

def compute_score_remote(data_sources, solution_strs, ground_truths, extra_infos):
    if 'test' in data_sources[0]:
        return compute_score_arl_test(data_sources, solution_strs, ground_truths, extra_infos)
    else:
        return compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos)


def compute_score_arl_test(data_sources, solution_strs, ground_truths, extra_infos):
    """Test-time evaluation: run tests in ARL pods + save JSONL for records.

    Replaces the old compute_score_record (save only) and compute_score_bench
    (external harness) with direct ARL pod testing.
    """
    from verl_utils.reward.arl_reward import compute_score_arl

    # Save JSONL record (same as compute_score_record, for bookkeeping)
    ts = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    patch_strs = [extract_patch(sol) for sol in solution_strs]
    payload = []
    for idx, (patch, extra_info) in enumerate(zip(patch_strs, extra_infos)):
        if patch.strip():
            payload.append({
                "model_name_or_path": 'trae-lite-ossi',
                "instance_id": extra_info["instance_id"],
                "model_patch": patch
            })
    if payload:
        jsonl_content = "\n".join([json.dumps(p) for p in payload])
        with open(f'cached_submission_{ts}.jsonl', 'w') as f:
            f.write(jsonl_content)
        if ROOT_DIR:
            with open(os.path.join(ROOT_DIR, f'cached_submission_{ts}.jsonl'), 'w') as f:
                f.write(jsonl_content)

    # Run actual tests in ARL pods
    scores = compute_score_arl(data_sources, solution_strs, ground_truths, extra_infos)

    # Log results
    resolved = sum(1 for s in scores if s == 1.0)
    total = len(scores)
    empty = sum(1 for s in scores if s == -1.0)
    print(f"[ARL Test] {resolved}/{total} resolved, {empty} empty patches")

    # Save ARL results alongside JSONL
    arl_results = {
        "timestamp": ts,
        "resolved": resolved,
        "total": total,
        "empty": empty,
        "scores": scores,
        "instance_ids": [info.get("instance_id", "") for info in extra_infos],
    }
    with open(f'cached_arl_results_{ts}.json', 'w') as f:
        f.write(json.dumps(arl_results))

    return scores


def compute_score_random(data_sources, solution_strs, ground_truths, extra_infos):
    return [float(random.choice([0, 1])) if extract_tool_format(sol) and extract_think_format(sol) else 0.0 for sol in solution_strs]

def compute_score_record(data_sources, solution_strs, ground_truths, extra_infos):
    ts = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    patch_strs = [extract_patch(sol) for sol in solution_strs]
    payload = []
    for idx, (patch, extra_info) in enumerate(zip(patch_strs, extra_infos)):
        if patch.strip():
            payload.append({
                "model_name_or_path": 'trae-lite-ossi',
                "instance_id": extra_info["instance_id"],
                "model_patch": patch
            })

    jsonl_content = "\n".join([json.dumps(p) for p in payload])
    with open(f'cached_submission_{ts}.jsonl', 'w') as f:
        f.write(jsonl_content)
    if ROOT_DIR:
        with open(os.path.join(ROOT_DIR, f'cached_submission_{ts}.jsonl'), 'w') as f:
            f.write(jsonl_content)
    else:
        print("WARNING: ROOT_DIR not set, backup not saved.")

    return [0.0] * len(patch_strs)

def compute_score_bench(data_sources, solution_strs, ground_truths, extra_infos):

    ts = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    
    payload = []
    empty_indices = []
    empty_instances = []
    valid_indices_map = {}

    patch_strs = [extract_patch(sol) for sol in solution_strs]

    for idx, (patch, extra_info) in enumerate(zip(patch_strs, extra_infos)):
        instance_id = extra_info["instance_id"]
        if not patch.strip():
            empty_indices.append(idx)
            empty_instances.append(instance_id)
        else:
            payload.append({
                "model_name_or_path": 'trae-lite-ossi',
                "instance_id": instance_id,
                "model_patch": patch
            })
            if instance_id not in valid_indices_map:
                valid_indices_map[instance_id] = idx

    with open(f'cached_empty_ids_{ts}.txt', 'w') as f:
        f.write('\n'.join(empty_instances))

    if not payload:
        scores = [0.0] * len(patch_strs)
        for idx in empty_indices:
            scores[idx] = -1.0
        return scores

    run_id = None
    try:
        jsonl_content = "\n".join([json.dumps(p) for p in payload])

        with open(f'cached_submission_{ts}.jsonl', 'w') as f:
            f.write(jsonl_content)
        
        data = {"dataset": "SWE-bench/SWE-bench_Verified"}
        files = {'file': ('predictions.jsonl', jsonl_content.encode('utf-8'), 'application/octet-stream')}
        
        print("Uploading patches to evaluation server...")
        submit_url = f"{HARNESS_URL}evaluate"
        response = requests.post(submit_url, data=data, files=files, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        run_id = result.get("run_id")
        if not run_id:
            raise ValueError("Server response did not include a run_id.")
        print(f"Submission successful. Received run_id: {run_id}")

    except Exception as e:
        print(f"Error submitting evaluation request: {e}")
        print("All submissions get 0 reward.")
        return [0.0] * len(patch_strs)

    result_filename = None
    polling_url = f"{HARNESS_URL}progress/{run_id}"
    POLLING_INTERVAL = 15
    MAX_POLLING_ATTEMPTS = 300
    
    print("Polling for results...")
    for attempt in range(MAX_POLLING_ATTEMPTS):
        try:
            response = requests.get(polling_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            status = data.get('status')
            if status == 'completed':
                print("Evaluation completed.")
                result_filename = data.get('result_file')
                break
            elif status == 'error':
                print(f"Evaluation failed on the server. Log: {data.get('output')}")
                print("All submissions get 0 reward.")
                return [0.0] * len(patch_strs)
            elif status == 'running':
                print(f"Attempt {attempt + 1}/{MAX_POLLING_ATTEMPTS}: Status is 'running'. Waiting for {POLLING_INTERVAL}s...")
                time.sleep(POLLING_INTERVAL)
            else:
                print(f"Unknown status received: {status}. Aborting.")
                return [0.0] * len(patch_strs)

        except requests.exceptions.RequestException as e:
            print(f"Error polling for progress: {e}. Retrying in {POLLING_INTERVAL}s...")
            time.sleep(POLLING_INTERVAL)
    
    if not result_filename:
        print("Polling timed out. Could not retrieve results.")
        return [0.0] * len(patch_strs)

    final_scores = [0.0] * len(patch_strs)
    try:
        download_url = f"{HARNESS_URL}download/{result_filename}"
        print(f"Downloading result file from: {download_url}")
        response = requests.get(download_url, timeout=60)
        response.raise_for_status()
        eval_results = response.json()

        with open(f"cached_harness_{ts}.json", 'w') as f:
            f.write(json.dumps(eval_results))
            
        for resolved_id in eval_results['resolved_ids']:
            original_index = valid_indices_map[resolved_id]
            final_scores[original_index] = 1.0

    except requests.exceptions.RequestException as e:
        print(f"Failed to download or parse result file: {e}")
        return [0.0] * len(patch_strs)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing result file: {e}")
        return [0.0] * len(patch_strs)

    for idx in empty_indices:
        final_scores[idx] = -1.0
        
    return final_scores

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    grouped_data = defaultdict(list)
    patch_strs = [extract_patch(sol) for sol in solution_strs]
    tool_format_flags = [extract_tool_format(sol) for sol in solution_strs]
    think_format_flags = [extract_think_format(sol) for sol in solution_strs]
    for idx, (sol, info) in enumerate(zip(patch_strs, extra_infos)):
        instance_id = info["instance_id"]
        issue = info['issue']
        patch = sol
        grouped_data[instance_id].append((idx, issue, patch))
    
    batch_items_pydantic: List[BatchItem] = [] 
    index_mapping = {}
    
    for instance_id, items in grouped_data.items():
        assert len(items) % RM_BATCH_SIZE == 0, f"Instance {instance_id} has {len(items)} rollouts, which can not be divided by batch size {RM_BATCH_SIZE}."
        items.sort(key=lambda x: x[0])
        
        for i in range(0, len(items), RM_BATCH_SIZE):
            batch_id = f"{instance_id}_{i//RM_BATCH_SIZE}"
            batch_items_slice = items[i:i+RM_BATCH_SIZE]
            
            batch_issues = [item[1] for item in batch_items_slice]
            assert len(set(batch_issues)) == 1, "Error occurs when wrapping batches for rewarding."
            batch_solutions = [item[2] for item in batch_items_slice]
            
            request_data = BatchRequest(
                issue=batch_issues[0],
                patch_list=batch_solutions
            )
            batch_item = BatchItem(
                batch_id=batch_id,
                data=request_data
            )
            batch_items_pydantic.append(batch_item)
            
            index_mapping[batch_id] = [item[0] for item in batch_items_slice]
    
    if not batch_items_pydantic:
        return [0.0] * len(patch_strs)

    payload = MultiBatchRequest(batches=batch_items_pydantic)

    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    SERVER_URL,
                    json=payload.model_dump(mode="json"),
                    headers={"Content-Type": "application/json"},
                    proxies={"http": None, "https": None} # do not use proxy
                )
                response.raise_for_status()
                break
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                if attempt < max_retries - 1:
                    print(f"Request failed: {str(e)}. Retry attempts: {attempt+1}.")
                else:
                    raise RuntimeError(f"Request failed.") from e
        
        result = response.json()
        batch_results = result.get("scores", {})
        
        if not batch_results:
            raise ValueError("Empty response from reward server.")
    
    except Exception as e:
        print(f"Rewarding failed: {str(e)}")
        return [0.0] * len(patch_strs)
    
    final_scores = [0.0] * len(patch_strs)
    
    for batch in payload.batches:
        batch_id = batch.batch_id
        if batch_id in batch_results:
            scores = batch_results[batch_id]
            orig_indices = index_mapping[batch_id] 
            for score, orig_idx in zip(scores, orig_indices):
                final_scores[orig_idx] = score

    for idx, (patch, tool_flag, think_flag) in enumerate(zip(patch_strs, tool_format_flags, think_format_flags)):
        if not patch or not tool_flag or not think_flag:
            final_scores[idx] = -1.0
    
    return final_scores

def evaluate_jsonl(jsonl_path):
    """Evaluate patches from a cached_submission JSONL using ARL pods.

    Usage:
        python verl_utils/reward/model_client.py cached_submission_2026-03-26_17-24-34.jsonl

    The JSONL format (one JSON per line):
        {"model_name_or_path": "...", "instance_id": "astropy__astropy-13398", "model_patch": "diff --git ..."}

    Extra info (docker_image etc.) is loaded from R2E-Gym/SWE-Bench-Verified.
    Patches produced before the strip() bugfix are handled: trailing newline is ensured.
    """
    from datasets import load_dataset
    from verl_utils.reward.arl_reward import compute_score_arl

    # Read JSONL
    with open(jsonl_path, 'r') as f:
        objs = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(objs)} patches from {jsonl_path}")

    # Load R2E dataset for extra_info (need docker_image for ARL pods)
    print("Loading R2E-Gym/SWE-Bench-Verified for extra_info ...")
    ds = load_dataset("R2E-Gym/SWE-Bench-Verified")
    split = ds["test"] if "test" in ds else ds["train"]
    instance_map = {row["instance_id"]: dict(row) for row in split}

    data_sources = []
    solution_strs = []
    ground_truths = []
    extra_infos = []
    skipped = []

    for obj in objs:
        instance_id = obj["instance_id"]
        patch = obj.get("model_patch", "")

        if instance_id not in instance_map:
            print(f"  SKIP: {instance_id} not found in R2E dataset")
            skipped.append(instance_id)
            continue

        # Fix strip() damage from old code: ensure trailing newline
        if patch and not patch.endswith("\n"):
            patch += "\n"

        data_sources.append("r2e_test")
        # Wrap in format that extract_patch() expects
        solution_strs.append(f"[PATCH]\n{patch}\n[/PATCH]")
        ground_truths.append("")
        extra_infos.append(instance_map[instance_id])

    print(f"Evaluating {len(solution_strs)} patches ({len(skipped)} skipped) ...")
    scores = compute_score_arl(data_sources, solution_strs, ground_truths, extra_infos)

    # Report
    resolved = sum(1 for s in scores if s == 1.0)
    failed = sum(1 for s in scores if s == 0.0)
    empty = sum(1 for s in scores if s == -1.0)
    print(f"\nResults: {resolved}/{len(scores)} resolved, {failed} failed, {empty} empty")

    evaluated_objs = [o for o in objs if o["instance_id"] not in skipped]
    for obj, score in zip(evaluated_objs, scores):
        status = "RESOLVED" if score == 1.0 else ("EMPTY" if score == -1.0 else "FAILED")
        print(f"  {obj['instance_id']}: {status}")

    # Save results
    ts = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    results = {
        "source": jsonl_path,
        "timestamp": ts,
        "total": len(scores),
        "resolved": resolved,
        "failed": failed,
        "empty": empty,
        "scores": {obj["instance_id"]: s for obj, s in zip(evaluated_objs, scores)},
    }
    result_path = f"eval_results_{ts}.json"
    with open(result_path, 'w') as f:
        f.write(json.dumps(results, indent=2))
    print(f"Results saved to {result_path}")

    return scores


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate cached_submission JSONL via ARL pods")
    parser.add_argument("jsonl_path", help="Path to cached_submission_*.jsonl file")
    parser.add_argument("--arl-gateway-url", default="http://118.145.210.10:8080")
    parser.add_argument("--arl-mirror-namespace", default="code")
    parser.add_argument("--arl-concurrency", type=int, default=1024)
    parser.add_argument("--arl-timeout", type=int, default=600)
    parser.add_argument("--arl-experiment-id", default="eval")
    args = parser.parse_args()

    # Set ARL env vars (defaults match TRAE_R2E.sh)
    os.environ.setdefault("ARL_GATEWAY_URL", args.arl_gateway_url)
    os.environ.setdefault("ARL_MIRROR_NAMESPACE", args.arl_mirror_namespace)
    os.environ.setdefault("ARL_REWARD_CONCURRENCY", str(args.arl_concurrency))
    os.environ.setdefault("ARL_REWARD_TIMEOUT", str(args.arl_timeout))
    os.environ.setdefault("ARL_EXPERIMENT_ID", args.arl_experiment_id)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    evaluate_jsonl(args.jsonl_path)