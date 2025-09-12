import argparse
import os
import datasets
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import verl_utils.data.source as src

def make_map_fn(data_source):

    if 'naive' in data_source:
        process_fn = src.naive_prompt.process_fn
    elif 'batch' in data_source:
        process_fn = src.batch_prompt.process_fn
    elif 'sft' in data_source:
        process_fn = src.sft_prompt.process_fn
    elif 'embed' in data_source:
        process_fn = src.embed_prompt.process_fn
    elif 'formal' in data_source:
        process_fn = src.formal_prompt.process_fn
    elif 'base' in data_source:
        process_fn = src.base_prompt.process_fn
    elif 'rm' in data_source:
        process_fn = src.rm_prompt.process_fn
    elif 'vm' in data_source:
        process_fn = src.vm_prompt.process_fn
    elif 'rubric' in data_source:
        process_fn = src.rubric_prompt.process_fn
    elif 'tts' in data_source:
        process_fn = src.tts_prompt.process_fn
    elif 'gen' in data_source:
        process_fn = src.gen_prompt.process_fn
    elif 'ver' in data_source:
        process_fn = src.ver_prompt.process_fn
    elif 'pair' in data_source:
        process_fn = src.pair_prompt.process_fn
    else:
        raise ValueError(f"Unknown dataset: {data_source}")

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="agentic")
    parser.add_argument("--file_path", default='data/info_train.parquet')

    args = parser.parse_args()

    dataset = datasets.Dataset.from_parquet(args.file_path)
    dataset = dataset.map(function=make_map_fn(args.data_source), with_indices=True)
    print(dataset)
    dataset.to_parquet(args.file_path.replace("info", "data"))
