import os
import argparse
import json

import pandas as pd


def json_data_gen(sorted_samples):
    json_data_list = []
    for row in sorted_samples:
        task_name = row.get("task_name", "")
        response = row.get("response", "")
        query = row.get("query", "")
        gt = row.get("ground_truth", "")
        # print(query)
        # print(gt)
        json_entry = {
            "instruction": "Answer the following questions as required.",
            "input": str(query),
            "output": str(gt)
        }
        json_data_list.append(json_entry)

    return json_data_list


# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/vllm_infer.py --model_name_or_path meta-llama/Llama-3.1-8B-Instruct --adapter_name_or_path saves/llama3.1-8b/lora/xxx --dataset router_test --cutoff_len 2048
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default="./LLaMA-Factory/data")
    parser.add_argument('--csv_path', type=str, default="./dataset/router_data_test.csv")
    args = parser.parse_args()

    SAVE_PATH = args.save_path
    df = pd.read_csv(args.csv_path)

    total_rows = len(df)
    print(total_rows)
    num_samples = total_rows // 40

    all_samples = []
    for i in range(num_samples):
        start = i * 40
        end = start + 40
        sample = df.iloc[start:end].reset_index(drop=True)
        all_samples.append(sample.iloc[0])

    json_data_list = json_data_gen(sorted_samples=all_samples)
    with open(os.path.join(SAVE_PATH, "router_test.json"), 'w', encoding='utf-8') as f:
        json.dump(json_data_list, f, ensure_ascii=False, indent=2)
