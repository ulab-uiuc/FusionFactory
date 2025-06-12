import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import ast
import re
import json
import argparse

import pandas as pd


def preprocess_performance(value):
    if isinstance(value, float):
        return value
    elif isinstance(value, int):
        return float(value)
    elif isinstance(value, str):
        value = value.strip()

        try:
            return float(value)
        except ValueError:
            pass

        try:
            value_clean = re.sub(r'np\.float64\((.*?)\)', r'\1', value)
            parsed = ast.literal_eval(value_clean)
            if isinstance(parsed, dict) and "pass@1" in parsed:
                return float(parsed["pass@1"])
            else:
                print(f"CANNOT EXTRACT 'pass@1'：'{value}'")
                return None
        except Exception as e:
            print(f"NEITHER float NOR dict：'{value}'，ERROR: {e}")
            return None
    else:
        print(f"Unknown performance value: {value}（Type: {type(value)}）")
        return None


def sort_list(samples, setting="perf"):
    sorted_samples = []
    for sample in samples:
        sample = sample.copy()
        sample['performance_processed'] = sample['performance'].apply(preprocess_performance)
        sample['llm_judge_processed'] = sample['llm_judge'].apply(preprocess_performance)
        sample['output_tokens_num_processed'] = pd.to_numeric(sample['output_tokens_num'], errors='coerce')
        if setting == "perf":
            sort_by = ['performance_processed', 'output_tokens_num_processed']
        elif setting == "judge":
            sort_by = ['llm_judge_processed', 'output_tokens_num_processed']
        elif setting == "hybrid":
            sort_by = ['performance_processed', 'llm_judge_processed']
        elif setting == "baseline":
            sort_by = ['performance_processed', 'output_tokens_num_processed']
        else:
            raise NotImplementedError

        sample_sorted = sample.sort_values(
            by=sort_by,
            ascending=[False, False]
        ).reset_index(drop=True)
        sorted_samples.append(sample_sorted)

    return sorted_samples


def json_data_gen(sorted_samples, setting="perf", is_small=False):
    SMALL_MODEL = ["qwen2-7b-instruct", "qwen2.5-7b-instruct", "gemma-7b", "codegemma-7b", "gemma-2-9b-it",
                   "llama-3.1-8b-instruct", "granite-3.0-8b-instruct", "llama3-chatqa-1.5-8b", "mistral-nemo-12b-instruct",
                   "mistral-7b-instruct-v0.3"]
    TEMP = []
    for i in SMALL_MODEL:
        TEMP.append(i + "_think")
    SMALL_MODEL.extend(TEMP)
    json_data_list = []
    for sorted_sample in sorted_samples:
        is_select = False
        cnt = 0
        for idx, row in sorted_sample.iterrows():
            task_name = row.get("task_name", "")
            llm_name = row.get("llm", "")
            if llm_name not in SMALL_MODEL and is_small:
                continue
            response = row.get("response", "")
            query = row.get("query", "")
            gt = row.get("ground_truth", "")
            perf = row.get('performance_processed', "")
            if not is_select:
                if task_name in ["mbpp", "human_eval"]:
                    reformat_gt = "[BEGIN]\n {} \n[DONE]".format(str(gt))
                elif task_name in ["mmlu", "gpqa", "commonsense_qa", "openbook_qa", "arc_challenge", "hellaswag"]:
                    reformat_gt = "({})".format(str(gt))
                else:
                    reformat_gt = str(gt)
                json_entry = {
                    "instruction": "Answer the following questions as required.",
                    "input": str(query).replace("let's think step by step", "").replace("Let's think step by step", ""),
                    "output": reformat_gt
                }
                json_data_list.append(json_entry)
                is_select = True
            if perf == 0:
                break
            if setting != "baseline" and cnt < TOP_K:
                json_entry = {
                    "instruction": "Answer the following questions as required.",
                    "input": str(query),
                    "output": str(response)
                }
                json_data_list.append(json_entry)
                cnt += 1

    return json_data_list


# FORCE_TORCHRUN=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=6,7,8,9 llamafactory-cli train examples/train_full/llama3_lora_sft_5.yaml
# FORCE_TORCHRUN=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=6,7,8,9 llamafactory-cli train examples/train_lora/llama3_lora_sft_5.yaml
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default="perf", choices=["perf", "judge", "hybrid", "baseline"])
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--save_path', type=str, default="/data/taofeng2/Router_bench/data_process/LLaMA-Factory/data")
    parser.add_argument('--csv_path_with_judge', type=str, default="/data/taofeng2/Router_bench/router_data/router_zijie_train_0603_with_judge.csv")
    args = parser.parse_args()

    TOP_K = args.k
    SMALL = args.small
    SETTING = args.setting
    SAVE_PATH = args.save_path
    df = pd.read_csv(args.csv_path_with_judge)
    total_rows = len(df)
    num_samples = total_rows // 40

    all_samples = []
    for i in range(num_samples):
        start = i * 40
        end = start + 40
        sample = df.iloc[start:end].reset_index(drop=True)
        all_samples.append(sample)

    sorted_all_samples = sort_list(samples=all_samples, setting=SETTING)

    json_data_list = json_data_gen(sorted_samples=sorted_all_samples, setting=SETTING, is_small=SMALL)
    if SETTING == "baseline":
        json_data_list = json_data_list * (TOP_K + 1)

    print(len(json_data_list))
    with open(os.path.join(SAVE_PATH, "{}_sft_top_{}.json".format(SETTING, TOP_K)), 'w', encoding='utf-8') as f:
        json.dump(json_data_list, f, ensure_ascii=False, indent=2)
