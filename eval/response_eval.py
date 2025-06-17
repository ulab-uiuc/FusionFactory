import os
import time
import numpy as np
import pandas as pd
import torch
import ast
import re
import json
from tqdm import tqdm
from collections import defaultdict

from utils import model_prompting, f1_score, exact_match_score, get_bert_score, evaluate_code, cem_score
from beartype.typing import Any, Dict, List, Tuple, Optional
from human_eval.evaluate_functional_correctness import entry_point_item
from mbpp.mbpp_eval import entry_point_item_mbpp
from math_eval import last_boxed_only_string, remove_boxed, is_equiv


def eval_perf(metric, prediction, ground_truth, task_name, task_id=None):
    if task_name in ["natural_qa", "trivia_qa", "squad", "boolq"]:
        metric = "cem"

    if metric == 'em':
        result = exact_match_score(prediction, ground_truth)
        return float(result)
    elif metric == 'cem':
        result = cem_score(prediction, ground_truth)
        return float(result)
    elif metric == 'em_mc':
        result = exact_match_score(prediction, ground_truth, normal_method="mc")
        return float(result)
    elif metric == 'bert_score':
        result = get_bert_score([prediction], [ground_truth])
        return result
    elif metric == 'GSM8K':
        ground_truth = ground_truth.split("####")[-1].replace(',', '').replace('$', '').replace('.', '').strip()
        answer = re.findall("(\\-?[0-9\\.\\,]+)", prediction)
        final_answer = None
        if len(answer) == 0:
            return 0
        else:
            invalid_str = ['', '.']
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
        final_answer = final_answer.replace(',', '').replace('$', '').replace('.', '').strip()
        if final_answer == ground_truth:
            return 1
        else:
            return 0
    elif metric == 'MATH':
        ground_truth = remove_boxed(last_boxed_only_string(ground_truth))
        try:
            string_in_last_boxed = last_boxed_only_string(prediction)
            if string_in_last_boxed is not None:
                answer = remove_boxed(string_in_last_boxed)
                if is_equiv(answer, ground_truth):
                    return 1
        except Exception as e:
            return 0
        return 0
    elif metric == 'f1_score' or task_name in ['quac']:
        f1, prec, recall = f1_score(prediction, ground_truth)
        return f1
    elif metric == 'code_eval':
        if task_id is None:
            raise ValueError("task_id is required for code_eval metric")

        is_mbpp = not str(task_id).startswith("HumanEval")

        if is_mbpp:
            code_match = re.search(r'\[BEGIN\](.*?)(?:\[DONE\]|\[Done\]|$)', prediction, re.DOTALL | re.IGNORECASE)

            if code_match:
                code = code_match.group(1).strip()
            else:
                code = prediction.strip()

            mbpp_sample = {"task_id": task_id, "completion": code}
            pass_1 = entry_point_item_mbpp(mbpp_sample, './dataset/Code/mbpp.jsonl')
            return pass_1
        else:
            code_match = re.search(r'\[BEGIN\](.*?)(?:\[DONE\]|$)', prediction, re.DOTALL | re.IGNORECASE)
            if code_match:
                raw_code = code_match.group(1).strip()
                if raw_code.lstrip().startswith("def "):
                    code = raw_code
                else:
                    code = "    " + raw_code.replace("\n", "\n    ")
            else:
                code = prediction.strip()
            dict = {"task_id": task_id, "completion": code}
            pass_1 = entry_point_item(dict, './dataset/Code/HumanEval.jsonl')
            return pass_1
    else:
        return 0


if __name__ == '__main__':
    file_path = './dataset/router_data_with_response.csv'
    model = str(file_path.split('/')[-1].split('_')[-4:-2])
    df = pd.read_csv(file_path)
    print(df['output'].iloc[0])

    MATH_TASK = ['gsm8k', 'math']
    CODE_TASK = ["mbpp", "human_eval"]
    COMMONSENSE_TASK = ['commonsense_qa', 'openbook_qa', 'arc_challenge', 'hellaswag']
    WORLD_KNOWLEDGE_TASK = ["natural_qa", "trivia_qa"]
    READING_TASK = ["squad", 'quac', "boolq"]
    POPULAR_TASK = ["mmlu", "gpqa"]

    task_results = defaultdict(list)
    category_results = {
        'math': [],
        'code': [],
        'commonsense': [],
        'world_knowledge': [],
        'reading': [],
        'popular': []
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing", ncols=100):
        prediction = row["output"]
        gt = row["ground_truth"]
        task_name = row["task_name"]
        metric = row["metric"]
        task_id = row["task_id"]

        if task_name in CODE_TASK:
            if not pd.isna(task_id) and not str(task_id).startswith("HumanEval"):
                task_id = int(str(task_id).strip())

        try:
            res = eval_perf(metric=metric, prediction=prediction if prediction else "", ground_truth=gt, task_name=task_name, task_id=task_id)
        except Exception as e:
            print(e)
            continue

        if isinstance(res, dict):
            res = res['pass@1']

        task_results[task_name].append(res)

        if task_name in MATH_TASK:
            category_results['math'].append(res)
        elif task_name in CODE_TASK:
            category_results['code'].append(res)
        elif task_name in COMMONSENSE_TASK:
            category_results['commonsense'].append(res)
        elif task_name in WORLD_KNOWLEDGE_TASK:
            category_results['world_knowledge'].append(res)
        elif task_name in READING_TASK:
            category_results['reading'].append(res)
        elif task_name in POPULAR_TASK:
            category_results['popular'].append(res)

    print(f"Model: {model}")
    print("\n=== Results by Task Category ===")
    for category, results in category_results.items():
        if results:
            print(f"{category.title()} Tasks: {np.mean(results):.4f}")

    print("\n=== Results by Individual Task ===")
    for task_name, results in sorted(task_results.items()):
        if results:
            print(f"{task_name}: {np.mean(results):.4f}")

    print("\n=== Sample Counts ===")
    print(f"Total samples: {len(df)}")
    for task_name, results in sorted(task_results.items()):
        print(f"{task_name}: {len(results)}")