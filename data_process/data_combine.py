import os
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
import csv
import random
import json
import pickle
import argparse
import yaml
from datetime import datetime

import pandas as pd
from datasets import load_dataset
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import threading

from llm_engine import LLMEngine
from utils import loadjson, get_longformer_representation, hellaswag_preprocess
from prompt_pool import *


save_lock = threading.Lock()


def parallel_task_1(data):
    success = True
    id, query_t, task_description = data
    try:
        query_t_embedding = get_longformer_representation([query_t])
        task_description_embedding = get_longformer_representation([task_description])
    except Exception as e:
        print("!!!!!!!!!!!!!!!", e)
        query_t_embedding = ""
        task_description_embedding = ""
        success = False

    return id, query_t_embedding, task_description_embedding, success


def parallel_task_2(data):
    success = True
    rid, a_t, config, llm_names, MyLLMEngine, query_t, query_think_t, ground_truth_t, metric_t, task_id_k, task_id_t = data
    print("info:", task_id_t, llm_names[a_t])
    try:
        response_t = MyLLMEngine.get_llm_response(query=query_t, llm_idx=a_t)
        response_think_t = MyLLMEngine.get_llm_response(query=query_think_t, llm_idx=a_t)
        reward_t = MyLLMEngine.eval(prediction=response_t, ground_truth=ground_truth_t, metric=metric_t,
                                         task_id=task_id_k)
        reward_think_t = MyLLMEngine.eval(prediction=response_think_t, ground_truth=ground_truth_t, metric=metric_t,
                                    task_id=task_id_k)
        cost_t,input_price,output_price,input_size,output_size = MyLLMEngine.compute_cost(llm_idx=a_t, input_text=query_t,
                                               output_text=response_t)
        cost_think_t,_,_,input_size_think,output_size_think = MyLLMEngine.compute_cost(llm_idx=a_t, input_text=query_think_t,
                                          output_text=response_think_t)
    except Exception as e:
        print("!!!!!!!!!!!!!!!", e)
        response_t = ""
        response_think_t = ""
        reward_t = ""
        cost_t = ""
        reward_think_t=""
        cost_think_t=""
        input_price = ""
        output_price = ""
        input_size = ""
        output_size = ""
        input_size_think=""
        output_size_think=""
        success = False

    return rid, a_t, response_t, response_think_t, reward_t, cost_t,reward_think_t,cost_think_t,input_price,output_price,input_size,output_size,input_size_think,output_size_think, success


def save_data_atomic(data, filename):
    tmp_file = filename + ".tmp"
    tmp_file = os.path.join(CACHE_SAVE_PATH, tmp_file)
    filename = os.path.join(CACHE_SAVE_PATH, filename)
    with open(tmp_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_file, filename)


def async_save(data_snapshot, base_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{timestamp}.pkl"

    def save_job():
        with save_lock:
            save_data_atomic(data_snapshot, filename)

    threading.Thread(target=save_job, daemon=True).start()


class data_building:
    def __init__(self, qa_path, llm_path, round, config):
        self.qa_data = qa_path
        self.llm_description = loadjson(llm_path)
        self.llm_names = list(self.llm_description.keys())
        self.all_llm_description = []
        for inter in self.llm_names:
            self.all_llm_description.append(self.llm_description[inter]['feature'])
        self.MyLLMEngine = LLMEngine(llm_names=self.llm_names, llm_description=self.llm_description)
        self.config = config
        self.round = round
        self.construct_data_with_LLM()
        self.final_gathering()

    def construct_data_with_LLM(self, p_num=100, task_1_save_every=1000, task_2_save_every=1000):
        if os.path.exists(os.path.join(CACHE_SAVE_PATH, "ret_1_final.pkl")):
            with open(os.path.join(CACHE_SAVE_PATH, "ret_1_final.pkl"), 'rb') as f:
                ret_1_tmp = pickle.load(f)
            print(f"Loaded existing results: {len(ret_1_tmp)}")
            ret_1_tmp.sort(key=lambda x: x[0], reverse=False)
            task_1_args = [(id, row['query'], row['task_description']) for id, row in enumerate(self.qa_data) if not ret_1_tmp[id][-1]]
            ret_1 = [i for i in ret_1_tmp if i[-1]]
        else:
            ret_1 = []
            task_1_args = [(id, row['query'], row['task_description']) for id, row in enumerate(self.qa_data)]

        print("Task 1 Start")
        with ThreadPool(p_num) as p:
            for i, r in enumerate(tqdm(p.imap_unordered(parallel_task_1, task_1_args), total=len(task_1_args), desc="Processing", ncols=100)):
                ret_1.append(r)

                if (i + 1) % task_1_save_every == 0:
                    snapshot = ret_1.copy()
                    async_save(snapshot, "ret_1")
        ret_1.sort(key=lambda x: x[0], reverse=False)
        ret_1_fail_num = 0
        for ii in ret_1:
            if not ii[-1]:
                ret_1_fail_num += 1
        save_data_atomic(ret_1, "ret_1_final.pkl")
        print("Task 1 Complete: Success: {}, Fail: {}".format(len(ret_1) - ret_1_fail_num, ret_1_fail_num))

        while self.round > 0:
            self.round -= 1
            if os.path.exists(os.path.join(CACHE_SAVE_PATH, "ret_2_final.pkl")):
                with open(os.path.join(CACHE_SAVE_PATH, "ret_2_final.pkl"), 'rb') as f:
                    ret_2_tmp = pickle.load(f)
                print(f"Loaded existing results: {len(ret_2_tmp)}")
                ret_2_tmp.sort(key=lambda x: (x[0], x[1]), reverse=False)
                task_2_args = [(rid, a_t, self.config, self.llm_names, self.MyLLMEngine, row['query'], row['query_think'], row['gt'], row['metric'], row['task_id'] if 'task_id' in row else None, row['task_name']) for rid, row in enumerate(self.qa_data) for a_t in range(len(self.llm_names)) if not ret_2_tmp[rid * len(self.llm_names) + a_t][-1]]
                ret_2 = [i for i in ret_2_tmp if i[-1]]
            else:
                ret_2 = []
                task_2_args = [(rid, a_t, self.config, self.llm_names, self.MyLLMEngine, row['query'], row['query_think'], row['gt'], row['metric'], row['task_id'] if 'task_id' in row else None, row['task_name']) for rid, row in enumerate(self.qa_data) for a_t in range(len(self.llm_names))]

            print("Task 2 Start")
            with ThreadPool(p_num) as p:
                for i, r in enumerate(tqdm(p.imap_unordered(parallel_task_2, task_2_args), total=len(task_2_args), desc="Processing", ncols=100)):
                    ret_2.append(r)

                    if (i + 1) % task_2_save_every == 0:
                        snapshot = ret_2.copy()
                        async_save(snapshot, "ret_2")
            ret_2.sort(key=lambda x: (x[0], x[1]), reverse=False)
            ret_2_fail_num = 0
            for ii in ret_2:
                if not ii[-1]:
                    ret_2_fail_num += 1
            save_data_atomic(ret_2, "ret_2_final.pkl")
            print("Task 2 Complete: Success: {}, Fail: {}".format(len(ret_2) - ret_2_fail_num, ret_2_fail_num))
            if ret_2_fail_num == 0:
                break

    def final_gathering(self):
        with open(os.path.join(CACHE_SAVE_PATH, "ret_1_final.pkl"), 'rb') as f:
            ret_1 = pickle.load(f)
        print(f"Loaded existing results: {len(ret_1)}")
        ret_1.sort(key=lambda x: x[0], reverse=False)

        with open(os.path.join(CACHE_SAVE_PATH, "ret_2_final.pkl"), 'rb') as f:
            ret_2 = pickle.load(f)
        print(f"Loaded existing results: {len(ret_2)}")
        ret_2.sort(key=lambda x: (x[0], x[1]), reverse=False)

        # Create a DataFrame with your columns
        df = pd.DataFrame(
            columns=[
                'task_name',
                'task_id',
                'task_description',
                'task_description_embedding',
                'query',
                'query_embedding',
                'ground_truth',
                'metric',
                'llm',
                'input_price',
                'output_price',
                'input_tokens_num',
                'output_tokens_num',  # Note: Changed from 'output_tokens_think' to match new_row
                'performance',
                'cost',
                'response',
                'llm_description'
            ])

        # Create the CSV file with headers
        df.to_csv(self.config['saved_router_data_path'], index=False)

        # Process each row
        for rid, row in enumerate(tqdm(self.qa_data, total=len(self.qa_data), desc="Processing", ncols=100)):
            task_id_t = row['task_name']
            query_t = row['query']
            query_think_t = row['query_think']
            task_description = row['task_description']
            query_t_embedding = ret_1[rid][1]
            task_description_embedding = ret_1[rid][2]
            ground_truth_t = row['gt']
            metric_t = row['metric']
            task_id_k = row['task_id'] if 'task_id' in row else None

            for a_t in range(len(self.llm_names)):
                new_idx = rid * len(self.llm_names) + a_t
                response_t = ret_2[new_idx][2]
                response_think_t = ret_2[new_idx][3]
                reward_t = ret_2[new_idx][4]
                cost_t = ret_2[new_idx][5]
                reward_think_t = ret_2[new_idx][6]
                cost_think_t = ret_2[new_idx][7]
                input_price = ret_2[new_idx][8]
                output_price = ret_2[new_idx][9]
                input_tokens = ret_2[new_idx][10]
                output_tokens = ret_2[new_idx][11]
                input_tokens_think = ret_2[new_idx][12]
                output_tokens_think = ret_2[new_idx][13]
                llm_t = self.llm_names[a_t]

                # Create regular row
                new_row = {'task_name': task_id_t, 'task_id': task_id_k, 'task_description': task_description,
                           'task_description_embedding': task_description_embedding if a_t == 0 else "", 'query': query_t,
                           'query_embedding': query_t_embedding if a_t == 0 else "",
                           'ground_truth': ground_truth_t, 'metric': metric_t,
                           'llm': llm_t, 'input_price': input_price, 'output_price': output_price,
                           'input_tokens_num': input_tokens, 'output_tokens_num': output_tokens,
                           'performance': reward_t, 'cost': cost_t,
                           'response': response_t,
                           'llm_description': self.llm_description[llm_t]['feature']}

                # Create think row
                new_row_think = {'task_name': task_id_t, 'task_id': task_id_k, 'task_description': task_description,
                                 'task_description_embedding': "", 'query': query_think_t,
                                 'query_embedding': "",
                                 'ground_truth': ground_truth_t, 'metric': metric_t,
                                 'llm': llm_t + '_think', 'input_price': input_price, 'output_price': output_price,
                                 'input_tokens_num': input_tokens_think, 'output_tokens_num': output_tokens_think,
                                 'performance': reward_think_t, 'cost': cost_think_t,
                                 'response': response_think_t,
                                 'llm_description': self.llm_description[llm_t]['feature']}

                # Append rows to DataFrame
                temp_df = pd.DataFrame([new_row, new_row_think])

                # Write immediately to CSV in append mode (mode='a') without writing the header again (header=False)
                temp_df.to_csv(self.config['saved_router_data_path'], mode='a', header=False, index=False, quoting=csv.QUOTE_MINIMAL, escapechar='\\')


def get_n_samples(N=10, random_seed=42, split="train"):
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Initialize empty lists for each dataset
    natural_qa_samples = []
    trivia_qa_samples = []
    squad_samples = []
    quac_samples = []
    boolq_samples = []
    mmlu_samples = []
    gpqa_samples = []
    mbpp_samples = []
    humaneval_samples = []
    gsm8k_samples = []
    commonsense_qa_samples = []
    math_samples = []
    openbook_qa_samples = []
    arc_challenge_samples = []
    hellaswag_samples = []

    # 1. Natural QA dataset
    try:
        natural_qa = load_dataset('RUC-NLPIR/FlashRAG_datasets', 'nq',
                                  cache_dir='/data/taofeng2/Router_bench/dataset/World Knowledge')
        # Get N random samples from the training split (or another split if available)
        if split == "train":
            split_name = 'train' if 'train' in natural_qa else list(natural_qa.keys())[0]
        else:
            split_name = 'test' if 'test' in natural_qa else list(natural_qa.keys())[0]
        indices = random.sample(range(len(natural_qa[split_name])), min(N, len(natural_qa[split_name])))
        natural_qa_samples = [natural_qa[split_name][i] for i in indices]
        print(f"Successfully extracted {len(natural_qa_samples)} samples from Natural QA")
    except Exception as e:
        print(f"Error extracting from Natural QA: {e}")

    # 2. Trivia QA dataset
    try:
        trivia_qa = load_dataset("trivia_qa", "rc.nocontext",
                                 cache_dir='/data/taofeng2/Router_bench/dataset/World Knowledge')
        if split == "train":
            split_name = 'train' if 'train' in trivia_qa else list(trivia_qa.keys())[0]
        else:
            split_name = 'validation' if 'validation' in trivia_qa else list(trivia_qa.keys())[0]
        indices = random.sample(range(len(trivia_qa[split_name])), min(N, len(trivia_qa[split_name])))
        trivia_qa_samples = [trivia_qa[split_name][i] for i in indices]
        print(f"Successfully extracted {len(trivia_qa_samples)} samples from Trivia QA")
    except Exception as e:
        print(f"Error extracting from Trivia QA: {e}")

    # 3. SQUAD dataset
    if split == "train":
        try:
            SQUAD = pd.read_parquet("/data/taofeng2/Router_bench/dataset/Reading Comprehension/SQUAD.parquet")
            # Convert to list of dicts for consistency
            squad_list = SQUAD.to_dict('records')
            indices = random.sample(range(len(squad_list)), min(N, len(squad_list)))
            squad_samples = [squad_list[i] for i in indices]
            print(f"Successfully extracted {len(squad_samples)} samples from SQUAD")
        except Exception as e:
            print(f"Error extracting from SQUAD: {e}")
    else:
        try:
            squad = load_dataset('RUC-NLPIR/FlashRAG_datasets', 'squad',
                                 cache_dir='/data/taofeng2/Router_bench/dataset/World Knowledge')
            # Get N random samples from the training split (or another split if available)
            split_name = 'dev' if 'dev' in squad else list(squad.keys())[0]
            indices = random.sample(range(len(squad[split_name])), min(N, len(squad[split_name])))
            squad_samples = [squad[split_name][i] for i in indices]
            print(f"Successfully extracted {len(squad_samples)} samples from SQUAD")
        except Exception as e:
            print(f"Error extracting from SQUAD: {e}")

    # 4. QuAC dataset
    try:
        quac = load_dataset("quac", trust_remote_code=True,
                            cache_dir='/data/taofeng2/Router_bench/dataset/Reading Comprehension')
        if split == "train":
            split_name = 'train' if 'train' in quac else list(quac.keys())[0]
        else:
            split_name = 'test' if 'test' in quac else list(quac.keys())[0]
        indices = random.sample(range(len(quac[split_name])), N)
        quac_samples = [quac[split_name][i] for i in indices]
        print(f"Successfully extracted {len(quac_samples)} samples from QuAC")
    except Exception as e:
        print(f"Error extracting from QuAC: {e}")

    # 5. BoolQ dataset
    try:
        boolq = load_dataset("boolq", cache_dir='/data/taofeng2/Router_bench/dataset/Reading Comprehension')
        if split == "train":
            split_name = 'train' if 'train' in boolq else list(boolq.keys())[0]
        else:
            split_name = 'validation' if 'validation' in boolq else list(boolq.keys())[0]
        indices = random.sample(range(len(boolq[split_name])), min(N, len(boolq[split_name])))
        boolq_samples = [boolq[split_name][i] for i in indices]
        print(f"Successfully extracted {len(boolq_samples)} samples from BoolQ")
    except Exception as e:
        print(f"Error extracting from BoolQ: {e}")

    # 6. MMLU dataset
    try:
        mmlu = load_dataset("cais/mmlu", "all", cache_dir='/data/taofeng2/Router_bench/dataset/Popular')
        if split == "train":
            split_name = 'auxiliary_train' if 'auxiliary_train' in mmlu else list(mmlu.keys())[0]
        else:
            split_name = 'test' if 'test' in mmlu else list(mmlu.keys())[0]
        indices = random.sample(range(len(mmlu[split_name])), min(N, len(mmlu[split_name])))
        mmlu_samples = [mmlu[split_name][i] for i in indices]
        print(f"Successfully extracted {len(mmlu_samples)} samples from MMLU")
    except Exception as e:
        print(f"Error extracting from MMLU: {e}")

    # 7. GPQA dataset
    try:
        gpqa = load_dataset("Idavidrein/gpqa", "gpqa_main",
                            cache_dir='/data/taofeng2/Router_bench/dataset/Popular/gpqa')
        split_name = 'train' if 'train' in gpqa else list(gpqa.keys())[0]
        if split == "train":
            indices = random.sample(range(len(gpqa[split_name])), min(N, len(gpqa[split_name])))[:404]
        else:
            indices = random.sample(range(len(gpqa[split_name])), min(N, len(gpqa[split_name])))[404:]
        gpqa_samples = [gpqa[split_name][i] for i in indices]
        print(f"Successfully extracted {len(gpqa_samples)} samples from GPQA")
    except Exception as e:
        print(f"Error extracting from GPQA: {e}")

    # 8. MBPP dataset
    try:
        mdpp_path = '/data/taofeng2/Router_bench/dataset/Code/mbpp.jsonl'
        with open(mdpp_path, 'r') as f:
            lines = f.readlines()

        mbpp_samples_all = [json.loads(line) for line in lines]
        indices = random.sample(range(len(mbpp_samples_all)), min(N, len(mbpp_samples_all)))
        mbpp_samples = [mbpp_samples_all[i] for i in indices]

        print(f"Successfully extracted {len(mbpp_samples)} samples from MDPP")
    except Exception as e:
        print(f"Error extracting from MDPP: {e}")

    # 9. HumanEval dataset
    try:
        humaneval_path = '/data/taofeng2/Router_bench/dataset/Code/HumanEval.jsonl'
        with open(humaneval_path, 'r') as f:
            lines = f.readlines()

        humaneval_samples_all = [json.loads(line) for line in lines]
        indices = random.sample(range(len(humaneval_samples_all)), min(N, len(humaneval_samples_all)))
        humaneval_samples = [humaneval_samples_all[i] for i in indices]

        print(f"Successfully extracted {len(humaneval_samples)} samples from HumanEval")
    except Exception as e:
        print(f"Error extracting from HumanEval: {e}")

    # 10. GSM8K dataset
    try:
        gsm8k = load_dataset('gsm8k', 'main',
                             cache_dir='/data/taofeng2/Router_bench/dataset/Math')
        # Get N random samples from the training split (or another split if available)
        if split == "train":
            split_name = 'train' if 'train' in gsm8k else list(gsm8k.keys())[0]
        else:
            split_name = 'test' if 'test' in gsm8k else list(gsm8k.keys())[0]
        indices = random.sample(range(len(gsm8k[split_name])), min(N, len(gsm8k[split_name])))
        gsm8k_samples = [gsm8k[split_name][i] for i in indices]
        print(f"Successfully extracted {len(gsm8k_samples)} samples from GSM8K")
    except Exception as e:
        print(f"Error extracting from GSM8K: {e}")

    # 11. CommonsenseQA dataset
    try:
        commonsense_qa = load_dataset('commonsense_qa',
                                      cache_dir='/data/taofeng2/Router_bench/dataset/Commonsense Reasoning')
        # Get N random samples from the training split (or another split if available)
        if split == "train":
            split_name = 'train' if 'train' in commonsense_qa else list(commonsense_qa.keys())[0]
        else:
            split_name = 'validation' if 'validation' in commonsense_qa else list(commonsense_qa.keys())[0]
        indices = random.sample(range(len(commonsense_qa[split_name])), min(N, len(commonsense_qa[split_name])))
        commonsense_qa_samples = [commonsense_qa[split_name][i] for i in indices]
        print(f"Successfully extracted {len(commonsense_qa_samples)} samples from CommonsenseQA")
    except Exception as e:
        print(f"Error extracting from CommonsenseQA: {e}")

    # 12. Hellaswag dataset
    try:
        hellaswag = load_dataset('Rowan/hellaswag',
                                 cache_dir='/data/taofeng2/Router_bench/dataset/Commonsense Reasoning')
        # Get N random samples from the training split (or another split if available)
        if split == "train":
            split_name = 'train' if 'train' in hellaswag else list(hellaswag.keys())[0]
        else:
            split_name = 'validation' if 'validation' in hellaswag else list(hellaswag.keys())[0]
        indices = random.sample(range(len(hellaswag[split_name])), min(N, len(hellaswag[split_name])))
        hellaswag_samples = [hellaswag[split_name][i] for i in indices]
        print(f"Successfully extracted {len(hellaswag_samples)} samples from Hellaswag")
    except Exception as e:
        print(f"Error extracting from Hellaswag: {e}")

    # 13. ARC-Challenge dataset
    try:
        arc_challenge = load_dataset('allenai/ai2_arc', 'ARC-Challenge',
                                     cache_dir='/data/taofeng2/Router_bench/dataset/Commonsense Reasoning')
        # Get N random samples from the training split (or another split if available)
        if split == "train":
            split_name = 'train' if 'train' in arc_challenge else list(arc_challenge.keys())[0]
        else:
            split_name = 'test' if 'test' in arc_challenge else list(arc_challenge.keys())[0]
        indices = random.sample(range(len(arc_challenge[split_name])), min(N, len(arc_challenge[split_name])))
        arc_challenge_samples = [arc_challenge[split_name][i] for i in indices]
        print(f"Successfully extracted {len(arc_challenge_samples)} samples from ARC-Challenge")
    except Exception as e:
        print(f"Error extracting from ARC-Challenge: {e}")

    # 14. OpenbookQA dataset
    try:
        openbook_qa = load_dataset('allenai/openbookqa', 'main',
                                   cache_dir='/data/taofeng2/Router_bench/dataset/Commonsense Reasoning')
        # Get N random samples from the training split (or another split if available)
        if split == "train":
            split_name = 'train' if 'train' in openbook_qa else list(openbook_qa.keys())[0]
        else:
            split_name = 'test' if 'test' in openbook_qa else list(openbook_qa.keys())[0]
        indices = random.sample(range(len(openbook_qa[split_name])), min(N, len(openbook_qa[split_name])))
        openbook_qa_samples = [openbook_qa[split_name][i] for i in indices]
        print(f"Successfully extracted {len(openbook_qa_samples)} samples from OpenbookQA")
    except Exception as e:
        print(f"Error extracting from OpenbookQA: {e}")

    # 15. MATH dataset
    try:
        CATEGORY = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory',
                    'prealgebra', 'precalculus']
        for cate in CATEGORY:
            math = load_dataset('EleutherAI/hendrycks_math', cate,
                                cache_dir='/data/taofeng2/Router_bench/dataset/Math')
            # Get N random samples from the training split (or another split if available)
            if split == "train":
                split_name = 'train' if 'train' in math else list(math.keys())[0]
            else:
                split_name = 'test' if 'test' in math else list(math.keys())[0]
            indices = random.sample(range(len(math[split_name])), min(N // len(CATEGORY) + 1, len(math[split_name])))
            math_samples.extend([math[split_name][i] for i in indices])
        print(f"Successfully extracted {len(math_samples)} samples from MATH")
    except Exception as e:
        print(f"Error extracting from MATH: {e}")

    return {
        "natural_qa": natural_qa_samples,
        "trivia_qa": trivia_qa_samples,
        "squad": squad_samples,
        "quac": quac_samples,
        "boolq": boolq_samples,
        "mmlu": mmlu_samples,
        'gpqa': gpqa_samples,
        'mbpp': mbpp_samples,
        'human_eval': humaneval_samples,
        'gsm8k': gsm8k_samples,
        'commonsense_qa': commonsense_qa_samples,
        'math': math_samples,
        'openbook_qa': openbook_qa_samples,
        'arc_challenge': arc_challenge_samples,
        'hellaswag': hellaswag_samples,
    }


task_description = {}
task_description[
    "natural_qa"] = 'Natural Questions consists of real Google search queries paired with full Wikipedia articles. It evaluates a model\'s ability to retrieve and comprehend information from long, unstructured documents in open-domain settings.'
task_description[
    "trivia_qa"] = 'TriviaQA features complex trivia-style questions with evidence from multiple web sources. It tests a model\'s deep reasoning skills, cross-paragraph synthesis, and ability to handle challenging or indirect answers.'
task_description[
    "squad"] = 'SQuAD provides questions based on short Wikipedia passages where the answer is explicitly found in the text. It measures sentence-level comprehension and the model\'s precision in extracting factual information from concise contexts.'
task_description[
    "quac"] = 'QuAC is a conversational QA dataset where each question builds on the previous dialogue turn. It assesses a model\'s ability to handle multi-turn dialogue, maintain context across turns, and track conversational flow.'
task_description[
    "boolq"] = 'BoolQ contains yes/no questions based on a given paragraph, written in natural language. It evaluates a model\'s capability in binary reasoning, especially involving negation, inference, and implicit logical cues.'
task_description[
    "gsm8k"] = 'GSM8K is a benchmark of grade school math word problems designed to evaluate a model\'s numerical reasoning, problem-solving skills, and ability to generate step-by-step solutions using arithmetic and logical reasoning.'
task_description[
    "commonsense_qa"] = 'CommonsenseQA is a multiple-choice question dataset that requires models to apply commonsense knowledge beyond factual recall. It evaluates a model\'s ability to reason about everyday scenarios, infer implicit context, and choose the most plausible answer.'
task_description[
    "mmlu"] = 'MMLU (Massive Multitask Language Understanding) covers 57 subjects ranging from STEM to humanities, evaluating a model\'s breadth of knowledge and ability to apply concepts across multiple domains with varying complexity.'
task_description[
    "gpqa"] = 'GPQA evaluates a model\'s ability to answer challenging graduate-level multiple-choice questions spanning physics, chemistry, biology, and other scientific fields.'
task_description[
    "mbpp"] = 'MBPP (Mostly Basic Python Programming) features Python programming tasks of varying complexity with test cases, measuring a model\'s ability to generate syntactically correct and functionally accurate Python code.'
task_description[
    "human_eval"] = 'HumanEval is a challenging programming benchmark that evaluates a model\'s ability to both understand problem descriptions and generate code that implements the required functionality correctly.'
task_description[
    "math"] = 'MATH is a dataset of high school and competition-level mathematics problems, requiring detailed multi-step solutions across algebra, geometry, calculus, and more. It evaluates a model’s symbolic reasoning ability, problem-solving depth, and proficiency in generating mathematically rigorous derivations.'
task_description[
    "arc_challenge"] = 'ARC-Challenge is a benchmark of difficult grade-school science questions requiring complex reasoning, knowledge retrieval, and elimination strategies. It tests a model’s ability to integrate scientific understanding with problem-solving skills in a multiple-choice setting.'
task_description[
    "hellaswag"] = 'HellaSwag is a challenging commonsense reasoning benchmark featuring sentence completion tasks with deceptively similar distractors. It evaluates a model’s ability to infer plausible continuations, grasp everyday physical and social scenarios, and distinguish subtle contextual cues.'
task_description[
    "openbook_qa"] = 'OpenbookQA consists of elementary science questions that require combining core scientific facts with broad commonsense knowledge. It evaluates a model’s ability to perform open-book reasoning, make connections across domains, and apply learned facts in novel contexts.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--case_num', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--round', type=int, default=5)
    parser.add_argument('--cache_save_path', type=str, default="/data/taofeng2/Router_bench/router_data/cache")
    args = parser.parse_args()

    SPLIT = args.split
    CASE_NUM = args.case_num
    SEED = args.seed
    random.seed(SEED)
    CACHE_SAVE_PATH = args.cache_save_path
    ROUND = args.round

    data_all = []
    samples = get_n_samples(N=CASE_NUM, random_seed=SEED, split=SPLIT)
    for inter in samples:
        if inter == "natural_qa":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['query'] = inter_['question']
                case[
                    'query_think'] = inter_['question'] + "\nLet's think step by step."
                case['gt'] = inter_['golden_answers'][0]
                case['metric'] = 'cem'
                data_all.append(case)
        elif inter == "trivia_qa":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['query'] = inter_['question']
                case[
                    'query_think'] = inter_['question'] + "\nLet's think step by step."
                case['gt'] = inter_['answer']['normalized_aliases'][0]
                case['metric'] = 'cem'
                data_all.append(case)
        elif inter == "quac":
            for inter_ in samples[inter]:
                context = inter_['context']
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['query'] = format_quac_prompt(inter_, 0)
                case['query_think'] = format_quac_prompt(inter_, 0, thought=True)
                case['gt'] = inter_['answers']['texts'][0][0]
                case["metric"] = "f1_score"
                data_all.append(case)
        elif inter == "squad":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['query'] = format_qa_prompt(inter_['context'], inter_['question'])
                case['query_think'] = format_qa_prompt(inter_['context'], inter_['question'], thought=True)
                case['gt'] = inter_['answers']['text'][0]
                case['metric'] = 'cem'
                data_all.append(case)
        elif inter == "boolq":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['query'] = format_bool_prompt(inter_['passage'], inter_['question'])
                case['query_think'] = format_bool_prompt(inter_['passage'], inter_['question'], thought=True)
                case['gt'] = str(inter_['answer'])
                case['metric'] = 'cem'
                data_all.append(case)
        elif inter == "mmlu":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['query'] = format_mc_prompt(inter_['question'], inter_['choices'])
                # For think version, we could modify this to include reasoning
                case[
                    'query_think'] = format_mc_prompt(inter_['question'],
                                                      inter_['choices']) + "\nLet's think step by step."
                answer_index = inter_['answer']  # this is an index, like 0, 1, 2, 3
                letter_answer = chr(65 + answer_index)  # Convert the index to A, B, C, D
                case['gt'] = letter_answer
                case['metric'] = 'em_mc'
                data_all.append(case)
        elif inter == "gpqa":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                # Prepare the multiple choice options
                options = [
                    inter_['Correct Answer'], inter_['Incorrect Answer 1'], inter_['Incorrect Answer 2'],
                    inter_['Incorrect Answer 3']]
                correct_index = 0  # Index of correct answer before shuffling
                mapping = list(range(len(options)))  # Shuffle
                random.shuffle(mapping)
                new_correct_index = mapping.index(correct_index)
                shuffled_options = [options[mapping.index(i)] for i in range(len(options))]

                case['query'] = format_mc_prompt(inter_['Question'], shuffled_options)
                case[
                    'query_think'] = format_mc_prompt(inter_['Question'],
                                                      shuffled_options) + "\nLet's think step by step."
                letter_answer = chr(65 + new_correct_index)
                case['gt'] = letter_answer
                case['metric'] = 'em_mc'
                data_all.append(case)
        elif inter == "mbpp":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['task_id'] = inter_['task_id']
                case['query'] = format_mbpp_prompt(inter_['text'], inter_['test_list'])
                # print(case['query'])
                case[
                    'query_think'] = format_mbpp_prompt(inter_['text'],
                                                        inter_['test_list']) + "\nLet's think step by step."
                case['gt'] = inter_['test_list']
                case['metric'] = 'code_eval'
                data_all.append(case)
        elif inter == "human_eval":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['query'] = format_humaneval_prompt(inter_['prompt'])
                case[
                    'query_think'] = format_humaneval_prompt(inter_['prompt']) + "\nLet's think step by step."
                case['gt'] = inter_['test']
                case['task_id'] = inter_['task_id']
                case['metric'] = 'code_eval'
                data_all.append(case)
        elif inter == "gsm8k":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['query'] = format_gsm8k_prompt(query=inter_['question'], thought=False)
                case['query_think'] = format_gsm8k_prompt(query=inter_['question'], thought=True)
                case['gt'] = inter_['answer']
                case['metric'] = 'GSM8K'
                data_all.append(case)
        elif inter == "commonsense_qa":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['query'] = format_commonsense_qa_prompt(query=inter_['question'], choices=inter_['choices'],
                                                             thought=False)
                case['query_think'] = format_commonsense_qa_prompt(query=inter_['question'], choices=inter_['choices'],
                                                                   thought=True)
                case['gt'] = inter_['answerKey']
                case['metric'] = 'em_mc'
                data_all.append(case)
        elif inter == "math":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['query'] = format_math_prompt(query=inter_['problem'], thought=False)
                case['query_think'] = format_math_prompt(query=inter_['problem'], thought=True)
                case['gt'] = inter_['solution']
                case['metric'] = 'MATH'
                data_all.append(case)
        elif inter == "openbook_qa":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['query'] = format_commonsense_qa_prompt(query=inter_['question_stem'], choices=inter_['choices'],
                                                             thought=False)
                case['query_think'] = format_commonsense_qa_prompt(query=inter_['question_stem'],
                                                                   choices=inter_['choices'],
                                                                   thought=True)
                case['gt'] = inter_['answerKey']
                case['metric'] = 'em_mc'
                data_all.append(case)
        elif inter == "arc_challenge":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                case['query'] = format_commonsense_qa_prompt(query=inter_['question'], choices=inter_['choices'],
                                                             thought=False)
                case['query_think'] = format_commonsense_qa_prompt(query=inter_['question'], choices=inter_['choices'],
                                                                   thought=True)
                case['gt'] = inter_['answerKey']
                case['metric'] = 'em_mc'
                data_all.append(case)
        elif inter == "hellaswag":
            for inter_ in samples[inter]:
                case = {}
                case['task_name'] = inter
                case['task_description'] = task_description[inter]
                ctx = inter_["ctx_a"] + " " + inter_["ctx_b"].capitalize()
                input_query = hellaswag_preprocess(inter_["activity_label"] + ": " + ctx)
                input_choices = dict()
                input_choices["text"] = [hellaswag_preprocess(ending) for ending in inter_["endings"]]
                input_choices["label"] = ["A", "B", "C", "D"]
                case['query'] = format_hellaswag_prompt(query=input_query, choices=input_choices,
                                                        thought=False)
                case['query_think'] = format_hellaswag_prompt(query=input_query, choices=input_choices,
                                                              thought=True)
                case['gt'] = chr(ord('A') + int(inter_['label']))
                case['metric'] = 'em_mc'
                data_all.append(case)
        else:
            raise Exception("Error")

    with open("/data/taofeng2/Router_bench/configs/config_test.yaml", 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    data_building(qa_path=data_all,llm_path=config['llm_description_path'],config=config, round=ROUND)