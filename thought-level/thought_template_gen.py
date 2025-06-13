#!/usr/bin/env python3
"""
Thought Template Generation Script

This script generates thought templates for different types of queries by analyzing
high-performing responses from small language models (7B-12B parameters).
"""

import os
import pickle
import time
import threading
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd
import numpy as np
import re
import json
from collections import defaultdict
from utils import model_prompting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CHECKPOINT_DIR = Path("./checkpoints_thought_template_hybrid_small_8b")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# List of small models to filter by
SMALL_MODELS = [
    "qwen2-7b-instruct",
    "qwen2-7b-instruct_think",
    "qwen2.5-7b-instruct",
    "qwen2.5-7b-instruct_think",
    "gemma-7b",
    "gemma-7b_think",
    "codegemma-7b",
    "codegemma-7b_think",
    "gemma-2-9b-it",
    "gemma-2-9b-it_think",
    "llama-3.1-8b-instruct",
    "llama-3.1-8b-instruct_think",
    "granite-3.0-8b-instruct",
    "granite-3.0-8b-instruct_think",
    "llama3-chatqa-1.5-8b",
    "llama3-chatqa-1.5-8b_think",
    "mistral-nemo-12b-instruct",
    "mistral-nemo-12b-instruct_think",
    "mistral-7b-instruct-v0.3",
    "mistral-7b-instruct-v0.3_think"
]

@dataclass
class Config:
    """Configuration for thought template generation."""
    file_path: str
    top_n: int = 3
    output_path: str = "router_analysis_results.json"
    template_dataset_path: str = "thought_templates_hybrid.csv"
    llm_model: str = "nvdev/nvidia/llama-3.1-8b-instruct"
    queries_per_task: int = 3
    p_num: int = 6
    checkpoint_interval: int = 1000
    retry_failed: bool = False
    max_retries: int = 3
    error_threshold: float = 0.1

class CheckpointManager:
    """Manages checkpoints for query processing."""
    
    def __init__(self, prefix: str = "query_checkpoint", save_interval: int = 10):
        self.prefix = prefix
        self.save_interval = save_interval
        self.results: Dict[str, Tuple] = {}
        self.save_lock = threading.Lock()
        self.processed_count = 0
        self.error_count = 0
        self.load_latest_checkpoint()
    
    def add_result(self, query: str, best_performers: pd.DataFrame, template: str, success: bool) -> None:
        """Add a processed query result."""
        self.results[query] = (best_performers, template, success)
        self.processed_count += 1
        
        if not success:
            self.error_count += 1
            
        if self.processed_count % self.save_interval == 0:
            self.save_checkpoint()
            logger.info(f"Progress: Processed {self.processed_count} queries, Errors: {self.error_count}")
    
    def save_checkpoint(self) -> None:
        """Save current results to a checkpoint file."""
        timestamp = int(time.time())
        filename = f"{self.prefix}_{timestamp}.pkl"
        filepath = CHECKPOINT_DIR / filename
        tmp_filepath = filepath.with_suffix('.pkl.tmp')
        
        with self.save_lock:
            with open(tmp_filepath, 'wb') as f:
                pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_filepath, filepath)
    
    def load_latest_checkpoint(self) -> None:
        """Load most recent checkpoint if available."""
        checkpoint_files = list(CHECKPOINT_DIR.glob(f"{self.prefix}_*.pkl"))
        
        if not checkpoint_files:
            logger.info("No checkpoints found. Starting fresh.")
            return
        
        # Sort by timestamp (newest first)
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_checkpoint, 'rb') as f:
                self.results = pickle.load(f)
            
            # Count success/failures in loaded results
            self.processed_count = len(self.results)
            self.error_count = sum(1 for _, _, success in self.results.values() if not success)
            
            logger.info(f"Loaded checkpoint with {self.processed_count} queries, Errors: {self.error_count}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            self.results = {}
    
    def get_successful_queries(self) -> List[str]:
        """Get list of successfully processed queries."""
        return [query for query, (_, _, success) in self.results.items() if success]
    
    def get_failed_queries(self) -> List[str]:
        """Get list of failed queries."""
        return [query for query, (_, _, success) in self.results.items() if not success]
    
    def get_best_performers(self, query: str) -> Optional[pd.DataFrame]:
        """Get best performers for a query if successful."""
        if query in self.results and self.results[query][2]:  # Check success flag
            return self.results[query][0]
        return None
    
    def get_template(self, query: str) -> Optional[str]:
        """Get template for a query if successful."""
        if query in self.results and self.results[query][2]:  # Check success flag
            return self.results[query][1]
        return None

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the router dataset from a file."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix == '.json':
        return pd.read_json(file_path)
    elif file_path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def normalize_query_boolq(q: str) -> str:
    """Special normalization function for BoolQ queries."""
    if not isinstance(q, str):
        return q
    q = re.sub(r'\s+', ' ', q)
    q = re.sub(r"\s*let'?s think step by step\.?\s*$", "", q, flags=re.IGNORECASE)
    return q.strip()

def group_by_task_and_query(data: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Group the data first by task_name and then by query."""
    data_copy = data.copy()
    
    def normalize_with_task(row):
        if row['task_name'] == 'boolq':
            return normalize_query_boolq(row['query'])
        else:
            return re.sub(r"\s*let's think step by step\.?\s*$", "", row['query'], flags=re.IGNORECASE).strip() if isinstance(row['query'], str) else row['query']
    
    data_copy['normalized_query'] = data_copy.apply(normalize_with_task, axis=1)
    
    task_query_groups = defaultdict(dict)
    
    for task_name, task_group in data_copy.groupby('task_name'):
        for query, query_group in task_group.groupby('normalized_query'):
            task_query_groups[task_name][query] = query_group
    
    return task_query_groups

def find_best_performers(query_group: pd.DataFrame, n: int) -> pd.DataFrame:
    """Find the top n performers based on performance and cost."""
    first_stage = query_group.sort_values(by=['performance', 'cost'], ascending=[False, True])
    first_stage_performers = first_stage.head(5)
    second_stage = first_stage_performers.sort_values(by=['llm_judge', 'cost'], ascending=[False, True])
    return second_stage.head(n)

def generate_thought_template_prompt(query: str, best_responses: List[str]) -> str:
    """Create a prompt for the LLM to generate a thought template."""
    prompt = f"""Given this question and example solutions, extract a concise thought template that captures the effective reasoning pattern and can serve as guidance:

Question: {query}

Here are {len(best_responses)} high-performing solutions:
"""
    
    for i, response in enumerate(best_responses, 1):
        if len(response) > 1000:
            response = response[:1000] + "... [truncated]"
        prompt += f"Solution {i}:\n{response}\n\n"
    
    prompt += """Create a concise and clear thought template (1-5 sentences total) focusing on:

1. Core Task Summarization: Identify the core problem type and general appraoch needed (in one very concise sentence)
2. Reasoning Step: Give a clear chain of thought that can help model address this problem (1-3 sentences)
2. Answer Template: What is the prefered answer format / structure for the given query (1 sentence) 

Your template should be specific enough to guide similar problems but general enough to work across variations. Use 1-2 sentences for straightforward problems and 3-5 sentences for complex ones.

Thought Template:"""
    
    return prompt

def generate_thought_template_with_llm(
    query: str, 
    task_description: str,
    best_performers: pd.DataFrame,
    llm_model: str = "deepseek-ai/deepseek-r1",
    max_token_num: int = 1024,
) -> str:
    """Generate a thought template using the specified LLM."""
    responses = best_performers['response'].tolist()
    prompt = generate_thought_template_prompt(query, responses)
    
    try:
        thought_template = model_prompting(
            llm_model=llm_model,
            prompt=prompt,
            max_token_num=max_token_num,
            temperature=0.2,
            stream=False
        )
    except AttributeError:
        from openai import OpenAI
        client = OpenAI(
            base_url="",
            api_key="",
            timeout=300,
            max_retries=2
        )
        
        completion = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_token_num,
            temperature=0.2,
            stream=False
        )
        
        thought_template = completion.choices[0].message.content
    
    return thought_template

def process_query_parallel(query_data: Tuple[str, pd.DataFrame, str, int]) -> Tuple[str, Optional[pd.DataFrame], str, bool]:
    """Process a single query with error handling."""
    query, group, llm_model, top_n = query_data
    
    try:
        small_model_group = group[group['llm'].isin(SMALL_MODELS)]

        if small_model_group.empty:
            raise ValueError(f"No small models found for query: {query[:50]}...")
        
        best_perf = find_best_performers(small_model_group, top_n)
        task_desc = group['task_description'].iloc[0]
        template = generate_thought_template_with_llm(
            query=query,
            task_description=task_desc,
            best_performers=best_perf,
            llm_model=llm_model
        )
        
        return (query, best_perf, template, True)
    except Exception as e:
        logger.error(f"Error processing query '{query[:50]}...': {e}")
        return (query, None, str(e), False)

def create_summarized_dataset(
    data: pd.DataFrame,
    selected_queries: Dict[str, List[str]],
    query_groups: Dict[str, pd.DataFrame],
    templates: Dict[str, str]
) -> pd.DataFrame:
    """Create a new dataset with the thought templates."""
    unique_data = []
    
    for task_name, queries in selected_queries.items():
        for query in queries:
            sample_row = query_groups[query].iloc[0].copy()
            
            new_row = {
                'task_name': sample_row['task_name'],
                'task_id': sample_row['task_id'],
                'task_description': sample_row['task_description'],
                'task_description_embedding': sample_row['task_description_embedding'],
                'query': sample_row['query'],
                'query_embedding': sample_row['query_embedding'],
                'ground_truth': sample_row['ground_truth'],
                'metric': sample_row['metric'],
                'thought_template': templates[query]
            }
            
            unique_data.append(new_row)
    
    return pd.DataFrame(unique_data)

def analyze_router_dataset(config: Config) -> Dict[str, Any]:
    """Main function to analyze the router dataset."""
    logger.info(f"Loading dataset from {config.file_path}...")
    data = load_dataset(config.file_path)
    
    logger.info("Grouping data by task and query...")
    task_query_groups = group_by_task_and_query(data)
    logger.info(f"Found {len(task_query_groups)} unique tasks")
    
    all_query_groups = {}
    for task_name, query_dict in task_query_groups.items():
        for query, group in query_dict.items():
            all_query_groups[query] = group
    
    selected_queries = {}
    for task_name, query_dict in task_query_groups.items():
        task_queries = list(query_dict.keys())
        n_to_select = min(config.queries_per_task, len(task_queries))
        selected = task_queries[:n_to_select]
        selected_queries[task_name] = selected
        logger.info(f"Task '{task_name}': Selected {len(selected)} queries out of {len(task_queries)}")
    
    queries_to_process = []
    for task_queries in selected_queries.values():
        queries_to_process.extend(task_queries)
    
    checkpoint_prefix = Path(config.file_path).stem + "_checkpoint"
    checkpoint_mgr = CheckpointManager(prefix=checkpoint_prefix, save_interval=config.checkpoint_interval)
    
    if config.retry_failed:
        failed_queries = checkpoint_mgr.get_failed_queries()
        queries_to_process = [q for q in queries_to_process if q in failed_queries]
        logger.info(f"Retrying {len(queries_to_process)} failed queries from previous run")
    else:
        successful_queries = checkpoint_mgr.get_successful_queries()
        queries_to_process = [q for q in queries_to_process if q not in successful_queries]
    
    logger.info(f"Processing {len(queries_to_process)} queries across {len(selected_queries)} tasks")
    
    if not queries_to_process:
        logger.info("All queries already processed successfully. Skipping to results collection.")
    else:
        retry_count = 0
        while retry_count < config.max_retries:
            parallel_args = [(query, all_query_groups[query], config.llm_model, config.top_n) 
                           for query in queries_to_process]
            
            logger.info(f"Processing queries in parallel with {config.p_num} workers... (Attempt {retry_count + 1}/{config.max_retries})")
            
            with ThreadPool(config.p_num) as pool:
                for query, best_perf, template, success in tqdm(
                    pool.imap_unordered(process_query_parallel, parallel_args),
                    total=len(parallel_args),
                    desc="Processing queries"
                ):
                    checkpoint_mgr.add_result(query, best_perf, template, success)
            
            checkpoint_mgr.save_checkpoint()
            
            failed_queries = checkpoint_mgr.get_failed_queries()
            error_rate = len(failed_queries) / len(queries_to_process)
            
            logger.info(f"\nAttempt {retry_count + 1} completed. Error rate: {error_rate:.2%}")
            
            if error_rate <= config.error_threshold:
                logger.info(f"Error rate ({error_rate:.2%}) is below threshold ({config.error_threshold:.2%}). Stopping retries.")
                break
            
            if retry_count < config.max_retries - 1:
                logger.info(f"Error rate above threshold. Retrying {len(failed_queries)} failed queries...")
                queries_to_process = failed_queries
            else:
                logger.info(f"Maximum retries reached. {len(failed_queries)} queries still failed.")
            
            retry_count += 1
    
    best_performers = {}
    templates = {}
    
    for query in all_query_groups.keys():
        best_perf = checkpoint_mgr.get_best_performers(query)
        template = checkpoint_mgr.get_template(query)
        
        if best_perf is not None and template is not None:
            best_performers[query] = best_perf
            templates[query] = template
    
    failed_queries = checkpoint_mgr.get_failed_queries()
    if failed_queries:
        logger.warning(f"\nWarning: {len(failed_queries)} queries failed processing after {retry_count + 1} attempts:")
        for q in failed_queries[:min(5, len(failed_queries))]:
            logger.warning(f"  - '{q[:50]}...'")
        if len(failed_queries) > 5:
            logger.warning(f"  ... and {len(failed_queries) - 5} more")
    
    logger.info("Creating summarized dataset...")
    filtered_selected_queries = {}
    for task_name, queries in selected_queries.items():
        filtered_selected_queries[task_name] = [q for q in queries if q in templates]
    
    summarized_df = create_summarized_dataset(
        data=data,
        selected_queries=filtered_selected_queries,
        query_groups=all_query_groups,
        templates=templates
    )
    
    if config.template_dataset_path:
        logger.info(f"Saving summarized dataset with thought templates to {config.template_dataset_path}...")
        summarized_df.to_csv(config.template_dataset_path, index=False)
    
    logger.info("Formatting detailed results...")
    formatted_results = {}
    for query, perf in best_performers.items():
        if query not in templates:
            continue
            
        group = all_query_groups[query]
        formatted_results[query] = {
            'task_id': group['task_id'].iloc[0],
            'task_name': group['task_name'].iloc[0],
            'task_description': group['task_description'].iloc[0],
            'query': query,
            'ground_truth': group['ground_truth'].iloc[0],
            'metric': group['metric'].iloc[0],
            'best_performers': perf.to_dict(orient='records'),
            'thought_template': templates[query]
        }
    
    if config.output_path:
        logger.info(f"Saving detailed results to {config.output_path}...")
        with open(config.output_path, 'w') as f:
            json.dump(formatted_results, f, indent=2)
    
    logger.info("\nTask distribution in processed dataset:")
    task_counts = {}
    for task_name, queries in filtered_selected_queries.items():
        task_counts[task_name] = len(queries)
    
    for task_name, count in task_counts.items():
        logger.info(f"- {task_name}: {count} queries")
    
    success_count = len(formatted_results)
    total_queries = len(queries_to_process) + len(checkpoint_mgr.get_successful_queries())
    logger.info(f"\nAnalysis complete! Successfully processed {success_count} out of {total_queries} queries")
    logger.info(f"Top {config.top_n} performers and thought templates for each query saved to {config.output_path}")
    logger.info(f"Summarized dataset with thought templates saved to {config.template_dataset_path}")
    
    return formatted_results

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze router dataset")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the router dataset file")
    parser.add_argument("--top_n", type=int, default=3, help="Number of top performers to select for each query")
    parser.add_argument("--output_path", type=str, default="router_analysis_results.json", 
                        help="Path to save the detailed results to")
    parser.add_argument("--template_dataset_path", type=str, default="",
                        help="Path to save the summarized dataset with templates")
    parser.add_argument("--llm_model", type=str, default="nvdev/nvidia/llama-3.1-8b-instruct",
                        help="Model to use for generating templates")
    parser.add_argument("--queries_per_task", type=int, default=3,
                        help="Number of queries to process per task type")
    parser.add_argument("--p_num", type=int, default=6,
                        help="Number of parallel workers")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="How often to save checkpoints (number of queries)")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Retry only failed queries from previous run")
    
    args = parser.parse_args()
    
    config = Config(
        file_path=args.file_path,
        top_n=args.top_n,
        output_path=args.output_path,
        template_dataset_path=args.template_dataset_path,
        llm_model=args.llm_model,
        queries_per_task=args.queries_per_task,
        p_num=args.p_num,
        checkpoint_interval=args.checkpoint_interval,
        retry_failed=args.retry_failed
    )
    
    analyze_router_dataset(config)

if __name__ == "__main__":
    main()