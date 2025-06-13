#this file contains the code to generate thought templates. Given a trainset, we summarize thought template for each query
import os
import pickle
import time
import threading
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import re
import json
from collections import defaultdict
from utils import model_prompting

# Create checkpoint directory
CHECKPOINT_DIR = "./checkpoints_thought_template_hybrid_small_8b"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

class CheckpointManager:
    """Simple checkpoint manager for query processing."""
    
    def __init__(self, prefix="query_checkpoint", save_interval=10):
        self.prefix = prefix
        self.save_interval = save_interval
        self.results = {}  # query -> (best_performers, template, success)
        self.save_lock = threading.Lock()
        self.processed_count = 0
        self.error_count = 0
        self.load_latest_checkpoint()
    
    def add_result(self, query, best_performers, template, success):
        """Add a processed query result."""
        self.results[query] = (best_performers, template, success)
        self.processed_count += 1
        
        if not success:
            self.error_count += 1
            
        if self.processed_count % self.save_interval == 0:
            self.save_checkpoint()
            print(f"Progress: Processed {self.processed_count} queries, Errors: {self.error_count}")
    
    def save_checkpoint(self):
        """Save current results to a checkpoint file."""
        timestamp = int(time.time())
        filename = f"{self.prefix}_{timestamp}.pkl"
        filepath = os.path.join(CHECKPOINT_DIR, filename)
        tmp_filepath = f"{filepath}.tmp"
        
        with self.save_lock:
            with open(tmp_filepath, 'wb') as f:
                pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_filepath, filepath)
    
    def load_latest_checkpoint(self):
        """Load most recent checkpoint if available."""
        checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) 
                           if f.startswith(self.prefix) and f.endswith('.pkl')]
        
        if not checkpoint_files:
            print("No checkpoints found. Starting fresh.")
            return
        
        # Sort by timestamp (newest first)
        checkpoint_files.sort(reverse=True)
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoint_files[0])
        
        try:
            with open(latest_checkpoint, 'rb') as f:
                self.results = pickle.load(f)
            
            # Count success/failures in loaded results
            self.processed_count = len(self.results)
            self.error_count = sum(1 for _, _, success in self.results.values() if not success)
            
            print(f"Loaded checkpoint with {self.processed_count} queries, Errors: {self.error_count}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            self.results = {}
    
    def get_successful_queries(self):
        """Get list of successfully processed queries."""
        return [query for query, (_, _, success) in self.results.items() if success]
    
    def get_failed_queries(self):
        """Get list of failed queries."""
        return [query for query, (_, _, success) in self.results.items() if not success]
    
    def get_best_performers(self, query):
        """Get best performers for a query if successful."""
        if query in self.results and self.results[query][2]:  # Check success flag
            return self.results[query][0]
        return None
    
    def get_template(self, query):
        """Get template for a query if successful."""
        if query in self.results and self.results[query][2]:  # Check success flag
            return self.results[query][1]
        return None

# Include these original functions from your code
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the router dataset from a file.
    Handles different file formats like CSV, JSON, or parquet.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        DataFrame containing the router dataset
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def normalize_query_boolq(q):
    """
    Special normalization function for BoolQ queries that handles whitespace properly.
    
    Args:
        q: Query string
        
    Returns:
        Normalized query string
    """
    if not isinstance(q, str):
        return q
    # First normalize all whitespace (convert newlines, tabs, multiple spaces to single space)
    q = re.sub(r'\s+', ' ', q)
    # Then remove the phrase with a more flexible regex
    q = re.sub(r"\s*let'?s think step by step\.?\s*$", "", q, flags=re.IGNORECASE)
    # Trim any remaining whitespace
    return q.strip()

def group_by_task_and_query(data: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Group the data first by task_name and then by query.
    
    Args:
        data: DataFrame containing the router dataset
        
    Returns:
        Nested dictionary mapping task_name -> query -> dataframe
    """
    # Create a copy
    data_copy = data.copy()
    
    # Handle different normalization for different tasks
    def normalize_with_task(row):
        if row['task_name'] == 'boolq':
            # Use special handling for BoolQ
            return normalize_query_boolq(row['query'])
        else:
            # Use standard normalization for other tasks
            return re.sub(r"\s*let's think step by step\.?\s*$", "", row['query'], flags=re.IGNORECASE).strip() if isinstance(row['query'], str) else row['query']
    
    # Apply normalization based on task
    data_copy['normalized_query'] = data_copy.apply(normalize_with_task, axis=1)
    
    # Create nested dictionary
    task_query_groups = defaultdict(dict)
    
    # First group by task_name
    for task_name, task_group in data_copy.groupby('task_name'):
        # Then group each task's data by normalized query
        for query, query_group in task_group.groupby('normalized_query'):
            task_query_groups[task_name][query] = query_group
    
    return task_query_groups

def find_best_performers(query_group: pd.DataFrame, n: int, metric: str = 'llm_judge') -> pd.DataFrame:
    """
    For each query, find the top n performers based on performance and cost.
    When performance is the same, select the ones with lower cost.
    
    Args:
        query_group: DataFrame containing data for a specific query
        n: Number of top performers to select
        metric: Metric to use for sorting (default: 'performance')
        
    Returns:
        DataFrame containing the top n performers
    """
    # # Sort by performance (descending) and then by cost (ascending)
    # sorted_group = query_group.sort_values(by=[metric, 'cost'], ascending=[False, True])

    # First stage: Sort by performance (descending) and cost (ascending)
    first_stage = query_group.sort_values(by=['performance', 'cost'], ascending=[False, True])
    
    # Take the top first_stage_n performers from first stage
    first_stage_performers = first_stage.head(5)
    
    # Second stage: Sort these by llm_judge (descending) and cost (ascending)
    second_stage = first_stage_performers.sort_values(by=['llm_judge', 'cost'], ascending=[False, True])

    top_performers = second_stage.head(3)

    
    # Take the top n performers
    # top_performers = sorted_group.head(n)
    
    return top_performers

def generate_thought_template_prompt(query: str, best_responses: List[str]) -> str:
    """
    Create a prompt for the LLM to generate a concise thought template based on top responses,
    with emphasis on chain-of-thought reasoning for specific problem types like GSM8K.
    
    Args:
        query: The query that was answered
        best_responses: List of top-performing responses
        
    Returns:
        Prompt for the LLM
    """
    prompt = f"""Given this question and example solutions, extract a concise thought template that captures the effective reasoning pattern and can serve as guidance:

Question: {query}

Here are {len(best_responses)} high-performing solutions:
"""
    
    # Add each response
    for i, response in enumerate(best_responses, 1):
        # Truncate very long responses
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
    # Extract responses only
    responses = best_performers['response'].tolist()
    
    # Create the prompt for the LLM without metrics
    prompt = generate_thought_template_prompt(query, responses)

    
    try:
        # Try to use the existing function with stream=False
        thought_template = model_prompting(
            llm_model=llm_model,
            prompt=prompt,
            max_token_num=max_token_num,
            temperature=0.2,
            stream=False
        )
    except AttributeError:
        # If it fails, handle it as a string-by-string collection
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
            stream=False  # Important: No streaming here
        )
        
        thought_template = completion.choices[0].message.content
    
    return thought_template

def create_summarized_dataset(
    data: pd.DataFrame,
    selected_queries: Dict[str, List[str]],
    query_groups: Dict[str, pd.DataFrame],
    templates: Dict[str, str]
) -> pd.DataFrame:
    """
    Create a new dataset with the thought templates.
    
    Args:
        data: Original dataset
        selected_queries: Dictionary mapping task_name to list of selected queries
        query_groups: Dictionary mapping each query to its corresponding data
        templates: Dictionary mapping each query to its thought template
        
    Returns:
        New DataFrame with thought templates
    """
    # Get unique task IDs, descriptions, queries, etc.
    unique_data = []
    
    for task_name, queries in selected_queries.items():
        for query in queries:
            # Take the first row for this query to get the common fields
            sample_row = query_groups[query].iloc[0].copy()
            
            # Create a new row with the fields we want
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
    
    # Create a new DataFrame
    summarized_df = pd.DataFrame(unique_data)
    
    return summarized_df

def process_query_parallel(query_data):
    """
    Process a single query with error handling.
    
    Args:
        query_data: Tuple containing (query, group, llm_model, top_n)
        
    Returns:
        Tuple containing results for this query with success flag
    """
    query, group, llm_model, top_n = query_data
    
    try:
        #This is added to only use small models' response
        small_model_group = group[group['llm'].isin(SMALL_MODELS)]

        if small_model_group.empty:
            raise ValueError(f"No small models found for query: {query[:50]}...")
        
        # Find best performers
        best_perf = find_best_performers(small_model_group, top_n)
        
        # best_perf = find_best_performers(group, top_n)
        
        # Generate thought template
        task_desc = group['task_description'].iloc[0]
        template = generate_thought_template_with_llm(
            query=query,
            task_description=task_desc,
            best_performers=best_perf,
            llm_model=llm_model
        )
        
        return (query, best_perf, template, True)  # Success flag
    except Exception as e:
        # print(f"Error processing query '{query[:50]}...': {e}")
        return (query, None, str(e), False)  # Error flag

def analyze_router_dataset(
    file_path: str, 
    top_n: int = 3, 
    output_path: str = "router_analysis_results.json",
    template_dataset_path: str = "thought_templates_hybrid.csv",
    llm_model: str = "nvdev/nvidia/llama-3.1-nemotron-70b-instruct",
    queries_per_task: int = 3,
    p_num: int = 20,
    checkpoint_interval: int = 5,
    retry_failed: bool = False,
    max_retries: int = 3,
    error_threshold: float = 0.1  # 10% error threshold
) -> Dict[str, Any]:
    """
    Main function to analyze the router dataset with checkpointing and error handling.
    Revised 05/11/2025 -> Only considers responses from small models (7b-12b parameters).
    
    Args:
        file_path: Path to the dataset file
        top_n: Number of top performers to select for each query
        output_path: Path to save the detailed results to
        template_dataset_path: Path to save the summarized dataset with templates
        llm_model: Model to use for generating templates
        queries_per_task: Number of queries to process per task type
        p_num: Number of concurrent processes/threads
        checkpoint_interval: How often to save checkpoints (in queries)
        retry_failed: Whether to retry failed queries from previous run
        max_retries: Maximum number of retry attempts for failed queries
        error_threshold: Maximum acceptable error rate (0.0 to 1.0)
        
    Returns:
        Formatted results dictionary
    """
    # Load dataset
    print(f"Loading dataset from {file_path}...")
    data = load_dataset(file_path)
    
    # Group by task and then by query
    print(f"Grouping data by task and query...")
    task_query_groups = group_by_task_and_query(data)
    print(f"Found {len(task_query_groups)} unique tasks")
    
    # Flatten the groups for easier processing while preserving task info
    all_query_groups = {}
    for task_name, query_dict in task_query_groups.items():
        for query, group in query_dict.items():
            all_query_groups[query] = group
    
    # Select N queries per task
    selected_queries = {}
    for task_name, query_dict in task_query_groups.items():
        # Get list of queries for this task
        task_queries = list(query_dict.keys())
        
        # Select minimum of queries_per_task or available queries
        n_to_select = min(queries_per_task, len(task_queries))
        selected = task_queries[:n_to_select]
        
        selected_queries[task_name] = selected
        print(f"Task '{task_name}': Selected {len(selected)} queries out of {len(task_queries)}")
    
    # Flatten the selected queries for processing
    queries_to_process = []
    for task_queries in selected_queries.values():
        queries_to_process.extend(task_queries)
    
    # Initialize checkpoint manager
    checkpoint_prefix = os.path.basename(file_path).split('.')[0] + "_checkpoint"
    checkpoint_mgr = CheckpointManager(prefix=checkpoint_prefix, save_interval=checkpoint_interval)
    
    # Determine which queries to process
    if retry_failed:
        # Only retry failed queries from previous run
        failed_queries = checkpoint_mgr.get_failed_queries()
        queries_to_process = [q for q in queries_to_process if q in failed_queries]
        print(f"Retrying {len(queries_to_process)} failed queries from previous run")
    else:
        # Skip already successful queries
        successful_queries = checkpoint_mgr.get_successful_queries()
        queries_to_process = [q for q in queries_to_process if q not in successful_queries]
    
    print(f"Processing {len(queries_to_process)} queries across {len(selected_queries)} tasks")
    
    # If all queries are already processed, skip to results
    if not queries_to_process:
        print("All queries already processed successfully. Skipping to results collection.")
    else:
        # Process queries with retries
        retry_count = 0
        while retry_count < max_retries:
            # Prepare data for parallel processing
            parallel_args = [(query, all_query_groups[query], llm_model, top_n) 
                            for query in queries_to_process]
            
            # Process in parallel
            print(f"Processing queries in parallel with {p_num} workers... (Attempt {retry_count + 1}/{max_retries})")
            
            with ThreadPool(p_num) as pool:
                for query, best_perf, template, success in tqdm(
                    pool.imap_unordered(process_query_parallel, parallel_args),
                    total=len(parallel_args),
                    desc="Processing queries"
                ):
                    # Add result to checkpoint manager
                    checkpoint_mgr.add_result(query, best_perf, template, success)
            
            # Save checkpoint after each attempt
            checkpoint_mgr.save_checkpoint()
            
            # Check error rate
            failed_queries = checkpoint_mgr.get_failed_queries()
            error_rate = len(failed_queries) / len(queries_to_process)
            
            print(f"\nAttempt {retry_count + 1} completed. Error rate: {error_rate:.2%}")
            
            if error_rate <= error_threshold:
                print(f"Error rate ({error_rate:.2%}) is below threshold ({error_threshold:.2%}). Stopping retries.")
                break
            
            if retry_count < max_retries - 1:
                print(f"Error rate above threshold. Retrying {len(failed_queries)} failed queries...")
                queries_to_process = failed_queries
            else:
                print(f"Maximum retries reached. {len(failed_queries)} queries still failed.")
            
            retry_count += 1
    
    # Collect all results from checkpoint manager
    best_performers = {}
    templates = {}
    
    for query in all_query_groups.keys():
        # Try to get from checkpoint if available
        best_perf = checkpoint_mgr.get_best_performers(query)
        template = checkpoint_mgr.get_template(query)
        
        if best_perf is not None and template is not None:
            best_performers[query] = best_perf
            templates[query] = template
    
    # Report failures
    failed_queries = checkpoint_mgr.get_failed_queries()
    if failed_queries:
        print(f"\nWarning: {len(failed_queries)} queries failed processing after {retry_count + 1} attempts:")
        for q in failed_queries[:min(5, len(failed_queries))]:
            print(f"  - '{q[:50]}...'")
        if len(failed_queries) > 5:
            print(f"  ... and {len(failed_queries) - 5} more")
    
    # Create summarized dataset with thought templates for successful queries
    print(f"Creating summarized dataset...")
    # Filter selected_queries to only include successful ones
    filtered_selected_queries = {}
    for task_name, queries in selected_queries.items():
        filtered_selected_queries[task_name] = [q for q in queries if q in templates]
    
    summarized_df = create_summarized_dataset(
        data=data,
        selected_queries=filtered_selected_queries,
        query_groups=all_query_groups,
        templates=templates
    )
    
    # Save summarized dataset
    if template_dataset_path:
        print(f"Saving summarized dataset with thought templates to {template_dataset_path}...")
        summarized_df.to_csv(template_dataset_path, index=False)
    
    # Format detailed results for successful queries
    print("Formatting detailed results...")
    formatted_results = {}
    for query, perf in best_performers.items():
        if query not in templates:
            continue  # Skip if no template (failed query)
            
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
    
    # Save detailed results
    if output_path:
        print(f"Saving detailed results to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(formatted_results, f, indent=2)
    
    # Print task distribution
    print("\nTask distribution in processed dataset:")
    task_counts = {}
    for task_name, queries in filtered_selected_queries.items():
        task_counts[task_name] = len(queries)
    
    for task_name, count in task_counts.items():
        print(f"- {task_name}: {count} queries")
    
    # Final summary
    success_count = len(formatted_results)
    total_queries = len(queries_to_process) + len(checkpoint_mgr.get_successful_queries())
    print(f"\nAnalysis complete! Successfully processed {success_count} out of {total_queries} queries")
    print(f"Top {top_n} performers and thought templates for each query saved to {output_path}")
    print(f"Summarized dataset with thought templates saved to {template_dataset_path}")
    
    return formatted_results

# Example usage
if __name__ == "__main__":
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
    
    results = analyze_router_dataset(
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