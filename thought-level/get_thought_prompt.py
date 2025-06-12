import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count
import os
import time
from functools import partial
from tqdm import tqdm

def parse_embedding(embedding_str):
    # import pdb; pdb.set_trace()
    """Parse embedding string to numpy array, handling different formats."""
    if embedding_str is None or pd.isna(embedding_str):
        return None
        
    try:
        # Handle PyTorch tensor format
        if isinstance(embedding_str, str) and 'tensor' in embedding_str:
            # Remove tensor( and ) parts, device info, and any extra brackets
            clean_str = embedding_str.replace('tensor(', '').replace(')', '')
            # Remove device information
            if 'device=' in clean_str:
                clean_str = clean_str.split('device=')[0].strip()
            # Remove newlines and extra spaces
            clean_str = clean_str.replace('\n', '').replace(' ', '')
            # Use eval to safely convert the string to a Python object
            embedding = np.array(eval(clean_str))
            return embedding
        
        # Handle standard array format [x, y, z]
        elif isinstance(embedding_str, str):
            # Remove brackets and split by commas
            clean_str = embedding_str.replace('[', '').replace(']', '')
            return np.array([float(x) for x in clean_str.split(',')])
            
        # Handle numeric values or numpy arrays directly
        elif isinstance(embedding_str, (int, float, np.ndarray)):
            return np.array(embedding_str)
            
        else:
            return None
            
    except Exception as e:
        print(f"Error parsing embedding: {str(e)}")
        return None

def create_sample_dataframe(queries_file_path, samples_per_task=3, random_state=42):
    """
    Create a sampled DataFrame with a specified number of queries per task.
    
    Args:
        queries_file_path: Path to the CSV with queries to process
        samples_per_task: Number of samples per unique task description
        random_state: Random seed for reproducible sampling
    
    Returns:
        DataFrame with sampled queries
    """
    # Load the queries file
    print(f"Loading queries from {queries_file_path}...")
    queries_df = pd.read_csv(queries_file_path)
    original_count = len(queries_df)
    print(f"Original queries count: {original_count}")
    
    # Get unique task descriptions
    unique_tasks = queries_df['task_description'].unique()
    print(f"Found {len(unique_tasks)} unique task descriptions")
    
    # Create empty DataFrame for samples
    sampled_queries = pd.DataFrame()
    
    # Sample queries for each task
    for task in unique_tasks:
        task_queries = queries_df[queries_df['task_description'] == task]
        print(f"Task: {task[:50]}... - {len(task_queries)} queries")
        
        # Take min(samples_per_task, available) samples
        n_samples = min(samples_per_task, len(task_queries))
        task_samples = task_queries.sample(n=n_samples, random_state=random_state)
        
        # Add to our sampled DataFrame
        sampled_queries = pd.concat([sampled_queries, task_samples])
    
    # Reset index and create a clean copy
    sampled_queries = sampled_queries.reset_index(drop=True).copy()
    print(f"Final sampled queries count: {len(sampled_queries)}")
    
    return sampled_queries

def process_query_chunk(chunk_data, template_df):
    """Process a chunk of queries."""
    chunk_df, chunk_start_idx = chunk_data
    results = []
    
    for i, (idx, query_row) in enumerate(chunk_df.iterrows()):
        try:
            query = query_row['query']
            
            # Check if the required columns exist
            if 'query_embedding' not in query_row or 'task_description' not in query_row:
                print(f"Missing required columns for query {idx}")
                continue
            
            query_embedding = parse_embedding(query_row['query_embedding'])
            task_description = query_row['task_description']
            
            if query_embedding is None:
                print(f"Failed to parse embedding for query {idx}")
                continue
            
            # Filter templates with the same task description
            matching_templates = template_df[template_df['task_description'] == task_description]
            
            if matching_templates.empty:
                # Try to find partial matches (case-insensitive)
                task_desc_lower = task_description.lower()
                partial_matches = template_df[template_df['task_description'].str.lower().str.contains(task_desc_lower.split()[0], na=False)]
                
                if not partial_matches.empty:
                    matching_templates = partial_matches
                else:
                    print(f"No matching templates found for task: {task_description[:50]}...")
                    continue
            
            # Parse embeddings and calculate similarities
            similarities = []
            for t_idx, t_row in matching_templates.iterrows():
                if 'query_embedding' not in t_row:
                    continue
                    
                template_embedding = parse_embedding(t_row['query_embedding'])
                
                if template_embedding is not None:
                    try:
                        # Reshape for sklearn's cosine_similarity
                        q_emb = query_embedding.reshape(1, -1)
                        t_emb = template_embedding.reshape(1, -1)
                        
                        # Check dimensions match
                        if q_emb.shape[1] != t_emb.shape[1]:
                            continue
                            
                        sim = cosine_similarity(q_emb, t_emb)[0][0]
                        similarities.append((t_idx, sim))
                    except Exception:
                        continue
            
            if not similarities:
                continue
            
            # Sort by similarity (highest first) and get top 3 (or fewer if less available)
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_n = min(3, len(similarities))
            top_indices = [idx for idx, _ in similarities[:top_n]]
            
            # Get the rows from matching_templates
            top_templates = matching_templates.loc[top_indices]
            
            # Format thought prompt
            thought_prompt = "Here are some similar questions and guidelines in how to solve them:\n\n"
            
            for i, (t_idx, t_row) in enumerate(top_templates.iterrows(), 1):
                thought_prompt += f"Question{i}: {t_row['query']}\n\n"
                thought_prompt += f"Thought Template {i}: {t_row['thought_template']}\n\n"
            
            thought_prompt += "Now, please solve the following question:\n\n"
            thought_prompt += query
            thought_prompt += "\n\n Use the thought templates above as guidance. Reason step by step. And provide the final answer! The final answer should be enclosed in <answer> and </answer> tags."
            
            # Store result
            results.append({
                'idx': idx,
                'thought_prompt_all': thought_prompt,
                'similar_queries': ', '.join(top_templates['query'].tolist())
            })
            
        except Exception as e:
            continue
    
    return results

def process_queries_parallel(input_df, template_file_path, num_processes=None, chunk_size=100):
    """
    Process queries in parallel using the provided DataFrame.
    
    Args:
        input_df: DataFrame containing queries to process
        template_file_path: Path to the CSV with thought templates
        num_processes: Number of processes to use (default: CPU count - 1)
        chunk_size: Number of queries to process in each chunk
        
    Returns:
        DataFrame with original queries and generated thought prompts
    """
    start_time = time.time()
    
    # Load template file
    template_df = pd.read_csv(template_file_path)
    print(f"Loaded {len(template_df)} templates")
    
    # Create a copy of input DataFrame for results
    results_df = input_df.copy()
    
    # Add columns for thought prompts if they don't exist
    if 'thought_prompt_all' not in results_df.columns:
        results_df['thought_prompt_all'] = None
    if 'similar_queries' not in results_df.columns:
        results_df['similar_queries'] = None
    
    # Create chunks of the queries dataframe
    total_queries = len(input_df)
    chunks = []
    
    for i in range(0, total_queries, chunk_size):
        end_idx = min(i + chunk_size, total_queries)
        chunks.append((input_df.iloc[i:end_idx], i))
    
    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    
    print(f"Processing {total_queries} queries in {len(chunks)} chunks using {num_processes} processes")
    # Create a partial function with the template_df already set
    process_func = partial(process_query_chunk, template_df=template_df)
    
    # Create a pool of workers and map the function to the chunks
    with Pool(processes=num_processes) as pool:
        # Process chunks in parallel with progress bar
        results = list(tqdm(pool.imap(process_func, chunks), total=len(chunks), desc="Processing chunks"))
    
    # Flatten results
    flat_results = []
    for chunk_results in results:
        flat_results.extend(chunk_results)
    
    # Create a dictionary to lookup results by index
    results_dict = {result['idx']: result for result in flat_results}
    
    # Update results dataframe with processed thought prompts
    for idx, row in results_df.iterrows():
        if idx in results_dict:
            results_df.at[idx, 'thought_prompt_all'] = results_dict[idx]['thought_prompt_all']
            results_df.at[idx, 'similar_queries'] = results_dict[idx]['similar_queries']
    
    # Count successful generations
    success_count = results_df['thought_prompt_all'].notna().sum()
    elapsed_time = time.time() - start_time
    
    print(f"Successfully generated {success_count} thought prompts out of {total_queries} queries")
    print(f"Processing took {elapsed_time:.2f} seconds ({total_queries/elapsed_time:.2f} queries/second)")
    
    return results_df

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    template_file = "/data/taofeng2/Router_bench/zijie/full_test_data_process/hybrid_thought_template/thought_template_70b_small.csv"
    queries_file = "/data/taofeng2/Router_bench/router_data/router_quac_test_full_0608.csv"
    output_file = "/data/taofeng2/Router_bench/zijie/full_test_data_process/new_benchmark_data/quac_hybrid_70b_small_0609.csv"
    
    # Step 1: Create a sampled DataFrame (for testing)
    # Comment this out and use the full queries_df for production
    sample_mode = False  # Set to False for full processing
    
    if sample_mode:
        print("SAMPLE MODE: Creating a sample of 3 queries per task")
        queries_df = create_sample_dataframe(queries_file, samples_per_task=3)
        # Save the sample for inspection if needed
        queries_df.to_csv("sample_queries.csv", index=False)
    else:
        print("FULL MODE: Processing all queries")
        queries_df = pd.read_csv(queries_file)
        queries_df = queries_df[::2]

    if sample_mode:
        # For sample mode, use smaller chunks to get better parallelism
        chunk_size = 3  # Makes 12 chunks with 36 samples
        num_processes = min(12, max(1, cpu_count() - 1))
    else: 
        # Step 2: Process the queries in parallel
        num_processes = 20
        chunk_size = 200  # Process 200 queries per chunk
    
    result_df = process_queries_parallel(
        queries_df,
        template_file,
        num_processes=num_processes,
        chunk_size=chunk_size
    )

    
    # Save results
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Show a sample of the results
    if not result_df.empty and 'thought_prompt_all' in result_df.columns:
        successful_samples = result_df[result_df['thought_prompt_all'].notna()]
        if not successful_samples.empty:
            sample_query = successful_samples.iloc[0]
            print("\nSample successful result:")
            print(f"Query: {sample_query['query'][:100]}...")
            print(f"Thought prompt: {sample_query['thought_prompt_all'][:200]}...")