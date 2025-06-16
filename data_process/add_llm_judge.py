import pandas as pd
import numpy as np
from openai import OpenAI
from typing import Optional, Union, List, Tuple
import time
import json
import re
import random
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Initialize OpenAI client with placeholder values
client = OpenAI(
    base_url="YOUR_API_BASE_URL",  # Replace with your API base URL
    api_key="YOUR_API_KEY",        # Replace with your API key
    timeout=60,
    max_retries=2
)

# Global lock for rate limiting
rate_limit_lock = Lock()
last_request_time = 0
MIN_REQUEST_INTERVAL = 0.1  # Minimum time between requests in seconds

def rate_limited_request():
    """Ensure minimum time between requests"""
    global last_request_time
    with rate_limit_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - time_since_last)
        last_request_time = time.time()

def model_prompting(
    llm_model: str,
    prompt: str,
    max_token_num: Optional[int] = 1024,
    temperature: Optional[float] = 0.2,
    top_p: Optional[float] = 0.7,
    stream: Optional[bool] = True,
    max_retries: int = 3,
    base_delay: float = 2.0
) -> Union[str, None]:
    """
    Get a response from an LLM model using the OpenAI-compatible API with retry logic.
    """
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_token_num,
                temperature=temperature,
                top_p=top_p,
                stream=stream
            )

            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    response_text += chunk.choices[0].delta.content
            return response_text
            
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
            else:
                raise e
    
    return None

def get_llm_judge_score(query: str, ground_truth: str, response: str) -> float:
    """
    Get a score from the LLM judge based on the quality of the response.
    """
    prompt = f"""You are an expert judge evaluating the quality of an AI model's response. Please score the response based on the following criteria:

1. Correctness (0-1): Is the answer correct according to the ground truth?
2. Thought Process (0-1): Does the response show clear reasoning and explanation?
3. Training Data Quality (0-1): Is the response well-structured and suitable for supervised fine-tuning?

Query: {query}
Ground Truth: {ground_truth}
Response: {response}

Please provide a single score from 0 to 1, where:
- 0: Incorrect answer
- 0.3: Correct answer but minimal thought process
- 0.6: Correct answer with some thought process
- 0.8: Correct answer with good thought process
- 1.0: Correct answer with excellent thought process and well-suited for training

Return the score in the following format:
<answer>SCORE</answer>
where SCORE is a number between 0 and 1."""

    try:
        response = model_prompting(
            llm_model="YOUR_MODEL_NAME",  # Replace with your model name
            prompt=prompt,
            max_token_num=100,
            temperature=0.1,
            top_p=0.7,
            stream=True,
            max_retries=3,
            base_delay=2.0
        )
        
        if response is None:
            print("Failed to get response after all retries")
            return None
        
        # Extract the numerical score from the response using answer tags
        try:
            # Find content between answer tags
            match = re.search(r'<answer>(.*?)</answer>', response)
            if match:
                score = float(match.group(1).strip())
                return min(max(score, 0), 1)  # Ensure score is between 0 and 1
            else:
                print(f"Could not find score in answer tags. Response: {response}")
                return None
        except ValueError:
            print(f"Could not parse score from response: {response}")
            return None
            
    except Exception as e:
        print(f"Error getting LLM judge score: {e}")
        return None

def process_row(row_data: Tuple[int, pd.Series]) -> Tuple[int, Optional[float]]:
    """
    Process a single row and return its index and score.
    """
    idx, row = row_data
    rate_limited_request()
    
    try:
        score = get_llm_judge_score(
            query=row['query'],
            ground_truth=row['ground_truth'],
            response=row['response']
        )
        return idx, score
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return idx, None

def process_csv(input_file: str = "./dataset/router_data.csv", 
                output_file: str = "./dataset/router_data_with_judge.csv",
                max_workers: int = 5, 
                batch_size: int = 10):
    """
    Process a CSV file and add LLM judge scores using parallel processing.
    If output file exists, resume from the first NaN row.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        max_workers: Maximum number of concurrent workers
        batch_size: Number of rows to process before saving checkpoint
    """
    print(f"Processing {input_file}...")
    df = pd.read_csv(input_file)
    
    # Check if output file exists and load it
    start_idx = 0
    if os.path.exists(output_file):
        print(f"Found existing output file {output_file}, checking for progress...")
        existing_df = pd.read_csv(output_file)
        
        # Find first NaN row from the start
        nan_mask = existing_df['llm_judge'].isna()
        if nan_mask.any():
            # Get the first index with NaN value
            first_nan_idx = nan_mask.idxmax()
            start_idx = first_nan_idx
            print(f"Found first NaN row at index {first_nan_idx}")
            print(f"Starting from row {start_idx}")
            
            # Copy existing scores
            df.loc[:start_idx-1, 'llm_judge'] = existing_df.loc[:start_idx-1, 'llm_judge']
        else:
            print("No NaN rows found, all rows already processed")
            return
    else:
        print("No existing output file found, starting from beginning")
    
    # Add new column for LLM judge scores if not already present
    if 'llm_judge' not in df.columns:
        df['llm_judge'] = 0.0
    
    total_rows = len(df)
    failed_indices = []
    
    print(f"Starting parallel processing with {max_workers} workers...")
    
    # Process rows in batches starting from start_idx
    for batch_start in range(start_idx, total_rows, batch_size):
        batch_end = min(batch_start + batch_size, total_rows)
        print(f"Processing batch {batch_start}-{batch_end}/{total_rows}")
        
        # Create a list of row data to process
        row_data = [(idx, row) for idx, row in df.iloc[batch_start:batch_end].iterrows()]
        
        # Process rows in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {executor.submit(process_row, row_data): row_data for row_data in row_data}
            
            for future in as_completed(future_to_row):
                idx, score = future.result()
                if score is not None:
                    df.at[idx, 'llm_judge'] = score
                else:
                    failed_indices.append(idx)
                    print(f"Failed to get score for row {idx}")
        
        # Save checkpoint after each batch
        df.to_csv(output_file, index=False)
        print(f"Checkpoint saved at row {batch_end}")
        
        # Add a small delay between batches to prevent overwhelming the API
        time.sleep(1.0)
    
    # Second pass: Process failed rows
    if failed_indices:
        print(f"\nStarting second pass for {len(failed_indices)} failed rows...")
        failed_row_data = [(idx, df.iloc[idx]) for idx in failed_indices]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {executor.submit(process_row, row_data): row_data for row_data in failed_row_data}
            
            for future in as_completed(future_to_row):
                idx, score = future.result()
                if score is not None:
                    df.at[idx, 'llm_judge'] = score
                    print(f"Successfully processed row {idx} on second attempt")
                else:
                    print(f"Failed to get score for row {idx} on second attempt")
    
    # Save the final processed dataframe
    df.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")
    
    # Print statistics
    total_processed = total_rows - len(failed_indices)
    print(f"\nProcessing Statistics:")
    print(f"Total rows: {total_rows}")
    print(f"Successfully processed: {total_processed}")
    print(f"Failed rows: {len(failed_indices)}")
    if failed_indices:
        print(f"Failed row indices: {failed_indices}")

def main():
    # Process the CSV file with parallel processing
    process_csv(
        input_file="./dataset/router_data.csv",
        output_file="./dataset/router_data_with_judge.csv",
        max_workers=8,  # Adjust based on API rate limits
        batch_size=300
    )

if __name__ == "__main__":
    main() 