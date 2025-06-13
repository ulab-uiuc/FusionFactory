import pandas as pd
import numpy as np
import json
import os
from datasets import load_dataset

def extract_numeric_value(value):
    """Extract a numeric value from a float or a stringified dict with 'pass@1'."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.number)):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return np.nan

    return np.nan

# Load data from HuggingFace dataset
dataset = load_dataset("ulab-ai/FusionBench")
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['partial_test'])

# Recalculate cost for deepseek-r1 rows with new prices
for df in [train_df, test_df]:
    deepseek_mask = df['llm'].str.contains('deepseek-r1', na=False)
    df.loc[deepseek_mask, 'cost'] = df.loc[deepseek_mask, 'input_tokens_num'] * 0.55 + df.loc[deepseek_mask, 'output_tokens_num'] * 2.19

# Filter out rows where response is NaN or empty string
train_df = train_df[train_df['response'].notna() & (train_df['response'] != '')]
test_df = test_df[test_df['response'].notna() & (test_df['response'] != '')]

# Convert performance and cost columns to numeric
for df in [train_df, test_df]:
    df['eval_performance'] = df['performance'].apply(extract_numeric_value)
    df['eval_cost'] = df['cost'].apply(extract_numeric_value)

    # Normalize performance and cost within each task_name
    df['normalized_performance'] = df.groupby('task_name')['eval_performance'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
    df['normalized_cost'] = df.groupby('task_name')['eval_cost'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

def process_chunks(df, chunk_size=40):
    output_data = []
    total_rows = len(df)
    
    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        if len(chunk) == 0:
            continue
            
        query = chunk['query'].iloc[0]
        candidates = []
        for _, row in chunk.iterrows():
            candidate_info = {
                'candidate_name': row['llm'],
                'performance': float(row['normalized_performance']),
                'cost': float(row['normalized_cost']),
                'score': float(row['score'])
            }
            candidates.append(candidate_info)
        
        valid_chunk = chunk.dropna(subset=['score'])
        best_candidate = valid_chunk.loc[valid_chunk['score'].idxmax(), 'llm'] if len(valid_chunk) > 0 else candidates[0]['candidate_name']
        
        output_data.append({
            'query': query,
            'candidates': candidates,
            'ground_truth': best_candidate
        })
    
    return output_data

def create_json_output(train_df, test_df, alpha, beta, output_name):
    # Calculate score for each row using the formula: Reward = α · Performance - β · Cost
    train_df['score'] = alpha * train_df['normalized_performance'] - beta * train_df['normalized_cost']
    test_df['score'] = alpha * test_df['normalized_performance'] - beta * test_df['normalized_cost']

    if output_name == 'llm_judge':  
        train_df['score'] = train_df['llm_judge']
        test_df['score'] = test_df['llm_judge']
    
    train_data = process_chunks(train_df)
    test_data = process_chunks(test_df)
    
    with open(f'{output_name}_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(f'{output_name}_test.json', 'w') as f:
        json.dump(test_data, f, indent=2)

# Create JSON files with different scoring configurations
create_json_output(train_df, test_df, 1.0, 0.0, 'performance')  # Performance First (α=1, β=0)
create_json_output(train_df, test_df, 0.5, 0.5, 'balance')      # Balanced (α=0.5, β=0.5)
create_json_output(train_df, test_df, 0.2, 0.8, 'cost')         # Cost First (α=0.2, β=0.8)
create_json_output(train_df, test_df, 1.0, 0.0, 'llm_judge')    # LLM Judge scores