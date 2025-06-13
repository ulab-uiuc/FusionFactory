# Thought Template Generation and Prompt Creation

This module provides two main functionalities:
1. Generating thought templates by analyzing high-performing responses from small language models (7B-12B parameters)
2. Creating thought prompts for new queries by finding similar queries and their templates

## Overview

### Thought Template Generation (`thought_template_gen.py`)
The thought template generation process:
1. Takes a dataset of queries and their responses
2. Identifies the best performing responses for each query
3. Generates a concise thought template that captures effective reasoning patterns
4. Creates a summarized dataset with thought templates

### Thought Prompt Creation (`get_thought_prompt.py`)
The thought prompt creation process:
1. Takes a dataset of queries and a template dataset
2. For each query, finds similar queries from the template dataset
3. Creates a thought prompt that includes similar questions and their templates
4. Generates a final prompt that guides the model to solve the query using the templates

## Usage

### 1. Generating Thought Templates (`thought_template_gen.py`)

#### Basic Usage
```bash
python thought_template_gen.py --file_path <path_to_dataset> --template_dataset_path <output_path>
```

#### Full Configuration Options
```bash
python thought_template_gen.py \
    --file_path <path_to_dataset> \
    --top_n <number_of_top_performers> \
    --output_path <detailed_results_path> \
    --template_dataset_path <template_dataset_path> \
    --llm_model <model_name> \
    --queries_per_task <queries_per_task> \
    --p_num <parallel_workers> \
    --checkpoint_interval <checkpoint_frequency> \
    --retry_failed
```

#### Parameters for Template Generation
- `--file_path`: Path to the input dataset file (required)
- `--top_n`: Number of top performers to select for each query (default: 3)
- `--output_path`: Path to save detailed results (default: "router_analysis_results.json")
- `--template_dataset_path`: Path to save the summarized dataset with templates
- `--llm_model`: Model to use for generating templates (default: "nvdev/nvidia/llama-3.1-8b-instruct")
- `--queries_per_task`: Number of queries to process per task type (default: 3)
- `--p_num`: Number of parallel workers (default: 6)
- `--checkpoint_interval`: How often to save checkpoints (default: 1000)
- `--retry_failed`: Flag to retry only failed queries from previous run

### 2. Creating Thought Prompts (`get_thought_prompt.py`)

The script is configured with the following hardcoded parameters in the main section:

```python
# Input/Output Configuration
template_file = "thought_template_70b_full.csv"  # Path to template dataset
output_file = "thought_prompt_output.csv"        # Path to save generated prompts

# Processing Configuration
sample_mode = False                              # Set to True for sample processing
samples_per_task = 3                             # Number of samples per task in sample mode
chunk_size = 200                                 # Number of queries to process per chunk
num_processes = 20                               # Number of parallel processes
```

#### Usage Steps
1. Ensure you have the required input files:
   - A template dataset file (default: "thought_template_70b_full.csv")
   - The script will load queries from the HuggingFace dataset "ulab-ai/FusionBench"

2. Modify the configuration in the script's main section if needed:
   - Change `template_file` to point to your template dataset
   - Adjust `output_file` for your desired output location
   - Set `sample_mode` to True for testing with a subset of queries
   - Modify `chunk_size` and `num_processes` based on your system resources

3. Run the script:
```bash
python get_thought_prompt.py
```

#### Output
The script generates a CSV file containing:
- Original query information
- Generated thought prompt
- List of similar queries used
- Task information

## Output

### Template Generation Output
1. A detailed results file (JSON) containing:
   - Task information
   - Query details
   - Best performing responses
   - Generated thought templates

2. A summarized dataset (CSV) containing:
   - Task information
   - Queries
   - Generated thought templates

### Prompt Creation Output
A CSV file containing:
- Original query information
- Generated thought prompt
- List of similar queries used
- Task information

## Checkpointing

The template generation script implements checkpointing to save progress and allow resuming from failures:
- Checkpoints are saved in the `./checkpoints_thought_template_hybrid_small_8b` directory
- Use `--retry_failed` to retry failed queries from a previous run
- Checkpoint frequency can be adjusted with `--checkpoint_interval`

## Supported Models

The template generation script is configured to work with small language models (7B-12B parameters) including:
- Qwen2-7B
- Gemma-7B
- CodeGemma-7B
- Llama-3.1-8B
- Granite-3.0-8B
- Mistral-7B/12B