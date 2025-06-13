# Thought Template Generation

This module generates thought templates for different types of queries by analyzing high-performing responses from small language models (7B-12B parameters).

## Overview

The thought template generation process:
1. Takes a dataset of queries and their responses
2. Identifies the best performing responses for each query
3. Generates a concise thought template that captures effective reasoning patterns
4. Creates a summarized dataset with thought templates

## Usage

### Basic Usage

```bash
python thought_template_gen.py --file_path <path_to_dataset> --template_dataset_path <output_path>
```

### Full Configuration Options

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

### Parameters

- `--file_path`: Path to the input dataset file (required)
- `--top_n`: Number of top performers to select for each query (default: 3)
- `--output_path`: Path to save detailed results (default: "router_analysis_results.json")
- `--template_dataset_path`: Path to save the summarized dataset with templates
- `--llm_model`: Model to use for generating templates (default: "nvdev/nvidia/llama-3.1-8b-instruct")
- `--queries_per_task`: Number of queries to process per task type (default: 3)
- `--p_num`: Number of parallel workers (default: 6)
- `--checkpoint_interval`: How often to save checkpoints (default: 1000)
- `--retry_failed`: Flag to retry only failed queries from previous run

### Example Configurations

1. Basic template generation:
```bash
python thought_template_gen.py --file_path data/input.json --template_dataset_path output/templates.csv
```

2. Process more queries with higher parallelism:
```bash
python thought_template_gen.py \
    --file_path data/input.json \
    --queries_per_task 5 \
    --p_num 12 \
    --template_dataset_path output/templates.csv
```

3. Retry failed queries from previous run:
```bash
python thought_template_gen.py \
    --file_path data/input.json \
    --retry_failed \
    --template_dataset_path output/templates.csv
```

## Output

The script generates two main outputs:

1. A detailed results file (JSON) containing:
   - Task information
   - Query details
   - Best performing responses
   - Generated thought templates

2. A summarized dataset (CSV) containing:
   - Task information
   - Queries
   - Generated thought templates

## Checkpointing

The script implements checkpointing to save progress and allow resuming from failures:
- Checkpoints are saved in the `./checkpoints_thought_template_hybrid_small_8b` directory
- Use `--retry_failed` to retry failed queries from a previous run
- Checkpoint frequency can be adjusted with `--checkpoint_interval`

## Supported Models

The script is configured to work with small language models (7B-12B parameters) including:
- Qwen2-7B
- Gemma-7B
- CodeGemma-7B
- Llama-3.1-8B
- Granite-3.0-8B
- Mistral-7B/12B 