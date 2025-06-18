# FusionBench Evaluation Framework

This directory contains the evaluation framework for FusionBench, a comprehensive benchmark for evaluating language model performance across various tasks.

## Overview

The evaluation framework supports multiple types of tasks and metrics:

- **Mathematical Reasoning**: GSM8K, MATH
- **Code Generation**: MBPP, HumanEval
- **Commonsense Reasoning**: CommonsenseQA, OpenBookQA, ARC Challenge, HellaSwag
- **World Knowledge**: Natural Questions, TriviaQA
- **Reading Comprehension**: SQuAD, BoolQ
- **Popular Benchmarks**: MMLU, GPQA

## Key Components

### Main Evaluation Script (`response_eval.py`)

The main evaluation script that processes model outputs and calculates performance metrics across different tasks. It supports:

- Task-specific evaluation metrics
- Category-wise performance aggregation
- Detailed per-task performance reporting
- Sample count statistics

### Utility Functions (`utils.py`)

Contains essential helper functions for:

- Text normalization and preprocessing
- Various evaluation metrics (F1, Exact Match, BERT Score)
- Code evaluation utilities
- Model prompting and embedding functions
- File I/O operations

### Math Evaluation (`math_eval.py`)

Specialized functions for evaluating mathematical reasoning tasks:

- LaTeX equation normalization
- Mathematical expression comparison
- Boxed answer extraction and validation

### Task-Specific Evaluators

- `human_eval/`: Evaluation framework for HumanEval code generation tasks
- `mbpp/`: Evaluation framework for MBPP code generation tasks

## Usage

1. Prepare your model outputs in a CSV file with the following columns:
   - `output`: Model's response
   - `ground_truth`: Correct answer
   - `task_name`: Name of the task
   - `metric`: Evaluation metric to use
   - `task_id`: Task identifier (required for code evaluation)

2. Run the evaluation:
```bash
python response_eval.py
```

The script will process the outputs and provide:
- Category-wise performance metrics
- Individual task performance
- Sample counts per task

## Supported Metrics

- `em`: Exact Match
- `cem`: Contains Exact Match
- `em_mc`: Exact Match for Multiple Choice
- `bert_score`: BERT-based semantic similarity
- `f1_score`: F1 score for text matching
- `code_eval`: Code execution-based evaluation
- Task-specific metrics for GSM8K and MATH

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Pandas
- SentenceTransformers
- BERT-Score
- LiteLLM
- Transformers (Longformer)

## Notes

- The framework uses CUDA when available for faster processing
- Code evaluation includes timeout protection
- Mathematical expressions are normalized for fair comparison
- BERT-Score is used for semantic similarity evaluation 