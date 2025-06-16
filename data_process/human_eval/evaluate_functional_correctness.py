import fire

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness_item
import json


def entry_point_item(sample_input, problem_file=HUMAN_EVAL):
    """
    Evaluates the functional correctness of a single sample.
    
    Args:
        sample_input: Either a path to a JSON file or a dictionary with task_id and completion
        problem_file: Path to the HumanEval problems file
    """
    # Check if sample_input is a string (path) or dict
    if isinstance(sample_input, str):
        # Load the sample from file
        with open(sample_input, 'r') as f:
            sample = json.load(f)
    else:
        # Already a dictionary
        sample = sample_input
    
    result = evaluate_functional_correctness_item(sample, problem_file)
    return result


def main():
    fire.Fire(entry_point_item)


# sys.exit(main())