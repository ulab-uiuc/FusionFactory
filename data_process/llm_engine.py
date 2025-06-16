import json
import re
import time
from transformers import GPT2Tokenizer
from utils import model_prompting, f1_score, exact_match_score, cem_score
from beartype.typing import Any, Dict, List
from human_eval.evaluate_functional_correctness import entry_point_item
from mbpp.mbpp_eval import entry_point_item_mbpp
from math_eval import last_boxed_only_string, remove_boxed, is_equiv

# Initialize tokenizer for token counting (used in cost calculation)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


class LLMEngine:
    """
    A class to manage interactions with multiple language models and evaluate their performance.

    Handles model selection, querying, cost calculation, and performance evaluation
    using various metrics for different tasks.
    """

    def __init__(self, llm_names: List[str], llm_description: Dict[str, Dict[str, Any]]):
        """
        Initialize the LLM Engine with available models and their descriptions.

        Args:
            llm_names: List of language model names available in the engine
            llm_description: Dictionary containing model configurations and pricing details
                Structure: {
                    "model_name": {
                        "model": "api_identifier",
                        "input_price": cost_per_input_token,
                        "output_price": cost_per_output_token,
                        ...
                    },
                    ...
                }
        """
        self.llm_names = llm_names
        self.llm_description = llm_description

    def compute_cost(self, llm_idx: int, input_text: str, output_text: str) -> float:
        """
        Calculate the cost of a model query based on input and output token counts.

        Args:
            llm_idx: Index of the model in the llm_names list
            input_text: The input prompt sent to the model
            output_size: Number of tokens in the model's response

        Returns:
            float: The calculated cost in currency units
        """
        # Count input tokens
        input_size = len(tokenizer(input_text)['input_ids'])
        output_size = len(tokenizer(output_text)['input_ids'])

        # Get pricing information for the selected model
        llm_name = self.llm_names[llm_idx]
        input_price = self.llm_description[llm_name]["input_price"]
        output_price = self.llm_description[llm_name]["output_price"]

        # Calculate total cost
        cost = input_size * input_price + output_size * output_price
        return cost, input_price, output_price, input_size, output_size

    def get_llm_response(self, query: str, llm_idx: int, api_base: str, api_key: str) -> str:
        """
        Send a query to a language model and get its response.

        Args:
            query: The prompt text to send to the model
            llm_idx: Index of the model in the llm_names list

        Returns:
            str: The model's text response

        Note:
            Includes a retry mechanism with a 2-second delay if the first attempt fails
        """
        llm_name = self.llm_names[llm_idx]
        model = self.llm_description[llm_name]["model"]

        try:
            response = model_prompting(llm_model=model, prompt=query, base_url=api_base, api_key=api_key)
        except:
            # If the request fails, wait and retry once
            time.sleep(2)
            response = model_prompting(llm_model=model, prompt=query, base_url=api_base, api_key=api_key)

        return response

    def eval(self, prediction: str, ground_truth: str, metric: str, task_id=None, entry_point=None) -> float:
        """
        Evaluate the model's prediction against the ground truth using the specified metric.

        Args:
            prediction: The model's output text
            ground_truth: The correct expected answer
            metric: The evaluation metric to use (e.g., 'em', 'f1_score', 'GSM8K')
            task_id: Optional identifier for the specific task being evaluated

        Returns:
            float: Evaluation score (typically between 0 and 1)
        """
        # Exact match evaluation
        if metric == 'em':
            result = exact_match_score(prediction, ground_truth)
            return float(result)

        # Multiple choice exact match
        elif metric == 'em_mc':
            result = exact_match_score(prediction, ground_truth, normal_method="mc")
            return float(result)

        elif metric == 'cem':
            result = cem_score(prediction, ground_truth)
            return float(result)

        # GSM8K math problem evaluation
        elif metric == 'GSM8K':
            ground_truth = ground_truth.split("####")[-1].replace(',', '').replace('$', '').replace('.', '').strip()
            answer = re.findall("(\\-?[0-9\\.\\,]+)", prediction)
            final_answer = None
            if len(answer) == 0:
                return 0
            else:
                invalid_str = ['', '.']
                # find the last number that is not '.'
                for final_answer in reversed(answer):
                    if final_answer not in invalid_str:
                        break
            final_answer = final_answer.replace(',', '').replace('$', '').replace('.', '').strip()
            if final_answer == ground_truth:
                return 1
            else:
                return 0
        elif metric == 'MATH':
            ground_truth = remove_boxed(last_boxed_only_string(ground_truth))
            try:
                string_in_last_boxed = last_boxed_only_string(prediction)
                if string_in_last_boxed is not None:
                    answer = remove_boxed(string_in_last_boxed)
                    if is_equiv(answer, ground_truth):
                        return 1
            except Exception as e:
                return 0

            return 0
        # F1 score for partial matching (used in QA tasks)
        elif metric == 'f1_score':
            f1, prec, recall = f1_score(prediction, ground_truth)
            return f1

        elif metric == 'code_eval':
            if task_id is None:
                raise ValueError("task_id is required for code_eval metric")

            # Check if this is MBPP or HumanEval based on task_id format
            is_mbpp = not str(task_id).startswith("HumanEval")

            if is_mbpp:
                # Case-insensitive pattern to match between [BEGIN] and [DONE]/[Done]
                code_match = re.search(r'\[BEGIN\](.*?)(?:\[DONE\]|\[Done\]|$)', prediction, re.DOTALL | re.IGNORECASE)

                if code_match:
                    code = code_match.group(1).strip()
                else:
                    code = prediction.strip()

                mbpp_sample = {"task_id": task_id, "completion": code}
                pass_1 = entry_point_item_mbpp(mbpp_sample, './dataset/mbpp.jsonl')
                return pass_1

            else:
                # Extract code between [BEGIN] and optional [DONE]
                # code_match = re.search(r'\[BEGIN\](.*?)(?:\[DONE\]|$)', prediction, re.DOTALL)
                code_match = re.search(r'\[BEGIN\](.*?)(?:\[DONE\]|$)', prediction, re.DOTALL | re.IGNORECASE)
                if code_match:
                    # code = code_match.group(1).strip()
                    raw_code = code_match.group(1).strip()
                    if raw_code.lstrip().startswith("def "):
                        code = raw_code
                    else:
                        code = "    " + raw_code.replace("\n", "\n    ") 
                else:
                    # If no tags found, use the entire prediction
                    code = prediction.strip()
                # Create a dict with task_id and completion for evaluation
                dict = {"task_id": task_id, "completion": code}

                # Use the existing entry_point_item function for evaluation
                pass_1 = entry_point_item(dict, './dataset/HumanEval.jsonl')
                return pass_1
        # Default case for unrecognized metrics
        else:
            return 0