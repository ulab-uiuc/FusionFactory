import pandas as pd
from vllm import LLM, SamplingParams
from typing import List
import os

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
SYSTEM_PROMPT = "You are a helpful AI assistant. Please provide clear and accurate responses."
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def format_prompt(system_prompt: str, user_query: str) -> str:
    return (
        f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}\n<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n{user_query}\n<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

class ModelManager:
    def __init__(self):
        self.model = None
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.7,
            max_tokens=5000
        )

    def get_model(self) -> LLM:
        if self.model is None:
            print(f"Initializing Llama 3.1 8B model")
            self.model = LLM(
                model=MODEL_PATH,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=5000,
                device="cuda"
            )
        return self.model

    def process_batch(self, queries: List[str], batch_size: int = 16) -> List[str]:
        model = self.get_model()
        formatted = [format_prompt(SYSTEM_PROMPT, q) for q in queries]
        outputs = []

        for i in range(0, len(formatted), batch_size):
            batch = formatted[i:i + batch_size]
            result = model.generate(batch, self.sampling_params)
            outputs.extend([r.outputs[0].text for r in result])

        return outputs

def process_csv(input_file: str, output_file: str, batch_size: int = 16):
    df = pd.read_csv(input_file)
    model_manager = ModelManager()
    queries = df['query'].tolist()
    print(f"Processing {len(queries)} queries")
    responses = model_manager.process_batch(queries, batch_size=batch_size)
    df['output'] = responses
    df.to_csv(output_file, index=False)
    print(f"Done. Output saved to {output_file}")

if __name__ == "__main__":
    input_file = "./dataset/router_data.csv"
    output_file = "./dataset/router_data_with_response.csv"
    process_csv(input_file, output_file, batch_size=2048)
