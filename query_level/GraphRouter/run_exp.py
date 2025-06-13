import wandb
import yaml
import argparse
import sys
import os
from datasets import load_dataset
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from model.multi_task_graph_router import graph_router_prediction
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="configs/config.yaml")
args = parser.parse_args()

with open(args.config_file, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Load data from HuggingFace dataset
dataset = load_dataset("ulab-ai/FusionBench")
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['partial_test'])

new_df = pd.concat([train_df, test_df], ignore_index=True)
wandb_key = config['wandb_key']
wandb.login(key=wandb_key)
wandb.init(project="graph_router")
graph_router_prediction(router_data_path=new_df,llm_path=config['llm_description_path'],
                        llm_embedding_path=config['llm_embedding_path'],config=config,wandb=wandb)