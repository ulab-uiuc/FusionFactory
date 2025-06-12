import wandb
import yaml
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from model.multi_task_graph_router import graph_router_prediction
import pandas as pd

def evaluate_model(config_file, test_data_path):
    # Load configuration
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Initialize wandb (optional, can be removed if not needed)
    wandb_key = config.get('wandb_key')
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(project="graph_router")
    
    # Initialize and evaluate
    router = graph_router_prediction(
        router_data_path=test_df,
        llm_path=config['llm_description_path'],
        llm_embedding_path=config['llm_embedding_path'],
        config=config,
        wandb=wandb if wandb_key else None
    )
    
    # Perform evaluation
    router.test_GNN()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/config.yaml",
                      help="Path to the configuration file")
    parser.add_argument("--test_data_path", type=str, default="/data/taofeng2/Router_bench/zijie/full_test_data_process/benchmark_data/benchmark_data.csv",
                      help="Path to the test data CSV file")
    args = parser.parse_args()
    
    evaluate_model(args.config_file, args.test_data_path) 