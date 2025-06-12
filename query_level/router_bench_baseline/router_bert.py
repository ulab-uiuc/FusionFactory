import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

import json
from typing import List, Dict, Any
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.nn.functional import cosine_similarity
import re
from datasets import Dataset

class PassageBERTRouter:
    def __init__(self, model_name: str = "AmirMohseni/BERT-Router-large-v2"):
        """
        Initialize BERT-Router for medical QA data.
        
        Args:
            model_name: Name of the BERT-Router model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)  # Move model to device before setting eval mode
        self.model.eval()
        
    def get_embedding(self, text: str) -> torch.Tensor:
        """Get the embedding for a given text using BERT-Router."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the [CLS] token embedding as the document representation
            embedding = outputs.logits
        
        return embedding

    def prepare_training_data(self, json_file: str) -> Dataset:
        """Prepare training data for finetuning."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Convert data to format expected by transformers
        processed_data = []
        for item in data:
            query = str(item["query"]).strip()  # Ensure query is string
            candidates = item["candidates"]
            gt_llm = str(item["ground_truth"]).strip()  # Ensure ground truth is string
            
            # Create positive and negative pairs
            for candidate in candidates:
                candidate_name = str(candidate["candidate_name"]).strip()  # Ensure candidate name is string
                label = 1 if candidate_name == gt_llm else 0
                processed_data.append({
                    "query": query,
                    "llm": candidate_name,
                    "label": label
                })
        
        return Dataset.from_list(processed_data)

    def train(self, train_file: str, output_dir: str = "finetuned_router", num_epochs: int = 1):
        """Finetune the model on the training data."""
        # Prepare training data
        train_dataset = self.prepare_training_data(train_file)
        
        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["query"],
                examples["llm"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        
        # Calculate steps for 0.1 epoch
        batch_size = 8
        total_steps = len(tokenized_train) // batch_size
        steps_for_0_1_epoch = int(total_steps * 0.1)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=steps_for_0_1_epoch,  # Stop after 0.1 epoch
            per_device_train_batch_size=batch_size,
            save_steps=1000,
            save_total_limit=2,
            logging_steps=100,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=500,
            report_to="none",  # Disable all logging integrations
            no_cuda=False,  # Use CUDA
            local_rank=-1,  # Disable distributed training
            dataloader_num_workers=0,  # Disable multiprocessing for data loading
            ddp_find_unused_parameters=False,  # Disable unused parameter detection
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Load the finetuned model
        self.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        self.model = self.model.to(self.device)  # Move model to device before setting eval mode
        self.model.eval()
        
    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single item from the JSON file.
        
        Args:
            item: Dictionary containing query, candidates, and ground_truth
            
        Returns:
            Dictionary containing query, ranked LLMs, and metrics
        """
        # Get query
        query = str(item["query"]).strip()  # Ensure query is string
        
        # Get candidates
        candidates = item["candidates"]
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Get candidate embeddings and calculate similarities
        candidate_embeddings = []
        for candidate in candidates:
            candidate_embedding = self.get_embedding(candidate["candidate_name"])
            candidate_embeddings.append(candidate_embedding)
        
        # Calculate similarities
        similarities = []
        for candidate_embedding in candidate_embeddings:
            similarity = cosine_similarity(query_embedding, candidate_embedding)
            similarities.append(similarity.item())
        
        # Get ranked results
        top_indices = np.argsort(similarities)[::-1]
        
        ranked_results = []
        for idx in top_indices:
            ranked_results.append({
                "llm": candidates[idx]["candidate_name"],
                "score": float(similarities[idx]),
                "cost": candidates[idx]["cost"],
                "performance": candidates[idx]["performance"],
                "score": candidates[idx]["score"]
            })
            
        return {
            "query": query,
            "ranked_llms": ranked_results,
            "ground_truth": str(item["ground_truth"]).strip()
        }
        
    def process_file(self, json_file: str) -> List[Dict[str, Any]]:
        """
        Process the entire JSON file.
        
        Args:
            json_file: Path to the JSON file
            
        Returns:
            List of processed items with BERT-Router rankings
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        results = []
        for item in data:
            results.append(self.process_item(item))
            
        return results

def main():
    # Initialize the processor
    processor = PassageBERTRouter()
    
    # Paths for training and testing
    train_file = "../router_bench_data/balance_train.json"
    test_file = "../router_bench_data/balance_test.json"
    
    # Finetune the model
    print("Starting finetuning...")
    processor.train(train_file, output_dir="finetuned_router")
    print("Finetuning completed!")
    
    # Process the test file
    print("Evaluating on test set...")
    results = processor.process_file(test_file)
    
    # Calculate average metrics
    total_cost = 0
    total_performance = 0
    total_score = 0
    count = 0
    
    for result in results:
        # Get the top-ranked model's metrics
        if not result["ranked_llms"]:
            print(f"Warning: No valid predictions for query: {result['query'][:50]}...")
            continue
            
        top_model = result["ranked_llms"][0]
        total_cost += top_model["cost"]
        total_performance += top_model["performance"]
        total_score += top_model["score"]
        count += 1
    
    if count == 0:
        print("Warning: No valid predictions were made!")
        avg_metrics = {
            "Average Cost": 0.0,
            "Average Performance": 0.0,
            "Average Score": 0.0
        }
    else:
        # Calculate averages
        avg_metrics = {
            "Average Cost": total_cost / count,
            "Average Performance": total_performance / count,
            "Average Score": total_score / count
        }
    
    # Print metrics
    print("\nEvaluation Results:")
    for metric_name, value in avg_metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()
