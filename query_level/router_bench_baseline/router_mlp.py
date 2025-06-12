from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
import tqdm
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor
import json
from sentence_transformers import SentenceTransformer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MLPRouter():
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",  # Default to consistent model
        hidden_layer_sizes: list[int] = (50,),
        activation_function: str = "relu",
        learning_rate_method: str = "constant",
        learning_rate: float = 0.0001,
        models_to_route: list[str] = None,
        **kwargs,
    ) -> None:
        self.models_to_route = models_to_route
        self.mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=200,
            random_state=1234,
            activation=activation_function,
            learning_rate=learning_rate_method,
            learning_rate_init=learning_rate,
            alpha=0.01,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False,
        )
        self.embedding_model = embedding_model
        self.mlps = {}
        self.embedding_model_instance = None  # Cache the embedding model

    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Train the router using the provided training data.
        
        Args:
            train_data: List of dictionaries containing training examples
        """
        # Shuffle the training data
        np.random.seed(1234)
        train_data = np.random.permutation(train_data).tolist()
        
        # Extract queries and ground truth scores
        queries = [item["query"] for item in train_data]
        
        # Get embeddings for all queries
        embeddings = self.get_embeddings(queries)
        
        # Get unique models from training data
        if self.models_to_route is None:
            # Get all unique models from the entire training set
            all_candidates = set()
            for item in train_data:
                # Get unique candidate names
                candidates = [c['candidate_name'] for c in item['candidates']]
                all_candidates.update(candidates)
            self.models_to_route = sorted(list(all_candidates))
        print('candidate llm: ', self.models_to_route)
        
        # Train MLPs for each model
        for model in tqdm.tqdm(self.models_to_route):
            # Create binary labels (1 if model is ground truth, 0 otherwise)
            labels = np.array([1 if item["ground_truth"] == model else 0 for item in train_data])
            
            # Initialize MLP for this model if not already done
            if model not in self.mlps:
                self.mlps[model] = MLPRegressor(
                    hidden_layer_sizes=self.mlp.hidden_layer_sizes,
                    max_iter=500,
                    random_state=1234,
                    activation=self.mlp.activation,
                    learning_rate=self.mlp.learning_rate,
                    learning_rate_init=self.mlp.learning_rate_init,
                    alpha=0.01,
                    early_stopping=True,
                    validation_fraction=0.1,
                    verbose=False,
                )
            
            # Train the MLP
            self.mlps[model].fit(embeddings, labels)

    def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the router on test data using average metrics.
        
        Args:
            test_data: List of dictionaries containing test examples
            
        Returns:
            Dictionary containing evaluation metrics
        """
        total_cost = 0
        total_performance = 0
        total_score = 0
        count = 0
        
        for item in test_data:
            query = item["query"]
            
            # Get query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Get scores for all models
            model_scores = {}
            for model in self.models_to_route:
                score = self.mlps[model].predict([query_embedding])[0]
                model_scores[model] = score
            
            # Filter model scores to only include models available in test data
            available_models = {c['candidate_name'] for c in item['candidates']}
            filtered_scores = {model: score for model, score in model_scores.items() 
                             if model in available_models}
            
            if not filtered_scores:
                print(f"Warning: No matching models found for query: {query[:50]}...")
                continue
                
            # Find the predicted best model from available models
            predicted_model = max(filtered_scores.items(), key=lambda x: x[1])[0]
            
            # Find the predicted candidate
            predicted_candidate = next(c for c in item['candidates'] if c['candidate_name'] == predicted_model)
            
            # Add metrics
            total_cost += predicted_candidate['cost']
            total_performance += predicted_candidate['performance']
            total_score += predicted_candidate['score']
            count += 1
        
        if count == 0:
            print("Warning: No valid predictions were made!")
            return {
                "Average Performance": 0.0,
                "Average Cost": 0.0,
                "Average Score": 0.0
            }
        
        # Calculate averages
        avg_cost = total_cost / count
        avg_performance = total_performance / count
        avg_score = total_score / count
        
        return {
            "Average Performance": avg_performance,
            "Average Cost": avg_cost,
            "Average Score": avg_score
        }
    
    def predict_rankings(self, query: str) -> List[Dict[str, Any]]:
        """
        Predict rankings for a given query.
        
        Args:
            query: The input query
            
        Returns:
            List of dictionaries containing model names and their scores, sorted by score
        """
        # Get query embedding
        query_embedding = self.get_embeddings([query])[0]
        
        # Get scores for all models
        model_scores = {}
        for model in self.models_to_route:
            score = self.mlps[model].predict([query_embedding])[0]
            model_scores[model] = score
        
        # Create ranked list of models
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"llm": model, "score": score} for model, score in ranked_models]

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts using the specified embedding model.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            numpy array of embeddings
        """
        if self.embedding_model_instance is None:
            self.embedding_model_instance = SentenceTransformer(self.embedding_model)
        embeddings = self.embedding_model_instance.encode(texts)
        return embeddings

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries containing the data
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    # Set your configuration here
    train_path = "../router_bench_data/llm_judge_train.json"
    test_path = "../router_bench_data/performance_test.json"
    
    # Model configuration
    embedding_model = "all-MiniLM-L6-v2"  # or any other model you want to use
    hidden_layer_sizes = [1]   # MLP hidden layer sizes
    activation = "relu"                    # Activation function
    learning_rate = 0.001                  # Learning rate
    
    # Load train and test data
    print("Loading training data...")
    train_data = load_data(train_path)
    print(f"Loaded {len(train_data)} training examples")
    
    print("\nLoading test data...")
    test_data = load_data(test_path)
    print(f"Loaded {len(test_data)} test examples")
    
    # Initialize router
    print("\nInitializing router...")
    router = MLPRouter(
        embedding_model=embedding_model,
        hidden_layer_sizes=hidden_layer_sizes,
        activation_function=activation,
        learning_rate=learning_rate
    )
    
    # Train router
    print("\nTraining router...")
    router.train(train_data)
    
    # Evaluate router
    print("\nEvaluating router...")
    metrics = router.evaluate(test_data)
    
    # Print metrics
    print("\nEvaluation Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    main()