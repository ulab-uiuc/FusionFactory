from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
import tqdm
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class KNNRouter:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_neighbors: int = 50,
        distance_metric: str = "cosine",
        leaf_size: int = 30,
        models_to_route: list[str] = None,
        **kwargs,
    ) -> None:
        self.models_to_route = models_to_route
        self.embedding_model = embedding_model
        self.embedding_model_instance = None
        self.knn = None
        self.train_data = None
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.leaf_size = leaf_size

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
        
        # Store training data
        self.train_data = train_data
        
        # Initialize and fit KNN
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.distance_metric,
            leaf_size=self.leaf_size,
            n_jobs=-1,
        ).fit(embeddings)

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
            ground_truth = item["ground_truth"]
            
            # Get query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Get scores for all models
            model_scores = self._get_model_scores(query_embedding)
            
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
        model_scores = self._get_model_scores(query_embedding)
        
        # Create ranked list of models
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"llm": model, "score": score} for model, score in ranked_models]

    def _get_model_scores(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """
        Get scores for all models based on KNN neighbors.
        
        Args:
            query_embedding: Embedding of the query
            
        Returns:
            Dictionary mapping model names to scores
        """
        # Find nearest neighbors
        distances, indices = self.knn.kneighbors([query_embedding])
        
        # Get the corresponding training examples
        neighbors = [self.train_data[i] for i in indices[0]]
        
        # Calculate scores for each model
        model_scores = {}
        for model in self.models_to_route:
            # Score is the fraction of neighbors where this model was the ground truth
            score = sum(1 for neighbor in neighbors if neighbor["ground_truth"] == model) / len(neighbors)
            model_scores[model] = score
        
        return model_scores

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
    import json
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    # Set your configuration here
    train_path = "../router_bench_data/llm_judge_train.json"
    test_path = "../router_bench_data/performance_test.json"
    
    # Model configuration
    embedding_model = "all-MiniLM-L6-v2"  # or any other model you want to use
    n_neighbors = 50                      # Number of neighbors to consider
    distance_metric = "cosine"            # Distance metric for KNN
    leaf_size = 30                        # Leaf size for KD-tree
    
    # Load train and test data
    print("Loading training data...")
    train_data = load_data(train_path)
    print(f"Loaded {len(train_data)} training examples")
    
    print("\nLoading test data...")
    test_data = load_data(test_path)
    print(f"Loaded {len(test_data)} test examples")
    
    # Initialize router
    print("\nInitializing router...")
    router = KNNRouter(
        embedding_model=embedding_model,
        n_neighbors=n_neighbors,
        distance_metric=distance_metric,
        leaf_size=leaf_size
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