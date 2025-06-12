from typing import List, Dict, Any, Optional
import numpy as np
import json
from sklearn.svm import LinearSVR
from sentence_transformers import SentenceTransformer
import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SVMRouter:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        models_to_route: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the SVM-based router.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
            models_to_route: List of model names to route between. If None, will be determined from training data.
        """
        self.models_to_route = models_to_route
        self.svrs = {}
        self.embedding_model = embedding_model
        self.embedding_model_instance = None

    def train(self, train_data: List[Dict[str, Any]]) -> None:
        """
        Train the router using the provided training data.
        
        Args:
            train_data: List of dictionaries containing training examples with:
                - "query": The input query
                - "candidates": List of candidate models with their metrics
                - "ground_truth": The correct model to route to
        """
        # Shuffle the training data
        np.random.seed(1234)
        train_data = np.random.permutation(train_data).tolist()
        
        # Extract queries
        queries = [item["query"] for item in train_data]
        
        # Get embeddings for all queries
        embeddings = self.get_embeddings(queries)
        
        # Get unique models from training data if not provided
        if self.models_to_route is None:
            all_candidates = set()
            for item in train_data:
                candidates = [c['candidate_name'] for c in item['candidates']]
                all_candidates.update(candidates)
            self.models_to_route = sorted(list(all_candidates))
        print('Candidate models:', self.models_to_route)
        
        # Train SVMs for each model
        for model in tqdm.tqdm(self.models_to_route):
            # Create binary labels (1 if model is ground truth, 0 otherwise)
            labels = np.array([1 if item["ground_truth"] == model else 0 for item in train_data])
            
            # Initialize and train SVM for this model
            self.svrs[model] = LinearSVR(random_state=1234)
            self.svrs[model].fit(embeddings, labels)

    def predict(self, query: str) -> List[Dict[str, Any]]:
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
            score = self.svrs[model].predict([query_embedding])[0]
            model_scores[model] = score
        
        # Create ranked list of models
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"model": model, "score": score} for model, score in ranked_models]

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


def evaluate_predictions(predictions: List[Dict[str, Any]], test_item: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Evaluate predictions against ground truth.
    
    Args:
        predictions: List of dictionaries containing model names and scores
        test_item: The test item containing candidates and ground truth
        
    Returns:
        Dictionary containing evaluation metrics, or None if no valid prediction could be made
    """
    # Get available models in test data
    available_models = {c['candidate_name'] for c in test_item['candidates']}
    
    # Filter predictions to only include available models
    valid_predictions = [p for p in predictions if p["model"] in available_models]
    
    if not valid_predictions:
        print(f"Warning: No matching models found for query: {test_item['query'][:50]}...")
        return None
    
    # Get the predicted model (highest score among available models)
    predicted_model = valid_predictions[0]["model"]
    
    # Find the predicted candidate
    predicted_candidate = next(c for c in test_item['candidates'] if c['candidate_name'] == predicted_model)
    
    # Return metrics for the predicted candidate
    return {
        "cost": predicted_candidate['cost'],
        "performance": predicted_candidate['performance'],
        "score": predicted_candidate['score']
    }


def main():
    # Set your configuration here
    train_path = "../router_bench_data/performance_train.json"
    test_path = "../router_bench_data/performance_test.json"
    
    # Model configuration
    embedding_model = "all-MiniLM-L6-v2"
    
    # Load train and test data
    print("Loading training data...")
    train_data = load_data(train_path)
    print(f"Loaded {len(train_data)} training examples")
    
    print("\nLoading test data...")
    test_data = load_data(test_path)
    print(f"Loaded {len(test_data)} test examples")
    
    # Initialize router
    print("\nInitializing router...")
    router = SVMRouter(embedding_model=embedding_model)
    
    # Train router
    print("\nTraining router...")
    router.train(train_data)
    
    # Evaluate router
    print("\nEvaluating router...")
    all_metrics = []
    for item in test_data:
        query = item["query"]
        
        # Get predictions
        predictions = router.predict(query)
        
        # Evaluate predictions
        metrics = evaluate_predictions(predictions, item)
        if metrics is not None:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("Warning: No valid predictions were made!")
        avg_metrics = {
            "Average Performance": 0.0,
            "Average Cost": 0.0,
            "Average Score": 0.0
        }
    else:
        # Calculate and print average metrics
        avg_metrics = {
            "Average Performance": np.mean([m["performance"] for m in all_metrics]),
            "Average Cost": np.mean([m["cost"] for m in all_metrics]),
            "Average Score": np.mean([m["score"] for m in all_metrics])
        }
    
    print("\nEvaluation Results:")
    for metric_name, value in avg_metrics.items():
        print(f"{metric_name}: {value:.4f}")


if __name__ == '__main__':
    main()