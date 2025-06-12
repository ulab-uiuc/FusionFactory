import numpy as np
from typing import List, Dict, Any

def calculate_ndcg(relevance_scores: List[int], k: int = None) -> float:
    """
    Calculate NDCG@k for a list of relevance scores.
    
    Args:
        relevance_scores: List of binary relevance scores (1 for relevant, 0 for non-relevant)
        k: Cutoff for NDCG@k. If None, uses the full list length.
        
    Returns:
        NDCG@k score
    """
    if k is None:
        k = len(relevance_scores)
    k = min(k, len(relevance_scores))
    
    # Calculate DCG
    dcg = 0
    for i in range(k):
        dcg += relevance_scores[i] / np.log2(i + 2)
    
    # Calculate IDCG (ideal DCG)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = 0
    for i in range(k):
        idcg += ideal_scores[i] / np.log2(i + 2)
    
    # Calculate NDCG
    return dcg / idcg if idcg > 0 else 0

def calculate_hit_at_k(relevance_scores: List[int], k: int) -> float:
    """
    Calculate Hit@k for a list of relevance scores.
    
    Args:
        relevance_scores: List of binary relevance scores (1 for relevant, 0 for non-relevant)
        k: Cutoff for Hit@k
        
    Returns:
        Hit@k score (1 if any relevant item in top k, 0 otherwise)
    """
    k = min(k, len(relevance_scores))
    return 1 if any(relevance_scores[:k]) else 0

def calculate_mrr(relevance_scores: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for a list of relevance scores.
    
    Args:
        relevance_scores: List of binary relevance scores (1 for relevant, 0 for non-relevant)
        
    Returns:
        MRR score
    """
    for i, score in enumerate(relevance_scores):
        if score == 1:
            return 1.0 / (i + 1)
    return 0.0

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate average metrics across all results.
    
    Args:
        results: List of dictionaries containing metrics for each query
        
    Returns:
        Dictionary containing average metrics
    """
    metrics = {}
    
    # Get all metric names from the first result
    metric_names = [key for key in results[0].keys() if key not in ["query", "ranked_llms", "ground_truth"]]
    
    # Calculate average for each metric
    for metric in metric_names:
        values = [result[metric] for result in results]
        metrics[metric] = sum(values) / len(values)
    
    return metrics

def print_metrics(metrics: Dict[str, float]) -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary containing metric names and values
    """
    print("\nEvaluation Metrics:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-" * 50) 