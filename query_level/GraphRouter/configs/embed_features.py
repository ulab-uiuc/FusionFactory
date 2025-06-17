import json
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from transformers import LongformerTokenizer, LongformerModel

def get_cls_embedding(text, model_name="allenai/longformer-base-4096", device="cuda:0"):
    """
    Extracts the [CLS] embedding from a given text using Longformer.

    Args:
        text (str): Input text
        model_name (str): Hugging Face model name
        device (str): "cpu" or "cuda"

    Returns:
        torch.Tensor: CLS embedding of shape (1, hidden_size)
    """
    # Load tokenizer and model
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    model = LongformerModel.from_pretrained(model_name).to(device)
    model.eval()

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)

    return cls_embedding

def main():
    # Load the JSON file
    with open('large_LLM_Description_with_think.json', 'r') as f:
        llm_descriptions = json.load(f)
    
    # Initialize BERT model and tokenizer
    model_name = "allenai/longformer-base-4096"  # This model outputs 768-dimensional embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Generate embeddings for each feature
    embeddings = {}
    for model_name, info in llm_descriptions.items():
        feature_text = info['feature']
        embedding = get_cls_embedding(feature_text, model_name)
        embeddings[model_name] = embedding
    
    # Convert to numpy array
    model_names = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[name] for name in model_names])
    
    # Save embeddings as numpy array
    with open('large_llm_description_embedding_with_think.pkl', 'wb') as f:
        pickle.dump(embedding_matrix, f)
    
    print(f"Generated embeddings of shape: {embedding_matrix.shape}")
    print(f"Saved embeddings to configs/large_llm_description_embedding_with_think.pkl")

if __name__ == "__main__":
    main() 