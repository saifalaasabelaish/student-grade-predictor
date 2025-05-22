"""
Model serialization utilities for saving and loading trained models.
"""
import joblib
import os
from typing import Any, Dict
from .naive_bayes import NaiveBayesClassifier


def save_model(model: NaiveBayesClassifier, filepath: str, metadata: Dict[str, Any] = None) -> None:
    """
    Save a trained Naive Bayes model to disk.
    
    Args:
        model: Trained NaiveBayesClassifier instance
        filepath: Path where to save the model
        metadata: Optional metadata to save with the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Prepare data to save
    model_data = {
        'model': model,
        'model_info': model.get_model_info(),
        'metadata': metadata or {}
    }
    
    # Save the model
    joblib.dump(model_data, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath: str) -> tuple[NaiveBayesClassifier, Dict[str, Any]]:
    """
    Load a trained Naive Bayes model from disk.
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        Tuple of (loaded model, metadata)
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Load the model data
    model_data = joblib.load(filepath)
    
    model = model_data['model']
    metadata = model_data.get('metadata', {})
    
    print(f"Model loaded from: {filepath}")
    print(f"Model info: {model_data.get('model_info', {})}")
    
    return model, metadata


def get_model_size(filepath: str) -> float:
    """
    Get the size of a saved model file in MB.
    
    Args:
        filepath: Path to the model file
        
    Returns:
        Size in MB
    """
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)  # Convert to MB
    return 0.0


def list_saved_models(directory: str) -> list[str]:
    """
    List all saved model files in a directory.
    
    Args:
        directory: Directory to search for model files
        
    Returns:
        List of model file paths
    """
    if not os.path.exists(directory):
        return []
    
    model_files = []
    for file in os.listdir(directory):
        if file.endswith('.pkl') or file.endswith('.joblib'):
            model_files.append(os.path.join(directory, file))
    
    return sorted(model_files)