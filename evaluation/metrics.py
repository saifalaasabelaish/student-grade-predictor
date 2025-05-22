"""
Simple evaluation metrics for the student grade prediction project.
"""
from typing import List


def calculate_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """
    Calculate accuracy between true and predicted labels.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(y_true) == 0:
        return 0.0
    
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)