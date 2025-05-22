"""
Evaluation module for student grade prediction project.
"""

from .metrics import calculate_accuracy
from .analysis import plot_model_comparison

__all__ = [
    'calculate_accuracy',
    'plot_model_comparison'
]