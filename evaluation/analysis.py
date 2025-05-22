"""
Analysis utilities for model evaluation and comparison.
"""
import matplotlib.pyplot as plt
from typing import Dict


def plot_model_comparison(results: Dict[str, Dict[str, float]], 
                         title: str = "Model Performance Comparison on Holdout Dataset",
                         save_path: str = None) -> plt.Figure:
    """
    Compare the performance of each classifier by plotting the accuracy of the versions 
    on the holdout dataset where x-axis is the version number and y-axis is the 
    percentage of the true prediction.
    
    Args:
        results: Dictionary with model results {version: {'True': accuracy, 'False': 1-accuracy}}
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Extract version numbers and accuracies
    versions = sorted(results.keys())  # e.g., ['v1', 'v2', 'v3', 'v4']
    version_numbers = [int(v.replace('v', '')) for v in versions]  # [1, 2, 3, 4]
    accuracies = [results[v]['True'] * 100 for v in versions]  # Convert to percentage
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot line with markers
    ax.plot(version_numbers, accuracies, marker='o', linewidth=2, markersize=8, color='blue')
    
    # Customize the plot exactly as specified
    ax.set_xlabel('Version Number')
    ax.set_ylabel('Percentage of True Prediction (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(version_numbers)
    
    # Add accuracy labels on points
    for x, acc in zip(version_numbers, accuracies):
        ax.annotate(f'{acc:.1f}%', 
                   (x, acc), 
                   textcoords="offset points", 
                   xytext=(0, 10), 
                   ha='center')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig