"""
Improved main pipeline for integer-encoded preprocessed data.
"""
import os
import sys
import random
import numpy as np
import pandas as pd  # Added missing import
from sklearn.model_selection import StratifiedShuffleSplit

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import load_student_data, extract_features_target
from models import NaiveBayesClassifier, save_model
from evaluation import calculate_accuracy, plot_model_comparison


def stratified_split_data(X, y, train_ratio=0.8, holdout_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Use stratified sampling to ensure balanced class distribution in all splits.
    """
    # First split: train vs (holdout + test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(holdout_ratio + test_ratio), random_state=random_state)
    train_idx, temp_idx = next(sss1.split(X, y))
    
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    
    X_temp = X.iloc[temp_idx].reset_index(drop=True)
    y_temp = y.iloc[temp_idx].reset_index(drop=True)
    
    # Second split: holdout vs test
    test_size_ratio = test_ratio / (holdout_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size_ratio, random_state=random_state)
    holdout_idx, test_idx = next(sss2.split(X_temp, y_temp))
    
    X_holdout = X_temp.iloc[holdout_idx].reset_index(drop=True)
    y_holdout = y_temp.iloc[holdout_idx].reset_index(drop=True)
    
    X_test = X_temp.iloc[test_idx].reset_index(drop=True)
    y_test = y_temp.iloc[test_idx].reset_index(drop=True)
    
    return X_train, X_holdout, X_test, y_train, y_holdout, y_test


def analyze_data_distribution(X, y):
    """Analyze the data distribution to understand patterns."""
    print("\n" + "="*50)
    print("DATA ANALYSIS")
    print("="*50)
    
    # Class distribution
    print("Class Distribution:")
    class_counts = y.value_counts().sort_index()
    for grade, count in class_counts.items():
        percentage = (count / len(y)) * 100
        print(f"  Grade {grade}: {count} samples ({percentage:.1f}%)")
    
    # Feature analysis
    print(f"\nFeature Value Distributions:")
    for feature in X.columns:
        print(f"\n{feature}:")
        value_counts = X[feature].value_counts().sort_index()
        for value, count in value_counts.items():
            percentage = (count / len(X)) * 100
            print(f"  {value}: {count} ({percentage:.1f}%)")
    
    # Cross-tabulation analysis
    print(f"\nFeature-Grade Relationships:")
    for feature in X.columns:
        print(f"\n{feature} vs Grade (percentages by feature value):")
        crosstab = pd.crosstab(X[feature], y, normalize='index') * 100
        print(crosstab.round(1))


def train_and_evaluate_models(X_train, X_holdout, X_test, y_train, y_holdout, y_test):
    """Train multiple models with different k values."""
    print("\n" + "="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    k_values = [0, 1, 2, 3]
    holdout_results = {}
    classifiers = {}
    
    for i, k in enumerate(k_values):
        version_name = f'v{i+1}'
        print(f"\nTraining {version_name} with k={k}...")
        
        # Train classifier
        classifier = NaiveBayesClassifier(k=k)
        classifier.fit(X_train, y_train)
        classifiers[version_name] = classifier
        
        # Evaluate on holdout set
        holdout_predictions = classifier.predict(X_holdout)
        holdout_accuracy = calculate_accuracy(y_holdout, holdout_predictions)
        
        # Evaluate on training set (to check overfitting)
        train_predictions = classifier.predict(X_train)
        train_accuracy = calculate_accuracy(y_train, train_predictions)
        
        holdout_results[version_name] = {
            'True': holdout_accuracy,
            'False': 1 - holdout_accuracy,
            'k': k,
            'train_accuracy': train_accuracy
        }
        
        print(f"  Training accuracy: {train_accuracy:.3f}")
        print(f"  Holdout accuracy: {holdout_accuracy:.3f}")
        print(f"  Overfitting: {train_accuracy - holdout_accuracy:.3f}")
    
    # Find best model
    best_version = max(holdout_results, key=lambda x: holdout_results[x]['True'])
    best_classifier = classifiers[best_version]
    
    print(f"\nBest performing model: {best_version} (k={holdout_results[best_version]['k']})")
    print(f"Best holdout accuracy: {holdout_results[best_version]['True']:.3f}")
    
    # Final test evaluation
    test_predictions = best_classifier.predict(X_test)
    test_accuracy = calculate_accuracy(y_test, test_predictions)
    
    print(f"Final test accuracy: {test_accuracy:.3f}")
    
    # Detailed analysis of predictions
    print(f"\nDetailed Test Results:")
    print(f"Actual distribution: {dict(y_test.value_counts().sort_index())}")
    predicted_counts = pd.Series(test_predictions).value_counts().sort_index()
    print(f"Predicted distribution: {dict(predicted_counts)}")
    
    return {
        'holdout_results': holdout_results,
        'best_model': best_classifier,
        'best_version': best_version,
        'test_accuracy': test_accuracy,
        'test_predictions': test_predictions,
        'all_classifiers': classifiers
    }


def main():
    """Improved main function for better results with encoded data."""
    print("IMPROVED Student Grade Prediction Pipeline")
    print("=" * 60)
    
    # 1. Load data (keeping it as integers since that's how it's encoded)
    print("\n1. Loading preprocessed data...")
    df = load_student_data('preprocessed_data.csv')
    
    # Don't decode - work with integers directly since they're properly encoded
    feature_columns = ['Gender', 'Transporttion', 'Accomodation', 
                      'Preparation to midterm', 'Taking notes in classes']
    target_column = 'GRADE'
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    print(f"Data loaded: {len(X)} samples, {len(X.columns)} features")
    print(f"Classes: {sorted(y.unique())}")
    
    # 2. Rename columns for consistency
    X.columns = ['Gender', 'Transportation', 'Accommodation', 'MidExam', 'TakingNotes']
    
    # 3. Analyze data distribution
    analyze_data_distribution(X, y)
    
    # 4. Use stratified splitting for better results
    print(f"\n2. Stratified data splitting...")
    X_train, X_holdout, X_test, y_train, y_holdout, y_test = stratified_split_data(X, y)
    
    print(f"Train: {len(X_train)} samples")
    print(f"Holdout: {len(X_holdout)} samples") 
    print(f"Test: {len(X_test)} samples")
    
    # Print class distributions for each split
    print(f"\nClass distributions:")
    print(f"Train: {dict(y_train.value_counts().sort_index())}")
    print(f"Holdout: {dict(y_holdout.value_counts().sort_index())}")
    print(f"Test: {dict(y_test.value_counts().sort_index())}")
    
    # 5. Train and evaluate models
    results = train_and_evaluate_models(X_train, X_holdout, X_test, y_train, y_holdout, y_test)
    
    # 6. Create performance plot
    print(f"\n3. Creating performance comparison plot...")
    plot_model_comparison(
        results['holdout_results'],
        title="Naive Bayes Performance Comparison (Stratified Validation)",
        save_path="improved_model_comparison.png"
    )
    print("Plot saved as 'improved_model_comparison.png'")
    
    # 7. Save best model
    print(f"\n4. Saving best model...")
    os.makedirs('saved_models', exist_ok=True)
    
    metadata = {
        'version': results['best_version'],
        'k_value': results['holdout_results'][results['best_version']]['k'],
        'holdout_accuracy': results['holdout_results'][results['best_version']]['True'],
        'test_accuracy': results['test_accuracy'],
        'training_method': 'stratified_sampling',
        'data_encoding': 'integer_encoded',
        'features': list(X.columns),
        'classes': sorted(y.unique())
    }
    
    save_model(results['best_model'], 'saved_models/best_model_improved.pkl', metadata)
    
    # 8. Final summary
    print(f"\n" + "="*60)
    print("IMPROVED PIPELINE COMPLETED!")
    print("="*60)
    print(f"Best Model: {results['best_version']} (k={metadata['k_value']})")
    print(f"Holdout Accuracy: {metadata['holdout_accuracy']:.3f} ({metadata['holdout_accuracy']*100:.1f}%)")
    print(f"Test Accuracy: {metadata['test_accuracy']:.3f} ({metadata['test_accuracy']*100:.1f}%)")
    print(f"Improvement method: Stratified sampling with integer-encoded data")
    print(f"Model saved to: saved_models/best_model_improved.pkl")
    print("="*60)


if __name__ == "__main__":
    main()