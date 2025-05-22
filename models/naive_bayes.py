"""
Naive Bayes classifier implementation for student grade prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union


class NaiveBayesClassifier:
    """
    Naive Bayes classifier with Laplace smoothing.
    """
    
    def __init__(self, k: float = 1.0):
        """
        Initialize Naive Bayes Classifier with Laplace smoothing parameter k.
        
        Args:
            k: Laplace smoothing parameter (default: 1.0)
        """
        self.k = k
        self.target_prob = {}
        self.conditional_probabilities = {}
        self.features = []
        self.classes = []
        self.feature_values = {}
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Train the Naive Bayes classifier.
        
        Args:
            X_train: Training features DataFrame
            y_train: Training target Series
        """
        # Store feature names and classes
        self.features = list(X_train.columns)
        self.classes = sorted(list(set(y_train)))
        
        # Store unique values for each feature
        self.feature_values = {}
        for feature in self.features:
            self.feature_values[feature] = sorted(list(set(X_train[feature])))
        
        # Calculate prior probabilities with Laplace smoothing
        self._calculate_prior_probabilities(y_train)
        
        # Calculate conditional probabilities
        self._calculate_conditional_probabilities(X_train, y_train)
    
    def _calculate_prior_probabilities(self, y_train: pd.Series) -> None:
        """
        Calculate prior probabilities P(class) with Laplace smoothing.
        
        Args:
            y_train: Training target Series
        """
        total_samples = len(y_train)
        num_classes = len(self.classes)
        
        self.target_prob = {}
        for class_label in self.classes:
            count = sum(1 for y in y_train if y == class_label)
            # P(class) = (count + k) / (total + k * num_classes)
            self.target_prob[class_label] = (count + self.k) / (total_samples + self.k * num_classes)
    
    def _calculate_conditional_probabilities(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Calculate conditional probabilities P(feature=value|class) with Laplace smoothing.
        
        Args:
            X_train: Training features DataFrame
            y_train: Training target Series
        """
        self.conditional_probabilities = {}
        
        for feature in self.features:
            self.conditional_probabilities[feature] = {}
            feature_values = self.feature_values[feature]
            
            for class_label in self.classes:
                self.conditional_probabilities[feature][class_label] = {}
                
                # Get samples for this class
                class_mask = y_train == class_label
                class_samples = X_train[class_mask]
                class_count = len(class_samples)
                
                for feature_value in feature_values:
                    # Count occurrences of this feature value in this class
                    value_count = sum(1 for val in class_samples[feature] if val == feature_value)
                    # P(feature=value|class) = (count + k) / (class_count + k * num_feature_values)
                    self.conditional_probabilities[feature][class_label][feature_value] = \
                        (value_count + self.k) / (class_count + self.k * len(feature_values))
    
    def predict_single(self, sample: Union[pd.Series, Dict[str, Any]]) -> str:
        """
        Predict class for a single sample.
        
        Args:
            sample: Single sample as pandas Series or dictionary
            
        Returns:
            Predicted class label
        """
        if isinstance(sample, dict):
            sample = pd.Series(sample)
        
        class_scores = {}
        
        for class_label in self.classes:
            # Start with log prior probability
            score = np.log(self.target_prob[class_label])
            
            # Add log conditional probabilities
            for feature in self.features:
                if feature in sample:
                    feature_value = sample[feature]
                    
                    if (feature_value in self.conditional_probabilities[feature][class_label]):
                        prob = self.conditional_probabilities[feature][class_label][feature_value]
                        score += np.log(prob)
                    else:
                        # Handle unseen feature values with smoothing
                        num_feature_values = len(self.feature_values[feature])
                        prob = self.k / (sum(self.conditional_probabilities[feature][class_label].values()) + 
                                       self.k * (num_feature_values + 1))
                        score += np.log(prob)
            
            class_scores[class_label] = score
        
        # Return class with highest score
        return max(class_scores, key=class_scores.get)
    
    def predict(self, X_test: pd.DataFrame) -> List[str]:
        """
        Predict classes for test data.
        
        Args:
            X_test: Test features DataFrame
            
        Returns:
            List of predicted class labels
        """
        predictions = []
        for _, sample in X_test.iterrows():
            predictions.append(self.predict_single(sample))
        return predictions
    
    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predict class probabilities for test data.
        
        Args:
            X_test: Test features DataFrame
            
        Returns:
            DataFrame with class probabilities
        """
        probabilities = []
        
        for _, sample in X_test.iterrows():
            class_scores = {}
            
            for class_label in self.classes:
                score = np.log(self.target_prob[class_label])
                
                for feature in self.features:
                    if feature in sample:
                        feature_value = sample[feature]
                        
                        if feature_value in self.conditional_probabilities[feature][class_label]:
                            prob = self.conditional_probabilities[feature][class_label][feature_value]
                            score += np.log(prob)
                        else:
                            # Handle unseen feature values
                            num_feature_values = len(self.feature_values[feature])
                            prob = self.k / (sum(self.conditional_probabilities[feature][class_label].values()) + 
                                           self.k * (num_feature_values + 1))
                            score += np.log(prob)
                
                class_scores[class_label] = score
            
            # Convert log scores to probabilities
            max_score = max(class_scores.values())
            exp_scores = {cls: np.exp(score - max_score) for cls, score in class_scores.items()}
            total = sum(exp_scores.values())
            probabilities.append({cls: prob / total for cls, prob in exp_scores.items()})
        
        return pd.DataFrame(probabilities)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'k': self.k,
            'num_features': len(self.features),
            'features': self.features,
            'num_classes': len(self.classes),
            'classes': self.classes,
            'feature_values': self.feature_values
        }