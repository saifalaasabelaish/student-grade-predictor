"""
Data loading utilities for the student grade prediction project.
"""
import pandas as pd
from typing import Tuple


def load_student_data(filepath: str = "preprocessed_data.csv") -> pd.DataFrame:
    """
    Load student dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file (defaults to preprocessed_data.csv)
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Dataset file is empty: {filepath}")


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate that the dataset has the required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = [
        'Gender', 'Transporttion', 'Accomodation', 
        'Preparation to midterm', 'Taking notes in classes', 'GRADE'
    ]
    
    return all(col in df.columns for col in required_columns)


def extract_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target variable from the preprocessed dataset.
    
    Args:
        df: Preprocessed dataset DataFrame
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if not validate_dataset(df):
        raise ValueError("Dataset missing required columns")
    
    # Select relevant columns (excluding STUDENT ID)
    feature_columns = [
        'Gender', 'Transporttion', 'Accomodation', 
        'Preparation to midterm', 'Taking notes in classes'
    ]
    target_column = 'GRADE'
    
    # Extract features and target
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    return X, y