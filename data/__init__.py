"""
Data module for student grade prediction project.
"""

from .loader import load_student_data, extract_features_target

__all__ = [
    'load_student_data',
    'extract_features_target'
]