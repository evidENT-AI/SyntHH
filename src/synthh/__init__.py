"""
SyntHH: Synthetic Hearing Health Data Generation and Validation

A research project for generating synthetic audiometric data that preserves
statistical properties of real hearing measurements while protecting patient privacy.
"""

__version__ = "0.1.0"
__author__ = "LB"

# Import main preprocessing components
from .data_loader import NHANESDataLoader, load_nhanes_data
from .data_cleaner import NHANESDataCleaner, clean_nhanes_data  
from .feature_engineering import NHANESFeatureEngineer, engineer_nhanes_features
from .preprocessing_pipeline import NHANESPreprocessingPipeline, preprocess_nhanes_data

__all__ = [
    # Data Loading
    'NHANESDataLoader',
    'load_nhanes_data',
    
    # Data Cleaning  
    'NHANESDataCleaner',
    'clean_nhanes_data',
    
    # Feature Engineering
    'NHANESFeatureEngineer', 
    'engineer_nhanes_features',
    
    # Preprocessing Pipeline
    'NHANESPreprocessingPipeline',
    'preprocess_nhanes_data'
]