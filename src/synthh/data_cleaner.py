"""
NHANES Data Cleaning and Validation Module

This module provides functions for cleaning, validating, and preprocessing 
NHANES audiometric data, including handling missing values, outlier detection,
and data quality assessment.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


class NHANESDataCleaner:
    """
    A class for cleaning and validating NHANES audiometric data.
    
    This class provides methods for handling missing values, detecting outliers,
    validating data quality, and preparing data for analysis.
    """
    
    def __init__(self):
        """Initialize the NHANES data cleaner with default parameters."""
        # Define physiologically plausible hearing threshold ranges
        self.threshold_ranges = {
            'min_threshold': -20,  # Minimum plausible hearing threshold (dB HL)
            'max_threshold': 120,  # Maximum plausible hearing threshold (dB HL)
            'normal_range': (-10, 25),  # Normal hearing range
            'mild_loss_range': (26, 40),  # Mild hearing loss
            'moderate_loss_range': (41, 70),  # Moderate hearing loss
            'severe_loss_range': (71, 95),  # Severe hearing loss
            'profound_loss_range': (96, 120)  # Profound hearing loss
        }
        
        # Age ranges for validation
        self.age_ranges = {
            'min_age': 12,  # NHANES minimum age for audiometry
            'max_age': 85,  # Practical maximum age
            'pediatric': (12, 18),
            'adult': (18, 65),
            'older_adult': (65, 85)
        }
    
    def validate_data_ranges(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate that data values are within expected physiological ranges.
        
        Args:
            df: DataFrame containing NHANES data.
            
        Returns:
            Dictionary containing validation results and any issues found.
        """
        issues = {
            'threshold_outliers': [],
            'age_outliers': [],
            'missing_seqn': [],
            'invalid_gender': [],
            'invalid_ethnicity': []
        }
        
        # Check hearing thresholds
        pta_columns = [col for col in df.columns if 'kHz' in col]
        for col in pta_columns:
            if col in df.columns:
                outliers = df[
                    (df[col] < self.threshold_ranges['min_threshold']) |
                    (df[col] > self.threshold_ranges['max_threshold'])
                ][col].dropna()
                
                if len(outliers) > 0:
                    issues['threshold_outliers'].append({
                        'column': col,
                        'count': len(outliers),
                        'values': outliers.tolist()
                    })
        
        # Check age ranges
        if 'Age (years)' in df.columns:
            age_outliers = df[
                (df['Age (years)'] < self.age_ranges['min_age']) |
                (df['Age (years)'] > self.age_ranges['max_age'])
            ]['Age (years)'].dropna()
            
            if len(age_outliers) > 0:
                issues['age_outliers'] = age_outliers.tolist()
        
        # Check for missing SEQN (should never happen)
        if 'SEQN' in df.columns:
            missing_seqn = df[df['SEQN'].isna()]
            if len(missing_seqn) > 0:
                issues['missing_seqn'] = missing_seqn.index.tolist()
        
        # Check gender coding
        if 'RIAGENDR' in df.columns:
            invalid_gender = df[~df['RIAGENDR'].isin([1.0, 2.0, np.nan])]
            if len(invalid_gender) > 0:
                issues['invalid_gender'] = invalid_gender['RIAGENDR'].tolist()
        
        # Check ethnicity coding
        if 'RIDRETH1' in df.columns:
            invalid_ethnicity = df[~df['RIDRETH1'].isin([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])]
            if len(invalid_ethnicity) > 0:
                issues['invalid_ethnicity'] = invalid_ethnicity['RIDRETH1'].tolist()
        
        return issues
    
    def detect_statistical_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Detect statistical outliers in hearing threshold data.
        
        Args:
            df: DataFrame containing hearing threshold data.
            method: Method for outlier detection ('iqr', 'zscore', or 'modified_zscore').
            threshold: Threshold for outlier detection.
            
        Returns:
            Dictionary containing outlier information for each PTA column.
        """
        outliers = {}
        pta_columns = [col for col in df.columns if 'kHz' in col]
        
        for col in pta_columns:
            if col in df.columns:
                data = df[col].dropna()
                
                if method == 'iqr':
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outlier_mask = (data < lower_bound) | (data > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(data))
                    outlier_mask = z_scores > threshold
                    
                elif method == 'modified_zscore':
                    median = np.median(data)
                    mad = np.median(np.abs(data - median))
                    modified_z_scores = 0.6745 * (data - median) / mad
                    outlier_mask = np.abs(modified_z_scores) > threshold
                
                if outlier_mask.any():
                    outlier_indices = data[outlier_mask].index
                    outliers[col] = df.loc[outlier_indices, ['SEQN', col]].copy()
        
        return outliers
    
    def assess_missing_data(self, df: pd.DataFrame) -> Dict[str, Union[int, float, pd.DataFrame]]:
        """
        Assess missing data patterns in the dataset.
        
        Args:
            df: DataFrame to assess.
            
        Returns:
            Dictionary containing missing data statistics.
        """
        missing_stats = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2),
            'complete_cases': df.dropna().shape[0],
            'incomplete_cases': df.shape[0] - df.dropna().shape[0]
        }
        
        # Find patterns of missingness
        if missing_stats['total_missing'] > 0:
            missing_patterns = df.isnull().value_counts().head(10)
            missing_stats['missing_patterns'] = missing_patterns
        
        return missing_stats
    
    def clean_hearing_thresholds(
        self,
        df: pd.DataFrame,
        round_to_nearest: int = 5,
        handle_outliers: str = 'clip',
        outlier_method: str = 'physiological'
    ) -> pd.DataFrame:
        """
        Clean hearing threshold values.
        
        Args:
            df: DataFrame containing hearing threshold data.
            round_to_nearest: Round thresholds to nearest N dB (typically 5).
            handle_outliers: How to handle outliers ('clip', 'remove', or 'keep').
            outlier_method: Method for outlier detection ('physiological' or 'statistical').
            
        Returns:
            Cleaned DataFrame.
        """
        df_clean = df.copy()
        pta_columns = [col for col in df.columns if 'kHz' in col]
        
        for col in pta_columns:
            if col in df_clean.columns:
                # Round to nearest 5 dB (standard audiometric practice)
                if round_to_nearest > 0:
                    df_clean[col] = (df_clean[col] / round_to_nearest).round() * round_to_nearest
                
                # Handle outliers
                if handle_outliers != 'keep':
                    if outlier_method == 'physiological':
                        if handle_outliers == 'clip':
                            df_clean[col] = df_clean[col].clip(
                                lower=self.threshold_ranges['min_threshold'],
                                upper=self.threshold_ranges['max_threshold']
                            )
                        elif handle_outliers == 'remove':
                            outlier_mask = (
                                (df_clean[col] < self.threshold_ranges['min_threshold']) |
                                (df_clean[col] > self.threshold_ranges['max_threshold'])
                            )
                            df_clean.loc[outlier_mask, col] = np.nan
                    
                    elif outlier_method == 'statistical':
                        outliers = self.detect_statistical_outliers(
                            df_clean[[col]], method='iqr', threshold=1.5
                        )
                        if col in outliers:
                            outlier_indices = outliers[col].index
                            if handle_outliers == 'remove':
                                df_clean.loc[outlier_indices, col] = np.nan
        
        return df_clean
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'listwise',
        min_valid_frequencies: int = 3,
        imputation_method: str = 'median'
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potentially missing values.
            strategy: Strategy for handling missing data:
                     'listwise' - remove rows with any missing PTA data
                     'pairwise' - keep all available data
                     'impute' - impute missing values
                     'partial' - require minimum number of valid frequencies
            min_valid_frequencies: Minimum valid frequencies required (for 'partial').
            imputation_method: Method for imputation ('median', 'mean', 'forward_fill').
            
        Returns:
            DataFrame with missing values handled according to strategy.
        """
        df_clean = df.copy()
        pta_columns = [col for col in df.columns if 'kHz' in col]
        
        if strategy == 'listwise':
            # Remove any rows with missing PTA data
            df_clean = df_clean.dropna(subset=pta_columns)
            
        elif strategy == 'partial':
            # Keep rows with at least min_valid_frequencies non-missing values
            valid_counts = df_clean[pta_columns].notna().sum(axis=1)
            df_clean = df_clean[valid_counts >= min_valid_frequencies]
            
        elif strategy == 'impute':
            for col in pta_columns:
                if col in df_clean.columns:
                    if imputation_method == 'median':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    elif imputation_method == 'mean':
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                    elif imputation_method == 'forward_fill':
                        df_clean[col] = df_clean[col].fillna(method='ffill')
        
        # 'pairwise' strategy keeps data as-is
        
        return df_clean.reset_index(drop=True)
    
    def validate_audiometric_patterns(self, df: pd.DataFrame) -> Dict[str, List]:
        """
        Validate audiometric patterns for clinical plausibility.
        
        Args:
            df: DataFrame containing audiometric data.
            
        Returns:
            Dictionary containing validation results.
        """
        issues = {
            'implausible_asymmetry': [],
            'unusual_configurations': [],
            'inconsistent_thresholds': []
        }
        
        pta_columns = [col for col in df.columns if 'kHz' in col]
        
        # Check for implausible inter-aural asymmetry (>40 dB difference)
        frequencies = ['0.5kHz', '1kHz', '2kHz', '4kHz', '8kHz']
        for freq in frequencies:
            right_col = f'{freq} Right'
            left_col = f'{freq} Left'
            
            if right_col in df.columns and left_col in df.columns:
                asymmetry = abs(df[right_col] - df[left_col])
                implausible = df[asymmetry > 40].index.tolist()
                if implausible:
                    issues['implausible_asymmetry'].append({
                        'frequency': freq,
                        'cases': implausible
                    })
        
        # Check for unusual audiometric configurations
        for idx, row in df.iterrows():
            if pd.notna(row[pta_columns]).all():
                # Check for "corner audiogram" - sudden drop at high frequencies
                right_thresholds = [
                    row['0.5kHz Right'], row['1kHz Right'], 
                    row['2kHz Right'], row['4kHz Right'], row['8kHz Right']
                ]
                left_thresholds = [
                    row['0.5kHz Left'], row['1kHz Left'], 
                    row['2kHz Left'], row['4kHz Left'], row['8kHz Left']
                ]
                
                # Flag sudden drops >30 dB between adjacent frequencies
                for thresholds, ear in [(right_thresholds, 'Right'), (left_thresholds, 'Left')]:
                    for i in range(len(thresholds) - 1):
                        if abs(thresholds[i+1] - thresholds[i]) > 30:
                            issues['unusual_configurations'].append({
                                'index': idx,
                                'ear': ear,
                                'pattern': f'Sudden drop between freq {i} and {i+1}'
                            })
        
        return issues
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive data quality report.
        
        Args:
            df: DataFrame to assess.
            
        Returns:
            Dictionary containing comprehensive quality assessment.
        """
        report = {
            'dataset_info': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'pta_columns': len([col for col in df.columns if 'kHz' in col])
            },
            'range_validation': self.validate_data_ranges(df),
            'missing_data': self.assess_missing_data(df),
            'statistical_outliers': self.detect_statistical_outliers(df),
            'audiometric_validation': self.validate_audiometric_patterns(df)
        }
        
        # Add summary statistics
        pta_columns = [col for col in df.columns if 'kHz' in col]
        if pta_columns:
            report['summary_statistics'] = df[pta_columns].describe()
        
        return report


def clean_nhanes_data(
    df: pd.DataFrame,
    missing_strategy: str = 'listwise',
    round_thresholds: bool = True,
    handle_outliers: str = 'clip',
    validate_patterns: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to clean NHANES data with standard parameters.
    
    Args:
        df: Raw NHANES DataFrame.
        missing_strategy: Strategy for handling missing values.
        round_thresholds: Whether to round thresholds to nearest 5 dB.
        handle_outliers: How to handle outliers ('clip', 'remove', or 'keep').
        validate_patterns: Whether to validate audiometric patterns.
        
    Returns:
        Tuple of (cleaned_dataframe, quality_report).
    """
    cleaner = NHANESDataCleaner()
    
    # Generate initial quality report
    initial_report = cleaner.generate_data_quality_report(df)
    
    # Clean the data
    df_clean = df.copy()
    
    if round_thresholds:
        df_clean = cleaner.clean_hearing_thresholds(
            df_clean,
            round_to_nearest=5,
            handle_outliers=handle_outliers
        )
    
    df_clean = cleaner.handle_missing_values(df_clean, strategy=missing_strategy)
    
    # Generate final quality report
    final_report = cleaner.generate_data_quality_report(df_clean)
    
    quality_report = {
        'initial': initial_report,
        'final': final_report,
        'cleaning_summary': {
            'records_before': len(df),
            'records_after': len(df_clean),
            'records_removed': len(df) - len(df_clean),
            'removal_rate': (len(df) - len(df_clean)) / len(df) * 100
        }
    }
    
    return df_clean, quality_report