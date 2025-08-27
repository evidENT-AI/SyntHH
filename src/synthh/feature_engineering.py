"""
NHANES Feature Engineering Module

This module provides functions for creating derived features from NHANES 
audiometric data, including data format transformations, hearing loss coding,
aggregated measures, and clinical feature extraction.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class NHANESFeatureEngineer:
    """
    A class for engineering features from NHANES audiometric data.
    
    This class provides methods for creating derived features, transforming
    data formats, and extracting clinical audiometric measures.
    """
    
    def __init__(self):
        """Initialize the feature engineer with standard parameters."""
        self.frequencies = [500, 1000, 2000, 4000, 8000]  # Standard audiometric frequencies
        self.frequency_labels = ['0.5kHz', '1kHz', '2kHz', '4kHz', '8kHz']
        
        # Hearing loss thresholds (dB HL)
        self.hearing_loss_threshold = 25
        self.hearing_loss_categories = {
            'normal': (-10, 25),
            'mild': (26, 40), 
            'moderate': (41, 70),
            'severe': (71, 95),
            'profound': (96, 120)
        }
    
    def wide_to_long_format(
        self,
        df: pd.DataFrame,
        include_demographics: bool = False,
        id_column: str = 'SEQN'
    ) -> pd.DataFrame:
        """
        Convert wide-format PTA data to long format with ear and frequency coding.
        
        Args:
            df: Wide-format DataFrame with PTA data.
            include_demographics: Whether to include demographic variables.
            id_column: Name of the ID column.
            
        Returns:
            Long-format DataFrame with columns: ID, Hearing Threshold, Frequency, Ear, [Demographics].
        """
        # Get PTA columns
        pta_columns = [col for col in df.columns if 'kHz' in col and ('Right' in col or 'Left' in col)]
        
        long_df_list = []
        
        for col in pta_columns:
            # Extract frequency and ear information
            parts = col.split(' ')
            freq_str = parts[0]
            ear = parts[1]
            
            # Convert frequency string to Hz
            if freq_str == '0.5kHz':
                frequency = 500
            elif freq_str.endswith('kHz'):
                frequency = int(float(freq_str.replace('kHz', '')) * 1000)
            else:
                continue
                
            # Create long-format subset
            subset_df = df[id_column].to_frame()
            subset_df['Hearing Threshold (dB HL)'] = df[col]
            subset_df['Frequency (Hz)'] = frequency
            subset_df['Ear'] = ear
            
            # Add demographics if requested
            if include_demographics:
                demo_cols = ['Gender', 'Age (years)', 'Race/ethnicity']
                for demo_col in demo_cols:
                    if demo_col in df.columns:
                        subset_df[demo_col] = df[demo_col]
            
            long_df_list.append(subset_df)
        
        # Combine all subsets
        long_df = pd.concat(long_df_list, axis=0, ignore_index=True)
        
        # Remove 3kHz and 6kHz if present (not standard)
        long_df = long_df[~long_df['Frequency (Hz)'].isin([3000, 6000])]
        
        # Replace infinite values with NaN
        long_df = long_df.replace([np.inf, -np.inf], np.nan)
        
        return long_df.reset_index(drop=True)
    
    def create_hearing_loss_coding(
        self,
        df: pd.DataFrame,
        threshold: float = None,
        method: str = 'any_frequency'
    ) -> Tuple[pd.DataFrame, int]:
        """
        Create hearing loss coding based on PTA thresholds.
        
        Args:
            df: DataFrame with PTA data (wide format).
            threshold: Hearing loss threshold in dB HL (default: 25).
            method: Method for determining hearing loss:
                   'any_frequency' - hearing loss if any frequency > threshold
                   'pta_average' - hearing loss based on PTA average
                   'high_frequency' - hearing loss based on high frequencies (4-8kHz)
                   
        Returns:
            Tuple of (dataframe_with_coding, count_with_hearing_loss).
        """
        if threshold is None:
            threshold = self.hearing_loss_threshold
            
        df_coded = df.copy()
        pta_columns = [col for col in df.columns if 'kHz' in col and ('Right' in col or 'Left' in col)]
        
        if method == 'any_frequency':
            # Hearing loss if any frequency > threshold
            hearing_loss_mask = (df[pta_columns] > threshold).any(axis=1)
            
        elif method == 'pta_average':
            # Calculate PTA average (0.5, 1, 2 kHz) for each ear
            pta_freqs = ['0.5kHz', '1kHz', '2kHz']
            
            right_pta = df[[f'{freq} Right' for freq in pta_freqs if f'{freq} Right' in df.columns]].mean(axis=1)
            left_pta = df[[f'{freq} Left' for freq in pta_freqs if f'{freq} Left' in df.columns]].mean(axis=1)
            
            hearing_loss_mask = (right_pta > threshold) | (left_pta > threshold)
            
        elif method == 'high_frequency':
            # High-frequency hearing loss (4kHz, 8kHz)
            hf_columns = [col for col in pta_columns if '4kHz' in col or '8kHz' in col]
            hearing_loss_mask = (df[hf_columns] > threshold).any(axis=1)
        
        df_coded['Hearing Loss'] = hearing_loss_mask.astype(int)
        hearing_loss_count = hearing_loss_mask.sum()
        
        return df_coded, hearing_loss_count
    
    def create_hearing_loss_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical hearing loss severity coding.
        
        Args:
            df: DataFrame with PTA data.
            
        Returns:
            DataFrame with additional severity category columns.
        """
        df_categorized = df.copy()
        pta_columns = [col for col in df.columns if 'kHz' in col and ('Right' in col or 'Left' in col)]
        
        # Create severity categories for each ear
        for ear in ['Right', 'Left']:
            ear_columns = [col for col in pta_columns if ear in col]
            
            if ear_columns:
                # Calculate PTA average for severity classification
                pta_freqs = [f'0.5kHz {ear}', f'1kHz {ear}', f'2kHz {ear}']
                available_pta_freqs = [col for col in pta_freqs if col in df.columns]
                
                if available_pta_freqs:
                    ear_pta = df[available_pta_freqs].mean(axis=1)
                    
                    # Assign categories
                    conditions = [
                        ear_pta <= self.hearing_loss_categories['normal'][1],
                        (ear_pta > self.hearing_loss_categories['mild'][0]) & 
                        (ear_pta <= self.hearing_loss_categories['mild'][1]),
                        (ear_pta > self.hearing_loss_categories['moderate'][0]) & 
                        (ear_pta <= self.hearing_loss_categories['moderate'][1]),
                        (ear_pta > self.hearing_loss_categories['severe'][0]) & 
                        (ear_pta <= self.hearing_loss_categories['severe'][1]),
                        ear_pta > self.hearing_loss_categories['profound'][0]
                    ]
                    
                    choices = ['Normal', 'Mild', 'Moderate', 'Severe', 'Profound']
                    
                    df_categorized[f'Hearing Loss Severity {ear}'] = np.select(
                        conditions, choices, default='Unknown'
                    )
        
        return df_categorized
    
    def create_aggregated_frequencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated frequency measures (average of left and right ears).
        
        Args:
            df: DataFrame with bilateral PTA data.
            
        Returns:
            DataFrame with additional aggregated frequency columns.
        """
        df_agg = df.copy()
        
        for freq in self.frequency_labels:
            right_col = f'{freq} Right'
            left_col = f'{freq} Left'
            
            if right_col in df.columns and left_col in df.columns:
                # Calculate mean of both ears
                df_agg[freq] = (df[right_col] + df[left_col]) / 2
                
                # Calculate absolute difference (asymmetry measure)
                df_agg[f'{freq} Asymmetry'] = abs(df[right_col] - df[left_col])
        
        return df_agg
    
    def create_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create clinical audiometric features.
        
        Args:
            df: DataFrame with PTA data.
            
        Returns:
            DataFrame with additional clinical features.
        """
        df_clinical = df.copy()
        
        for ear in ['Right', 'Left']:
            ear_columns = [f'{freq} {ear}' for freq in self.frequency_labels 
                          if f'{freq} {ear}' in df.columns]
            
            if len(ear_columns) >= 3:
                # Pure Tone Average (0.5, 1, 2 kHz)
                pta_columns = [f'{freq} {ear}' for freq in ['0.5kHz', '1kHz', '2kHz'] 
                              if f'{freq} {ear}' in df.columns]
                if len(pta_columns) == 3:
                    df_clinical[f'PTA {ear}'] = df[pta_columns].mean(axis=1)
                
                # High Frequency Average (4, 8 kHz)
                hfa_columns = [f'{freq} {ear}' for freq in ['4kHz', '8kHz'] 
                              if f'{freq} {ear}' in df.columns]
                if len(hfa_columns) == 2:
                    df_clinical[f'HFA {ear}'] = df[hfa_columns].mean(axis=1)
                
                # Speech Frequency Average (0.5, 1, 2, 4 kHz)
                sfa_columns = [f'{freq} {ear}' for freq in ['0.5kHz', '1kHz', '2kHz', '4kHz'] 
                              if f'{freq} {ear}' in df.columns]
                if len(sfa_columns) == 4:
                    df_clinical[f'SFA {ear}'] = df[sfa_columns].mean(axis=1)
                
                # Calculate audiometric slope (change from low to high frequencies)
                if f'0.5kHz {ear}' in df.columns and f'8kHz {ear}' in df.columns:
                    df_clinical[f'Slope {ear}'] = df[f'8kHz {ear}'] - df[f'0.5kHz {ear}']
                
                # Identify configuration patterns
                thresholds = df[ear_columns].values
                df_clinical[f'Config {ear}'] = df[ear_columns].apply(
                    self._classify_audiometric_configuration, axis=1
                )
        
        return df_clinical
    
    def _classify_audiometric_configuration(self, thresholds: pd.Series) -> str:
        """
        Classify audiometric configuration based on threshold pattern.
        
        Args:
            thresholds: Series of hearing thresholds across frequencies.
            
        Returns:
            String describing the audiometric configuration.
        """
        if thresholds.isna().all():
            return 'Unknown'
        
        values = thresholds.dropna().values
        
        if len(values) < 3:
            return 'Insufficient Data'
        
        # Calculate differences between adjacent frequencies
        diffs = np.diff(values)
        
        # Classification rules (simplified)
        if all(abs(d) <= 10 for d in diffs):
            return 'Flat'
        elif all(d >= 5 for d in diffs):
            return 'Rising'
        elif all(d <= -5 for d in diffs):
            return 'Sloping'
        elif len(diffs) >= 2 and diffs[0] <= -10 and diffs[-1] >= 10:
            return 'U-shaped'
        elif len(diffs) >= 2 and diffs[0] >= 10 and diffs[-1] <= -10:
            return 'Inverted-U'
        else:
            return 'Irregular'
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional demographic features.
        
        Args:
            df: DataFrame with demographic data.
            
        Returns:
            DataFrame with additional demographic features.
        """
        df_demo = df.copy()
        
        if 'Age (years)' in df.columns:
            # Age groups
            age_bins = [0, 18, 30, 50, 65, 100]
            age_labels = ['Child', 'Young Adult', 'Middle Age', 'Older Adult', 'Elderly']
            df_demo['Age Group'] = pd.cut(
                df['Age (years)'], bins=age_bins, labels=age_labels, right=False
            )
            
            # Age-related hearing loss risk
            df_demo['ARHL Risk'] = df['Age (years)'].apply(
                lambda x: 'Low' if x < 50 else 'Moderate' if x < 65 else 'High'
            )
        
        if 'Gender' in df.columns:
            # Binary gender coding for modeling
            df_demo['Gender_Numeric'] = df['Gender'].map({'Female': 0, 'Male': 1})
        
        if 'Race/ethnicity' in df.columns:
            # Create binary indicators for each ethnicity
            ethnicities = df['Race/ethnicity'].unique()
            for ethnicity in ethnicities:
                if pd.notna(ethnicity):
                    col_name = f'Ethnicity_{ethnicity.replace(" ", "_").replace("-", "_")}'
                    df_demo[col_name] = (df['Race/ethnicity'] == ethnicity).astype(int)
        
        return df_demo
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from cohort information.
        
        Args:
            df: DataFrame with Cohort column.
            
        Returns:
            DataFrame with additional time features.
        """
        df_time = df.copy()
        
        if 'Cohort' in df.columns:
            # Extract year information
            df_time['Cohort_Start_Year'] = df['Cohort'].str.extract(r'(\d{4})').astype(float)
            
            # Create decade groups
            df_time['Decade'] = ((df_time['Cohort_Start_Year'] // 10) * 10).astype('Int64')
            
            # Create early vs late NHANES periods
            df_time['Period'] = df_time['Cohort_Start_Year'].apply(
                lambda x: 'Early (1999-2009)' if x < 2010 else 'Later (2010+)'
            )
        
        return df_time


def engineer_nhanes_features(
    df: pd.DataFrame,
    include_hearing_loss: bool = True,
    include_clinical: bool = True,
    include_aggregated: bool = True,
    include_demographics: bool = True,
    hearing_loss_method: str = 'any_frequency'
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to engineer comprehensive features from NHANES data.
    
    Args:
        df: Raw NHANES DataFrame with wide-format PTA data.
        include_hearing_loss: Whether to create hearing loss coding.
        include_clinical: Whether to create clinical audiometric features.
        include_aggregated: Whether to create aggregated frequency measures.
        include_demographics: Whether to create demographic features.
        hearing_loss_method: Method for hearing loss classification.
        
    Returns:
        Dictionary containing multiple feature-engineered datasets:
        - 'wide': Wide-format data with all features
        - 'long': Long-format data for visualization
        - 'modeling': Data prepared for machine learning
    """
    engineer = NHANESFeatureEngineer()
    
    # Start with input dataframe
    df_wide = df.copy()
    
    # Create hearing loss coding
    if include_hearing_loss:
        df_wide, hl_count = engineer.create_hearing_loss_coding(
            df_wide, method=hearing_loss_method
        )
        df_wide = engineer.create_hearing_loss_categories(df_wide)
    
    # Create clinical features
    if include_clinical:
        df_wide = engineer.create_clinical_features(df_wide)
    
    # Create aggregated frequencies
    if include_aggregated:
        df_wide = engineer.create_aggregated_frequencies(df_wide)
    
    # Create demographic features
    if include_demographics:
        df_wide = engineer.create_demographic_features(df_wide)
        df_wide = engineer.create_time_features(df_wide)
    
    # Create long-format dataset
    df_long = engineer.wide_to_long_format(
        df, include_demographics=include_demographics
    )
    
    # Create modeling dataset (numeric only, handle missing values)
    df_modeling = df_wide.select_dtypes(include=[np.number]).copy()
    
    return {
        'wide': df_wide,
        'long': df_long,
        'modeling': df_modeling
    }