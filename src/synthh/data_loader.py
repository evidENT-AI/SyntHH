"""
NHANES Data Loading Module

This module provides functions for loading and filtering NHANES audiometric data
from multiple cohorts (1999-2020) including demographic, pure tone audiometry (PTA),
acoustic reflex, and tympanometry data.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class NHANESDataLoader:
    """
    A class for loading and filtering NHANES audiometric data.
    
    This class provides methods to load data from multiple NHANES cohorts,
    filter relevant columns, and combine datasets for analysis.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the NHANES data loader.
        
        Args:
            data_dir: Path to the NHANES data directory containing subdirectories
                     for demo, pta, reflex, and tymp data.
        """
        self.data_dir = Path(data_dir)
        self.cohort_suffixes = [
            '1999-2000.csv', '2001-02.csv', '2003-04.csv', '2005-06.csv',
            '2007-08.csv', '2009-10.csv', '2011-12.csv', '2015-16.csv',
            '2017-18.csv', '2017-20.csv'
        ]
        
        # Define column mappings
        self.pta_columns = [
            'SEQN', 'AUXU1K1R', 'AUXU500R', 'AUXU1K2R', 'AUXU2KR', 'AUXU3KR',
            'AUXU4KR', 'AUXU6KR', 'AUXU8KR', 'AUXU1K1L', 'AUXU500L', 'AUXU1K2L',
            'AUXU2KL', 'AUXU3KL', 'AUXU4KL', 'AUXU6KL', 'AUXU8KL'
        ]
        
        self.demo_columns = ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDAGEMN', 'RIDRETH1']
        
        self.error_codes = [888, 666]  # NHANES error codes to replace with NaN
    
    def load_cohort_data(self, cohort_suffix: str) -> Tuple[pd.DataFrame, ...]:
        """
        Load NHANES data for a single cohort.
        
        Args:
            cohort_suffix: Filename suffix for the cohort (e.g., '1999-2000.csv').
            
        Returns:
            Tuple containing (demo_df, pta_df, auxr_df, auxt_df) DataFrames.
            
        Raises:
            FileNotFoundError: If any required data files are not found.
        """
        try:
            demo_df = pd.read_csv(self.data_dir / 'demo' / f'nhanes_demo_{cohort_suffix}')
            pta_df = pd.read_csv(self.data_dir / 'pta' / f'nhanes_aux_{cohort_suffix}')
            auxr_df = pd.read_csv(self.data_dir / 'reflex' / f'nhanes_auxr_{cohort_suffix}')
            auxt_df = pd.read_csv(self.data_dir / 'tymp' / f'nhanes_auxt_{cohort_suffix}')
            
            return demo_df, pta_df, auxr_df, auxt_df
            
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not find data for cohort {cohort_suffix}. "
                f"Please check that data directory contains required files."
            ) from e
    
    def filter_pta_data(self, pta_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter and clean pure tone audiometry data.
        
        Args:
            pta_df: Raw PTA DataFrame from NHANES.
            
        Returns:
            Filtered DataFrame with selected columns and cleaned error codes.
        """
        # Select relevant columns
        filtered_df = pta_df[self.pta_columns].copy()
        
        # Replace error codes with NaN
        for error_code in self.error_codes:
            filtered_df = filtered_df.replace(error_code, np.nan)
            
        return filtered_df
    
    def filter_demo_data(self, demo_df: pd.DataFrame, pta_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter demographic data to match patients in PTA data.
        
        Args:
            demo_df: Raw demographic DataFrame from NHANES.
            pta_df: Filtered PTA DataFrame to match patients against.
            
        Returns:
            Filtered demographic DataFrame.
        """
        # Select relevant columns
        filtered_df = demo_df[self.demo_columns].copy()
        
        # Match patients in demographic data to patients in PTA data
        filtered_df = filtered_df[filtered_df['SEQN'].isin(pta_df['SEQN'])]
        filtered_df = filtered_df.reset_index(drop=True)
        
        return filtered_df
    
    def load_and_filter_cohort(self, cohort_suffix: str) -> pd.DataFrame:
        """
        Load and filter data for a single cohort, combining demo and PTA data.
        
        Args:
            cohort_suffix: Filename suffix for the cohort (e.g., '1999-2000.csv').
            
        Returns:
            DataFrame containing combined and processed cohort data.
        """
        # Load raw data
        demo_df, pta_df, auxr_df, auxt_df = self.load_cohort_data(cohort_suffix)
        
        # Filter data
        filtered_pta_df = self.filter_pta_data(pta_df)
        filtered_demo_df = self.filter_demo_data(demo_df, filtered_pta_df)
        
        # Combine filtered data
        cohort_df = pd.concat([
            filtered_demo_df,
            filtered_pta_df.iloc[:, 1:]  # Exclude SEQN column to avoid duplication
        ], axis=1)
        
        # Add cohort identifier
        cohort = cohort_suffix.replace('.csv', '')
        cohort_df['Cohort'] = cohort
        cohort_df = cohort_df.reset_index(drop=True)
        
        return cohort_df
    
    def load_all_cohorts(self, cohort_suffixes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load and combine data from all NHANES cohorts.
        
        Args:
            cohort_suffixes: Optional list of cohort suffixes to load. 
                           If None, loads all available cohorts.
                           
        Returns:
            Combined DataFrame containing all cohort data.
        """
        if cohort_suffixes is None:
            cohort_suffixes = self.cohort_suffixes
        
        combined_df = pd.DataFrame()
        
        for cohort_suffix in cohort_suffixes:
            cohort_data = self.load_and_filter_cohort(cohort_suffix)
            combined_df = pd.concat([combined_df, cohort_data], axis=0, ignore_index=True)
        
        return combined_df.reset_index(drop=True)
    
    def create_clean_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create clean, human-readable labels for demographic variables.
        
        Args:
            df: DataFrame with raw NHANES demographic codes.
            
        Returns:
            DataFrame with additional clean label columns.
        """
        df_clean = df.copy()
        
        # Create clean age column
        df_clean['Age (years)'] = df_clean['RIDAGEYR']
        
        # Create clean gender labels
        df_clean['Gender'] = df_clean['RIAGENDR'].replace({
            1.0: 'Male',
            2.0: 'Female'
        })
        
        # Create clean race/ethnicity labels
        df_clean['Race/ethnicity'] = df_clean['RIDRETH1'].replace({
            1.0: 'Mexican American',
            2.0: 'Other Hispanic',
            3.0: 'Non-Hispanic White',
            4.0: 'Non-Hispanic Black',
            5.0: 'Other Race - Including Multi-Racial'
        })
        
        return df_clean
    
    @staticmethod
    def get_standard_pta_columns() -> List[str]:
        """
        Get standardized PTA column names for consistent ordering.
        
        Returns:
            List of standard PTA column names in frequency order.
        """
        return [
            '0.5kHz Right', '0.5kHz Left',
            '1kHz Right', '1kHz Left',
            '2kHz Right', '2kHz Left',
            '4kHz Right', '4kHz Left',
            '8kHz Right', '8kHz Left'
        ]
    
    def get_pta_subset(self, df: pd.DataFrame, relabel: bool = True) -> pd.DataFrame:
        """
        Extract and optionally relabel PTA columns from combined dataset.
        
        Args:
            df: Combined NHANES DataFrame.
            relabel: Whether to create human-readable column labels.
            
        Returns:
            DataFrame with PTA data only.
        """
        # Select PTA columns (excluding retest and 3k/6k Hz)
        pta_columns_subset = [
            'AUXU1K1R', 'AUXU500R', 'AUXU2KR', 'AUXU4KR', 'AUXU8KR',
            'AUXU1K1L', 'AUXU500L', 'AUXU2KL', 'AUXU4KL', 'AUXU8KL'
        ]
        
        pta_df = df[pta_columns_subset].copy()
        
        if relabel:
            # Create human-readable column names
            label_mapping = {
                'AUXU1K1R': '1kHz Right', 'AUXU500R': '0.5kHz Right',
                'AUXU2KR': '2kHz Right', 'AUXU4KR': '4kHz Right', 'AUXU8KR': '8kHz Right',
                'AUXU1K1L': '1kHz Left', 'AUXU500L': '0.5kHz Left',
                'AUXU2KL': '2kHz Left', 'AUXU4KL': '4kHz Left', 'AUXU8KL': '8kHz Left'
            }
            pta_df = pta_df.rename(columns=label_mapping)
            
            # Reorder columns by frequency
            standard_columns = self.get_standard_pta_columns()
            pta_df = pta_df[standard_columns]
        
        return pta_df


def load_nhanes_data(
    data_dir: Union[str, Path],
    cohort_suffixes: Optional[List[str]] = None,
    include_clean_labels: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load and process NHANES data.
    
    Args:
        data_dir: Path to NHANES data directory.
        cohort_suffixes: Optional list of cohort suffixes to load.
        include_clean_labels: Whether to include clean demographic labels.
        
    Returns:
        Dictionary containing processed DataFrames:
        - 'combined': Full combined dataset
        - 'pta': PTA data only  
        - 'demo_pta': Demographics + PTA data
    """
    loader = NHANESDataLoader(data_dir)
    
    # Load all cohorts
    combined_df = loader.load_all_cohorts(cohort_suffixes)
    
    if include_clean_labels:
        combined_df = loader.create_clean_labels(combined_df)
    
    # Extract PTA subset
    pta_df = loader.get_pta_subset(combined_df, relabel=True)
    
    # Create demographics + PTA dataset
    demo_pta_columns = ['SEQN'] + loader.get_standard_pta_columns()
    if include_clean_labels:
        demo_pta_columns.extend(['Gender', 'Age (years)', 'Race/ethnicity'])
    
    demo_pta_df = combined_df[demo_pta_columns].copy()
    
    return {
        'combined': combined_df,
        'pta': pta_df,
        'demo_pta': demo_pta_df
    }