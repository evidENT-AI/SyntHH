"""
NHANES Preprocessing Pipeline Module

This module provides a unified preprocessing pipeline that orchestrates
data loading, cleaning, validation, and feature engineering for NHANES
audiometric data. It serves as the main interface for preparing data
for synthetic data generation models.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .data_loader import NHANESDataLoader, load_nhanes_data
from .data_cleaner import NHANESDataCleaner, clean_nhanes_data  
from .feature_engineering import NHANESFeatureEngineer, engineer_nhanes_features


class NHANESPreprocessingPipeline:
    """
    Unified preprocessing pipeline for NHANES audiometric data.
    
    This class orchestrates the complete preprocessing workflow from raw
    NHANES data files to analysis-ready datasets for synthetic data generation.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        log_level: str = 'INFO'
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            data_dir: Path to NHANES data directory.
            output_dir: Path to output directory for processed data.
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'processed'
        
        # Initialize components
        self.loader = NHANESDataLoader(data_dir)
        self.cleaner = NHANESDataCleaner()
        self.engineer = NHANESFeatureEngineer()
        
        # Setup logging
        self._setup_logging(log_level)
        
        # Pipeline state
        self.raw_data = {}
        self.cleaned_data = {}
        self.engineered_data = {}
        self.quality_reports = {}
        
        # Default pipeline configuration
        self.config = {
            'cohorts': None,  # All cohorts by default
            'missing_strategy': 'listwise',
            'round_thresholds': True,
            'handle_outliers': 'clip',
            'hearing_loss_method': 'any_frequency',
            'include_clinical_features': True,
            'include_demographic_features': True,
            'validate_patterns': True,
            'export_formats': ['csv', 'parquet']
        }
    
    def _setup_logging(self, level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def configure(self, **kwargs) -> 'NHANESPreprocessingPipeline':
        """
        Configure pipeline parameters.
        
        Args:
            **kwargs: Configuration parameters to update.
            
        Returns:
            Self for method chaining.
        """
        self.config.update(kwargs)
        self.logger.info(f"Pipeline configured with: {kwargs}")
        return self
    
    def load_data(self, cohorts: Optional[List[str]] = None) -> 'NHANESPreprocessingPipeline':
        """
        Load raw NHANES data.
        
        Args:
            cohorts: Optional list of cohort suffixes to load.
            
        Returns:
            Self for method chaining.
        """
        self.logger.info("Loading NHANES data...")
        
        cohorts = cohorts or self.config['cohorts']
        
        try:
            # Load combined dataset
            combined_df = self.loader.load_all_cohorts(cohorts)
            combined_df = self.loader.create_clean_labels(combined_df)
            
            # Create different views of the data
            self.raw_data = {
                'combined': combined_df,
                'pta': self.loader.get_pta_subset(combined_df, relabel=True),
                'demo_pta': combined_df[
                    ['SEQN'] + self.loader.get_standard_pta_columns() + 
                    ['Gender', 'Age (years)', 'Race/ethnicity']
                ].copy()
            }
            
            self.logger.info(f"Loaded {len(combined_df)} records from NHANES data")
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
            
        return self
    
    def clean_data(self) -> 'NHANESPreprocessingPipeline':
        """
        Clean and validate the loaded data.
        
        Returns:
            Self for method chaining.
        """
        self.logger.info("Cleaning NHANES data...")
        
        if not self.raw_data:
            raise RuntimeError("No data loaded. Call load_data() first.")
        
        try:
            for dataset_name, df in self.raw_data.items():
                self.logger.info(f"Cleaning {dataset_name} dataset...")
                
                cleaned_df, quality_report = clean_nhanes_data(
                    df,
                    missing_strategy=self.config['missing_strategy'],
                    round_thresholds=self.config['round_thresholds'],
                    handle_outliers=self.config['handle_outliers'],
                    validate_patterns=self.config['validate_patterns']
                )
                
                self.cleaned_data[dataset_name] = cleaned_df
                self.quality_reports[dataset_name] = quality_report
                
                # Log cleaning results
                summary = quality_report['cleaning_summary']
                self.logger.info(
                    f"{dataset_name}: {summary['records_before']} -> "
                    f"{summary['records_after']} records "
                    f"({summary['removal_rate']:.1f}% removed)"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to clean data: {e}")
            raise
            
        return self
    
    def engineer_features(self) -> 'NHANESPreprocessingPipeline':
        """
        Engineer features from cleaned data.
        
        Returns:
            Self for method chaining.
        """
        self.logger.info("Engineering features...")
        
        if not self.cleaned_data:
            raise RuntimeError("No cleaned data available. Call clean_data() first.")
        
        try:
            # Use the demo_pta dataset as primary for feature engineering
            primary_df = self.cleaned_data['demo_pta']
            
            engineered_datasets = engineer_nhanes_features(
                primary_df,
                include_hearing_loss=True,
                include_clinical=self.config['include_clinical_features'],
                include_aggregated=True,
                include_demographics=self.config['include_demographic_features'],
                hearing_loss_method=self.config['hearing_loss_method']
            )
            
            self.engineered_data = engineered_datasets
            
            # Log feature engineering results
            wide_df = engineered_datasets['wide']
            self.logger.info(f"Created {wide_df.shape[1]} features for {len(wide_df)} records")
            
            # Log hearing loss statistics if available
            if 'Hearing Loss' in wide_df.columns:
                hl_count = wide_df['Hearing Loss'].sum()
                hl_pct = (hl_count / len(wide_df)) * 100
                self.logger.info(f"Hearing loss prevalence: {hl_count}/{len(wide_df)} ({hl_pct:.1f}%)")
                
        except Exception as e:
            self.logger.error(f"Failed to engineer features: {e}")
            raise
            
        return self
    
    def export_data(
        self,
        datasets: Optional[List[str]] = None,
        formats: Optional[List[str]] = None
    ) -> 'NHANESPreprocessingPipeline':
        """
        Export processed datasets to files.
        
        Args:
            datasets: List of dataset names to export. If None, exports all.
            formats: List of file formats ('csv', 'parquet'). If None, uses config.
            
        Returns:
            Self for method chaining.
        """
        self.logger.info("Exporting processed datasets...")
        
        if not self.engineered_data:
            raise RuntimeError("No engineered data available. Call engineer_features() first.")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        datasets = datasets or list(self.engineered_data.keys())
        formats = formats or self.config['export_formats']
        
        try:
            for dataset_name in datasets:
                if dataset_name not in self.engineered_data:
                    self.logger.warning(f"Dataset '{dataset_name}' not found, skipping...")
                    continue
                
                df = self.engineered_data[dataset_name]
                
                for fmt in formats:
                    filename = f"nhanes_{dataset_name}_{len(df)}records.{fmt}"
                    filepath = self.output_dir / filename
                    
                    if fmt == 'csv':
                        df.to_csv(filepath, index=False)
                    elif fmt == 'parquet':
                        df.to_parquet(filepath, index=False)
                    else:
                        self.logger.warning(f"Unsupported format '{fmt}', skipping...")
                        continue
                    
                    self.logger.info(f"Exported {dataset_name} to {filepath}")
                    
        except Exception as e:
            self.logger.error(f"Failed to export data: {e}")
            raise
            
        return self
    
    def export_quality_report(self, filename: str = 'quality_report.json') -> 'NHANESPreprocessingPipeline':
        """
        Export data quality reports to file.
        
        Args:
            filename: Name of the quality report file.
            
        Returns:
            Self for method chaining.
        """
        import json
        
        if not self.quality_reports:
            self.logger.warning("No quality reports available")
            return self
        
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            serializable_reports = convert_numpy_types(self.quality_reports)
            
            filepath = self.output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(serializable_reports, f, indent=2)
            
            self.logger.info(f"Exported quality report to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export quality report: {e}")
            raise
            
        return self
    
    def get_modeling_data(self, dataset: str = 'modeling') -> pd.DataFrame:
        """
        Get data prepared for machine learning modeling.
        
        Args:
            dataset: Name of the dataset to retrieve.
            
        Returns:
            DataFrame ready for modeling.
        """
        if dataset not in self.engineered_data:
            raise ValueError(f"Dataset '{dataset}' not available")
        
        df = self.engineered_data[dataset].copy()
        
        # Additional modeling preparations
        # Remove non-numeric columns that aren't useful for modeling
        exclude_columns = ['SEQN', 'Gender', 'Race/ethnicity', 'Cohort']
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        
        # Remove completely NaN columns
        numeric_df = numeric_df.dropna(axis=1, how='all')
        
        self.logger.info(f"Modeling dataset prepared with {numeric_df.shape[1]} features")
        
        return numeric_df
    
    def run_full_pipeline(
        self,
        cohorts: Optional[List[str]] = None,
        export: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            cohorts: Optional list of cohort suffixes to process.
            export: Whether to export results to files.
            
        Returns:
            Dictionary containing all processed datasets.
        """
        self.logger.info("Starting full preprocessing pipeline...")
        
        # Run pipeline steps
        self.load_data(cohorts)
        self.clean_data()
        self.engineer_features()
        
        if export:
            self.export_data()
            self.export_quality_report()
        
        self.logger.info("Preprocessing pipeline completed successfully!")
        
        return self.engineered_data
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for the processed datasets.
        
        Returns:
            Dictionary containing summary statistics.
        """
        summary = {
            'raw_data_records': len(self.raw_data.get('combined', [])),
            'cleaned_data_records': len(self.cleaned_data.get('demo_pta', [])),
            'final_features': 0,
            'hearing_loss_prevalence': None
        }
        
        if self.engineered_data and 'wide' in self.engineered_data:
            wide_df = self.engineered_data['wide']
            summary['final_features'] = wide_df.shape[1]
            
            if 'Hearing Loss' in wide_df.columns:
                hl_count = wide_df['Hearing Loss'].sum()
                summary['hearing_loss_prevalence'] = (hl_count / len(wide_df)) * 100
        
        return summary


def preprocess_nhanes_data(
    data_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    cohorts: Optional[List[str]] = None,
    missing_strategy: str = 'listwise',
    export: bool = True,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to run NHANES preprocessing with default settings.
    
    Args:
        data_dir: Path to NHANES data directory.
        output_dir: Path to output directory.
        cohorts: Optional list of cohort suffixes.
        missing_strategy: Strategy for handling missing values.
        export: Whether to export processed data.
        **kwargs: Additional configuration parameters.
        
    Returns:
        Dictionary containing processed datasets.
    """
    pipeline = NHANESPreprocessingPipeline(data_dir, output_dir)
    
    # Configure pipeline
    config_updates = {
        'missing_strategy': missing_strategy,
        **kwargs
    }
    pipeline.configure(**config_updates)
    
    # Run pipeline
    return pipeline.run_full_pipeline(cohorts, export)