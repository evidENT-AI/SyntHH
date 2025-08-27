"""
NHANES Tympanometry Data Loading Module

This module provides functions for loading and processing NHANES tympanometry data,
including parsing raw tympanogram curves and creating pressure-compliance datasets
for visualization and analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class NHANESTympanometryLoader:
    """
    A class for loading and processing NHANES tympanometry data.
    
    This class handles the complex structure of NHANES tympanometry data,
    which contains 84 measurement points per ear across different pressure levels.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the tympanometry data loader.
        
        Args:
            data_dir: Path to the NHANES data directory containing tympanometry files.
        """
        self.data_dir = Path(data_dir)
        
        # NHANES tympanometry specifications
        self.n_measurements = 84
        self.pressure_start = -300  # daPa (dekaPascals) 
        self.pressure_end = 198     # daPa
        self.pressure_increment = 6  # daPa
        
        # Generate pressure values for each measurement point
        self.pressure_values = self._generate_pressure_values()
        
        # Column name patterns
        self.right_ear_columns = [f'AUDTYR{i:02d}' for i in range(1, self.n_measurements + 1)]
        self.left_ear_columns = [f'AUDTYL{i:02d}' for i in range(1, self.n_measurements + 1)]
        
        # Cohort suffixes available
        self.cohort_suffixes = [
            '1999-2000.csv', '2001-02.csv', '2003-04.csv', '2005-06.csv',
            '2007-08.csv', '2009-10.csv', '2011-12.csv', '2015-16.csv',
            '2017-18.csv', '2017-20.csv'
        ]
    
    def _generate_pressure_values(self) -> np.ndarray:
        """
        Generate the pressure values corresponding to each measurement point.
        
        Note: NHANES collects data from positive to negative pressure, but 
        traditionally tympanograms are plotted from negative to positive.
        The data file stores measurements in traditional plotting order.
        
        Returns:
            Array of pressure values in daPa from -300 to +198.
        """
        return np.arange(
            self.pressure_start, 
            self.pressure_end + self.pressure_increment, 
            self.pressure_increment
        )
    
    def load_cohort_data(self, cohort_suffix: str) -> pd.DataFrame:
        """
        Load tympanometry data for a single cohort.
        
        Args:
            cohort_suffix: Filename suffix for the cohort (e.g., '1999-2000.csv').
            
        Returns:
            DataFrame containing raw tympanometry data.
            
        Raises:
            FileNotFoundError: If the tympanometry file is not found.
        """
        filepath = self.data_dir / 'nhanes' / 'tymp' / f'nhanes_auxt_{cohort_suffix}'
        
        try:
            df = pd.read_csv(filepath)
            return df
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Tympanometry file not found: {filepath}. "
                f"Please ensure data directory structure is correct."
            )
    
    def extract_tympanogram_data(
        self, 
        df: pd.DataFrame, 
        seqn: Union[int, float]
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract tympanogram data for a specific participant.
        
        Args:
            df: DataFrame containing tympanometry data.
            seqn: Participant sequence number (SEQN).
            
        Returns:
            Dictionary containing tympanogram data for each ear:
            - 'right': DataFrame with pressure and compliance for right ear
            - 'left': DataFrame with pressure and compliance for left ear
            
        Raises:
            ValueError: If participant SEQN is not found in the dataset.
        """
        # Find participant data
        participant_data = df[df['SEQN'] == seqn]
        
        if participant_data.empty:
            raise ValueError(f"Participant SEQN {seqn} not found in dataset")
        
        participant_data = participant_data.iloc[0]  # Get first (should be only) row
        
        # Extract right ear data
        right_ear_data = pd.DataFrame({
            'Pressure_daPa': self.pressure_values,
            'Compliance_ml': [participant_data[col] for col in self.right_ear_columns],
            'Ear': 'Right'
        })
        
        # Extract left ear data  
        left_ear_data = pd.DataFrame({
            'Pressure_daPa': self.pressure_values,
            'Compliance_ml': [participant_data[col] for col in self.left_ear_columns],
            'Ear': 'Left'
        })
        
        return {
            'right': right_ear_data,
            'left': left_ear_data
        }
    
    def extract_multiple_participants(
        self, 
        df: pd.DataFrame, 
        seqn_list: List[Union[int, float]]
    ) -> pd.DataFrame:
        """
        Extract tympanogram data for multiple participants in long format.
        
        Args:
            df: DataFrame containing tympanometry data.
            seqn_list: List of participant sequence numbers.
            
        Returns:
            Long-format DataFrame with columns: SEQN, Pressure_daPa, Compliance_ml, Ear.
        """
        all_data = []
        
        for seqn in seqn_list:
            try:
                participant_curves = self.extract_tympanogram_data(df, seqn)
                
                # Add SEQN to each ear's data
                for ear_data in participant_curves.values():
                    ear_data['SEQN'] = seqn
                    all_data.append(ear_data)
                    
            except ValueError:
                # Skip participants not found
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df[['SEQN', 'Pressure_daPa', 'Compliance_ml', 'Ear']]
    
    def get_participant_list(self, df: pd.DataFrame) -> List[float]:
        """
        Get list of available participant SEQNs in the dataset.
        
        Args:
            df: DataFrame containing tympanometry data.
            
        Returns:
            List of participant SEQN values.
        """
        return df['SEQN'].dropna().unique().tolist()
    
    def load_all_cohorts_tympanometry(
        self, 
        cohort_suffixes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load and combine tympanometry data from multiple cohorts.
        
        Args:
            cohort_suffixes: Optional list of cohort suffixes to load.
                           If None, loads all available cohorts.
                           
        Returns:
            Combined DataFrame with all cohort tympanometry data.
        """
        if cohort_suffixes is None:
            cohort_suffixes = self.cohort_suffixes
        
        combined_data = []
        
        for suffix in cohort_suffixes:
            try:
                cohort_data = self.load_cohort_data(suffix)
                
                # Add cohort identifier
                cohort = suffix.replace('.csv', '')
                cohort_data['Cohort'] = cohort
                
                combined_data.append(cohort_data)
                
            except FileNotFoundError:
                print(f"Warning: Cohort file not found for {suffix}, skipping...")
                continue
        
        if not combined_data:
            raise FileNotFoundError("No tympanometry data files found")
        
        return pd.concat(combined_data, ignore_index=True)
    
    def calculate_tympanometric_parameters(
        self, 
        pressure_values: np.ndarray, 
        compliance_values: np.ndarray,
        use_curve_fitting: bool = True
    ) -> Dict[str, float]:
        """
        Calculate standard tympanometric parameters from pressure-compliance curve.
        
        Args:
            pressure_values: Array of pressure values in daPa.
            compliance_values: Array of compliance values in ml.
            use_curve_fitting: Whether to use curve fitting for more accurate peak detection.
            
        Returns:
            Dictionary containing calculated parameters:
            - peak_pressure: Pressure at peak compliance (daPa)
            - peak_compliance: Maximum compliance value (ml)  
            - gradient: Tympanometric width at half-peak height (daPa)
            - equivalent_volume: Ear canal volume at +200 daPa (ml) - estimated
            - baseline_compliance: Compliance at high positive pressure (ml)
            - curve_quality: Quality measure of tympanogram shape (0-1)
        """
        # Remove NaN and zero values for cleaner analysis
        valid_mask = ~np.isnan(compliance_values) & (compliance_values >= 0)
        if not np.any(valid_mask) or np.sum(valid_mask) < 5:
            return {
                'peak_pressure': np.nan,
                'peak_compliance': np.nan,
                'gradient': np.nan,
                'equivalent_volume': np.nan,
                'baseline_compliance': np.nan,
                'curve_quality': 0.0
            }
        
        valid_pressure = pressure_values[valid_mask]
        valid_compliance = compliance_values[valid_mask]
        
        # Sort by pressure for consistent analysis
        sort_idx = np.argsort(valid_pressure)
        sorted_pressure = valid_pressure[sort_idx]
        sorted_compliance = valid_compliance[sort_idx]
        
        # Calculate baseline compliance (ear canal volume at high positive pressure)
        # Use average of highest 10% of pressures for stability
        high_pressure_mask = sorted_pressure >= np.percentile(sorted_pressure, 90)
        if np.any(high_pressure_mask):
            baseline_compliance = np.mean(sorted_compliance[high_pressure_mask])
        else:
            baseline_compliance = sorted_compliance[-1] if len(sorted_compliance) > 0 else np.nan
        
        # Find peak using smoothed curve if requested
        if use_curve_fitting and len(sorted_compliance) > 10:
            try:
                # Simple moving average smoothing
                window = min(5, len(sorted_compliance) // 3)
                smoothed_compliance = np.convolve(sorted_compliance, 
                                                 np.ones(window)/window, mode='same')
                peak_idx = np.argmax(smoothed_compliance)
                peak_compliance = smoothed_compliance[peak_idx]
                peak_pressure = sorted_pressure[peak_idx]
            except:
                # Fallback to simple max finding
                peak_idx = np.argmax(sorted_compliance)
                peak_compliance = sorted_compliance[peak_idx]
                peak_pressure = sorted_pressure[peak_idx]
        else:
            # Simple peak detection
            peak_idx = np.argmax(sorted_compliance)
            peak_compliance = sorted_compliance[peak_idx]
            peak_pressure = sorted_pressure[peak_idx]
        
        # Calculate tympanometric gradient (width at half-peak height)
        # This is measured from the baseline-corrected curve
        baseline_corrected_compliance = sorted_compliance - baseline_compliance
        peak_height = peak_compliance - baseline_compliance
        
        if peak_height > 0.05:  # Only calculate if there's a meaningful peak
            half_peak_height = baseline_compliance + (peak_height / 2)
            
            # Find points closest to half-peak height on either side of peak
            left_side = sorted_pressure[:peak_idx+1]
            right_side = sorted_pressure[peak_idx:]
            left_compliance = sorted_compliance[:peak_idx+1]
            right_compliance = sorted_compliance[peak_idx:]
            
            # Find left and right half-peak points
            left_half_peak = np.nan
            right_half_peak = np.nan
            
            if len(left_side) > 1:
                left_diffs = np.abs(left_compliance - half_peak_height)
                left_half_peak = left_side[np.argmin(left_diffs)]
            
            if len(right_side) > 1:
                right_diffs = np.abs(right_compliance - half_peak_height)
                right_half_peak = right_side[np.argmin(right_diffs)]
            
            if not np.isnan(left_half_peak) and not np.isnan(right_half_peak):
                gradient = right_half_peak - left_half_peak
            else:
                gradient = np.nan
        else:
            gradient = np.nan
        
        # Estimate equivalent volume (ear canal volume)
        # This is an approximation - true values should come from main audiometry file
        # Typically ear canal volume ranges from 0.4-1.0 ml in adults
        if not np.isnan(baseline_compliance):
            # Apply typical correction factors based on clinical knowledge
            if baseline_compliance < 0.2:
                equivalent_volume = 0.5  # Assume small ear canal if very low compliance
            elif baseline_compliance > 2.0:
                equivalent_volume = baseline_compliance * 0.4  # Likely overestimate, scale down
            else:
                equivalent_volume = baseline_compliance
        else:
            equivalent_volume = np.nan
        
        # Calculate curve quality measure (0-1 scale)
        curve_quality = self._assess_curve_quality(
            sorted_pressure, sorted_compliance, peak_compliance, baseline_compliance
        )
        
        return {
            'peak_pressure': peak_pressure,
            'peak_compliance': peak_compliance,
            'gradient': gradient,
            'equivalent_volume': equivalent_volume,
            'baseline_compliance': baseline_compliance,
            'curve_quality': curve_quality
        }
    
    def _assess_curve_quality(
        self, 
        pressure: np.ndarray, 
        compliance: np.ndarray, 
        peak_compliance: float,
        baseline_compliance: float
    ) -> float:
        """
        Assess the quality of a tympanogram curve.
        
        Args:
            pressure: Sorted pressure values.
            compliance: Corresponding compliance values.
            peak_compliance: Peak compliance value.
            baseline_compliance: Baseline compliance value.
            
        Returns:
            Quality score between 0 and 1 (1 = excellent quality).
        """
        try:
            # Quality factors
            quality_score = 1.0
            
            # Factor 1: Peak height (good curves have clear peaks)
            peak_height = peak_compliance - baseline_compliance
            if peak_height < 0.1:
                quality_score *= 0.3  # Very poor peak
            elif peak_height < 0.3:
                quality_score *= 0.6  # Poor peak
            elif peak_height > 3.0:
                quality_score *= 0.7  # Suspiciously high peak
            
            # Factor 2: Smoothness (penalize very noisy curves)
            if len(compliance) > 3:
                compliance_diff = np.diff(compliance)
                noise_level = np.std(compliance_diff) / np.mean(np.abs(compliance_diff))
                if noise_level > 2.0:
                    quality_score *= 0.5
                elif noise_level > 1.0:
                    quality_score *= 0.8
            
            # Factor 3: Physiological plausibility
            if peak_compliance > 5.0 or peak_compliance < 0.01:
                quality_score *= 0.4  # Physiologically implausible
            
            # Factor 4: Data completeness
            data_completeness = len(compliance) / 84  # Should have 84 points
            quality_score *= min(1.0, data_completeness)
            
            return max(0.0, min(1.0, quality_score))
            
        except:
            return 0.0
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for the tympanometry dataset.
        
        Args:
            df: DataFrame containing tympanometry data.
            
        Returns:
            Dictionary with summary statistics.
        """
        stats = {
            'total_participants': len(df),
            'participants_with_right_ear': 0,
            'participants_with_left_ear': 0,
            'mean_compliance_range_right': np.nan,
            'mean_compliance_range_left': np.nan
        }
        
        # Check for valid data in each ear
        right_ear_data = df[self.right_ear_columns].values
        left_ear_data = df[self.left_ear_columns].values
        
        # Count participants with valid data (at least some non-zero values)
        stats['participants_with_right_ear'] = np.sum(np.any(right_ear_data > 0, axis=1))
        stats['participants_with_left_ear'] = np.sum(np.any(left_ear_data > 0, axis=1))
        
        # Calculate mean compliance ranges
        right_ranges = np.max(right_ear_data, axis=1) - np.min(right_ear_data, axis=1)
        left_ranges = np.max(left_ear_data, axis=1) - np.min(left_ear_data, axis=1)
        
        stats['mean_compliance_range_right'] = np.mean(right_ranges[right_ranges > 0])
        stats['mean_compliance_range_left'] = np.mean(left_ranges[left_ranges > 0])
        
        return stats


def load_nhanes_tympanometry(
    data_dir: Union[str, Path],
    cohort_suffixes: Optional[List[str]] = None
) -> Tuple[NHANESTympanometryLoader, pd.DataFrame]:
    """
    Convenience function to load NHANES tympanometry data.
    
    Args:
        data_dir: Path to NHANES data directory.
        cohort_suffixes: Optional list of cohort suffixes to load.
        
    Returns:
        Tuple of (loader_instance, combined_dataframe).
    """
    loader = NHANESTympanometryLoader(data_dir)
    df = loader.load_all_cohorts_tympanometry(cohort_suffixes)
    
    return loader, df