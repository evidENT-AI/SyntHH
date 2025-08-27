"""
NHANES Acoustic Reflex Data Loading Module

This module provides functions for loading and processing NHANES acoustic reflex data,
including parsing raw reflex response curves and extracting reflex parameters
for clinical interpretation and analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class NHANESAcousticReflexLoader:
    """
    A class for loading and processing NHANES acoustic reflex data.
    
    The acoustic reflex is a protective reflex contraction of the middle ear muscles
    in response to loud sounds. This data contains compliance measurements over
    1.5 seconds following acoustic stimulation at 1000 Hz and 2000 Hz.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the acoustic reflex data loader.
        
        Args:
            data_dir: Path to the NHANES data directory containing reflex files.
        """
        self.data_dir = Path(data_dir)
        
        # NHANES acoustic reflex specifications
        self.n_measurements = 84  # 84 time points over 1.5 seconds
        self.total_duration_ms = 1500  # Total measurement duration in milliseconds
        self.time_interval_ms = self.total_duration_ms / self.n_measurements  # ~17.9 ms per point
        
        # Generate time values for each measurement point
        self.time_values = self._generate_time_values()
        
        # Test frequencies available
        self.frequencies = [1000, 2000]  # Hz
        
        # Column name patterns for different conditions
        # Format: AUX[L/R]R[frequency][measurement_number]
        self.column_patterns = {
            1000: {
                'right': [f'AUXRR1{i:02d}' for i in range(1, self.n_measurements + 1)],
                'left': [f'AUXLR1{i:02d}' for i in range(1, self.n_measurements + 1)]
            },
            2000: {
                'right': [f'AUXRR2{i:02d}' for i in range(1, self.n_measurements + 1)],
                'left': [f'AUXLR2{i:02d}' for i in range(1, self.n_measurements + 1)]
            }
        }
        
        # Cohort suffixes available
        self.cohort_suffixes = [
            '1999-2000.csv', '2001-02.csv', '2003-04.csv', '2005-06.csv',
            '2007-08.csv', '2009-10.csv', '2011-12.csv', '2015-16.csv',
            '2017-18.csv', '2017-20.csv'
        ]
        
        # Clinical reference values
        self.reflex_thresholds = {
            'normal_threshold_db': 85,     # dB - typical reflex threshold
            'abnormal_threshold_db': 100,  # dB - above this suggests pathology
            'decay_threshold_percent': 50, # % - significant decay if >50% in 10 seconds
            'min_reflex_magnitude': 0.1    # ml - minimum detectable reflex
        }
    
    def _generate_time_values(self) -> np.ndarray:
        """
        Generate the time values corresponding to each measurement point.
        
        Returns:
            Array of time values in milliseconds from 0 to 1500 ms.
        """
        return np.linspace(0, self.total_duration_ms, self.n_measurements)
    
    def load_cohort_data(self, cohort_suffix: str) -> pd.DataFrame:
        """
        Load acoustic reflex data for a single cohort.
        
        Args:
            cohort_suffix: Filename suffix for the cohort (e.g., '1999-2000.csv').
            
        Returns:
            DataFrame containing raw acoustic reflex data.
            
        Raises:
            FileNotFoundError: If the acoustic reflex file is not found.
        """
        filepath = self.data_dir / 'nhanes' / 'reflex' / f'nhanes_auxr_{cohort_suffix}'
        
        try:
            df = pd.read_csv(filepath)
            return df
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Acoustic reflex file not found: {filepath}. "
                f"Please ensure data directory structure is correct."
            )
    
    def extract_reflex_data(
        self,
        df: pd.DataFrame,
        seqn: Union[int, float],
        frequency: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract acoustic reflex data for a specific participant and frequency.
        
        Args:
            df: DataFrame containing acoustic reflex data.
            seqn: Participant sequence number (SEQN).
            frequency: Test frequency in Hz (1000 or 2000).
            
        Returns:
            Dictionary containing reflex response data for each ear:
            - 'right': DataFrame with time and compliance change for right ear
            - 'left': DataFrame with time and compliance change for left ear
            
        Raises:
            ValueError: If participant SEQN is not found or frequency is invalid.
        """
        if frequency not in self.frequencies:
            raise ValueError(f"Frequency {frequency} not supported. Use {self.frequencies}")
        
        # Find participant data
        participant_data = df[df['SEQN'] == seqn]
        
        if participant_data.empty:
            raise ValueError(f"Participant SEQN {seqn} not found in dataset")
        
        participant_data = participant_data.iloc[0]  # Get first (should be only) row
        
        # Get column names for this frequency
        right_columns = self.column_patterns[frequency]['right']
        left_columns = self.column_patterns[frequency]['left']
        
        # Extract right ear data
        right_ear_data = pd.DataFrame({
            'Time_ms': self.time_values,
            'Compliance_Change_ml': [participant_data[col] if col in participant_data.index 
                                   else np.nan for col in right_columns],
            'Ear': 'Right',
            'Frequency_Hz': frequency
        })
        
        # Extract left ear data
        left_ear_data = pd.DataFrame({
            'Time_ms': self.time_values,
            'Compliance_Change_ml': [participant_data[col] if col in participant_data.index 
                                   else np.nan for col in left_columns],
            'Ear': 'Left',
            'Frequency_Hz': frequency
        })
        
        return {
            'right': right_ear_data,
            'left': left_ear_data
        }
    
    def extract_all_frequencies(
        self,
        df: pd.DataFrame,
        seqn: Union[int, float]
    ) -> Dict[int, Dict[str, pd.DataFrame]]:
        """
        Extract acoustic reflex data for all frequencies for a participant.
        
        Args:
            df: DataFrame containing acoustic reflex data.
            seqn: Participant sequence number.
            
        Returns:
            Nested dictionary with structure:
            {frequency: {'right': DataFrame, 'left': DataFrame}}
        """
        all_data = {}
        
        for freq in self.frequencies:
            try:
                freq_data = self.extract_reflex_data(df, seqn, freq)
                all_data[freq] = freq_data
            except ValueError:
                # Skip if data not available for this frequency
                continue
        
        return all_data
    
    def calculate_reflex_parameters(
        self,
        time_values: np.ndarray,
        compliance_changes: np.ndarray,
        stimulus_intensity_db: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate acoustic reflex parameters from compliance change curve.
        
        Args:
            time_values: Array of time values in milliseconds.
            compliance_changes: Array of compliance changes in ml.
            stimulus_intensity_db: Optional stimulus intensity in dB SPL.
            
        Returns:
            Dictionary containing reflex parameters:
            - reflex_present: Whether reflex is present (boolean as float)
            - reflex_magnitude: Maximum compliance change (ml)
            - reflex_latency: Time to peak response (ms)  
            - reflex_decay: Decay from peak over 500ms (%)
            - reflex_duration: Duration of response >50% peak (ms)
            - baseline_stability: Stability of pre-stimulus baseline (ml)
        """
        # Remove NaN values
        valid_mask = ~np.isnan(compliance_changes)
        if not np.any(valid_mask) or np.sum(valid_mask) < 10:
            return {
                'reflex_present': 0.0,
                'reflex_magnitude': np.nan,
                'reflex_latency': np.nan,
                'reflex_decay': np.nan,
                'reflex_duration': np.nan,
                'baseline_stability': np.nan
            }
        
        valid_time = time_values[valid_mask]
        valid_compliance = compliance_changes[valid_mask]
        
        # Calculate baseline (first 100ms should be pre-stimulus)
        baseline_mask = valid_time <= 100
        if np.any(baseline_mask):
            baseline_values = valid_compliance[baseline_mask]
            baseline_mean = np.mean(baseline_values)
            baseline_stability = np.std(baseline_values)
        else:
            baseline_mean = 0.0
            baseline_stability = np.nan
        
        # Baseline-correct the compliance changes
        corrected_compliance = valid_compliance - baseline_mean
        
        # Find reflex magnitude (maximum absolute change)
        abs_compliance = np.abs(corrected_compliance)
        max_idx = np.argmax(abs_compliance)
        reflex_magnitude = abs_compliance[max_idx]
        
        # Determine if reflex is present
        reflex_present = float(reflex_magnitude >= self.reflex_thresholds['min_reflex_magnitude'])
        
        if reflex_present:
            # Calculate reflex latency (time to peak)
            reflex_latency = valid_time[max_idx]
            
            # Calculate reflex decay (change from peak over 500ms)
            peak_time = valid_time[max_idx]
            decay_window_start = peak_time + 100  # Start 100ms after peak
            decay_window_end = peak_time + 600    # End 600ms after peak
            
            decay_mask = (valid_time >= decay_window_start) & (valid_time <= decay_window_end)
            if np.any(decay_mask):
                decay_values = abs_compliance[decay_mask]
                if len(decay_values) > 0:
                    final_magnitude = np.mean(decay_values[-3:])  # Average of last 3 points
                    reflex_decay = ((reflex_magnitude - final_magnitude) / reflex_magnitude) * 100
                else:
                    reflex_decay = np.nan
            else:
                reflex_decay = np.nan
            
            # Calculate reflex duration (time above 50% of peak)
            half_peak = reflex_magnitude * 0.5
            above_half_peak = abs_compliance >= half_peak
            
            if np.any(above_half_peak):
                above_half_times = valid_time[above_half_peak]
                reflex_duration = above_half_times.max() - above_half_times.min()
            else:
                reflex_duration = 0.0
                
        else:
            reflex_latency = np.nan
            reflex_decay = np.nan
            reflex_duration = 0.0
        
        return {
            'reflex_present': reflex_present,
            'reflex_magnitude': reflex_magnitude,
            'reflex_latency': reflex_latency,
            'reflex_decay': reflex_decay,
            'reflex_duration': reflex_duration,
            'baseline_stability': baseline_stability
        }
    
    def classify_reflex_response(
        self,
        reflex_params: Dict[str, float],
        stimulus_intensity_db: Optional[float] = None
    ) -> Dict[str, Union[str, bool]]:
        """
        Classify acoustic reflex response based on calculated parameters.
        
        Args:
            reflex_params: Dictionary of reflex parameters.
            stimulus_intensity_db: Optional stimulus intensity in dB SPL.
            
        Returns:
            Dictionary containing classification results:
            - response_type: 'Normal', 'Absent', 'Elevated', 'Pathological'
            - is_normal: Boolean indicating normal response
            - clinical_significance: Description of clinical findings
        """
        reflex_present = reflex_params.get('reflex_present', 0.0)
        reflex_magnitude = reflex_params.get('reflex_magnitude', 0.0)
        reflex_decay = reflex_params.get('reflex_decay', 0.0)
        reflex_latency = reflex_params.get('reflex_latency', 0.0)
        
        if not reflex_present:
            response_type = 'Absent'
            is_normal = False
            clinical_significance = 'No reflex detected - may indicate conductive or retrocochlear pathology'
        
        elif stimulus_intensity_db and stimulus_intensity_db > self.reflex_thresholds['abnormal_threshold_db']:
            response_type = 'Elevated'
            is_normal = False
            clinical_significance = 'Elevated reflex threshold - possible conductive hearing loss'
        
        elif not np.isnan(reflex_decay) and reflex_decay > self.reflex_thresholds['decay_threshold_percent']:
            response_type = 'Pathological'
            is_normal = False
            clinical_significance = 'Excessive reflex decay - suggestive of retrocochlear pathology'
        
        elif not np.isnan(reflex_latency) and reflex_latency > 200:  # Normal latency <150ms
            response_type = 'Pathological'
            is_normal = False
            clinical_significance = 'Prolonged reflex latency - possible neural conduction delay'
        
        else:
            response_type = 'Normal'
            is_normal = True
            clinical_significance = 'Normal acoustic reflex response'
        
        return {
            'response_type': response_type,
            'is_normal': is_normal,
            'clinical_significance': clinical_significance
        }
    
    def get_participant_list(self, df: pd.DataFrame) -> List[float]:
        """
        Get list of available participant SEQNs in the dataset.
        
        Args:
            df: DataFrame containing acoustic reflex data.
            
        Returns:
            List of participant SEQN values.
        """
        return df['SEQN'].dropna().unique().tolist()
    
    def load_all_cohorts_reflex(
        self,
        cohort_suffixes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load and combine acoustic reflex data from multiple cohorts.
        
        Args:
            cohort_suffixes: Optional list of cohort suffixes to load.
                           If None, loads all available cohorts.
                           
        Returns:
            Combined DataFrame with all cohort reflex data.
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
            raise FileNotFoundError("No acoustic reflex data files found")
        
        return pd.concat(combined_data, ignore_index=True)
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for the acoustic reflex dataset.
        
        Args:
            df: DataFrame containing acoustic reflex data.
            
        Returns:
            Dictionary with summary statistics.
        """
        stats = {
            'total_participants': len(df),
            'participants_with_1khz_right': 0,
            'participants_with_1khz_left': 0,
            'participants_with_2khz_right': 0,
            'participants_with_2khz_left': 0
        }
        
        # Check for valid data in each condition
        for freq in self.frequencies:
            right_columns = self.column_patterns[freq]['right']
            left_columns = self.column_patterns[freq]['left']
            
            # Count participants with valid data (at least some non-zero values)
            right_data = df[right_columns].values if all(col in df.columns for col in right_columns) else np.array([[]])
            left_data = df[left_columns].values if all(col in df.columns for col in left_columns) else np.array([[]])
            
            if right_data.size > 0:
                stats[f'participants_with_{freq//1000}khz_right'] = np.sum(
                    np.any(np.abs(right_data) > 1, axis=1)
                )
            
            if left_data.size > 0:
                stats[f'participants_with_{freq//1000}khz_left'] = np.sum(
                    np.any(np.abs(left_data) > 1, axis=1)
                )
        
        return stats


def load_nhanes_acoustic_reflex(
    data_dir: Union[str, Path],
    cohort_suffixes: Optional[List[str]] = None
) -> Tuple[NHANESAcousticReflexLoader, pd.DataFrame]:
    """
    Convenience function to load NHANES acoustic reflex data.
    
    Args:
        data_dir: Path to NHANES data directory.
        cohort_suffixes: Optional list of cohort suffixes to load.
        
    Returns:
        Tuple of (loader_instance, combined_dataframe).
    """
    loader = NHANESAcousticReflexLoader(data_dir)
    df = loader.load_all_cohorts_reflex(cohort_suffixes)
    
    return loader, df