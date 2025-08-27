"""
NHANES Tympanometry Visualization Module

This module provides functions for visualizing tympanometry data from NHANES,
including individual tympanograms, comparative plots, and summary statistics
visualization.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

from .tympanometry_loader import NHANESTympanometryLoader


class TympanometryVisualizer:
    """
    A class for creating tympanometry visualizations.
    
    This class provides methods for plotting tympanograms, comparing measurements
    across participants, and creating publication-quality figures.
    """
    
    def __init__(self, style: str = 'clinical'):
        """
        Initialize the tympanometry visualizer.
        
        Args:
            style: Plotting style ('clinical', 'scientific', or 'minimal').
        """
        self.style = style
        self._setup_style()
        
        # Clinical reference ranges for tympanometric parameters
        self.reference_ranges = {
            'peak_pressure': (-150, 50),    # daPa, normal middle ear pressure
            'peak_compliance': (0.3, 1.7),  # ml, normal tympanic membrane compliance
            'gradient': (50, 150),          # daPa, normal tympanometric width
            'equivalent_volume': (0.4, 1.0) # ml, normal ear canal volume
        }
        
        # Tympanogram type classifications
        self.tymp_types = {
            'Type A': {'description': 'Normal', 'color': 'green'},
            'Type As': {'description': 'Shallow/Stiff', 'color': 'orange'}, 
            'Type Ad': {'description': 'Deep/Flaccid', 'color': 'blue'},
            'Type B': {'description': 'Flat/Fluid', 'color': 'red'},
            'Type C': {'description': 'Negative Pressure', 'color': 'purple'}
        }
    
    def _setup_style(self):
        """Setup matplotlib plotting style based on selected style."""
        if self.style == 'clinical':
            plt.style.use('seaborn-v0_8-whitegrid')
            self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
            self.grid_alpha = 0.3
        elif self.style == 'scientific':
            plt.style.use('seaborn-v0_8-paper')
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            self.grid_alpha = 0.5
        else:  # minimal
            plt.style.use('seaborn-v0_8-white')
            self.colors = ['#333333', '#666666', '#999999', '#CCCCCC']
            self.grid_alpha = 0.2
    
    def plot_single_tympanogram(
        self,
        pressure_values: np.ndarray,
        compliance_values: np.ndarray,
        ear: str,
        seqn: Optional[Union[int, float]] = None,
        ax: Optional[plt.Axes] = None,
        show_reference_ranges: bool = True,
        classify_type: bool = True
    ) -> plt.Axes:
        """
        Plot a single tympanogram curve.
        
        Args:
            pressure_values: Array of pressure values in daPa.
            compliance_values: Array of compliance values in ml.
            ear: Ear designation ('Left' or 'Right').
            seqn: Optional participant sequence number for title.
            ax: Optional matplotlib axes object.
            show_reference_ranges: Whether to show normal reference ranges.
            classify_type: Whether to classify and display tympanogram type.
            
        Returns:
            Matplotlib axes object with the plotted tympanogram.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(compliance_values)
        plot_pressure = pressure_values[valid_mask]
        plot_compliance = compliance_values[valid_mask]
        
        # Plot the tympanogram curve
        ax.plot(plot_pressure, plot_compliance, 
               linewidth=2.5, color=self.colors[0], 
               marker='o', markersize=3, alpha=0.8,
               label=f'{ear} Ear')
        
        # Fill area under curve for better visualization
        ax.fill_between(plot_pressure, plot_compliance, alpha=0.3, color=self.colors[0])
        
        # Add reference ranges if requested
        if show_reference_ranges:
            self._add_reference_ranges(ax)
        
        # Calculate and display tympanometric parameters
        from .tympanometry_loader import NHANESTympanometryLoader
        loader = NHANESTympanometryLoader('.')  # Dummy loader for parameter calculation
        params = loader.calculate_tympanometric_parameters(plot_pressure, plot_compliance)
        
        # Mark peak if valid
        if not np.isnan(params['peak_pressure']) and not np.isnan(params['peak_compliance']):
            ax.plot(params['peak_pressure'], params['peak_compliance'], 
                   'ro', markersize=8, label=f"Peak: {params['peak_compliance']:.2f} ml")
        
        # Classify tympanogram type if requested
        if classify_type:
            tymp_type = self._classify_tympanogram_type(params)
            type_info = self.tymp_types.get(tymp_type, {'description': 'Unknown', 'color': 'gray'})
            ax.text(0.02, 0.98, f'Type: {tymp_type}\n({type_info["description"]})',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=type_info['color'], alpha=0.3))
        
        # Customize axes
        ax.set_xlabel('Pressure (daPa)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Compliance (ml)', fontsize=12, fontweight='bold')
        
        title = f'Tympanogram - {ear} Ear'
        if seqn is not None:
            title += f' (SEQN: {seqn})'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set axis limits and grid
        ax.set_xlim(-350, 250)
        ax.set_ylim(0, max(plot_compliance) * 1.1 if len(plot_compliance) > 0 else 2)
        ax.grid(True, alpha=self.grid_alpha)
        ax.legend(fontsize=10)
        
        # Add parameter text box
        if not all(np.isnan(list(params.values()))):
            param_text = self._format_parameter_text(params)
            ax.text(0.98, 0.02, param_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ax
    
    def plot_bilateral_tympanograms(
        self,
        loader: NHANESTympanometryLoader,
        df: pd.DataFrame,
        seqn: Union[int, float],
        figsize: Tuple[float, float] = (15, 6)
    ) -> plt.Figure:
        """
        Plot bilateral tympanograms for a single participant.
        
        Args:
            loader: NHANESTympanometryLoader instance.
            df: DataFrame containing tympanometry data.
            seqn: Participant sequence number.
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure object containing both tympanograms.
        """
        # Extract participant data
        try:
            ear_data = loader.extract_tympanogram_data(df, seqn)
        except ValueError as e:
            raise ValueError(f"Cannot plot tympanograms: {e}")
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot right ear
        self.plot_single_tympanogram(
            ear_data['right']['Pressure_daPa'].values,
            ear_data['right']['Compliance_ml'].values,
            'Right', seqn, ax1
        )
        
        # Plot left ear
        self.plot_single_tympanogram(
            ear_data['left']['Pressure_daPa'].values,
            ear_data['left']['Compliance_ml'].values,
            'Left', seqn, ax2
        )
        
        plt.tight_layout()
        return fig
    
    def plot_multiple_participants(
        self,
        loader: NHANESTympanometryLoader,
        df: pd.DataFrame,
        seqn_list: List[Union[int, float]],
        ear: str = 'Right',
        max_plots: int = 6,
        figsize: Tuple[float, float] = (15, 10)
    ) -> plt.Figure:
        """
        Plot tympanograms for multiple participants.
        
        Args:
            loader: NHANESTympanometryLoader instance.
            df: DataFrame containing tympanometry data.
            seqn_list: List of participant sequence numbers.
            ear: Which ear to plot ('Left' or 'Right').
            max_plots: Maximum number of plots to display.
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure object with multiple tympanogram subplots.
        """
        n_plots = min(len(seqn_list), max_plots)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, seqn in enumerate(seqn_list[:max_plots]):
            ax = axes[i] if n_plots > 1 else axes[0]
            
            try:
                ear_data = loader.extract_tympanogram_data(df, seqn)
                ear_key = ear.lower()
                
                self.plot_single_tympanogram(
                    ear_data[ear_key]['Pressure_daPa'].values,
                    ear_data[ear_key]['Compliance_ml'].values,
                    ear, seqn, ax, show_reference_ranges=False, classify_type=False
                )
                
            except ValueError:
                ax.text(0.5, 0.5, f'No data\nSEQN: {seqn}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_xlim(-350, 250)
                ax.set_ylim(0, 2)
        
        # Hide empty subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_overlay_comparison(
        self,
        loader: NHANESTympanometryLoader,
        df: pd.DataFrame,
        seqn_list: List[Union[int, float]],
        ear: str = 'Right',
        figsize: Tuple[float, float] = (12, 8)
    ) -> plt.Figure:
        """
        Plot multiple tympanograms overlaid on the same axes for comparison.
        
        Args:
            loader: NHANEsTympanometryLoader instance.
            df: DataFrame containing tympanometry data.
            seqn_list: List of participant sequence numbers.
            ear: Which ear to plot ('Left' or 'Right').
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure object with overlaid tympanograms.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(seqn_list)))
        
        for i, seqn in enumerate(seqn_list):
            try:
                ear_data = loader.extract_tympanogram_data(df, seqn)
                ear_key = ear.lower()
                
                pressure = ear_data[ear_key]['Pressure_daPa'].values
                compliance = ear_data[ear_key]['Compliance_ml'].values
                
                # Remove NaN values
                valid_mask = ~np.isnan(compliance)
                plot_pressure = pressure[valid_mask]
                plot_compliance = compliance[valid_mask]
                
                ax.plot(plot_pressure, plot_compliance, 
                       linewidth=2, color=colors[i], alpha=0.7,
                       label=f'SEQN {seqn}')
                
            except ValueError:
                continue
        
        self._add_reference_ranges(ax)
        
        ax.set_xlabel('Pressure (daPa)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Compliance (ml)', fontsize=12, fontweight='bold')
        ax.set_title(f'Tympanogram Comparison - {ear} Ear', fontsize=14, fontweight='bold')
        ax.set_xlim(-350, 250)
        ax.grid(True, alpha=self.grid_alpha)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def _add_reference_ranges(self, ax: plt.Axes):
        """Add reference range shading to tympanogram plot."""
        # Normal pressure range
        pressure_range = self.reference_ranges['peak_pressure']
        ax.axvspan(pressure_range[0], pressure_range[1], alpha=0.1, color='green', 
                  label='Normal Pressure Range')
        
        # Normal compliance range (horizontal band)
        compliance_range = self.reference_ranges['peak_compliance']
        ax.axhspan(compliance_range[0], compliance_range[1], alpha=0.1, color='blue',
                  label='Normal Compliance Range')
    
    def _classify_tympanogram_type(self, params: Dict[str, float]) -> str:
        """
        Classify tympanogram type based on calculated parameters.
        
        Args:
            params: Dictionary of tympanometric parameters.
            
        Returns:
            String describing the tympanogram type.
        """
        peak_pressure = params.get('peak_pressure', np.nan)
        peak_compliance = params.get('peak_compliance', np.nan)
        
        if np.isnan(peak_pressure) or np.isnan(peak_compliance):
            return 'Type B'  # Flat/no clear peak
        
        # Type C: Negative pressure
        if peak_pressure < -150:
            return 'Type C'
        
        # Type A variants based on compliance
        if peak_compliance < 0.3:
            return 'Type As'  # Shallow/stiff
        elif peak_compliance > 1.7:
            return 'Type Ad'  # Deep/flaccid
        else:
            return 'Type A'   # Normal
    
    def _format_parameter_text(self, params: Dict[str, float]) -> str:
        """Format tympanometric parameters for display."""
        lines = []
        
        if not np.isnan(params['peak_pressure']):
            lines.append(f"Peak Pressure: {params['peak_pressure']:.0f} daPa")
        
        if not np.isnan(params['peak_compliance']):
            lines.append(f"Peak Compliance: {params['peak_compliance']:.2f} ml")
        
        if not np.isnan(params['gradient']):
            lines.append(f"Gradient: {params['gradient']:.0f} daPa")
        
        if not np.isnan(params['equivalent_volume']):
            lines.append(f"Equiv. Volume: {params['equivalent_volume']:.2f} ml")
        
        return '\n'.join(lines)
    
    def plot_summary_statistics(
        self,
        df: pd.DataFrame,
        loader: NHANESTympanometryLoader,
        sample_size: int = 100,
        figsize: Tuple[float, float] = (15, 10)
    ) -> plt.Figure:
        """
        Plot summary statistics for tympanometric parameters across participants.
        
        Args:
            df: DataFrame containing tympanometry data.
            loader: NHANESTympanometryLoader instance.
            sample_size: Number of participants to sample for analysis.
            figsize: Figure size as (width, height).
            
        Returns:
            Matplotlib figure with summary statistics plots.
        """
        # Sample participants and calculate parameters
        available_seqns = loader.get_participant_list(df)
        sample_seqns = np.random.choice(available_seqns, 
                                       size=min(sample_size, len(available_seqns)), 
                                       replace=False)
        
        # Collect parameters for both ears
        parameters_data = []
        
        for seqn in sample_seqns:
            try:
                ear_data = loader.extract_tympanogram_data(df, seqn)
                
                for ear in ['right', 'left']:
                    pressure = ear_data[ear]['Pressure_daPa'].values
                    compliance = ear_data[ear]['Compliance_ml'].values
                    params = loader.calculate_tympanometric_parameters(pressure, compliance)
                    
                    params['SEQN'] = seqn
                    params['Ear'] = ear.title()
                    parameters_data.append(params)
                    
            except ValueError:
                continue
        
        params_df = pd.DataFrame(parameters_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        parameters = ['peak_pressure', 'peak_compliance', 'gradient', 'equivalent_volume']
        titles = ['Peak Pressure (daPa)', 'Peak Compliance (ml)', 
                 'Tympanometric Width (daPa)', 'Equivalent Volume (ml)']
        
        for i, (param, title) in enumerate(zip(parameters, titles)):
            ax = axes[i]
            
            # Box plot by ear
            params_df_clean = params_df.dropna(subset=[param])
            if not params_df_clean.empty:
                sns.boxplot(data=params_df_clean, x='Ear', y=param, ax=ax)
                ax.set_title(title, fontweight='bold')
                ax.grid(True, alpha=self.grid_alpha)
                
                # Add reference range if available
                if param in self.reference_ranges:
                    ref_range = self.reference_ranges[param]
                    ax.axhspan(ref_range[0], ref_range[1], alpha=0.2, color='green',
                              label='Normal Range')
                    ax.legend()
        
        plt.tight_layout()
        return fig


def visualize_participant_tympanograms(
    data_dir: str,
    seqn: Union[int, float],
    cohort: str = '1999-2000',
    style: str = 'clinical'
) -> plt.Figure:
    """
    Convenience function to visualize tympanograms for a single participant.
    
    Args:
        data_dir: Path to NHANES data directory.
        seqn: Participant sequence number.
        cohort: Cohort identifier (e.g., '1999-2000').
        style: Visualization style ('clinical', 'scientific', or 'minimal').
        
    Returns:
        Matplotlib figure with bilateral tympanograms.
    """
    from .tympanometry_loader import load_nhanes_tympanometry
    
    # Load data
    loader, df = load_nhanes_tympanometry(data_dir, [f'{cohort}.csv'])
    
    # Create visualizer and plot
    visualizer = TympanometryVisualizer(style=style)
    fig = visualizer.plot_bilateral_tympanograms(loader, df, seqn)
    
    return fig