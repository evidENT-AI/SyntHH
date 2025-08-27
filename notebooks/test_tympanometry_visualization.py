"""
Test script for tympanometry visualization functionality.

This script demonstrates how to load and visualize NHANES tympanometry data
using the new tympanometry modules.
"""

import sys
sys.path.append('../src')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthh.tympanometry_loader import NHANESTympanometryLoader, load_nhanes_tympanometry
from synthh.tympanometry_visualizer import TympanometryVisualizer, visualize_participant_tympanograms

# Set random seed for reproducibility
np.random.seed(42)

def main():
    """Main function to test tympanometry visualization."""
    
    print("Testing NHANES Tympanometry Visualization")
    print("=" * 50)
    
    # Set data directory
    data_dir = '../data'
    
    try:
        # 1. Load tympanometry data
        print("1. Loading tympanometry data...")
        loader, df = load_nhanes_tympanometry(data_dir, ['1999-2000.csv'])
        print(f"   Loaded {len(df)} participants")
        
        # Get summary statistics
        stats = loader.get_summary_statistics(df)
        print(f"   Participants with right ear data: {stats['participants_with_right_ear']}")
        print(f"   Participants with left ear data: {stats['participants_with_left_ear']}")
        
        # 2. Get available participants
        participant_list = loader.get_participant_list(df)
        print(f"   Available participants: {len(participant_list)}")
        
        if len(participant_list) == 0:
            print("   No participants found with tympanometry data!")
            return
        
        # 3. Test single participant visualization
        print("\\n2. Testing single participant visualization...")
        test_seqn = participant_list[0]
        print(f"   Using participant SEQN: {test_seqn}")
        
        try:
            # Extract data for test participant
            participant_data = loader.extract_tympanogram_data(df, test_seqn)
            print(f"   Successfully extracted data for SEQN {test_seqn}")
            
            # Print some sample data
            right_ear = participant_data['right']
            print(f"   Right ear pressure range: {right_ear['Pressure_daPa'].min():.0f} to {right_ear['Pressure_daPa'].max():.0f} daPa")
            print(f"   Right ear compliance range: {right_ear['Compliance_ml'].min():.3f} to {right_ear['Compliance_ml'].max():.3f} ml")
            
            # 4. Create visualizations
            print("\\n3. Creating tympanometry visualizations...")
            
            # Initialize visualizer
            visualizer = TympanometryVisualizer(style='clinical')
            
            # Plot bilateral tympanograms
            print("   Creating bilateral tympanogram plot...")
            fig1 = visualizer.plot_bilateral_tympanograms(loader, df, test_seqn)
            plt.show()
            plt.close(fig1)
            
            # Plot multiple participants (first 6)
            print("   Creating multiple participant comparison...")
            sample_participants = participant_list[:6]
            fig2 = visualizer.plot_multiple_participants(
                loader, df, sample_participants, ear='Right'
            )
            plt.show()
            plt.close(fig2)
            
            # Overlay comparison
            print("   Creating overlay comparison...")
            fig3 = visualizer.plot_overlay_comparison(
                loader, df, sample_participants[:4], ear='Right'
            )
            plt.show()
            plt.close(fig3)
            
            # Summary statistics
            print("   Creating summary statistics plot...")
            fig4 = visualizer.plot_summary_statistics(df, loader, sample_size=50)
            plt.show()
            plt.close(fig4)
            
            print("\\n4. Testing parameter calculations...")
            # Test parameter calculations
            right_ear_data = participant_data['right']
            pressure = right_ear_data['Pressure_daPa'].values
            compliance = right_ear_data['Compliance_ml'].values
            
            params = loader.calculate_tympanometric_parameters(pressure, compliance)
            print(f"   Peak pressure: {params['peak_pressure']:.1f} daPa")
            print(f"   Peak compliance: {params['peak_compliance']:.3f} ml")
            print(f"   Tympanometric gradient: {params['gradient']:.1f} daPa")
            print(f"   Equivalent volume: {params['equivalent_volume']:.3f} ml")
            
            # Test convenience function
            print("\\n5. Testing convenience function...")
            fig5 = visualize_participant_tympanograms(data_dir, test_seqn, '1999-2000', 'scientific')
            plt.show()
            plt.close(fig5)
            
            print("\\n✓ All tympanometry visualization tests completed successfully!")
            
        except ValueError as e:
            print(f"   Error with participant {test_seqn}: {e}")
            # Try with a different participant
            if len(participant_list) > 1:
                test_seqn = participant_list[1]
                print(f"   Trying with participant {test_seqn}...")
                participant_data = loader.extract_tympanogram_data(df, test_seqn)
                fig = visualizer.plot_bilateral_tympanograms(loader, df, test_seqn)
                plt.show()
                plt.close(fig)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find tympanometry data files.")
        print(f"Make sure the data directory structure is correct: {e}")
        return
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return


def demonstrate_tympanogram_types():
    """Demonstrate different tympanogram types with synthetic data."""
    
    print("\\nDemonstrating Different Tympanogram Types")
    print("=" * 50)
    
    # Create synthetic tympanogram examples
    pressure_values = np.arange(-300, 199, 6)
    
    # Type A - Normal tympanogram
    peak_pressure = 0
    peak_compliance = 1.0
    type_a = generate_synthetic_tympanogram(pressure_values, peak_pressure, peak_compliance, width=100)
    
    # Type As - Shallow/stiff
    type_as = generate_synthetic_tympanogram(pressure_values, 0, 0.2, width=80)
    
    # Type Ad - Deep/flaccid  
    type_ad = generate_synthetic_tympanogram(pressure_values, 0, 2.5, width=120)
    
    # Type B - Flat
    type_b = np.full_like(pressure_values, 0.1) + np.random.normal(0, 0.02, len(pressure_values))
    
    # Type C - Negative pressure
    type_c = generate_synthetic_tympanogram(pressure_values, -200, 0.8, width=90)
    
    # Plot all types
    visualizer = TympanometryVisualizer(style='clinical')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    types_data = [
        (type_a, "Type A (Normal)", 'green'),
        (type_as, "Type As (Stiff)", 'orange'), 
        (type_ad, "Type Ad (Flaccid)", 'blue'),
        (type_b, "Type B (Flat)", 'red'),
        (type_c, "Type C (Negative)", 'purple')
    ]
    
    for i, (compliance, title, color) in enumerate(types_data):
        ax = axes[i]
        ax.plot(pressure_values, compliance, linewidth=3, color=color)
        ax.fill_between(pressure_values, compliance, alpha=0.3, color=color)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Pressure (daPa)')
        ax.set_ylabel('Compliance (ml)')
        ax.set_xlim(-350, 250)
        ax.set_ylim(0, max(compliance) * 1.2)
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Tympanogram Type Classification Examples', fontsize=16, fontweight='bold', y=0.98)
    plt.show()


def generate_synthetic_tympanogram(pressure_values, peak_pressure, peak_compliance, width):
    """Generate a synthetic tympanogram curve."""
    # Create Gaussian-like curve
    compliance = peak_compliance * np.exp(-0.5 * ((pressure_values - peak_pressure) / (width/2))**2)
    
    # Add some noise
    compliance += np.random.normal(0, peak_compliance * 0.05, len(pressure_values))
    
    # Ensure no negative compliance
    compliance = np.maximum(compliance, 0.01)
    
    return compliance


if __name__ == "__main__":
    # Run the main test
    main()
    
    # Demonstrate tympanogram types
    demonstrate_tympanogram_types()