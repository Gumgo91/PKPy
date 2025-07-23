#!/usr/bin/env python3
"""
Complete Theophylline data analysis with AFE/AAFE metrics
Demonstrates proper data loading and workflow integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pkpy.theophylline_loader import load_theophylline_data, prepare_for_workflow
from pkpy.models import create_pkpy_model
from pkpy.workflow import BasePKWorkflow
from pkpy.fitting import PKPyFit
from pkpy.utils import calculate_validation_metrics, calculate_nca_parameters

def analyze_theophylline():
    """Comprehensive Theophylline analysis with all metrics"""
    print("=" * 70)
    print("THEOPHYLLINE DATA ANALYSIS WITH AFE/AAFE METRICS")
    print("=" * 70)
    
    # Load Theophylline data
    print("\n1. Loading Theophylline data...")
    data = load_theophylline_data(
        'Examples/concentrations.csv',
        'Examples/times.csv',
        'Examples/demographics.csv'
    )
    
    # Prepare data for workflow
    times, concentrations, demographics, median_dose = prepare_for_workflow(data)
    
    print(f"   - Loaded {len(concentrations)} subjects")
    print(f"   - Time points: {len(times)} measurements")
    print(f"   - Median dose: {median_dose:.1f} mg")
    print(f"   - Dose range: {data['doses'].min():.1f} - {data['doses'].max():.1f} mg")
    
    # Create model
    print("\n2. Creating one-compartment absorption model...")
    param_specs = {
        'Ka': {'value': 1.5, 'cv_percent': 50},
        'CL': {'value': 2.8, 'cv_percent': 30},
        'V': {'value': 32.0, 'cv_percent': 30}
    }
    model = create_pkpy_model('onecomp_abs', param_specs)
    
    # Create workflow and prepare data
    print("\n3. Setting up workflow...")
    workflow = BasePKWorkflow(model=model)
    workflow.prop_err = 0.15
    
    # Properly set up the data structure
    workflow.times = times
    workflow.data = {
        'times': times,  # Add times to data dict
        'concentrations': concentrations,
        'demographics': demographics,
        'dose': median_dose,  # Add dose to data dict
        'individual_parameters': pd.DataFrame({
            'ID': range(1, len(concentrations) + 1),
            'Ka': np.random.lognormal(np.log(1.5), 0.5, len(concentrations)),
            'CL': np.random.lognormal(np.log(2.8), 0.3, len(concentrations)),
            'V': np.random.lognormal(np.log(32.0), 0.3, len(concentrations))
        })
    }
    
    # Run model fitting
    print("\n4. Fitting population PK model...")
    try:
        workflow.run_model_fitting()
        print("   ✓ Model fitting completed successfully")
    except Exception as e:
        print(f"   ✗ Model fitting error: {str(e)}")
        return None
    
    # Display fit metrics
    if 'fit_metrics' in workflow.results:
        print("\n5. MODEL FIT METRICS:")
        print("-" * 40)
        metrics = workflow.results['fit_metrics']
        
        # Standard metrics
        print(f"   R²:              {metrics.get('R2', np.nan):.4f}")
        print(f"   RMSE:            {metrics.get('RMSE', np.nan):.4f}")
        print(f"   MAE:             {metrics.get('MAE', np.nan):.4f}")
        
        # New fold-error metrics
        afe = metrics.get('AFE', np.nan)
        aafe = metrics.get('AAFE', np.nan)
        
        print(f"\n   AFE:             {afe:.4f}")
        if not np.isnan(afe):
            if afe > 1:
                print(f"                    (Model overpredicts by {(afe-1)*100:.1f}%)")
            elif afe < 1:
                print(f"                    (Model underpredicts by {(1-afe)*100:.1f}%)")
            else:
                print("                    (No systematic bias)")
        
        print(f"\n   AAFE:            {aafe:.4f}")
        if not np.isnan(aafe):
            if aafe < 1.5:
                print("                    (Excellent prediction accuracy)")
            elif aafe < 2.0:
                print("                    (Good prediction accuracy)")
            else:
                print("                    (Poor prediction accuracy)")
    
    # Run NCA analysis
    print("\n\n6. Running Non-Compartmental Analysis...")
    try:
        workflow.run_nca_analysis()
        print("   ✓ NCA analysis completed")
        
        if 'nca' in workflow.results:
            nca_summary = workflow.results['nca']['summary']
            print("\n   NCA SUMMARY (population mean ± SD):")
            print("   " + "-" * 35)
            
            params_to_show = ['AUC', 'Cmax', 'Tmax', 'half_life', 'clearance']
            for param in params_to_show:
                if param in nca_summary.columns:
                    mean_val = nca_summary[param]['mean']
                    std_val = nca_summary[param]['std']
                    if param == 'AUC':
                        print(f"   {param:<12}: {mean_val:>8.1f} ± {std_val:>6.1f} mg·h/L")
                    elif param == 'Cmax':
                        print(f"   {param:<12}: {mean_val:>8.2f} ± {std_val:>6.2f} mg/L")
                    elif param == 'Tmax':
                        print(f"   {param:<12}: {mean_val:>8.2f} ± {std_val:>6.2f} h")
                    elif param == 'half_life':
                        print(f"   Half-life    : {mean_val:>8.2f} ± {std_val:>6.2f} h")
                    elif param == 'clearance':
                        print(f"   CL (NCA)     : {mean_val:>8.2f} ± {std_val:>6.2f} L/h")
            
            success_rate = workflow.results['nca']['success_rate']
            print(f"\n   Success rate: {success_rate*100:.0f}%")
            
    except Exception as e:
        print(f"   ✗ NCA analysis error: {str(e)}")
    
    # Create diagnostic plots
    print("\n7. Creating diagnostic plots...")
    try:
        workflow.create_diagnostic_plots(log_scale=False)
        plt.savefig('theophylline_diagnostics.png', dpi=300, bbox_inches='tight')
        print("   ✓ Saved diagnostic plots to 'theophylline_diagnostics.png'")
    except Exception as e:
        print(f"   ✗ Error creating plots: {str(e)}")
    
    # Individual analysis example
    print("\n8. Example Individual Analysis (Subject 1):")
    print("-" * 40)
    
    subj_idx = 0
    subj_times = data['individual_times'][subj_idx]
    subj_conc = data['individual_concentrations'][subj_idx]
    subj_dose = data['doses'][subj_idx]
    
    try:
        nca_params = calculate_nca_parameters(subj_times, subj_conc, subj_dose)
        print(f"   Subject dose: {subj_dose:.1f} mg")
        print(f"   AUC:          {nca_params['AUC']:.1f} mg·h/L")
        print(f"   Cmax:         {nca_params['Cmax']:.2f} mg/L")
        print(f"   Tmax:         {nca_params['Tmax']:.2f} h")
        print(f"   Half-life:    {nca_params['half_life']:.2f} h")
        print(f"   Clearance:    {nca_params['clearance']:.2f} L/h")
    except Exception as e:
        print(f"   Individual analysis error: {str(e)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    if 'fit_metrics' in workflow.results:
        metrics = workflow.results['fit_metrics']
        r2 = metrics.get('R2', np.nan)
        afe = metrics.get('AFE', np.nan)
        aafe = metrics.get('AAFE', np.nan)
        
        print(f"\nModel Fit Quality:")
        if r2 > 0.9:
            print(f"  • Excellent model fit (R² = {r2:.3f})")
        elif r2 > 0.8:
            print(f"  • Good model fit (R² = {r2:.3f})")
        else:
            print(f"  • Poor model fit (R² = {r2:.3f})")
        
        print(f"\nPrediction Accuracy:")
        if not np.isnan(afe):
            print(f"  • AFE = {afe:.3f} - ", end="")
            if 0.8 < afe < 1.25:
                print("Acceptable bias")
            else:
                print("Significant bias detected")
        
        if not np.isnan(aafe):
            print(f"  • AAFE = {aafe:.3f} - ", end="")
            if aafe < 1.5:
                print("Excellent accuracy")
            elif aafe < 2.0:
                print("Good accuracy")
            else:
                print("Poor accuracy")
    
    return workflow

def create_individual_plots(data):
    """Create plots showing individual concentration-time profiles"""
    print("\n9. Creating individual profile plots...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(12):  # Plot all 12 subjects
        ax = axes[i]
        times = data['individual_times'][i]
        concs = data['individual_concentrations'][i]
        
        ax.scatter(times, concs, s=50, alpha=0.7)
        ax.plot(times, concs, '-', alpha=0.5)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Conc (mg/L)')
        ax.set_title(f'Subject {i+1}')
        ax.set_xlim(0, 25)
        ax.set_ylim(0, max(concs) * 1.1)
    
    plt.tight_layout()
    plt.savefig('theophylline_individual_profiles.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved individual profiles to 'theophylline_individual_profiles.png'")

if __name__ == "__main__":
    # Run the analysis
    workflow = analyze_theophylline()
    
    # Create additional plots
    data = load_theophylline_data(
        'Examples/concentrations.csv',
        'Examples/times.csv',
        'Examples/demographics.csv'
    )
    create_individual_plots(data)
    
    print("\n✓ Analysis complete!")