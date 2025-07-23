#!/usr/bin/env python3
"""
Comprehensive test of all validation metrics in PKPy
Tests both simulated and real data scenarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkpy.workflow import BasePKWorkflow
from pkpy.models import create_pkpy_model
from pkpy.utils import calculate_validation_metrics

def test_basic_metrics():
    """Test the validation metrics function directly"""
    print("=" * 50)
    print("TESTING BASIC VALIDATION METRICS")
    print("=" * 50)
    
    # Test 1: Perfect prediction
    obs = np.array([10, 8, 6, 4, 2, 1])
    pred = obs.copy()
    metrics = calculate_validation_metrics(obs, pred)
    print("\nTest 1: Perfect Prediction")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test 2: 20% overestimation
    pred_over = obs * 1.2
    metrics = calculate_validation_metrics(obs, pred_over)
    print("\nTest 2: 20% Overestimation")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test 3: 20% underestimation
    pred_under = obs * 0.8
    metrics = calculate_validation_metrics(obs, pred_under)
    print("\nTest 3: 20% Underestimation")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

def test_onecomp_workflow():
    """Test one-compartment model workflow with all metrics"""
    print("\n" + "=" * 50)
    print("TESTING ONE-COMPARTMENT MODEL WORKFLOW")
    print("=" * 50)
    
    # Create model
    param_specs = {
        'CL': {'value': 5.0, 'cv_percent': 30},
        'V': {'value': 30.0, 'cv_percent': 30}
    }
    model = create_pkpy_model('onecomp', param_specs)
    
    # Create workflow
    workflow = BasePKWorkflow(model=model, n_subjects=30)
    workflow.prop_err = 0.1
    
    # Generate virtual population
    print("\nGenerating virtual population...")
    workflow.generate_virtual_population(
        time_points=np.array([0.5, 1, 2, 4, 6, 8, 12, 16, 20, 24]),
        dose=100.0,
        demographic_covariates={
            'CRCL': {'mean': 90, 'sd': 20},
            'WT': {'mean': 70, 'sd': 15},
            'AGE': {'mean': 45, 'sd': 10}
        }
    )
    
    # Run analysis
    print("Running PK analysis...")
    workflow.run_analysis(create_plots=False)
    
    # Display results
    if 'fit_metrics' in workflow.results:
        print("\nFIT METRICS:")
        print("-" * 30)
        for key, value in workflow.results['fit_metrics'].items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                print(f"  {key}: {value:.4f}")
            elif np.isnan(value):
                print(f"  {key}: NaN (calculation failed)")
            else:
                print(f"  {key}: {value}")
    
    # Print parameter estimates
    if 'model_fit' in workflow.results:
        params = workflow.results['model_fit'].get('params', {})
        print("\nPARAMETER ESTIMATES:")
        print("-" * 30)
        for param, value in params.items():
            if isinstance(value, (int, float)):
                print(f"  {param}: {value:.4f}")
    
    return workflow

def test_onecomp_abs_workflow():
    """Test one-compartment absorption model workflow"""
    print("\n" + "=" * 50)
    print("TESTING ONE-COMPARTMENT ABSORPTION MODEL WORKFLOW")
    print("=" * 50)
    
    # Create model
    param_specs = {
        'Ka': {'value': 1.5, 'cv_percent': 30},
        'CL': {'value': 3.0, 'cv_percent': 30},
        'V': {'value': 35.0, 'cv_percent': 30}
    }
    model = create_pkpy_model('onecomp_abs', param_specs)
    
    # Create workflow
    workflow = BasePKWorkflow(model=model, n_subjects=30)
    workflow.prop_err = 0.15
    
    # Generate virtual population
    print("\nGenerating virtual population...")
    workflow.generate_virtual_population(
        time_points=np.array([0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 24]),
        dose=100.0,
        demographic_covariates={
            'CRCL': {'mean': 90, 'sd': 20},
            'WT': {'mean': 70, 'sd': 15},
            'AGE': {'mean': 45, 'sd': 10}
        }
    )
    
    # Run analysis
    print("Running PK analysis...")
    workflow.run_analysis(create_plots=False)
    
    # Display results
    if 'fit_metrics' in workflow.results:
        print("\nFIT METRICS:")
        print("-" * 30)
        for key, value in workflow.results['fit_metrics'].items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                print(f"  {key}: {value:.4f}")
            elif np.isnan(value):
                print(f"  {key}: NaN (calculation failed)")
            else:
                print(f"  {key}: {value}")
    
    # Print parameter estimates
    if 'model_fit' in workflow.results:
        params = workflow.results['model_fit'].get('params', {})
        print("\nPARAMETER ESTIMATES:")
        print("-" * 30)
        for param, value in params.items():
            if isinstance(value, (int, float)):
                print(f"  {param}: {value:.4f}")
    
    return workflow

def test_theophylline_data():
    """Test with real Theophylline data"""
    print("\n" + "=" * 50)
    print("TESTING WITH THEOPHYLLINE DATA")
    print("=" * 50)
    
    try:
        # Load data
        concentrations = pd.read_csv('Examples/concentrations.csv')
        times = pd.read_csv('Examples/times.csv')
        demographics = pd.read_csv('Examples/demographics.csv')
        
        print(f"\nLoaded data for {len(concentrations)} subjects")
        print(f"Time points: {times['time'].values}")
        
        # Create model
        param_specs = {
            'Ka': {'value': 1.5, 'cv_percent': 50},
            'CL': {'value': 2.8, 'cv_percent': 30},
            'V': {'value': 32.0, 'cv_percent': 30}
        }
        model = create_pkpy_model('onecomp_abs', param_specs)
        
        # Create workflow
        workflow = BasePKWorkflow(model=model)
        workflow.prop_err = 0.15
        
        # Prepare data for workflow
        workflow.times = times['time'].values
        
        # Convert concentration data to numpy array
        conc_array = concentrations.iloc[:, 1:].values
        
        # Set dose (320 mg for Theophylline dataset)
        workflow.dose = 320.0
        
        # Create data structure
        workflow.data = {
            'concentrations': conc_array,
            'demographics': demographics,
            'individual_parameters': pd.DataFrame({
                'Ka': np.random.lognormal(np.log(1.5), 0.5, len(concentrations)),
                'CL': np.random.lognormal(np.log(2.8), 0.3, len(concentrations)),
                'V': np.random.lognormal(np.log(32.0), 0.3, len(concentrations))
            })
        }
        
        # Run model fitting
        print("\nRunning model fitting...")
        workflow.run_model_fitting()
        
        # Display results
        if 'fit_metrics' in workflow.results:
            print("\nFIT METRICS:")
            print("-" * 30)
            for key, value in workflow.results['fit_metrics'].items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    print(f"  {key}: {value:.4f}")
                elif np.isnan(value):
                    print(f"  {key}: NaN (calculation failed)")
                else:
                    print(f"  {key}: {value}")
        
        # Print parameter estimates
        if 'model_fit' in workflow.results:
            params = workflow.results['model_fit'].get('params', {})
            print("\nPARAMETER ESTIMATES:")
            print("-" * 30)
            for param, value in params.items():
                if isinstance(value, (int, float)):
                    print(f"  {param}: {value:.4f}")
        
        # Run NCA analysis
        print("\nRunning NCA analysis...")
        workflow.run_nca_analysis()
        
        if 'nca' in workflow.results:
            nca_summary = workflow.results['nca']['summary']
            print("\nNCA SUMMARY (mean values):")
            print("-" * 30)
            for param in ['AUC', 'Cmax', 'Tmax', 'half_life']:
                if param in nca_summary.columns:
                    mean_val = nca_summary[param]['mean']
                    std_val = nca_summary[param]['std']
                    print(f"  {param}: {mean_val:.2f} ± {std_val:.2f}")
        
        return workflow
        
    except Exception as e:
        print(f"\nError in Theophylline analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_metrics_comparison_plot():
    """Create a comprehensive metrics comparison plot"""
    print("\n" + "=" * 50)
    print("CREATING METRICS COMPARISON VISUALIZATION")
    print("=" * 50)
    
    # Run different scenarios
    scenarios = []
    
    # Scenario 1: Low noise, good initial parameters
    print("\nScenario 1: Low noise (10%), good initial parameters")
    param_specs = {
        'CL': {'value': 5.0, 'cv_percent': 20},
        'V': {'value': 30.0, 'cv_percent': 20}
    }
    model = create_pkpy_model('onecomp', param_specs)
    workflow = BasePKWorkflow(model=model, n_subjects=50)
    workflow.prop_err = 0.1
    workflow.generate_virtual_population(
        time_points=np.linspace(0.5, 24, 12),
        dose=100.0
    )
    workflow.run_analysis(create_plots=False)
    scenarios.append(('Low Noise\n(10%)', workflow.results.get('fit_metrics', {})))
    
    # Scenario 2: Medium noise
    print("\nScenario 2: Medium noise (20%)")
    workflow2 = BasePKWorkflow(model=model, n_subjects=50)
    workflow2.prop_err = 0.2
    workflow2.generate_virtual_population(
        time_points=np.linspace(0.5, 24, 12),
        dose=100.0
    )
    workflow2.run_analysis(create_plots=False)
    scenarios.append(('Medium Noise\n(20%)', workflow2.results.get('fit_metrics', {})))
    
    # Scenario 3: High noise
    print("\nScenario 3: High noise (30%)")
    workflow3 = BasePKWorkflow(model=model, n_subjects=50)
    workflow3.prop_err = 0.3
    workflow3.generate_virtual_population(
        time_points=np.linspace(0.5, 24, 12),
        dose=100.0
    )
    workflow3.run_analysis(create_plots=False)
    scenarios.append(('High Noise\n(30%)', workflow3.results.get('fit_metrics', {})))
    
    # Scenario 4: Sparse sampling
    print("\nScenario 4: Sparse sampling (6 time points)")
    workflow4 = BasePKWorkflow(model=model, n_subjects=50)
    workflow4.prop_err = 0.15
    workflow4.generate_virtual_population(
        time_points=np.array([0.5, 2, 4, 8, 16, 24]),
        dose=100.0
    )
    workflow4.run_analysis(create_plots=False)
    scenarios.append(('Sparse\nSampling', workflow4.results.get('fit_metrics', {})))
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = ['R2', 'RMSE', 'AFE', 'AAFE']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        scenario_names = [s[0] for s in scenarios]
        values = [s[1].get(metric, np.nan) for s in scenarios]
        
        # Create bar plot
        bars = ax.bar(range(len(scenarios)), values, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom')
            else:
                ax.text(bar.get_x() + bar.get_width()/2, 0.01,
                       'N/A', ha='center', va='bottom')
        
        # Customize plot
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenario_names)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Across Different Scenarios')
        
        # Add reference lines
        if metric == 'R2':
            ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.95)')
            ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Good (>0.8)')
        elif metric == 'AFE':
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='No bias')
            ax.axhline(y=1.2, color='orange', linestyle='--', alpha=0.5, label='20% threshold')
            ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5)
        elif metric == 'AAFE':
            ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='Excellent (<1.5)')
            ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='Good (<2.0)')
        
        if idx == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle('Comprehensive Metrics Comparison Across Different Analysis Scenarios', fontsize=14)
    plt.tight_layout()
    plt.savefig('comprehensive_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("\nSaved comprehensive metrics comparison plot")
    
    return scenarios

def generate_summary_report(scenarios):
    """Generate a summary report of all metrics"""
    print("\n" + "=" * 50)
    print("COMPREHENSIVE METRICS SUMMARY REPORT")
    print("=" * 50)
    
    # Create summary DataFrame
    summary_data = []
    for scenario_name, metrics in scenarios:
        row = {'Scenario': scenario_name.replace('\n', ' ')}
        row.update(metrics)
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Display summary statistics
    print("\nMETRICS SUMMARY TABLE:")
    print("-" * 80)
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Calculate and display interpretations
    print("\n\nMETRICS INTERPRETATION:")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"\n{row['Scenario']}:")
        
        # R² interpretation
        r2 = row.get('R2', np.nan)
        if not np.isnan(r2):
            if r2 > 0.95:
                print(f"  - R² = {r2:.3f}: Excellent model fit")
            elif r2 > 0.8:
                print(f"  - R² = {r2:.3f}: Good model fit")
            else:
                print(f"  - R² = {r2:.3f}: Poor model fit")
        
        # AFE interpretation
        afe = row.get('AFE', np.nan)
        if not np.isnan(afe):
            if 0.8 < afe < 1.2:
                print(f"  - AFE = {afe:.3f}: Minimal bias (within ±20%)")
            elif afe > 1.2:
                print(f"  - AFE = {afe:.3f}: Model overpredicts by {(afe-1)*100:.1f}%")
            else:
                print(f"  - AFE = {afe:.3f}: Model underpredicts by {(1-afe)*100:.1f}%")
        
        # AAFE interpretation
        aafe = row.get('AAFE', np.nan)
        if not np.isnan(aafe):
            if aafe < 1.5:
                print(f"  - AAFE = {aafe:.3f}: Excellent prediction accuracy")
            elif aafe < 2.0:
                print(f"  - AAFE = {aafe:.3f}: Good prediction accuracy")
            else:
                print(f"  - AAFE = {aafe:.3f}: Poor prediction accuracy")
    
    # Save summary to CSV
    df.to_csv('comprehensive_metrics_summary.csv', index=False)
    print("\n\nSaved summary to 'comprehensive_metrics_summary.csv'")
    
    return df

if __name__ == "__main__":
    # Run all tests
    print("PKPy COMPREHENSIVE METRICS TESTING")
    print("=" * 50)
    print("Testing validation metrics implementation including AFE and AAFE")
    print()
    
    # Test 1: Basic metrics
    test_basic_metrics()
    
    # Test 2: One-compartment workflow
    workflow1 = test_onecomp_workflow()
    
    # Test 3: One-compartment absorption workflow
    workflow2 = test_onecomp_abs_workflow()
    
    # Test 4: Theophylline data
    workflow3 = test_theophylline_data()
    
    # Test 5: Create comprehensive comparison
    scenarios = create_metrics_comparison_plot()
    
    # Generate summary report
    summary_df = generate_summary_report(scenarios)
    
    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)