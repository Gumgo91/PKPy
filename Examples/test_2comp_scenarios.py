#!/usr/bin/env python
"""
Test scenarios including 2-compartment models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from pkpy import create_pkpy_model, BasePKWorkflow

# Set random seed
np.random.seed(42)

# Test scenarios
scenarios = {
    'onecomp': {
        'params': {
            "CL": {"value": 5.0, "cv_percent": 25},
            "V": {"value": 50.0, "cv_percent": 20}
        },
        'covariates': {
            'CL': {'CRCL': {'type': 'power', 'coefficient': 0.75}},
            'V': {'WT': {'type': 'power', 'coefficient': 1.0}}
        },
        'times': np.linspace(0, 24, 13)
    },
    'onecomp_abs': {
        'params': {
            "Ka": {"value": 1.0, "cv_percent": 30},
            "CL": {"value": 5.0, "cv_percent": 25},
            "V": {"value": 50.0, "cv_percent": 20}
        },
        'covariates': {
            'CL': {'CRCL': {'type': 'power', 'coefficient': 0.75}},
            'V': {'WT': {'type': 'power', 'coefficient': 1.0}},
            'Ka': {'AGE': {'type': 'exponential', 'coefficient': -0.02}}
        },
        'times': np.concatenate([np.linspace(0, 2, 8), np.linspace(3, 24, 8)])
    },
    'twocomp': {
        'params': {
            "CL": {"value": 5.0, "cv_percent": 25},
            "V1": {"value": 30.0, "cv_percent": 20},
            "Q": {"value": 10.0, "cv_percent": 30},
            "V2": {"value": 50.0, "cv_percent": 25}
        },
        'covariates': {
            'CL': {'CRCL': {'type': 'power', 'coefficient': 0.75}},
            'V1': {'WT': {'type': 'power', 'coefficient': 1.0}},
            'V2': {'WT': {'type': 'power', 'coefficient': 1.0}}
        },
        'times': np.concatenate([np.linspace(0, 2, 8), np.linspace(3, 8, 5), np.linspace(12, 48, 5)])
    },
    'twocomp_abs': {
        'params': {
            "Ka": {"value": 1.0, "cv_percent": 30},
            "CL": {"value": 5.0, "cv_percent": 25},
            "V1": {"value": 30.0, "cv_percent": 20},
            "Q": {"value": 10.0, "cv_percent": 30},
            "V2": {"value": 50.0, "cv_percent": 25}
        },
        'covariates': {
            'CL': {'CRCL': {'type': 'power', 'coefficient': 0.75}},
            'V1': {'WT': {'type': 'power', 'coefficient': 1.0}},
            'Ka': {'AGE': {'type': 'exponential', 'coefficient': -0.02}}
        },
        'times': np.concatenate([np.linspace(0, 3, 10), np.linspace(4, 10, 5), np.linspace(12, 48, 5)])
    }
}

# Results storage
results_summary = []

print("Testing PKPy scenarios with 1-comp and 2-comp models...")
print("="*60)

for scenario_name, scenario_config in scenarios.items():
    print(f"\nScenario: {scenario_name}")
    print("-"*40)
    
    # Create model
    model = create_pkpy_model(scenario_name, scenario_config['params'])
    workflow = BasePKWorkflow(model, n_subjects=20)
    
    # Generate population
    workflow.generate_virtual_population(
        scenario_config['times'],
        dose=100.0,
        demographic_covariates={
            'WT': ('truncnorm', 70, 15, 40, 120),
            'AGE': ('truncnorm', 45, 15, 20, 80),
            'CRCL': ('truncnorm', 100, 25, 50, 150)
        },
        covariate_models=scenario_config['covariates']
    )
    
    # Run analysis
    try:
        results = workflow.run_analysis(create_plots=False)
        
        if results['model_fit']['success']:
            # Calculate metrics
            est_params = results['model_fit']['parameters']
            true_params = {k: v['value'] for k, v in scenario_config['params'].items()}
            
            biases = {}
            for param in true_params:
                bias = ((est_params[param] - true_params[param]) / true_params[param]) * 100
                biases[param] = bias
            
            # Covariate detection
            cov_detected = 0
            if results.get('covariate_analysis'):
                selected = results['covariate_analysis'].get('selected_relationships', [])
                cov_detected = len(selected)
            
            # Store results
            result_row = {
                'Scenario': scenario_name,
                'Success': True,
                'R2': results['fit_metrics']['R2'],
                'Successful_Subjects': results['model_fit']['successful_subjects'],
                'Total_Subjects': results['model_fit']['total_subjects'],
                'Covariates_Detected': cov_detected,
                **{f'{param}_bias': biases[param] for param in biases}
            }
            results_summary.append(result_row)
            
            # Print summary
            print(f"  Success: {results['model_fit']['successful_subjects']}/{results['model_fit']['total_subjects']} subjects")
            print(f"  R²: {results['fit_metrics']['R2']:.3f}")
            print(f"  Parameter biases:")
            for param, bias in biases.items():
                print(f"    {param}: {bias:+.1f}%")
            print(f"  Covariates detected: {cov_detected}")
            
    except Exception as e:
        print(f"  Error: {str(e)}")
        results_summary.append({
            'Scenario': scenario_name,
            'Success': False,
            'Error': str(e)
        })

# Create summary DataFrame
df_summary = pd.DataFrame(results_summary)

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. R² by scenario
scenarios_success = df_summary[df_summary['Success']==True]
ax1.bar(scenarios_success['Scenario'], scenarios_success['R2'])
ax1.set_xlabel('Scenario')
ax1.set_ylabel('R²')
ax1.set_title('Model Fit Quality by Scenario')
ax1.set_ylim(0.9, 1.0)
ax1.grid(True, alpha=0.3)
for i, (x, y) in enumerate(zip(scenarios_success['Scenario'], scenarios_success['R2'])):
    ax1.text(i, y + 0.001, f'{y:.3f}', ha='center', va='bottom')

# 2. Parameter bias heatmap
# Extract bias columns
bias_cols = [col for col in df_summary.columns if col.endswith('_bias')]
if bias_cols:
    bias_data = scenarios_success[['Scenario'] + bias_cols].set_index('Scenario')
    bias_data.columns = [col.replace('_bias', '') for col in bias_data.columns]
    
    im = ax2.imshow(bias_data.T.values, cmap='RdBu_r', vmin=-20, vmax=20, aspect='auto')
    ax2.set_xticks(range(len(bias_data.index)))
    ax2.set_xticklabels(bias_data.index, rotation=45)
    ax2.set_yticks(range(len(bias_data.columns)))
    ax2.set_yticklabels(bias_data.columns)
    ax2.set_title('Parameter Bias Heatmap (%)')
    
    # Add values
    for i in range(len(bias_data.columns)):
        for j in range(len(bias_data.index)):
            val = bias_data.iloc[j, i]
            if not np.isnan(val):
                ax2.text(j, i, f'{val:.0f}', ha='center', va='center',
                        color='white' if abs(val) > 10 else 'black')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Bias (%)')

# 3. Success rate
ax3.bar(scenarios_success['Scenario'], 
        scenarios_success['Successful_Subjects'] / scenarios_success['Total_Subjects'] * 100)
ax3.set_xlabel('Scenario')
ax3.set_ylabel('Success Rate (%)')
ax3.set_title('Individual Fitting Success Rate')
ax3.set_ylim(0, 105)
ax3.grid(True, alpha=0.3)

# 4. Model complexity vs performance
complexity = {'onecomp': 2, 'onecomp_abs': 3, 'twocomp': 4, 'twocomp_abs': 5}
scenarios_success['n_params'] = scenarios_success['Scenario'].map(complexity)
ax4.scatter(scenarios_success['n_params'], scenarios_success['R2'], s=200)
for _, row in scenarios_success.iterrows():
    ax4.annotate(row['Scenario'], (row['n_params'], row['R2']), 
                xytext=(5, 5), textcoords='offset points')
ax4.set_xlabel('Number of Parameters')
ax4.set_ylabel('R²')
ax4.set_title('Model Complexity vs Fit Quality')
ax4.grid(True, alpha=0.3)

plt.suptitle('PKPy Scenario Test Results', fontsize=16)
plt.tight_layout()
plt.savefig('scenario_test_results.png', dpi=300, bbox_inches='tight')
print(f"\nFigure saved: scenario_test_results.png")

# Save detailed results
df_summary.to_csv('scenario_test_summary.csv', index=False)
print(f"Summary saved: scenario_test_summary.csv")

print("\n" + "="*60)
print("OVERALL SUMMARY")
print("="*60)
print(f"Scenarios tested: {len(scenarios)}")
print(f"Successful scenarios: {df_summary['Success'].sum()}")
print(f"Average R² (successful): {scenarios_success['R2'].mean():.3f}")
print(f"Average success rate: {(scenarios_success['Successful_Subjects'].sum() / scenarios_success['Total_Subjects'].sum() * 100):.1f}%")