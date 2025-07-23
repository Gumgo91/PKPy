#!/usr/bin/env python
"""
Compare 1-compartment and 2-compartment models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import os
import sys
sys.path.append('..')

from pkpy import create_pkpy_model, BasePKWorkflow
from pkpy.simulation import SimulationEngine

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters for both models
params_1comp = {
    "CL": {"value": 5.0, "cv_percent": 25, "lower_bound": 0.1},
    "V": {"value": 50.0, "cv_percent": 20, "lower_bound": 1.0}
}

params_2comp = {
    "CL": {"value": 5.0, "cv_percent": 25, "lower_bound": 0.1},
    "V1": {"value": 30.0, "cv_percent": 20, "lower_bound": 1.0},
    "Q": {"value": 10.0, "cv_percent": 30, "lower_bound": 0.1},
    "V2": {"value": 50.0, "cv_percent": 25, "lower_bound": 1.0}
}

# Define covariate models
covariate_models_1comp = {
    'CL': {'CRCL': {'type': 'power', 'coefficient': 0.75}},
    'V': {'WT': {'type': 'power', 'coefficient': 1.0}}
}

covariate_models_2comp = {
    'CL': {'CRCL': {'type': 'power', 'coefficient': 0.75}},
    'V1': {'WT': {'type': 'power', 'coefficient': 1.0}},
    'V2': {'WT': {'type': 'power', 'coefficient': 1.0}}
}

# Sample sizes to test
sample_sizes = [20, 50]  # Reduced for faster execution
n_simulations = 5  # Number of simulations per condition

# Results storage
results = {
    'model_type': [],
    'n_subjects': [],
    'simulation': [],
    'param_name': [],
    'true_value': [],
    'estimated_value': [],
    'bias_percent': [],
    'rmse': [],
    'r2': [],
    'computation_time': [],
    'covariate_detected': []
}

def calculate_bias_percent(true_val, est_val):
    """Calculate percentage bias"""
    return ((est_val - true_val) / true_val) * 100

def run_simulation(model_type, params, covariate_models, n_subjects, sim_num):
    """Run a single simulation"""
    print(f"  Simulation {sim_num+1}/{n_simulations} for {model_type} with n={n_subjects}")
    
    start_time = time()
    
    # Create model and workflow
    model = create_pkpy_model(model_type, params)
    workflow = BasePKWorkflow(model, n_subjects=n_subjects)
    
    # Generate time points (longer for 2-comp to capture distribution phase)
    if model_type == "twocomp":
        times = np.concatenate([
            np.linspace(0, 2, 10),    # Dense early sampling
            np.linspace(3, 8, 6),     # Medium density
            np.linspace(10, 48, 8)    # Sparse later sampling
        ])
    else:
        times = np.linspace(0, 24, 13)
    
    # Generate virtual population
    workflow.generate_virtual_population(
        times, 
        dose=100.0,
        demographic_covariates={
            'WT': ('truncnorm', 70, 15, 40, 120),
            'AGE': ('truncnorm', 40, 15, 20, 80),
            'CRCL': ('truncnorm', 100, 25, 50, 150)
        },
        covariate_models=covariate_models
    )
    
    # Run analysis
    try:
        results_sim = workflow.run_analysis(create_plots=False)
        computation_time = time() - start_time
        
        # Extract results
        if 'model_fit' in results_sim and results_sim['model_fit']['success']:
            estimated_params = results_sim['model_fit']['parameters']
            true_params = workflow.data['individual_parameters'].mean()
            
            # Record parameter estimates
            for param_name in params.keys():
                # Map parameter names for comparison
                true_name = param_name
                if model_type == "twocomp" and param_name == "V1":
                    compare_name = "V"  # Compare V1 of 2-comp with V of 1-comp
                elif model_type == "onecomp" and param_name == "V":
                    compare_name = "V1"  # For consistency
                else:
                    compare_name = param_name
                
                true_val = true_params[param_name]
                est_val = estimated_params[param_name]
                
                results['model_type'].append(model_type)
                results['n_subjects'].append(n_subjects)
                results['simulation'].append(sim_num)
                results['param_name'].append(compare_name)
                results['true_value'].append(true_val)
                results['estimated_value'].append(est_val)
                results['bias_percent'].append(calculate_bias_percent(true_val, est_val))
                results['rmse'].append(np.sqrt((true_val - est_val)**2))
                results['r2'].append(results_sim['fit_metrics'].get('R2', np.nan))
                results['computation_time'].append(computation_time)
                
                # Check covariate detection
                cov_detected = False
                if 'covariate_analysis' in results_sim and results_sim['covariate_analysis']:
                    selected_rels = results_sim['covariate_analysis'].get('selected_relationships', [])
                    for rel in selected_rels:
                        if rel.parameter == param_name:
                            cov_detected = True
                            break
                results['covariate_detected'].append(cov_detected)
        else:
            print(f"    Warning: Fitting failed for {model_type} with n={n_subjects}")
            
    except Exception as e:
        print(f"    Error in simulation: {str(e)}")
        
# Run simulations
print("Starting model comparison simulations...")
for n_subjects in sample_sizes:
    for sim_num in range(n_simulations):
        # 1-compartment model
        run_simulation("onecomp", params_1comp, covariate_models_1comp, n_subjects, sim_num)
        
        # 2-compartment model
        run_simulation("twocomp", params_2comp, covariate_models_2comp, n_subjects, sim_num)

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Create visualizations
print("\nCreating visualizations...")

# Figure 1: Concentration-time profiles comparison
fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
fig1.suptitle('Concentration-Time Profiles: 1-Comp vs 2-Comp Models', fontsize=14)

for idx, n in enumerate(sample_sizes):
    ax = axes[idx]
    
    # Generate example profiles
    times_plot = np.linspace(0, 48, 200)
    
    # 1-comp profile
    model_1comp = create_pkpy_model("onecomp", params_1comp)
    conc_1comp = model_1comp.solve_ode(times_plot, 100, 
                                       {'CL': 5.0, 'V': 50.0})
    
    # 2-comp profile
    model_2comp = create_pkpy_model("twocomp", params_2comp)
    conc_2comp = model_2comp.solve_ode(times_plot, 100,
                                       {'CL': 5.0, 'V1': 30.0, 'Q': 10.0, 'V2': 50.0})
    
    ax.semilogy(times_plot, conc_1comp, 'b-', label='1-Comp', linewidth=2)
    ax.semilogy(times_plot, conc_2comp, 'r-', label='2-Comp', linewidth=2)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Concentration (mg/L)')
    ax.set_title(f'n = {n} subjects')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('figure1_concentration_profiles.png', dpi=300, bbox_inches='tight')

# Figure 2: Parameter estimation accuracy
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
fig2.suptitle('Parameter Estimation Accuracy', fontsize=14)

# Bias by parameter
ax = axes[0, 0]
param_order = ['CL', 'V1', 'Q', 'V2']
df_bias = df_results.groupby(['model_type', 'param_name'])['bias_percent'].agg(['mean', 'std'])
df_bias = df_bias.reindex(param_order, level=1)

x = np.arange(len(param_order))
width = 0.35

for i, model in enumerate(['onecomp', 'twocomp']):
    model_data = df_bias.xs(model, level=0) if model in df_bias.index.get_level_values(0) else pd.DataFrame()
    if not model_data.empty:
        positions = x + (i - 0.5) * width
        ax.bar(positions, model_data['mean'], width, 
               yerr=model_data['std'], capsize=5,
               label=f'{model}', alpha=0.8)

ax.set_xlabel('Parameter')
ax.set_ylabel('Bias (%)')
ax.set_title('Parameter Bias by Model Type')
ax.set_xticks(x)
ax.set_xticklabels(param_order)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

# Bias by sample size
ax = axes[0, 1]
for model in ['onecomp', 'twocomp']:
    model_data = df_results[df_results['model_type'] == model]
    bias_by_n = model_data.groupby('n_subjects')['bias_percent'].agg(['mean', 'std'])
    ax.errorbar(bias_by_n.index, bias_by_n['mean'], yerr=bias_by_n['std'],
                marker='o', label=model, capsize=5, linewidth=2)

ax.set_xlabel('Sample Size')
ax.set_ylabel('Mean Bias (%)')
ax.set_title('Bias vs Sample Size')
ax.legend()
ax.grid(True, alpha=0.3)

# RMSE comparison
ax = axes[1, 0]
rmse_data = df_results.groupby(['model_type', 'param_name'])['rmse'].mean().unstack()
rmse_data.plot(kind='bar', ax=ax)
ax.set_xlabel('Model Type')
ax.set_ylabel('RMSE')
ax.set_title('Root Mean Square Error by Parameter')
ax.legend(title='Parameter')
ax.grid(True, alpha=0.3)

# Precision (CV%) by sample size
ax = axes[1, 1]
for model in ['onecomp', 'twocomp']:
    model_data = df_results[df_results['model_type'] == model]
    cv_data = []
    for n in sample_sizes:
        n_data = model_data[model_data['n_subjects'] == n]
        cv = (n_data.groupby('param_name')['estimated_value'].std() / 
              n_data.groupby('param_name')['estimated_value'].mean() * 100).mean()
        cv_data.append(cv)
    ax.plot(sample_sizes, cv_data, marker='o', label=model, linewidth=2)

ax.set_xlabel('Sample Size')
ax.set_ylabel('Average CV (%)')
ax.set_title('Precision vs Sample Size')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_parameter_accuracy.png', dpi=300, bbox_inches='tight')

# Figure 3: Model fit comparison
fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Model Fit Quality Comparison', fontsize=14)

# R² distribution
ax = axes[0]
r2_data = df_results.groupby(['model_type', 'n_subjects'])['r2'].apply(list)
positions = []
labels = []
data_to_plot = []

for i, n in enumerate(sample_sizes):
    for j, model in enumerate(['onecomp', 'twocomp']):
        if (model, n) in r2_data.index:
            positions.append(i * 3 + j)
            labels.append(f'{model}\nn={n}')
            data_to_plot.append([x for x in r2_data[model, n] if not np.isnan(x)])

ax.boxplot(data_to_plot, positions=positions, widths=0.8)
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45)
ax.set_ylabel('R²')
ax.set_title('Model Fit (R²) Distribution')
ax.grid(True, alpha=0.3)

# Computation time
ax = axes[1]
time_data = df_results.groupby(['model_type', 'n_subjects'])['computation_time'].mean()
time_data.unstack().plot(kind='bar', ax=ax)
ax.set_xlabel('Model Type')
ax.set_ylabel('Computation Time (seconds)')
ax.set_title('Average Computation Time')
ax.legend(title='Sample Size')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure3_model_fit_comparison.png', dpi=300, bbox_inches='tight')

# Figure 4: Covariate detection rates
fig4, ax = plt.subplots(figsize=(10, 6))
fig4.suptitle('Covariate Detection Rates', fontsize=14)

cov_detection = df_results.groupby(['model_type', 'n_subjects', 'param_name'])['covariate_detected'].mean() * 100
cov_detection = cov_detection.unstack(level=[0, 1])

cov_detection.plot(kind='bar', ax=ax)
ax.set_xlabel('Parameter')
ax.set_ylabel('Detection Rate (%)')
ax.set_title('Covariate Relationship Detection Success')
ax.legend(title='Model Type / Sample Size', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure4_covariate_detection.png', dpi=300, bbox_inches='tight')

# Save summary statistics
print("\nSaving summary statistics...")
summary = df_results.groupby(['model_type', 'n_subjects', 'param_name']).agg({
    'bias_percent': ['mean', 'std'],
    'rmse': ['mean', 'std'],
    'r2': ['mean', 'std'],
    'computation_time': ['mean', 'std'],
    'covariate_detected': 'mean'
}).round(3)

summary.to_csv('model_comparison_summary.csv')

# Print summary
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

print("\nAverage Bias by Model Type:")
print(df_results.groupby('model_type')['bias_percent'].agg(['mean', 'std']).round(2))

print("\nAverage R² by Model Type:")
print(df_results.groupby('model_type')['r2'].agg(['mean', 'std']).round(3))

print("\nAverage Computation Time by Model Type:")
print(df_results.groupby('model_type')['computation_time'].agg(['mean', 'std']).round(2))

print("\nCovariate Detection Rate by Model Type:")
cov_summary = df_results.groupby('model_type')['covariate_detected'].mean() * 100
print(f"1-Compartment: {cov_summary.get('onecomp', 0):.1f}%")
print(f"2-Compartment: {cov_summary.get('twocomp', 0):.1f}%")

print("\nFigures saved:")
print("- figure1_concentration_profiles.png")
print("- figure2_parameter_accuracy.png")
print("- figure3_model_fit_comparison.png")
print("- figure4_covariate_detection.png")
print("- model_comparison_summary.csv")