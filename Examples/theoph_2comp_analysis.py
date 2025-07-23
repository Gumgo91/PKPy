#!/usr/bin/env python
"""
Theophylline 2-compartment analysis with PKPy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from pkpy import BasePKWorkflow, create_pkpy_model
from pkpy.utils import convert_data
import time

print("="*60)
print("Theophylline 2-Compartment Analysis with PKPy")
print("="*60)

# Load data
files = convert_data('Theoph.csv',
                    id_col='Subject',
                    time_col='Time',
                    conc_col='conc')

# 1. First try 2-compartment with absorption
print("\n1. Testing 2-compartment model with absorption")
print("-"*40)

params_2comp_abs = {
    'Ka': {
        'value': 1.5,
        'cv_percent': 50,
        'lower_bound': 0.1,
        'upper_bound': 10.0
    },
    'CL': {
        'value': 3.0,
        'cv_percent': 30,
        'lower_bound': 0.5,
        'upper_bound': 10.0
    },
    'V1': {
        'value': 20.0,
        'cv_percent': 25,
        'lower_bound': 5.0,
        'upper_bound': 50.0
    },
    'Q': {
        'value': 5.0,
        'cv_percent': 40,
        'lower_bound': 0.5,
        'upper_bound': 20.0
    },
    'V2': {
        'value': 30.0,
        'cv_percent': 30,
        'lower_bound': 10.0,
        'upper_bound': 100.0
    }
}

# Run analysis without initial parameters
print("\na) Without initial parameters:")
start_time = time.time()

workflow_no_init = BasePKWorkflow.from_files(
    model_type='twocomp_abs',
    conc_file=files['concentrations'],
    time_file=files['times'],
    demo_file=files['demographics'],
    dose=320
)

results_no_init = workflow_no_init.run_analysis(create_plots=False)
time_no_init = time.time() - start_time

print(f"Computation time: {time_no_init:.2f} seconds")
workflow_no_init.print_summary()

# Run analysis with initial parameters
print("\nb) With initial parameters:")
start_time = time.time()

workflow_with_init = BasePKWorkflow.from_files(
    model_type='twocomp_abs',
    conc_file=files['concentrations'],
    time_file=files['times'],
    demo_file=files['demographics'],
    dose=320,
    initial_params=params_2comp_abs
)

results_with_init = workflow_with_init.run_analysis(create_plots=False)
time_with_init = time.time() - start_time

print(f"Computation time: {time_with_init:.2f} seconds")
workflow_with_init.print_summary()

# 2. Compare with 1-compartment results
print("\n2. Comparison with 1-compartment model")
print("-"*40)

params_1comp_abs = {
    'Ka': {'value': 1.5, 'cv_percent': 30},
    'CL': {'value': 2.5, 'cv_percent': 25},
    'V': {'value': 30.0, 'cv_percent': 20}
}

workflow_1comp = BasePKWorkflow.from_files(
    model_type='onecomp_abs',
    conc_file=files['concentrations'],
    time_file=files['times'],
    demo_file=files['demographics'],
    dose=320,
    initial_params=params_1comp_abs
)

results_1comp = workflow_1comp.run_analysis(create_plots=False)

# AIC comparison
def calculate_aic(results, n_params):
    if 'fit_metrics' in results and 'R2' in results['fit_metrics']:
        n = 12 * 11  # 12 subjects × 11 timepoints
        r2 = results['fit_metrics']['R2']
        # Approximate RSS from R²
        ss_tot = 1000  # arbitrary scale
        ss_res = ss_tot * (1 - r2)
        aic = n * np.log(ss_res/n) + 2 * n_params
        return aic
    return None

aic_1comp = calculate_aic(results_1comp, 3)  # Ka, CL, V
aic_2comp = calculate_aic(results_with_init, 5)  # Ka, CL, V1, Q, V2

print("\nModel Comparison:")
print(f"1-compartment: R² = {results_1comp['fit_metrics']['R2']:.3f}, AIC ≈ {aic_1comp:.1f}")
print(f"2-compartment: R² = {results_with_init['fit_metrics']['R2']:.3f}, AIC ≈ {aic_2comp:.1f}")

if aic_2comp < aic_1comp:
    print("→ 2-compartment model is preferred (lower AIC)")
else:
    print("→ 1-compartment model is preferred (lower AIC)")

# 3. Create comparison plots
print("\n3. Creating visualization")
print("-"*40)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Select 3 subjects for detailed comparison
subjects = [0, 5, 10]

for idx, subj in enumerate(subjects):
    # Top row: 1-compartment fits
    ax = axes[0, idx]
    ax.plot(workflow_1comp.times, workflow_1comp.data['concentrations'][subj], 
            'bo', markersize=8, label='Observed')
    if 'predictions' in results_1comp:
        ax.plot(workflow_1comp.times, results_1comp['predictions'][subj], 
                'b-', linewidth=2, label='1-comp pred')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Concentration (mg/L)')
    ax.set_title(f'1-Comp: Subject {subj+1}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom row: 2-compartment fits
    ax = axes[1, idx]
    ax.plot(workflow_with_init.times, workflow_with_init.data['concentrations'][subj], 
            'ro', markersize=8, label='Observed')
    if 'predictions' in results_with_init:
        ax.plot(workflow_with_init.times, results_with_init['predictions'][subj], 
                'r-', linewidth=2, label='2-comp pred')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Concentration (mg/L)')
    ax.set_title(f'2-Comp: Subject {subj+1}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Theophylline: 1-Compartment vs 2-Compartment Model Fits', fontsize=14)
plt.tight_layout()
plt.savefig('theoph_1comp_vs_2comp.png', dpi=300, bbox_inches='tight')
print("Saved: theoph_1comp_vs_2comp.png")

# 4. Summary table
print("\n4. Summary Comparison Table")
print("-"*40)

summary_data = {
    'Model': ['1-Compartment', '2-Compartment (no init)', '2-Compartment (with init)'],
    'Parameters': [3, 5, 5],
    'R²': [
        results_1comp['fit_metrics']['R2'],
        results_no_init['fit_metrics']['R2'],
        results_with_init['fit_metrics']['R2']
    ],
    'Success Rate': [
        f"{results_1comp['model_fit']['successful_subjects']}/12",
        f"{results_no_init['model_fit']['successful_subjects']}/12",
        f"{results_with_init['model_fit']['successful_subjects']}/12"
    ],
    'Computation Time (s)': [
        'N/A',
        f"{time_no_init:.1f}",
        f"{time_with_init:.1f}"
    ]
}

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# Save detailed results
df_summary.to_csv('theoph_2comp_comparison_summary.csv', index=False)
print("\nSummary saved: theoph_2comp_comparison_summary.csv")

print("\n" + "="*60)
print("Analysis completed!")
print("="*60)