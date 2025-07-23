#!/usr/bin/env python
"""
Simple test of 2-compartment with absorption for Theophylline
"""
import numpy as np
import sys
sys.path.append('..')

from pkpy import BasePKWorkflow
from pkpy.utils import convert_data
import time

print("Testing improved 2-comp model with Theophylline data...")
print("="*60)

# Load data
files = convert_data('Theoph.csv',
                    id_col='Subject',
                    time_col='Time',
                    conc_col='conc')

# Set more conservative initial parameters
params_2comp_abs = {
    'Ka': {'value': 1.5, 'cv_percent': 30, 'lower_bound': 0.1, 'upper_bound': 5.0},
    'CL': {'value': 2.8, 'cv_percent': 25, 'lower_bound': 0.5, 'upper_bound': 10.0},
    'V1': {'value': 25.0, 'cv_percent': 20, 'lower_bound': 5.0, 'upper_bound': 50.0},
    'Q': {'value': 5.0, 'cv_percent': 30, 'lower_bound': 1.0, 'upper_bound': 20.0},
    'V2': {'value': 35.0, 'cv_percent': 25, 'lower_bound': 10.0, 'upper_bound': 100.0}
}

print("\n1. Testing with initial parameters")
print("-"*40)

start_time = time.time()

workflow = BasePKWorkflow.from_files(
    model_type='twocomp_abs',
    conc_file=files['concentrations'],
    time_file=files['times'],
    demo_file=files['demographics'],
    dose=320,
    initial_params=params_2comp_abs
)

# Just try fitting without full analysis
print("Running model fitting...")
try:
    workflow.run_model_fitting()
    time_elapsed = time.time() - start_time
    
    if workflow.results['model_fit']['success']:
        print(f"\nSuccess! Computation time: {time_elapsed:.1f} seconds")
        print(f"Successful subjects: {workflow.results['model_fit']['successful_subjects']}/12")
        
        print("\nParameter estimates:")
        for param, value in workflow.results['model_fit']['parameters'].items():
            print(f"  {param}: {value:.3f}")
    else:
        print("Fitting failed")
        
except Exception as e:
    print(f"Error: {str(e)}")

# Compare with 1-compartment
print("\n\n2. Comparing with 1-compartment model")
print("-"*40)

params_1comp_abs = {
    'Ka': {'value': 1.5, 'cv_percent': 30},
    'CL': {'value': 2.8, 'cv_percent': 25},
    'V': {'value': 32.0, 'cv_percent': 20}
}

start_time = time.time()

workflow_1comp = BasePKWorkflow.from_files(
    model_type='onecomp_abs',
    conc_file=files['concentrations'],
    time_file=files['times'],
    demo_file=files['demographics'],
    dose=320,
    initial_params=params_1comp_abs
)

workflow_1comp.run_model_fitting()
time_1comp = time.time() - start_time

print(f"\nComputation time: {time_1comp:.1f} seconds")
print(f"Successful subjects: {workflow_1comp.results['model_fit']['successful_subjects']}/12")

print("\nParameter estimates:")
for param, value in workflow_1comp.results['model_fit']['parameters'].items():
    print(f"  {param}: {value:.3f}")

print("\n" + "="*60)
print("Test completed!")