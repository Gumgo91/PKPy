#!/usr/bin/env python
"""
Simple test of 2-compartment model
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from pkpy import create_pkpy_model, BasePKWorkflow

# Set random seed
np.random.seed(42)

print("Testing 2-compartment model...")

# Test 1: Basic model creation and concentration profile
print("\n1. Testing basic 2-compartment model")
params_2comp = {
    "CL": {"value": 5.0, "cv_percent": 25},
    "V1": {"value": 30.0, "cv_percent": 20},
    "Q": {"value": 10.0, "cv_percent": 30},
    "V2": {"value": 50.0, "cv_percent": 25}
}

model_2comp = create_pkpy_model("twocomp", params_2comp)
times = np.linspace(0, 48, 100)
conc_2comp = model_2comp.solve_ode(times, 100, 
                                  {'CL': 5.0, 'V1': 30.0, 'Q': 10.0, 'V2': 50.0})

print(f"  Max concentration: {np.max(conc_2comp):.2f}")
print(f"  Terminal half-life: {0.693 * 30 / 5:.2f} h (approx)")

# Compare with 1-compartment
params_1comp = {
    "CL": {"value": 5.0, "cv_percent": 25},
    "V": {"value": 50.0, "cv_percent": 20}
}

model_1comp = create_pkpy_model("onecomp", params_1comp)
conc_1comp = model_1comp.solve_ode(times, 100, {'CL': 5.0, 'V': 50.0})

# Plot comparison
plt.figure(figsize=(10, 6))
plt.semilogy(times, conc_1comp, 'b-', label='1-Compartment', linewidth=2)
plt.semilogy(times, conc_2comp, 'r-', label='2-Compartment', linewidth=2)
plt.xlabel('Time (h)')
plt.ylabel('Concentration (mg/L)')
plt.title('1-Compartment vs 2-Compartment Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('test_concentration_profiles.png', dpi=150)
print("\n  Saved: test_concentration_profiles.png")

# Test 2: Simple population simulation
print("\n2. Testing population simulation (n=10)")
workflow = BasePKWorkflow(model_2comp, n_subjects=10)

# Generate data
times_sim = np.concatenate([
    np.linspace(0, 2, 5),    # Early sampling
    np.linspace(4, 24, 5)    # Later sampling
])

workflow.generate_virtual_population(
    times_sim, 
    dose=100.0,
    covariate_models={
        'CL': {'CRCL': {'type': 'power', 'coefficient': 0.75}},
        'V1': {'WT': {'type': 'power', 'coefficient': 1.0}}
    }
)

print("  Generated population data")

# Fit model
try:
    workflow.run_model_fitting()
    if workflow.results['model_fit']['success']:
        print("\n  Estimated parameters:")
        for param, value in workflow.results['model_fit']['parameters'].items():
            true_val = params_2comp[param]['value']
            bias = ((value - true_val) / true_val) * 100
            print(f"    {param}: {value:.3f} (true: {true_val}, bias: {bias:+.1f}%)")
except Exception as e:
    print(f"  Error in fitting: {str(e)}")

# Test 3: Compare fitting time
print("\n3. Comparing computation time")
import time

# 1-comp timing
model_1comp_wf = create_pkpy_model("onecomp", params_1comp)
workflow_1comp = BasePKWorkflow(model_1comp_wf, n_subjects=20)
workflow_1comp.generate_virtual_population(np.linspace(0, 24, 10), dose=100.0)

start = time.time()
workflow_1comp.run_model_fitting()
time_1comp = time.time() - start

# 2-comp timing
workflow_2comp = BasePKWorkflow(model_2comp, n_subjects=20)
workflow_2comp.generate_virtual_population(times_sim, dose=100.0)

start = time.time()
workflow_2comp.run_model_fitting()
time_2comp = time.time() - start

print(f"  1-compartment: {time_1comp:.2f} seconds")
print(f"  2-compartment: {time_2comp:.2f} seconds")
print(f"  Ratio: {time_2comp/time_1comp:.1f}x slower")

print("\nTest completed!")