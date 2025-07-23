#!/usr/bin/env python
"""
Test numerical stability improvements for 2-compartment models
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from pkpy import create_pkpy_model
import time

print("Testing numerical stability improvements...")
print("="*50)

# Test 1: Basic 2-compartment with absorption
print("\n1. Testing 2-compartment with absorption model")
print("-"*40)

params = {
    'Ka': {'value': 1.5, 'cv_percent': 30},
    'CL': {'value': 3.0, 'cv_percent': 25},
    'V1': {'value': 20.0, 'cv_percent': 20},
    'Q': {'value': 8.0, 'cv_percent': 30},
    'V2': {'value': 40.0, 'cv_percent': 25}
}

model = create_pkpy_model("twocomp_abs", params)
times = np.linspace(0, 24, 50)

# Test with different parameter combinations
test_params = [
    # Normal parameters
    {'Ka': 1.5, 'CL': 3.0, 'V1': 20.0, 'Q': 8.0, 'V2': 40.0},
    # Stiff system (very different rate constants)
    {'Ka': 10.0, 'CL': 1.0, 'V1': 10.0, 'Q': 20.0, 'V2': 100.0},
    # Near-equal Ka and k
    {'Ka': 0.2, 'CL': 4.0, 'V1': 20.0, 'Q': 5.0, 'V2': 30.0},
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, test_param in enumerate(test_params):
    print(f"\nTest case {idx+1}: {test_param}")
    
    start_time = time.time()
    try:
        conc = model.solve_ode(times, 100, test_param)
        elapsed = time.time() - start_time
        
        print(f"  Success! Time: {elapsed:.3f}s")
        print(f"  Max concentration: {np.max(conc):.2f}")
        print(f"  Terminal concentration: {conc[-1,0]:.4f}")
        
        # Plot
        ax = axes[idx]
        ax.semilogy(times, conc, 'b-', linewidth=2)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Concentration (mg/L)')
        ax.set_title(f'Test Case {idx+1}')
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        print(f"  Failed: {str(e)}")
        axes[idx].text(0.5, 0.5, 'Failed', ha='center', va='center', 
                      transform=axes[idx].transAxes, fontsize=14)

plt.suptitle('2-Compartment with Absorption: Numerical Stability Tests', fontsize=14)
plt.tight_layout()
plt.savefig('numerical_stability_test.png', dpi=150)
print("\nSaved: numerical_stability_test.png")

# Test 2: Compare with 1-compartment
print("\n\n2. Comparing with 1-compartment model")
print("-"*40)

params_1comp = {
    'Ka': {'value': 1.5, 'cv_percent': 30},
    'CL': {'value': 3.0, 'cv_percent': 25},
    'V': {'value': 30.0, 'cv_percent': 20}
}

model_1comp = create_pkpy_model("onecomp_abs", params_1comp)

# Use equivalent parameters
V_total = 20.0 + 40.0  # V1 + V2
params_1comp_equiv = {'Ka': 1.5, 'CL': 3.0, 'V': V_total}

conc_1comp = model_1comp.solve_ode(times, 100, params_1comp_equiv)
conc_2comp = model.solve_ode(times, 100, test_params[0])

plt.figure(figsize=(10, 6))
plt.semilogy(times, conc_1comp, 'b-', label='1-compartment', linewidth=2)
plt.semilogy(times, conc_2comp, 'r--', label='2-compartment', linewidth=2)
plt.xlabel('Time (h)')
plt.ylabel('Concentration (mg/L)')
plt.title('1-Compartment vs 2-Compartment Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('1comp_vs_2comp_stability.png', dpi=150)
print("\nSaved: 1comp_vs_2comp_stability.png")

print("\n" + "="*50)
print("Numerical stability test completed!")
print("Summary:")
print("- Stiff systems now use analytical approximation")
print("- More robust parameter bounds")
print("- Improved convergence with multiple fallback options")