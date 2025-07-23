#!/usr/bin/env python3
"""
Test script for validation metrics in PKPy
Tests the newly added AFE and AAFE metrics along with other validation metrics
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkpy.utils import calculate_validation_metrics

def test_validation_metrics():
    """Test the validation metrics with known inputs and expected outputs"""
    
    print("Testing validation metrics...")
    print("=" * 60)
    
    # Test Case 1: Perfect predictions
    print("\nTest Case 1: Perfect predictions")
    observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    metrics = calculate_validation_metrics(observed, predicted)
    print(f"Observed: {observed}")
    print(f"Predicted: {predicted}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Verify expected values for perfect predictions
    assert abs(metrics['R2'] - 1.0) < 1e-10, "R2 should be 1.0 for perfect predictions"
    assert abs(metrics['RMSE']) < 1e-10, "RMSE should be 0.0 for perfect predictions"
    assert abs(metrics['AFE'] - 1.0) < 1e-10, "AFE should be 1.0 for perfect predictions"
    assert abs(metrics['AAFE'] - 1.0) < 1e-10, "AAFE should be 1.0 for perfect predictions"
    
    # Test Case 2: Systematic overprediction by factor of 2
    print("\n\nTest Case 2: Systematic overprediction by factor of 2")
    observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    
    metrics = calculate_validation_metrics(observed, predicted)
    print(f"Observed: {observed}")
    print(f"Predicted: {predicted}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # AFE should be exactly 2.0 for systematic 2-fold overprediction
    assert abs(metrics['AFE'] - 2.0) < 1e-10, f"AFE should be 2.0, got {metrics['AFE']}"
    # AAFE should also be 2.0 since all errors are in the same direction
    assert abs(metrics['AAFE'] - 2.0) < 1e-10, f"AAFE should be 2.0, got {metrics['AAFE']}"
    
    # Test Case 3: Systematic underprediction by factor of 2
    print("\n\nTest Case 3: Systematic underprediction by factor of 2")
    observed = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    predicted = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    metrics = calculate_validation_metrics(observed, predicted)
    print(f"Observed: {observed}")
    print(f"Predicted: {predicted}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # AFE should be 0.5 for systematic 2-fold underprediction
    assert abs(metrics['AFE'] - 0.5) < 1e-10, f"AFE should be 0.5, got {metrics['AFE']}"
    # AAFE should be 2.0 (since 10^|log10(0.5)| = 10^0.301 = 2.0)
    assert abs(metrics['AAFE'] - 2.0) < 1e-10, f"AAFE should be 2.0, got {metrics['AAFE']}"
    
    # Test Case 4: Mixed over and underprediction
    print("\n\nTest Case 4: Mixed over and underprediction")
    observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted = np.array([2.0, 1.0, 6.0, 2.0, 10.0])
    
    metrics = calculate_validation_metrics(observed, predicted)
    print(f"Observed: {observed}")
    print(f"Predicted: {predicted}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # AFE should be close to 1.0 if over and underpredictions balance out
    print(f"\nAFE interpretation: {metrics['AFE']:.2f} means average prediction is {metrics['AFE']:.2f}-fold of observed")
    print(f"AAFE interpretation: {metrics['AAFE']:.2f} means predictions are within {metrics['AAFE']:.2f}-fold of observed on average")
    
    # Test Case 5: Realistic PK data with noise
    print("\n\nTest Case 5: Realistic PK data with noise")
    np.random.seed(42)
    time = np.linspace(0, 24, 10)
    true_conc = 10 * np.exp(-0.2 * time)  # Simple exponential decay
    noise = np.random.normal(1, 0.1, len(time))  # 10% CV noise
    observed = true_conc * noise
    predicted = true_conc  # Predictions are the true values
    
    metrics = calculate_validation_metrics(observed, predicted)
    print(f"Time points: {time}")
    print(f"Observed (with noise): {observed}")
    print(f"Predicted (true values): {predicted}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test Case 6: Handling zero and negative values
    print("\n\nTest Case 6: Handling zero and negative values")
    observed = np.array([1.0, 0.0, -1.0, 2.0, 3.0])
    predicted = np.array([1.1, 0.1, 0.5, 2.2, 2.8])
    
    metrics = calculate_validation_metrics(observed, predicted)
    print(f"Observed (with invalid values): {observed}")
    print(f"Predicted: {predicted}")
    print(f"Valid data points used: {np.sum((observed > 0) & (predicted > 0))}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("\nSummary of AFE and AAFE interpretation:")
    print("- AFE = 1.0: No systematic bias")
    print("- AFE > 1.0: Systematic overprediction")
    print("- AFE < 1.0: Systematic underprediction")
    print("- AAFE = 1.0: Perfect predictions")
    print("- AAFE = 2.0: Predictions within 2-fold of observed on average")
    print("- AAFE is always >= 1.0 and >= AFE")

if __name__ == "__main__":
    test_validation_metrics()