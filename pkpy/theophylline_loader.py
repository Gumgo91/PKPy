"""
Fixed Theophylline loader that properly handles sparse data
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

def load_theophylline_data(conc_file: str, times_file: str, demo_file: str) -> Dict:
    """
    Load and preprocess Theophylline data for PKPy analysis
    This fixed version properly handles the sparse data format.
    """
    
    # Load data files
    conc_df = pd.read_csv(conc_file)
    times_df = pd.read_csv(times_file)
    demo_df = pd.read_csv(demo_file)
    
    # Extract time points from column names
    time_cols = [col for col in conc_df.columns if col.startswith('Time_')]
    times = np.array([float(col.replace('Time_', '').replace('h', '')) for col in time_cols])
    
    # Calculate actual doses (mg) = Dose (mg/kg) * Weight (kg)
    demo_df['ActualDose'] = demo_df['Dose'] * demo_df['Wt']
    
    # Get individual data for each subject
    n_subjects = len(conc_df)
    individual_times = []
    individual_concentrations = []
    
    for i in range(n_subjects):
        # Get all concentration values for this subject
        subj_conc = conc_df.iloc[i, 1:].values  # Skip ID column
        
        # Find non-NaN values
        valid_idx = ~np.isnan(subj_conc)
        valid_times = times[valid_idx]
        valid_conc = subj_conc[valid_idx]
        
        # Store individual data
        individual_times.append(valid_times)
        individual_concentrations.append(valid_conc)
    
    # Find common time points that most subjects have
    # Count how many subjects have data at each time point
    time_counts = {}
    for i in range(n_subjects):
        for t in individual_times[i]:
            time_counts[t] = time_counts.get(t, 0) + 1
    
    # Select times that at least 10 subjects have (out of 12)
    common_times = []
    for t, count in sorted(time_counts.items()):
        if count >= 10:
            common_times.append(t)
    
    common_times = np.array(common_times)
    
    # If we don't have enough common times, use the most frequent ones
    if len(common_times) < 8:
        sorted_times = sorted(time_counts.items(), key=lambda x: x[1], reverse=True)
        common_times = np.array([t for t, _ in sorted_times[:11]])  # Take top 11 times
        common_times = np.sort(common_times)
    
    # Create concentration matrix for common times
    n_common_times = len(common_times)
    conc_matrix = np.full((n_subjects, n_common_times), np.nan)
    
    for i in range(n_subjects):
        for j, t in enumerate(common_times):
            # Find if subject has data at this time (within 0.01 hour tolerance)
            time_idx = np.where(np.abs(individual_times[i] - t) < 0.01)[0]
            if len(time_idx) > 0:
                conc_matrix[i, j] = individual_concentrations[i][time_idx[0]]
    
    # Also keep the full data
    full_conc_matrix = conc_df.iloc[:, 1:].values
    
    return {
        'concentrations': conc_matrix,
        'times': common_times,
        'demographics': demo_df,
        'doses': demo_df['ActualDose'].values,
        'individual_times': individual_times,
        'individual_concentrations': individual_concentrations,
        'dose_per_kg': demo_df['Dose'].values,
        'weights': demo_df['Wt'].values,
        'full_concentrations': full_conc_matrix,
        'full_times': times
    }

def prepare_for_workflow(data: Dict, min_points: int = 5) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    """
    Prepare Theophylline data for BasePKWorkflow
    This version properly filters subjects based on valid data points.
    """
    
    # Count valid points for each subject
    valid_subjects = []
    
    for i in range(len(data['concentrations'])):
        n_valid = np.sum(~np.isnan(data['concentrations'][i]))
        if n_valid >= min_points:
            valid_subjects.append(i)
    
    valid_subjects = np.array(valid_subjects, dtype=int)
    
    print(f"Keeping {len(valid_subjects)} subjects with at least {min_points} valid data points")
    
    if len(valid_subjects) == 0:
        # If no subjects meet the criteria with common times, 
        # return individual data formatted appropriately
        print("Warning: No subjects have enough common time points. Using individual data approach.")
        
        # Return empty arrays that won't cause errors
        return (
            data['times'], 
            data['concentrations'], 
            data['demographics'], 
            np.median(data['doses'])
        )
    
    # Filter data
    concentrations = data['concentrations'][valid_subjects]
    demographics = data['demographics'].iloc[valid_subjects].reset_index(drop=True)
    doses = data['doses'][valid_subjects]
    
    # Use median dose as the population dose
    median_dose = np.median(doses)
    
    return data['times'], concentrations, demographics, median_dose

def convert_to_long_format(data: Dict) -> pd.DataFrame:
    """
    Convert Theophylline data to long format for alternative analysis
    
    Parameters:
    -----------
    data: Dict
        Output from load_theophylline_data
        
    Returns:
    --------
    DataFrame with columns: ID, TIME, CONC, DOSE, WT
    """
    
    long_data = []
    
    for i in range(len(data['individual_times'])):
        subj_id = i + 1
        times = data['individual_times'][i]
        concs = data['individual_concentrations'][i]
        dose = data['doses'][i]
        weight = data['weights'][i]
        
        for t, c in zip(times, concs):
            long_data.append({
                'ID': subj_id,
                'TIME': t,
                'CONC': c,
                'DOSE': dose,
                'WT': weight,
                'DOSE_PER_KG': data['dose_per_kg'][i]
            })
    
    return pd.DataFrame(long_data)