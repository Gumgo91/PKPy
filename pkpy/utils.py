import numpy as np
from scipy import interpolate
from typing import Dict, List, Optional, Tuple

def calculate_nca_parameters(
    times: np.ndarray,
    concentrations: np.ndarray,
    dose: Optional[float] = None
) -> Dict[str, float]:
    """Calculate non-compartmental analysis parameters"""
    # Ensure arrays are 1D and properly aligned
    times = np.asarray(times).flatten()
    concentrations = np.asarray(concentrations).flatten()
    
    if len(times) != len(concentrations):
        raise ValueError(f"Length mismatch: times ({len(times)}) != concentrations ({len(concentrations)})")
    
    # Remove any nan or negative values
    mask = ~(np.isnan(concentrations) | (concentrations < 0))
    times = times[mask]
    concentrations = concentrations[mask]
    
    if len(concentrations) < 3:
        raise ValueError("Need at least 3 valid concentration points for NCA analysis")
    
    # Find Cmax and Tmax
    cmax = np.max(concentrations)
    tmax = times[np.argmax(concentrations)]
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(y=concentrations, x=times)
    
    # Calculate terminal half-life
    # Use last 3 points for terminal slope
    terminal_idx = -3
    # 0이나 음수 값 필터링
    valid_conc = concentrations[terminal_idx:] > 0
    if np.any(valid_conc):
        log_conc = np.log(concentrations[terminal_idx:][valid_conc])
        terminal_times = times[terminal_idx:][valid_conc]
        
        if len(terminal_times) >= 2:  # Need at least 2 points for linear regression
            slope, intercept = np.polyfit(terminal_times, log_conc, 1)
            half_life = -np.log(2) / slope if slope < 0 else np.nan
        else:
            half_life = np.nan
    else:
        half_life = np.nan
    
    # Calculate clearance if dose is provided
    if dose is not None and auc > 0:
        clearance = dose / auc
    else:
        clearance = np.nan
        
    # Calculate additional parameters
    aumc = np.trapz(y=concentrations * times, x=times)
    mrt = aumc / auc if auc > 0 else np.nan
    vss = clearance * mrt if clearance is not None and not np.isnan(mrt) else np.nan
    
    return {
        'Cmax': cmax,
        'Tmax': tmax,
        'AUC': auc,
        'AUMC': aumc,
        'MRT': mrt,
        'half_life': half_life,
        'clearance': clearance,
        'Vss': vss
    }

def check_identifiability(
    model,
    nominal_params: Dict[str, float],
    times: np.ndarray,
    perturbation: float = 0.1
) -> Dict[str, float]:
    """Check parameter identifiability through sensitivity analysis"""
    sensitivities = {}
    correlations = {}
    
    # Calculate nominal profile
    nominal_profile = model.solve_ode(times, 100, nominal_params)
    
    # Calculate sensitivity for each parameter
    for param_name, param_value in nominal_params.items():
        # Perturb parameter
        perturbed_params = nominal_params.copy()
        perturbed_params[param_name] = param_value * (1 + perturbation)
        
        # Calculate perturbed profile
        perturbed_profile = model.solve_ode(times, 100, perturbed_params)
        
        # Calculate sensitivity
        sensitivity = np.mean(
            np.abs(perturbed_profile - nominal_profile) / 
            (perturbation * param_value)
        )
        
        # Calculate correlation
        correlations[param_name] = np.corrcoef(nominal_profile.flatten(), 
                                             perturbed_profile.flatten())[0,1]
        
        sensitivities[param_name] = sensitivity
        
    return {
        'sensitivities': sensitivities,
        'correlations': correlations
    }

def power_analysis(
    model,
    parameters: Dict[str, float],
    n_subjects_range: List[int],
    effect_size: float,
    n_simulations: int = 1000,
    alpha: float = 0.05
) -> Dict[int, float]:
    """Perform power analysis for different sample sizes"""
    powers = {}
    
    for n_subjects in n_subjects_range:
        significant_tests = 0
        
        for _ in range(n_simulations):
            # Simulate control group
            control_data = model.simulate_population(
                n_subjects=n_subjects,
                times=np.linspace(0, 24, 10),
                parameters=parameters
            )
            
            # Simulate treatment group with effect
            treatment_params = parameters.copy()
            treatment_params['CL'] *= (1 + effect_size)
            
            treatment_data = model.simulate_population(
                n_subjects=n_subjects,
                times=np.linspace(0, 24, 10),
                parameters=treatment_params
            )
            
            # Perform statistical test
            control_auc = np.trapz(control_data['concentrations'].mean(axis=0),
                                 control_data['times'])
            treatment_auc = np.trapz(treatment_data['concentrations'].mean(axis=0),
                                   treatment_data['times'])
                                   
            # Simple t-test
            t_stat = (treatment_auc - control_auc) / np.sqrt(
                np.var(control_data['concentrations']) / n_subjects +
                np.var(treatment_data['concentrations']) / n_subjects
            )
            
            if abs(t_stat) > 1.96:  # Approximate critical value for alpha=0.05
                significant_tests += 1
                            
        power = significant_tests / n_simulations
        powers[n_subjects] = power
                    
    return powers

def plot_diagnostics(observed: np.ndarray, predicted: np.ndarray) -> Dict:
    """Generate standard diagnostic plots for PK analysis"""
    import matplotlib.pyplot as plt
    
    # Calculate residuals
    residuals = np.log(observed) - np.log(predicted)
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Observed vs Predicted
    axes[0,0].scatter(predicted, observed)
    min_val = min(predicted.min(), observed.min())
    max_val = max(predicted.max(), observed.max())
    axes[0,0].plot([min_val, max_val], [min_val, max_val], 'k--')
    axes[0,0].set_xlabel('Predicted Concentration')
    axes[0,0].set_ylabel('Observed Concentration')
    axes[0,0].set_title('Observed vs Predicted')
    
    # Residuals vs Predicted
    axes[0,1].scatter(predicted, residuals)
    axes[0,1].axhline(y=0, color='k', linestyle='--')
    axes[0,1].set_xlabel('Predicted Concentration')
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].set_title('Residuals vs Predicted')
    
    # QQ plot of residuals
    from scipy import stats
    stats.probplot(residuals, plot=axes[1,0])
    axes[1,0].set_title('Normal Q-Q Plot')
    
    # Histogram of residuals
    axes[1,1].hist(residuals, bins=20)
    axes[1,1].set_xlabel('Residuals')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Histogram of Residuals')
    
    plt.tight_layout()
    
    # Calculate goodness-of-fit statistics
    metrics = {
        'R2': 1 - np.sum(residuals**2) / np.sum((np.log(observed) - np.mean(np.log(observed)))**2),
        'RMSE': np.sqrt(np.mean(residuals**2)),
        'MAE': np.mean(np.abs(residuals)),
        'bias': np.mean(residuals)
    }
    
    return {'figure': fig, 'metrics': metrics}

def create_vpc_plot(
    observed_data: Dict,
    model,
    n_simulations: int = 1000,
    percentiles: List[float] = [5, 50, 95]
) -> Dict:
    """Create Visual Predictive Check (VPC) plot"""
    import matplotlib.pyplot as plt
    
    # Simulate multiple datasets
    all_simulated = []
    for _ in range(n_simulations):
        sim_data = model.simulate_population(
            n_subjects=len(observed_data['concentrations']),
            times=observed_data['times'],
            parameters=model.parameters
        )
        all_simulated.append(sim_data['concentrations'])
    
    # Calculate percentiles for simulated data
    sim_percentiles = np.percentile(all_simulated, percentiles, axis=0)
    
    # Calculate percentiles for observed data
    obs_percentiles = np.percentile(observed_data['concentrations'], 
                                  percentiles, axis=0)
    
    # Create VPC plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot simulation intervals
    ax.fill_between(observed_data['times'], 
                   sim_percentiles[0], 
                   sim_percentiles[2],
                   alpha=0.2, 
                   label=f'{percentiles[0]}-{percentiles[2]} percentile PI')
    
    # Plot median
    ax.plot(observed_data['times'], 
            sim_percentiles[1], 
            '--', 
            label='Median prediction')
    
    # Plot observed percentiles
    ax.plot(observed_data['times'], 
            obs_percentiles[1], 
            'k-', 
            label='Observed median')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.set_title('Visual Predictive Check')
    ax.legend()
    
    return {'figure': fig, 'simulated_percentiles': sim_percentiles}


def convert_data(
    file: str, 
    output_dir: str = '.',
    id_col: str = 'Subject',
    time_col: str = 'Time',
    conc_col: str = 'conc'
) -> Dict[str, str]:
    """
    Convert longitudinal PK data to PKPy format files
    
    Parameters:
    -----------
    file: str
        csv file path
    output_dir: str
        directory to save output files
    id_col: str
        subject ID column name
    time_col: str
        time column name
    conc_col: str
        concentration column name
        
    Returns:
    --------
    Dict[str, str]
        paths of generated files
    """
    import os
    import pandas as pd
    
    # load data
    data = pd.read_csv(file)
    
    # check required columns
    required_cols = [id_col, time_col, conc_col]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Required columns {required_cols} not found in the data")
    
    # extract and sort unique times
    unique_times = sorted(data[time_col].unique())
    
    # pivot concentration data
    conc_df = data.pivot(
        index=id_col,
        columns=time_col,
        values=conc_col
    ).reset_index()
    
    # rename columns
    conc_df.columns = ['ID'] + [f'Time_{t}h' for t in unique_times]
    
    # identify covariate columns
    covariate_cols = [col for col in data.columns 
                     if col not in [id_col, time_col, conc_col]]
    
    # prepare demographic/covariate data
    if covariate_cols:
        demo_df = data.groupby(id_col)[covariate_cols].first().reset_index()
        demo_df.columns = ['ID'] + covariate_cols
    else:
        demo_df = pd.DataFrame({
            'ID': data[id_col].unique()
        })
    
    # prepare time data
    time_df = pd.DataFrame({'time': unique_times})
    
    # save files
    os.makedirs(output_dir, exist_ok=True)
    
    conc_file = os.path.join(output_dir, 'concentrations.csv')
    time_file = os.path.join(output_dir, 'times.csv')
    demo_file = os.path.join(output_dir, 'demographics.csv')
    
    conc_df.to_csv(conc_file, index=False)
    time_df.to_csv(time_file, index=False)
    demo_df.to_csv(demo_file, index=False)

    # Print column classifications
    print("\nColumn Classifications:")
    print("-" * 50)
    print(f"ID Column: {id_col}")
    print(f"Time Column: {time_col}")
    print(f"Concentration Column: {conc_col}")
    if covariate_cols:
        print("Covariate Columns:")
        for col in covariate_cols:
            print(f"  - {col}")
    
    return {
        'concentrations': conc_file,
        'times': time_file,
        'demographics': demo_file
    }