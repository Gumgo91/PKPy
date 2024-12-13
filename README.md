# PKPy: A Python-Based Framework for Automated Population Pharmacokinetic Analysis

PKPy is an open-source Python framework designed to streamline population pharmacokinetic analysis and modeling. The framework combines user accessibility with computational performance to provide an efficient platform for PK/PD modeling.

## Key Features

- Multiple compartment models (1-compartment, 2-compartment, absorption models)
- Automated non-compartmental analysis (NCA)
- Covariate effect analysis
- Visual predictive checks (VPC)
- Diagnostic plot generation
- Simulation engine
- Parameter estimation
- Robust error handling

## Installation

To install PKPy, you can use pip:
```
!pip install git+https://github.com/gumgo91/pkpy.git
```

## Core Modules

- `models.py`: PK model definitions
- `workflow.py`: Analysis workflow management
- `covariate_analysis.py`: Covariate analysis tools
- `simulation.py`: Simulation engine
- `fitting.py`: Parameter estimation
- `utils.py`: Utility functions

## Features in Detail

### Compartment Models
- One-compartment model
- One-compartment model with absorption
- Two-compartment model
- Custom model support

### Non-Compartmental Analysis
- AUC calculation
- Terminal half-life estimation
- Clearance and volume estimation
- Multiple NCA metrics
- Robust error handling

### Covariate Analysis
- Automated covariate screening
- Multiple relationship types (linear, power, exponential)
- Forward selection algorithm
- Statistical significance testing
- Diagnostic plots

### Simulation Capabilities
- Virtual population generation
- Parameter variability
- Covariate effects
- Residual error models
- Trial design optimization

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- Pandas ≥ 1.3.0
- Matplotlib ≥ 3.4.0
- Seaborn ≥ 0.11.0
- Scikit-learn ≥ 0.24.0
- Numba ≥ 0.54.0

## Contact

- Email: hskong@snu.ac.kr
- Issue Tracker: https://github.com/gumgo91/PKPy/issues
