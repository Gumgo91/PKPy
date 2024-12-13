import numpy as np
from typing import Dict, List, Optional
from .models import CompartmentModel
import pandas as pd
from scipy import stats

class SimulationEngine:
    """Engine for PK model simulation with improved error structure"""
    def __init__(self, model: CompartmentModel):
        self.model = model
        
    def generate_individual_parameters(
        self, 
        parameters: Dict[str, float], 
        n_subjects: int,
        covariate_models: Optional[Dict] = None
    ) -> List[Dict[str, float]]:
        """Generate individual parameters with covariate effects"""
        individual_params = []
        
        # 1. Generate basic covariates
        covariates = {
            'WT': np.random.normal(70, 15, n_subjects),
            'AGE': np.random.uniform(20, 80, n_subjects),
            'CRCL': np.random.normal(100, 25, n_subjects)
        }
        
        # 2. Use provided covariate models or defaults
        if covariate_models is None:
            covariate_models = {
                'CL': {
                    'CRCL': {'type': 'power', 'coefficient': 0.75}
                },
                'V': {
                    'WT': {'type': 'power', 'coefficient': 1.0}
                }
            }
        
        # 3. Set reference values
        reference_values = {
            'WT': 70,   # kg
            'AGE': 40,  # years (adjusted to median)
            'CRCL': 100 # mL/min
        }
        
        # 4. Generate parameters for each subject
        for i in range(n_subjects):
            subject_params = {}
            
            for param_name, param in self.model.parameters.items():
                # Set typical value
                typical_value = parameters[param_name]
                
                # Apply covariate effects
                if param_name in covariate_models:
                    for cov_name, model in covariate_models[param_name].items():
                        cov_value = covariates[cov_name][i]
                        ref_value = reference_values[cov_name]
                        
                        if model['type'] == 'power':
                            # 0이나 음수 값 방지
                            cov_ratio = max(cov_value/ref_value, 1e-10)
                            typical_value *= cov_ratio**model['coefficient']
                        elif model['type'] == 'linear':
                            typical_value *= (1 + model['coefficient']*(cov_value - ref_value))
                        elif model['type'] == 'exp':
                            typical_value *= np.exp(model['coefficient']*(cov_value - ref_value))
                
                # Apply Between-subject variability (BSV)
                if param.cv_percent is not None:
                    omega = np.sqrt(np.log(1 + (param.cv_percent/100)**2))
                    eta = np.random.normal(0, omega)
                    value = typical_value * np.exp(eta)
                else:
                    value = typical_value
                
                # Apply bounds
                if param.lower_bound is not None:
                    value = max(value, param.lower_bound)
                if param.upper_bound is not None:
                    value = min(value, param.upper_bound)
                    
                subject_params[param_name] = value
                
            individual_params.append(subject_params)
        
        # Store covariate data
        self.covariates = pd.DataFrame(covariates)
        
        return individual_params
        
    def simulate_individual(self, times: np.ndarray, dose: float, parameters: Dict[str, float]) -> np.ndarray:
        """Simulate PK profile for a single individual"""
        concentrations = self.model.solve_ode(times, dose, parameters)
        return concentrations
        
    def add_residual_error(self, concentrations: np.ndarray,
                          prop_error: float = 0.05,    # 5%로 감소
                          add_error: float = 0.01,    # 1%로 감소
                          lloq: float = 0.01) -> np.ndarray:
        """더욱 개선된 오차 구조"""
        # 농도별 오차 스케일링
        conc_scale = np.maximum(concentrations, lloq)
        relative_error = prop_error * np.sqrt(conc_scale/np.max(conc_scale))
        
        # 결합 오차 모델
        eps_p = np.random.normal(0, relative_error)
        eps_a = np.random.normal(0, add_error)
        
        observed = concentrations * (1 + eps_p) + eps_a
        observed = np.maximum(observed, lloq)
        
        return observed
        
    def simulate_population(
        self,
        n_subjects: int,
        times: np.ndarray,
        parameters: Dict[str, float],
        dose: float = 100.0,
        add_noise: bool = True,
        prop_error: float = 0.08,
        add_error: float = 0.02,
        lloq: float = 0.01,
        demographic_covariates: Optional[Dict] = None,
        covariate_models: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None
    ) -> Dict:
        """Simulate PK profiles for a population"""
        # Default covariate models if none provided
        if covariate_models is None:
            covariate_models = {
                'CL': {
                    'CRCL': {'type': 'power', 'coefficient': 0.75}
                },
                'V': {
                    'WT': {'type': 'power', 'coefficient': 0.75}
                }
            }
        
        # Generate individual parameters with provided covariate models
        individual_params = self.generate_individual_parameters(
            parameters, 
            n_subjects,
            covariate_models=covariate_models
        )
        all_concentrations = []
        
        for subject_idx, params in enumerate(individual_params):
            concentrations = self.simulate_individual(times, dose, params)
            
            if add_noise:
                concentrations = self.add_residual_error(
                    concentrations,
                    prop_error=prop_error,
                    add_error=add_error,
                    lloq=lloq
                )
                
            all_concentrations.append(concentrations)
        
        # Convert individual parameters to DataFrame
        param_df = pd.DataFrame(individual_params)
        
        demographics = pd.DataFrame(self.covariates)
        demographics['ID'] = range(n_subjects)
        
        return {
            'times': times,
            'concentrations': np.array(all_concentrations),
            'individual_parameters': param_df,
            'demographics': demographics
        }