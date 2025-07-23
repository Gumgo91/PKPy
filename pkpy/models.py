import numpy as np
from scipy.integrate import odeint, solve_ivp
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import warnings

@dataclass
class Parameter:
    """Parameter class for PK model parameters"""
    value: float
    cv_percent: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    fixed: bool = False

class CompartmentModel:
    """Base class for compartmental PK models"""
    def __init__(self, parameters: Dict[str, Union[float, Dict]]):
        self.parameters = {}
        for name, param in parameters.items():
            if isinstance(param, dict):
                self.parameters[name] = Parameter(**param)
            else:
                self.parameters[name] = Parameter(value=float(param))
        
    def get_differential_equations(self, y: List[float], t: float, params: Dict[str, float]) -> List[float]:
        """Define the differential equations for the model"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def solve_ode(self, times: np.ndarray, dose: float, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Solve ODE system for given times and parameters"""
        if params is None:
            params = {name: param.value for name, param in self.parameters.items()}
            
        initial_conditions = [0.0] * self.n_compartments
        initial_conditions[0] = dose
        
        solution = odeint(
            self.get_differential_equations,
            initial_conditions,
            times,
            args=(params,)
        )
        return solution

class OneCompartmentModel(CompartmentModel):
    """One compartment PK model with analytical solution"""
    def __init__(self, parameters: Dict[str, Union[float, Dict]]):
        super().__init__(parameters)
        self.n_compartments = 1
        
    def solve_ode(self, times: np.ndarray, dose: float, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Analytical solution for one compartment model:
        C(t) = D/V * exp(-CL/V * t)
        """
        if params is None:
            params = {name: param.value for name, param in self.parameters.items()}
            
        CL = params['CL']
        V = params['V']
        k = CL/V
        
        concentrations = dose/V * np.exp(-k * times)
        return concentrations.reshape(-1, 1)

class TwoCompartmentModel(CompartmentModel):
    """Two compartment PK model with analytical solution"""
    def __init__(self, parameters: Dict[str, Union[float, Dict]]):
        super().__init__(parameters)
        self.n_compartments = 2
        
    def get_differential_equations(self, y: List[float], t: float, params: Dict[str, float]) -> List[float]:
        """
        Define ODEs for two compartment model:
        dC1/dt = -(CL/V1 + Q/V1) * C1 + Q/V2 * C2
        dC2/dt = Q/V1 * C1 - Q/V2 * C2
        """
        C1, C2 = y
        CL = params['CL']
        V1 = params['V1']
        Q = params['Q']
        V2 = params['V2']
        
        dC1dt = -(CL/V1 + Q/V1) * C1 + Q/V2 * C2
        dC2dt = Q/V1 * C1 - Q/V2 * C2
        return [dC1dt, dC2dt]
    
    def solve_ode(self, times: np.ndarray, dose: float, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Analytical solution for two compartment model:
        C1(t) = A * exp(-alpha * t) + B * exp(-beta * t)
        
        For IV bolus administration
        """
        if params is None:
            params = {name: param.value for name, param in self.parameters.items()}
            
        CL = params['CL']
        V1 = params['V1']
        Q = params['Q']
        V2 = params['V2']
        
        # Calculate micro rate constants
        k10 = CL / V1  # elimination from central
        k12 = Q / V1   # central to peripheral
        k21 = Q / V2   # peripheral to central
        
        # Calculate macro rate constants (alpha, beta)
        a = k10 + k12 + k21
        b = k10 * k21
        
        # Protect against negative discriminant
        discriminant = a**2 - 4*b
        if discriminant < 0:
            discriminant = 0
            
        alpha = (a + np.sqrt(discriminant)) / 2
        beta = (a - np.sqrt(discriminant)) / 2
        
        # Ensure alpha > beta
        if alpha < beta:
            alpha, beta = beta, alpha
            
        # Calculate coefficients for IV bolus
        # Handle special case when alpha ≈ beta
        if abs(alpha - beta) < 1e-10:
            # Use L'Hôpital's rule or limiting case
            A = dose / V1
            B = 0
            concentrations = A * np.exp(-alpha * times) * (1 + alpha * times)
        else:
            A = dose / V1 * (alpha - k21) / (alpha - beta)
            B = dose / V1 * (k21 - beta) / (alpha - beta)
            
            # Calculate concentrations in central compartment
            concentrations = A * np.exp(-alpha * times) + B * np.exp(-beta * times)
        
        # Ensure non-negative concentrations
        concentrations = np.maximum(concentrations, 0)
        
        return concentrations.reshape(-1, 1)

import numpy as np
from scipy.integrate import odeint, solve_ivp
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import warnings

class OneCompartmentModelWithAbsorption(CompartmentModel):
    """One compartment PK model with first-order absorption"""
    def __init__(self, parameters: Dict[str, Union[float, Dict]]):
        super().__init__(parameters)
        self.n_compartments = 2
        
        # add default limit for Ka
        if 'Ka' in parameters:
            if isinstance(parameters['Ka'], dict):
                parameters['Ka'].setdefault('upper_bound', 10.0)  # 생리학적으로 타당한 상한값
            else:
                parameters['Ka'] = {'value': parameters['Ka'], 'upper_bound': 10.0}
        
    def get_differential_equations(self, y: List[float], t: float, params: Dict[str, float]) -> List[float]:
        """
        Define ODEs for one compartment model with absorption:
        dA_depot/dt = -Ka * A_depot
        dC/dt = Ka * A_depot/V - CL/V * C
        
        Parameters:
        -----------
        y: List[float]
            y[0]: Amount in depot compartment
            y[1]: Concentration in central compartment
        params: Dict[str, float]
            Ka: Absorption rate constant
            CL: Clearance
            V: Volume of distribution
        """
        A_depot, C = y
        Ka = params['Ka']
        CL = params['CL']
        V = params['V']
        
        dA_depot_dt = -Ka * A_depot
        dC_dt = Ka * A_depot/V - CL/V * C
        
        return [dA_depot_dt, dC_dt]
        
    def solve_ode(self, times: np.ndarray, dose: float, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Solve ODE for one-compartment model with absorption - enhanced stability"""
        if params is None:
            params = {name: param.value for name, param in self.parameters.items()}
            
        # More conservative parameter bounds
        Ka = np.clip(params['Ka'], 0.01, 20.0)
        CL = np.clip(params['CL'], 0.01, 100.0) 
        V = np.clip(params['V'], 0.1, 1000.0)
        
        k = CL / V
        
        # Check for numerical issues
        if Ka <= 0 or k <= 0 or V <= 0:
            warnings.warn("Invalid parameters detected, using defaults")
            Ka = 1.0
            k = 0.1
            V = 50.0
        
        # Protection against overflow
        max_exp_arg = 700.0  # Conservative limit
        
        concentrations = np.zeros_like(times)
        
        for i, t in enumerate(times):
            if t == 0:
                concentrations[i] = 0
                continue
                
            # Check for overflow potential
            if Ka * t > max_exp_arg or k * t > max_exp_arg:
                concentrations[i] = 0
                continue
            
            # Handle near-equal rate constants
            if abs(Ka - k) < 1e-8:
                # L'Hôpital's rule approximation
                if k * t < 10:  # For reasonable time values
                    concentrations[i] = dose * k / V * t * np.exp(-k * t)
                else:
                    concentrations[i] = 0
            else:
                # Standard biphasic solution
                try:
                    exp_kt = np.exp(-k * t)
                    exp_Kat = np.exp(-Ka * t)
                    
                    # Additional check for numerical stability
                    if np.isfinite(exp_kt) and np.isfinite(exp_Kat):
                        concentrations[i] = dose * Ka / (V * (Ka - k)) * (exp_kt - exp_Kat)
                    else:
                        concentrations[i] = 0
                except:
                    concentrations[i] = 0
        
        # Ensure non-negative concentrations
        concentrations = np.maximum(concentrations, 0)
        
        return concentrations.reshape(-1, 1)

class TwoCompartmentModelWithAbsorption(CompartmentModel):
    """Two compartment PK model with first-order absorption"""
    def __init__(self, parameters: Dict[str, Union[float, Dict]]):
        super().__init__(parameters)
        self.n_compartments = 3  # depot, central, peripheral
        
        # add default limit for Ka
        if 'Ka' in parameters:
            if isinstance(parameters['Ka'], dict):
                parameters['Ka'].setdefault('upper_bound', 10.0)
            else:
                parameters['Ka'] = {'value': parameters['Ka'], 'upper_bound': 10.0}
        
    def get_differential_equations(self, y: List[float], t: float, params: Dict[str, float]) -> List[float]:
        """
        Define ODEs for two compartment model with absorption:
        dA_depot/dt = -Ka * A_depot
        dA1/dt = Ka * A_depot - (CL/V1 + Q/V1) * A1 + Q/V2 * A2
        dA2/dt = Q/V1 * A1 - Q/V2 * A2
        """
        A_depot, A1, A2 = y
        Ka = params['Ka']
        CL = params['CL']
        V1 = params['V1']
        Q = params['Q']
        V2 = params['V2']
        
        dA_depot_dt = -Ka * A_depot
        dA1_dt = Ka * A_depot - (CL/V1 + Q/V1) * A1 + Q/V2 * A2
        dA2_dt = Q/V1 * A1 - Q/V2 * A2
        
        return [dA_depot_dt, dA1_dt, dA2_dt]
    
    def solve_ode(self, times: np.ndarray, dose: float, params: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Solve ODE for two-compartment model with absorption using improved numerical stability"""
        if params is None:
            params = {name: param.value for name, param in self.parameters.items()}
        
        # Parameter validation and bounds
        Ka = np.clip(params['Ka'], 0.01, 20.0)
        CL = np.clip(params['CL'], 0.1, 100.0)
        V1 = np.clip(params['V1'], 1.0, 1000.0)
        Q = np.clip(params['Q'], 0.1, 100.0)
        V2 = np.clip(params['V2'], 1.0, 1000.0)
        
        validated_params = {'Ka': Ka, 'CL': CL, 'V1': V1, 'Q': Q, 'V2': V2}
        
        # Check for potential stiffness
        k10 = CL / V1
        k12 = Q / V1
        k21 = Q / V2
        
        # If rate constants are very different, use analytical approximation
        rates = [Ka, k10, k12, k21]
        if max(rates) / min(rates) > 1000:
            warnings.warn("Highly stiff system detected. Using analytical approximation.")
            # Fall back to sequential approximation
            return self._analytical_approximation(times, dose, validated_params)
        
        # Initial conditions with small offset to avoid singularity
        initial_conditions = [dose * 0.9999, dose * 0.0001, 0.0]
        
        # Define ODE function for solve_ivp (different signature)
        def ode_func(t, y):
            return self.get_differential_equations(y, t, validated_params)
        
        try:
            # Use solve_ivp with BDF method for stiff equations
            solution = solve_ivp(
                ode_func,
                t_span=(times[0], times[-1]),
                y0=initial_conditions,
                t_eval=times,
                method='BDF',  # Good for stiff ODEs
                rtol=1e-8,
                atol=1e-10,
                max_step=0.5  # Limit step size
            )
            
            if solution.success:
                # Return concentration in central compartment (A1/V1)
                concentrations = solution.y[1] / V1
                return concentrations.reshape(-1, 1)
            else:
                warnings.warn(f"ODE solver failed: {solution.message}")
                return self._analytical_approximation(times, dose, validated_params)
                
        except Exception as e:
            warnings.warn(f"ODE solver error: {str(e)}")
            # Fall back to odeint with adjusted tolerances
            try:
                solution = odeint(
                    self.get_differential_equations,
                    initial_conditions,
                    times,
                    args=(validated_params,),
                    rtol=1e-6,
                    atol=1e-8,
                    hmax=0.1
                )
                concentrations = solution[:, 1] / V1
                return concentrations.reshape(-1, 1)
            except:
                # Final fallback to analytical approximation
                return self._analytical_approximation(times, dose, validated_params)
    
    def _analytical_approximation(self, times: np.ndarray, dose: float, params: Dict[str, float]) -> np.ndarray:
        """Analytical approximation for 2-compartment with absorption"""
        Ka = params['Ka']
        CL = params['CL']
        V1 = params['V1']
        Q = params['Q']
        V2 = params['V2']
        
        # Approximate as 1-compartment with modified parameters
        # This is a rough approximation but numerically stable
        V_apparent = V1 + V2 * Q/(Q + CL)
        k = CL / V_apparent
        
        # Use 1-compartment with absorption solution
        eps = 1e-10
        if abs(Ka - k) < eps:
            Ka = k * (1 + eps)
            concentrations = dose * Ka / V_apparent * times * np.exp(-k * times)
        else:
            concentrations = dose * Ka / (V_apparent * (Ka - k)) * (np.exp(-k * times) - np.exp(-Ka * times))
        
        # Adjust for 2-compartment behavior (empirical correction)
        # Early phase correction for distribution
        distribution_factor = 1 + (V2/V1 - 1) * np.exp(-Q/V2 * times)
        concentrations = concentrations / distribution_factor
        
        return np.maximum(concentrations, 0).reshape(-1, 1)

def create_pkpy_model(model_type: str, parameters: Dict[str, Dict]) -> CompartmentModel:
    """Factory function to create PK models"""
    model_types = {
        "onecomp": OneCompartmentModel,
        "onecomp_abs": OneCompartmentModelWithAbsorption,
        "twocomp": TwoCompartmentModel,
        "twocomp_abs": TwoCompartmentModelWithAbsorption
    }
    
    if model_type not in model_types:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model_types[model_type](parameters)