import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

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
    """Two compartment PK model"""
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

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

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
        """Solve ODE for one-compartment model with absorption"""
        if params is None:
            params = {name: param.value for name, param in self.parameters.items()}
            
        Ka = np.clip(params['Ka'], 0.1, 10.0)  # limit Ka value range
        
        V = params['V']
        k = params['CL'] / V
        
        # protection against very large time values or rate constants
        max_exp_arg = 709.0  # np.log(np.finfo(np.float64).max)
        safe_times = np.minimum(times, max_exp_arg/max(Ka, k))
        
        # special case when Ka and k are very similar
        eps = 1e-10
        if abs(Ka - k) < eps:
            Ka = k * (1 + eps)  # ensure Ka and k are not exactly the same
            # apply L'Hôpital's rule
            concentrations = dose * Ka / V * safe_times * np.exp(-k * safe_times)
        else:
            # separate exponential term calculation for numerical stability
            exp_k = np.exp(-k * safe_times)
            exp_Ka = np.exp(-Ka * safe_times)
            concentrations = dose * Ka / (V * (Ka - k)) * (exp_k - exp_Ka)
        
        # handle very large time values by setting concentrations to 0
        concentrations[times > safe_times] = 0
        
        return concentrations.reshape(-1, 1)

def create_pkpy_model(model_type: str, parameters: Dict[str, Dict]) -> CompartmentModel:
    """Factory function to create PK models"""
    model_types = {
        "onecomp": OneCompartmentModel,
        "onecomp_abs": OneCompartmentModelWithAbsorption,
        "twocomp": TwoCompartmentModel
    }
    
    if model_type not in model_types:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model_types[model_type](parameters)