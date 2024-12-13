import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple, List
from .models import CompartmentModel
from numba import jit

@jit(nopython=True)
def _calculate_objective(log_obs, log_pred, valid_mask):
    """목적 함수의 계산 부분을 최적화"""
    return np.sum((log_obs[valid_mask] - log_pred[valid_mask]) ** 2)

@jit(nopython=True)
def _calculate_penalty(log_params, lower_bounds, upper_bounds):
    """파라미터 페널티 계산을 최적화"""
    penalty = 0.0
    for i in range(len(log_params)):
        val = np.exp(log_params[i])
        if lower_bounds[i] > 0 and val < lower_bounds[i]:
            penalty += 1e3 * (lower_bounds[i] - val)**2
        if upper_bounds[i] > 0 and val > upper_bounds[i]:
            penalty += 1e3 * (val - upper_bounds[i])**2
    return penalty

class PKPyFit:
    def __init__(self, data: Dict, model: CompartmentModel):
        self.data = data
        self.model = model
        self.results = None
        self.individual_params = None
        
    def _solve_ode_wrapper(self, times: np.ndarray, dose: float, params: Dict) -> np.ndarray:
        pred = self.model.solve_ode(times, dose, params)
        return pred.flatten() if len(pred.shape) == 2 else pred
        
    def _calculate_individual_predictions(self, log_params: np.ndarray) -> np.ndarray:
        # 파라미터 경계 설정
        min_param_value = 1e-10
        params = {}
        for name, log_val in zip(self.param_names, log_params):
            param_obj = self.model.parameters[name]
            val = np.exp(log_val)
            
            # 경계값 적용
            if param_obj.lower_bound is not None:
                val = max(val, param_obj.lower_bound)
            if param_obj.upper_bound is not None:
                val = min(val, param_obj.upper_bound)
                
            val = max(val, min_param_value)  # 최소값 보장
            params[name] = val
            
        return self._solve_ode_wrapper(self.data['times'], self.data['dose'], params)
        
    def _objective_individual(self, log_params: np.ndarray, observed: np.ndarray) -> float:
        try:
            predictions = self._calculate_individual_predictions(log_params)
            valid_mask = ~np.isnan(observed)
            
            log_obs = np.log(np.maximum(observed, 1e-10))
            log_pred = np.log(np.maximum(predictions, 1e-10))
            
            # numba 최적화된 함수들 호출
            objective = _calculate_objective(log_obs, log_pred, valid_mask)
            
            # 경계값 배열 준비
            lower_bounds = np.array([
                param_obj.lower_bound if param_obj.lower_bound is not None else -1
                for param_obj in [self.model.parameters[name] for name in self.param_names]
            ])
            upper_bounds = np.array([
                param_obj.upper_bound if param_obj.upper_bound is not None else -1
                for param_obj in [self.model.parameters[name] for name in self.param_names]
            ])
            
            penalty = _calculate_penalty(log_params, lower_bounds, upper_bounds)
            return objective + penalty
            
        except Exception as e:
            return 1e10
        
    def _safe_geometric_mean(self, values: np.ndarray) -> float:
        """안전한 기하평균 계산"""
        min_value = 1e-10
        safe_values = np.maximum(values, min_value)
        return np.exp(np.mean(np.log(safe_values)))
        
    def fit(self, method: str = "fo", **kwargs) -> Dict:
        self.param_names = [name for name, param in self.model.parameters.items() 
                          if not param.fixed]
        
        # 초기값 설정
        pop_params = {name: param.value for name, param in self.model.parameters.items()
                     if not param.fixed}
        pop_params_array = np.array([pop_params[name] for name in self.param_names])
        
        # 로그 스케일 초기값 조정
        log_pop_params = np.log(np.maximum(pop_params_array, 1e-10))
        
        concentrations = self.data['concentrations']
        if len(concentrations.shape) == 3:
            concentrations = concentrations.squeeze(2)
            
        individual_params = []
        successful_fits = 0
        all_results = []
        
        for subj_idx in range(len(concentrations)):
            subj_conc = concentrations[subj_idx]
            best_obj = float('inf')
            best_params = None
            
            # try multiple initial values
            for scale_factor in [0.1, 0.5, 1.0, 1.5, 2.0]:
                try:
                    init_params = log_pop_params + np.log(scale_factor) + np.random.normal(0, 0.1, len(log_pop_params))
                    result = minimize(
                        self._objective_individual,
                        init_params,
                        args=(subj_conc,),
                        method='Nelder-Mead',
                        options={'maxiter': 9999}
                    )
                    
                    if result.fun < best_obj:
                        best_obj = result.fun
                        best_params = result.x
                        
                except:
                    continue
                    
            if best_params is not None:
                params = {name: np.exp(val) for name, val in zip(self.param_names, best_params)}
                successful_fits += 1
            else:
                params = {name: val for name, val in pop_params.items()}
                
            individual_params.append(params)
            all_results.append(best_obj if best_obj != float('inf') else None)
            
        # Population parameter calculation
        final_pop_params = {}
        param_matrix = np.array([[params[name] for name in self.param_names] 
                                for params in individual_params])
        
        for i, name in enumerate(self.param_names):
            final_pop_params[name] = self._safe_geometric_mean(param_matrix[:, i])
            
        # calculate variability
        log_param_matrix = np.log(np.maximum(param_matrix, 1e-10))
        avg_log_params = np.mean(log_param_matrix, axis=0)
        centered_log_params = log_param_matrix - avg_log_params
        omega = np.cov(centered_log_params.T)
        
        # save results
        self.results = {
            'success': successful_fits > 0,
            'parameters': final_pop_params,
            'individual_parameters': individual_params,
            'omega': omega,
            'successful_subjects': successful_fits,
            'total_subjects': len(concentrations),
            'individual_objectives': all_results
        }
        
        self.individual_params = individual_params
        return self.results
        
    def get_predictions(self, times: Optional[np.ndarray] = None) -> np.ndarray:
        if self.results is None:
            raise ValueError("Model must be fitted before getting predictions")
            
        if times is None:
            times = self.data['times']
            
        predictions = []
        for params in self.individual_params:
            pred = self._solve_ode_wrapper(times, self.data['dose'], params)
            predictions.append(pred)
            
        return np.array(predictions)