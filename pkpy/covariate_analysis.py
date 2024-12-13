import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

@dataclass
class CovariateRelationship:
    """Class to define covariate relationships"""
    parameter: str  # PK parameter name (e.g., 'CL', 'V')
    covariate: str  # Covariate name (e.g., 'WT', 'AGE')
    relationship: str  # Type of relationship ('linear', 'power', 'exponential')
    coefficient: float = 0.75  # Default coefficient for allometric scaling
    center: Optional[float] = None  # Reference/center value for the covariate

class CovariateAnalyzer:
    """Class for performing covariate analysis on PK data"""
    
    def __init__(self, pk_data: pd.DataFrame):
        """
        Initialize with PK data containing individual parameter estimates
        and covariate information
        
        Parameters:
        -----------
        pk_data: pd.DataFrame
            DataFrame containing individual PK parameters and covariates
        """
        self.data = pk_data
        self.relationships = []
        self.results = {}
        
    def add_relationship(self, relationship: CovariateRelationship):
        """Add a covariate-parameter relationship to test"""
        self.relationships.append(relationship)
        
    def _apply_relationship(self, theta: float, cov: np.ndarray, 
                           relationship: CovariateRelationship) -> np.ndarray:
        """Apply covariate relationship with given parameter value"""
        if relationship.center is None:
            relationship.center = np.median(cov)
            
        # Add eps for numerical stability
        eps = np.finfo(float).eps
        
        if relationship.relationship == 'linear':
            return theta * (1 + relationship.coefficient * (cov - relationship.center))
        elif relationship.relationship == 'power':
            # Prevent division by zero
            safe_center = np.maximum(relationship.center, eps)
            safe_cov = np.maximum(cov, eps)
            return theta * (safe_cov / safe_center) ** relationship.coefficient
        elif relationship.relationship == 'exponential':
            return theta * np.exp(relationship.coefficient * (cov - relationship.center))
        else:
            raise ValueError(f"Unknown relationship type: {relationship.relationship}")
            
    def _calculate_objective(self, params: np.ndarray, relationship: CovariateRelationship) -> float:
        """Calculate objective function with improvements"""
        theta, coefficient = params
        relationship.coefficient = coefficient
        
        # Observed and predicted values
        cov = self.data[relationship.covariate].values
        param_obs = self.data[relationship.parameter].values
        
        eps = np.finfo(float).eps
        
        param_pred = np.maximum(self._apply_relationship(theta, cov, relationship), eps)
        param_obs = np.maximum(param_obs, eps)
        
        # 로그 변환된 잔차
        eta = np.log(param_obs) - np.log(param_pred)
        
        # 개선된 분산 추정
        omega = np.var(eta)
        if omega < 1e-6:
            omega = 1e-6
        
        # 수정된 objective function - AIC 기반
        n = len(param_obs)
        obj = (
            n * np.log(omega) +  # 분산 페널티
            np.sum(eta**2) / omega +  # 가중치가 적용된 잔차 제곱합
            np.sum(np.log(param_pred)) +  # Jacobian 기여도
            2  # AIC 보정항 (파라미터 수에 대한 페널티)
        )
        
        return obj if np.isfinite(obj) else 1e10
        
    def analyze_relationship(self, relationship: CovariateRelationship) -> Dict:
        """Analyze a single covariate-parameter relationship with multiple initial guesses"""
        from scipy.optimize import minimize
        
        param_data = np.maximum(self.data[relationship.parameter].values, np.finfo(float).eps)
        
        # 다양한 초기값 설정
        theta_guesses = [
            np.median(param_data),
            np.mean(param_data),
            np.percentile(param_data, 75)
        ]
        coef_guesses = [0.75, 0.5, 1.0]  # 일반적인 생리학적 범위의 계수들
        
        best_result = None
        best_obj = np.inf
        
        # 여러 초기값으로 최적화 시도
        for theta_init in theta_guesses:
            for coef_init in coef_guesses:
                try:
                    result = minimize(
                        self._calculate_objective,
                        x0=[theta_init, coef_init],
                        args=(relationship,),
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5}
                    )
                    
                    if result.success and result.fun < best_obj:
                        best_result = result
                        best_obj = result.fun
                        
                except Exception:
                    continue
        
        if best_result is None:
            warnings.warn(f"Optimization failed for {relationship.parameter}-{relationship.covariate}")
            return {
                'theta': theta_init,
                'coefficient': coef_init,
                'r_squared': 0,
                'p_value': 1,
                'relationship': relationship,
                'fitted_values': None,
                'convergence_failed': True,
                'ss_res': np.inf
            }
        
        theta_opt, coef_opt = best_result.x
        relationship.coefficient = coef_opt
        
        fitted_values = self._apply_relationship(
            theta_opt,
            self.data[relationship.covariate].values,
            relationship
        )
        
        # 안전한 계산을 위한 전처리
        eps = np.finfo(float).eps
        param_obs = np.maximum(self.data[relationship.parameter].values, eps)
        fitted_values = np.maximum(fitted_values, eps)
        
        # 로그 변환된 값으로 계산
        log_obs = np.log(param_obs)
        log_fit = np.log(fitted_values)
        
        # 잔차 제곱합 계산 추가
        ss_res = np.sum((log_obs - log_fit) ** 2)
        
        # 안전한 R-squared 계산
        mean_log_obs = np.mean(log_obs)
        ss_tot = np.sum((log_obs - mean_log_obs) ** 2)
        
        try:
            if ss_tot < eps:
                r_squared = 0.0
            else:
                r_squared = np.clip(1 - (ss_res / ss_tot), 0, 1)
        except:
            r_squared = 0.0
            warnings.warn(f"R-squared calculation failed for {relationship.parameter}-{relationship.covariate}")
        
        # 통계 검정
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_fit,
                log_obs
            )
        except:
            p_value = 1.0
            warnings.warn(f"Statistical test failed for {relationship.parameter}-{relationship.covariate}")
        
        return {
            'theta': theta_opt,
            'coefficient': coef_opt,
            'r_squared': r_squared,
            'p_value': p_value,
            'relationship': relationship,
            'fitted_values': fitted_values,
            'ss_res': ss_res
        }
        
    def run_forward_selection(self, significance_level: float = 0.05) -> Dict:
        """Run forward selection with improved criteria"""
        selected_relationships = []
        
        # 파라미터별로 관계들을 그룹화
        param_groups = {}
        for rel in self.relationships:
            if rel.parameter not in param_groups:
                param_groups[rel.parameter] = []
            param_groups[rel.parameter].append(rel)
        
        # 각 파라미터에 대해 가장 좋은 관계 선택
        for param, relationships in param_groups.items():
            best_relationship = None
            best_aic = np.inf
            
            for rel in relationships:
                result = self.analyze_relationship(rel)
                if result and 'ss_res' in result:
                    n = len(self.data)
                    aic = n * np.log(result['ss_res']/n) + 2  # 각 관계당 1개 파라미터
                    
                    # p-value와 AIC 모두 고려
                    if result['p_value'] < significance_level and aic < best_aic:
                        best_aic = aic
                        best_relationship = rel
            
            if best_relationship is not None:
                selected_relationships.append(best_relationship)
        
        # 최종 AIC 계산
        final_aic = 0
        if selected_relationships:
            total_ss_res = 0
            for rel in selected_relationships:
                result = self.analyze_relationship(rel)
                if result and 'ss_res' in result:
                    total_ss_res += result['ss_res']
            n = len(self.data)
            final_aic = n * np.log(total_ss_res/n) + 2 * len(selected_relationships)
        
        return {
            'selected_relationships': selected_relationships,
            'final_aic': final_aic
        }
        
    def plot_relationships(self) -> Dict:
        """Create diagnostic plots for covariate relationships"""
        import matplotlib.pyplot as plt
        
        results = {}
        for rel in self.relationships:
            result = self.analyze_relationship(rel)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Scatter plot of parameter vs covariate
            ax1.scatter(self.data[rel.covariate], self.data[rel.parameter])
            ax1.plot(self.data[rel.covariate], result['fitted_values'], 'r-')
            ax1.set_xlabel(rel.covariate)
            ax1.set_ylabel(rel.parameter)
            ax1.set_title(f'{rel.parameter} vs {rel.covariate}')
            
            # Residual plot
            residuals = np.log(self.data[rel.parameter]) - np.log(result['fitted_values'])
            ax2.scatter(result['fitted_values'], residuals)
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel('Fitted Values')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Residual Plot')
            
            plt.tight_layout()
            
            results[f"{rel.parameter}_{rel.covariate}"] = {
                'figure': fig,
                'analysis_result': result
            }
            
        return results
        
    def run_stepwise_selection(self, 
                             forward_alpha: float = 0.05, 
                             backward_alpha: float = 0.01) -> Dict:
        """Perform stepwise selection with both forward and backward steps"""
        selected_relationships = []
        remaining_relationships = self.relationships.copy()
        current_objective = float('inf')
        
        # Backward elimination
        while selected_relationships:
            worst_relationship = None
            worst_p_value = 0
            
            for rel in selected_relationships:
                result = self.analyze_relationship(rel)
                if result['p_value'] > backward_alpha:
                    if result['p_value'] > worst_p_value:
                        worst_p_value = result['p_value']
                        worst_relationship = rel
            
            if worst_relationship is None:
                break
                
            selected_relationships.remove(worst_relationship)
        
        # Forward selection
        while remaining_relationships:
            best_relationship = None
            best_result = None
            best_improvement = 0
            
            for rel in remaining_relationships:
                result = self.analyze_relationship(rel)
                
                if result['p_value'] < forward_alpha:
                    improvement = current_objective - result['r_squared']
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_relationship = rel
                        best_result = result
                        
            if best_relationship is None:
                break
                
            selected_relationships.append(best_relationship)
            remaining_relationships.remove(best_relationship)
            current_objective = best_result['r_squared']
            
        return {
            'selected_relationships': selected_relationships,
            'final_r_squared': current_objective
        }

def test_covariate_effects(
    pk_data: pd.DataFrame,
    parameter_cols: List[str],
    covariate_cols: List[str],
    relationships: Optional[List[str]] = None
) -> Dict:
    """
    Convenience function to test multiple covariate effects
    
    Parameters:
    -----------
    pk_data: pd.DataFrame
        DataFrame containing PK parameters and covariates
    parameter_cols: List[str]
        Names of columns containing PK parameters
    covariate_cols: List[str]
        Names of columns containing covariates
    relationships: Optional[List[str]]
        Types of relationships to test ('linear', 'power', 'exponential')
        
    Returns:
    --------
    Dict containing analysis results
    """
    if relationships is None:
        relationships = ['linear', 'power', 'exponential']
        
    analyzer = CovariateAnalyzer(pk_data)
    
    # Add all possible parameter-covariate relationships
    for param in parameter_cols:
        for cov in covariate_cols:
            for rel_type in relationships:
                relationship = CovariateRelationship(
                    parameter=param,
                    covariate=cov,
                    relationship=rel_type
                )
                analyzer.add_relationship(relationship)
                
    # Run forward selection
    selection_results = analyzer.run_forward_selection()
    
    # Create diagnostic plots
    plot_results = analyzer.plot_relationships()
    
    return {
        'selection_results': selection_results,
        'plot_results': plot_results
    }