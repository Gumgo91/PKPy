import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional
from .models import CompartmentModel
from .simulation import SimulationEngine
from .fitting import PKPyFit
from .covariate_analysis import CovariateAnalyzer, CovariateRelationship
from .utils import calculate_nca_parameters, calculate_validation_metrics
from .models import create_pkpy_model

class BasePKWorkflow:
    """Base class for PK analysis workflows"""
    
    def __init__(self, model: CompartmentModel, n_subjects: int = 20):
        """
        Initialize the workflow
        
        Parameters:
        -----------
        model: CompartmentModel
            PK model to use for analysis
        n_subjects: int
            Number of subjects to simulate
        """
        self.model = model
        self.sim_engine = SimulationEngine(model)
        self.n_subjects = n_subjects
        self.results = {}
        
    def generate_virtual_population(self, 
                                  time_points: np.ndarray,
                                  dose: float = 100.0,
                                  demographic_covariates: Optional[Dict] = None,
                                  covariate_models: Optional[Dict] = None) -> None:
        """
        Generate virtual population with covariates
        
        Parameters:
        -----------
        time_points: np.ndarray
            Time points for simulation
        dose: float
            Administered dose
        demographic_covariates: Optional[Dict]
            Dictionary of demographic covariate specifications
        covariate_models: Optional[Dict]
            Dictionary specifying covariate-parameter relationships
            Example:
            {
                'CL': {
                    'WT': {'type': 'power', 'coefficient': 0.75},
                    'CRCL': {'type': 'power', 'coefficient': 0.75}
                },
                'V': {
                    'WT': {'type': 'power', 'coefficient': 1.0}
                }
            }
        """
        # SimulationEngine을 통해 데이터 생성
        sim_data = self.sim_engine.simulate_population(
            n_subjects=self.n_subjects,
            times=time_points,
            parameters={name: param.value for name, param in self.model.parameters.items()},
            dose=dose,
            add_noise=True,
            demographic_covariates=demographic_covariates,
            covariate_models=covariate_models  # covariate_models 전달
        )
        
        # Store data
        self.times = time_points
        self.data = {
            'times': sim_data['times'],
            'concentrations': sim_data['concentrations'],
            'dose': dose,
            'demographics': sim_data['demographics'],
            'individual_parameters': pd.DataFrame(sim_data['individual_parameters'])
        }
        
    def run_model_fitting(self) -> None:
        """Fit the PK model to data with improved metrics handling"""
        if not hasattr(self, 'data'):
            raise ValueError("No data available. Run generate_virtual_population first.")
            
        try:
            fit_data = {
                'times': self.data['times'],
                'concentrations': self.data['concentrations'],
                'dose': self.data['dose']
            }
            
            self.fit = PKPyFit(fit_data, self.model)
            fit_results = self.fit.fit(method="basic")
            
            # save all results
            self.results['model_fit'] = fit_results
            self.results['predictions'] = fit_results.get('predictions', self.fit.get_predictions())
            self.results['fit_metrics'] = fit_results.get('fit_metrics', {
                'R2': np.nan,
                'RMSE': np.nan,
                'AFE': np.nan,
                'AAFE': np.nan,
                'MAE': np.nan,
                'Mean_Residual': np.nan
            })
            
            # recalculate metrics if not available or nan
            if not self.results.get('fit_metrics') or np.isnan(self.results['fit_metrics'].get('R2', np.nan)):
                obs = self.data['concentrations'].flatten()
                pred = self.results['predictions'].flatten()
                
                # select valid data points only
                valid_mask = ~(np.isnan(obs) | np.isnan(pred) | 
                            np.isinf(obs) | np.isinf(pred))
                
                if np.any(valid_mask):
                    obs = obs[valid_mask]
                    pred = pred[valid_mask]
                    residuals = obs - pred
                    
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((obs - np.mean(obs))**2)
                    
                    # Try to use the validation metrics function
                    try:
                        validation_metrics = calculate_validation_metrics(obs, pred)
                        self.results['fit_metrics'] = {
                            'R2': validation_metrics['R2'],
                            'RMSE': validation_metrics['RMSE'],
                            'AFE': validation_metrics['AFE'],
                            'AAFE': validation_metrics['AAFE'],
                            'MAE': np.mean(np.abs(residuals)),
                            'Mean_Residual': np.mean(residuals)
                        }
                    except:
                        # Fallback calculation
                        self.results['fit_metrics'] = {
                            'R2': 1 - ss_res/ss_tot if ss_tot != 0 else np.nan,
                            'RMSE': np.sqrt(np.mean(residuals**2)),
                            'AFE': np.nan,
                            'AAFE': np.nan,
                            'MAE': np.mean(np.abs(residuals)),
                            'Mean_Residual': np.mean(residuals)
                        }
                    
        except Exception as e:
            print(f"Error in model fitting: {str(e)}")
            self.results['fit_metrics'] = {
                'R2': np.nan,
                'RMSE': np.nan,
                'AFE': np.nan,
                'AAFE': np.nan,
                'MAE': np.nan,
                'Mean_Residual': np.nan
            }
            
        # set initial value range for Ka
        if 'Ka' in self.model.parameters:
            ka_param = self.model.parameters['Ka']
            ka_param.lower_bound = 0.1
            ka_param.upper_bound = 5.0
            ka_param.cv_percent = min(ka_param.cv_percent, 50)
            
    def run_nca_analysis(self) -> None:
        """Perform NCA analysis with robust error handling"""
        if not hasattr(self, 'data'):
            raise ValueError("No data available. Run generate_virtual_population first.")
            
        all_nca_params = []
        param_names = list(self.model.parameters.keys())
        successful_subjects = 0
        
        concentrations = self.data['concentrations']
        
        for subj_idx, subj_conc in enumerate(concentrations):
            try:
                # NCA analysis
                nca_params = calculate_nca_parameters(
                    times=self.times,
                    concentrations=subj_conc,
                    dose=self.data['dose']
                )
                nca_params['Subject_ID'] = subj_idx
                
                # add individual parameters
                for param in param_names:
                    nca_params[param] = self.data['individual_parameters'].iloc[subj_idx][param]
                    
                all_nca_params.append(nca_params)
                successful_subjects += 1
                
            except ValueError as e:
                print(f"Warning: NCA failed for subject {subj_idx}: {str(e)}")
                # use simulated parameters if NCA fails
                nca_params = {'Subject_ID': subj_idx}
                for param in param_names:
                    nca_params[param] = self.data['individual_parameters'].iloc[subj_idx][param]
                all_nca_params.append(nca_params)
                
        if all_nca_params:
            nca_df = pd.DataFrame(all_nca_params)
            self.results['nca'] = {
                'individual': nca_df,
                'summary': nca_df.describe(),
                'success_rate': successful_subjects / len(concentrations)
            }
            print(f"NCA analysis completed. Success rate: {successful_subjects}/{len(concentrations)} subjects")
        else:
            raise ValueError("NCA analysis failed for all subjects")
            
    def run_covariate_analysis(self, relationships: Optional[List[CovariateRelationship]] = None) -> None:
        """Perform covariate analysis with robust error handling"""
        if not hasattr(self, 'data'):
            raise ValueError("No data available")
        
        # prepare improved data
        pk_params = self.results.get('nca', {}).get('individual', 
                    pd.DataFrame(self.data['individual_parameters']))
        
        # create standardized covariate data
        covariates = self.data['demographics'].copy()
        for col in covariates.columns:
            if col != 'ID':
                covariates[col] = (covariates[col] - covariates[col].mean()) / covariates[col].std()
        
        pk_data = pd.concat([
            self.data['demographics'],
            pk_params
        ], axis=1)
        
        analyzer = CovariateAnalyzer(pk_data)
        
        if relationships is None:
            # include only available covariates
            available_covs = set(self.data['demographics'].columns) - {'ID'}
            relationships = []
            for param in self.model.parameters.keys():
                for cov in available_covs:  # use only existing covariates
                    for rel_type in ['linear', 'power']:
                        relationships.append(
                            CovariateRelationship(
                                parameter=param,
                                covariate=cov,
                                relationship=rel_type
                            )
                        )
                        
        for rel in relationships:
            # check if covariate actually exists
            if rel.covariate in pk_data.columns:
                analyzer.add_relationship(rel)
            else:
                print(f"Warning: Covariate '{rel.covariate}' not found in data, skipping...")
                
        try:
            self.results['covariate_analysis'] = analyzer.run_forward_selection()
        except Exception as e:
            print(f"Warning: Covariate analysis failed: {str(e)}")
            self.results['covariate_analysis'] = None
        
    def create_diagnostic_plots(self, log_scale: bool = False) -> None:
        """
        Create comprehensive diagnostic plots with robust metric calculations
        
        Parameters:
        -----------
        log_scale: bool
            Whether to use log scale for concentration plots
        """
        predictions = self.results.get('predictions')
        if predictions is None:
            print("Warning: No predictions available")
            self.results['fit_metrics'] = {
                'R2': np.nan, 'RMSE': np.nan,
                'MAE': np.nan, 'Mean_Residual': np.nan
            }
            return

        # Create figure
        fig = plt.figure(figsize=(15, 15))

        # 1. Individual PK profiles
        ax1 = plt.subplot(321)
        plot_func = ax1.semilogy if log_scale else ax1.plot
        
        try:
            for i in range(len(self.data['concentrations'])):
                plot_func(self.times, self.data['concentrations'][i], 'b-', alpha=0.3)
                plot_func(self.times, predictions[i], 'r--', alpha=0.3)
                
            plot_func(self.times, self.data['concentrations'].mean(axis=0), 
                    'b-', linewidth=2, label='Mean Observed')
            plot_func(self.times, predictions.mean(axis=0), 
                    'r--', linewidth=2, label='Mean Predicted')
            
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Concentration' + (' (log scale)' if log_scale else ''))
            ax1.set_title('Individual PK Profiles')
            ax1.grid(True)
            ax1.legend()
        except Exception as e:
            print(f"Warning: Error plotting PK profiles: {str(e)}")

        # Prepare data for metrics and other plots
        observed = np.asarray(self.data['concentrations']).flatten()
        predicted = np.asarray(predictions).flatten()

        # Remove any invalid values
        valid_mask = ~(np.isnan(observed) | np.isnan(predicted) | 
                    np.isinf(observed) | np.isinf(predicted) |
                    (observed < 0) | (predicted < 0))
        
        observed = observed[valid_mask]
        predicted = predicted[valid_mask]
        times_flat = np.repeat(self.times, len(self.data['concentrations']))[valid_mask]

        if len(observed) == 0 or len(predicted) == 0:
            print("Warning: No valid data points for metric calculation")
            self.results['fit_metrics'] = {
                'R2': np.nan, 'RMSE': np.nan,
                'MAE': np.nan, 'Mean_Residual': np.nan
            }
            return

        try:
            # Calculate residuals
            residuals = observed - predicted
            
            # 2. Observed vs Predicted
            ax2 = plt.subplot(322)
            ax2.scatter(predicted, observed, alpha=0.5)
            min_val = min(predicted.min(), observed.min())
            max_val = max(predicted.max(), observed.max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--')
            ax2.set_xlabel('Predicted Concentration')
            ax2.set_ylabel('Observed Concentration')
            ax2.set_title('Observed vs Predicted')
            
            # 3. Residuals vs Predicted
            ax3 = plt.subplot(323)
            ax3.scatter(predicted, residuals, alpha=0.5)
            ax3.axhline(y=0, color='k', linestyle='--')
            ax3.set_xlabel('Predicted Concentration')
            ax3.set_ylabel('Residuals')
            ax3.set_title('Residuals vs Predicted')
            
            # 4. QQ plot
            ax4 = plt.subplot(324)
            std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
            stats.probplot(std_residuals, dist="norm", plot=ax4)
            ax4.set_title('Normal Q-Q Plot of Residuals')
            
            # 5. Time vs Residuals
            ax5 = plt.subplot(325)
            ax5.scatter(times_flat, residuals, alpha=0.5)
            ax5.axhline(y=0, color='k', linestyle='--')
            ax5.set_xlabel('Time')
            ax5.set_ylabel('Residuals')
            ax5.set_title('Time vs Residuals')
            
            # 6. Residuals vs Observed
            ax6 = plt.subplot(326)
            ax6.scatter(observed, residuals, alpha=0.5)
            ax6.axhline(y=0, color='k', linestyle='--')
            ax6.set_xlabel('Observed Concentration')
            ax6.set_ylabel('Residuals')
            ax6.set_title('Residuals vs Observed')
            
            plt.tight_layout()
            self.results['diagnostic_plots'] = fig

            # Calculate metrics using the comprehensive validation function
            try:
                validation_metrics = calculate_validation_metrics(observed, predicted)
                
                # Add additional metrics that are not in the validation function
                mae = np.mean(np.abs(residuals)) if len(residuals) > 0 else np.nan
                mean_residual = np.mean(residuals) if len(residuals) > 0 else np.nan
                
                self.results['fit_metrics'] = {
                    'R2': validation_metrics['R2'],
                    'RMSE': validation_metrics['RMSE'],
                    'AFE': validation_metrics['AFE'],
                    'AAFE': validation_metrics['AAFE'],
                    'MAE': mae,
                    'Mean_Residual': mean_residual,
                    'N_valid_points': len(observed)
                }
            except Exception as e:
                print(f"Warning: Error calculating validation metrics: {str(e)}")
                # Fallback to basic metrics
                obs_mean = np.mean(observed)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((observed - obs_mean)**2)
                
                r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
                rmse = np.sqrt(np.mean(residuals**2)) if len(residuals) > 0 else np.nan
                mae = np.mean(np.abs(residuals)) if len(residuals) > 0 else np.nan
                mean_residual = np.mean(residuals) if len(residuals) > 0 else np.nan
                
                self.results['fit_metrics'] = {
                    'R2': r2,
                    'RMSE': rmse,
                    'AFE': np.nan,
                    'AAFE': np.nan,
                    'MAE': mae,
                    'Mean_Residual': mean_residual,
                    'N_valid_points': len(observed)
                }
            
            print(f"Metrics calculated using {len(observed)} valid data points")
            
        except Exception as e:
            print(f"Error in plot creation or metric calculation: {str(e)}")
            self.results['fit_metrics'] = {
                'R2': np.nan,
                'RMSE': np.nan,
                'AFE': np.nan,
                'AAFE': np.nan,
                'MAE': np.nan,
                'Mean_Residual': np.nan
            }
        
    def _add_standard_diagnostics(self, fig, predictions):
        """Helper method to add standard diagnostic plots"""
        observed = self.data['concentrations'].flatten()
        predicted = predictions.flatten()
        residuals = observed - predicted
        times_flat = np.repeat(self.times, len(self.data['concentrations']))
        
        # 2. Observed vs Predicted
        ax2 = plt.subplot(322)
        ax2.scatter(predicted, observed, alpha=0.5)
        min_val = min(predicted.min(), observed.min())
        max_val = max(predicted.max(), observed.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--')
        ax2.set_xlabel('Predicted Concentration')
        ax2.set_ylabel('Observed Concentration')
        ax2.set_title('Observed vs Predicted')
        
        # 3. Residuals vs Predicted
        ax3 = plt.subplot(323)
        ax3.scatter(predicted, residuals, alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='--')
        ax3.set_xlabel('Predicted Concentration')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residuals vs Predicted')
        
        # 4. QQ plot
        ax4 = plt.subplot(324)
        std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
        stats.probplot(std_residuals, dist="norm", plot=ax4)
        ax4.set_title('Normal Q-Q Plot of Residuals')
        
        # 5. Time vs Residuals
        ax5 = plt.subplot(325)
        ax5.scatter(times_flat, residuals, alpha=0.5)
        ax5.axhline(y=0, color='k', linestyle='--')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Residuals')
        ax5.set_title('Time vs Residuals')
        
        # 6. Residuals vs Observed
        ax6 = plt.subplot(326)
        ax6.scatter(observed, residuals, alpha=0.5)
        ax6.axhline(y=0, color='k', linestyle='--')
        ax6.set_xlabel('Observed Concentration')
        ax6.set_ylabel('Residuals')
        ax6.set_title('Residuals vs Observed')
        
    def run_analysis(self, create_plots: bool = False) -> Dict:
        """Run complete analysis workflow"""
        if not hasattr(self, 'data'):
            raise ValueError("No data available. Run generate_virtual_population first.")
            
        try:
            self.run_model_fitting()
        except Exception as e:
            print(f"Warning: Model fitting failed: {str(e)}")
            
        try:
            self.run_nca_analysis()
        except Exception as e:
            print(f"Warning: NCA analysis failed: {str(e)}")
            
        try:
            self.run_covariate_analysis()
        except Exception as e:
            print(f"Warning: Covariate analysis failed: {str(e)}")
            
        if create_plots:
            try:
                self.create_diagnostic_plots()
            except Exception as e:
                print(f"Warning: Creating diagnostic plots failed: {str(e)}")
            
        return self.results
        
    def print_summary(self) -> None:
        """Print comprehensive analysis summary"""
        print("\n===============================================")
        print("           PK Analysis Summary")
        print("===============================================")
        
        if 'model_fit' in self.results and self.results['model_fit']:
            print("\nFINAL PARAMETER ESTIMATES")
            print("-----------------------------------------------")
            print("Parameter    Estimate    CV%    [95% CI]")
            print("-----------------------------------------------")
            
            # Calculate parameter statistics
            for param, value in self.results['model_fit']['parameters'].items():
                if param in self.data['individual_parameters']:
                    values = self.data['individual_parameters'][param]
                    cv_percent = np.std(values) / np.mean(values) * 100
                    ci_lower = np.percentile(values, 2.5)
                    ci_upper = np.percentile(values, 97.5)
                    print(f"{param:<10} {value:9.3f} {cv_percent:7.1f}  [{ci_lower:.3f}, {ci_upper:.3f}]")
            
            # Between-subject variability (BSV)
            print("\nBETWEEN-SUBJECT VARIABILITY (CV%)")
            print("-----------------------------------------------")
            for param in self.model.parameters:
                if param in self.data['individual_parameters']:
                    values = self.data['individual_parameters'][param]
                    omega = np.std(np.log(values)) * 100  # Approximate CV%
                    print(f"{param:<10} {omega:7.1f}%")
            
            # Residual variability
            if 'predictions' in self.results:
                obs = self.data['concentrations'].flatten()
                pred = self.results['predictions'].flatten()
                valid_mask = (obs > 0) & (pred > 0)
                
                if np.any(valid_mask):
                    residuals = np.log(obs[valid_mask]) - np.log(pred[valid_mask])
                    sigma = np.std(residuals) * 100  # CV%
                    print("\nRESIDUAL VARIABILITY")
                    print("-----------------------------------------------")
                    print(f"Proportional Error (CV%): {sigma:.1f}%")
            
            # Goodness-of-fit statistics
            if 'fit_metrics' in self.results:
                print("\nGOODNESS-OF-FIT STATISTICS")
                print("-----------------------------------------------")
                metrics = self.results['fit_metrics']
                print(f"R-squared:        {metrics['R2']:.3f}")
                print(f"RMSE:             {metrics['RMSE']:.3f}")
                if 'AFE' in metrics and not np.isnan(metrics['AFE']):
                    print(f"AFE:              {metrics['AFE']:.3f}")
                if 'AAFE' in metrics and not np.isnan(metrics['AAFE']):
                    print(f"AAFE:             {metrics['AAFE']:.3f}")
                print(f"MAE:              {metrics['MAE']:.3f}")
                print(f"Mean Residual:    {metrics['Mean_Residual']:.3f}")
        else:
            print("\nModel fitting results not available")
        
        if 'nca' in self.results:
            print("\nNCA PARAMETERS SUMMARY")
            print("-----------------------------------------------")
            print(self.results['nca']['summary'].round(3))
            print(f"NCA Success Rate: {self.results['nca']['success_rate']*100:.1f}%")
        
        if 'covariate_analysis' in self.results and self.results['covariate_analysis']:
            print("\nSIGNIFICANT COVARIATES")
            print("-----------------------------------------------")
            for rel in self.results['covariate_analysis']['selected_relationships']:
                print(f"{rel.parameter}-{rel.covariate}: {rel.relationship}")
        
        print("\n===============================================")

    @classmethod
    def from_files(cls, model_type: str, 
                conc_file: str, 
                time_file: str, 
                demo_file: str = None,
                dose: float = 100.0,
                initial_params: dict = None):
        """
        Create workflow instance from data files
        
        Parameters:
        -----------
        model_type: str
            'onecomp' or 'onecomp_abs'
        conc_file: str
            Path to concentration CSV file
        time_file: str
            Path to time points CSV file
        demo_file: str, optional
            Path to demographics CSV file
        dose: float, optional
            Administered dose
        initial_params: dict, optional
            Initial parameter estimates. If None, uses default values
            
        Returns:
        --------
        BasePKWorkflow instance
        """
        # Load data
        conc_df = pd.read_csv(conc_file)
        time_df = pd.read_csv(time_file)
        demo_df = None if demo_file is None else pd.read_csv(demo_file)
        
        # Get concentration data
        time_cols = [col for col in conc_df.columns if col.startswith('Time_')]
        times = time_df['time'].values
        concentrations = conc_df[time_cols].values
        
        # Set default parameters if not provided
        if initial_params is None:
            if model_type == 'onecomp':
                initial_params = {
                    'CL': {'value': 5.0, 'cv_percent': 25},
                    'V': {'value': 50.0, 'cv_percent': 20}
                }
            elif model_type == 'onecomp_abs':
                initial_params = {
                    'Ka': {'value': 1.0, 'cv_percent': 30},
                    'CL': {'value': 5.0, 'cv_percent': 25},
                    'V': {'value': 50.0, 'cv_percent': 20}
                }
            elif model_type == 'twocomp':
                initial_params = {
                    'CL': {'value': 5.0, 'cv_percent': 25},
                    'V1': {'value': 30.0, 'cv_percent': 20},
                    'Q': {'value': 10.0, 'cv_percent': 30},
                    'V2': {'value': 50.0, 'cv_percent': 25}
                }
            elif model_type == 'twocomp_abs':
                initial_params = {
                    'Ka': {'value': 1.0, 'cv_percent': 30},
                    'CL': {'value': 5.0, 'cv_percent': 25},
                    'V1': {'value': 30.0, 'cv_percent': 20},
                    'Q': {'value': 10.0, 'cv_percent': 30},
                    'V2': {'value': 50.0, 'cv_percent': 25}
                }
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create model and workflow
        model = create_pkpy_model(model_type, initial_params)
        workflow = cls(model, n_subjects=len(conc_df))
    
        # Ensure concentration data is properly formatted
        concentrations = np.array(concentrations, dtype=float)
        
        # Set data with validation
        workflow.times = np.array(times, dtype=float)
        workflow.data = {
            'times': workflow.times,
            'concentrations': concentrations,
            'dose': float(dose),
            'demographics': demo_df
        }
        
        # Initial model fitting to get predictions
        try:
            workflow.run_model_fitting()
            if workflow.results.get('model_fit'):
                workflow.data['individual_parameters'] = pd.DataFrame(
                    workflow.results['model_fit']['individual_parameters']
                )
            else:
                print("Warning: Initial model fitting failed")
        except Exception as e:
            print(f"Warning: Error during initial fitting: {str(e)}")
        
        return workflow

    def save_results(self, output_dir: str = '.') -> None:
        """
        Save analysis results to files
        
        Parameters:
        -----------
        output_dir: str
            Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save NCA results if available
        if 'nca' in self.results:
            self.results['nca']['individual'].to_csv(
                os.path.join(output_dir, 'nca_results.csv'), 
                index=False
            )
            
        # Save model fitting results if available
        if 'model_fit' in self.results:
            pd.DataFrame([self.results['model_fit']['parameters']]).to_csv(
                os.path.join(output_dir, 'model_parameters.csv'),
                index=False
            )
            
        # Save covariate analysis results if available
        if 'covariate_analysis' in self.results and self.results['covariate_analysis']:
            covariate_results = []
            for rel in self.results['covariate_analysis']['selected_relationships']:
                covariate_results.append({
                    'parameter': rel.parameter,
                    'covariate': rel.covariate,
                    'relationship': rel.relationship,
                    'coefficient': rel.coefficient,
                    'center': rel.center
                })
            
            pd.DataFrame(covariate_results).to_csv(
                os.path.join(output_dir, 'covariate_relationships.csv'),
                index=False
            )
            
        # Save diagnostic plots if available
        if 'diagnostic_plots' in self.results:
            self.results['diagnostic_plots'].savefig(
                os.path.join(output_dir, 'diagnostics.png')
            )

    def create_nca_plots(self) -> Dict:
        """Create NCA diagnostic plots"""
        if 'nca' not in self.results:
            raise ValueError("No NCA results available. Run run_nca_analysis first.")
        
        nca_data = self.results['nca']['individual']
        
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Mean concentration-time profile with SD
        ax1 = plt.subplot(231)
        ax2 = plt.subplot(232)
        
        # 데이터 차원 처리
        concentrations = self.data['concentrations']
        if len(concentrations.shape) == 3:
            concentrations = concentrations.squeeze(2)  # 3차원을 2차원으로
        
        mean_conc = np.mean(concentrations, axis=0)
        std_conc = np.std(concentrations, axis=0)
        
        # 선형 스케일 플롯
        ax1.plot(self.times, mean_conc, 'r-', linewidth=2, label='Mean')
        ax1.fill_between(
            self.times,
            mean_conc - std_conc,
            mean_conc + std_conc,
            color='r',
            alpha=0.2,
            label='±1 SD'
        )
        ax1.set_xlabel('Time (h)')
        ax1.set_ylabel('Concentration')
        ax1.set_title('Mean Concentration-Time Profile')
        ax1.legend()
        
        # 로그 스케일 플롯
        ax2.plot(self.times, mean_conc, 'r-', linewidth=2, label='Mean')
        ax2.fill_between(
            self.times,
            np.maximum(mean_conc - std_conc, 1e-10),  # 음수 방지
            mean_conc + std_conc,
            color='r',
            alpha=0.2,
            label='±1 SD'
        )
        ax2.set_yscale('log')
        ax2.set_xlabel('Time (h)')
        ax2.set_ylabel('Concentration (log scale)')
        ax2.set_title('Mean Concentration-Time Profile (Log Scale)')
        ax2.legend()
        
        # 2. Terminal elimination phase visualization
        ax3 = plt.subplot(233)
        terminal_idx = -3  # Last 3 points for terminal phase
        
        for conc in self.data['concentrations']:
            log_conc = np.log(np.maximum(conc, 1e-10))
            ax3.plot(self.times[terminal_idx:], log_conc[terminal_idx:], 'b-', alpha=0.2)
        
        log_mean = np.log(np.maximum(mean_conc, 1e-10))
        ax3.plot(self.times[terminal_idx:], log_mean[terminal_idx:], 'r-', linewidth=2)
        ax3.set_xlabel('Time (h)')
        ax3.set_ylabel('Log Concentration')
        ax3.set_title('Terminal Elimination Phase')
        ax3.grid(True)
        
        # 3. Distribution of key NCA parameters
        ax4 = plt.subplot(234)
        ax5 = plt.subplot(235)
        ax6 = plt.subplot(236)
        
        sns.histplot(data=nca_data['AUC'], ax=ax4, bins=15)
        ax4.set_xlabel('AUC')
        ax4.set_title('AUC Distribution')
        
        sns.histplot(data=nca_data['half_life'], ax=ax5, bins=15)
        ax5.set_xlabel('Half-life (h)')
        ax5.set_title('Half-life Distribution')
        
        sns.histplot(data=nca_data['clearance'], ax=ax6, bins=15)
        ax6.set_xlabel('Clearance (L/h)')
        ax6.set_title('Clearance Distribution')
        
        plt.tight_layout()
        
        # Calculate summary statistics
        summary_stats = {
            param: {
                'mean': nca_data[param].mean(),
                'sd': nca_data[param].std(),
                'cv%': (nca_data[param].std() / nca_data[param].mean() * 100),
                'median': nca_data[param].median(),
                'range': [nca_data[param].min(), nca_data[param].max()]
            }
            for param in ['AUC', 'Cmax', 'Tmax', 'half_life', 'clearance', 'MRT']
        }
        
        return {
            'figure': fig,
            'summary_stats': summary_stats
        }

    def create_nca_summary(self) -> Dict:
        """
        Create a comprehensive summary of NCA analysis results
        
        Returns:
        --------
        Dict containing summary statistics and flags for potential issues
        """
        if 'nca' not in self.results:
            raise ValueError("NCA analysis must be run first using run_nca_analysis()")
        
        nca_data = self.results['nca']['individual']
        
        # Calculate basic statistics for each parameter
        params_of_interest = ['Cmax', 'Tmax', 'AUC', 'half_life', 'clearance', 'MRT', 'Vss']
        basic_stats = {}
        
        for param in params_of_interest:
            if param in nca_data:
                data = nca_data[param].values
                valid_data = data[~np.isnan(data) & ~np.isinf(data)]
                
                if len(valid_data) > 0:
                    # Arithmetic statistics
                    arithmetic_mean = np.mean(valid_data)
                    arithmetic_sd = np.std(valid_data)
                    cv_percent = (arithmetic_sd / arithmetic_mean * 100) if arithmetic_mean != 0 else np.nan
                    
                    # Geometric statistics
                    log_data = np.log(np.maximum(valid_data, 1e-10))
                    geometric_mean = np.exp(np.mean(log_data))
                    geometric_cv = np.sqrt(np.exp(np.var(log_data)) - 1) * 100
                    
                    basic_stats[param] = {
                        'n': len(valid_data),
                        'arithmetic_mean': arithmetic_mean,
                        'arithmetic_sd': arithmetic_sd,
                        'cv_percent': cv_percent,
                        'geometric_mean': geometric_mean,
                        'geometric_cv': geometric_cv,
                        'median': np.median(valid_data),
                        'range': [np.min(valid_data), np.max(valid_data)],
                        'missing_values': len(data) - len(valid_data)
                    }
                else:
                    basic_stats[param] = None
        
        # Calculate additional metrics and quality checks
        quality_flags = {
            'terminal_r2': [],  # R² for terminal elimination fit
            'auc_extrapolation': [],  # % AUC extrapolated
            'sampling_adequacy': []  # Check if sampling schedule is adequate
        }
        
        # Check terminal phase estimation quality
        if 'half_life' in nca_data:
            terminal_idx = -3
            log_conc = np.log(np.maximum(self.data['concentrations'][:, terminal_idx:], 1e-10))
            terminal_times = self.times[terminal_idx:]
            
            for i in range(len(log_conc)):
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        terminal_times, log_conc[i]
                    )
                    quality_flags['terminal_r2'].append(r_value**2)
                except:
                    quality_flags['terminal_r2'].append(np.nan)
        
        # Check AUC extrapolation
        if 'AUC' in nca_data:
            total_auc = nca_data['AUC'].values
            observed_auc = np.trapz(
                self.data['concentrations'],
                self.times,
                axis=1
            )
            quality_flags['auc_extrapolation'] = (
                (total_auc - observed_auc) / total_auc * 100
            )
        
        # Check sampling schedule adequacy
        diff_times = np.diff(self.times)
        quality_flags['sampling_adequacy'] = {
            'absorption_density': np.min(diff_times[:3]),  # Time gap in absorption phase
            'elimination_density': np.max(diff_times[-3:]),  # Time gap in elimination phase
            'total_duration': self.times[-1] - self.times[0]  # Total observation period
        }
        
        return {
            'basic_stats': basic_stats,
            'quality_flags': quality_flags,
            'success_rate': self.results['nca'].get('success_rate', 0)
        }

    def print_nca_summary(self) -> None:
        """Print formatted NCA analysis summary"""
        summary = self.create_nca_summary()
        
        print("\n============================================")
        print("         NCA Analysis Summary")
        print("============================================")
        
        # Print success rate
        print(f"\nAnalysis Success Rate: {summary['success_rate']*100:.1f}%")
        
        # Print basic statistics for each parameter
        print("\nParameter Estimates")
        print("--------------------------------------------")
        
        for param, stats in summary['basic_stats'].items():
            if stats is not None:
                print(f"\n{param}:")
                print(f"  N = {stats['n']} (missing: {stats['missing_values']})")
                print(f"  Arithmetic mean ± SD: {stats['arithmetic_mean']:.3f} ± {stats['arithmetic_sd']:.3f}")
                print(f"  CV%: {stats['cv_percent']:.1f}")
                print(f"  Geometric mean (CV%): {stats['geometric_mean']:.3f} ({stats['geometric_cv']:.1f}%)")
                print(f"  Median: {stats['median']:.3f}")
                print(f"  Range: [{stats['range'][0]:.3f}, {stats['range'][1]:.3f}]")
        
        # Print quality checks
        print("\nQuality Checks")
        print("--------------------------------------------")
        
        # Terminal phase quality
        if summary['quality_flags']['terminal_r2']:
            r2_values = np.array(summary['quality_flags']['terminal_r2'])
            valid_r2 = r2_values[~np.isnan(r2_values)]
            if len(valid_r2) > 0:
                print(f"\nTerminal phase R²:")
                print(f"  Mean: {np.mean(valid_r2):.3f}")
                print(f"  Range: [{np.min(valid_r2):.3f}, {np.max(valid_r2):.3f}]")
                print(f"  Subjects with R² < 0.9: {np.sum(valid_r2 < 0.9)}")
        
        # AUC extrapolation
        if len(summary['quality_flags']['auc_extrapolation']) > 0:
            extrap = summary['quality_flags']['auc_extrapolation']
            print(f"\nAUC extrapolation:")
            print(f"  Mean %: {np.mean(extrap):.1f}")
            print(f"  Subjects with >20% extrapolation: {np.sum(extrap > 20)}")
        
        # Sampling adequacy
        samp = summary['quality_flags']['sampling_adequacy']
        print(f"\nSampling Schedule:")
        print(f"  Minimum time gap (absorption): {samp['absorption_density']:.2f} h")
        print(f"  Maximum time gap (elimination): {samp['elimination_density']:.2f} h")
        print(f"  Total observation period: {samp['total_duration']:.1f} h")
        
        print("\n============================================")

    
def generate_example_data(model_type='onecomp_abs'):
    """Generate example data for testing"""
    # Set parameters based on model type
    if model_type == 'onecomp':
        parameters = {
            'CL': {'value': 5.0, 'cv_percent': 25},
            'V': {'value': 50.0, 'cv_percent': 20}
        }
    else:  # onecomp_abs
        parameters = {
            'Ka': {'value': 1.0, 'cv_percent': 30},
            'CL': {'value': 5.0, 'cv_percent': 25},
            'V': {'value': 50.0, 'cv_percent': 20}
        }
    
    # Create model and workflow
    model = create_pkpy_model(model_type, parameters)
    workflow = BasePKWorkflow(model, n_subjects=50)
    
    # Generate time points based on model type
    if model_type == 'onecomp':
        time_points = np.linspace(0, 24, 13)  # Uniform sampling
    else:
        time_points = np.concatenate([
            np.linspace(0, 4, 9),    # Dense early sampling
            np.linspace(6, 24, 7)     # Sparse later sampling
        ])
    
    # Generate population
    workflow.generate_virtual_population(
        time_points=time_points,
        dose=500.0,
        demographic_covariates={
            'WT': ('truncnorm', 70, 20, 40, 120),
            'AGE': ('truncnorm', 45, 20, 18, 80),
            'CRCL': ('truncnorm', 100, 30, 30, 150)
        }
    )
    
    # Save files in format ready for analysis
    conc_df = pd.DataFrame(
        workflow.data['concentrations'],
        columns=[f'Time_{t:.1f}h' for t in time_points]
    )
    conc_df['ID'] = range(len(conc_df))
    
    time_df = pd.DataFrame({'time': time_points})
    demo_df = workflow.data['demographics']
    
    # Save to CSV
    conc_df.to_csv('concentrations.csv', index=False)
    time_df.to_csv('times.csv', index=False)
    demo_df.to_csv('demographics.csv', index=False)