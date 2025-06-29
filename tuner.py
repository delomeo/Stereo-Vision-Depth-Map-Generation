# pylint: skip-file
import numpy as np
import cv2 as cv
import optuna
from stereoobject import ImageLR
from metrics import mae, rmse


class StereoSGBMParameterTuner:
    """
    Class to tune StereoSGBM parameters using grid search.
    """


    def __init__(self, param_grid, metric='mae', verbose=True):
        """
        Initialize the tuner with a parameter grid.
        
        :param param_grid: Dictionary of parameters to tune.
        """
        self.param_grid = param_grid
        self.metric = metric
        self.verbose = verbose

    def tune(self, stereo_obj, disp0, disp1):
        """
        Perform grid search to find the best parameters.
        
        :param compute_disparity_func: Function to compute disparity.
        :param imgLR: StereoObject containing left and right images.
        :param disp0: Ground truth disparity for left image.
        :param disp1: Ground truth disparity for right image.
        :return: Best parameters and their corresponding error.
        """
        best_params = None
        best_error = float('inf')

        if self.method == 'grid':
            study = optuna.create_study(sampler=optuna.samplers.GridSampler(self.param_grid))
        elif self.method == 'tpe':
            study = optuna.create_study(sampler=optuna.samplers.TPESampler(self.param_grid))
        elif self.method == 'cmaes':
            study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(self.param_grid))
        elif self.method == 'nsga2':
            study = optuna.create_study(sampler=optuna.samplers.NSGAIISampler(self.param_grid))
        elif self.method == 'random':
            study = optuna.create_study(sampler=optuna.samplers.RandomSampler(self.param_grid))
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        def objective(trial, stereo_obj, disp0, disp1):

            error = None
            params = {
                'minDisparity': trial.suggest_int('minDisparity', 0, 32),
                'numDisparities': trial.suggest_int('numDisparities', 16, 128, step=16),
                'blockSize': trial.suggest_int('blockSize', 5, 21, step=2),
                'P1': trial.suggest_int('P1', 8 * 3 * 5 ** 2, 8 * 3 * 21 ** 2),
                'P2': trial.suggest_int('P2', 32 * 3 * 5 ** 2, 32 * 3 * 21 ** 2),
                'disp12MaxDiff': trial.suggest_int('disp12MaxDiff', 1, 10),
                'uniquenessRatio': trial.suggest_int('uniquenessRatio', 5, 20),
                'speckleWindowSize': trial.suggest_int('speckleWindowSize', 50, 200),
                'speckleRange': trial.suggest_int('speckleRange', 1, 50),
                'mode': trial.suggest_categorical('mode', ['SGBM_3WAY', 'SGBM'])
            }
            disp = self.compute_disparity(stereo_obj, **params)
            if disp is None:
                continue

            if self.metric == 'rmse':
                error = self._use_rmse(disp, disp0, disp1)
            elif self.metric == 'mae':
                error = self._use_mae(disp, disp0, disp1)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            if self.verbose:
                # Print the parameters and error for each iteration
                print(f"Parameters: {params}, Error: {error}")
            if error < best_error:
                best_error = error
                best_params = params
        return best_params, best_error
        # Run the optimization
        study.optimize(lambda trial: objective(trial, stereo_obj, disp0, disp1), n_trials=100, n_jobs=-1, show_progress_bar=self.verbose)
        
        return study.best_params, study.best_value
        
    
    def _use_rmse(self, disp, disp0, disp1):
        """
        Compute RMSE for left and right disparities.
        
        :param disp: Computed disparity map.
        :param disp0: Ground truth disparity for left image.
        :param disp1: Ground truth disparity for right image.
        :return: Average RMSE error.
        """
        error_left = rmse(disp, disp0)
        error_right = rmse(disp, disp1)
        return (error_left + error_right) / 2
    
    def _use_mae(self, disp, disp0, disp1):
        """
        Compute MAE for left and right disparities.
        
        :param disp: Computed disparity map.
        :param disp0: Ground truth disparity for left image.
        :param disp1: Ground truth disparity for right image.
        :return: Average MAE error.
        """
        error_left = mae(disp, disp0)
        error_right = mae(disp, disp1)
        return (error_left + error_right) / 2