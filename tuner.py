import numpy as np
from sklearn.model_selection import ParameterGrid
from stereoobject import ImageLR, compute_disparity
from metrics import mae, rmse


class StereoSGBMParameterTuner:
    """
    Class to tune StereoSGBM parameters using grid search.
    """


    def __init__(self, param_grid, metric='mae'):
        """
        Initialize the tuner with a parameter grid.
        
        :param param_grid: Dictionary of parameters to tune.
        """
        self.param_grid = param_grid
        self.metric = metric

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
        error = None

        for params in ParameterGrid(self.param_grid):
            disp = compute_disparity(stereo_obj, **params)
            if disp is None:
                continue

            if self.metric == 'rmse':
                error = self._use_rmse(disp, disp0, disp1)
            elif self.metric == 'mae':
                error = self._use_mae(disp, disp0, disp1)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            print(f"Parameters: {params}, Error: {error}")
            if error < best_error:
                best_error = error
                best_params = params
        return best_params, best_error
    
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