# pylint: skip-file
import glob
import os
import numpy as np
import cv2 as cv
import re
import matplotlib.pyplot as plt
from stereoobject import ImageLR, show_image_pairs
from utils import read_image, read_disparity
from metrics import mae, rmse

from tuner import StereoSGBMParameterTuner

    
def main():

    '''
    Main function to compute disparity map using StereoSGBM.
    1. Reads left and right images.
    2. Reads ground truth disparity maps.
    3. Computes disparity map using StereoSGBM.
    4. Reprojects disparity map to 3D points.
    5. Displays the left image and computed disparity map.
    6. Prints 'Done' when finished.
    '''
    # Define the root directory for the dataset
    # This should match the directory structure of your dataset
    # The glob pattern will match all directories starting with 'artroom'
    for dir in glob.glob('data/*'):
        
        ROOT_DIR = dir + '/'

        imgL_filename = 'im0.png'
        imgR_filename = 'im1.png'

        # Ground truth disparity filename
        disp0_filename = 'disp0.pfm'
        disp1_filename = 'disp1.pfm'
        print('-' * 30)
        print(f'Processing directory: {os.path.basename(dir)}')
        # Read the images
        imgL_path = ROOT_DIR + imgL_filename
        imgR_path = ROOT_DIR + imgR_filename
        print('-' * 30)    
        # print(f'Reading images from {os.path.basename(dir)}...')
        imgL = read_image(imgL_path)
        imgR = read_image(imgR_path)
        # print('-' * 30)
        # Read the disparity maps
        disp0_path = ROOT_DIR + disp0_filename
        disp1_path = ROOT_DIR + disp1_filename

        # print(f'Reading disparity maps from {os.path.basename(dir)}...') 
        disp0, disp0_scale = read_disparity(disp0_path)
        disp1, disp1_scale = read_disparity(disp1_path)
        # print('-' * 30)

        # Pack an LR StereoObject
        artroomLR = ImageLR(imgL, imgR)

        stereo_default = cv.StereoSGBM_create()
        # print('Computing disparity...')
        disp = stereo_default.compute(artroomLR.left, artroomLR.right).astype(np.float32) / 16.0

        # print('Done!')
        # print('-' * 30)

        # Compute metrics
        # print('Computing metrics...')
        # mae_value_left = mae(disp, disp0)
        rmse_value_left = rmse(disp, disp0)

        # mae_value_right = mae(disp, disp1)
        rmse_value_right = rmse(disp, disp1)

        # avg_mae = (mae_value_left + mae_value_right) / 2
        avg_rmse = (rmse_value_left + rmse_value_right) / 2
        print('-' * 30)
        # Automatic tuning of StereoSGBM parameters
        print('Tuning StereoSGBM parameters...')
        param_grid = {
            'minDisparity': [0, 16, 32],
            'numDisparities': [16, 32, 64],
            'blockSize': [5, 7, 9, 11],
            'P1': [8 * 3 * 5 ** 2, 8 * 3 * 7 ** 2],
            'P2': [32 * 3 * 5 ** 2, 32 * 3 * 7 ** 2],
            'disp12MaxDiff': [1, 2, 3, 4, 5],
            'uniquenessRatio': [5, 10],
            'speckleWindowSize': [50, 100],
            'speckleRange': [1, 5, 10, 20, 50],
            'mode': ['SGBM_3WAY', 'SGBM']
        }

        tuner = StereoSGBMParameterTuner(param_grid, method="grid", metric='rmse', verbose=False, save_results=True)
        best_params, best_error = tuner.tune((artroomLR), disp0, disp1)
        print('-' * 60)
        print('Tuning completed!')
        print('-' * 60)
        print('Best parameters found:')
        print(best_params)
        print('-' * 60)
        print(f'Best error: {best_error}')
        print('-' * 60)
        print(f'Initial Error (rand values): {avg_rmse}')

if __name__ == "__main__":
    main()