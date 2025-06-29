# pylint: skip-file
import numpy as np
import cv2 as cv
import re
import matplotlib.pyplot as plt
from stereoobject import ImageLR, show_image_pairs, compute_disparity
from utils import read_image, read_disparity
from metrics import mae, rmse

    
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

    ROOT_DIR = './data/artroom1/'

    imgL_filename = 'im0.png'
    imgR_filename = 'im1.png'

    # Ground truth disparity filename
    disp0_filename = 'disp0.pfm'
    disp1_filename = 'disp1.pfm'

    # Read the images
    imgL_path = ROOT_DIR + imgL_filename
    imgR_path = ROOT_DIR + imgR_filename

    imgL = read_image(imgL_path)
    imgR = read_image(imgR_path)

    # Read the disparity maps
    disp0_path = ROOT_DIR + disp0_filename
    disp1_path = ROOT_DIR + disp1_filename

    disp0, disp0_scale = read_disparity(disp0_path)
    disp1, disp1_scale = read_disparity(disp1_path)

    # Pack an LR StereoObject
    artroomLR = ImageLR(imgL, imgR)

    print('computing disparity...')
    disp = compute_disparity(artroomLR)

    print('Done!')
    
    # Compute metrics
    print('computing metrics...')
    mae_value_left = mae(disp, disp0)
    rmse_value_left = rmse(disp, disp0)

    mae_value_right = mae(disp, disp1)
    rmse_value_right = rmse(disp, disp1)

    print(f'MAE Left: {mae_value_left}')
    print(f'MAE Right: {mae_value_right}')
    
    print(f'RMSE Left: {rmse_value_left}')
    print(f'RMSE Right: {rmse_value_right}')
    
if __name__ == "__main__":
    main()