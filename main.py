# pylint: skip-file
import numpy as np
import cv2 as cv
import re
import matplotlib.pyplot as plt
from stereoobject import ImageLR, show_image_pairs
from utils import read_image, read_disparity

    
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

    disp0 = read_disparity(disp0_path)
    disp1 = read_disparity(disp1_path)

     
    # Pack an LR StereoObject
    artroomLR = ImageLR(imgL, imgR)

    # Stereo SGBM Parameters
    min_disp = 0
    num_disp = 16 * 15  # Needs to be divisible by 16
    window_size = 7  # Must be odd

    # Create StereoSGBM object
    # Note: The parameters here are just an example, you may need to adjust them based
    stereo = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=9,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63
    )
    
    print('computing disparity...')
    disp = stereo.compute(artroomLR.left, artroomLR.right).astype(np.float32) / 32.0

    
    h, w = imgL.shape[:2]

    f = 0.8*w     
    
                         # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    
    points = cv.reprojectImageTo3D(disp, Q)
    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp-min_disp)/num_ disp)
    cv.waitKey()

    print('Done')

if __name__ == "__main__":
    main()