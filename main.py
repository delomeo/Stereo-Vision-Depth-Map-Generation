# pylint: skip-file
from dataclasses import dataclass

import numpy as np
import cv2 as cv
import re
import matplotlib.pyplot as plt

@dataclass
class ImageLR:
    left: np.array
    right: np.array

def read_image(image_path) -> np.array: 
    """
    Reads an image from the specified path.
    
    :param image_path: Path to the image file.
    :return: The image as a numpy array.
    """
    image = cv.imread(image_path, cv.IMREAD_ANYCOLOR)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be read.")
    return image

def read_disparity(disp_path) -> np.array:
    """
    Reads a disparity map from the specified path.
    
    :param disp_path: Path to the disparity file.
    :return: The disparity map as a numpy array.
    """
    file = open(disp_path, 'rb')
    
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode('ascii') == 'PF':
        color = True    
    elif header.decode('ascii') == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.search(r'(\d+)\s(\d+)', file.readline().decode('ascii'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape), scale

def show_image_pairs(image_pairs: ImageLR) -> None:
    """
    Displays a pair of images side by side.

    :param image_pairs: An ImageLR object containing left and right images.
    :raises TypeError: If image_pairs is not an instance of ImageLR.
    """
    # Check if type is correct
    if not isinstance(image_pairs, ImageLR): 
        raise TypeError("Image Pairs must always be passed as a Stereo Object!")
    
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(cv.cvtColor(image_pairs.left, cv.COLOR_BGR2RGB))
    ax[0].set_title('Left Image')
    ax[0].axis('off')   
    ax[1].imshow(cv.cvtColor(image_pairs.right, cv.COLOR_BGR2RGB))
    ax[1].set_title('Right Image')
    ax[1].axis('off')
    plt.show()

    
def main():

    # EXAMPLE: 

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
     
    # Pack an LR StereoObject
    artroomLR = ImageLR(imgL, imgR)

    # show_image_pairs(artroomLR)

    # Read the disparity maps
    disp0_path = ROOT_DIR + disp0_filename
    disp1_path = ROOT_DIR + disp1_filename
    disp0 = read_disparity(disp0_path)
    disp1 = read_disparity(disp1_path)

    print(disp0)
    
    # # Stereo SGBM Parameters
    # min_disp = 0
    # num_disp = 16 * 15  # Needs to be divisible by 16
    # window_size = 7  # Must be odd

    # # Create StereoSGBM object
    # # Note: The parameters here are just an example, you may need to adjust them based
    # stereo = cv.StereoSGBM_create(
    #     minDisparity=min_disp,
    #     numDisparities=num_disp,
    #     blockSize=window_size,
    #     P1=8 * 3 * window_size ** 2,
    #     P2=32 * 3 * window_size ** 2,
    #     disp12MaxDiff=9,
    #     uniquenessRatio=10,
    #     speckleWindowSize=100,
    #     speckleRange=32,
    #     preFilterCap=63
    # )
    
    # print('computing disparity...')
    # disp = stereo.compute(artroomLR.left, artroomLR.right).astype(np.float32) / 32.0

    
    # h, w = imgL.shape[:2]

    # f = 0.8*w     
    
    #                      # guess for focal length
    # Q = np.float32([[1, 0, 0, -0.5*w],
    #                 [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
    #                 [0, 0, 0,     -f], # so that y-axis looks up
    #                 [0, 0, 1,      0]])
    
    # points = cv.reprojectImageTo3D(disp, Q)
    # colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    # mask = disp > disp.min()
    # out_points = points[mask]
    # out_colors = colors[mask]

    # cv.imshow('left', imgL)
    # cv.imshow('disparity', (disp-min_disp)/num_ disp)
    # cv.waitKey()

    # print('Done')

if __name__ == "__main__":
    main()