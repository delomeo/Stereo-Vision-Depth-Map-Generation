# pylint: skip-file
import numpy as np  
import cv2 as cv
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ImageLR:
    left: np.array
    right: np.array

def compute_disparity(image_pairs: ImageLR, min_disp: int = 0, num_disp: int = 16 * 15, window_size: int = 7) -> np.array:
    """
    Computes the disparity map using StereoSGBM.

    :param image_pairs: An ImageLR object containing left and right images.
    :param min_disp: Minimum disparity value.
    :param num_disp: Number of disparities (must be divisible by 16).
    :param window_size: Size of the block window (must be odd).
    :return: Disparity map as a numpy array.
    """
    # Check if type is correct
    if not isinstance(image_pairs, ImageLR): 
        raise TypeError("Image Pairs must always be passed as a Stereo Object!")
    
    # Create StereoSGBM object
    stereo = cv.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63
    )
    
    # Compute disparity map
    disparity = stereo.compute(image_pairs.left, image_pairs.right).astype(np.float32) / 16.0
    
    return disparity

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
