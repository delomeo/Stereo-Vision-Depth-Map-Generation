# pylint: skip-file
import numpy as np  
import cv2 as cv
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ImageLR:
    left: np.array
    right: np.array

def compute_disparity(image_pairs: ImageLR, **params) -> np.array:
    """
    Computes the disparity map using StereoSGBM.

    :param image_pairs: An ImageLR object containing left and right images.
    :param params: Parameters for StereoSGBM.
    :return: Disparity map as a numpy array.
    :raises TypeError: If image_pairs is not an instance of ImageLR.
    """
    # Check if type is correct
    if not isinstance(image_pairs, ImageLR): 
        raise TypeError("Image Pairs must always be passed as a Stereo Object!")
    
    # Create StereoSGBM object
    stereo = cv.StereoSGBM_create(
        minDisparity=params.get('minDisparity', 0),
        numDisparities=params.get('numDisparities', 16),
        blockSize=params.get('blockSize', 5),
        P1=params.get('P1', 8 * 3 * 5 ** 2),
        P2=params.get('P2', 32 * 3 * 5 ** 2),
        disp12MaxDiff=params.get('disp12MaxDiff', 1),   
        uniquenessRatio=params.get('uniquenessRatio', 10),
        speckleWindowSize=params.get('speckleWindowSize', 50),
        speckleRange=params.get('speckleRange', 1),
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY if params.get('mode', 'SGBM_3WAY') == 'SGBM_3WAY' else cv.STEREO_SGBM_MODE_SGBM
    )

    print("Using StereoSGBM with parameters:")
    # Get parameters from StereoSGBM object
    for key, value in stereo.getParams().items():
        print(f"{key}: {value}")
    
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
