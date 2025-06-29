# pylint: skip-file
import numpy as np  
import cv2 as cv
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ImageLR:
    left: np.array
    right: np.array


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
