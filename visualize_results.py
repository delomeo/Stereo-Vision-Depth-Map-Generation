# pylint: skip-file

import os
import numpy as np
from stereoobject import ImageLR, compute_disparity
from utils import read_image, read_disparity, show_object_depth_map

def visualize_results(image_pairs: ImageLR, disparity: np.array) -> None:
    """
    Visualizes the results of the disparity computation.
    
    :param image_pairs: An ImageLR object containing left and right images.
    :param disparity: Computed disparity map.
    :raises TypeError: If image_pairs is not an instance of ImageLR.
    """    # Check if type is correct
    if not isinstance(image_pairs, ImageLR):
        raise TypeError("Image Pairs must always be passed as a Stereo Object!")
    
    # Show the left image and the computed disparity map
    show_object_depth_map(image_pairs, disparity)   

    print("Visualization complete.")

if __name__ == "__main__":
    # Example usage
    imgL_path = 'data/artroom2/im0.png'
    imgR_path = 'data/artroom2/im1.png'
    disp0_path = 'data/artroom2/disp0.pfm'
    disp1_path = 'data/artroom2/disp1.pfm'

    params = {
        'P1': 1176, 'P2': 4704, 'blockSize': 5, 'disp12MaxDiff': 2, 'minDisparity': 16, 'numDisparities': 64, 'speckleRange': 2, 'speckleWindowSize': 50, 'uniquenessRatio': 5
    }
    imgL = read_image(imgL_path)
    imgR = read_image(imgR_path)
    # disp0, _ = read_disparity(disp0_path)
    # disp1, _ = read_disparity(disp1_path)

    image_pairs = ImageLR(imgL, imgR)
    disparity = compute_disparity(image_pairs, **params)

    visualize_results(image_pairs, disparity)