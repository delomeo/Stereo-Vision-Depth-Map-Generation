# pylint: skip-file
from dataclasses import dataclass

import numpy as np
import cv2 as cv

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
    # Open pair of images
    imgL_path = './data/artroom1/im0.png'
    imgR_path = './data/artroom1/im1.png'

    imgL = read_image(imgL_path)
    imgR = read_image(imgR_path)
     
    # Pack an LR StereoObject
    artroomLR = ImageLR(imgL, imgR)

    # show_image_pairs(artroomLR)

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
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()

    print('Done')

if __name__ == "__main__":
    main()