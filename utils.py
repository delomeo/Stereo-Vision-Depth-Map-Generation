# pylint: skip-file
import numpy as np
import cv2 as cv
import re

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

    if np.isinf(data).any():
        data[data == np.inf] = -1  # Handle infinite values
    return np.reshape(data, shape), scale
def show_object_depth_map(image_pairs: ImageLR, disparity: np.array) -> None:
    """
    Displays the left image and the computed disparity map side by side.
    
    :param image_pairs: An ImageLR object containing left and right images.
    :param disparity: The computed disparity map.
    """
