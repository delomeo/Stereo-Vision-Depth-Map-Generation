# pylint: skip-file
import numpy as np
import cv2 as cv
from typing import Tuple
from stereoobject import ImageLR

def mae(src: np.array, groundtruth: np.array) -> float:
    """
    Computes the Mean Absolute Error (MAE) between the source and ground truth disparity maps.
    :param src: Source disparity map.
    :param groundtruth: Ground truth disparity map.
    :return: Mean Absolute Error.
    """
    if src.shape != groundtruth.shape:
        raise ValueError("Source and ground truth disparity maps must have the same shape.")
    
    return np.mean(np.abs(src - groundtruth))

def rmse(src: np.array, groundtruth: np.array) -> float:
    """
    Computes the Root Mean Square Error (RMSE) between the source and ground truth disparity maps.
    :param src: Source disparity map.
    :param groundtruth: Ground truth disparity map.
    :return: Root Mean Square Error.
    """
    if src.shape != groundtruth.shape:
        raise ValueError("Source and ground truth disparity maps must have the same shape.")
    
    return np.sqrt(np.mean((src - groundtruth) ** 2))
