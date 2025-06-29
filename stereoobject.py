# pylint: skip-file
import numpy as np  
from dataclasses import dataclass

@dataclass
class ImageLR:
    left: np.array
    right: np.array
