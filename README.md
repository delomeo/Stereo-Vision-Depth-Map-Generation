# Stereo-Vision-Depth-Map-Generation

## Overview
This project aims to create an automatic tuner for rectified stereo image pairs using optimization algorithms. It showcases applications of stereo matching algorithms and compares them to recent techniques leveraging Deep Learning.

## Features
- **Stereo Image Pair Handling**: Efficiently processes left and right images.
- **Disparity Map Computation**: Uses StereoSGBM to compute disparity maps.
- **3D Point Reprojection**: Converts disparity maps into 3D points.
- **Visualization**: Displays stereo image pairs and computed disparity maps.
- **Optimization Algorithms**: Implements tuners for parameter optimization.

## Code Structure
- `main.py`: Contains the main function to compute disparity maps and reproject them to 3D points.
- `utils.py`: Provides utility functions for reading images and disparity maps.
- `stereoobject.py`: Defines the `ImageLR` class and functions for visualizing stereo image pairs.
- `tuner.py`: Placeholder for optimization algorithms to tune stereo matching parameters.
- `metrics.py`: Placeholder for evaluating the performance of stereo matching algorithms.

## Data
The `data/` directory contains rectified stereo image pairs and ground truth disparity maps for testing.

## License
This project is licensed under the GNU General Public License v3.0. See the `LICENSE` file for details.

## How to Run
1. Clone the repository.
2. Ensure the required dependencies are installed.
3. Run `main.py` to compute and visualize disparity maps.

## Future Work
- Implement optimization algorithms in `tuner.py`.
- Add evaluation metrics in `metrics.py`.
- Explore deep learning-based stereo matching techniques.
