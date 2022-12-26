from typing import Union

import numpy as np
import cupy as cp
import cv2


def compute_jacobian_of_warp_function(
    pointcloud: Union[np.ndarray, cp.ndarray], calibration_matrix: Union[np.ndarray, cp.ndarray], use_gpu: bool = False
) -> Union[np.ndarray, cp.ndarray]:
    """
        Computes the Jacobian of a warp function

    Parameters
    ----------
    pointcloud : np.ndarray
        Resulting pointcloud of deprojecting and image (4xN)
    calibration_matrix : np.ndarray
        4x4 camera calibration matrix used on warping function

    Notes
    -----
    `J_w = J_pi * J_g * J_G` Where: `J_pi` is the 2x3 matrix of derivatives of the projection function with respect to
    points coordinates, `J_g` is the 3x12 Jacobian of the rigid body transformation with respect to its 12 parameters
    and `J_G` is the 12x6 Jacobian matrix of the exponential map (Lie Algebra). So `J_w` is a 2x6 matrix
    """
    module = np if not use_gpu else cp

    fx = calibration_matrix[0, 0]
    fy = calibration_matrix[1, 1]

    x = pointcloud[0, :]
    y = pointcloud[1, :]
    z = pointcloud[2, :]

    zeros = module.zeros_like(x)

    J_w = module.array([
        [fx / z, zeros, -fx * x / z ** 2, -fx * (x * y) / z ** 2, fx * (1 + (x ** 2 / z ** 2)), -fx * y / z],
        [zeros, fy / z, -fy * y / z ** 2, -fy * (1 + (y ** 2 / z ** 2)), fy * (x * y) / z ** 2, fy * x / z]
    ], dtype=module.float32)

    # Transpose array to be of shape Nx2x6
    J_w = module.transpose(J_w, [2, 0, 1])

    return J_w


def compute_gradients(image: np.ndarray, kernel_size: int, ddepth: int = cv2.CV_32FC1):
    """
        Computes the gradients of an image along the x and y axis using Sobel's approximation.
        See https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html

    Parameters
    ----------
    image : np.ndarray
        Input image
    kernel_size : int
        Kernel size to use on Sobel's filter. Must be an odd number
    ddepth : int, optional
        Output image depth. It defaults to 32-bits floating point numbers

    Returns
    -------
    gradx : np.ndarray
        Numpy array with the same shape as `image` containing gradient values on x-axis
    grady : np.ndarray
        Numpy array with the same shape as `image` containing gradient values on y-axis
    """
    assert (kernel_size % 2) != 0, "Expected 'kernel_size' to be an odd number, got '{}' instead".format(kernel_size)

    gradx = cv2.Sobel(image, ddepth=ddepth, dx=1, dy=0, ksize=kernel_size, borderType=cv2.BORDER_REFLECT)
    grady = cv2.Sobel(image, ddepth=ddepth, dx=0, dy=1, ksize=kernel_size, borderType=cv2.BORDER_REFLECT)

    return gradx, grady
