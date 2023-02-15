import numpy as np
import numpy.typing as npt
import numba as nb
import cv2


@nb.njit("float32[:,:,:](float32[:,:], float32[:,:])", parallel=True, fastmath=True)
def compute_jacobian_of_warp_function(
    pointcloud: npt.NDArray[np.float32], calibration_matrix: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
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
    `J_w_i = J_pi * J_g * J_G` Where: `J_pi` is the 2x3 matrix of derivatives of the projection function with respect to
    points coordinates, `J_g` is the 3x12 Jacobian of the rigid body transformation with respect to its 12 parameters
    and `J_G` is the 12x6 Jacobian matrix of the exponential map (Lie Algebra). So `J_w` is a Nx2x6 matrix
    """
    fx = calibration_matrix[0, 0]
    fy = calibration_matrix[1, 1]

    N = pointcloud.shape[1]

    J_w = np.empty((N, 2, 6), dtype=np.float32)

    for i in nb.prange(N):
        x = pointcloud[0, i]
        y = pointcloud[1, i]
        z = pointcloud[2, i]

        J_w[i] = np.array([
            [fx / z, 0.0, -fx * x / z ** 2, -fx * (x * y) / z ** 2, fx * (1 + (x ** 2 / z ** 2)), -fx * y / z],
            [0.0, fy / z, -fy * y / z ** 2, -fy * (1 + (y ** 2 / z ** 2)), fy * (x * y) / z ** 2, fy * x / z]
        ], dtype=np.float32)

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
