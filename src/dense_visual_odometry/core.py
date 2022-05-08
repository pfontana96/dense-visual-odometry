import logging
import numpy as np
from pathlib import Path

from dense_visual_odometry.utils.lie_algebra.special_euclidean_group import SE3
from dense_visual_odometry.utils.interpolate import Interp2D
from dense_visual_odometry.camera_model import RGBDCameraModel


logger = logging.getLogger(__name__)


class DenseVisualOdometry:

    def __init__(self, camera_model: RGBDCameraModel, initial_pose: np.ndarray):
        """
        Parameters
        ----------
        camera_model : dense_visual_odometry.camera_model.RGBDCameraModel
            Camera model used
        initial_pose : np.ndarray
            Initial camera pose w.r.t the World coordinate frame expressed as a 6x1 se(3) vector
        """
        self.camera_model = camera_model
        self.initial_pose = initial_pose

        self.current_pose = initial_pose.copy()

    def compute_residuals(self, gray_image: np.ndarray, gray_image_prev: np.ndarray, depth_image_prev: np.ndarray,
                          transformation: np.ndarray, keep_dims: bool=True, return_mask: bool=False):
        """
            Deprojects `depth_image_prev` into a 3d space, then it transforms this pointcloud using the estimated
            `transformation` between the 2 different camera poses and projects this pointcloud back to an
            image plane. Then it interpolates values for this new synthetic image using `gray_image` and compares
            this result with the intensities values of `gray_image_prev`. It returns:

        Parameters
        ----------
        gray_image : np.ndarray
            Intensity image corresponding to t (height, width).
        gray_image_prev : np.ndarray
            Intensity image corresponding to t-1 (height, width).
        depth_image_prev : np.ndarray
            Depth image corresponding to t-1 (height, width). Invalid pixels should have 0 as value
        transformation : np.ndarray
            Transformation to be applied. It might be expressed as a (4,4) SE(3) matrix or a (6,1) se(3) vector.
        keep_dims : bool
            If True then the function returns an array of the same shape as `gray_image` (height, width), otherwise
            it returns an array of shape (-1, 1) with only valid pixels (i.e where `depth_image_prev` is not zero)
        return_mask : bool
            If True then the binary mask computed by this function (i.e. boolean array where `depth_image_prev` is
            zero is also returned). If `keep_dims` is False then this parameter won't be taken into account (there is
            no sense in returning a boolean mask if there is no interest in visualizing `residuals` as an image)

        Returns
        -------
        residuals : np.ndarray
            Residuals image. If `keep_dims` is set to True, an image with the same shape
        mask : np.ndarray, optional
            Boolean mask mask computed by this function (i.e. boolean array where `depth_image_prev` is
            zero is also returned) 
        """
        assertion_message = f"`gray_image` {gray_image.shape} and `depth_image` {depth_image_prev.shape} should have the same shape"
        assert gray_image.shape == depth_image_prev.shape, assertion_message
        assertion_message = f"Expected 'transformation' shape to be either (4,4) or (6,1) got {transformation.shape} instead"
        assert transformation.shape == (6,1) or transformation.shape == (4,4), assertion_message

        # Deproject image into 3d space w.r.t the first camera position
        # NOTE: Assuming origin is first camera position
        pointcloud, mask = self.camera_model.deproject(depth_image_prev, np.zeros((6, 1), dtype=np.float32),
                                                       return_mask=True)

        # Transform pointcloud using estimated rigid motion, i.e. `transformation`
        if transformation.shape == (4, 4):
            transformation = SE3.log(transformation)

        warped_pixels = self.camera_model.project(pointcloud, transformation)
        logger.debug("Warped Pixels shape: {}".format(warped_pixels.shape))

        # Interpolate intensity values for warped pixels projected coordinates
        new_gray_image = np.zeros_like(gray_image_prev, dtype=np.float32)
        new_gray_image[mask] = Interp2D.bilinear(warped_pixels[0], warped_pixels[1], gray_image)

        residuals = np.zeros_like(gray_image_prev, dtype=np.float32)
        residuals[mask] = new_gray_image[mask] - gray_image_prev[mask]
        logger.debug(f"Residuals (min, max, mean): ({residuals.min()}, {residuals.max()}, {residuals.mean()})")

        if not keep_dims:
            residuals = residuals[mask].reshape(-1, 1)

        elif return_mask:
            return (residuals, mask)

        return residuals
