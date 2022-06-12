import logging
import numpy as np

from dense_visual_odometry.utils.lie_algebra.special_euclidean_group import SE3
from dense_visual_odometry.utils.interpolate import Interp2D
from dense_visual_odometry.utils.jacobian import compute_jacobian_of_warp_function, compute_gradients
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.weighter.base_weighter import BaseWeighter


logger = logging.getLogger(__name__)


class DenseVisualOdometry:

    def __init__(self, camera_model: RGBDCameraModel, initial_pose: np.ndarray, weighter: BaseWeighter = None):
        """
        Parameters
        ----------
        camera_model : dense_visual_odometry.camera_model.RGBDCameraModel
            Camera model used
        initial_pose : np.ndarray
            Initial camera pose w.r.t the World coordinate frame expressed as a 6x1 se(3) vector
        weighter : BaseWeighter | None, optional
            Weighter functions to apply on residuals to remove dynamic object. If None, then no weighting is applied
        """
        self.camera_model = camera_model
        self.initial_pose = initial_pose

        self.current_pose = initial_pose.copy()

        self.weighter = weighter

    def compute_residuals(self, gray_image: np.ndarray, gray_image_prev: np.ndarray, depth_image_prev: np.ndarray,
                          transformation: np.ndarray, keep_dims: bool = True, return_mask: bool = False):
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
        assertion_message = "`gray_image` {} and `depth_image` {} should have the same shape".format(
            gray_image.shape, depth_image_prev.shape
        )
        assert gray_image.shape == depth_image_prev.shape, assertion_message
        assertion_message = "Expected 'transformation' shape to be either (4,4) or (6,1) got {} instead".format(
            transformation.shape
        )
        assert transformation.shape == (6, 1) or transformation.shape == (4, 4), assertion_message

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

    def compute_jacobian(self, image: np.ndarray, depth_image: np.ndarray, camera_pose: np.ndarray):
        """
            Computes the jacobian of an image with respect to a camera pose as: `J = Jl*Jw` where `Jl` is a Nx2 matrix
            containing the gradiends of `image` along the x and y directions and `Jw` is a 2x6 matrix containing the
            jacobian of the warping function (i.e. `J` is a Nx6 matrix). N is the number of valid pixels (i.e. with
            depth information not equal to zero)

        Parameters
        ----------
        image : np.ndarray
            Grayscale image
        depth_image : np.ndarray
            Aligned depth image for `image`
        camera_pose : np.ndarray
            Camera pose expressed in Lie algebra as a matrix of shape 6x1 (i.e. se(3))

        Returns
        -------
        J : np.ndarray
            NX6 array containing the jacobian of `image` with respect to the six parameters of `camera_pose`
        """
        pointcloud, mask = self.camera_model.deproject(
            depth_image=depth_image, camera_pose=camera_pose, return_mask=True
        )

        J_w = compute_jacobian_of_warp_function(
            pointcloud=pointcloud, calibration_matrix=self.camera_model.calibration_matrix
        )

        gradx, grady = compute_gradients(image=image, kernel_size=3)

        # Filter out invalid pixels
        gradx = gradx[mask]
        grady = grady[mask]

        J = np.zeros((image.size, 6))
        for i, gradients in enumerate(np.hstack((gradx.reshape(-1, 1), grady.reshape(-1, 1)))):
            J[i] = np.dot(gradients.reshape(1, 2), J_w[i])

        return J

    def gauss_newton(
        self, gray_image: np.ndarray, gray_image_prev: np.ndarray, depth_image_prev: np.ndarray,
        max_iter: int = 100, ratio: float = 0.995
    ):
        pass
