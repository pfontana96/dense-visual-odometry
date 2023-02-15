from typing import Tuple
import logging

import numpy as np
import numpy.typing as npt
import numba as nb
from scipy.interpolate import RegularGridInterpolator

from dense_visual_odometry.core.robust_dense_visual_odometry.base_robust_dvo import BaseRobustDVO
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.utils.lie_algebra import Se3
from dense_visual_odometry.utils.image_pyramid import ImagePyramid
from dense_visual_odometry.utils.jacobian import compute_jacobian_of_warp_function, compute_gradients


logger = logging.getLogger(__name__)


class RobustDVOCPU(BaseRobustDVO):
    def __init__(
        self, camera_model: RGBDCameraModel, initial_pose: Se3, levels: int, use_weighter: bool = False,
        max_increased_steps_allowed: int = 0, sigma: float = None, tolerance: float = 1e-6, max_iterations: int = 100,
        approximate_image2_gradient: bool = False
    ):
        super(RobustDVOCPU, self).__init__(
            camera_model=camera_model, initial_pose=initial_pose, levels=levels, use_weighter=use_weighter,
            max_increased_steps_allowed=max_increased_steps_allowed, sigma=sigma, tolerance=tolerance,
            max_iterations=max_iterations, approximate_image2_gradient=approximate_image2_gradient
        )

        self._curr_gray_image_pyr = None
        self._prev_gray_image_pyr = None

        self._curr_depth_image_pyr = None
        self._prev_depth_image_pyr = None

        self._gray_image_interpolator = None

        if not self._approximate_image2_gradients:
            self._grady_interpolator = None
            self._gradx_interpolator = None

        else:
            self._jacobian = None

    def _build_pyramids(
        self, gray_image: npt.NDArray[np.uint8], depth_image: npt.NDArray[np.uint16]
    ):
        """Method resposible for setting `self._curr_gray_image_pyr` and `self._curr_depth_image_pyr`
        """
        self._prev_gray_image_pyr = ImagePyramid(self._levels, self._gray_image_prev)
        self._prev_depth_image_pyr = ImagePyramid(self._levels, self._depth_image_prev)
        self._curr_gray_image_pyr = ImagePyramid(self._levels, gray_image)
        self._curr_depth_image_pyr = ImagePyramid(self._levels, depth_image)

    def _setup(self, level: int):
        height, width = self._prev_gray_image_pyr.at(level).shape

        # Init interpolators
        self._gray_image_interpolator = RegularGridInterpolator(
            points=(np.arange(height, dtype=int), np.arange(width, dtype=int)),
            values=self._curr_gray_image_pyr.at(level), method="linear", bounds_error=False
        )

        if not self._approximate_image2_gradients:

            gradx, grady = compute_gradients(image=self._curr_gray_image_pyr.at(level), kernel_size=3)

            self._gradx_interpolator = RegularGridInterpolator(
                points=(np.arange(height, dtype=int), np.arange(width, dtype=int)),
                values=gradx, method="linear", bounds_error=False
            )

            self._grady_interpolator = RegularGridInterpolator(
                points=(np.arange(height, dtype=int), np.arange(width, dtype=int)),
                values=grady, method="linear", bounds_error=False
            )

        else:
            # Directly compute Jacobian once using prev frame

            pointcloud, mask = self._camera_model.deproject(
                self._prev_depth_image_pyr.at(level), level=-level, return_mask=True
            )
            J_w = compute_jacobian_of_warp_function(
                pointcloud=pointcloud, calibration_matrix=self._camera_model.at(-level)
            )

            gradx, grady = compute_gradients(image=self._prev_gray_image_pyr.at(level), kernel_size=3)

            # gradx_values = gradx[mask].reshape(-1, 1)
            # grady_values = grady[mask].reshape(-1, 1)

            # self._jacobian = np.empty((gradx_values.size, 6), dtype=np.float32)
            # for i, gradients in enumerate(np.hstack((gradx_values, grady_values))):
            #     self._jacobian[i] = np.dot(gradients.reshape(1, 2), J_w[i])
            gradx_values = gradx[mask].reshape(-1, 1)
            grady_values = grady[mask].reshape(-1, 1)

            gradients = np.ascontiguousarray(np.hstack((gradx_values, grady_values)))

            self._jacobian = self._compute_jacobian_approximate_image2_gradient(
                np.ascontiguousarray(J_w), gradients
            )

    @staticmethod
    @nb.njit("float32[:,:](float32[:,:,:], float32[:,:])", parallel=True, fastmath=True)
    def _compute_jacobian_approximate_image2_gradient(
        Jw: npt.NDArray[np.float32], gradients: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:

        N = gradients.shape[0]
        jacobian = np.empty((N, 6), dtype=np.float32)
        for i in nb.prange(N):
            jacobian[i] = gradients[i] @ Jw[i]

        return jacobian

    def _cleanup(self):
        self._gray_image_interpolator = None

        if not self._approximate_image2_gradients:
            self._grady_interpolator = None
            self._gradx_interpolator = None

        else:
            self._jacobian = None

    def _compute_jacobian(
        self, J_w: npt.NDArray[np.float32], warped_pixels: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
            Computes the jacobian of an image with respect to a camera pose as: `J = Jl*Jw` where `Jl` is a 2xN matrix
            containing the gradients of `image` along the x and y directions and `Jw` is a Nx2x6 matrix containing the
            jacobian of the warping function (i.e. `J` is a Nx6 matrix). N is the number of valid pixels (i.e. with
            depth information not equal to zero)

        Parameters
        ----------
        J_w : npt.NDArray
            Jacobian of the warping function (Nx2x6)
        warped_pixels : npt.NDArray
            Warped pixels from previous frame to current one (2xN)

        Returns
        -------
        J : npt.NDArray
            NX6 array containing the jacobian of `image` with respect to the six parameters of the estimated camera
            motion.
        """
        gradx_values = self._gradx_interpolator(
            np.roll(warped_pixels[:2, :].T, 1, axis=1)
        ).astype(np.float32).reshape(-1, 1)
        grady_values = self._grady_interpolator(
            np.roll(warped_pixels[:2, :].T, 1, axis=1)
        ).astype(np.float32).reshape(-1, 1)

        J = np.zeros((gradx_values.size, 6), dtype=np.float32)
        for i, gradients in enumerate(np.hstack((gradx_values, grady_values))):
            J[i] = np.dot(gradients.reshape(1, 2), J_w[i])

        return J

    def compute_residuals_and_jacobian(
        self, estimate: Se3, level: int = 0
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.bool8]]:
        """Computes residuals of warping pixels in `gray_image_prev` onto `gray_image` by using depth
        information available in `depth_image_prev` and a given estimate of the transform between the
        frames (`estimate`). It also computes the Jacobian of the residuals with respect to the
        parameters of the transform in `estimate`.

        Parameters
        ----------
        estimate : Se3
            Estimate between previous frame and current one.
        level : int, optional
            Image pyramid level to process, by default 0.

        Returns
        -------
        Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.bool8]]
            Tuple containing residuals array (Nx1), jacobian array (Nx6) and valid pixels mask (same shape
            as `gray_image`) where N is the number of valid pixels (the sum of ones in valid pixels mask)
        """
        # Deproject image into 3d space w.r.t the first camera position
        pointcloud, mask = self._camera_model.deproject(
            self._prev_depth_image_pyr.at(level), return_mask=True, level=level
        )

        if self._approximate_image2_gradients:
            # NOTE: `_compute_jacobian_approximate_I2` is cached so it's not being computed every iteration
            # NOTE: Normally the jacobian to use is J = J_i(w(se3, x)) * Jw where only Jw is a constant term.
            # Nonetheless `J_i(w(se3, x)) = [I2x(w(se3, x)), I2y(w(se3, x))].T`
            # can be approximated by `J_i = [I1x(x), I1y(x)].T` which allows us NOT to recompute J_i at every iteration
            jacobian = self._jacobian

        else:
            J_w = compute_jacobian_of_warp_function(
                pointcloud=pointcloud, calibration_matrix=self._camera_model.at(level)
            )

        # Transform pointcloud to second camera frame using estimated rigid motion
        pointcloud = np.dot(estimate.exp(), pointcloud)

        # Warp I1 pixels
        warped_pixels = self._camera_model.project(pointcloud, level=level)
        warped_pixels_intensities = self._gray_image_interpolator(
            np.roll(warped_pixels[:2, :].T, 1, axis=1)
        ).astype(np.float32)
        valid_warped_pixels_mask = ~np.isnan(warped_pixels_intensities).flatten()

        if not self._approximate_image2_gradients:
            jacobian = self._compute_jacobian(J_w, warped_pixels[:, valid_warped_pixels_mask])

        else:
            jacobian = jacobian[valid_warped_pixels_mask]

        # Interpolate intensity values on I2 for warped pixels projected coordinates
        residuals = (
            warped_pixels_intensities[valid_warped_pixels_mask] -
            self._prev_gray_image_pyr.at(level)[mask][valid_warped_pixels_mask]
        ).reshape(-1, 1)

        logger.debug("Intensity Residuals (min, max, mean): ({}, {}, {})".format(
            residuals.min(), residuals.max(), residuals.mean()
        ))

        return residuals, jacobian, mask
