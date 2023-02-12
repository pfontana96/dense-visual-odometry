import logging
import ctypes as C

import numpy as np
import numba as nb
from numba import cuda
from scipy.interpolate import RegularGridInterpolator

from dense_visual_odometry.core.base_dense_visual_odometry import BaseDenseVisualOdometry
from dense_visual_odometry.utils.image_pyramid import CoarseToFineMultiImagePyramid
from dense_visual_odometry.utils.lie_algebra import Se3
from dense_visual_odometry.utils.jacobian import compute_jacobian_of_warp_function, compute_gradients
from dense_visual_odometry.utils.numpy_cache import np_cache
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.weighter.t_weighter import TDistributionWeighter
from dense_visual_odometry.cuda import CUDA_BLOCKSIZE, residuals_kernel


logger = logging.getLogger(__name__)


@nb.njit("float32[:,:](float32[:,:], float32[:,:])", fastmath=True)
def nb_lstsq(a, b):
    result, _, _, _ = np.linalg.lstsq(a=a, b=b)
    return result


class KerlDVO(BaseDenseVisualOdometry):
    """
        Class for performing dense visual odometry by minimizing the photometric error (see [1]_).

    Attributes
    ----------
    camera_model : dense_visual_odometry.camera_model.RGBDCameraModel
        Camera model used
    initial_pose : np.ndarray
            Initial camera pose w.r.t the World coordinate frame expressed as a 6x1 se(3) vector
    weighter : BaseWeighter | None, optional
        Weighter functions to apply on residuals to remove dynamic object. If None, then no weighting is applied
    gray_image_prev : np.ndarray
        Previous frame's grayscale image
    depth_image_prev : np.ndarray
        Previous frame's depth image

    Notes
    ----------
    .. [1] Kerl, C., Sturm, J., Cremers, D., "Robust Odometry Estimation for RGB-D Cameras"
    """
    # TODO: Update doc
    def __init__(
        self, camera_model: RGBDCameraModel, initial_pose: Se3, levels: int, use_weighter: bool = False,
        max_increased_steps_allowed: int = 0, sigma: float = None, use_gpu: bool = False, tolerance: float = 1e-6,
        max_iterations: int = 100, approximate_image2_gradient: bool = False
    ):
        """
        Parameters
        ----------
        camera_model : dense_visual_odometry.camera_model.RGBDCameraModel
            Camera model used
        initial_pose : np.ndarray
            Initial camera pose w.r.t the World coordinate frame expressed as a 6x1 se(3) vector
        levels : int
            Pyramid octaves to use
        weighter : BaseWeighter | None, optional
            Weighter functions to apply on residuals to remove dynamic object. If None, then no weighting is applied
        """
        weighter = TDistributionWeighter() if use_weighter else None
        super(KerlDVO, self).__init__(camera_model=camera_model, initial_pose=initial_pose, weighter=weighter)
        self.levels = levels
        self._max_increased_steps_allowed = max_increased_steps_allowed

        self._sigma = sigma
        if self._sigma is not None:
            self._inv_cov = (1 / self._sigma) * np.eye(6, dtype=np.float32)

        self._use_gpu = use_gpu

        self._tolerance = tolerance
        self._max_iter = max_iterations

        self._gray_image_interpolator = None

        self._grady_interpolator = None
        self._gradx_interpolator = None

        self._approximate_image2_gradients = approximate_image2_gradient

    def _init_gray_image_interpolator(self, gray_image: np.ndarray):

        height, width = gray_image.shape
        self._gray_image_interpolator = RegularGridInterpolator(
            points=(np.arange(height, dtype=int), np.arange(width, dtype=int)),
            values=gray_image, method="linear", bounds_error=False
        )

    def _clear_gray_image_interpolator(self):
        self._gray_image_interpolator = None

    def _init_gradients_interpolators(self, gray_image: np.ndarray):

        height, width = gray_image.shape

        gradx, grady = compute_gradients(image=gray_image, kernel_size=3)

        self._gradx_interpolator = RegularGridInterpolator(
            points=(np.arange(height, dtype=int), np.arange(width, dtype=int)),
            values=gradx, method="linear", bounds_error=False
        )

        self._grady_interpolator = RegularGridInterpolator(
            points=(np.arange(height, dtype=int), np.arange(width, dtype=int)),
            values=grady, method="linear", bounds_error=False
        )

    def _clear_gradients_interpolators(self):
        self._gradx_interpolator = None
        self._grady_interpolator = None

    # TODO: Update doc
    def _compute_residuals(
        self, gray_image: np.ndarray, gray_image_prev: np.ndarray, depth_image_prev: np.ndarray,
        estimate: Se3, level: int = 0
    ):
        """
            Deprojects `depth_image_prev` into a 3d space, then it transforms this pointcloud using the estimated
            `transformation` between the 2 different camera poses and projects this pointcloud back to an
            image plane. Then it interpolates values for this new synthetic image using `gray_image_prev` and compares
            this result with the intensities values of `gray_image`

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
        keep_dims : bool, optional
            If True then the function returns an array of the same shape as `gray_image` (height, width), otherwise
            it returns an array of shape (-1, 1) with only valid pixels (i.e where `depth_image_prev` is not zero).
            Defaults to `True`
        return_mask : bool, optional
            If True then the binary mask computed by this function (i.e. boolean array where `depth_image_prev` is
            zero is also returned). If `keep_dims` is False then this parameter won't be taken into account (there is
            no sense in returning a boolean mask if there is no interest in visualizing `residuals` as an image).
            Defaults to `False`

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

        # Deproject image into 3d space w.r.t the first camera position
        pointcloud, mask = self._camera_model.deproject(depth_image_prev, return_mask=True, level=level)

        if self._approximate_image2_gradients:
            # NOTE: `_compute_jacobian_approximate_I2` is cached so it's not being computed every iteration
            # NOTE: Normally the jacobian to use is J = J_i(w(se3, x)) * Jw where only Jw is a constant term.
            # Nonetheless `J_i(w(se3, x)) = [I2x(w(se3, x)), I2y(w(se3, x))].T`
            # can be approximated by `J_i = [I1x(x), I1y(x)].T` which allows us NOT to recompute J_i at every iteration
            jacobian = self._compute_jacobian_approximate_I2(
                image=gray_image_prev, depth_image=depth_image_prev, level=level
            )

        else:
            J_w = compute_jacobian_of_warp_function(
                pointcloud=pointcloud, calibration_matrix=self._camera_model.at(level)
            )

        # Transform pointcloud to second camera frame using estimated rigid motion
        pointcloud = np.dot(estimate.exp(), pointcloud)

        # Warp I1 pixels
        warped_pixels = self._camera_model.project(pointcloud, level=level)
        warped_pixels_intensities = self._gray_image_interpolator(np.roll(warped_pixels[:2, :].T, 1, axis=1))
        valid_warped_pixels_mask = ~np.isnan(warped_pixels_intensities).flatten()

        if not self._approximate_image2_gradients:
            jacobian = self._compute_jacobian(J_w, warped_pixels[:, valid_warped_pixels_mask])

        else:
            jacobian = jacobian[valid_warped_pixels_mask]

        # Interpolate intensity values on I2 for warped pixels projected coordinates
        residuals_intensity = (
            warped_pixels_intensities[valid_warped_pixels_mask] - gray_image_prev[mask][valid_warped_pixels_mask]
        ).reshape(-1, 1)

        logger.debug("Intensity Residuals (min, max, mean): ({}, {}, {})".format(
            residuals_intensity.min(), residuals_intensity.max(), residuals_intensity.mean()
        ))

        return residuals_intensity, jacobian

    # TODO: Fix documentation
    def _compute_jacobian(self, J_w: np.ndarray, warped_pixels: np.ndarray):
        """
            Computes the jacobian of an image with respect to a camera pose as: `J = Jl*Jw` where `Jl` is a Nx2 matrix
            containing the gradients of `image` along the x and y directions and `Jw` is a 2x6 matrix containing the
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
        gradx_values = self._gradx_interpolator(np.roll(warped_pixels[:2, :].T, 1, axis=1)).reshape(-1, 1)
        grady_values = self._grady_interpolator(np.roll(warped_pixels[:2, :].T, 1, axis=1)).reshape(-1, 1)

        J = np.zeros((gradx_values.size, 6), dtype=np.float32)
        for i, gradients in enumerate(np.hstack((gradx_values, grady_values))):
            J[i] = np.dot(gradients.reshape(1, 2), J_w[i])

        return J

    @np_cache
    def _compute_jacobian_approximate_I2(self, image: np.ndarray, depth_image: np.ndarray, level: int):
        """
            Computes the jacobian of a residual with respect to a camera pose (se(3)) as:
            `dr(w(xi, x))/dxi = J = Ji * Jw`
            where `Ji` is a
            Nx2 matrix containing the gradients of `image` along the x and y directions and `Jw` is a 2x6 matrix
            containing the jacobian of the warping function (i.e. `J` is a Nx6 matrix). N is the number of valid pixels
            (i.e. with depth information not equal to zero)
        """
        pointcloud, mask = self._camera_model.deproject(depth_image, level=level, return_mask=True)
        J_w = compute_jacobian_of_warp_function(
            pointcloud=pointcloud, calibration_matrix=self._camera_model.at(level)
        )

        gradx, grady = compute_gradients(image=image, kernel_size=3)

        gradx_values = gradx[mask].reshape(-1, 1)

        grady_values = grady[mask].reshape(-1, 1)

        J = np.zeros((gradx_values.size, 6), dtype=np.float32)
        for i, gradients in enumerate(np.hstack((gradx_values, grady_values))):
            J[i] = np.dot(gradients.reshape(1, 2), J_w[i])

        return J

    def _step(self, gray_image: np.ndarray, depth_image: np.ndarray, init_guess: Se3 = Se3.identity()):
        # Create coarse to fine Image Pyramids
        image_pyramids = CoarseToFineMultiImagePyramid(
            images=[gray_image, self._gray_image_prev, depth_image, self._depth_image_prev],
            levels=self.levels
        )

        estimate = init_guess.copy()

        for i, (gray_image_l, gray_image_prev_l, depth_image_l, depth_image_prev_l) in enumerate(image_pyramids):

            level = self.levels - 1 - i

            if self._use_gpu:
                estimate = self._non_linear_least_squares_gpu(
                    init_guess=estimate, gray_image=gray_image_l, depth_image=depth_image_l,
                    gray_image_prev=gray_image_prev_l, depth_image_prev=depth_image_prev_l, level=-level
                )

            else:
                self._init_gray_image_interpolator(gray_image_l)
                if not self._approximate_image2_gradients:
                    self._init_gradients_interpolators(gray_image_l)

                estimate = self._non_linear_least_squares(
                    init_guess=estimate, gray_image=gray_image_l, depth_image=depth_image_l,
                    gray_image_prev=gray_image_prev_l, depth_image_prev=depth_image_prev_l, level=-level
                )

                self._clear_gray_image_interpolator()
                self._clear_gradients_interpolators()

        # Clean cache (only used when running on CPU)
        if not self._use_gpu:
            self._camera_model.deproject.cache_clear()
            if not self._approximate_image2_gradients:
                self._compute_jacobian_approximate_I2.cache_clear()
            compute_jacobian_of_warp_function.cache_clear()

        return estimate

    def _non_linear_least_squares_gpu(
        self, init_guess: Se3, gray_image: np.ndarray, gray_image_prev: np.ndarray, depth_image: np.ndarray,
        depth_image_prev: np.ndarray, level: int = 0
    ):

        old = self._last_estimated_transform.copy()
        estimate = init_guess.copy()
        initial = init_guess.copy()

        err_prev = np.finfo("float32").max
        err_increased_count = 0

        height, width = gray_image.shape

        gray_image = np.ascontiguousarray(gray_image)
        gray_image_prev = np.ascontiguousarray(gray_image_prev)
        depth_image_prev = np.ascontiguousarray(depth_image_prev)

        # Images
        gpu_gray_image_buffer = cuda.mapped_array(
            (height, width), dtype=np.uint8, strides=None, order='C', stream=0, portable=False, wc=True
        )
        gpu_gray_image_buffer[...] = gray_image

        gpu_gray_image_prev_buffer = cuda.mapped_array(
            (height, width), dtype=np.uint8, strides=None, order='C', stream=0, portable=False, wc=True
        )
        gpu_gray_image_prev_buffer[...] = gray_image_prev

        gpu_depth_image_prev_buffer = cuda.mapped_array(
            (height, width), dtype=np.uint16, strides=None, order='C', stream=0, portable=False, wc=True
        )
        gpu_depth_image_prev_buffer[...] = depth_image_prev

        # Estimate
        gpu_R_buffer = cuda.mapped_array(
            (3, 3), dtype=np.float32, strides=None, order='C', stream=0, portable=False, wc=False
        )
        gpu_tvec_buffer = cuda.mapped_array(
            (3,), dtype=np.float32, strides=None, order='C', stream=0, portable=False, wc=False
        )

        # Residuals and jacobian
        gpu_mask_buffer = cuda.managed_array((height, width), dtype=bool, strides=None, order='C', stream=0)
        gpu_residuals_buffer = cuda.managed_array((height, width), dtype=np.float32, strides=None, order='C', stream=0)
        gpu_jacobian_buffer = cuda.managed_array(
            (gray_image.size, 6), dtype=np.float32, strides=None, order='C', stream=0
        )

        residuals_complete = np.ctypeslib.as_array(
            C.cast(gpu_residuals_buffer.ctypes.data, C.POINTER(C.c_float)), shape=gpu_residuals_buffer.shape
        )
        mask = np.ctypeslib.as_array(
            C.cast(gpu_mask_buffer.ctypes.data, C.POINTER(C.c_bool)), shape=gpu_mask_buffer.shape
        )
        jacobian_complete = np.ctypeslib.as_array(
            C.cast(gpu_jacobian_buffer.ctypes.data, C.POINTER(C.c_float)), shape=gpu_jacobian_buffer.shape
        )

        block_dim = (CUDA_BLOCKSIZE, CUDA_BLOCKSIZE)
        grid_dim = (
            int((width + (CUDA_BLOCKSIZE - 1)) // CUDA_BLOCKSIZE),
            int((height + (CUDA_BLOCKSIZE - 1)) // CUDA_BLOCKSIZE)
        )

        for i in range(self._max_iter):

            # Compute residuals
            T = np.ascontiguousarray(estimate.exp())

            # Update data from estimate to gpu buffers
            gpu_R_buffer[...] = T[:3, :3]
            gpu_tvec_buffer[...] = T[:3, 3]

            intrinsics = self._camera_model.at(level)

            residuals_kernel[grid_dim, block_dim](
                gpu_gray_image_buffer, gpu_gray_image_prev_buffer, gpu_depth_image_prev_buffer, gpu_R_buffer,
                gpu_tvec_buffer, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2],
                self._camera_model.depth_scale, gpu_mask_buffer, gpu_residuals_buffer, gpu_jacobian_buffer,
                height, width
            )

            cuda.synchronize()

            residuals = residuals_complete[mask].reshape(-1, 1)
            jacobian = jacobian_complete[mask.reshape(-1)]

            jacobian_t = jacobian.T.copy()

            # Computes weights if required
            if self._weighter is not None:

                residuals_squared = residuals * residuals

                weights = self._weighter.weight(residuals_squared=residuals_squared)

                err = np.mean(weights * residuals_squared)

                residuals = weights * residuals
                jacobian = weights * jacobian

            else:

                err = np.mean(residuals ** 2)

            # Solve linear system: (Jt * W * J) * delta_xi = (-Jt * W * r) -> H * delta_xi = b
            H = jacobian_t @ jacobian
            b = - jacobian_t @ residuals

            if (self._sigma is not None) and (old is not None):
                H += self._inv_cov
                b += self._inv_cov @ old.log()

                err += 0.5 * self._sigma * np.linalg.norm(old.log())

            # inc_xi, _, _, _ = lstsq(
            #     a=H, b=b, lapack_driver="gelsy", overwrite_a=True, overwrite_b=True, check_finite=False
            # )
            inc_xi = nb_lstsq(a=H, b=b)

            inc = Se3.from_se3(inc_xi)

            err_diff = err - err_prev

            logger.debug("Iteration {} -> error: {:.4f}".format(i + 1, err))

            if abs(err_diff) < self._tolerance:
                logger.info("Found convergence on iteration {} (error: {:.4f})".format(i + 1, err))
                break

            # Stopping criteria (error function always displays a global minima)
            if (err_diff < 0.0):
                # Error decreased, so compute increment
                estimate = inc * estimate
                err_prev = err

                if (self._sigma is not None) and (old is not None):
                    old = inc.inverse() * old

                err_increased_count = 0

            else:
                err_increased_count += 1

            if err_increased_count > self._max_increased_steps_allowed:
                logger.info("Error increased on iteration '{}' (error: {:.4f})".format(i, err))
                break

            if i == (self._max_iter - 1):
                logger.warning("Exceeded maximum number of iterations '{}' (error: {:.4f})".format(
                    self._max_iter, err
                ))

        return estimate

    def _non_linear_least_squares(
        self, init_guess: Se3, gray_image: np.ndarray, gray_image_prev: np.ndarray, depth_image: np.ndarray,
        depth_image_prev: np.ndarray, level: int = 0
    ):

        old = self._current_pose.copy()
        estimate = init_guess.copy()
        initial = init_guess.copy()

        err_prev = np.finfo("float32").max
        err_increased_count = 0

        for i in range(self._max_iter):

            residuals, jacobian = self._compute_residuals(
                gray_image=gray_image, gray_image_prev=gray_image_prev, depth_image_prev=depth_image_prev,
                estimate=estimate, level=level
            )
            jacobian_t = jacobian.T.copy()

            # Computes weights if required
            if self._weighter is not None:

                residuals_squared = residuals * residuals

                weights = self._weighter.weight(residuals_squared=residuals_squared)

                err = np.mean(weights * residuals_squared)

                residuals = weights * residuals
                jacobian = weights * jacobian

            else:
                err = np.mean(residuals ** 2)

            # Solve linear system: (Jt * W * J) * delta_xi = (-Jt * W * r) -> H * delta_xi = b
            H = np.dot(jacobian_t, jacobian)
            b = - np.dot(jacobian_t, residuals)

            if self._sigma is not None:
                # maybe_old = SE3.log(np.dot(SE3.inverse(SE3.exp(estimate)), SE3.exp(self._current_pose)))
                H += self._inv_cov
                b += np.dot(self._inv_cov, old.log())

                err += 0.5 * self._sigma * np.linalg.norm(old.log())

            inc_xi, _, _, _ = np.linalg.lstsq(a=H, b=b, rcond=None)

            inc = Se3.from_se3(inc_xi)

            err_diff = err - err_prev

            logger.debug("Iteration {} -> error: {:.4f}".format(i + 1, err))

            if abs(err_diff) < self._tolerance:
                logger.info("Found convergence on iteration {} (error: {:.4f})".format(i + 1, err))
                break

            # Stopping criteria (error function always displays a global minima)
            if (err_diff < 0.0):
                # Error decreased, so compute increment
                estimate = inc * estimate
                err_prev = err

                if self._sigma is not None:
                    old = inc.inverse() * old

                err_increased_count = 0

            else:
                err_increased_count += 1

            if err_increased_count > self._max_increased_steps_allowed:
                logger.info("Error increased on iteration '{}' (error: {:.4f})".format(i, err))
                break

            if i == (self._max_iter - 1):
                logger.warning("Exceeded maximum number of iterations '{}' (error: {:.4f})".format(
                    self._max_iter, err
                ))

        return estimate
