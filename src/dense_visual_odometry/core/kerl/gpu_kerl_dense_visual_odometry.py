import logging

import numpy as np
from numba import cuda
from scipy.interpolate import RegularGridInterpolator

from dense_visual_odometry.core.kerl.base_kerl_dense_visual_odometry import BaseKerlDVO
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.utils.lie_algebra import Se3
from dense_visual_odometry.weighter.t_weighter import TDistributionWeighter
from dense_visual_odometry.cuda import CUDA_BLOCKSIZE, residuals_kernel, weighting_kernel


logger = logging.getLogger(__name__)


class GPUKerlDVO(BaseKerlDVO):
    """
        GPU Implementation of Kerl's approach
    """
    # TODO: Update doc
    def __init__(
        self, camera_model: RGBDCameraModel, initial_pose: Se3, levels: int, use_weighter: bool = False,
        max_increased_steps_allowed: int = 0, sigma: float = None, use_gpu: bool = False, tolerance: float = 1e-6,
        max_iterations: int = 100, mu: float = None, approximate_image2_gradient: bool = False
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
        super(GPUKerlDVO, self).__init__(
            camera_model=camera_model, initial_pose=initial_pose, levels=levels, use_weighter=use_weighter,
            max_increased_steps_allowed=max_increased_steps_allowed, sigma=sigma, tolerance=tolerance,
            max_iterations=max_iterations, mu=mu, approximate_image2_gradient=approximate_image2_gradient
        )

        self._pointers = None  # GPU allocated pointers (Unified memory for Jetson Nano)

    def _compute_residuals(
        self, gray_image: np.ndarray, gray_image_prev: np.ndarray, depth_image_prev: np.ndarray,
        estimate: Se3, level: int = 0
    ):
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

        old = self._current_pose.copy()
        estimate = init_guess.copy()
        initial = init_guess.copy()

        err_prev = np.finfo("float32").max
        err_increased_count = 0

        height, width = gray_image.shape

        residuals_complete = np.zeros(gray_image.shape, dtype=np.float32, order="C")
        mask = np.zeros(gray_image.shape, dtype=bool, order="C")
        jacobian_complete = np.zeros((gray_image.size, 6), dtype=np.float32, order="C")

        if self._weighter is not None:
            weights_complete = np.zeros_like(residuals_complete, order="C")

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
            (3, 3), dtype=np.float32, strides=None, order='C', stream=0, portable=False, wc=True
        )
        gpu_tvec_buffer = cuda.mapped_array(
            (3,), dtype=np.float32, strides=None, order='C', stream=0, portable=False, wc=True
        )

        # Residuals and jacobian
        gpu_mask_buffer = cuda.mapped_array(
            (height, width), dtype=bool, strides=None, order='C', stream=0, portable=False, wc=True
        )
        gpu_residuals_buffer = cuda.mapped_array(
            (height, width), dtype=np.float32, strides=None, order='C', stream=0, portable=False, wc=True
        )
        gpu_jacobian_buffer = cuda.mapped_array(
            (gray_image.size, 6), dtype=np.float32, strides=None, order='C', stream=0, portable=False, wc=True
        )

        if self._weighter is not None:
            gpu_weights_buffer = cuda.mapped_array(
                weights_complete.shape, dtype=np.float32, strides=None, order='C', stream=0, portable=False, wc=True
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

            residuals_complete[...] = gpu_residuals_buffer
            jacobian_complete[...] = gpu_jacobian_buffer
            mask[...] = gpu_mask_buffer

            residuals = residuals_complete[mask].reshape(-1, 1)
            jacobian = jacobian_complete[mask.reshape(-1)]

            jacobian_t = jacobian.T.copy()

            # Computes weights if required
            if self._weighter is not None:

                weighting_kernel[grid_dim, block_dim](
                    gpu_residuals_buffer, gpu_mask_buffer, gpu_weights_buffer, 5.0, 5.0, 1e-3, 100, height, width
                )

                cuda.synchronize()

                weights_complete[...] = gpu_weights_buffer
                weights = weights_complete[mask].reshape(-1, 1)

                logger.info("Weights (min, max, mean): ({}, {}, {})".format(
                    weights.min(), weights.max(), weights.mean()
                ))

                residuals = weights * residuals
                jacobian = weights * jacobian

            err = np.mean(residuals ** 2)

            # Solve linear system: (Jt * W * J) * delta_xi = (-Jt * W * r) -> H * delta_xi = b
            H = np.dot(jacobian_t, jacobian)
            b = - np.dot(jacobian_t, residuals)

            if self._sigma is not None:
                # maybe_old = SE3.log(np.dot(SE3.inverse(SE3.exp(estimate)), SE3.exp(self._current_pose)))
                H += self._inv_cov
                b += np.dot(self._inv_cov, old.log())

                err += 0.5 * self._sigma * np.linalg.norm(old.log())

            if self._mu is not None:
                err += 0.5 * self._mu * np.linalg.norm(initial.log())

            inc_xi, _, _, _ = np.linalg.lstsq(a=H, b=b)

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

                if self._mu is not None:
                    initial = inc.inverse() * initial

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

            if self._mu is not None:
                err += 0.5 * self._mu * np.linalg.norm(initial.log())

            inc_xi, _, _, _ = np.linalg.lstsq(a=H, b=b)

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

                if self._mu is not None:
                    initial = inc.inverse() * initial

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
