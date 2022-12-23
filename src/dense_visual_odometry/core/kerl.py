import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_triangular

from dense_visual_odometry.core.base_dense_visual_odometry import BaseDenseVisualOdometry
from dense_visual_odometry.utils.image_pyramid import CoarseToFineMultiImagePyramid
from dense_visual_odometry.utils.lie_algebra import SE3
from dense_visual_odometry.utils.jacobian import compute_jacobian_of_warp_function, compute_gradients
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.weighter.t_weighter import TDistributionWeighter


logger = logging.getLogger(__name__)


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

    def __init__(
        self, camera_model: RGBDCameraModel, initial_pose: np.ndarray, levels: int, use_weighter: bool = False
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

    # TODO: Update doc
    def _compute_residuals(
        self, gray_image: np.ndarray, gray_image_prev: np.ndarray, depth_image_prev: np.ndarray,
        transformation: np.ndarray, level: int = 0, compute_jacobian: bool = True
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
        assertion_message = "Expected 'transformation' shape to be either (4,4) or (6,1) got {} instead".format(
            transformation.shape
        )
        assert transformation.shape == (6, 1) or transformation.shape == (4, 4), assertion_message

        # Deproject image into 3d space w.r.t the first camera position
        # NOTE: Assuming origin is first camera position
        pointcloud, mask = self._camera_model.deproject(
            depth_image_prev, np.zeros((6, 1), dtype=np.float32), return_mask=True, level=level
        )

        # Compute Jacobian at identity
        if compute_jacobian:
            jacobian = self._compute_jacobian(
                image=gray_image_prev, pointcloud=pointcloud, mask=mask
            )

        # Transform pointcloud using estimated rigid motion, i.e. `transformation`
        if transformation.shape == (6, 1):
            transformation = SE3.exp(transformation)

        pointcloud = np.dot(transformation, pointcloud)

        warped_pixels = self._camera_model.project(pointcloud, np.zeros((6, 1), dtype=np.float32), level=level)

        # Interpolate intensity values for warped pixels projected coordinates

        # NOTE: By the current implementation (17/04/2022) of 'Interp2D.bilinear' if we give the exact grid to retrieve
        # the same image, then last row and last column will be 0.0
        # residuals = (
        #     Interp2D.bilinear(warped_pixels[0], warped_pixels[1], gray_image) - gray_image_prev[mask]
        # ).reshape(-1, 1)
        height, width = gray_image.shape
        interp = RegularGridInterpolator(
            points=(np.arange(height, dtype=int), np.arange(width, dtype=int)),
            values=gray_image, method="linear"
        )

        residuals = (
            interp(
                np.clip(np.roll(warped_pixels[:2, :].T, 1, axis=1), a_min=[0, 0], a_max=[height - 1, width - 1])
            ) - gray_image_prev[mask]
        ).reshape(-1, 1)

        logger.debug(f"Residuals (min, max, mean): ({residuals.min()}, {residuals.max()}, {residuals.mean()})")

        result = (residuals, )

        if compute_jacobian:
            result += (jacobian, )

        return result

    # TODO: Fix documentation
    def _compute_jacobian(self, image: np.ndarray, pointcloud: np.ndarray, mask: np.ndarray):
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
        J_w = compute_jacobian_of_warp_function(
            pointcloud=pointcloud, calibration_matrix=self._camera_model._intrinsics
        )

        gradx, grady = compute_gradients(image=image, kernel_size=3)

        # Filter out invalid pixels
        gradx = gradx[mask]
        grady = grady[mask]

        J = np.zeros((gradx.size, 6), dtype=np.float32)
        for i, gradients in enumerate(np.hstack((gradx.reshape(-1, 1), grady.reshape(-1, 1)))):
            J[i] = np.dot(gradients.reshape(1, 2), J_w[i])

        return J

    def _find_optimal_transformation(
        self, gray_image: np.ndarray, gray_image_prev: np.ndarray, depth_image_prev: np.ndarray, level: int = 0,
        init_guess: np.ndarray = np.zeros((6, 1), dtype=np.float32), max_iter: int = 100, tolerance: float = 1e-6
    ):
        """
            Given a pair of grayscale images and the corresponding depth one for the first image on the pair, it
            estimates the transformation between the two frames by means of the Newton-Gauss method.

        Parameters
        ----------
        gray_image : np.ndarray
            Grayscale image for current frame
        gray_image_prev : np.ndarray
            Grayscale image for previous frame
        depth_image_prev : np.ndarray
            Depth image for previous frame
        init_guess : np.ndarray, optional
            Init-guess for the transformation between the 2 frames expressed as a se(3) 6x1 array. Defaults to
            `[[0], [0], [0], [0], [0], [0]]`
        max_iter : int, optional
            Max number of iterations. Defaults to `100`
        tolerance : float, optional
            Tolerance used on stopping criteria. Defaults to `0.000001`

        Returns
        -------
        xi : np.ndarray
            Estimated transformation between frames expressed as a se(3) 6x1 array.
        info : dict, optional
            Information about the estimated transformation.

        Notes
        -----
        This method computes the residuals and jacobian at each iteration and then solving the system:
        :math:`(J_t \cdot W \cdot J) \cdot \Delta xi = - (J_t \cdot W \cdot r)` where :math:`J_t` is the transponse of
        the Jacobian (6xN), :math:`J` is the Jacobian (Nx6), :math:`W` is the weights matrix (i.e. NxN diagonal matrix
        with the corresponding weights for each residual) and :math:`r` is the residuals vector (Nx1). This method only
        considers valid pixels (i.e. where `depth_image_prev` is not equal to zero)
        """
        # Newton-Gauss method
        err_prev = np.finfo("float32").max
        xi = init_guess

        for i in range(max_iter):
            # Compute residuals
            if i == 0:
                # Compute Jacobian at identity
                residuals, jacobian_at_identity = self._compute_residuals(
                    gray_image=gray_image, gray_image_prev=gray_image_prev, depth_image_prev=depth_image_prev,
                    transformation=xi, level=level
                )
                jacobian_t = jacobian_at_identity.T

            else:
                (residuals, ) = self._compute_residuals(
                    gray_image=gray_image, gray_image_prev=gray_image_prev, depth_image_prev=depth_image_prev,
                    transformation=xi, level=level, compute_jacobian=False
                )

            # Computes weights if required
            if self._weighter is not None:

                weights = self._weighter.weight(residuals=residuals)
                err = np.sum(weights * (residuals ** 2))
                residuals = weights * residuals
                jacobian = weights * jacobian_at_identity

            else:
                jacobian = jacobian_at_identity
                err = np.linalg.norm(residuals) / np.sqrt(len(residuals))

            # Solve linear system: (Jt * W * J) * delta_xi = (-Jt * W * r) -> H * delta_xi = b
            H = np.dot(jacobian_t, jacobian)
            b = - np.dot(jacobian_t, residuals)

            # delta_xi = np.linalg.solve(H, b)

            # Q, R = np.linalg.qr(H)
            # delta_xi = solve_triangular(R, Q.T.dot(b))

            L = np.linalg.cholesky(H)
            y = solve_triangular(L, b, lower=True)
            delta_xi = solve_triangular(L.T, y)

            err_diff = err - err_prev

            logger.debug("Iteration {} -> error: {:.4f}".format(i + 1, err))

            # Stopping criteria (as shown on paper, error function always displays a global minima)
            if err_diff > 0.0:

                # Error increse, we keep last estimate and try best luck in next pyramid level
                logger.warning("Error increased on iteration {}, breaking out..".format(i + 1))
                break

            elif abs(err_diff) < tolerance:
                logger.info("Found convergence on iteration {}".format(i + 1))
                break

            err_prev = err

            # Update
            xi = SE3.log(np.dot(SE3.exp(delta_xi), SE3.exp(xi)))

            if i == (max_iter - 1):
                logger.warning("Exceeded maximum number of iterations ({})".format(max_iter))

        return xi

    def _step(
        self, gray_image: np.ndarray, depth_image: np.ndarray,
        init_guess: np.ndarray = np.zeros((6, 1), dtype=np.float32), max_iter: int = 100, tolerance: float = 1e-6,
    ):
        # Create coarse to fine Image Pyramids
        image_pyramids = CoarseToFineMultiImagePyramid(
            images=[gray_image, self._gray_image_prev, self._depth_image_prev],
            levels=self.levels
        )

        for i, (gray_image, gray_image_prev, depth_image_prev) in enumerate(image_pyramids):
            transformation = self._find_optimal_transformation(
                gray_image=gray_image, gray_image_prev=gray_image_prev, depth_image_prev=depth_image_prev,
                init_guess=init_guess, tolerance=tolerance, max_iter=max_iter, level=(i - self.levels + 1)
            )
            init_guess = transformation

        # Clean cache
        self._camera_model.deproject.cache_clear()
        self._camera_model.project.cache_clear()

        return transformation
