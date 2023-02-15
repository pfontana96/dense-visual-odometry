import abc
import logging
from typing import Tuple

import numpy as np
import numpy.typing as npt
import numba as nb
from scipy.linalg import lstsq

from dense_visual_odometry.utils.lie_algebra import Se3
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.weighter import TDistributionWeighter
from dense_visual_odometry.core.base_dense_visual_odometry import BaseDenseVisualOdometry


logger = logging.getLogger(__name__)


@nb.njit("float32[:,:](float32[:,:], float32[:,:])", fastmath=True)
def nb_lstsq(a, b):
    result, _, _, _ = np.linalg.lstsq(a=a, b=b)
    return result


class BaseRobustDVO(BaseDenseVisualOdometry, abc.ABC):
    """
        Class for performing dense visual odometry by minimizing the photometric error based on [1]_.

    Notes
    ----------
    .. [1] Kerl, C., Sturm, J., Cremers, D., "Robust Odometry Estimation for RGB-D Cameras"
    """

    def __init__(
        self, camera_model: RGBDCameraModel, initial_pose: Se3, levels: int, use_weighter: bool = False,
        max_increased_steps_allowed: int = 0, sigma: float = None, tolerance: float = 1e-6, max_iterations: int = 100,
        approximate_image2_gradient: bool = False
    ):
        """
        Parameters
        ----------
        camera_model : RGBDCameraModel
            Camera model to use.
        initial_pose : Se3
            Initial pose of camera.
        levels : int
            Number of levels to build the image pyramids.
        use_weighter : bool, optional
            If True then `TDistributionWeighter` is use as a scale method, by default False.
        max_increased_steps_allowed : int, optional
            Maximum numbers of step to allow the photometric error to increase in the non-linear least squares
            minimization, by default 0.
        sigma : float, optional
            Hyperparameter to control the influence of the prior estimate, by default None. It turns the
            non-linear least squares into:
            `(Jt * W * J + (1/sigma) * I) * delta_xi = -Jt * W * residuals[0] + (1/sigma) * I * (xi[t - 1] - xi[t]_k)`
        tolerance : float, optional
            Tolerance to achieve convergence, by default 1e-6.
        max_iterations : int, optional
            Maximum number of iterations per pyramid level, by default 100.
        approximate_image2_gradient : bool, optional
            Whether to approximate the gradient of `I2(warped(x1))` as the gradient of `I1(x1)`, by default False.
        """
        weighter = TDistributionWeighter() if use_weighter else None
        super(BaseRobustDVO, self).__init__(camera_model=camera_model, initial_pose=initial_pose, weighter=weighter)
        self._levels = levels
        self._max_increased_steps_allowed = max_increased_steps_allowed

        self._sigma = sigma
        if self._sigma is not None:
            self._inv_cov = (1 / self._sigma) * np.eye(6, dtype=np.float32)

        self._tolerance = tolerance
        self._max_iter = max_iterations

        self._approximate_image2_gradients = approximate_image2_gradient

        # Pyramids depends on implementation (whether CPU or GPU)
        self._curr_gray_image_pyr = None
        self._prev_gray_image_pyr = None

        self._curr_depth_image_pyr = None
        self._prev_depth_image_pyr = None

    @property
    def levels(self) -> int:
        """Number of levels of the Image pyramid
        """
        return self._levels

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def _build_pyramids(
        self, gray_image: npt.NDArray[np.uint8], depth_image: npt.NDArray[np.uint16]
    ):
        """Method resposible for setting `self._prev_gray_image_pyr`, `self._prev_depth_image_pyr`,
        `self._curr_gray_image_pyr` and `self._curr_depth_image_pyr`
        """
        pass

    @abc.abstractmethod
    def _setup(self, level: int):
        """Optional set up method called before executing Non-Linear Least Squares approximation at each pyramid level.
        """
        pass

    @abc.abstractmethod
    def _cleanup(self):
        """Optional clean up method called before executing Non-Linear Least Squares approximation at each pyramid
        level.
        """
        pass

    def _step(
        self, gray_image: npt.NDArray[np.uint8], depth_image: npt.NDArray[np.uint16], init_guess: Se3 = Se3.identity()
    ):
        # Create coarse to fine Image Pyramids
        self._build_pyramids(gray_image=gray_image, depth_image=depth_image)

        if (self._curr_gray_image_pyr is None) or (self._curr_depth_image_pyr is None):
            raise NotImplementedError("Call to _build_curr_pyramids did not correctly set pyramids")

        if (self._prev_gray_image_pyr is None) or (self._prev_depth_image_pyr is None):
            raise NotImplementedError("Call to _build_prev_pyramids did not correctly set pyramids")

        estimate = init_guess.copy()

        # Non linear least squares
        for level in range(self.levels - 1, -1, -1):

            old = self._last_estimated_transform.copy()
            # NOTE: Consider implementing initial smoothing
            initial = estimate.copy()  # noqa

            err_prev = np.finfo("float32").max
            err_increased_count = 0

            self._setup(level=level)

            for i in range(self._max_iter):

                # Compute residuals
                residuals, jacobian, mask = self.compute_residuals_and_jacobian(estimate=estimate, level=level)

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

                inc_xi, _, _, _ = lstsq(
                    a=H, b=b, lapack_driver="gelsy", overwrite_a=True, overwrite_b=True, check_finite=False
                )
                # inc_xi = nb_lstsq(a=H, b=b)

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

            self._cleanup()

        return estimate
