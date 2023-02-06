import logging
import abc

import numpy as np

from dense_visual_odometry.core.base_dense_visual_odometry import BaseDenseVisualOdometry
from dense_visual_odometry.utils.image_pyramid import CoarseToFineMultiImagePyramid
from dense_visual_odometry.utils.lie_algebra import Se3
from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.weighter.t_weighter import TDistributionWeighter


logger = logging.getLogger(__name__)


class BaseKerlDVO(BaseDenseVisualOdometry, abc.ABC):
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
        max_increased_steps_allowed: int = 0, sigma: float = None, tolerance: float = 1e-6, max_iterations: int = 100,
        mu: float = None, approximate_image2_gradient: bool = False
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
        super(BaseKerlDVO, self).__init__(camera_model=camera_model, initial_pose=initial_pose, weighter=weighter)
        self.levels = levels
        self._max_increased_steps_allowed = max_increased_steps_allowed

        self._sigma = sigma
        if self._sigma is not None:
            self._inv_cov = (1 / self._sigma) * np.eye(6, dtype=np.float32)

        self._mu = mu

        self._tolerance = tolerance
        self._max_iter = max_iterations

        self._approximate_image2_gradients = approximate_image2_gradient

    @abc.abstractmethod
    def _least_squares_setup(
        self, gray_image: np.ndarray, depth_image: np.ndarray, gray_image_prev: np.ndarray,
        depth_image_prev: np.ndarray, level: int
    ):
        """
        Setup method prior to the call of `_non_linear_least_squares`
        """
        pass

    @abc.abstractmethod
    def _least_squares_cleanup(
        self, gray_image: np.ndarray, depth_image: np.ndarray, gray_image_prev: np.ndarray,
        depth_image_prev: np.ndarray, level: int
    ):
        """
        Clean method posterior to the call of `_non_linear_least_squares`
        """
        pass

    def _step(self, gray_image: np.ndarray, depth_image: np.ndarray, init_guess: Se3 = Se3.identity()):
        # Create coarse to fine Image Pyramids
        image_pyramids = CoarseToFineMultiImagePyramid(
            images=[gray_image, self._gray_image_prev, depth_image, self._depth_image_prev],
            levels=self.levels
        )

        estimate = init_guess.copy()

        for i, (gray_image_l, gray_image_prev_l, depth_image_l, depth_image_prev_l) in enumerate(image_pyramids):

            level = self.levels - 1 - i

            self._least_squares_setup(
                gray_image=gray_image_l, depth_image=depth_image_l, gray_image_prev=gray_image_prev_l, 
                level=level
            )

            estimate = self._non_linear_least_squares(
                init_guess=estimate, gray_image=gray_image_l, depth_image=depth_image_l,
                gray_image_prev=gray_image_prev_l, depth_image_prev=depth_image_prev_l, level=-level
            )

            self._least_squares_cleanup(
                gray_image=gray_image_l, depth_image=depth_image_l, gray_image_prev=gray_image_prev_l, 
                level=level
            )

        return estimate

    @abc.abstractmethod
    def _non_linear_least_squares(
        self, init_guess: Se3, gray_image: np.ndarray, gray_image_prev: np.ndarray, depth_image: np.ndarray,
        depth_image_prev: np.ndarray, level: int = 0
    ):
        pass
