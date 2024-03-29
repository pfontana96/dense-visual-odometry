from typing import Type
import abc
from pathlib import Path
import logging

import numpy as np
import cv2

from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.weighter import BaseWeighter
from dense_visual_odometry.utils.lie_algebra import Se3


logger = logging.getLogger(__name__)


class DVOError(Exception):
    pass


class BaseDenseVisualOdometry(abc.ABC):
    """
        Base class for performing Visual odometry using RGBD images
    """
    def __init__(
        self, camera_model: RGBDCameraModel, weighter: Type[BaseWeighter] = None,
        initial_pose: Se3 = Se3.identity(), max_distance: float = 5.0, debug_dir: Path = None
    ):
        self._camera_model = camera_model

        self._weighter = weighter

        self._initial_pose = initial_pose
        self._current_pose = self._initial_pose.copy()
        self._last_pose = None

        self._last_estimated_transform = None

        self._max_distance = max_distance

        # We need to save the last frame on memory
        self._gray_image_prev = None
        self._depth_image_prev = None

        self._debug_dir = debug_dir

    @abc.abstractmethod
    def _step(
        self, gray_image: np.ndarray, depth_image: np.ndarray,
        init_guess: Se3 = Se3.identity(), **kwargs
    ):
        pass

    def step(
        self, color_image: np.ndarray, depth_image: np.ndarray,
        init_guess: Se3 = Se3.identity(), **kwargs
    ):
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        depth_image[(depth_image * self._camera_model.depth_scale) > self._max_distance] = 0  # Possible noisy points

        if (self._gray_image_prev is None) and (self._depth_image_prev is None):
            # First frame
            transform = Se3.identity()

        else:
            # Call subclass method (transform must be in se(3))
            transform = self._step(
                gray_image=gray_image, depth_image=depth_image, init_guess=init_guess, **kwargs
            )

        # Update if transform estimated is not None
        # 'transform' is transform of camera {t-1}_to_{t} and '_current_pose' is {t-1}_to_{world} so most recent current
        # pose (ergo {t}_to_{world}) will be:
        # current_pose = {t-1}_to_{world} * {t}_to_{t-1} = {t-1}_to_{world} * ({t-1}_to_{t})^(-1)
        if transform is not None:
            self._last_pose = self._current_pose.copy()
            self._last_estimated_transform = transform.copy()

            self._current_pose = self._current_pose * transform.inverse()

            self._gray_image_prev = gray_image
            self._depth_image_prev = depth_image

        else:
            logger.warning("DVO could not estimate transform, trying luck on next frame..")

        return transform

    @property
    def current_pose(self):
        return self._current_pose
