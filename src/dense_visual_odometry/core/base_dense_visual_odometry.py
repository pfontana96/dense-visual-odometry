from typing import Type
import abc

import numpy as np
import cv2

from dense_visual_odometry.camera_model import RGBDCameraModel
from dense_visual_odometry.weighter import BaseWeighter
from dense_visual_odometry.utils.lie_algebra import SE3


class BaseDenseVisualOdometry(abc.ABC):
    """
        Base class for performing Visual odometry using RGBD images
    """
    def __init__(
        self, camera_model: RGBDCameraModel, weighter: Type[BaseWeighter] = None,
        initial_pose: np.array = np.zeros((6, 1), dtype=np.float32)
    ):
        self._camera_model = camera_model

        self._weighter = weighter

        if initial_pose.shape == (4, 4):
            self._initial_pose = SE3.log(initial_pose)
        elif initial_pose.shape == (6, 1):
            self._initial_pose = initial_pose
        else:
            raise ValueError("Expected 'initial_pose' to have shape either (4, 4) or (6, 1) go '{}' instead".format(
                initial_pose.shape
            ))
        self._current_pose = self._initial_pose.copy()

        # We need to save the last frame on memory
        self.gray_image_prev = None
        self.depth_image_prev = None

    @abc.abstractmethod
    def _step(
        self, gray_image: np.ndarray, depth_image: np.ndarray,
        init_guess: np.ndarray = np.zeros((6, 1), dtype=np.float32), **kwargs
    ):
        pass

    def step(
        self, color_image: np.ndarray, depth_image: np.ndarray,
        init_guess: np.ndarray = np.zeros((6, 1), dtype=np.float32), **kwargs
    ):
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        if (self.gray_image_prev is None) and (self.depth_image_prev is None):
            # First frame
            transform = np.zeros((6, 1), dtype=np.float32)

        else:
            # Call subclass method (transform must be in se(3))
            transform = self._step(
                gray_image=gray_image, depth_image=depth_image, init_guess=init_guess, **kwargs
            )

        # Update
        self._current_pose = SE3.log(np.dot(SE3.exp(transform), SE3.exp(self._current_pose)))
        self.gray_image_prev = gray_image
        self.depth_image_prev = depth_image

        return transform

    @property
    def current_pose(self):
        return self._current_pose
