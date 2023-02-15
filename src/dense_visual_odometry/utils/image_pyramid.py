import logging
import math
import abc

import cv2
import numpy as np
import numpy.typing as npt

from dense_visual_odometry.cuda import UnifiedMemoryArray


logger = logging.getLogger(__name__)


class ImagePyramidError(Exception):
    pass


def pyrDownMedianSmooth(image: np.ndarray):
    out = cv2.medianBlur(image, 3)
    return out[::2, ::2]  # Drop even rows and columns


class BaseImagePyramid(abc.ABC):

    @abc.abstractmethod
    def at(self, **kwargs):
        pass

    @abc.abstractproperty
    def levels(self):
        pass


class ImagePyramid(BaseImagePyramid):
    def __init__(self, levels: int, image: npt.ArrayLike):

        if (levels < 0):
            raise ValueError("Expected 'levels' to be non-negative, got '{}' instead".format(
                levels
            ))

        self._levels = levels

        self._pyramid = [None] * self._levels
        # Create Pyramids
        try:
            self._pyramid[0] = image

            for level in range(1, self._levels):
                self._pyramid[level] = pyrDownMedianSmooth(self._pyramid[level - 1])

        except Exception as e:
            raise ImagePyramidError("Could not create Image Pyramid, got '{}'".format(e))

    @property
    def levels(self):
        return self._levels

    def at(self, level: int) -> npt.ArrayLike:
        if (level < 0) or level >= (self._levels):
            raise IndexError("'level' out of range [0, {}], got {} instead".format(
                self.levels - 1, level
            ))
        return self._pyramid[level]


class ImagePyramidGPU(BaseImagePyramid):

    def __init__(self, levels: int, image: npt.ArrayLike, dtype: npt.DTypeLike):
        self._levels = levels
        self._dtype = dtype

        self._shape = image.shape

        # Create Pyramid
        self._pyramid = [None] * self._levels
        self._pyramid[0] = UnifiedMemoryArray(self._shape, self._dtype, image)

        try:
            height, width = self._shape

            for i in range(1, self._levels):
                height = math.ceil(height / 2)
                width = math.ceil(width / 2)

                self._pyramid[i] = UnifiedMemoryArray(
                    (height, width), self._dtype, pyrDownMedianSmooth(self._pyramid[i - 1].get("cpu"))
                )

        except Exception as e:
            raise ImagePyramidError("Could not create RGBD Image Pyramid, got '{}'".format(e))

    @property
    def levels(self):
        return self._levels

    def update(self, image: npt.NDArray):

        self._pyramid[0].get("cpu")[...] = image

        try:
            for i in range(1, self._levels):
                self._pyramid[i].get("cpu")[...] = pyrDownMedianSmooth(self._pyramid[i - 1].get("cpu"))

        except Exception as e:
            raise ImagePyramidError("Could not create RGBD Image Pyramid, got '{}'".format(e))

    def at(self, level: int, device: str = "cpu") -> npt.ArrayLike:
        assert (level >= 0) and level < (self._levels), "'level' out of range [0, {}], got {} instead".format(
            self.levels - 1, level
        )

        return self._pyramid[level].get(device)
