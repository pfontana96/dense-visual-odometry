import cv2
import numpy as np
import logging


logger = logging.getLogger(__name__)


class ImagePyramidError(Exception):
    pass


class ImagePyramid:

    def __init__(self, image: np.ndarray, levels: int):
        assert levels > 0, f"Expected number of levels to be greater than 0, got {levels} instead"
        self.levels = levels
        self.pyramid = [None] * self.levels

        # Create Pyramid
        try:
            self.pyramid[0] = image
            for level in range(1, self.levels):
                self.pyramid[level] = cv2.pyrDown(self.pyramid[level - 1])
        except cv2.error as e:
            logger.error(e)
            raise ImagePyramidError("Could not create Image Pyramid")

    def get(self, level: int):
        """
            Returns a level of the pyramid

        Parameters
        ----------
        level : int
            Level to return

        Returns
        -------
        image : np.ndarray
            Image corresponding to 'level'
        """
        assertion_message = f"'level' should be in the range [0, {self.levels - 1}], got {level} instead"
        assert (level < self.levels) and (level >= 0), assertion_message

        return self.pyramid[level]
