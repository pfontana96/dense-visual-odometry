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

    def __getitem__(self, level: int):
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
        if (level >= self.levels) or (level < 0):
            raise IndexError("Expected 'level' to be in the range [0, {}], got {} instead".format(
                self.levels - 1, level
            ))

        return self.pyramid[level]


class MultiImagePyramid:

    def __init__(self, images: list, levels: int):
        self._levels = levels
        self._nb_of_images = len(images)
        self._pyramids = [ImagePyramid(image=image, levels=self._levels) for image in images]

    @property
    def levels(self):
        return self._levels

    @property
    def images_count(self):
        return self._nb_of_images


class CoarseToFineMultiImagePyramid(MultiImagePyramid):

    def __init__(self, images: list, levels: int):
        super(CoarseToFineMultiImagePyramid, self).__init__(images=images, levels=levels)
        self._counter = self.levels

    def __iter__(self):
        self._counter = self.levels
        return self

    def __next__(self):
        if self._counter >= self.levels:
            raise StopIteration

        result = (pyramid[self._counter] for pyramid in self.pyramids)
        self._counter -= 1

        return result
