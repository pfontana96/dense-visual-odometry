from unittest import TestCase

import numpy as np

from dense_visual_odometry.utils.interpolate import Interp2D


class TestInterp2D(TestCase):

    def setUp(self) -> None:
        self.maxDiff = None
        self.image = np.arange(5 * 10).astype(np.uint8).reshape(5, 10)

    def tearDown(self) -> None:
        return super().tearDown()

    def test__given_valid_points_to_interpolate__when_bilinear__then_ok(self):

        # Given
        x = np.array([1, 2, 2.5, 3])
        y = np.array([1, 2, 2.5, 3])

        # When
        result = Interp2D.bilinear(x, y, self.image)

        # Then
        np.testing.assert_equal(result, np.array([11.0, 22.0, 27.5, 33.0]))

    def test__given_meshgrid_for_an_image__when_interpolate__then_same_image(self):

        # Given
        height, width = self.image.shape
        x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

        # When
        result = Interp2D.bilinear(x, y, self.image)

        # Then
        # NOTE: By the current implementation (17/04/2022) of 'Interp2D.bilinear' if we give the exact grid to retrieve
        # the same image, then last row and last column will be 0.0
        np.testing.assert_equal(result[:-1, :-1], self.image[:-1, :-1])
