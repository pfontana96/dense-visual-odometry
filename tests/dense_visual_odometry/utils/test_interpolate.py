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
