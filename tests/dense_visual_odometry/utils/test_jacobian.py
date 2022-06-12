from unittest import TestCase

import numpy as np

from dense_visual_odometry.utils.jacobian import compute_jacobian_of_warp_function, compute_gradients


class TestComputeJacobianOfWarpFunction(TestCase):

    def setUp(self) -> None:
        self.maxDiff = None

    def tearDown(self) -> None:
        return super().tearDown()

    def test__given_pointcloud__when_compute_jacobian_of_warp_function__then_ok(self):

        # Given
        pointcloud = np.array([
            [1, 4],
            [2, 5],
            [3, 6],
            [1, 1],
        ], dtype=np.float32)

        calibration_matrix = np.eye(3, dtype=np.float32)

        # When
        result = compute_jacobian_of_warp_function(pointcloud=pointcloud, calibration_matrix=calibration_matrix)

        # Then
        expected_result = np.array([
            [[1 / 3, 0, -(1 / 9), -(2 / 9), 10 / 9, -(2 / 3)],
             [0, 1 / 3, -(2 / 9), -(13 / 9), 2 / 9, 1 / 3]],
            [[1 / 6, 0, -(1 / 9), -(5 / 9), 13 / 9, -(5 / 6)],
             [0, 1 / 6, -(5 / 36), -(61 / 36), 5 / 9, 2 / 3]]
        ], dtype=np.float32)
        np.testing.assert_array_equal(result, expected_result)


class TestComputeGradients(TestCase):

    def setUp(self) -> None:
        self.maxDiff = None

    def tearDown(self) -> None:
        return super().tearDown()

    def test__given_known_image__when_compute_gradients__then_ok(self):

        # Given
        # Horizontal line
        image = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)

        # When
        gradx, grady = compute_gradients(image=image, kernel_size=3)

        # Then
        expected_grady = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 4.0, 4.0, 4.0, 4.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-4.0, -4.0, -4.0, -4.0, -4.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]
        ], dtype=np.float32)
        np.testing.assert_almost_equal(gradx, np.zeros_like(image))
        np.testing.assert_almost_equal(grady, expected_grady)
