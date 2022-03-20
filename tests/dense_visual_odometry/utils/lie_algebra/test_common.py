import numpy as np

from dense_visual_odometry.utils.lie_algebra.common import is_rotation_matrix, wrap_angle

from unittest import TestCase


class TestIsRotationMatrix(TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test__given_a_valid_rotation_matrix__when_is_rotation_matrix__then_true(self):

        # Given
        matrix = np.array([[-0.0195074,  0.4347243, -0.9003523],
                           [0.57775190,  0.7398499,  0.3447099],
                           [0.81597920, -0.5134559, -0.2655953]])

        # When
        result = is_rotation_matrix(matrix)

        # Then
        self.assertTrue(isinstance(result, bool))
        self.assertTrue(result)

    def test__given_a_non_valid_rotation_matrix__when_is_rotation_matrix__then_false(self):

        # Given
        matrix = np.array([[-0.0195074,  0.4347243, -0.9003523],
                           [0.81597920, -0.5134559, -0.2655953],
                           [0.57775190,  0.7398499,  0.3447099]])

        # When
        result = is_rotation_matrix(matrix)

        # Then
        self.assertTrue(isinstance(result, bool))
        self.assertFalse(result)

    def test__given_an_incorrect_matrix_shape__when_is_rotation_matrix__then_raises_assertion(self):

        # Given
        matrix = np.zeros((2, 2))

        # When and Then
        with self.assertRaises(AssertionError):
            _ = is_rotation_matrix(matrix)


class TestWrapAngle(TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test__given_an_angle_of_3_pi__when_wrap_angle__then_minus_pi(self):

        # Given
        angle = 3 * np.pi

        # When
        wrapped_angle = wrap_angle(angle)

        # Then
        self.assertTrue(isinstance(wrapped_angle, float))
        np.testing.assert_almost_equal(wrapped_angle, -np.pi)

    def test__given_an_angle_of_0__when_wrap_angle__then_0(self):

        # Given
        angle = 0.0

        # When
        wrapped_angle = wrap_angle(angle)

        # Then
        self.assertTrue(isinstance(wrapped_angle, float))
        np.testing.assert_almost_equal(wrapped_angle, 0.0)

    def test__given_an_array__when_wrap_angle__then_ok(self):

        # Given
        angles = np.array([0.0, 3 * np.pi, 5 * np.pi / 2, 4.95 * np.pi])

        # When
        wrapped_angles = wrap_angle(angles)

        # Then
        self.assertTrue(isinstance(wrapped_angles, np.ndarray))
        np.testing.assert_array_almost_equal(wrapped_angles, np.array([0.0, -np.pi, np.pi / 2, 0.95 * np.pi]))
