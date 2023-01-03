import math

import numpy as np

from dense_visual_odometry.utils.lie_algebra import Se3, So3

from unittest import TestCase


class TestSE3(TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test__given_xi__when_hat__then_ok(self):

        # Given
        i = math.sqrt(1 / 3)
        xi = np.array([1, 2, 3, i, i, i], dtype=np.float32).reshape(6, 1)
        pose = Se3.from_se3(xi)

        # When
        xi_hat = pose.hat()

        # Then
        expected_xi_hat = np.array(
            [[0.0, -i, i, 1.0],
             [i, 0.0, -i, 2.0],
             [-i, i, 0.0, 3.0],
             [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32
        )

        np.testing.assert_almost_equal(xi_hat, expected_xi_hat, decimal=6)

    def test__given_wrong_shape__when_from_se3__then_raises_assertion(self):

        # Given
        xi = np.array([1, 2, 3, 4, 5, 6])

        # When and Then
        with self.assertRaises(AssertionError):
            _ = Se3.from_se3(xi)

    # def test__given_xi__when_curly_hat__then_ok(self):

    #     # Given
    #     xi = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape(6, 1)

    #     # When
    #     xi_curly_hat = SE3.curly_hat(xi)

    #     # Then
    #     expected_xi_curly_hat = np.array(
    #         [[0.0, -6., 5.0, 0.0, -3., 2.0],
    #          [6.0, 0.0, -4., 3.0, 0.0, -1.],
    #          [-5., 4.0, 0.0, -2., 1.0, 0.0],
    #          [0.0, 0.0, 0.0, 0.0, -6., 5.0],
    #          [0.0, 0.0, 0.0, 6.0, 0.0, -4.],
    #          [0.0, 0.0, 0.0, -5., 4.0, 0.0]],
    #         dtype=np.float32
    #     )

    #     np.testing.assert_almost_equal(xi_curly_hat, expected_xi_curly_hat, decimal=6)

    # def test__given_wrong_type__when_curly_hat__then_raises_assertion(self):

    #     # Given
    #     xi = [1, 2, 3, 4, 5, 6]

    #     # When and Then
    #     with self.assertRaises(AssertionError):
    #         _ = SE3.curly_hat(xi)

    # def test__given_wrong_shape__when_curly_hat__then_raises_assertion(self):

    #     # Given
    #     xi = np.array([1, 2, 3, 4, 5, 6])

    #     # When and Then
    #     with self.assertRaises(AssertionError):
    #         _ = SE3.curly_hat(xi)

    def test__given_xi__when_exp__then_ok(self):

        # Given
        xi = np.array([1.4852, -3.156, -4.578, 0.4893, 0.3232, -1.2345], dtype=np.float32).reshape(6, 1)
        pose = Se3.from_se3(xi)

        # When
        T = pose.exp()

        # Then
        expected_T = np.array(
            [[0.30488090, 0.9520282, -0.0262667, -0.8325922],
             [-0.8170195, 0.2472735, -0.5208982, -1.8249098],
             [-0.4894148, 0.1802723,  0.8532146, -5.1481801],
             [0.00000000, 0.0000000,  0.0000000, 1.00000000]],
            dtype=np.float32
        )

        np.testing.assert_almost_equal(T, expected_T, decimal=6)

    def test__given_singular_xi__when_exp__then_identity(self):

        # Given
        xi = np.zeros((6, 1), dtype=np.float32)
        pose = Se3.from_se3(xi)

        # When
        T = pose.exp()

        # Then
        expected_T = np.eye(4, dtype=np.float32)

        np.testing.assert_almost_equal(T, expected_T, decimal=6)

    def test__given_T__when_log__then_ok(self):

        # Given
        T = np.array(
            [[0.30488090, 0.9520282, -0.0262667, -0.8325922],
             [-0.8170195, 0.2472735, -0.5208982, -1.8249098],
             [-0.4894148, 0.1802723,  0.8532146, -5.1481801],
             [0.00000000, 0.0000000,  0.0000000, 1.00000000]],
            dtype=np.float32
        )
        pose = Se3(So3(T[:3, :3]), T[:3, 3].reshape(3, 1))

        # When
        xi = pose.log()

        # Then
        expected_xi = np.array([1.4852, -3.156, -4.578, 0.4893, 0.3232, -1.2345], dtype=np.float32).reshape(6, 1)

        np.testing.assert_almost_equal(xi, expected_xi, decimal=6)

    def test__given_identity_T__when_log__then_zeros(self):

        # Given
        T = np.eye(4, dtype=np.float32)
        pose = Se3(So3(T[:3, :3]), T[:3, 3].reshape(3, 1))

        # When
        xi = pose.log()

        # Then
        expected_xi = np.zeros((6, 1), dtype=np.float32)

        np.testing.assert_almost_equal(xi, expected_xi, decimal=6)
