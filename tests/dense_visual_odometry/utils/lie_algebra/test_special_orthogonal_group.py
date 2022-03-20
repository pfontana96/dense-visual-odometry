import numpy as np

from dense_visual_odometry.utils.lie_algebra.special_orthogonal_group import SO3

from unittest import TestCase


class TestSO3(TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test__given_phi__when_hat__then_ok(self):

        # Given
        phi = np.array([1, 2, 3], dtype=np.float32).reshape(3, 1)

        # When
        phi_hat = SO3.hat(phi)

        # Then
        expected_phi_hat = np.array(
            [[0.0, -3., 2.0],
             [3.0, 0.0, -1.],
             [-2., 1.0, 0.0]],
            dtype=np.float32
        )

        np.testing.assert_almost_equal(phi_hat, expected_phi_hat, decimal=6)

    def test__given_phi__when_exp__then_ok(self):

        # Given
        phi = np.array([30, 60, -5], dtype=np.float32).reshape(3, 1) * (np.pi / 180.0)

        # When
        rot_mat = SO3.exp(phi)

        # Then
        expected_rot_mat = np.array(
            [[0.50845740, 0.3126321,  0.8023292],
             [0.17552060, 0.8745719, -0.4520139],
             [-0.8430086, 0.3706551,  0.3898093]],
            dtype=np.float32
        )

        np.testing.assert_almost_equal(rot_mat, expected_rot_mat, decimal=6)

    def test__given_singular_phi__when_exp__then_identity(self):

        # Given
        phi = np.zeros((3, 1), dtype=np.float32)

        # When
        rot_mat = SO3.exp(phi)

        # Then
        expected_rot_mat = np.eye(3, dtype=np.float32)

        np.testing.assert_almost_equal(rot_mat, expected_rot_mat, decimal=6)

    def test__given_rot_mat__when_log__then_ok(self):

        # Given
        rot_mat = np.array(
            [[0.50845740, 0.3126321,  0.8023292],
             [0.17552060, 0.8745719, -0.4520139],
             [-0.8430086, 0.3706551,  0.3898093]],
            dtype=np.float32
        )

        # When
        phi = SO3.log(rot_mat)

        # Then
        expected_phi = np.array([30, 60, -5], dtype=np.float32).reshape(3, 1) * (np.pi / 180.0)

        np.testing.assert_almost_equal(phi, expected_phi, decimal=6)

    def test__given_identity_rot_mat__when_log__then_zeros(self):

        # Given
        rot_mat = np.eye(3, dtype=np.float32)

        # When
        phi = SO3.log(rot_mat)

        # Then
        expected_phi = np.zeros((3, 1), dtype=np.float32)

        np.testing.assert_almost_equal(phi, expected_phi, decimal=6)
