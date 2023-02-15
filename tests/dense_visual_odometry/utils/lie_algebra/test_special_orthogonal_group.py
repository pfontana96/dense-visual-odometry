import numpy as np

from dense_visual_odometry.utils.lie_algebra import So3

from unittest import TestCase


class TestSO3(TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test__given_phi__when_hat__then_ok(self):

        # Given
        array = np.array([1, 2, 3], dtype=np.float32).reshape(3, 1)
        theta = np.pi / 5
        phi = theta * array / np.linalg.norm(array)
        rot = So3(phi)

        # When
        phi_hat = rot.hat()

        # Then
        expected_phi_hat = np.array(
            [[0.0, -phi[2], phi[1]],
             [phi[2], 0.0, -phi[0]],
             [-phi[1], phi[0], 0.0]],
            dtype=np.float32
        )

        np.testing.assert_almost_equal(phi_hat, expected_phi_hat, decimal=6)

    def test__given_wrong_type__when_init__then_raises_assertion(self):

        # Given
        phi = [1, 2, 3]

        # When and Then
        with self.assertRaises(AssertionError):
            _ = So3(phi)

    def test__given_wrong_shape__when_init__then_raises_value(self):

        # Given
        phi = np.array([1, 2, 3, 4, 5, 6])

        # When and Then
        with self.assertRaises(ValueError):
            _ = So3(phi)

    def test__given_phi__when_exp__then_ok(self):

        # Given
        phi = np.array([30, 60, -5], dtype=np.float32).reshape(3, 1) * (np.pi / 180.0)
        rot = So3(phi)

        # When
        rot_mat = rot.exp()

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
        rot = So3(phi)

        # When
        rot_mat = rot.exp()

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
        rot = So3(rot_mat)

        # When
        phi = rot.log()

        # Then
        expected_phi = np.array([30, 60, -5], dtype=np.float32).reshape(3, 1) * (np.pi / 180.0)

        assert (np.allclose(phi, expected_phi, atol=1e-6) or np.allclose(-phi, expected_phi, atol=1e-6))
        np.testing.assert_almost_equal(rot.exp(), rot_mat, decimal=6)

    def test__given_identity_rot_mat__when_log__then_zeros(self):

        # Given
        rot_mat = np.eye(3, dtype=np.float32)
        rot = So3(rot_mat)

        # When
        phi = rot.log()

        # Then
        expected_phi = np.zeros((3, 1), dtype=np.float32)

        np.testing.assert_almost_equal(phi, expected_phi, decimal=6)
