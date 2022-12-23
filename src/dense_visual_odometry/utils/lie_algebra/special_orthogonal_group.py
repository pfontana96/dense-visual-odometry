# This auxiliary module contains a numpy implementation of some functions for manipulating
# Special Orthogonal and Special Eucledian groups (Lie's algebra)
# It was based on Tim Barfoot's book available on:
# http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf

import numpy as np

from dense_visual_odometry.utils.lie_algebra.base_special_group import BaseSpecialGroup, _LIE_EPSILON
from dense_visual_odometry.utils.lie_algebra.common import is_rotation_matrix, wrap_angle


class SO3(BaseSpecialGroup):

    @staticmethod
    def hat(phi: np.ndarray):
        assert type(phi) == np.ndarray, "'phi' should be a numpy array"
        assert phi.shape == (3, 1), "Expected shape of (3,1) for 'phi', got {} instead".format(phi.shape)
        hat = np.zeros((3, 3), dtype=np.float32)
        hat[0, 1] = -phi[2]
        hat[0, 2] = phi[1]
        hat[1, 0] = phi[2]
        hat[1, 2] = -phi[0]
        hat[2, 0] = -phi[1]
        hat[2, 1] = phi[0]

        return hat

    @staticmethod
    def exp(phi: np.ndarray):
        """
            Exponential mapping from so(3) to SO(3), i.e, R3 --> R3x3

        Parameters
        ----------
        phi : np.ndarray
            so(3) array

        Returns
        -------
        rot_mat : np.ndarray
            Rotation matrix (3,3) corresponding to the exponential map of 'phi'
        """
        assert type(phi) == np.ndarray, "'phi' should be a numpy array"
        assert phi.shape == (3, 1), "Expected shape of (3,1) for 'phi', got {} instead".format(phi.shape)
        theta = np.linalg.norm(phi)

        # Check for 0 rad rotation
        if np.abs(theta) < _LIE_EPSILON:
            return np.eye(3)
        a = phi / theta

        # After normalizing the axis-angle vector, wrap the angle between [-pi;pi]
        theta = wrap_angle(theta)
        a_hat = SO3.hat(a)

        return np.eye(3, dtype=np.float32) + np.sin(theta)*a_hat + (1.0 - np.cos(theta))*np.dot(a_hat, a_hat)

    @staticmethod
    def log(rot_mat):
        """
            Logarithmic mapping from SO(3) to so(3), i.e, R3x3 --> R3

        Parameters
        ----------
        rot_mat : np.ndarray
            Rotation matrix

        Returns
        -------
        phi : np.ndarray
            so(3) vector corresponding to the logarithmic map of 'rot_map'
        """
        assert type(rot_mat) == np.ndarray, "'rot_mat' should be a numpy array"
        assert is_rotation_matrix(rot_mat), "'rot_mat' is not a valid rotation matrix"

        phi = np.zeros((3, 1), dtype=np.float32)
        theta = np.arccos((np.trace(rot_mat) - 1.0) / 2.0)
        if np.abs(theta) < _LIE_EPSILON:
            return phi

        # Wrap the angle between [-pi;pi]
        theta = wrap_angle(theta)

        phi_hat = (theta/(2.0*np.sin(theta)))*(rot_mat - rot_mat.T)
        phi[0] = phi_hat[2, 1]
        phi[1] = phi_hat[0, 2]
        phi[2] = phi_hat[1, 0]

        return phi

    @staticmethod
    def left_jacobian(phi: np.ndarray):
        """
            Computes SO(3) left jacobian

        Parameters
        ----------
        phi : np.ndarray
            so(3) array

        Returns
        -------
        left_jacobian : np.ndarray
            Left jacobian of 'phi' (3,3)
        """
        assert type(phi) == np.ndarray, "'phi' should be a numpy array"
        assert phi.shape == (3, 1), "Expected shape of (3,1) for 'phi', got {} instead".format(phi.shape)
        theta = np.linalg.norm(phi)
        a = phi / theta

        # Check for singularity at theta = 0
        if theta < _LIE_EPSILON:
            # Use first order Taylor's expansion
            return np.eye(3, dtype=np.float32) + 0.5 * SO3.hat(phi)

        sin_theta = np.sin(theta)/theta
        return (
            sin_theta * np.eye(3, dtype=np.float32) + (1 - sin_theta) * np.dot(a, a.T)
            + ((1 - np.cos(theta)) / theta) * SO3.hat(a)
        )
