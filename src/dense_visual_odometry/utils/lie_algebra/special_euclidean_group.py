# This auxiliary module contains a numpy implementation of some functions for manipulating
# Special Orthogonal and Special Eucledian groups (Lie's algebra)
# It was based on Tim Barfoot's book available on:
# http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf

import numpy as np
import logging

from dense_visual_odometry.utils.lie_algebra.base_special_group import BaseSpecialGroup, _LIE_EPSILON
from dense_visual_odometry.utils.lie_algebra.special_orthogonal_group import SO3
from dense_visual_odometry.utils.lie_algebra.common import wrap_angle


logger = logging.getLogger(__name__)


class SE3(BaseSpecialGroup):

    @staticmethod
    def hat(xi: np.ndarray):
        assert type(xi) == np.ndarray, "'xi' should be a numpy array"
        assert xi.shape == (6, 1), "Expected shape of (6,1) for 'xi', got {} instead".format(xi.shape)

        hat = np.zeros((4, 4), dtype=np.float32)
        hat[:3, :3] = SO3.hat(xi[3:, :])
        hat[:3, 3] = xi[:3, 0]

        return hat

    @staticmethod
    def curly_hat(xi: np.ndarray):
        """
            Returns the adjoint of an element of s3(3)

        Parameters
        ----------
        xi : np.ndarray
            se(3) array

        Returns
        -------
        curly_hat : np.ndarray
            Adjoint matrix (6,6)
        """
        assert type(xi) == np.ndarray, "'xi' should be a numpy array"
        assert xi.shape == (6, 1), "Expected shape of (6,1) for 'xi', got {} instead".format(xi.shape)

        phi_hat = SO3.hat(xi[3:, :])
        rho_hat = SO3.hat(xi[:3, :])

        curly_hat = np.zeros((6, 6), dtype=np.float32)
        curly_hat[:3, :3] = phi_hat
        curly_hat[3:, 3:] = phi_hat
        curly_hat[:3, 3:] = rho_hat

        return curly_hat

    @staticmethod
    def exp(xi: np.ndarray):
        """
        Exponential mapping from se(3) to SE(3), i.e, R6 --> R4x4

        Parameters
        ----------
        xi : np.ndarray
            se(3) array

        Returns
        -------
        T : np.ndarray
            Transformation matrix (4,4) corresponding to the exponential map of 'xi'
        """
        assert type(xi) == np.ndarray, "'xi' should be a numpy array"
        assert xi.shape == (6, 1), "Expected shape of (6,1) for 'xi', got {} instead".format(xi.shape)

        xi_hat = SE3.hat(xi)

        theta = np.linalg.norm(xi[3:, 0])

        # Wrap angle between [-pi,pi)
        theta = wrap_angle(theta)

        T = np.eye(4, dtype=np.float32)

        # Check for singularity at 0 deg rotation
        if theta < _LIE_EPSILON:
            logger.debug("Singular rotation (i.e 'theta' = 0)")
            T[:3, 3] = xi[:3, 0]

        else:
            xi_hat = SE3.hat(xi)
            xi_hat_2 = np.dot(xi_hat, xi_hat)
            theta_2 = theta*theta
            T += (
                xi_hat + ((1.0 - np.cos(theta)) / theta_2) * xi_hat_2
                + ((theta - np.sin(theta)) / (theta_2 * theta)) * np.dot(xi_hat_2, xi_hat)
            )

        return T

    @staticmethod
    def log(T: np.ndarray):
        """
        Logarithmic mapping from SE(3) to se(3), i.e, R4x4 --> R6

        Parameters
        ----------
        T : np.ndarray
            Transformation matrix

        Returns
        -------
        xi : np.ndarray
            se(3) vector corresponding to the logarithmic map of 'T'
        """
        assert type(T) == np.ndarray, "'T' should be a numpy array"
        assert T.shape == (4, 4), "Expected shape (4,4) for 'T', got {} instead".format(T.shape)

        xi = np.zeros((6, 1), dtype=np.float32)
        xi[3:, 0] = SO3.log(T[:3, :3]).reshape(-1)

        theta = np.linalg.norm(xi[3:, 0])

        if np.abs(theta) < _LIE_EPSILON:
            xi[3:, 0] = np.zeros(3, dtype=np.float32)
            xi[:3, 0] = T[:3, 3]
        else:
            a = (xi[3:, 0] / theta).reshape(3, 1)
            a_hat = SO3.hat(a)
            theta_2 = theta / 2
            A = theta_2 * np.cos(theta_2) / np.sin(theta_2)
            V_inv = A * np.eye(3, dtype=np.float32) + (1 - A) * np.dot(a, a.T) - theta_2 * a_hat
            xi[:3, 0] = np.dot(V_inv, T[:3, 3]).reshape(-1)

        return xi

    @staticmethod
    def inverse(T: np.ndarray):
        """
            Returns the inverse of a transformation matrix (4x4)

        Parameters
        ----------
        T : np.ndarray:
            4x4 Transformation matrix

        Returns
        -------
        T : np.ndarray
            4x4 Transformation matrix
        """
        assert type(T) == np.ndarray, "'T' should be a numpy array"
        assert T.shape == (4, 4), "Expected shape (4,4) for 'T', got {} instead".format(T.shape)

        T_inverse = np.eye(4, 4, dtype=np.float32)
        T_inverse[:3, :3] = T[:3, :3].T
        T_inverse[:3, 3] = -np.dot(T[:3, :3].T, T[:3, 3])

        return T_inverse
