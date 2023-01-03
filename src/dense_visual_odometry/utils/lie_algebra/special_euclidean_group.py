# This auxiliary module contains a numpy implementation of some functions for manipulating
# Special Orthogonal and Special Eucledian groups (Lie's algebra)
# It was based on Tim Barfoot's book available on:
# http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf
import math

import numpy as np

from dense_visual_odometry.utils.lie_algebra.base_special_group import BaseSpecialGroup, _LIE_EPSILON
from dense_visual_odometry.utils.lie_algebra.special_orthogonal_group import So3


class Se3(BaseSpecialGroup):

    def __init__(self, so3: So3, tvec: np.ndarray):
        assert isinstance(so3, So3), "Expected 'so3' to be of type 'So3', got '{}' instead".format(type(so3))
        assert tvec.shape == (3, 1), "Expected 'tvec' to have shape '(3, 1)', got '{}' instead".format(tvec.shape)

        self._so3 = so3
        self._tvec = tvec.copy()

        self._matrix = None
        self._xi = None
        self._hat = None

    def hat(self):

        if self._hat is None:
            self._hat = np.zeros((4, 4), dtype=np.float32)
            self._hat[:3, :3] = self._so3.hat()
            self._hat[:3, 3] = self.log()[:3, 0]

        return self._hat

    def exp(self):
        if self._matrix is None:

            T = np.eye(4, dtype=np.float32)

            phi = self._so3.log()
            theta = np.linalg.norm(phi)

            if abs(theta) < _LIE_EPSILON:
                T[:3, 3] = self._tvec.flatten()

            else:
                T[:3, :3] = self._so3.exp()
                T[:3, 3] = self._tvec.flatten()

            self._matrix = T

        return self._matrix

    def log(self):

        if self._xi is None:

            xi = np.zeros((6, 1), dtype=np.float32)

            phi = self._so3.log()
            theta = np.linalg.norm(phi)

            if abs(theta) < _LIE_EPSILON:
                xi[:3, 0] = self._tvec.flatten()

            else:
                a = So3(phi.reshape(3, 1) / theta)
                a_hat = a.hat()
                theta_2 = theta / 2
                A = theta_2 * np.cos(theta_2) / np.sin(theta_2)
                V_inv = A * np.eye(3, dtype=np.float32) + (1 - A) * np.dot(a.log(), a.log().T) - theta_2 * a_hat
                xi[:3, 0] = np.dot(V_inv, self._tvec).flatten()
                xi[3:, 0] = self._so3.log().flatten()

            self._xi = xi

        return self._xi

    def inverse(self):
        so3_inv = self._so3.inverse()
        return Se3(so3_inv, -np.dot(so3_inv.exp(), self._tvec))

    @property
    def so3(self):
        return self._so3

    @property
    def tvec(self):
        return self._tvec

    def __mul__(self, right):
        assert isinstance(right, Se3), "Expected 'right' to be of type 'Se3', got '{}' instead.".format(type(right))
        so3 = self._so3 * right.so3
        tvec = self._tvec + np.dot(self._so3.exp(), right.tvec)

        return Se3(so3, tvec)

    @classmethod
    def identity(cls):
        return cls(So3.identity(), np.zeros((3, 1), dtype=np.float32))

    def copy(self):
        return Se3(self.so3.copy(), self.tvec.copy())

    @classmethod
    def from_se3(cls, xi: np.ndarray):
        assert xi.shape == (6, 1), "Expected 'xi' to have shape '(6, 1)', got '{}' instead".format(xi.shape)

        upsilon = xi[:3]
        so3 = So3(xi[3:])
        phi_hat = so3.hat()
        phi_hat_sq = np.dot(phi_hat, phi_hat)
        theta = so3.theta

        if theta < _LIE_EPSILON:
            return cls(So3.identity(), upsilon)

        V = (
            np.eye(3, dtype=np.float32) + ((1 - math.cos(theta)) / (theta**2)) * phi_hat +
            ((theta - math.sin(theta)) / (theta ** 3)) * phi_hat_sq
        )

        return cls(So3(xi[3:]), np.dot(V, upsilon))

    def __eq__(self, other) -> bool:
        assert isinstance(other, Se3), "Expected 'other' to be of type 'Se3', got '{}' instead".format(type(other))

        return np.allclose(self.log(), other.log(), atol=_LIE_EPSILON)
