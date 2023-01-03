# This auxiliary module contains a numpy implementation of some functions for manipulating
# Special Orthogonal and Special Eucledian groups (Lie's algebra)
# It was based on Tim Barfoot's book available on:
# http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf
import math

import numpy as np

from dense_visual_odometry.utils.lie_algebra.base_special_group import BaseSpecialGroup, _LIE_EPSILON
from dense_visual_odometry.utils.lie_algebra.common import is_rotation_matrix, wrap_angle, quat_mult


class So3(BaseSpecialGroup):

    _QUAT_REPR_VALUES = ["wxyz", "xyzw"]

    def __init__(self, rot: np.ndarray, quat_repr: str = "wxyz"):
        assert isinstance(rot, np.ndarray), "Expected 'rot' to be a numpy array, got {} instead".format(type(rot))
        assert quat_repr in self._QUAT_REPR_VALUES, "Invalid 'quat_repr': '{}'. Options are: {}".format(
            quat_repr, self._QUAT_REPR_VALUES
        )

        # Rotation is internally represented as a quaternion
        self._quat = None
        self._matrix = None
        self._phi = None
        self._hat = None
        self._quat_repr = quat_repr

        if rot.shape == (4, 1):
            self._quat = rot.copy()

        elif rot.shape == (3, 1):

            theta = np.linalg.norm(rot)

            if theta < _LIE_EPSILON:
                self._phi = np.zeros((3, 1), dtype=np.float32)

                if self._quat_repr == "xyzw":
                    self._quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).reshape(4, 1)

                else:
                    self._quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(4, 1)

            else:
                a = rot / theta

                theta = wrap_angle(theta)

                self._phi = theta * a
                self._quat = self._se3_to_quat(phi=self._phi, quat_repr=self._quat_repr)

        elif rot.shape == (3, 3):
            assert is_rotation_matrix(rot), "Got invalid rotation matrix '{}'".format(rot.tolist())

            self._matrix = rot.copy()
            self._quat = self._SE3_to_quat(R=self._matrix, quat_repr=self._quat_repr)

        else:
            raise ValueError("Expected 'rot' to have shape (4, 1), (3, 1) or (3, 3), got '{}' instead".format(
                rot.shape
            ))

    @staticmethod
    def _se3_to_quat(phi: np.ndarray, quat_repr: str = "wxyz"):
        assert phi.shape == (3, 1), "Expected 'phi' shape to be (3, 1), got '{}' instead".format(phi.shape)

        theta = np.linalg.norm(phi)

        a = phi / theta
        sin_theta2 = math.sin(theta / 2)
        x, y, z = a.flatten()

        if quat_repr == "wxyz":
            quat = np.array([
                math.cos(theta / 2), sin_theta2 * x, sin_theta2 * y, sin_theta2 * z
            ], dtype=np.float32).reshape(4, 1)

        elif quat_repr == "xyzw":
            quat = np.array([
                sin_theta2 * x, sin_theta2 * y, sin_theta2 * z, math.cos(theta / 2)
            ], dtype=np.float32).reshape(4, 1)

        return quat

    @staticmethod
    def _SE3_to_quat(R: np.ndarray, quat_repr: str = "wxyz"):
        assert R.shape == (3, 3), "Expected 'R' shape to be (3, 3), got '{}' instead".format(R.shape)

        quat = np.zeros(4)
        t = R[0, 0] + R[1, 1] + R[2, 2]

        if t > 0:
            # case 1
            t = np.sqrt(1 + t)
            quat[0] = 0.5 * t
            t = 0.5 / t
            quat[1] = (R[2, 1] - R[1, 2]) * t
            quat[2] = (R[0, 2] - R[2, 0]) * t
            quat[3] = (R[1, 0] - R[0, 1]) * t

        else:
            i = 0
            if R[1, 1] > R[0, 0]:
                i = 1

            if R[2, 2] > R[i, i]:
                i = 2

            j = (i + 1) % 3
            k = (j + 1) % 3
            t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1)

            quat[1+i] = 0.5 * t

            t = 0.5 / t

            quat[0] = (R[k, j] - R[j, k]) * t
            quat[1+j] = (R[j, i] + R[i, j]) * t
            quat[1+k] = (R[k, i] + R[i, k]) * t

        quat = quat.reshape(4, 1)

        if quat_repr == "xyzw":
            quat = np.roll(quat, 1, axis=0)

        return quat

    @classmethod
    def from_se3(cls, phi: np.ndarray):
        assert phi.shape == (3, 1), "Expected 'phi' shape to be (3, 1), got '{}' instead".format(phi.shape)

        quat = cls._se3_to_quat(phi=phi)
        return cls(quat)

    @classmethod
    def from_SE3(cls, R: np.ndarray):
        assert R.shape == (3, 3), "Expected 'R' shape to be (3, 3), got '{}' instead".format(R.shape)

        quat = cls._SE3_to_quat(R=R)
        return cls(quat)

    def hat(self):
        if self._hat is None:
            phi = self.log()

            self._hat = np.zeros((3, 3), dtype=np.float32)
            self._hat[0, 1] = -phi[2]
            self._hat[0, 2] = phi[1]
            self._hat[1, 0] = phi[2]
            self._hat[1, 2] = -phi[0]
            self._hat[2, 0] = -phi[1]
            self._hat[2, 1] = phi[0]

        return self._hat

    def exp(self):
        if self._matrix is None:
            if self._quat_repr == "wxyz":
                w, x, y, z = self._quat.flatten()

            else:
                x, y, z, w = self._quat.flatten()
                print("qx: {}, qy: {}, qz: {}, qw: {}".format(x, y, z, w))

            # First row of the rotation matrix
            r00 = 2 * (w * w + x * x) - 1
            r01 = 2 * (x * y - w * z)
            r02 = 2 * (x * z + w * y)

            # Second row of the rotation matrix
            r10 = 2 * (x * y + w * z)
            r11 = 2 * (w * w + y * y) - 1
            r12 = 2 * (y * z - w * x)

            # Third row of the rotation matrix
            r20 = 2 * (x * z - w * y)
            r21 = 2 * (y * z + w * x)
            r22 = 2 * (w * w + z * z) - 1

            self._matrix = np.array([
                [r00, r01, r02],
                [r10, r11, r12],
                [r20, r21, r22]
            ], dtype=np.float32)

        return self._matrix

    def log(self):
        if self._phi is None:
            if self._quat_repr == "wxyz":
                w = self._quat[0, 0]
                vec = self._quat[1:, 0]

            else:
                w = self._quat[3, 0]
                vec = self._quat[:3, 0]

            n = np.linalg.norm(vec)

            if n < _LIE_EPSILON:
                self._phi = np.zeros((3, 1), dtype=np.float32)

            else:
                theta = wrap_angle((2 * math.atan2(n, w) / n))
                self._phi = theta * vec.reshape(3, 1)

        return self._phi

    @property
    def quat(self):
        return self._quat

    @property
    def quat_repr(self):
        return self._quat_repr

    @property
    def theta(self):
        return np.linalg.norm(self.log())

    def __mul__(self, right):
        assert isinstance(right, So3), "Got invalid type '{}', expected So3".format(type(right))

        quat_a = self._quat if self._quat_repr == "wxyz" else np.roll(self._quat, 1, axis=0)
        quat_b = right.quat if right.quat_repr == "wxyz" else np.roll(right.quat, 1, axis=0)

        return So3(quat_mult(quat_a, quat_b), quat_repr="wxyz")

    def inverse(self):
        return So3(self.exp().T.copy())

    def copy(self):
        return So3(self.quat.copy())

    @classmethod
    def identity(cls):
        return cls(np.zeros((3, 1), dtype=np.float32))

    def __eq__(self, other) -> bool:
        assert isinstance(other, So3), "Expected 'other' to be of type 'So3', got '{}' instead".format(type(other))

        return np.allclose(self.log(), other.log(), atol=_LIE_EPSILON)
