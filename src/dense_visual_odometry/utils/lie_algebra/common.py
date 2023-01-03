import numpy as np


def is_rotation_matrix(matrix: np.ndarray) -> bool:
    """
        Checks whether a matrix is a valid rotation matrix based on the criteria 'M is a rotational matrix
        if and only if M is orthogonal' which implies: M * M.T = M.T * M = I and det(M) = 1

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to test

    Returns
    -------
    valid_rotation_matrix : bool
        True if 'matrix' is a valid rotation matrix, False otherwise
    """
    assert matrix.shape == (3, 3), f"Expected matrix shape to be (3,3), got {matrix.shape} instead"

    first_condition = (
        np.allclose(np.dot(matrix, matrix.T), np.eye(3), atol=1e-3)
        and np.allclose(np.dot(matrix.T, matrix), np.eye(3), atol=1e-3)
    )
    second_condition = np.isclose(np.linalg.det(matrix), 1.0, atol=1e-3)

    return bool(first_condition and second_condition)


def wrap_angle(angle: float) -> float:
    """
        Wraps angles between [-np.pi; np.pi)

    Arguments
    ---------
    angle : np.ndarray
        Angle (or array of angles) to be wrapped

    Returns
    -------
    wrapped_angle : np.ndarray
        Angle (or array of angles) wrapped to [-np.pi; np.pi)
    """
    return (angle + np.pi) % (2*np.pi) - np.pi


def squared_norm(x: np.ndarray) -> float:
    return np.sum(x ** 2)


def quat_mult(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    """Quaternion multiplication

    Parameters
    ----------
    quat_a : np.ndarray
        4x1 quaternion in format `[qw, qx, qy, qz].T`
    quat_b : np.ndarray
        4x1 quaternion in format `[qw, qx, qy, qz].T`

    Returns
    -------
    quat : np.ndarray
        4x1 quaternion in format `[qw, qx, qy, qz].T`
    """
    assert quat_a.shape == quat_b.shape == (4, 1), "Both quaternions should have shape (4, 1) got '{}' and '{}".format(
        quat_a.shape, quat_b.shape
    )
    w = quat_a[0] * quat_b[0] - np.dot(quat_a[1:].T, quat_b[1:]).flatten()
    vec = quat_a[0, 0] * quat_b[1:, 0] + quat_b[0, 0] * quat_a[1:, 0] + np.cross(quat_a[1:, 0], quat_b[1:, 0])
    result = np.concatenate((w, vec)).astype(np.float32).reshape(4, 1)

    return result
