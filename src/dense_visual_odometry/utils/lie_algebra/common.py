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
        np.isclose(np.dot(matrix, matrix.T), np.eye(3), atol=1e-6).all()
        and np.isclose(np.dot(matrix.T, matrix), np.eye(3), atol=1e-6).all()
    )
    second_condition = np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6)

    return bool(first_condition and second_condition)


def wrap_angle(angle: np.ndarray):
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
