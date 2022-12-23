import logging

import numpy as np

from dense_visual_odometry.utils.lie_algebra.common import is_rotation_matrix


logger = logging.getLogger(__name__)


class EstimationError(Exception):
    pass


def find_rigid_body_transform_from_pointclouds(src_pc: np.ndarray, dst_pc: np.ndarray, weights: np.ndarray = None):
    """Finds optimal rigid body transform given 2 pointcloud of correspondances via an equivalent least squares
    solution.

    Parameters
    ----------
    src_pc : np.ndarray
        Source pointcloud (3xN).
    dst_pc : np.ndarray
        Destination pointcloud (3xN).
    weights : np.ndarray
        Weights of each match (correspondance) with shape (N,)

    Returns
    -------
    np.ndarray :
        3x3 rotation matrix.
    np.ndarray :
        3x1 translation vector.
    np.ndarray :
        3xN residuals points.

    Raises
    ------
    ValueError
        If shapes of inputs do not match.
    EstimationError
        If no valid rotation matrix could be estimated.
    """
    # See https://stackoverflow.com/questions/66923224/rigid-registration-of-two-point-clouds-with-known-correspondence

    if weights is not None:
        if weights.shape != (src_pc.shape[1],):
            raise ValueError("Expected weights shape to be '({},)', got '{}' instead".format(
                src_pc.shape[1], weights.shape
            ))

        src_centroids = ((weights * src_pc).sum(axis=1) / weights.sum()).reshape(-1, 1)
        dst_centroids = ((weights * dst_pc).sum(axis=1) / weights.sum()).reshape(-1, 1)
    else:
        src_centroids = src_pc.mean(axis=1).reshape(-1, 1)
        dst_centroids = dst_pc.mean(axis=1).reshape(-1, 1)

    q1 = src_pc - src_centroids
    q2 = dst_pc - dst_centroids

    # Compute covariance matrix
    H = np.dot(q1, q2.T) / src_centroids.shape[1]

    # Compute singular value decomposition
    U, X, Vt = np.linalg.svd(H, full_matrices=False)

    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Check for possible reflection
    if np.allclose(np.linalg.det(R), -1.0):

        lambdas = np.isclose(X, 0.0, atol=0.005)
        logger.debug("Determinant equals -1, lambdas: {}".format(X.tolist()))

        if lambdas.any():
            V_1 = Vt.T.copy()
            V_1[:, lambdas] *= (-1)  # Inverse sign
            R = np.dot(V_1, U.T)
        # R = np.dot(Vt.T * np.array([1, 1, -1]), U.T)

    if not np.allclose(np.linalg.det(R), 1.0, atol=0.01):
        raise EstimationError("Could not estimate rotation matrix (determinant: {})".format(np.linalg.det(R)))

    # Compute translation
    T = dst_centroids - np.dot(R, src_centroids)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = T.reshape(-1)

    return transform


def find_rigid_body_transform_from_pointclouds_1(src_pc: np.ndarray, dst_pc: np.ndarray, weights: np.ndarray = None):
    """Finds optimal rigid body transform given 2 pointcloud of correspondances via an equivalent least squares
    solution.

    Parameters
    ----------
    src_pc : np.ndarray
        Source pointcloud (3xN).
    dst_pc : np.ndarray
        Destination pointcloud (3xN).
    weights : np.ndarray
        Weights of each match (correspondance) with shape (N,)

    Returns
    -------
    np.ndarray :
        3x3 rotation matrix.
    np.ndarray :
        3x1 translation vector.
    np.ndarray :
        3xN residuals points.

    Raises
    ------
    ValueError
        If shapes of inputs do not match.
    EstimationError
        If no valid rotation matrix could be estimated.
    """
    # See https://stackoverflow.com/questions/66923224/rigid-registration-of-two-point-clouds-with-known-correspondence
    if weights is not None:
        if weights.shape != (src_pc.shape[1],):
            raise ValueError("Expected weights shape to be '({},)', got '{}' instead".format(
                src_pc.shape[1], weights.shape
            ))

        src_centroids = ((weights * src_pc).sum(axis=1) / weights.sum()).reshape(-1, 1)
        dst_centroids = ((weights * dst_pc).sum(axis=1) / weights.sum()).reshape(-1, 1)
    else:
        src_centroids = src_pc.mean(axis=1).reshape(-1, 1)
        dst_centroids = dst_pc.mean(axis=1).reshape(-1, 1)

    q1 = src_pc - src_centroids
    q2 = dst_pc - dst_centroids

    # Compute covariance matrix
    H = np.dot(q1, q2.T) / src_pc.shape[1]

    # Anti-symmetric matrix
    A = H - H.T
    delta = np.array([A[1, 2], A[2, 0], A[0, 1]])
    Q = np.eye(4, dtype=np.float32)
    Q[0, 0] = H.trace()
    Q[1:, 0] = Q[0, 1:] = delta
    Q[1:, 1:] = H + H.T - np.eye(3, dtype=np.float32) * H.trace()

    # Compute eigen decomposition
    lambdas, v = np.linalg.eig(Q)
    max_index = np.argmax(lambdas)
    q = v[:, max_index]

    R = np.array([
        [q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2, 2 * (q[1] * q[2] - q[0] * q[3]), 2 * (q[1] * q[3] + q[0] * q[2])],  # noqa
        [2 * (q[1] * q[2] + q[0] * q[3]), q[0] ** 2 + q[2] ** 2 - q[1] ** 2 - q[3] ** 2, 2 * (q[2] * q[3] - q[0] * q[1])],  # noqa
        [2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), q[0] ** 2 + q[3] ** 2 - q[1] ** 2 - q[2] ** 2],  # noqa
    ], dtype=np.float32)

    if not is_rotation_matrix(R):
        raise EstimationError("Could not estimate rotation matrix (determinant: {})".format(np.linalg.det(R)))

    # Compute translation
    T = dst_centroids - np.dot(R, src_centroids)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = T.reshape(-1)

    return transform
