import logging

import numpy as np

from dense_visual_odometry.utils.lie_algebra.common import is_rotation_matrix


logger = logging.getLogger(__name__)


class EstimationError(Exception):
    pass


def find_rigid_body_transform_from_pointclouds_SVD(src_pc: np.ndarray, dst_pc: np.ndarray, weights: np.ndarray = None):
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
    H = np.dot(q1, q2.T) / src_centroids.shape[0]

    # Compute singular value decomposition
    U, X, Vt = np.linalg.svd(H, full_matrices=False)

    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Check for possible reflection
    if np.allclose(np.linalg.det(R), -1.0, atol=1e-3):

        lambdas = np.isclose(X, 0.0, atol=0.005)

        if lambdas.any():
            V_1 = Vt.T.copy()
            V_1[:, lambdas] *= (-1)  # Inverse sign
            R = np.dot(V_1, U.T)
        # R = np.dot(Vt.T * np.array([1, 1, -1]).reshape(3, 1), U.T)

    if not is_rotation_matrix(R):
        raise EstimationError("Could not estimate rotation matrix (determinant: {})".format(np.linalg.det(R)))

    # Compute translation
    T = dst_centroids - np.dot(R, src_centroids)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = T.reshape(-1)

    return transform


def find_rigid_body_transform_from_pointclouds_SVD1(src_pc: np.ndarray, dst_pc: np.ndarray, weights: np.ndarray = None):
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
    H = np.dot(q1, q2.T) / src_centroids.shape[0]
    H_rank = np.linalg.matrix_rank(H)

    # Compute singular value decomposition
    U, _, Vt = np.linalg.svd(H, full_matrices=False)

    if H_rank > 2:

        S = np.eye(3, dtype=np.float32)

        if np.linalg.det(H) < 1e-8:
            S[2, 2] = -1

    elif H_rank == 2:
        U_det = np.linalg.det(U)
        V_det = np.linalg.det(Vt.T)

        if np.isclose(U_det * V_det, 1.0, atol=1e-6):
            S = np.eye(3, dtype=np.float32)

        elif np.isclose(U_det * V_det, 1.0, atol=1e-6):
            S = np.eye(3, dtype=np.float32)
            S[2, 2] = -1

        else:
            raise EstimationError("Covariance rank is 2 but det(U) * det(V) is neither 1.0 nor -1.0")

    # Compute rotation matrix
    R = np.dot(U, np.dot(S, Vt))

    if not is_rotation_matrix(R):
        raise EstimationError("Could not estimate rotation matrix (determinant: {})".format(np.linalg.det(R)))

    # Compute translation
    T = dst_centroids - np.dot(R, src_centroids)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = T.reshape(-1)

    return transform


def find_rigid_body_transform_from_pointclouds_quat(src_pc: np.ndarray, dst_pc: np.ndarray, weights: np.ndarray = None):
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
    H = np.dot(q2, q1.T) / src_pc.shape[1]

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
    t = dst_centroids - np.dot(R, src_centroids)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = t.reshape(-1)

    return transform
