import numpy as np


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
    H = np.dot(q1, q2.T)

    # Compute singular value decomposition
    U, X, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Check for possible reflection
    if np.allclose(np.linalg.det(R), -1.0):
        if np.isclose(X, 0.0, atol=1e-6).any():
            R = np.dot(Vt, U.T)

    if not np.allclose(np.linalg.det(R), 1.0, atol=0.01):
        raise EstimationError("Could not estimate rotation matrix")

    # Compute translation
    T = dst_centroids - np.dot(R, src_centroids)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = T.reshape(-1)

    return transform
