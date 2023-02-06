import numpy as np
import numba as nb

from dense_visual_odometry.weighter.base_weighter import BaseWeighter


class TDistributionWeighter(BaseWeighter):
    """Weights residuals supposing they follow a t-distribution
    """

    def __init__(self, dof: int = 5, initial_sigma: int = 5.0, tolerance: float = 1e-3, max_iterations: int = 50):
        self._dof = dof
        self._tolerance = tolerance
        self._max_iter = max_iterations
        self._init_sigma = initial_sigma
        self._init_lambda = 1.0 / (self._init_sigma ** 2)

        # Compile compute_scale
        self._compute_scale(np.zeros((3, 1), dtype=np.float32), self._dof, self._init_lambda)

    def weight(self, residuals_squared: np.ndarray):

        last_lambda = self._init_lambda
        for _ in range(self._max_iter):

            sigma_2 = self._compute_scale(residuals_squared, self._dof, last_lambda)

            curr_lambda = 1 / sigma_2
            if abs(curr_lambda - last_lambda) < self._tolerance:
                break

            last_lambda = curr_lambda

        return (self._dof + 1) / (self._dof + residuals_squared * curr_lambda)

    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def _compute_scale(residuals_squared, dof, last_lambda):
        N = residuals_squared.shape[0]

        sigma_2 = 0
        for i in nb.prange(N):
            sigma_2 += np.mean(
                residuals_squared[i] * ((dof + 1) / (dof + residuals_squared[i] * last_lambda))
            )

        return sigma_2
