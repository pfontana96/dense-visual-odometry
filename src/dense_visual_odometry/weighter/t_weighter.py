import numpy as np

from dense_visual_odometry.weighter.base_weighter import BaseWeighter


class TDistributionWeighter(BaseWeighter):
    """Weights residuals supposing they follow a t-distribution
    """

    def __init__(self, dof: int = 5.0, initial_sigma: int = 5.0, tolerance: float = 1e-3, max_iterations: int = 50):
        self._dof = dof
        self._tolerance = tolerance
        self._max_iter = max_iterations
        self._init_sigma = initial_sigma
        self._init_lambda = 1.0 / (self._init_sigma ** 2)

    def weight(self, residuals: np.ndarray):

        residuals_squared = residuals ** 2
        last_lambda = self._init_lambda
        for _ in range(self._max_iter):
            sigma_2 = np.mean(residuals_squared * ((self._dof + 1) / (self._dof + residuals_squared * last_lambda)))

            curr_lambda = 1 / sigma_2
            if abs(curr_lambda - last_lambda) < self._tolerance:
                break

        return (self._dof + 1) / (self._dof + residuals_squared * curr_lambda)
