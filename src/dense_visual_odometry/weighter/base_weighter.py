import numpy as np
import abc


class BaseWeighter(abc.ABC):
    """
        Performs some weighting function
    """

    def weight(residuals: np.ndarray):
        """
        Returns
        -------
        weights : np.ndarray
            Array of the same shape as `residuals` with pixelwise weights
        """
        raise NotImplementedError()
