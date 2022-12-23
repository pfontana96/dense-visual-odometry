from typing import Callable, Tuple
import logging
import math

import numpy as np
from numpy.random import default_rng


logger = logging.getLogger(__name__)


class RANSAC:

    def __init__(self, model: Callable, loss: Callable, metric: Callable, dof: int):
        """
        Parameters
        ----------
        model : Callable
            Function to call, must return model parameters as the form of a numpy array.
        loss : Callable
            Function that compute losses, receives two numpy arrays `x` and `y` of shape (mxn) and returns an array
            of shape (n,) with the loss for each point.
        metric : Callable
            Function that computes metrics, takes as input the losses (computed by `loss`).
        dof : int
            Degrees-of-freedom or minimum number of points for the estimation of the model by `model`.
        """

        self._model = model
        self._loss = loss
        self._metric = metric
        self._dof = dof

        self._rng = default_rng()

    def __call__(
        self, x: np.ndarray, y: np.ndarray, max_iter: int, min_count: int, threshold: float,
        weights: np.ndarray = None, **model_kwargs
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Perform Random Sample Consensus

        Parameters
        ----------
        x : np.ndarray
            mxn data array where n is the number of points and m is the data dimension.
        y : np.ndarray
            mxn measured values for `x` array where n is the number of points and m is the data dimension.
        max_iter : int
            Maximum number of iteration.
        min_count : int
            Minimum number of inliers needed for considering the model valid
        threshold : float
            Loss threshold to consider a point an inlier or not.

        Returns
        -------
        np.ndarray :
            Parameters of best fitted model.
        np.ndarray :
            Indices of best consensus set (inliers).
        float :
            Metric for best fitted model on consensus set.
        """
        assert x.shape == y.shape, "Expected 'x' and 'y' to have the same shape, got '{}' and '{}'".format(
            x.shape, y.shape
        )

        N = x.shape[1]

        best_model = None
        best_consensus = np.array([])
        best_metric = np.inf

        ids = self._rng.permutation(N)
        current_set = ids[:self._dof]
        for _ in range(max_iter):

            try:
                current_set_weights = weights[current_set] if weights is not None else None
                model = self._model(x[:, current_set], y[:, current_set], weights=current_set_weights)

            except Exception as e:
                logger.debug("Model estimation failed with '{}'. Skipping..".format(e))

                ids = self._rng.permutation(N)
                current_set = ids[:self._dof]
                continue

            validate_set = ids[np.in1d(ids, current_set, assume_unique=True, invert=True)]
            losses = self._loss(x[:, validate_set], y[:, validate_set], model)
            consensus_set = validate_set[(losses <= threshold).flatten()]

            if len(consensus_set) > min_count:

                losses = self._loss(x[:, consensus_set], y[:, consensus_set], model)
                metric = self._metric(losses)

                if (
                    (len(consensus_set) > len(best_consensus))
                    or ((len(consensus_set) == len(best_consensus)) and metric < best_metric)
                ):
                    best_model = model
                    best_metric = metric
                    best_consensus = consensus_set.copy()

                    current_set = np.concatenate((current_set, consensus_set))
                    continue

            ids = self._rng.permutation(N)
            current_set = ids[:self._dof]

        # Estimate one last model using all best inliers (in case last was best model)
        if len(best_consensus) > 0:
            current_set_weights = weights[best_consensus] if weights is not None else None
            maybe_best_model = self._model(x[:, best_consensus], y[:, best_consensus], weights=current_set_weights)
            validate_set = ids[np.in1d(ids, best_consensus, assume_unique=True, invert=True)]
            losses = self._loss(x[:, validate_set], y[:, validate_set], maybe_best_model)
            maybe_best_metric = self._metric(losses)

            if maybe_best_metric < best_metric:
                best_model = maybe_best_model
                best_metric = maybe_best_metric
                best_consensus = validate_set[(losses <= threshold).flatten()]

        logger.debug("Best consensus size: {} (input: {})".format(len(best_consensus), x.shape[1]))
        return best_model, best_consensus, best_metric

    @staticmethod
    def max_samples_by_conf(n_inl: int, num_tc: int, sample_size: int, conf: float) -> float:
        """Formula to update max_iter in order to stop iterations earlier
        https://en.wikipedia.org/wiki/Random_sample_consensus."""
        if n_inl == num_tc:
            return 1.0
        return math.log(1.0 - conf) / math.log(1.0 - math.pow(n_inl / num_tc, sample_size))
