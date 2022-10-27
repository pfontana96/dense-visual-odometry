from typing import Callable

import numpy as np


class RANSAC:

    def __init__(self, model: Callable, loss: Callable, metric: Callable, dof: int):
        """_summary_

        Parameters
        ----------
        model : Callable
            _description_
        loss : Callable
            _description_
        metric : Callable
            _description_
        dof : int
            _description_
        """

        self._model = model
        self._loss = loss
        self._metric = metric
        self._dof = dof

    def __call__(self, x: np.ndarray, y: np.ndarray, max_iter: int, min_count: int, threshold: float, **kwargs):
        """_summary_

        Parameters
        ----------
        x : np.ndarray
            _description_
        y : np.ndarray
            _description_
        max_iter : int
            _description_
        min_count : int
            _description_
        threshold : float
            _description_

        Returns
        -------
        _type_
            _description_
        """
        assert x.shape == y.shape, "Expected 'x' and 'y' to have the same shape, got '{}' and '{}'".format(
            x.shape, y.shape
        )

        N = x.shape[1]

        best_model = None
        best_consensus = []
        best_metric = np.inf

        current_set = np.random.randint(0, N, size=self._dof)
        for _ in range(max_iter):
            model = self._model(x[:, current_set], y[:, current_set], **kwargs)
            losses = self._loss(x, y, model)
            consensus_set = np.where(losses <= threshold)[1]
            if len(consensus_set) > min_count:
                metric = self._metric(losses)

                if (
                    (len(consensus_set) > len(best_consensus))
                    or ((len(consensus_set) == len(best_consensus)) and metric < best_metric)
                ):
                    best_model = model
                    best_metric = metric
                    best_consensus = consensus_set

                current_set = consensus_set

            else:
                current_set = np.random.randint(0, N, size=self._dof)

        return best_model, best_consensus, best_metric
