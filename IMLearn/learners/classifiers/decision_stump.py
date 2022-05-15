from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_loss = np.inf
        for sign, j in product([-1, 1], range(X.shape[1])):
            thr, thr_err = self._find_threshold(X[:, j], y, sign)
            if thr_err < min_loss:
                self.threshold_ = thr
                self.j_ = j
                self.sign = sign
                min_loss = thr_err
        return

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return self.sign * ((X[:, self.j_] <= self.threshold_) * 2 - 1)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        n = len(values)
        # check input
        if n <= 1:
            return 0, 0
        # separate D and the labels from each other
        D = abs(labels)
        # the indices that would sort the values array
        sorted_ind = np.argsort(values)
        values = values[sorted_ind]
        labels = labels[sorted_ind]
        labels = np.sign(labels) + (labels == 0)
        D = D[sorted_ind]
        thr_pred = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
        thr_pred = np.concatenate([[-np.inf], thr_pred, [np.inf]])
        # min_loss represents the probability of the stump's classification
        min_loss = np.sum(D[labels == sign])
        losses = np.append(min_loss, min_loss - np.cumsum(D * (labels * sign)))
        min_loss_id = np.argmin(losses)
        return thr_pred[min_loss_id], losses[min_loss_id]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        self.fit(X, y)
        # D = abs(y)
        # y = np.sign(y) + (y == 0)
        return misclassification_error(X[self.j_], y)
