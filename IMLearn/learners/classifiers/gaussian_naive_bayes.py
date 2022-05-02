from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m, d = X.shape
        self.classes_ = np.unique(y)
        k = self.classes_.size
        self.pi_ = np.ndarray(k)
        self.mu_ = np.ndarray((k, d))
        self.vars_ = np.ndarray((k, d))
        for i, y_i in enumerate(self.classes_):
            x_i = X[np.where(y == y_i)]
            self.pi_[i] = x_i.shape[0] / m
            self.mu_[i] = np.mean(x_i, axis=0)
            self.vars_[i] = np.var(x_i, axis=0, ddof=1)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        m, d = X.shape
        prediction = np.ndarray(m)
        for i, sample in enumerate(self.likelihood(X)):
            prediction[i] = np.argmax(sample)
        return prediction

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        m, d = X.shape
        k = self.classes_.size
        likelihoods = []

        for i in range(k):
            v_sum = []
            for j in range(d):
                v_sum.append(- np.log(np.sqrt(2 * np.pi * self.vars_[i][j])) -
                             0.5 * (np.square(X[:, j] - self.mu_[i][j])) / self.vars_[i][j])
            likelihoods.append(np.sum(np.array(v_sum), axis=0) + np.log(self.pi_[i]))

        return np.array(likelihoods).T

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
        return misclassification_error(y, self._predict(X))
