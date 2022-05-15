import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
from IMLearn.metrics import loss_functions


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :-1], data[:, -1].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:

        X, y = load_dataset("../datasets/" + f)
        # Fit Perceptron and record loss in each fit iteration
        losses = []

        # retrieve the loss over a training set
        def callback_func(fit: Perceptron, x: np.ndarray, y_i: int):
            errors = fit.loss(X, y)
            losses.append(errors)
            fit.training_loss_ = np.append(fit.training_loss_, errors)

        p = Perceptron(callback=callback_func)
        p.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        fig = px.line(x=np.linspace(start=0, stop=len(losses)-1, num=len(losses)),
                      y=losses,
                      title=n,
                      labels={'x': 'iterations', 'y': 'loss values'})
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)
        y = y.astype(int)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)
        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            rf"$\text{{Gaussian Naive Bayes prediction, accuracy: {loss_functions.accuracy(y, gnb.predict(X))} }}$",
            rf"$\text{{Linear Discriminant Analysis prediction, accuracy: {loss_functions.accuracy(y, lda.predict(X))} }}$"
        ))

        # Add traces for data-points setting symbols and colors
        # GNB
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=gnb.predict(X), symbol=y)), row=1, col=1)
        # LDA
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=lda.predict(X), symbol=y)), row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        # GNB
        fig.add_trace(
            go.Scatter(x=gnb.mu_.transpose()[0], y=gnb.mu_.transpose()[1], mode="markers",
                       showlegend=False,
                       marker=dict(color="black", symbol="x")), row=1, col=1)
        # LDA
        fig.add_trace(
            go.Scatter(x=lda.mu_.transpose()[0], y=lda.mu_.transpose()[1], mode="markers",
                       showlegend=False,
                       marker=dict(color="black", symbol="x")), row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        # GNB
        for i in range(3):
            fig.add_trace(get_ellipse(gnb.mu_[i], np.diag(gnb.vars_[i])), row=1, col=1)
        # LDA
        for i in range(3):
            fig.add_trace(get_ellipse(lda.mu_[i], lda.cov_), row=1, col=2)

        fig.update_layout(
            title_text=rf"$\textbf{{Comparing Gaussian Naive Bayes and LDA on {f}}}$")

        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()