import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    train_errors = [adaboost.partial_loss(train_X, train_y, i + 1) for i in range(n_learners)]
    test_errors = [adaboost.partial_loss(test_X, test_y, i + 1) for i in range(n_learners)]
    # Plot training and test errors as a function of the number of fitted learners
    fig = go.Figure()
    fig.update_layout(title='(1) Training and test errors as a function of the number of fitted learners',
                      xaxis_title='number of learners',
                      yaxis_title='error rate')
    fig.add_trace(go.Scatter(x=np.linspace(start=1, stop=n_learners, num=n_learners),
                             y=train_errors,
                             name="train errors",
                             line_shape='linear'))
    fig.add_trace(go.Scatter(x=np.linspace(start=1, stop=n_learners, num=n_learners),
                             y=test_errors,
                             name="test errors",
                             line_shape='linear'))
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$\textbf{{Decision boundary up to iteration {t}}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda X: adaboost.partial_predict(X, t), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries Of N Learners}}$")
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_idx = np.argmin(test_errors)
    fig = go.Figure()
    fig.update_layout(title=f"(3) Decision surface of best performing ensemble\n"
                            f"Size: {best_idx + 1}, Accuracy: {accuracy(adaboost.partial_predict(test_X, best_idx), test_y)}")
    fig.add_traces(
        [decision_surface(lambda X: adaboost.partial_predict(X, best_idx), lims[0], lims[1], showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                    mode="markers", showlegend=False,
                    marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))])
    fig.show()

    # Question 4: Decision surface with weighted samples
    D = adaboost.D_
    D = (D / np.max(D)) * 5
    fig = go.Figure()
    fig.update_layout(title=f"(4) Decision surface with weighted samples")
    fig.add_traces([decision_surface(lambda X: adaboost.predict(X), lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                    marker=dict(size=D, opacity=0.9, color=train_y, colorscale=class_colors(3)))])
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    # Question 5:
    fit_and_evaluate_adaboost(noise=0.4)
