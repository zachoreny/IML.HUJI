import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import plotly
import sklearn
from sklearn.metrics import auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    vals = []
    weights = []

    def callback(solver: GradientDescent, weights_arr: np.ndarray, val: np.ndarray, grad: np.ndarray, t: int,
                 eta: float, delta: float):
        vals.append(val)
        weights.append(weights_arr)

    return callback, vals, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for i, eta in enumerate(etas):
        fixed_lr = FixedLR(eta)
        callback_f, vals, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(fixed_lr, callback=callback_f)
        gd.fit(L1(init), X=None, y=None)
        plot_descent_path(L1, np.concatenate(weights, axis=0).reshape(len(weights), len(init)),
                          f"Descent Path for settings: L1, eta {eta}").show()
        go.Figure(go.Scatter(x=list(range(len(weights))), y=vals, mode='markers')) \
            .update_layout(title=f'Convergence Rate for settings: L1, eta {eta}').show()
        fixed_lr = FixedLR(eta)
        callback_f, vals, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(fixed_lr, callback=callback_f)
        gd.fit(L2(init), X=None, y=None)
        plot_descent_path(L1, np.concatenate(weights, axis=0).reshape(len(weights), len(init)),
                          f"Descent Path for settings: L2, eta {eta}").show()
        go.Figure(go.Scatter(x=list(range(len(weights))), y=vals, mode='markers')) \
            .update_layout(title=f'Convergence Rate for settings: L2, eta {eta}').show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = plotly.subplots.make_subplots(rows=2, cols=2)
    l1 = L1(init)
    for i, gamma in enumerate(gammas):
        lr = ExponentialLR(base_lr=eta, decay_rate=gamma)
        callback_f, vals, ws = get_gd_state_recorder_callback()
        gd = GradientDescent(learning_rate=lr, callback=callback_f)
        gd.fit(f=l1, X=None, y=None)
        fig.add_trace(go.Scatter(x=list(range(len(vals))), y=vals, mode='markers', name=f'rate: {gamma}'),
                      row=1 + (i % 2), col=1 + (i // 2))
    fig.update_layout(title=f'Convergence Rate for settings: L1, and different decay rates (see key)')

    # Plot algorithm's convergence for the different values of gamma
    fig.show()

    # Plot descent path for gamma=0.95
    lr = ExponentialLR(base_lr=eta, decay_rate=0.95)
    callback_f, vals, weights = get_gd_state_recorder_callback()
    gd = GradientDescent(lr, callback=callback_f)
    gd.fit(L1(init), X=None, y=None)
    plot_descent_path(L1, np.concatenate(weights, axis=0).reshape(len(weights), len(init)),
                      f"Descent Path for settings: L1, eta {eta}, decay rate 0.95").show()
    lr = ExponentialLR(base_lr=eta, decay_rate=0.95)
    callback_f, vals, weights = get_gd_state_recorder_callback()
    gd = GradientDescent(lr, callback=callback_f)
    gd.fit(L2(init), X=None, y=None)
    plot_descent_path(L2, np.concatenate(weights, axis=0).reshape(len(weights), len(init)),
                      f"Descent Path for settings: L2, eta {eta}, decay rate 0.95").show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    train_x, train_y, test_x, test_y = load_data()
    train_x, train_y, test_x, test_y = train_x.to_numpy(), train_y.to_numpy(), test_x.to_numpy(), test_y.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    lr = LogisticRegression()
    lr.fit(X=train_x, y=train_y)
    prob = lr.predict_proba(test_x)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_y, prob)
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="RCA"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="")],
        layout=go.Layout(title=rf"$\text{{ROC Curve, with AUC }}= {auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{FPR}$"),
                         yaxis=dict(title=r"$\text{TPR}$"))).show()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
