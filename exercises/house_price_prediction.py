from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from utils import animation_to_gif

pio.templates.default = "simple_white"

FEAT_NAMES = {"waterfront": "Waterfront", "view": "View", "condition": "Condition", "grade": "Grade",
              "price": "Price",
              "sqft_living": "Interior Living Space (sqft)",
              "sqft_above": "Interior Housing Space Above Ground Level (sqft)",
              "sqft_lot": "Land Space (sqft)",
              "yr_built": "Year of Initial Built",
              "sqft_living15": "Interior Living Space for the Nearest 15 Neighbors (sqft)",
              "sqft_lot15": "Land Space of the Nearest 15 Neighbors (sqft)",
              "bathrooms": "Bathrooms", "floors": "Floors",
              "sqft_basement": "Interior Housing Space Below Ground Level (sqft)",
              "yr_renovated": "Year of last renovation"}

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    # eliminate irrelevant columns "home id", "date of home sale" and "longitude/"latitude"
    irrelevant_feats = {"id", "lat", "long", "date"}
    for feat in irrelevant_feats:
        df = df.drop(feat, 1)
    # check feats in a specific range
    range_feats = {"waterfront": [0, 1], "view": range(5), "condition": range(1, 6), "grade": range(1, 14)}
    for feat in range_feats:
        df = df[df[feat].isin(range_feats[feat])]
    # check positive / non-negative feats
    positive_feats = {"price", "sqft_living", "sqft_lot", "sqft_above", "yr_built", "sqft_living15", "sqft_lot15"}
    non_negative_feats = {"bathrooms", "floors", "sqft_basement", "yr_renovated"}
    for feat in positive_feats:
        df = df[df[feat] > 0]
    for feat in non_negative_feats:
        df = df[df[feat] >= 0]
    # zipcode manipulation
    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])

    df.insert(0, 'intercept', 1, True)
    return df.drop("price", 1), df.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feat in X:
        if feat == "zipcode":
            pearson_cor = np.cov(X[feat], y) / (np.std(X[feat]) * np.std(y))
            name = feat
            if feat in FEAT_NAMES:
                name = FEAT_NAMES[feat]
            fig = px.scatter(pd.DataFrame({'x': X[feat], 'y': y}),
                             x="x", y="y", trendline="ols",
                             title=f"Correlation Between {name} and the Response <br>Pearson Correlation: {pearson_cor}",
                             labels={"x": f"{name} Values", "y": "Response Values"})
            fig.write_image("output_path/singular.values.scree.plot.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    final_results = []
    p_values = [p * 0.01 for p in range(10, 101)]
    for p in p_values:
        average_results = []
        for i in range(10):
            # first, copy the data and shuffle it to remain consistent
            p_sample_x = train_x.copy(deep=True)
            p_sample_x["price"] = train_y
            p_sample_x = p_sample_x.sample(frac=p)
            # return to X, y with their original features
            p_sample_y = p_sample_x.price
            p_sample_x = p_sample_x.drop("price", 1)
            # fit linear regression
            linear_reg = LinearRegression()
            linear_reg.fit(p_sample_x, p_sample_y)
            average_results.append(linear_reg.loss(test_x, test_y))
        final_results.append(float(sum(average_results) / 10))

    # fig = px.line(x=p_values, y=final_results)
    # fig.show()

    fig = go.Figure(data=p_values,
                    frames=final_results,
                    layout=go.Layout(
                        updatemenus=[dict(visible=True,
                                          type="buttons",
                                          buttons=[dict(label="Play",
                                                        method="animate",
                                                        args=[None, dict(frame={"duration": 1000})])])]))

    animation_to_gif(fig, f"../poly-deg-diff-samples.gif", 1000)
    fig.show()
