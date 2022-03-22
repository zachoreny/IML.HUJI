from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma, samples_num = 10, 1, 1000
    normal_var = np.random.normal(mu, sigma, size=samples_num)
    X = UnivariateGaussian()
    X.fit(normal_var)
    print(f"({X.get_mean()},{X.get_var()})")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    estimated_mean_diff = []
    for m in ms:
        gaussian_var = np.random.normal(mu, sigma, size=m)
        estimated_mean_diff.append(np.abs(np.mean(gaussian_var) - mu))
    go.Figure([go.Scatter(x=ms, y=estimated_mean_diff, mode='markers+lines', name='difference per size')],
              layout=go.Layout(title=r"$\text{Distance Between the Estimated and True Value of the Expectation as "
                                     r"Function of the Samples Size}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title=r'$|\mu - \widehat\mu|$',
                               height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
