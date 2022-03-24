from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma, samples_num = 10, 1, 1000
    X = np.random.normal(mu, sigma, size=samples_num)
    var = UnivariateGaussian()
    var = var.fit(X)
    print(f"({var.get_mean()},{var.get_var()})")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    estimated_mean_diff = []
    for m in ms:
        samples = X[:m]
        var.fit(samples)
        estimated_mean_diff.append(np.abs(var.get_mean() - mu))
    go.Figure([go.Scatter(x=ms, y=estimated_mean_diff, mode='markers+lines', name='difference per size')],
              layout=go.Layout(title=r"$\textbf{Distance Between the Estimated and True Value of the Expectation as "
                                     r"Function of the Samples Size}$",
                               xaxis_title='$m\\text{ - number of samples}$',
                               yaxis_title=r'$|\mu - \widehat\mu|$',
                               height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    X.sort()
    go.Figure([go.Scatter(x=X, y=var.pdf(X), mode='markers+lines', name='difference per size')],
              layout=go.Layout(title=r"$\textbf{Distance Between the Estimated and True Value of the Expectation as "
                                     r"Function of the Samples Size}$",
                               xaxis_title='$m\\text{ - number of samples}$',
                               yaxis_title=r'$|\mu - \widehat\mu|$',
                               height=500)).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples_num = 1000
    X = np.random.multivariate_normal(mu, sigma, samples_num)
    var = MultivariateGaussian()
    var = var.fit(X)
    print(f"{var.get_mean()}")
    print(f"{var.get_cov()}")

    # Question 5 - Likelihood evaluation
    samples = np.linspace(-10, 10, 200)
    pairs = (np.array([np.repeat(samples, len(samples)), np.tile(samples, len(samples))])).T

    def _create_log_likelihood(pair):
        mu = np.array([pair[0], 0, pair[1], 0], dtype=float).T
        return MultivariateGaussian.log_likelihood(mu, sigma, X)

    # log likelihoods:
    ll = np.array(list(map(_create_log_likelihood, pairs)))
    ll_resized = ll.reshape(len(samples), len(samples))

    fig = go.Figure(go.Heatmap(x=samples, y=samples, z=ll_resized),
                    layout=go.Layout(title="Log Likelihood as Function of Two Samples f1, f3",
                                     height=500, width=500))
    fig.update_xaxes(title_text="f1")
    fig.update_yaxes(title_text="f3")
    fig.show()

    # Question 6 - Maximum likelihood
    maximum_likelihood = np.max(ll)
    max_index = np.argmax(ll)
    f1, f3 = pairs[max_index]
    print(f"The maximum log-likelihood value is {maximum_likelihood}, and the matching pair is ({f1}, {f3})")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
