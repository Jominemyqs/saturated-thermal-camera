import unittest
import numpy as np
from scipy.special import logsumexp
from src.one_step_smoothing import censored_log_likelihood
from src.smoothing_collapse import log_likelihood_terms, marginalized_tempered_weights, exact_gaussian_log_likelihood


class CollapseTests(unittest.TestCase):
    def test_exact_integration(self):
        from scipy.stats import multivariate_normal
        q = np.array([[1., .2], [.2, .7]])
        forecast = np.array([[0., 1.], [2., 3.]])
        y = np.array([.1, .5])
        expected = [multivariate_normal.logpdf(y, mean=m, cov=q+.25*np.eye(2)) for m in forecast]
        np.testing.assert_allclose(exact_gaussian_log_likelihood(forecast, y, q, .5), expected)

    def test_terms_match_original(self):
        states = np.array([[[1., 3.], [2., 4.]], [[0., 2.], [1., 5.]]])
        y = np.array([1., 2.])
        sat = np.array([False, True])
        np.testing.assert_allclose(log_likelihood_terms(states, y, sat, 2., .5).sum(-1),
                                   censored_log_likelihood(states, y, sat, 2., .5))

    def test_tempering_and_exclusion(self):
        logs = np.array([[-1., -10., -3.], [-4., -2., -1.]])
        w = marginalized_tempered_weights(logs, 0, exclude=1)
        np.testing.assert_allclose(w, [.5, 0, .5])
        expected = np.exp(logsumexp(.5 * logs, axis=0))
        np.testing.assert_allclose(marginalized_tempered_weights(logs, .5), expected / expected.sum())
