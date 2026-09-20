"""Diagnostics for a frozen one-step importance proposal, not an SMC filter."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_ndtr, logsumexp


def log_likelihood_terms(states, observed, saturated, ceiling, noise_sd):
    if noise_sd <= 0:
        raise ValueError("Positive observation noise required")
    terms = -.5 * ((states - observed) / noise_sd) ** 2
    terms -= np.log(noise_sd * np.sqrt(2 * np.pi))
    terms[..., saturated] = log_ndtr((states[..., saturated] - ceiling) / noise_sd)
    return terms


def marginalized_tempered_weights(loglik, power, exclude=None):
    """Integrate powered path likelihoods; never power an averaged likelihood."""
    if not 0 <= power <= 1:
        raise ValueError("Likelihood power must be in [0, 1]")
    logs = logsumexp(power * loglik, axis=0) - np.log(len(loglik))
    if exclude is not None:
        logs[exclude] = -np.inf
    return np.exp(logs - logsumexp(logs))


def weight_summary(weights):
    return dict(ess=float(1 / np.sum(weights ** 2)),
                max_weight=float(weights.max()))


def target_summary(draws, weights, top, truth):
    mean = np.sum(weights[:, None] * draws, axis=0)
    values = draws[:, top]
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    cumulative = np.cumsum(weights[order], axis=0)
    columns = np.arange(len(top))
    lo = sorted_values[np.argmax(cumulative >= .025, axis=0), columns]
    hi = sorted_values[np.argmax(cumulative >= .975, axis=0), columns]
    error = mean[top] - truth[top]
    return mean, dict(top1_rmse_K=float(np.sqrt(np.mean(error ** 2))),
                      top1_bias_K=float(error.mean()),
                      top1_width_K=float(np.mean(hi - lo)),
                      top1_coverage=float(np.mean((lo <= truth[top]) & (truth[top] <= hi))))


def exact_gaussian_log_likelihood(forecast, observed, innovation_covariance, noise_sd):
    """Integrate Gaussian process innovation exactly for an unsaturated subset."""
    covariance = innovation_covariance + noise_sd**2 * np.eye(len(observed))
    factor = cho_factor(covariance, lower=True)
    error = (forecast - observed).T
    quadratic = np.sum(error * cho_solve(factor, error), axis=0)
    logdet = 2 * np.log(np.diag(factor[0])).sum()
    return -.5 * (quadratic + logdet + len(observed)*np.log(2*np.pi))
