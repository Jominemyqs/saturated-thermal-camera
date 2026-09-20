"""Diagnostic one-step importance reweighting; no resampling or tempering."""
from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, logsumexp


def censored_log_likelihood(states, observed, saturated, ceiling, noise_sd):
    if noise_sd <= 0:
        raise ValueError("Positive measurement noise required")
    states = np.asarray(states)
    y = np.asarray(observed)
    sat = np.asarray(saturated, dtype=bool)
    unsat = ~sat
    result = np.sum(-.5*((states[...,unsat]-y[unsat])/noise_sd)**2
                    - np.log(noise_sd*np.sqrt(2*np.pi)), axis=-1)
    result += np.sum(log_ndtr((states[...,sat]-ceiling)/noise_sd),axis=-1)
    return result


def importance_weights(log_weights):
    values = np.asarray(log_weights, dtype=float)
    if not np.isfinite(values).any():
        raise ValueError("All weights are nonfinite")
    weights = np.exp(values-logsumexp(values))
    sorted_weights = np.sort(weights)[::-1]
    positive = weights > 0
    return weights, dict(ess=float(1/np.sum(weights**2)),
        ess_fraction=float(1/(len(weights)*np.sum(weights**2))),
        max_weight=float(weights.max()),
        entropy_fraction=float(-np.sum(weights[positive]*np.log(weights[positive]))/np.log(len(weights))),
        particles_for_95pct=int(np.searchsorted(np.cumsum(sorted_weights),.95)+1))


def weighted_prediction(draws, weights, truth):
    """Weighted empirical CDF metrics, not an unbiased importance CRPS estimator."""
    x = np.asarray(draws)
    w = np.asarray(weights)
    w = w/w.sum()
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)):
        raise ValueError("Nonfinite particles or weights")
    mean = np.sum(w[:,None]*x,axis=0)
    sd = np.sqrt(np.sum(w[:,None]*(x-mean)**2,axis=0))
    order = np.argsort(x,axis=0)
    ordered = np.take_along_axis(x,order,axis=0)
    ws = w[order]
    cw = np.cumsum(ws,axis=0)
    cx = np.cumsum(ws*ordered,axis=0)
    dispersion = np.sum(ws*(ordered*(cw-ws)-(cx-ws*ordered)),axis=0)
    crps = np.sum(w[:,None]*abs(x-np.asarray(truth).ravel()),axis=0)-dispersion
    cols = np.arange(x.shape[1])
    lower = ordered[np.argmax(cw>=.025,axis=0),cols]
    upper = ordered[np.argmax(cw>=.975,axis=0),cols]
    return mean,sd,lower,upper,crps
