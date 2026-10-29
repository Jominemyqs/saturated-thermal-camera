"""Isolated noisy upper/lower censoring update; historical samplers unchanged."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .truncated_gaussian import sample_lower_truncated_gaussian_hmc


def spread_nodes(points, eligible, budget=256):
    """Deterministic farthest-point coverage in physical coordinates, no values."""
    xy = np.asarray(points, dtype=float)
    ids = np.flatnonzero(np.asarray(eligible, dtype=bool))
    if budget < 1 or not len(ids):
        raise ValueError('Need a positive budget and a nonempty eligible pool')
    pool = xy[ids]
    extent = float(np.max(np.ptp(pool, axis=0)))
    if extent > 0:
        pool = (pool-pool.min(axis=0))/extent
    center = (pool.min(axis=0) + pool.max(axis=0)) / 2
    index = int(np.argmin(np.round(np.sum((pool-center)**2, axis=1), 14)))
    distance = np.full(len(ids), np.inf)
    chosen = []
    for _ in range(min(budget, len(ids))):
        chosen.append(index)
        distance = np.minimum(distance, np.round(np.sum((pool-pool[index])**2, axis=1), 14))
        distance[chosen] = -np.inf
        index = int(np.argmax(distance))
    return np.sort(ids[chosen])


def likelihood_layout(ids, kinds, numerical_values):
    """0=numerical, +1=lower, -1=upper; flags cannot carry numerical readings."""
    ids = np.asarray(ids, dtype=int)
    kinds = np.asarray(kinds, dtype=int)
    values = np.asarray(numerical_values, dtype=float)
    if ids.ndim != 1 or kinds.shape != ids.shape or values.shape != ids.shape:
        raise ValueError('Observation IDs, types and values must align')
    if len(np.unique(ids)) != len(ids) or not np.isin(kinds, [-1, 0, 1]).all():
        raise ValueError('Duplicate observation or invalid likelihood type')
    if not np.isfinite(values[kinds == 0]).all() or not np.isnan(values[kinds != 0]).all():
        raise ValueError('Only numerical locations may contain readings')
    return kinds == 0, kinds != 0


def sign_transform(mean, covariance, cross_covariance, signs, bounds):
    signs = np.asarray(signs, dtype=float)
    if not np.isin(signs, [-1., 1.]).all():
        raise ValueError('Signs must be +1 or -1')
    return (signs*np.asarray(mean),
            signs[:, None]*np.asarray(covariance)*signs[None, :],
            np.asarray(cross_covariance)*signs[None, :],
            signs*np.broadcast_to(bounds, signs.shape))


def sample_mixed_noisy_blocks(*, ids, kinds, numerical_values, thresholds,
                              prediction_mean, observation_mean,
                              observed_covariance, pred_observed_covariance,
                              prediction_variance, noise_sd, n_samples,
                              burn_in, thin, seed, relative_jitter=1e-7):
    """Sample correlated noisy signals, then return latent marginal predictions.

    Noise enters once in K_ZZ. Mixed inequalities are sampled jointly after
    numerical conditioning and a diagonal sign transform. We invert that
    transform before predicting with the ORIGINAL K_TZ, exactly equivalent to
    transforming K_TZ as well. Prediction residuals retain the historical
    marginal-only representation, not full spatially coherent field draws.
    """
    numeric, constrained = likelihood_layout(ids, kinds, numerical_values)
    y = np.asarray(numerical_values, dtype=float)
    kinds = np.asarray(kinds)
    m = np.asarray(prediction_mean, dtype=float)
    mo = np.asarray(observation_mean, dtype=float)
    koo = np.asarray(observed_covariance, dtype=float)
    kpo = np.asarray(pred_observed_covariance, dtype=float)
    diag = np.asarray(prediction_variance, dtype=float)
    if koo.shape != (len(y), len(y)) or kpo.shape != (len(m), len(y)):
        raise ValueError('Inconsistent covariance blocks')
    if mo.shape != y.shape or diag.shape != m.shape or noise_sd <= 0:
        raise ValueError('Invalid mean, diagonal or noise')
    bounds = np.broadcast_to(thresholds, y.shape)
    rng = np.random.default_rng(seed)
    jitter = relative_jitter * max(float(np.max(diag)), 1.)
    noisy = .5 * (koo + koo.T)
    noisy[np.diag_indices_from(noisy)] += noise_sd**2 + jitter
    ci = np.flatnonzero(constrained)
    ni = np.flatnonzero(numeric)
    if len(ci):
        cm = mo[ci]
        ck = noisy[np.ix_(ci, ci)]
        if len(ni):
            knn = noisy[np.ix_(ni, ni)]
            kcn = noisy[np.ix_(ci, ni)]
            factor = cho_factor(knn, lower=True, check_finite=False)
            cm = cm + kcn.dot(cho_solve(factor, y[numeric]-mo[ni], check_finite=False))
            ck = ck - kcn.dot(cho_solve(factor, kcn.T, check_finite=False))
        signs = kinds[ci].astype(float)
        wm, wk, _, wb = sign_transform(cm, ck, kpo[:, ci], signs, bounds[ci])
        w = sample_lower_truncated_gaussian_hmc(
            wm, wk, wb, n_samples=n_samples, burn_in=burn_in, thin=thin,
            seed=seed+1_000_003, relative_jitter=relative_jitter)
        z = w * signs[None, :]
        observation_samples = np.empty((n_samples, len(y)))
        observation_samples[:, numeric] = y[numeric]
        observation_samples[:, constrained] = z
    else:
        z = np.empty((n_samples, 0))
        observation_samples = np.repeat(y[None, :], n_samples, axis=0)
    factor = cho_factor(noisy, lower=True, check_finite=False)
    weights = cho_solve(factor, (observation_samples-mo[None, :]).T, check_finite=False)
    conditional_means = m[None, :] + kpo.dot(weights).T
    solved = cho_solve(factor, kpo.T, check_finite=False)
    conditional_variance = np.maximum(diag-np.sum(kpo*solved.T, axis=1), 0.)
    mean = np.mean(conditional_means, axis=0)
    variance = conditional_variance + np.var(conditional_means, axis=0, ddof=1)
    draws = conditional_means + rng.normal(size=conditional_means.shape)*np.sqrt(conditional_variance)
    prediction = (mean, np.sqrt(np.maximum(variance, 0.)),
                  np.quantile(draws, .025, axis=0), np.quantile(draws, .975, axis=0), draws)
    return prediction, z
