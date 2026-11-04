"""Conditional-residual correlation audit; no changes to production inference."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import dgemm

from .censored_gp import kernel_diagonal, rbf_covariance


def product(left, right):
    """Explicit SciPy BLAS, matching the native drivers' covariance products."""
    result = dgemm(alpha=1., a=np.asfortranarray(left), b=np.asfortranarray(right))
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite covariance product')
    return result


def residual_covariance(observations, config):
    """Use exactly the fast RBF sampler's noisy observation covariance."""
    xo, xp = observations['x_obs'], observations['x_pred']
    koo = rbf_covariance(xo, xo, config)
    jitter = config.relative_jitter * config.signal_sd**2
    koo[np.diag_indices_from(koo)] += config.noise_sd**2 + jitter
    kpo = rbf_covariance(xp, xo, config)
    solved = cho_solve(cho_factor(koo, lower=True, check_finite=False),
                       kpo.T, check_finite=False)
    diagonal = np.maximum(kernel_diagonal(xp, config) -
                          np.sum(kpo * solved.T, axis=1), 0.)
    covariance = rbf_covariance(xp, xp, config) - product(kpo, solved)
    covariance = .5 * (covariance + covariance.T)
    np.testing.assert_allclose(np.diag(covariance), diagonal, atol=1e-12, rtol=0)
    np.fill_diagonal(covariance, diagonal)
    return covariance


def normal_chains(seed, chains, samples, dimension):
    """HMC has a separate RNG; the fast sampler's RNG only draws residuals."""
    return np.concatenate([np.random.default_rng(seed + 10000 * chain).normal(
        size=(samples, dimension)) for chain in range(chains)])


def recover_conditional_means(draws, covariance, *, seed, chains, samples,
                              truncated_sampler):
    """Undo saved HMC-path independent residuals without resampling inequalities.

    This is specific to sample_censored_ess_fast's HMC RNG contract, NOT its
    elliptical-slice path. The driver independently replays a legacy chain.
    """
    if truncated_sampler != 'hmc':
        raise ValueError('Residual recovery requires the audited HMC RNG path')
    draws = np.asarray(draws)
    if draws.shape != (chains * samples, len(covariance)):
        raise ValueError('Cache shape disagrees with saved chain settings')
    z = normal_chains(seed, chains, samples, draws.shape[1])
    residual = z * np.sqrt(np.diag(covariance))
    means = draws - residual
    np.testing.assert_allclose(means + residual, draws, atol=1e-12, rtol=0)
    return means, z


def psd_factor(covariance):
    """No added variance: clip only roundoff-sized negative eigenvalues."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1e-10 * max(1., float(np.max(np.diag(covariance))))
    if eigenvalues.min() < -tolerance:
        raise ValueError('Conditional covariance is not numerically PSD')
    factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.))
    difference = float(np.max(abs(product(factor, factor.T) - covariance)))
    if difference > tolerance:
        raise ValueError('PSD factor changes covariance beyond roundoff tolerance')
    return factor, dict(min_eigenvalue=float(eigenvalues.min()),
                       clipped_eigenvalues=int(np.sum(eigenvalues < 0)),
                       factor_max_error_K2=difference, added_jitter_K2=0.)


def propagated_variances(operator, means, covariance):
    """Law of total covariance, integrating conditional residuals analytically.

    The distribution of conditional means is the fixed empirical cached sample.
    Only the residual term differs between marginal and joint representations.
    """
    centered = means - means.mean(axis=0)
    shared = np.var(product(centered, operator.T), axis=0, ddof=1)
    independent = np.sum(operator**2 * np.diag(covariance), axis=1)
    joint = np.sum(product(operator, covariance) * operator, axis=1)
    if min(independent.min(), joint.min()) < -1e-10:
        raise ValueError('Negative propagated variance')
    return shared, independent, joint
