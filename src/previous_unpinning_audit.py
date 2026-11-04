"""All-camera predecessor conditioning, with analytic correlated residuals.

An isolated audit helper, not a replacement for the production hybrid sampler.
Observation jitter is inherited; no latent covariance jitter is introduced.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import issparse

from .censored_gp import rbf_covariance
from .previous_correlation_audit import product


def noisy_precision(points, config, block_size=256):
    """P=(K+v I)^-1, without constructing a full pairwise-coordinate tensor."""
    n = len(points)
    v = config.noise_sd**2 + config.relative_jitter * config.signal_sd**2
    matrix = np.empty((n, n), order='F')
    for first in range(0, n, block_size):
        matrix[first:first+block_size] = rbf_covariance(points[first:first+block_size], points, config)
    matrix[np.diag_indices(n)] += v
    factor = cho_factor(matrix, lower=True, overwrite_a=True, check_finite=False)
    precision = cho_solve(factor, np.eye(n, order='F'), overwrite_b=True, check_finite=False)
    precision = .5 * (precision + precision.T)
    # Check inverse columns against a separately reconstructed original block.
    selected = np.unique(np.linspace(0, n-1, min(n, 16), dtype=int))
    columns = rbf_covariance(points, points[selected], config)
    columns[selected, np.arange(len(selected))] += v
    expected = np.zeros_like(columns); expected[selected, np.arange(len(selected))] = 1.
    inverse_error = float(abs(product(precision, columns)-expected).max())
    if inverse_error > 1e-7:
        raise ValueError(f'Noisy precision verification failed: {inverse_error}')
    return precision, v, inverse_error


def conditional_camera(precision, camera, saturated, ambient):
    """Z_S | Z_U=y_U, from the full noisy-camera precision matrix.

    Saturated camera entries are never used as numerical readings. The returned
    completed signal at S is a conditional mean, NOT an observation at c.
    """
    camera = np.asarray(camera, float)
    sat = np.asarray(saturated, bool)
    ids = np.flatnonzero(sat)
    if precision.shape != (len(camera), len(camera)) or sat.shape != camera.shape:
        raise ValueError('Inconsistent camera and precision dimensions')
    if not np.isfinite(camera[~sat]).all():
        raise ValueError('Unsaturated numerical readings must be finite')
    fixed = np.where(sat, 0., camera-ambient)
    completed = np.where(sat, ambient, camera)
    if not len(ids):
        return ids, np.empty(0), np.empty((0, 0)), completed
    block = precision[np.ix_(ids, ids)]
    factor = cho_factor(block, lower=True, check_finite=False)
    covariance = cho_solve(factor, np.eye(len(ids)), check_finite=False)
    covariance = .5 * (covariance + covariance.T)
    rhs = product(precision[ids], fixed[:, None]).ravel()
    mean = ambient - cho_solve(factor, rhs, check_finite=False)
    completed[ids] = mean
    return ids, mean, covariance, completed


def latent_from_camera(precision, noise_variance, completed_signal, ambient, ids):
    """T|Z: mean=Z-vP(Z-m); covariance C=vI-v^2P.

    The columns W=(I-vP)[:, S] map subsequent censored-signal deviations to T.
    """
    v = noise_variance
    base = completed_signal - v*product(precision, (completed_signal-ambient)[:, None]).ravel()
    w = -v*precision[:, ids].copy()
    w[ids, np.arange(len(ids))] += 1.
    diagonal = v-v*v*np.diag(precision)
    if diagonal.min() < -1e-10:
        raise ValueError('Negative latent conditional variance')
    return base, w, np.maximum(diagonal, 0.)


def conditional_covariance_block(precision, v, rows, columns):
    rows, columns = np.asarray(rows, int), np.asarray(columns, int)
    return v*(rows[:, None] == columns[None, :]) - v*v*precision[np.ix_(rows, columns)]


def mixture_moments(base, weights, camera_draws, conditional_camera_mean, residual_diagonal,
                    block_size=256):
    """Rao-Blackwellized latent moments; no independent full-field residuals."""
    draws = np.asarray(camera_draws).reshape(-1, len(conditional_camera_mean))
    mean_z = draws.mean(axis=0)
    mean = base + product(weights, (mean_z-conditional_camera_mean)[:, None]).ravel()
    centered = draws-mean_z
    shared = np.empty(len(base))
    for first in range(0, len(base), block_size):
        conditional = product(centered, weights[first:first+block_size].T)
        shared[first:first+block_size] = np.var(conditional, axis=0, ddof=1)
    return mean, shared+residual_diagonal, shared


def propagated_conditional_residual(precision, v, operator, selected_columns, block_size=128):
    """diag(A C A^T) and selected columns of A C, preserving cross-covariance."""
    variance = np.empty(operator.shape[0])
    ac_columns = np.empty((operator.shape[0], len(selected_columns)))
    for first in range(0, operator.shape[0], block_size):
        a = operator[first:first+block_size]
        if issparse(a):
            ap = a @ precision
            dense = a.toarray()
        else:
            ap = product(a, precision)
            dense = a
        count = a.shape[0]
        variance[first:first+count] = v*np.sum(dense*dense, axis=1)-v*v*np.sum(ap*dense, axis=1)
        ac_columns[first:first+count] = v*dense[:, selected_columns]-v*v*ap[:, selected_columns]
    if variance.min() < -1e-9:
        raise ValueError('Negative propagated conditional variance')
    return np.maximum(variance, 0.), ac_columns


def propagated_state_components(operator, precision, v, ids, camera_draws,
                                ac_columns, full_residual_variance):
    """Separate censored, unsaturated and signed cross terms of A Sigma A^T.

    The censored-only term is exactly the matched all-readings PINNED arm.
    The joint full-latent arm adds the unsaturated and cross terms, not just
    independent diagonal noise at formerly pinned locations.
    """
    a_s = operator[:, ids]
    if issparse(a_s):
        a_s = a_s.toarray()
    c_ss = conditional_covariance_block(precision, v, ids, ids)
    a_s_c = product(a_s, c_ss)
    r_s = np.sum(a_s_c*a_s, axis=1)
    r_cross = 2*(np.sum(ac_columns*a_s, axis=1)-r_s)
    r_u = full_residual_variance-r_s-r_cross
    z = np.asarray(camera_draws).reshape(-1, len(ids))
    z = z-z.mean(axis=0)
    # Full and censored-only mean responses use the SAME completed-camera draws.
    full = product(z, (ac_columns/v).T)
    censored = product(z, (a_s_c/v).T)
    unsaturated = full-censored
    s_s = np.var(censored, axis=0, ddof=1)
    s_u = np.var(unsaturated, axis=0, ddof=1)
    s_cross = 2*np.sum(censored*unsaturated, axis=0)/(len(z)-1)
    shared_full = np.var(full, axis=0, ddof=1)
    np.testing.assert_allclose(s_s+s_u+s_cross, shared_full, atol=1e-10, rtol=1e-10)
    total = shared_full+full_residual_variance
    np.testing.assert_allclose((s_s+r_s)+(s_u+r_u)+(s_cross+r_cross), total,
                              atol=1e-10, rtol=1e-10)
    if min(r_s.min(), r_u.min(), total.min()) < -1e-9:
        raise ValueError('Substantial negative variance in state decomposition')
    return dict(censored=s_s+r_s, unsaturated=s_u+r_u, cross=s_cross+r_cross,
                total=total, residual_censored=r_s, residual_unsaturated=r_u,
                residual_cross=r_cross, shared=shared_full)
