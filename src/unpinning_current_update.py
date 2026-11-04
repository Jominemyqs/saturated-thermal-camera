"""Full propagated covariance blocks for the isolated output-60 follow-up."""
import numpy as np
from scipy.sparse import issparse

from .previous_correlation_audit import product
from .previous_unpinning_audit import conditional_covariance_block


def propagated_blocks(operator, precision, v, censored_ids, camera_chains,
                      observation_ids, *, pinned, operator_precision=None):
    """A Sigma A^T blocks, including analytic correlated conditional residuals.

    Sigma = W Cov(Z_S) W^T + C, C=vI-v^2P, W=C[:,S]/v.
    Pinned replaces A by A E_S, never by a diagonal covariance approximation.
    """
    a = operator
    ids = np.asarray(censored_ids, int)
    obs = np.asarray(observation_ids, int)
    z = np.asarray(camera_chains).reshape(-1, len(ids))
    z = z-z.mean(axis=0)
    a_s = a[:, ids].toarray() if issparse(a) else a[:, ids]
    if pinned:
        c_ss = conditional_covariance_block(precision, v, ids, ids)
        ac = product(a_s, c_ss)
        response = ac/v
        residual_po = product(ac, a_s[obs].T)
        residual_diag = np.sum(ac*a_s, axis=1)
    else:
        ap = operator_precision
        if ap is None:
            ap = a @ precision if issparse(a) else product(a, precision)
        response = a_s-v*ap[:, ids]
        if issparse(a):
            residual_po = v*(a@a[obs].T).toarray()-v*v*(a[obs]@ap.T).T
            residual_diag = v*np.asarray(a.multiply(a).sum(axis=1)).ravel()
            residual_diag -= v*v*np.asarray(a.multiply(ap).sum(axis=1)).ravel()
        else:
            residual_po = v*product(a, a[obs].T)-v*v*product(ap, a[obs].T)
            residual_diag = v*np.sum(a*a,axis=1)-v*v*np.sum(ap*a,axis=1)
    features = product(response,z.T)/np.sqrt(len(z)-1)
    cross = product(features,features[obs].T)+residual_po
    diagonal = np.sum(features*features,axis=1)+residual_diag
    observed = cross[obs].copy()
    np.testing.assert_allclose(observed,observed.T,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(np.diag(observed),diagonal[obs],atol=1e-10,rtol=1e-10)
    if diagonal.min() < -1e-9:
        raise ValueError('Negative propagated variance')
    return observed, cross, diagonal
