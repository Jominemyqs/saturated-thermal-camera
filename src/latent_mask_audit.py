"""Extend frozen full-latent prior blocks without reverting to hybrid features."""
import numpy as np

from .native_mask_pilot import likelihood_layout


def extend_blocks(base, base_ids, added_ids, added_cross, added_variance):
    """Append same-prior columns, preserving every cached baseline block exactly.

    Overlap and diagonal discrepancies must already be floating-point small.
    Reconciliation uses cached values, not a fitted covariance correction.
    """
    old = np.asarray(base_ids, int)
    added = np.asarray(added_ids, int)
    if len(np.unique(np.r_[old, added])) != len(old)+len(added):
        raise ValueError('Baseline and added IDs must be unique and disjoint')
    if not len(added):
        return old.copy(), dict(base), 0.
    cross = np.asarray(added_cross).copy()
    n = len(base['prediction_mean'])
    if cross.shape != (n, len(added)):
        raise ValueError('Added covariance columns have wrong shape')
    expected = base['pred_observed_covariance'][added].T
    np.testing.assert_allclose(cross[old], expected, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(added_variance, base['prediction_variance'], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(cross[added], cross[added].T, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.diag(cross[added]), base['prediction_variance'][added], atol=1e-10, rtol=1e-10)
    adjustment = max(float(abs(cross[old]-expected).max()),
                     float(abs(np.diag(cross[added])-base['prediction_variance'][added]).max()))
    cross[old] = expected
    cross[added, np.arange(len(added))] = base['prediction_variance'][added]
    ids = np.union1d(old, added)
    old_pos = np.searchsorted(ids, old)
    new_pos = np.searchsorted(ids, added)
    po = np.empty((n, len(ids)))
    po[:, old_pos] = base['pred_observed_covariance']
    po[:, new_pos] = cross
    oo = po[ids].copy()
    oo[np.ix_(old_pos, old_pos)] = base['observed_covariance']
    result = dict(base, observation_mean=base['prediction_mean'][ids],
                  observed_covariance=oo, pred_observed_covariance=po)
    np.testing.assert_array_equal(po[:, old_pos], base['pred_observed_covariance'])
    np.testing.assert_array_equal(oo[np.ix_(old_pos, old_pos)], base['observed_covariance'])
    return ids, result, adjustment


def additional_layout(base_ids, base_kinds, base_values, added_ids, arm, readings=None):
    """The mask arm cannot accept omitted numerical readings, even accidentally."""
    old = np.asarray(base_ids, int)
    added = np.asarray(added_ids, int)
    if arm not in ('C0', 'Cmask', 'Cnumeric'):
        raise ValueError('Unknown arm')
    if len(np.unique(np.r_[old, added])) != len(old)+len(added):
        raise ValueError('Duplicate likelihood location')
    if arm != 'Cnumeric' and readings is not None:
        raise ValueError('Only numerical control accepts added readings')
    if arm == 'C0':
        ids, kinds, values = old.copy(), np.asarray(base_kinds).copy(), np.asarray(base_values).copy()
    else:
        ids = np.union1d(old, added)
        kinds = np.empty(len(ids), int)
        values = np.full(len(ids), np.nan)
        where = np.searchsorted(ids, old)
        kinds[where] = base_kinds
        values[where] = base_values
        where = np.searchsorted(ids, added)
        kinds[where] = -1 if arm == 'Cmask' else 0
        if arm == 'Cnumeric':
            if readings is None or np.shape(readings) != added.shape:
                raise ValueError('Numerical control requires matching readings')
            values[where] = readings
    likelihood_layout(ids, kinds, values)
    return ids, kinds, values
