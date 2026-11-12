"""Prospective, outcome-blind selection for the bounded mask validation."""
import numpy as np
from .latent_mask_audit import extend_blocks


def select_heldout(records, development):
    """Smallest/largest run index per family, excluding historical development."""
    chosen = []
    for family in ('Diagonal', 'Horizontal', 'Spiral'):
        eligible = sorted((r for r in records if r.family == family and r.name not in development),
                          key=lambda r: r.run_index)
        if len(eligible) < 2:
            raise ValueError('Need two held-out trajectories per family')
        chosen.extend([eligible[0].name, eligible[-1].name])
    return chosen


def retained_layout(candidates, reference_saturated, camera, ceiling):
    """Retain O_star at every ceiling; never double-count a likelihood factor."""
    ids = np.flatnonzero(np.asarray(candidates, bool) | np.asarray(reference_saturated, bool))
    saturated = np.asarray(camera)[ids] == ceiling
    kinds = saturated.astype(int)
    values = np.where(saturated, np.nan, np.asarray(camera)[ids])
    return ids, kinds, values


def align_extended_prior(base, base_ids, added_ids, concatenated):
    """Reorder appended covariance columns to the likelihood's sorted union IDs."""
    old = np.asarray(base_ids, int)
    added = np.asarray(added_ids, int)
    order = np.r_[old, added]
    np.testing.assert_array_equal(concatenated['observation_mean'],base['prediction_mean'][order])
    np.testing.assert_array_equal(concatenated['pred_observed_covariance'][:,:len(old)],base['pred_observed_covariance'])
    np.testing.assert_array_equal(concatenated['observed_covariance'][:len(old),:len(old)],base['observed_covariance'])
    return extend_blocks(base,old,added,concatenated['pred_observed_covariance'][:,len(old):],
                         concatenated['prediction_variance'])


def verify_prior_layout(prior, ids):
    """Cheap fail-fast checks prevent silently attaching factors to wrong nodes."""
    np.testing.assert_array_equal(prior['observation_mean'],prior['prediction_mean'][ids])
    np.testing.assert_allclose(prior['observed_covariance'],prior['pred_observed_covariance'][ids],atol=1e-10,rtol=0)
    np.testing.assert_allclose(np.diag(prior['observed_covariance']),prior['prediction_variance'][ids],atol=1e-10,rtol=0)
