"""Mean-only ablation of the cached explicit current gradient increment."""
import numpy as np


def remove_current_increment(blocks, observation_ids, increment):
    """Return new means and the SAME covariance arrays, without reflooring."""
    ids = np.asarray(observation_ids, dtype=int)
    source = np.asarray(increment, dtype=float)
    mean = np.asarray(blocks['prediction_mean'])
    if source.shape != mean.shape or not np.isfinite(source).all():
        raise ValueError('Increment must be finite and match prediction locations')
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('Observation IDs must be unique and one-dimensional')
    if np.any(ids < 0) or np.any(ids >= len(mean)):
        raise ValueError('Observation outside prediction domain')
    np.testing.assert_array_equal(blocks['observation_mean'], mean[ids])
    result = dict(blocks)
    result['prediction_mean'] = mean-source
    result['observation_mean'] = result['prediction_mean'][ids]
    np.testing.assert_allclose(result['prediction_mean']+source, mean, rtol=0, atol=1e-12)
    return result
