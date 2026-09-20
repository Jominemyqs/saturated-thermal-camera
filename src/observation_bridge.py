"""Observation-only subsetting and observed-mask cooling diagnostics."""
from __future__ import annotations

import numpy as np


def subset_observation_blocks(observations, blocks, keep):
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != np.asarray(observations["y_obs"]).shape or not keep.any():
        raise ValueError("Nonempty observation subset required")
    obs = {**observations}
    for key in ("x_obs", "y_obs", "sat_mask"):
        obs[key] = np.asarray(obs[key])[keep]
    prior = {**blocks,
        "observation_mean": blocks["observation_mean"][keep],
        "observed_covariance": blocks["observed_covariance"][np.ix_(keep, keep)],
        "pred_observed_covariance": blocks["pred_observed_covariance"][:, keep]}
    return obs, prior


def cooling_visibility(observed, ceiling, target, lag, noise_sd, fixed_mask=None):
    """Counts use only noisy clipped observations, never latent temperatures."""
    values = np.asarray(observed)
    if target < 0 or lag < 1 or target + lag >= len(values):
        raise ValueError("Invalid target or lag")
    initial = values[target] == ceiling
    future = values[target + 1:target + lag + 1]
    unsaturated = future < ceiling
    # Three consecutive safely-below-ceiling frames reduce noise-flicker evidence.
    safe = future < ceiling - 2 * noise_sd
    persistent = np.zeros(initial.shape, dtype=bool)
    for start in range(max(0, lag - 2)):
        persistent |= np.all(safe[start:start + 3], axis=0)
    count = int(initial.sum())
    usable = np.ones(initial.shape, dtype=bool) if fixed_mask is None else np.asarray(fixed_mask, dtype=bool)
    if usable.shape != initial.shape:
        raise ValueError("Fixed mask shape differs from camera field")
    masks = {"ever_observable": np.any(unsaturated, axis=0),
             "endpoint_observable": unsaturated[-1],
             "persistent_observable": persistent}
    result = {"target_censored_count": count}
    for label, mask in masks.items():
        result[label + "_count"] = int(np.sum(mask & initial))
        result[label + "_fraction"] = float(np.sum(mask & initial) / count) if count else np.nan
        result[label + "_stride5_count"] = int(np.sum(mask & initial & usable))
        result[label + "_stride5_fraction"] = float(np.sum(mask & initial & usable) / count) if count else np.nan
    return result
