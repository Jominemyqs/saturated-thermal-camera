"""Observed-camera distance diagnostics for the frozen native-input pilot."""
import numpy as np
from scipy.spatial import cKDTree


def camera_at_ceiling(previous_signal, current_signal, ceiling, numeric_candidates):
    """Reclip pre-saturation signals, not a lower-ceiling camera image."""
    previous = np.minimum(previous_signal, ceiling)
    current = np.minimum(current_signal, ceiling)
    saturated = current == ceiling
    numeric = np.asarray(numeric_candidates, dtype=bool) & ~saturated
    return previous, current, saturated, numeric, np.flatnonzero(numeric | saturated)


def uncertainty_distance(points, numeric, saturated, weights, truth, prediction):
    """Distance is to retained numerical observations, not to all unsaturated nodes."""
    if not np.any(numeric):
        raise ValueError('No retained unsaturated numerical observations')
    distance = cKDTree(points[numeric]).query(points)[0] * 1000
    mean, sd, lower, upper = prediction[:4]
    error = mean-truth
    bins = np.array([0., .1, .25, .5, 1., 2., np.inf])
    rows = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = saturated & (distance >= left) & (distance < right)
        w = weights[mask]
        if not len(w):
            continue
        def avg(v):
            return float(np.sum(w*v[mask])/w.sum())
        ratio = np.divide(abs(error), sd, out=np.full_like(sd, np.nan), where=sd>0)
        rows.append(dict(distance_left_mm=left, distance_right_mm=right,
            distance_mm=avg(distance), nodes=int(mask.sum()), area_mm2=float(w.sum()*1e6),
            rmse_K=np.sqrt(avg(error**2)), mae_K=avg(abs(error)), bias_K=avg(error),
            posterior_sd_K=avg(sd), width95_K=avg(upper-lower),
            coverage95=avg((lower<=truth)&(truth<=upper)), mean_abs_error_over_sd=avg(ratio)))
    return distance, rows


def retained_reference_observations(reference_camera, reference_ceiling, camera,
                                    ceiling, numeric_candidates):
    """Keep candidate nodes plus observed reference saturation; no truth input."""
    reference_camera = np.asarray(reference_camera)
    camera = np.asarray(camera)
    candidates = np.asarray(numeric_candidates, dtype=bool)
    if camera.shape != reference_camera.shape or candidates.shape != camera.shape:
        raise ValueError('Camera and candidate shapes must agree')
    if ceiling < reference_ceiling:
        raise ValueError('Retention control requires a ceiling at least as high as reference')
    if not np.array_equal(np.minimum(camera, reference_ceiling), reference_camera):
        raise ValueError('Cameras must share the same pre-clipping signal')
    selected = candidates | (reference_camera == reference_ceiling)
    saturated = camera == ceiling
    if np.any(saturated & ~selected):
        raise ValueError('Higher-ceiling saturation must be contained in reference saturation')
    ids = np.flatnonzero(selected)
    return ids, selected & ~saturated, saturated
