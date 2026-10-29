"""Source-only quadrature; no inference, fitting, flooring or history rollout."""
import numpy as np
from numpy.polynomial.legendre import leggauss


def threshold_endpoints(g_previous, g_current, threshold):
    previous, current = np.broadcast_arrays(g_previous, g_current)
    if previous.ndim != 1 or not np.isfinite([previous, current]).all():
        raise ValueError("Expected finite one-dimensional native gradient fields")
    return tuple(np.where(g >= threshold, g, 0.0) for g in (previous, current))


def source_contributions(q_previous, q_current, *, gamma, dt, evolve):
    """evolve(q, age) acts linearly on a rate field, WITHOUT a state floor.

    Endpoint values have zero age and are used directly, as in the historical
    endpoint addition. All positive-age calls use the supplied frozen operator.
    """
    if dt < 0 or not np.isfinite([gamma, dt]).all():
        raise ValueError("Invalid coupling or timestep")
    qp, qc = np.broadcast_arrays(q_previous, q_current)
    result = {"S0": gamma * qc * dt}
    result["S_time"] = gamma * dt / 2 * (qp + qc)
    result["S_evolve"] = gamma * dt / 2 * (evolve(qc, dt) + qc)
    result["S_both"] = gamma * dt / 2 * (evolve(qp, dt) + qc)
    for order in (2, 4):
        nodes, weights = leggauss(order)
        theta, weights = (nodes + 1) / 2, weights / 2
        result[f"S_GL{order}"] = gamma * dt * sum(
            w * evolve((1 - t) * qp + t * qc, (1 - t) * dt)
            for t, w in zip(theta, weights)
        )
    return result


def native_source_contributions(operator, g_previous, g_current, *, threshold,
                                gamma, dt, diffusivity, cooling_rate):
    qp, qc = threshold_endpoints(g_previous, g_current, threshold)
    def evolve(values, age):
        return operator.propagate(values, "native", dt=age,
                                  diffusivity=diffusivity,
                                  cooling_rate=cooling_rate, floor=False)
    return source_contributions(qp, qc, gamma=gamma, dt=dt, evolve=evolve)


def error_metrics(values, truth, weights):
    weights = np.asarray(weights, float)
    if weights.sum() <= 0:
        raise ValueError("Empty evaluation region")
    error = values - truth
    return dict(rmse_K=float(np.sqrt(np.average(error**2, weights=weights))),
                bias_K=float(np.average(error, weights=weights)),
                mae_K=float(np.average(abs(error), weights=weights)))


def source_metrics(source, points, weights):
    mass = weights * source
    integral = float(mass.sum())
    peak = int(np.argmax(source))
    if integral > 0:
        center = np.average(points, axis=0, weights=mass)
        spread = np.sqrt(np.average(np.sum((points-center)**2, axis=1), weights=mass))
    else:
        center, spread = np.full(2, np.nan), np.nan
    return dict(integral_K_m2=integral, maximum_K=float(source[peak]), peak_node=peak,
                peak_x_mm=points[peak, 0]*1000, peak_y_mm=points[peak, 1]*1000,
                centroid_x_mm=center[0]*1000, centroid_y_mm=center[1]*1000,
                rms_radius_mm=float(spread*1000),
                area_above_1K_m2=float(weights[source >= 1].sum()))
