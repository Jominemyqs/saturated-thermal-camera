"""Deterministic projection attribution using the unchanged heat-mean formula."""
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .thermal_posterior_physics import diffuse_and_cool, current_source_field


def area_weights(xs, ys):
    """Tensor trapezoidal weights, including nonuniform rectangular coordinates."""
    def weights(x):
        delta = np.diff(x)
        if np.any(delta <= 0):
            raise ValueError("Coordinates must be strictly increasing")
        return np.r_[delta[0] / 2, (delta[:-1] + delta[1:]) / 2, delta[-1] / 2]
    return np.outer(weights(ys), weights(xs))


def lift_grid(values, xs, ys, target_xs, target_ys):
    x, y = np.meshgrid(target_xs, target_ys)
    points = np.column_stack([y.ravel(), x.ravel()])
    return RegularGridInterpolator((ys, xs), values, bounds_error=True)(points).reshape(x.shape)


def forecast_parts(previous, flux, xs, ys, ambient, dt, config):
    """Reuse the frozen floor, zero extension, Gaussian filter and source cutoff."""
    prepared = SimpleNamespace(xs=xs, ys=ys, ambient=ambient,
                               times=np.array([0., dt]), heat_flux=[flux, flux])
    background = diffuse_and_cool(
        prepared, previous, previous_index=0, current_index=1,
        diffusivity=config['diffusivity'], cooling_rate=config['cooling_rate'])
    source = current_source_field(
        prepared, previous_index=0, current_index=1,
        source_coupling=config['source_coupling'],
        source_flux_threshold=config['source_flux_threshold'])
    return background, source


def controlled_forecasts(native_previous, native_flux, projected_previous,
                         projected_flux, xs, ys, ambient, dt, config):
    bn, sn = forecast_parts(native_previous, native_flux, xs, ys, ambient, dt, config)
    bp, sp = forecast_parts(projected_previous, projected_flux, xs, ys, ambient, dt, config)
    return {
        'native_inputs': ambient + bn + sn,
        'project_previous_only': ambient + bp + sn,
        'project_source_only': ambient + bn + sp,
        'project_both_inputs': ambient + bp + sp,
        'previous_projection_effect': bp - bn,
        'source_projection_effect': sp - sn,
    }


def error_summary(error, weights):
    error, weights = np.asarray(error), np.asarray(weights)
    if error.shape != weights.shape or not np.isfinite(error).all() or weights.sum() <= 0:
        raise ValueError("Finite errors and positive matched area weights required")
    total = weights.sum()
    return dict(bias_K=float(np.sum(weights * error) / total),
                rmse_K=float(np.sqrt(np.sum(weights * error**2) / total)),
                mae_K=float(np.sum(weights * abs(error)) / total))


def bridge_components(native_forecast, projected_forecast, native_truth,
                      projected_truth, lifted_legacy_forecast):
    """Exact bridge; bias adds, RMSE does not."""
    return dict(
        native_oracle_error=native_forecast - native_truth,
        input_projection=projected_forecast - native_forecast,
        reference_projection=native_truth - projected_truth,
        coarse_discretization=lifted_legacy_forecast - projected_forecast,
        legacy_error=lifted_legacy_forecast - projected_truth,
    )
