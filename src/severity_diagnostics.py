"""Descriptive failure diagnostics; never used to fit or select a model."""
from __future__ import annotations
import numpy as np
from scipy.stats import skew


def region_masks(truth, censored):
    y = np.asarray(truth).ravel()
    top = np.zeros(len(y), bool)
    top[np.argsort(y, kind='mergesort')[-max(1, int(.01*len(y))):]] = True
    return {'overall': np.ones(len(y), bool), 'top_1pct': top,
            'observed_censored': np.asarray(censored, bool).ravel()}


def mean_diagnostics(mean, truth, masks):
    mu, y = np.asarray(mean).ravel(), np.asarray(truth).ravel()
    e = mu-y
    peak = float(mu.max()-y.max())
    at_peak = float(e[np.argmax(y)])
    return [dict(region=name, n_pixels=int(mask.sum()), bias_K=float(e[mask].mean()),
                 rmse_K=float(np.sqrt(np.mean(e[mask]**2))),
                 peak_signed_error_K=peak, peak_absolute_error_K=abs(peak),
                 error_at_true_peak_K=at_peak) for name, mask in masks.items()]


def posterior_diagnostics(prediction, truth, masks):
    mu, sd, lo, hi, draws = prediction
    y = np.asarray(truth).ravel()
    # Undefined standardized errors remain undefined, never silently floored.
    valid = np.isfinite(sd) & (sd>1e-12)
    z = np.full(len(y), np.nan)
    z[valid] = (y[valid]-mu[valid])/sd[valid]
    pit = np.mean(draws <= y[None], axis=0)
    median = np.median(draws, axis=0)
    skewness = skew(draws, axis=0, bias=False)
    rows=[]
    for name, mask in masks.items():
        v = mask & valid
        rows.append(dict(region=name, n_pixels=int(mask.sum()), n_invalid_sd=int((mask & ~valid).sum()),
            z_mean=float(np.mean(z[v])), z_sd=float(np.std(z[v], ddof=0)),
            z_rms=float(np.sqrt(np.mean(z[v]**2))),
            coverage_95=float(np.mean((lo[mask]<=y[mask])&(y[mask]<=hi[mask]))),
            width_95_K=float(np.mean(hi[mask]-lo[mask])), posterior_sd_K=float(sd[mask].mean()),
            pit_mean=float(pit[mask].mean()), pit_median=float(np.median(pit[mask])),
            pit_lower_05_fraction=float(np.mean(pit[mask]<.05)),
            pit_upper_95_fraction=float(np.mean(pit[mask]>.95)),
            posterior_skewness_mean=float(np.mean(skewness[v])),
            posterior_skewness_median=float(np.median(skewness[v])),
            mean_minus_median_K=float(np.mean(mu[mask]-median[mask]))))
    return rows, dict(z=z, pit=pit, skewness=skewness, posterior_median=median)


def local_source_coordinates(prepared, cutoff):
    centers=[]
    for flux in prepared.heat_flux[-2:]:
        weights=np.where(flux>=cutoff, flux, 0.).ravel()
        if weights.sum()<=0:
            return None, 'inactive source in one of the two frames'
        centers.append(np.sum(prepared.points*weights[:,None], axis=0)/weights.sum())
    displacement=centers[1]-centers[0]
    distance=np.linalg.norm(displacement)
    if distance<=1e-12:
        return None, 'stationary source; no motion direction'
    direction=displacement/distance
    cross=np.array([-direction[1],direction[0]])
    delta=prepared.points-centers[1]
    return dict(along=delta@direction, cross=delta@cross,
                center=centers[1], direction=direction, displacement=displacement), 'valid'


def directional_diagnostics(prior, truth, coordinates, length):
    if coordinates is None:
        return []
    e=np.asarray(prior).ravel()-np.asarray(truth).ravel()
    along=coordinates['along']/length
    cross=coordinates['cross']/length
    rows=[]
    for axis, position, transverse in [('along',along,cross),('cross',cross,along)]:
        edges=np.linspace(-8,8,17)
        for a,b in zip(edges[:-1],edges[1:]):
            mask=(position>=a)&(position<b)&(abs(transverse)<=2)
            if mask.any():
                rows.append(dict(axis=axis, bin_center=(a+b)/2, n_pixels=int(mask.sum()),
                    bias_K=float(e[mask].mean()),rmse_K=float(np.sqrt(np.mean(e[mask]**2)))))
    for side,mask in [('behind',(along<0)&(along>=-8)&(abs(cross)<=2)),
                      ('ahead',(along>=0)&(along<8)&(abs(cross)<=2))]:
        if mask.any():
            rows.append(dict(axis=side,bin_center=0.,n_pixels=int(mask.sum()),
                bias_K=float(e[mask].mean()),rmse_K=float(np.sqrt(np.mean(e[mask]**2)))))
    return rows
