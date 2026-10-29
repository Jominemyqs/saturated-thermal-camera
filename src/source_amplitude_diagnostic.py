"""Scalar amplitude diagnostics for fixed deterministic source shapes."""
import numpy as np


def scale_diagnostic(base, source, truth, weights, grid):
    """Closed-form weighted least squares; no inferred spatial parameter map."""
    base, source, truth, weights = map(lambda a: np.asarray(a, float),
                                       (base, source, truth, weights))
    if not (base.shape == source.shape == truth.shape == weights.shape):
        raise ValueError('Shapes must match')
    if not all(np.isfinite(a).all() for a in (base,source,truth,weights)):
        raise ValueError('Nonfinite input')
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Invalid region weights')
    w=weights/weights.sum()
    target=truth-base
    denominator=float(np.sum(w*source**2))
    numerator=float(np.sum(w*source*target))
    grid=np.asarray(grid,float)
    losses=np.array([np.sum(w*(base+k*source-truth)**2) for k in grid])
    identifiable=denominator > 1e-20
    optimum=numerator/denominator if identifiable else np.nan
    return dict(kappa_unconstrained=optimum,
                kappa_bounded=float(np.clip(optimum,grid.min(),grid.max())) if identifiable else np.nan,
                kappa_grid=float(grid[np.argmin(losses)]) if identifiable else np.nan,
                identifiable=identifiable,source_rms_K=np.sqrt(denominator),
                quadratic_curvature=2*denominator,
                optimum_outside_grid=bool(identifiable and not grid.min()<=optimum<=grid.max()))


def select_shared_scale(cases, grid):
    """Each supplied trajectory contributes equally, regardless of node count."""
    objectives=[]
    for k in grid:
        objectives.append(np.mean([np.average((b+k*s-t)**2,weights=w) for b,s,t,w in cases]))
    index=int(np.argmin(objectives))
    return float(grid[index]), np.asarray(objectives)
