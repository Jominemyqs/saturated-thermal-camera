"""Common-coordinate propagation for a matched projection-on/off GP audit."""
import numpy as np
from scipy.ndimage import gaussian_filter, shift as shift_image
from scipy.sparse import coo_matrix
from matplotlib.tri import Triangulation

from .thermal_trajectory import SurfaceGridProjector
from .native_surface import hottest_area_weights
from .metrics import empirical_crps


def bilinear_matrix(xs, ys, points):
    points = np.asarray(points)
    if (np.any(points[:, 0] < xs[0]-1e-12) or np.any(points[:, 0] > xs[-1]+1e-12)
            or np.any(points[:, 1] < ys[0]-1e-12) or np.any(points[:, 1] > ys[-1]+1e-12)):
        raise ValueError('Evaluation outside rectangle')
    ix = np.clip(np.searchsorted(xs, points[:, 0], side='right')-1, 0, len(xs)-2)
    iy = np.clip(np.searchsorted(ys, points[:, 1], side='right')-1, 0, len(ys)-2)
    u = (points[:, 0]-xs[ix])/(xs[ix+1]-xs[ix])
    v = (points[:, 1]-ys[iy])/(ys[iy+1]-ys[iy])
    cols = np.column_stack([iy*len(xs)+ix, iy*len(xs)+ix+1,
                            (iy+1)*len(xs)+ix, (iy+1)*len(xs)+ix+1])
    weights = np.column_stack([(1-u)*(1-v), u*(1-v), (1-u)*v, u*v])
    return coo_matrix((weights.ravel(), (np.repeat(np.arange(len(points)), 4), cols.ravel())),
                      shape=(len(points), len(xs)*len(ys))).tocsr()


def triangle_matrix(mesh, targets):
    tri = Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)
    ids = tri.get_trifinder()(targets[:, 0], targets[:, 1])
    if np.any(ids < 0):
        raise ValueError('Evaluation outside native surface')
    vertices = mesh.triangles[ids]
    a = mesh.points[vertices[:, 0]]
    b = mesh.points[vertices[:, 1]]-a
    c = mesh.points[vertices[:, 2]]-a
    d = targets-a
    det = b[:, 0]*c[:, 1]-b[:, 1]*c[:, 0]
    u = (d[:, 0]*c[:, 1]-d[:, 1]*c[:, 0])/det
    v = (b[:, 0]*d[:, 1]-b[:, 1]*d[:, 0])/det
    weights = np.column_stack([1-u-v, u, v])
    assert weights.min() > -1e-8
    return coo_matrix((weights.ravel(), (np.repeat(np.arange(len(targets)), 3), vertices.ravel())),
                      shape=(len(targets), len(mesh.points))).tocsr()


class MatchedHeatOperator:
    """Identical Gaussian propagation; only spatial input interpolation differs."""
    def __init__(self, mesh, coarse_xs, coarse_ys, *, nx=601, ny=401):
        self.xs = np.linspace(coarse_xs[0], coarse_xs[-1], nx)
        self.ys = np.linspace(coarse_ys[0], coarse_ys[-1], ny)
        self.shape = (ny, nx)
        x, y = np.meshgrid(self.xs, self.ys)
        targets = np.column_stack([x.ravel(), y.ravel()])
        p = SurfaceGridProjector(mesh.points, nx=len(coarse_xs), ny=len(coarse_ys))
        self.project = coo_matrix((p.weights.ravel(),
            (np.repeat(np.arange(len(p.vertices)), 3), p.vertices.ravel())),
            shape=(len(p.vertices), len(mesh.points))).tocsr()
        self.input_maps = {'native': triangle_matrix(mesh, targets),
                           'projected': bilinear_matrix(coarse_xs, coarse_ys, targets) @ self.project}
        self.evaluate = bilinear_matrix(self.xs, self.ys, mesh.points)
        self.roundtrip = bilinear_matrix(coarse_xs, coarse_ys, mesh.points) @ self.project
        for mapping in list(self.input_maps.values()) + [self.evaluate, self.roundtrip]:
            np.testing.assert_allclose(np.asarray(mapping.sum(axis=1)).ravel(), 1., atol=1e-12)

    def propagate(self, values, representation, *, dt, diffusivity, cooling_rate,
                  floor=False, batch_size=16, displacement=None):
        if min(dt, diffusivity, cooling_rate) < 0:
            raise ValueError('Negative physical parameter')
        original = np.asarray(values)
        flat = np.atleast_2d(original)
        result = np.empty_like(flat, dtype=float)
        sigma = (0, np.sqrt(2*diffusivity*dt)/np.mean(np.diff(self.ys)),
                 np.sqrt(2*diffusivity*dt)/np.mean(np.diff(self.xs)))
        for start in range(0, len(flat), batch_size):
            fields = (self.input_maps[representation] @ flat[start:start+batch_size].T).T
            fields = fields.reshape((-1,)+self.shape)
            if floor:
                fields = np.maximum(fields, 0)
            fields = gaussian_filter(fields, sigma=sigma, mode='constant', cval=0.)
            fields *= np.exp(-cooling_rate*dt)
            if displacement is not None:
                displacement = np.asarray(displacement, dtype=float)
                if displacement.shape != (2,) or not np.isfinite(displacement).all():
                    raise ValueError('Displacement must contain finite dx, dy')
                # Positive displacement moves a feature forward: output(x)=input(x-d).
                fields = shift_image(fields, (0, displacement[1]/np.mean(np.diff(self.ys)),
                                              displacement[0]/np.mean(np.diff(self.xs))),
                                     order=1, mode='constant', cval=0., prefilter=False)
            result[start:start+batch_size] = (self.evaluate @ fields.reshape(len(fields), -1).T).T
        return result[0] if original.ndim == 1 else result

    def mean(self, previous, flux, representation, *, ambient, source_coupling,
             source_flux_threshold, **physics):
        q = flux if representation == 'native' else self.roundtrip @ flux
        return (ambient + self.propagate(previous-ambient, representation, floor=True, **physics)
                + source_coupling * np.where(q >= source_flux_threshold, q, 0) * physics['dt'])


def weighted_metrics(prediction, truth, weights, ambient, censored):
    mean, sd, low, high, draws = prediction
    error = mean-truth
    crps = empirical_crps(draws, truth)
    rows = []
    for region, w in [('whole_area', weights), ('hot1pct_area', hottest_area_weights(truth, weights)),
                      ('observed_censored', weights*censored)]:
        def average(values):
            return float(np.sum(w*values)/w.sum())
        rows.append(dict(region=region,rmse_K=np.sqrt(average(error**2)),mae_K=average(abs(error)),
            bias_K=average(error),crps_K=average(crps),coverage95=average((low<=truth)&(truth<=high)),
            width95_K=average(high-low),posterior_sd_K=average(sd),
            relative_L2=np.sqrt(average(error**2)/average((truth-ambient)**2)),
            peak_error_K=float(mean.max()-truth.max()),peak_absolute_error_K=float(abs(mean.max()-truth.max()))))
    return rows
