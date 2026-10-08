"""Reference/320/325 K native B/C pilot; frozen model, regenerated shared camera."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.linalg.blas import dgemm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.thermal_trajectory import ThermalTrajectory, trajectory_catalog
from src.thermal_posterior_physics import prepare_trajectory
from src.native_surface import extract_surface, hottest_area_weights
from src.matched_projection import MatchedHeatOperator, weighted_metrics
from src.pseudo_censoring_runner import _previous_prediction, _current_prediction
from src.censored_gp import RBFConfig, rbf_covariance
from src.stochastic_heat_gp import StochasticHeatConfig, finite_step_innovation_covariance
from src.native_ceiling_diagnostics import camera_at_ceiling, uncertainty_distance

BASE = ROOT/'outputs/by_experiment/49_matched_native_grid_gp'
OUT = ROOT/'outputs/by_experiment/52_native_higher_ceiling'
DATA = ROOT.parent/'heat_eq_laser_trajectories'
NAMES = ['DiagonalScanPath_8', 'SpiralScanPath_13']
SUFFIXES = ['mean', 'sd', 'lower', 'upper', 'draws']


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def gram(a, b):
    return dgemm(alpha=1., a=np.asfortranarray(a), b=np.asfortranarray(b), trans_b=True)


def plot_case(name, levels, error_limit):
    sections = pd.read_csv(BASE/'native_diagnostics'/f'cross_sections_{name}.csv')
    sections = sections[sections.model=='native_B']
    table = []
    for section, nodes in sections.groupby('section', sort=False):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True, constrained_layout=True)
        ids = nodes.node.to_numpy(int); x = nodes.along_mm.to_numpy()
        for col, d in enumerate(levels):
            for row, model in enumerate(['B', 'C']):
                ax = axes[row, col]
                ax.fill_between(x, d[model+'_lower'][ids], d[model+'_upper'][ids], color='#0072B2', alpha=.22, label='Pointwise 95% interval')
                ax.plot(x, d['truth'][ids], color='black', lw=1.7, label='Native truth')
                ax.plot(x, d['physics_mean'][ids], '--', color='#D55E00', label='Physics forecast')
                ax.plot(x, d[model+'_mean'][ids], color='#0072B2', lw=1.2, label='Posterior mean')
                ax.axhline(float(d['ceiling']), color='#777777', ls=':', label='Camera ceiling')
                ax.set(title=f"{model} | ceiling {float(d['ceiling']):.2f} K", xlabel='Distance from true peak (mm)', ylabel='Temperature (K)')
                ax.grid(alpha=.15)
                for pos, node in enumerate(ids):
                    table.append(dict(trajectory=name, section=section, model=model,
                        ceiling_K=float(d['ceiling']), node=int(node), along_mm=x[pos],
                        normal_offset_mm=float(nodes.normal_offset_mm.iloc[pos]), truth_K=d['truth'][node],
                        physics_K=d['physics_mean'][node], mean_K=d[model+'_mean'][node],
                        lower_K=d[model+'_lower'][node], upper_K=d[model+'_upper'][node]))
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(name+' | '+section.replace('flux', 'gradient')+'\nB: posterior physics mean + RBF; C: same mean + sequential heat covariance\nSame native nodes as Experiment 49; connecting bands are visual guides, not off-mesh credible intervals', fontsize=11)
        fig.savefig(OUT/f'cross_sections_{name}_{section.split()[0].lower()}.png', dpi=170)
        plt.close(fig)
    pd.DataFrame(table).to_csv(OUT/f'cross_sections_{name}.csv', index=False)
    tri = Triangulation(levels[0]['points'][:, 0]*1000, levels[0]['points'][:, 1]*1000, levels[0]['triangles'])
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0, vmax=error_limit)
    for col, d in enumerate(levels):
        for row, (key, label) in enumerate([('physics_mean', 'Physics forecast'), ('B_mean', 'B posterior'), ('C_mean', 'C posterior')]):
            ax = axes[row, col]
            im = ax.tripcolor(tri, d[key]-d['truth'], shading='gouraud', cmap='RdBu_r', norm=norm)
            ax.tricontour(tri, (d['camera']==d['ceiling']).astype(float), levels=[.5], colors='black', linewidths=.6)
            ax.set(title=f"{label} | {float(d['ceiling']):.2f} K", xlabel='x (mm)', ylabel='y (mm)', aspect='equal')
    fig.colorbar(im, ax=list(axes.flat), label='Signed error (estimate - truth), K', shrink=.7)
    fig.suptitle(name+': same target frame, higher camera ceiling\nShared symmetric scale across both trajectories and all ceilings; black = observed saturation boundary')
    fig.savefig(OUT/f'signed_errors_{name}.png', dpi=170)
    center = levels[0]['points'][np.argmax(levels[0]['truth'])]*1000
    for ax in axes.flat:
        ax.set_xlim(center[0]-1.5, center[0]+1.5)
        ax.set_ylim(center[1]-1.5, center[1]+1.5)
    fig.suptitle(name+': 3 mm window around the true peak (evaluation only)\nIdentical signed-error scale; black = observed saturation boundary')
    fig.savefig(OUT/f'signed_errors_zoom_{name}.png', dpi=170); plt.close(fig)


def main():
    for variable in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
        if os.environ.get(variable) != '1':
            raise RuntimeError(f'Set {variable}=1 before launch to reproduce the frozen numerical environment')
    OUT.mkdir(parents=True, exist_ok=True)
    o = pd.read_csv(BASE/'frozen_configuration.csv').iloc[0].to_dict()
    pd.DataFrame([o]).to_csv(OUT/'frozen_configuration.csv', index=False)
    catalog = trajectory_catalog(DATA); assert len(catalog)==33
    timing = []
    for r in catalog:
        t = ThermalTrajectory(r.xdmf_path)
        intervals = np.diff(t.times)
        timing.append(dict(trajectory=r.name, frames=len(t.times), dt_min_s=intervals.min(),
            dt_max_s=intervals.max(), target_dt_s=t.times[-1]-t.times[-2]))
    pd.DataFrame(timing).to_csv(OUT/'dataset_timestamps.csv', index=False)
    results, checks, distances, severity = [], [], [], []
    oldchecks = pd.read_csv(BASE/'matching_checks.csv')
    for index, r in enumerate(catalog):
        if r.name not in NAMES:
            continue
        t = ThermalTrajectory(r.xdmf_path); mesh = extract_surface(t); xy = mesh.points; N = len(xy)
        p, _ = prepare_trajectory(DATA, r.name, nx=int(o['nx']), ny=int(o['ny']), heat_flux_cutoff=o['heat_flux_cutoff'])
        truth = t.read_temperature(-1, surface_only=True)
        previous_truth = t.read_temperature(-2, surface_only=True)
        flux = t.read_heat_flux_z(-1, surface_only=True)
        old = np.load(BASE/'by_trajectory'/r.name/'matched.npz')
        rng = np.random.default_rng(int(o['seed'])+10000*index)
        previous_signal = previous_truth+rng.normal(0, o['measurement_noise_sd'], N)
        current_signal = truth+rng.normal(0, o['measurement_noise_sd'], N)
        gx, gy = np.meshgrid(p.xs[::5], p.ys[::5])
        sensors = cKDTree(xy).query(np.column_stack([gx.ravel(), gy.ravel()]))[1]
        fixed = np.zeros(N, bool); fixed[np.unique(sensors)] = True
        points = np.column_stack([xy, np.full(N, t.times[-1])])
        length = p.source_lengthscale*o['length_multiplier']
        op = MatchedHeatOperator(mesh, p.xs, p.ys)
        physics = dict(dt=float(t.times[-1]-t.times[-2]), diffusivity=o['diffusivity'], cooling_rate=o['cooling_rate'])
        meanargs = dict(ambient=p.ambient, source_coupling=o['source_coupling'], source_flux_threshold=o['source_flux_threshold'], **physics)
        hcfg = StochasticHeatConfig(signal_sd=o['signal_sd'], forcing_lengthscale=length*o['forcing_length_multiplier'],
            diffusivity=o['diffusivity'], cooling_rate=o['cooling_rate'], quadrature_order=int(o['quadrature_order']))
        rbfcfg = RBFConfig(mean_temp=p.ambient, signal_sd=o['signal_sd'], lengthscale=length, noise_sd=o['current_noise_sd'])
        previous_seed = int(o['seed'])+50000*index; current_seed = int(o['seed'])+100000*index
        gradient_contribution = o['source_coupling']*np.where(flux>=o['source_flux_threshold'], flux, 0)*physics['dt']
        for level, ceiling in enumerate([float(old['ceiling']), 320., 325.]):
            start = perf_counter()
            previous_camera, camera, saturated, numeric, obsids = camera_at_ceiling(previous_signal, current_signal, ceiling, fixed)
            if level==0:
                for a, b in [(previous_camera, old['previous_camera']), (camera, old['camera']), (obsids, old['observation_ids'])]:
                    np.testing.assert_array_equal(a, b)
            obs = dict(x_obs=points[obsids], y_obs=camera[obsids], sat_mask=saturated[obsids],
                saturated_full=saturated, x_pred=points, threshold=ceiling)
            proxy = SimpleNamespace(points=xy, ambient=p.ambient, history=previous_truth.reshape(1, 1, N))
            frame = dict(time_index=0, time=float(t.times[-2]), clipped_full=previous_camera[None], saturated_full=(previous_camera==ceiling)[None])
            previous, draws, pc = _previous_prediction(proxy, frame=frame, fixed_mask=fixed[None], threshold=ceiling,
                signal_sd=o['signal_sd'], lengthscale=length, noise_sd=o['previous_noise_sd'], options=o, seed=previous_seed)
            previous = previous.ravel(); draws = draws[:, 0, :]
            if level==0:
                np.testing.assert_array_equal(previous, old['previous_mean'])
                expected = oldchecks[oldchecks.trajectory==r.name].iloc[0]
                assert digest(draws)==expected.previous_draws_sha256
            mean = op.mean(previous, flux, 'native', **meanargs)
            if level==0:
                np.testing.assert_array_equal(mean, old['native_physics_mean'])
            arrays = dict(points=xy, triangles=mesh.triangles, area_weights=mesh.weights, truth=truth,
                previous_truth=previous_truth, previous_mean=previous, previous_draws=draws,
                previous_signal=previous_signal, current_signal=current_signal, previous_camera=previous_camera,
                camera=camera, ceiling=ceiling, observation_ids=obsids, numeric_candidates=fixed,
                numeric_retained=numeric, physics_mean=mean, gradient_contribution=gradient_contribution,
                propagated_without_source=mean-gradient_contribution, ambient=p.ambient,
                hot_area_weights=hottest_area_weights(truth, mesh.weights))
            severity.append(dict(trajectory=r.name, level=level, ceiling_K=ceiling,
                peak_K=truth.max(), previous_peak_K=previous_truth.max(),
                area_censored_fraction=float(np.sum(mesh.weights*saturated)/mesh.weights.sum()),
                previous_area_censored_fraction=float(np.sum(mesh.weights*(previous_camera==ceiling))/mesh.weights.sum()),
                hidden_peak_excess_fraction=max(0., float((truth.max()-ceiling)/(truth.max()-p.ambient))),
                numeric_count=int(numeric.sum()), censored_count=int(saturated.sum()), native_nodes=N))
            for model in ['B', 'C']:
                tick = perf_counter()
                if model=='B':
                    koo = rbf_covariance(points[obsids], points[obsids], rbfcfg)
                    kpo = rbf_covariance(points, points[obsids], rbfcfg)
                    diag = np.full(N, o['signal_sd']**2)
                    inherited = np.zeros(N)
                else:
                    features = op.propagate(draws-previous, 'native', **physics).T/np.sqrt(len(draws)-1)
                    foo = features[obsids]
                    koo = finite_step_innovation_covariance(points[obsids], points[obsids], hcfg, physics['dt'])+gram(foo, foo)
                    kpo = finite_step_innovation_covariance(points, points[obsids], hcfg, physics['dt'])+gram(features, foo)
                    inherited = np.sum(features**2, axis=1)
                    diag = float(finite_step_innovation_covariance(points[:1], points[:1], hcfg, physics['dt'])[0, 0])+inherited
                    del features, foo
                blocks = dict(prediction_mean=mean, observation_mean=mean[obsids], observed_covariance=koo,
                    pred_observed_covariance=kpo, prediction_variance=diag)
                pred, cc = _current_prediction(obs, blocks, options=o, seed=current_seed)
                if level==0:
                    for suffix, value in zip(SUFFIXES, pred):
                        np.testing.assert_array_equal(value, old['native_'+model+'_'+suffix])
                for suffix, value in zip(SUFFIXES, pred):
                    arrays[model+'_'+suffix] = value
                arrays[model+'_prior_variance'] = diag
                arrays[model+'_inherited_variance'] = inherited
                distance, rows = uncertainty_distance(xy, numeric, saturated, mesh.weights, truth, pred)
                arrays['distance_to_retained_numeric_mm'] = distance
                distances.extend(dict(trajectory=r.name, level=level, ceiling_K=ceiling, model=model, **v) for v in rows)
                results.extend(dict(trajectory=r.name, level=level, ceiling_K=ceiling, model=model, **v)
                    for v in weighted_metrics(pred, truth, mesh.weights, p.ambient, saturated))
                checks.append(dict(trajectory=r.name, level=level, model=model, ceiling_K=ceiling,
                    previous_signal_sha256=digest(previous_signal), current_signal_sha256=digest(current_signal),
                    candidate_mask_sha256=digest(fixed), hot_area_sha256=digest(arrays['hot_area_weights']),
                    gradient_term_sha256=digest(gradient_contribution), previous_draws_sha256=digest(draws),
                    current_camera_sha256=digest(camera), observation_ids_sha256=digest(obsids),
                    previous_seed=previous_seed, current_seed=current_seed, lengthscale_m=length,
                    previous_rhat=pc['rhat_max'], current_rhat=cc['rhat_max'], current_censored_rhat=cc['selected_rhat_max'],
                    previous_retried=pc['previous_retried'], current_retried=cc['current_retried'],
                    model_seconds=perf_counter()-tick, level_elapsed_seconds=perf_counter()-start,
                    reference_exact_replay=level==0))
                print(r.name, ceiling, model, 'Rhat', cc['rhat_max'], 'seconds', perf_counter()-tick, flush=True)
            folder = OUT/'by_trajectory'/r.name; folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/f'level_{level}.npz', **arrays)
            for filename, rows in [('results', results), ('verification', checks), ('distance_summary', distances), ('camera_severity', severity)]:
                pd.DataFrame(rows).to_csv(OUT/(filename+'.csv'), index=False)
        old.close()
    check = pd.DataFrame(checks)
    for _, group in check.groupby('trajectory'):
        for col in ['previous_signal_sha256', 'current_signal_sha256', 'candidate_mask_sha256', 'hot_area_sha256', 'gradient_term_sha256', 'previous_seed', 'current_seed', 'lengthscale_m']:
            assert group[col].nunique()==1, col
    for _, group in check.groupby(['trajectory', 'level']):
        for col in ['previous_draws_sha256', 'current_camera_sha256', 'observation_ids_sha256']:
            assert group[col].nunique()==1, col
    plot_outputs()
    files = [Path(__file__), ROOT/'src/native_ceiling_diagnostics.py', ROOT/'src/matched_projection.py',
        ROOT/'src/pseudo_censoring_runner.py', ROOT/'src/metrics.py', BASE/'frozen_configuration.csv']
    (OUT/'source_fingerprints.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, indent=2))


def plot_outputs():
    limit = 0.
    forecasts, peak_nodes = [], []
    for name in NAMES:
        for level in range(3):
            with np.load(OUT/'by_trajectory'/name/f'level_{level}.npz') as d:
                limit = max(limit, *(float(abs(d[key]-d['truth']).max()) for key in ['physics_mean', 'B_mean', 'C_mean']))
                for region, w in [('whole_area', d['area_weights']), ('hot1pct_area', d['hot_area_weights'])]:
                    error = d['physics_mean']-d['truth']
                    forecasts.append(dict(trajectory=name, level=level, ceiling_K=float(d['ceiling']), region=region,
                        bias_K=float(np.sum(w*error)/w.sum()), rmse_K=float(np.sqrt(np.sum(w*error**2)/w.sum()))))
                node = int(np.argmax(d['truth']))
                for model in ['B', 'C']:
                    peak_nodes.append(dict(trajectory=name, level=level, ceiling_K=float(d['ceiling']), model=model,
                        node=node, truth_K=float(d['truth'][node]), previous_truth_K=float(d['previous_truth'][node]),
                        previous_mean_K=float(d['previous_mean'][node]), physics_K=float(d['physics_mean'][node]),
                        gradient_increment_K=float(d['gradient_contribution'][node]),
                        posterior_mean_K=float(d[model+'_mean'][node]), posterior_sd_K=float(d[model+'_sd'][node]),
                        lower95_K=float(d[model+'_lower'][node]), upper95_K=float(d[model+'_upper'][node])))
    pd.DataFrame(forecasts).to_csv(OUT/'physics_forecast_summary.csv', index=False)
    pd.DataFrame(peak_nodes).to_csv(OUT/'true_peak_node_summary.csv', index=False)
    metrics = pd.read_csv(OUT/'results.csv')
    metrics[metrics.region=='hot1pct_area'].to_csv(OUT/'hot_area_summary.csv', index=False)
    for name in NAMES:
        levels = [np.load(OUT/'by_trajectory'/name/f'level_{j}.npz') for j in range(3)]
        plot_case(name, levels, limit)
        for d in levels:
            d.close()
    d = pd.read_csv(OUT/'distance_summary.csv')
    for name, group in d.groupby('trajectory'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, (metric, label) in zip(axes.flat, [('rmse_K', 'Actual RMSE (K)'), ('posterior_sd_K', 'Posterior SD (K)'), ('coverage95', '95% coverage'), ('width95_K', '95% interval width (K)')]):
            for (level, model), g in group.groupby(['level', 'model']):
                ax.plot(g.distance_mm, g[metric], marker='o', ls='-' if model=='B' else '--',
                    color=['#777777', '#0072B2', '#D55E00'][int(level)], label=f"{model}, {g.ceiling_K.iloc[0]:.2f} K")
            ax.set(xlabel='Distance to retained uncensored measurement (mm)', ylabel=label)
            ax.grid(alpha=.15)
            if metric=='coverage95':
                ax.axhline(.95, color='black', ls=':', lw=.8); ax.set_ylim(0, 1.02)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(name+': within observed-censored region only\nArea-weighted distance bins; changing populations, not a controlled spatial intervention')
        fig.savefig(OUT/f'uncertainty_distance_{name}.png', dpi=170); plt.close(fig)
    (OUT/'plot_source_fingerprints.json').write_text(json.dumps({
        str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }, indent=2))


if __name__=='__main__':
    if '--plots-only' in sys.argv:
        plot_outputs()
    else:
        main()
