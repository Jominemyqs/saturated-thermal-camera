"""C-only current-observation retention control; all previous-state inputs cached."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import hashlib
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.linalg.blas import dgemm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.thermal_trajectory import ThermalTrajectory, trajectory_catalog
from src.matched_projection import MatchedHeatOperator, weighted_metrics
from src.stochastic_heat_gp import StochasticHeatConfig, finite_step_innovation_covariance
from src.pseudo_censoring_runner import _current_prediction
from src.native_ceiling_diagnostics import retained_reference_observations

BASE = ROOT/'outputs/by_experiment/52_native_higher_ceiling'
OUT = ROOT/'outputs/by_experiment/53_native_observation_retention'
DATA = ROOT.parent/'heat_eq_laser_trajectories'
NAMES = ['DiagonalScanPath_8', 'SpiralScanPath_13']
SUFFIXES = ['mean', 'sd', 'lower', 'upper', 'draws']


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def gram(a, b):
    return dgemm(alpha=1., a=np.asfortranarray(a), b=np.asfortranarray(b), trans_b=True)


def main():
    for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
        if os.environ.get(key) != '1':
            raise RuntimeError(f'Set {key}=1 for exact frozen replay')
    OUT.mkdir(parents=True, exist_ok=True)
    o = pd.read_csv(ROOT/'outputs/by_experiment/49_matched_native_grid_gp/frozen_configuration.csv').iloc[0].to_dict()
    assert o == pd.read_csv(BASE/'frozen_configuration.csv').iloc[0].to_dict()
    pd.DataFrame([o]).to_csv(OUT/'frozen_configuration.csv', index=False)
    checks52 = pd.read_csv(BASE/'verification.csv')
    results, checks, peaks, region_changes = [], [], [], []
    input_files = set()
    for r in trajectory_catalog(DATA):
        if r.name not in NAMES:
            continue
        trajectory = ThermalTrajectory(r.xdmf_path)
        reference_path = BASE/'by_trajectory'/r.name/'level_0.npz'
        input_files.add(reference_path)
        with np.load(reference_path) as reference:
            reference_camera = reference['camera']; reference_ceiling = float(reference['ceiling'])
        for level in [1, 2]:
            start = perf_counter()
            cache_path = BASE/'by_trajectory'/r.name/f'level_{level}.npz'
            input_files.add(cache_path)
            with np.load(cache_path) as cache:
                d = {key: cache[key] for key in cache.files}
            xy = d['points']; N = len(xy); ceiling = float(d['ceiling'])
            saved = checks52[(checks52.trajectory==r.name)&(checks52.level==level)&(checks52.model=='C')].iloc[0]
            seed = int(saved.current_seed)
            points = np.column_stack([xy, np.full(N, trajectory.times[-1])])
            mesh = SimpleNamespace(points=xy, triangles=d['triangles'])
            xs = np.linspace(xy[:, 0].min(), xy[:, 0].max(), int(o['nx']))
            ys = np.linspace(xy[:, 1].min(), xy[:, 1].max(), int(o['ny']))
            op = MatchedHeatOperator(mesh, xs, ys)
            physics = dict(dt=float(trajectory.times[-1]-trajectory.times[-2]),
                diffusivity=o['diffusivity'], cooling_rate=o['cooling_rate'])
            hcfg = StochasticHeatConfig(signal_sd=o['signal_sd'], forcing_lengthscale=float(saved.lengthscale_m)*o['forcing_length_multiplier'],
                diffusivity=o['diffusivity'], cooling_rate=o['cooling_rate'], quadrature_order=int(o['quadrature_order']))
            features = op.propagate(d['previous_draws']-d['previous_mean'], 'native', **physics).T/np.sqrt(len(d['previous_draws'])-1)
            inherited = np.sum(features**2, axis=1)
            diag = float(finite_step_innovation_covariance(points[:1], points[:1], hcfg, physics['dt'])[0, 0])+inherited
            np.testing.assert_array_equal(inherited, d['C_inherited_variance'])
            np.testing.assert_array_equal(diag, d['C_prior_variance'])
            assert digest(d['previous_draws']) == saved.previous_draws_sha256
            retained, numeric, saturated = retained_reference_observations(reference_camera, reference_ceiling,
                d['camera'], ceiling, d['numeric_candidates'])
            oldids = d['observation_ids']
            np.testing.assert_array_equal(retained, np.flatnonzero(d['numeric_candidates'] | (reference_camera==reference_ceiling)))
            locations = np.searchsorted(retained, oldids)
            np.testing.assert_array_equal(retained[locations], oldids)
            newmask = np.zeros(N, bool); newmask[np.setdiff1d(retained, oldids)] = True
            assert not np.any(newmask & saturated)
            common = dict(trajectory=r.name, level=level, ceiling_K=ceiling,
                previous_draws_sha256=digest(d['previous_draws']), mean_sha256=digest(d['physics_mean']),
                covariance_features_sha256=digest(features), prior_variance_sha256=digest(diag),
                camera_sha256=digest(d['camera']), current_signal_sha256=digest(d['current_signal']),
                hot_area_sha256=digest(d['hot_area_weights']), current_seed=seed,
                source_cache=str(cache_path.relative_to(ROOT)))
            arrays = {key: d[key] for key in ['points', 'triangles', 'truth', 'area_weights', 'hot_area_weights',
                'camera', 'ceiling', 'physics_mean', 'C_prior_variance']}
            arrays.update(old_observation_ids=oldids, retained_observation_ids=retained, newly_retained=newmask)
            baseline_koo = baseline_kpo = None
            for policy, ids in [('existing', oldids), ('retained', retained)]:
                tick = perf_counter()
                foo = features[ids]
                koo = finite_step_innovation_covariance(points[ids], points[ids], hcfg, physics['dt'])+gram(foo, foo)
                kpo = finite_step_innovation_covariance(points, points[ids], hcfg, physics['dt'])+gram(features, foo)
                overlap_max = 0.
                if policy=='existing':
                    baseline_koo, baseline_kpo = koo.copy(), kpo.copy()
                else:
                    for a, b in [(koo[np.ix_(locations, locations)], baseline_koo), (kpo[:, locations], baseline_kpo)]:
                        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-13)
                        overlap_max = max(overlap_max, float(np.max(abs(a-b))))
                obs = dict(x_obs=points[ids], y_obs=d['camera'][ids], sat_mask=saturated[ids],
                    saturated_full=saturated, x_pred=points, threshold=ceiling)
                blocks = dict(prediction_mean=d['physics_mean'], observation_mean=d['physics_mean'][ids],
                    observed_covariance=koo, pred_observed_covariance=kpo, prediction_variance=diag)
                pred, cc = _current_prediction(obs, blocks, options=o, seed=seed)
                if policy=='existing':
                    for suffix, values in zip(SUFFIXES, pred):
                        np.testing.assert_array_equal(values, d['C_'+suffix])
                for suffix, values in zip(SUFFIXES, pred):
                    arrays[policy+'_'+suffix] = values
                results.extend(dict(trajectory=r.name, level=level, ceiling_K=ceiling, policy=policy, **row)
                    for row in weighted_metrics(pred, d['truth'], d['area_weights'], float(d['ambient']), saturated))
                error = pred[0]-d['truth']; w = d['area_weights']*newmask
                region_changes.append(dict(trajectory=r.name, level=level, ceiling_K=ceiling, policy=policy,
                    nodes=int(newmask.sum()), bias_K=float(np.sum(w*error)/w.sum()),
                    rmse_K=float(np.sqrt(np.sum(w*error**2)/w.sum()))))
                node = int(np.argmax(d['truth']))
                peaks.append(dict(trajectory=r.name, level=level, ceiling_K=ceiling, policy=policy,
                    true_peak_K=float(d['truth'][node]), physics_at_true_peak_K=float(d['physics_mean'][node]),
                    prediction_at_true_peak_K=float(pred[0][node]), lower95_K=float(pred[2][node]), upper95_K=float(pred[3][node]),
                    predicted_max_K=float(pred[0].max())))
                checks.append(dict(**common, policy=policy, numeric_count=int(np.sum(~saturated[ids])),
                    censored_count=int(np.sum(saturated[ids])), total_count=len(ids),
                    added_numeric_count=int(newmask.sum()) if policy=='retained' else 0,
                    covariance_overlap_max_abs=overlap_max, rhat_max=cc['rhat_max'],
                    censored_rhat_max=cc['selected_rhat_max'], retried=cc['current_retried'],
                    reference_replay_exact=policy=='existing', seconds=perf_counter()-tick))
                print(r.name, ceiling, policy, 'Rhat', cc['rhat_max'], 'seconds', perf_counter()-tick, flush=True)
            assert digest(d['physics_mean'])==common['mean_sha256']
            assert digest(features)==common['covariance_features_sha256']
            folder = OUT/'by_trajectory'/r.name; folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder/f'level_{level}.npz', **arrays)
            print('case total seconds', perf_counter()-start, flush=True)
            for name, rows in [('results', results), ('verification', checks), ('peak_summary', peaks), ('newly_retained_region', region_changes)]:
                pd.DataFrame(rows).to_csv(OUT/(name+'.csv'), index=False)
    frame = pd.DataFrame(results)
    metrics = ['relative_L2', 'rmse_K', 'mae_K', 'bias_K', 'crps_K', 'coverage95', 'width95_K', 'peak_absolute_error_K']
    paired = []
    for keys, group in frame.groupby(['trajectory', 'level', 'ceiling_K', 'region']):
        g = group.set_index('policy')
        paired.append(dict(zip(['trajectory', 'level', 'ceiling_K', 'region'], keys),
            **{m:float(g.loc['retained', m]-g.loc['existing', m]) for m in metrics}))
    pd.DataFrame(paired).to_csv(OUT/'paired_retained_minus_existing.csv', index=False)
    for _, group in pd.DataFrame(checks).groupby(['trajectory', 'level']):
        for col in ['previous_draws_sha256', 'mean_sha256', 'covariance_features_sha256', 'prior_variance_sha256', 'camera_sha256', 'hot_area_sha256', 'current_seed']:
            assert group[col].nunique()==1, col
    files = [Path(__file__), ROOT/'src/native_ceiling_diagnostics.py', ROOT/'src/matched_projection.py',
        ROOT/'src/pseudo_censoring_runner.py', ROOT/'src/stochastic_heat_gp.py', BASE/'frozen_configuration.csv']
    (OUT/'source_fingerprints.json').write_text(json.dumps({str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, indent=2))
    (OUT/'input_cache_fingerprints.json').write_text(json.dumps({str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(input_files)}, indent=2))
    plot_sections()


def plot_sections():
    source = ROOT/'outputs/by_experiment/49_matched_native_grid_gp/native_diagnostics'
    for name in NAMES:
        sections = pd.read_csv(source/f'cross_sections_{name}.csv')
        sections = sections[sections.model=='native_C']
        data = [np.load(OUT/'by_trajectory'/name/f'level_{level}.npz') for level in [1, 2]]
        for section, nodes in sections.groupby('section', sort=False):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
            ids = nodes.node.to_numpy(int); x = nodes.along_mm.to_numpy()
            for col, d in enumerate(data):
                for row, policy in enumerate(['existing', 'retained']):
                    ax = axes[row, col]
                    ax.fill_between(x, d[policy+'_lower'][ids], d[policy+'_upper'][ids], alpha=.25, color='#0072B2', label='Pointwise 95% interval')
                    ax.plot(x, d['truth'][ids], color='black', label='Native truth')
                    ax.plot(x, d['physics_mean'][ids], '--', color='#D55E00', label='Identical physics forecast')
                    ax.plot(x, d[policy+'_mean'][ids], color='#0072B2', label='C posterior')
                    ax.axhline(float(d['ceiling']), ls=':', color='#777777', label='Ceiling')
                    ax.set(title=f"{policy.capitalize()} observations | {float(d['ceiling']):.0f} K", xlabel='Distance from true peak (mm)', ylabel='Temperature (K)')
                    ax.grid(alpha=.15)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(name+' | '+section.replace('flux', 'gradient')+'\nFrozen high-ceiling previous posterior, mean and covariance; only current observation retention changes\nSame native section nodes; connecting intervals are visual guides', fontsize=11)
            fig.savefig(OUT/f'cross_sections_{name}_{section.split()[0].lower()}.png', dpi=170); plt.close(fig)
        for d in data:
            d.close()


if __name__=='__main__':
    main()
