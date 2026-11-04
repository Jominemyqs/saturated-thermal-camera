"""Output 59: isolate missing conditional correlations in hybrid predecessors.

No current GP update, parameter fit, full-latent predecessor change or smoother.
"""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.censored_gp import RBFConfig, sample_censored_ess_fast
from src.matched_projection import MatchedHeatOperator
from src.native_mask_pilot import spread_nodes
from src.previous_correlation_audit import (
    normal_chains, product, propagated_variances, psd_factor,
    recover_conditional_means, residual_covariance,
)
from src.mcmc_diagnostics import gelman_rubin
from src.stochastic_heat_gp import StochasticHeatConfig, finite_step_innovation_covariance

BASE = ROOT / 'outputs/by_experiment/52_native_higher_ceiling'
P56 = ROOT / 'outputs/by_experiment/56_within_step_integration'
DEFAULT_OUT = ROOT / 'outputs/by_experiment/59_previous_conditional_correlation_audit'
NAMES = ['DiagonalScanPath_8', 'SpiralScanPath_13']


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def diagnose(chains, ids, kind):
    if not hasattr(scipy.signal, 'gaussian'):
        scipy.signal.gaussian = scipy.signal.windows.gaussian
    import arviz as az
    data = az.from_dict(posterior={'v': chains})
    return pd.DataFrame(dict(node=ids, variable=kind,
        classical_rhat=gelman_rubin(chains),
        rank_rhat=az.rhat(data, method='rank')['v'].values,
        ess_bulk=az.ess(data, method='bulk')['v'].values,
        ess_tail=az.ess(data, method='tail')['v'].values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        if os.environ.get(key) != '1':
            raise RuntimeError(f'Set {key}=1 to preserve the baseline runtime protocol')
    if not hasattr(scipy.signal, 'gaussian'):
        scipy.signal.gaussian = scipy.signal.windows.gaussian
    import arviz as az
    import xarray
    start = perf_counter()
    o = pd.read_csv(BASE/'frozen_configuration.csv').iloc[0].to_dict()
    metadata = pd.read_csv(BASE/'verification.csv')
    provenance = {r['trajectory']: r for r in json.loads((P56/'gradient_provenance.json').read_text())}
    assert o['truncated_sampler'] == 'hmc'
    paths = [Path(__file__), ROOT/'src/previous_correlation_audit.py',
             ROOT/'tests/test_previous_correlation_audit.py', ROOT/'src/censored_gp.py',
             ROOT/'src/truncated_gaussian.py', ROOT/'src/matched_projection.py',
             ROOT/'src/stochastic_heat_gp.py', ROOT/'src/thermal_posterior_physics.py',
             BASE/'frozen_configuration.csv', BASE/'verification.csv',
             P56/'gradient_provenance.json']
    for name in NAMES:
        paths.extend(BASE/'by_trajectory'/name/f'level_{level}.npz' for level in range(3))
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    out.mkdir(parents=True)
    write_json(out/'source_fingerprints.json', fingerprints)
    write_json(out/'frozen_configuration.json', dict(experiment=59,
        driver=str(Path(__file__).relative_to(ROOT)),
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        trajectories=NAMES, levels=['reference', 320, 325], inherited_options=o,
        change='Conditional residual off-diagonals at previous OBSERVED-CENSORED nodes only',
        pinned_unsaturated=True, physics_mean='Cached mean fixed; do not feed new empirical means into forecast',
        analytic_comparison='Cov(empirical conditional means) + C_cond versus its diagonal-residual counterpart',
        original_hmc_seed_offset=1000003, chain_seed_stride=10000,
        residual_repeat_seed_offsets=[0, 2000003, 4000007],
        independent_hmc_repeats=False, legacy_replay='First complete original HMC chain at each case',
        current_gp_inference=False, new_parameters=False, mask_likelihoods=False,
        evaluation='Native physical-area weights; fixed CURRENT hottest 1% area; truth evaluation only',
        numerical_checks=dict(replay_max_abs_K=1e-10, propagation_max_abs_K=1e-10,
                              saved_variance_max_abs_K2=1e-9, psd_tolerance=1e-10),
        expansion='Stop after covariance audit, no current update regardless of effect size',
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                         arviz=az.__version__, xarray=xarray.__version__,
                         diagnostics_pythonpath=os.environ.get('PYTHONPATH'),
                         threads={k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}),
        scope_notes='Older grid protocol/top25 labels are historical; this native audit uses cached area weights.'))
    rows, mcrows, checks, runtimes, sizes, diagnostic_rows = [], [], [], [], [], []
    for name in NAMES:
        basis = None
        for level in range(3):
            tick = perf_counter()
            path = BASE/'by_trajectory'/name/f'level_{level}.npz'
            keys = ['points','triangles','area_weights','hot_area_weights','truth','previous_camera',
                    'previous_mean','previous_draws','ceiling','numeric_candidates','ambient',
                    'physics_mean','C_inherited_variance','C_prior_variance']
            with np.load(path) as f:
                c = {k: f[k] for k in keys}
            meta = metadata[(metadata.trajectory == name) & (metadata.level == level) & (metadata.model == 'C')].iloc[0]
            assert not bool(meta.previous_retried)
            assert digest(c['previous_draws']) == meta.previous_draws_sha256
            xy = c['points']; n = len(xy); ceiling = float(c['ceiling'])
            sat = c['previous_camera'] == ceiling; ids = np.flatnonzero(sat)
            included = c['numeric_candidates'] | sat
            points = np.column_stack([xy, np.full(n, provenance[name]['previous_time_s'])])
            obs = dict(x_obs=points[included], y_obs=c['previous_camera'][included],
                       sat_mask=sat[included], x_pred=points[sat], threshold=ceiling)
            cfg = RBFConfig(mean_temp=float(c['ambient']), signal_sd=o['signal_sd'],
                            lengthscale=float(meta.lengthscale_m), noise_sd=o['previous_noise_sd'])
            chains, samples = int(o['previous_chains']), int(o['previous_samples_per_chain'])
            seed = int(meta.previous_seed)
            previous_draws = c.pop('previous_draws')
            np.testing.assert_array_equal(previous_draws[:, ~sat],
                np.broadcast_to(c['previous_camera'][~sat], (len(previous_draws), (~sat).sum())))
            old = previous_draws[:, sat].copy()
            np.testing.assert_array_equal(previous_draws.mean(axis=0), c['previous_mean'])
            del previous_draws
            covariance_start = perf_counter()
            cc = residual_covariance(obs, cfg)
            factor, factor_checks = psd_factor(cc)
            means, z = recover_conditional_means(old, cc, seed=seed, chains=chains,
                                                samples=samples, truncated_sampler=o['truncated_sampler'])
            conditional_seconds = perf_counter() - covariance_start
            # Independent call to the original sampler checks the cache/RNG reconstruction.
            replay_start = perf_counter()
            replay = sample_censored_ess_fast(obs, cfg, n_samples=samples,
                burn_in=int(o['previous_burn_in']), thin=int(o['previous_thin']),
                seed=seed, truncated_sampler='hmc')
            replay_error = float(abs(replay[4]-old[:samples]).max())
            np.testing.assert_allclose(replay[4], old[:samples], atol=1e-10, rtol=0)
            np.testing.assert_allclose(replay[0], means[:samples].mean(axis=0), atol=1e-10, rtol=0)
            replay_seconds = perf_counter() - replay_start
            physics = dict(dt=provenance[name]['dt_s'], diffusivity=o['diffusivity'], cooling_rate=o['cooling_rate'])
            basis_start = perf_counter()
            if basis is None:
                mesh = SimpleNamespace(points=xy, triangles=c['triangles'])
                op = MatchedHeatOperator(mesh, np.linspace(xy[:, 0].min(), xy[:, 0].max(), int(o['nx'])),
                                         np.linspace(xy[:, 1].min(), xy[:, 1].max(), int(o['ny'])))
                reference_ids = ids.copy()
                basis = np.empty((n, len(ids)))
                for first in range(0, len(ids), 32):
                    chunk = ids[first:first+32]
                    unit = np.zeros((len(chunk), n)); unit[np.arange(len(chunk)), chunk] = 1.
                    basis[:, first:first+len(chunk)] = op.propagate(unit, 'native', **physics).T
                reference_points = xy.copy()
            np.testing.assert_array_equal(xy, reference_points)
            positions = np.searchsorted(reference_ids, ids)
            np.testing.assert_array_equal(reference_ids[positions], ids)
            a = basis[:, positions]
            basis_seconds = perf_counter() - basis_start
            direct = np.zeros((4, n)); direct[:, sat] = old[:4] - c['previous_mean'][sat]
            propagation_error = float(abs(op.propagate(direct, 'native', **physics) -
                                          product(direct[:, sat], a.T)).max())
            assert propagation_error < 1e-10
            cached_variance = np.var(product(old-c['previous_mean'][sat], a.T), axis=0, ddof=1)
            variance_error = float(abs(cached_variance-c['C_inherited_variance']).max())
            assert variance_error < 1e-9
            shared, independent, joint = propagated_variances(a, means, cc)
            hcfg = StochasticHeatConfig(signal_sd=o['signal_sd'],
                forcing_lengthscale=cfg.lengthscale*o['forcing_length_multiplier'],
                diffusivity=o['diffusivity'], cooling_rate=o['cooling_rate'],
                quadrature_order=int(o['quadrature_order']))
            q = float(finite_step_innovation_covariance(points[:1], points[:1], hcfg, physics['dt'])[0, 0])
            np.testing.assert_allclose(c['C_prior_variance']-c['C_inherited_variance'], q, atol=1e-12, rtol=0)
            regions = dict(whole_area=c['area_weights'], hot1pct_area=c['hot_area_weights'],
                           previous_censored=c['area_weights']*sat,
                           previous_unsaturated=c['area_weights']*~sat)
            common = dict(trajectory=name, level=level, ceiling_K=ceiling)
            for region, w in regions.items():
                def avg(v): return float(np.sum(w * v) / w.sum())
                rows.append(dict(**common, region=region, nodes=int((w > 0).sum()),
                    shared_inherited_variance_K2=avg(shared),
                    residual_independent_variance_K2=avg(independent), residual_joint_variance_K2=avg(joint),
                    inherited_independent_variance_K2=avg(shared+independent),
                    inherited_joint_variance_K2=avg(shared+joint), Q_variance_K2=q,
                    prior_independent_variance_K2=avg(q+shared+independent),
                    prior_joint_variance_K2=avg(q+shared+joint),
                    mean_prior_independent_sd_K=avg(np.sqrt(q+shared+independent)),
                    mean_prior_joint_sd_K=avg(np.sqrt(q+shared+joint)),
                    prior_variance_change_percent=100*avg(joint-independent)/avg(q+shared+independent),
                    cached_inherited_variance_K2=avg(c['C_inherited_variance'])))
            folder = out/'by_trajectory'/name; folder.mkdir(parents=True, exist_ok=True)
            joint_draws = means + product(z, factor.T)
            # Repeats change Gaussian residual RNG ONLY, not the MCMC inequality samples.
            for repeat, offset in enumerate([0, 2000003, 4000007]):
                zr = z if repeat == 0 else normal_chains(seed+offset, chains, samples, len(ids))
                for arm, residual in [('independent', zr*np.sqrt(np.diag(cc))), ('joint', product(zr, factor.T))]:
                    draw = means + residual
                    propagated = product(draw-draw.mean(axis=0), a.T)
                    empirical = np.var(propagated, axis=0, ddof=1)
                    expected = shared + (independent if arm == 'independent' else joint)
                    for region, w in regions.items():
                        mcrows.append(dict(**common, region=region, repeat=repeat, arm=arm,
                            empirical_inherited_variance_K2=float(np.sum(w*empirical)/w.sum()),
                            analytic_inherited_variance_K2=float(np.sum(w*expected)/w.sum()),
                            empirical_minus_analytic_K2=float(np.sum(w*(empirical-expected))/w.sum())))
            monitor = spread_nodes(xy, sat, min(128, len(ids)))
            midx = np.searchsorted(ids, monitor)
            for arm, draw in [('cached_independent', old), ('paired_joint', joint_draws), ('conditional_mean', means)]:
                diag = diagnose(draw[:, midx].reshape(chains, samples, len(midx)), monitor, arm)
                for key, val in common.items(): diag[key] = val
                diagnostic_rows.append(diag)
            np.savez_compressed(folder/f'level_{level}.npz', points=xy, previous_censored_ids=ids,
                previous_observation_ids=np.flatnonzero(included), numerical_ids=np.flatnonzero(included & ~sat),
                frozen_previous_mean=c['previous_mean'], frozen_physics_mean=c['physics_mean'],
                conditional_means=means.reshape(chains, samples, len(ids)), conditional_covariance=cc,
                joint_censored_draws=joint_draws.reshape(chains, samples, len(ids)),
                shared_variance=shared, independent_residual_variance=independent,
                joint_residual_variance=joint, Q_variance=q, monitor_ids=monitor,
                previous_marginal_variance=np.var(means, axis=0, ddof=1)+np.diag(cc),
                previous_covariance_joint_monitor=np.cov(means[:, midx].T)+cc[np.ix_(midx, midx)],
                previous_covariance_independent_monitor=np.cov(means[:, midx].T)+np.diag(np.diag(cc)[midx]))
            checks.append(dict(**common, old_draw_hash_matches=True, pinned_values_identical=True,
                legacy_chain_max_abs_K=replay_error, propagation_max_abs_K=propagation_error,
                cached_inherited_variance_max_abs_K2=variance_error,
                conditional_mean_mc_difference_from_cached_mean_max_K=float(abs(means.mean(axis=0)-c['previous_mean'][sat]).max()),
                joint_empirical_mean_difference_max_K=float(abs(joint_draws.mean(axis=0)-c['previous_mean'][sat]).max()),
                mean_used_for_forecast='unchanged cached mean', **factor_checks))
            sizes.append(dict(**common, native_nodes=n, previous_numerical=int(np.sum(included & ~sat)),
                previous_censored=len(ids), previous_pinned_unsaturated=int((~sat).sum()),
                residual_covariance_MiB=cc.nbytes/2**20, propagation_matrix_MiB=a.nbytes/2**20,
                full_native_covariance_if_expanded_MiB=n*n*8/2**20,
                current_update_run=False))
            runtimes.append(dict(**common, conditional_covariance_factor_seconds=conditional_seconds,
                legacy_one_chain_replay_seconds=replay_seconds, propagation_basis_seconds=basis_seconds,
                case_seconds=perf_counter()-tick,
                process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
            for file, data in [('variance_summary', rows), ('residual_repeat_comparison', mcrows),
                               ('equality_checks', checks), ('runtime', runtimes), ('computational_size', sizes)]:
                pd.DataFrame(data).to_csv(out/f'{file}.csv', index=False)
            pd.concat(diagnostic_rows).to_csv(out/'sampling_diagnostics.csv', index=False)
            print(name, level, 'completed', round(runtimes[-1]['case_seconds'], 2), 's', flush=True)
    summary = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for col, name in enumerate(NAMES):
        for region, label, color in [('whole_area','Whole area','#0072B2'), ('hot1pct_area','Current hot 1% area','#D55E00')]:
            d = summary[(summary.trajectory == name) & (summary.region == region)]
            axes[0, col].plot(d.ceiling_K, d.prior_variance_change_percent, 'o-', color=color, label=label)
            axes[1, col].plot(d.ceiling_K, d.residual_independent_variance_K2, 'o--', color=color, label=label+' independent')
            axes[1, col].plot(d.ceiling_K, d.residual_joint_variance_K2, 's-', color=color, label=label+' joint')
        axes[0, col].set_title(name); axes[0, col].axhline(0, color='#999999', lw=.7)
        axes[0, col].set_ylabel('Change in area-mean total PRIOR variance (%)')
        axes[1, col].set_ylabel('Propagated conditional residual variance (K$^2$)')
        for ax in axes[:, col]:
            ax.set_xlabel('Camera ceiling (K)'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle('Same hybrid predecessor information and frozen forecast mean; no current GP update')
    fig.savefig(out/'covariance_summary.png', dpi=180); plt.close(fig)
    # Check that every pre-existing input stayed untouched throughout execution.
    assert all(sha(ROOT/p) == h for p, h in fingerprints.items())
    write_json(out/'verification.json', dict(cases=6, input_fingerprints_unchanged=True,
        marginal_previous_variances_identical_analytically=True, unsaturated_latent_uncertainty_added=False,
        current_posterior_inference=False, fitting=False, baseline_replaced=False,
        seconds=perf_counter()-start, peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        limitations=['Analytic residual effect, conditional-mean covariance uses cached MCMC draws',
                     'Residual repeats are not independent HMC repeats',
                     'Pinned unsaturated values unchanged; not a complete latent posterior',
                     'No coverage/CRPS improvement is established without a current update']))


if __name__ == '__main__':
    main()
