from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from itertools import combinations

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src').is_dir())
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.censored_gp import cholesky_with_jitter
from src.one_step_smoothing import importance_weights
from src.smoothing_collapse import (log_likelihood_terms, marginalized_tempered_weights,
                                    weight_summary, target_summary, exact_gaussian_log_likelihood)
from src.stochastic_heat_gp import (StochasticHeatConfig, finite_step_innovation_covariance,
                                    propagate_residual_draws)
from src.thermal_posterior_physics import DEVELOPMENT_TRAJECTORIES, trajectory_path
from src.thermal_trajectory import ThermalTrajectory, SurfaceGridProjector, trajectory_catalog

BASE = ROOT / 'outputs/archive/2026-09-20_deferred/42_one_step_smoothing_pilot'
OUT = ROOT / 'outputs/archive/2026-09-20_deferred/44_smoothing_collapse_diagnostic'
POWERS = [0., .01, .05, .1, .25, .5, 1.]
FRACTIONS = [.1, .25, .5, 1.]
Q_MULTIPLIERS = [.5, 1., 2., 4.]


def run(name, index, o, check):
    destination = OUT / name
    destination.mkdir(exist_ok=True)
    saved = np.load(BASE / name / 'pilot_particles.npz')
    draws = saved['target_draws']
    truth = saved['truth_evaluation_only'].ravel()
    shape = saved['truth_evaluation_only'].shape
    top = np.argsort(truth, kind='mergesort')[-25:]
    n = int(saved['target_frame'])
    ceiling = float(saved['ceiling'])
    trajectory = ThermalTrajectory(trajectory_path(ROOT.parent / 'heat_eq_laser_trajectories', name))
    projector = SurfaceGridProjector(trajectory.surface_coordinates(), nx=int(o['nx']), ny=int(o['ny']))
    _, raw_flux = trajectory.read_history('HeatFluxZ', time_indices=[n+1], surface_only=True)
    flux = projector.project(raw_flux)[0]
    dt = float(check.lag_s)
    ambient = float(check.ambient_from_initial_observation_K)
    excess = np.maximum(draws.reshape(-1, *shape) - ambient, 0.)
    forecast = ambient + propagate_residual_draws(excess,
        dx=float(np.diff(projector.xs).mean()), dy=float(np.diff(projector.ys).mean()),
        time_step=dt, diffusivity=o['diffusivity'], cooling_rate=o['cooling_rate'])
    forecast += o['source_coupling'] * dt * np.where(flux >= o['source_flux_threshold'], flux, 0.)
    camera = saved['future_observed']
    fixed = np.zeros(shape, bool)
    fixed[::int(o['observation_stride']), ::int(o['observation_stride'])] = True
    mask = fixed | (camera == ceiling)
    positions = projector.grid_points[mask.ravel()]
    y = camera[mask]
    sat = y == ceiling
    forecast = forecast.reshape(len(draws), -1)[:, mask.ravel()]
    heat = StochasticHeatConfig(signal_sd=o['signal_sd'],
        forcing_lengthscale=check.source_lengthscale_past_only_m * o['length_multiplier'] * o['forcing_length_multiplier'],
        diffusivity=o['diffusivity'], cooling_rate=o['cooling_rate'], quadrature_order=int(o['quadrature_order']))
    q = finite_step_innovation_covariance(positions, positions, heat, dt)
    factor = cholesky_with_jitter(q, 1e-7 * max(float(np.diag(q).max()), 1.))
    seed = int(o['seed']) + 100_000 * index
    rng = np.random.default_rng(seed + 8_000_000)
    innovation = np.asarray([np.einsum('ij,kj->ik', rng.normal(size=forecast.shape), factor) for _ in range(8)])
    audit_logs = log_likelihood_terms(forecast[None] + innovation, y, sat, ceiling, o['current_noise_sd']).sum(-1)
    difference = float(np.max(abs(audit_logs - saved['loglik'])))
    np.testing.assert_allclose(audit_logs, saved['loglik'], rtol=1e-10, atol=1e-7)
    audit_weights = marginalized_tempered_weights(audit_logs, 1.)
    np.testing.assert_allclose(audit_weights, saved['pooled_weights'], rtol=1e-7, atol=1e-9)
    audit = dict(trajectory=name, loglik_max_difference=difference,
                 weights_max_difference=float(np.max(abs(audit_weights-saved['pooled_weights']))),
                 target_draws_sha256=hashlib.sha256(draws.tobytes()).hexdigest(),
                 n_future_observations=len(y), **weight_summary(audit_weights))
    print(name, 'original pilot reproduced', audit, flush=True)
    # Random nested observation subsets are fixed across Q and temperature powers.
    # They are diagnostic subsets, not changes to the canonical camera protocol.
    permutation = np.random.default_rng(seed + 901).permutation(len(y))
    selections = {str(f): permutation[:max(1, int(round(f*len(y))))] for f in FRACTIONS}
    selections['unsaturated'] = np.flatnonzero(~sat)
    selections['censored'] = np.flatnonzero(sat)
    rows, stability = [], {}
    mismatch, exact_rows = [], []
    for repeat in range(3):
        noise_rng = np.random.default_rng(seed + 8_000_000 + repeat * 10_000_000)
        innovations = np.asarray([np.einsum('ij,kj->ik', noise_rng.normal(size=forecast.shape), factor) for _ in range(8)])
        synthetic_rng = np.random.default_rng(seed + 902 + repeat)
        generator = int(synthetic_rng.integers(len(draws)))
        synthetic_z = (forecast[generator] + np.einsum('j,kj->k', synthetic_rng.normal(size=len(y)), factor)
                       + synthetic_rng.normal(0, o['current_noise_sd'], len(y)))
        synthetic_y = np.minimum(synthetic_z, ceiling)
        for qmult in Q_MULTIPLIERS:
            states = forecast[None] + np.sqrt(qmult) * innovations
            for context, target_y, target_sat in [('real', y, sat),
                                                  ('matched_model', synthetic_y, synthetic_y == ceiling)]:
                terms = log_likelihood_terms(states, target_y, target_sat, ceiling, o['current_noise_sd'])
                if not np.isfinite(terms).all():
                    raise ValueError('Nonfinite likelihood terms')
                # Exact integration separates finite-path likelihood noise from
                # genuine importance concentration for the Gaussian-only subset.
                uidx = np.flatnonzero(~target_sat)
                exact_log = exact_gaussian_log_likelihood(forecast[:, uidx], target_y[uidx],
                    qmult * np.einsum('ij,kj->ik', factor, factor)[np.ix_(uidx, uidx)], o['current_noise_sd'])
                for estimator, logvalues in [('exact', exact_log[None]), ('eight_paths', terms[..., uidx].sum(-1))]:
                    w_exact = marginalized_tempered_weights(logvalues, 1., exclude=generator)
                    _, exact_metrics = target_summary(draws, w_exact, top, truth)
                    exact_rows.append(dict(trajectory=name, repeat=repeat, context=context,
                        q_variance_multiplier=qmult, estimator=estimator, n_observations=len(uidx),
                        **weight_summary(w_exact), **exact_metrics))
                for subset, indices in selections.items():
                    # Region labels refer to real camera regions even for matched-model control.
                    if not len(indices):
                        continue
                    logs = terms[..., indices].sum(-1)
                    for power in POWERS:
                        # Exclude the synthetic generating particle in BOTH contexts.
                        w = marginalized_tempered_weights(logs, power, exclude=generator)
                        mean, metrics = target_summary(draws, w, top, truth)
                        key = (qmult, context, subset, power)
                        stability.setdefault(key, []).append((mean, metrics['top1_width_K']))
                        rows.append(dict(trajectory=name, repeat=repeat, q_variance_multiplier=qmult,
                            context=context, subset=subset, n_observations=len(indices), power=power,
                            excluded_particle=generator, **weight_summary(w), **metrics))
                # Marginal forecast checks on available unsaturated measurements only;
                # selection truncation means these are descriptive, not calibrated z-tests.
                prior_sd = np.sqrt(np.var(forecast, axis=0) + qmult*np.diag(q) + o['current_noise_sd']**2)
                error = forecast.mean(0) - target_y
                u = ~target_sat
                mismatch.append(dict(trajectory=name, repeat=repeat, context=context,
                    q_variance_multiplier=qmult, n_unsaturated=int(u.sum()),
                    unsaturated_forecast_bias_K=float(error[u].mean()),
                    unsaturated_forecast_rmse_K=float(np.sqrt(np.mean(error[u]**2))),
                    unsaturated_mean_abs_z=float(np.mean(abs(error[u])/prior_sd[u]))))
        print(name, 'repeat', repeat, 'complete', flush=True)
    st = []
    for (qmult, context, subset, power), results in stability.items():
        pairs = [x[0]-y[0] for x,y in combinations(results, 2)]
        st.append(dict(trajectory=name, q_variance_multiplier=qmult, context=context, subset=subset,
            power=power, max_repeat_top1_mean_rmse_K=max(float(np.sqrt(np.mean(e[top]**2))) for e in pairs),
            max_repeat_global_mean_rmse_K=max(float(np.sqrt(np.mean(e**2))) for e in pairs),
            top1_width_range_K=max(x[1] for x in results)-min(x[1] for x in results)))
    pd.DataFrame(rows).to_csv(destination/'diagnostics.csv', index=False)
    pd.DataFrame(st).to_csv(destination/'repeat_stability.csv', index=False)
    pd.DataFrame(mismatch).to_csv(destination/'forecast_mismatch.csv', index=False)
    pd.DataFrame(exact_rows).to_csv(destination/'exact_unsaturated_check.csv', index=False)
    pd.DataFrame([audit]).to_csv(destination/'equality_audit.csv', index=False)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    o = pd.read_csv(BASE/'fixed_configuration.csv').iloc[0].to_dict()
    checks = pd.read_csv(BASE/'checks.csv').set_index('trajectory')
    for index, trajectory in enumerate(trajectory_catalog(ROOT.parent/'heat_eq_laser_trajectories')):
        if trajectory.name in DEVELOPMENT_TRAJECTORIES:
            run(trajectory.name, index, o, checks.loc[trajectory.name])
    for filename in ['diagnostics', 'repeat_stability', 'forecast_mismatch', 'equality_audit', 'exact_unsaturated_check']:
        pd.concat([pd.read_csv(OUT/name/f'{filename}.csv') for name in DEVELOPMENT_TRAJECTORIES]).to_csv(OUT/f'{filename}.csv', index=False)
    data = pd.read_csv(OUT/'diagnostics.csv')
    assert np.isfinite(data.select_dtypes(include='number').to_numpy()).all()
    means = data.groupby(['trajectory','q_variance_multiplier','context','subset','power']).mean(numeric_only=True).reset_index()
    means.to_csv(OUT/'summary.csv', index=False)
    stability = pd.read_csv(OUT/'repeat_stability.csv')
    rows = data[(data.context=='real') & (data.subset=='1.0') & (data.power==1) & (data.q_variance_multiplier==1)]
    repeat_rows = stability[(stability.context=='real') & (stability.subset=='1.0') & (stability.power==1) & (stability.q_variance_multiplier==1)]
    passed = bool((rows.ess>=120).all() and (rows.max_weight<=.02).all()
        and (repeat_rows.max_repeat_top1_mean_rmse_K<=.25).all()
        and (repeat_rows.max_repeat_global_mean_rmse_K<=.1).all())
    (OUT/'gate.json').write_text(json.dumps(dict(full_likelihood_reliability_passed=passed,
        gate_source='Experiment 42 ESS/max-weight/repeat thresholds, evaluated over three repeats',
        smc_implemented=False, heldout_evaluated=False, longer_lags_evaluated=False),indent=2))
    fig, axes = plt.subplots(2,3,figsize=(13,7), constrained_layout=True)
    for col, name in enumerate(DEVELOPMENT_TRAJECTORIES):
        for context, linestyle in [('real','-'), ('matched_model','--')]:
            part = means[(means.trajectory==name)&(means.q_variance_multiplier==1)&(means.context==context)&(means.subset=='1.0')]
            axes[0,col].plot(part.power, part.ess, linestyle, marker='o', label=context)
            part = means[(means.trajectory==name)&(means.q_variance_multiplier==1)&(means.context==context)&(means.power==1)&means.subset.isin([str(f) for f in FRACTIONS])].sort_values('n_observations')
            axes[1,col].plot(part.n_observations, part.ess, linestyle, marker='o', label=context)
        for ax in axes[:,col]:
            ax.set_yscale('log'); ax.axhline(120,color='gray',linestyle=':'); ax.set_ylabel('Mean particle ESS'); ax.legend()
        axes[0,col].set_title(name); axes[0,col].set_xlabel('Likelihood power (full observation set)')
        axes[1,col].set_xlabel('Number of future observations (power = 1)')
    fig.savefig(OUT/'collapse_decomposition.png',dpi=180); plt.close(fig)
    pd.DataFrame([{**o,'q_variance_multipliers':str(Q_MULTIPLIERS),'powers':str(POWERS),
        'dimension_fractions':str(FRACTIONS),'repeats':3,'paths_per_particle':8,
        'synthetic_generator_excluded':True,'heldout_evaluated':False}]).to_csv(OUT/'fixed_configuration.csv',index=False)
    paths = [Path(__file__), ROOT/'src/smoothing_collapse.py', BASE/'fixed_configuration.csv',
             *[BASE/name/'pilot_particles.npz' for name in DEVELOPMENT_TRAJECTORIES]]
    (OUT/'source_fingerprints.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))


if __name__ == '__main__':
    main()
