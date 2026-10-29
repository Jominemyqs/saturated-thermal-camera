"""Budgeted non-saturation-mask pilot; cached native C prior, current update only."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.linalg.blas import dgemm
from scipy.spatial import cKDTree
from scipy.special import ndtr
import scipy.signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.native_mask_pilot import spread_nodes, sample_mixed_noisy_blocks
from src.matched_projection import MatchedHeatOperator
from src.stochastic_heat_gp import StochasticHeatConfig, finite_step_innovation_covariance
from src.pseudo_censoring_runner import prediction_from_draws
from src.mcmc_diagnostics import chain_diagnostics
from src.metrics import empirical_crps
from src.thermal_trajectory import ThermalTrajectory, trajectory_catalog

BASE = ROOT/'outputs/by_experiment/52_native_higher_ceiling'
RETAINED = ROOT/'outputs/by_experiment/53_native_observation_retention'
OUT = ROOT/'outputs/by_experiment/54_native_non_saturation_mask'
NAMES = ['DiagonalScanPath_8', 'SpiralScanPath_13']
SUFFIXES = ['mean', 'sd', 'lower', 'upper', 'draws']
REPEAT_OFFSET = 7_000_019
GATE = dict(rank_rhat_max=1.01, ess_bulk_min=400., ess_tail_min=400.,
            arm_seconds_max=300., process_peak_rss_MiB_max=4096.)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def gram(a, b):
    return dgemm(alpha=1., a=np.asfortranarray(a), b=np.asfortranarray(b), trans_b=True)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(2**20 if sys.platform=='darwin' else 1024)


def arviz_diagnostics(chains, ids, kind):
    # ArviZ 0.17.1 imports this old plotting alias; diagnostics themselves are
    # unmodified. scipy >=1.13 moved it to signal.windows.
    if not hasattr(scipy.signal, 'gaussian'):
        scipy.signal.gaussian = scipy.signal.windows.gaussian
    import arviz as az
    data = az.from_dict(posterior={'v': chains})
    rank = az.rhat(data, method='rank')['v'].values
    bulk = az.ess(data, method='bulk')['v'].values
    tail = az.ess(data, method='tail')['v'].values
    mcse = az.mcse(data, method='mean')['v'].values
    return pd.DataFrame(dict(node=ids, variable=kind, rank_rhat=rank,
                             ess_bulk=bulk, ess_tail=tail, mean_mcse_K=mcse))


def select_locations(name):
    with np.load(BASE/'by_trajectory'/name/'level_0.npz') as d:
        xy = d['points']; weights = d['area_weights']
        reference_sat = d['camera']==d['ceiling']
        baseline = np.flatnonzero(d['numeric_candidates'] | reference_sat)
        pool = np.ones(len(xy), bool); pool[baseline] = False
        assert not reference_sat[pool].any()
        selected = spread_nodes(xy, pool, 256)
        nearest = cKDTree(xy[selected]).query(xy[pool])[1]
        represented = np.bincount(nearest, weights=weights[pool], minlength=len(selected))
        for level in [0, 1, 2]:
            with np.load(BASE/'by_trajectory'/name/f'level_{level}.npz') as other:
                np.testing.assert_array_equal(other['current_signal'], d['current_signal'])
                assert np.all(other['camera'][pool]<other['ceiling'])
        folder = OUT/'by_trajectory'/name; folder.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(dict(node=selected, x_m=xy[selected, 0], y_m=xy[selected, 1],
            selected_lumped_area_m2=weights[selected], pool_voronoi_area_m2=represented))
        path = folder/'selected_locations.csv'
        if path.exists():
            np.testing.assert_array_equal(pd.read_csv(path).node, selected)
        else:
            table.to_csv(path, index=False)
        np.savez_compressed(folder/'selection.npz', baseline_ids=baseline, added_ids=selected,
                            pool=pool, points=xy, area_weights=weights)
        return baseline, selected, pool


def inputs(name, level, options):
    path = BASE/'by_trajectory'/name/f'level_{level}.npz'
    with np.load(path) as cache:
        d = {k:cache[k] for k in cache.files if not k.startswith('B_')}
    check = pd.read_csv(BASE/'verification.csv')
    saved = check[(check.trajectory==name)&(check.level==level)&(check.model=='C')].iloc[0]
    assert digest(d['previous_draws'])==saved.previous_draws_sha256
    record = next(r for r in trajectory_catalog(ROOT.parent/'heat_eq_laser_trajectories') if r.name==name)
    trajectory = ThermalTrajectory(record.xdmf_path)
    xy = d['points']; mesh = SimpleNamespace(points=xy, triangles=d['triangles'])
    xs = np.linspace(xy[:, 0].min(), xy[:, 0].max(), int(options['nx']))
    ys = np.linspace(xy[:, 1].min(), xy[:, 1].max(), int(options['ny']))
    op = MatchedHeatOperator(mesh, xs, ys)
    physics = dict(dt=float(trajectory.times[-1]-trajectory.times[-2]), diffusivity=options['diffusivity'],
                   cooling_rate=options['cooling_rate'])
    cfg = StochasticHeatConfig(signal_sd=options['signal_sd'],
        forcing_lengthscale=float(saved.lengthscale_m)*options['forcing_length_multiplier'],
        diffusivity=physics['diffusivity'], cooling_rate=physics['cooling_rate'],
        quadrature_order=int(options['quadrature_order']))
    tick = perf_counter()
    h = op.propagate(d['previous_draws']-d['previous_mean'], 'native', **physics).T/np.sqrt(len(d['previous_draws'])-1)
    inherited = np.sum(h**2, axis=1)
    qdiag = float(finite_step_innovation_covariance(xy[:1], xy[:1], cfg, physics['dt'])[0, 0])
    diag = qdiag+inherited
    np.testing.assert_array_equal(inherited, d['C_inherited_variance'])
    np.testing.assert_array_equal(diag, d['C_prior_variance'])
    folder = OUT/'by_trajectory'/name
    np.savez_compressed(folder/f'prior_level_{level}.npz', features=h, prior_variance=diag,
                        physics_mean=d['physics_mean'], qdiag=qdiag)
    prior = dict(previous_draws_sha256=digest(d['previous_draws']),
        previous_mean_sha256=digest(d['previous_mean']), physics_mean_sha256=digest(d['physics_mean']),
        features_sha256=digest(h), prior_variance_sha256=digest(diag),
        camera_sha256=digest(d['camera']), current_seed=int(saved.current_seed),
        original_cache=str(path.relative_to(ROOT)), build_seconds=perf_counter()-tick)
    if level>0:
        oldcheck = pd.read_csv(RETAINED/'verification.csv')
        oldrow = oldcheck[(oldcheck.trajectory==name)&(oldcheck.level==level)].iloc[0]
        assert digest(h)==oldrow.covariance_features_sha256
    dump(folder/f'prior_level_{level}.json', prior)
    return d, h, cfg, physics['dt'], prior


def blocks(d, h, cfg, dt, ids):
    xy = d['points']; foo = h[ids]
    return dict(prediction_mean=d['physics_mean'], observation_mean=d['physics_mean'][ids],
        observed_covariance=finite_step_innovation_covariance(xy[ids], xy[ids], cfg, dt)+gram(foo, foo),
        pred_observed_covariance=finite_step_innovation_covariance(xy, xy[ids], cfg, dt)+gram(h, foo),
        prediction_variance=d['C_prior_variance'])


def arm_layout(d, base_ids, added, arm):
    ids = base_ids if arm=='C0' else np.union1d(base_ids, added)
    kinds = np.where(d['camera'][ids]==d['ceiling'], 1, 0)
    if arm=='Cmask':
        kinds[np.isin(ids, added)] = -1
    values = np.full(len(ids), np.nan)
    values[kinds==0] = d['camera'][ids[kinds==0]]
    assert len(np.unique(ids))==len(ids)
    assert np.isin(base_ids, ids).all()
    if arm=='Cmask':
        assert np.isnan(values[np.isin(ids, added)]).all()
    if arm=='Cnumeric':
        assert not np.any(kinds[np.isin(ids, added)])
    return ids, kinds, values


def run_arm(d, b, ids, kinds, values, options, seed, monitor, path, expected=None):
    start = perf_counter()
    settings = [(int(options['current_samples_per_chain']), int(options['current_burn_in']),
                 int(options['current_thin']), False),
                (int(options['current_retry_samples_per_chain']), int(options['current_retry_burn_in']),
                 int(options['current_retry_thin']), True)]
    attempts = []
    for samples, burn, thin, retried in settings:
        ps, zs = [], []
        for chain in range(int(options['current_chains'])):
            p, z = sample_mixed_noisy_blocks(ids=ids, kinds=kinds, numerical_values=values,
                thresholds=float(d['ceiling']), **b, noise_sd=options['current_noise_sd'],
                n_samples=samples, burn_in=burn, thin=thin, seed=seed+100000*chain)
            ps.append(p[4]); zs.append(z)
        latent = np.asarray(ps); noisy = np.asarray(zs)
        classic = chain_diagnostics(latent, selected_mask=d['camera']==d['ceiling'])
        noisy_classic = chain_diagnostics(noisy)
        attempts.append(dict(samples=samples, burn=burn, thin=thin, retried=retried,
                             **classic, noisy_rhat_max=noisy_classic['rhat_max']))
        # Historical retry policy is unchanged; advanced diagnostics gate expansion.
        if classic['rhat_max']<=options['maximum_rhat'] and classic['selected_rhat_max']<=options['maximum_rhat']:
            break
    per_chain = int(options['evaluation_draws_per_chain'])
    indices = np.linspace(0, samples-1, per_chain, dtype=int)
    evaluation = latent[:, indices, :]
    prediction = prediction_from_draws(evaluation.reshape(-1, latent.shape[-1]))
    replay = None
    if expected is not None:
        for a, v in zip(prediction, expected):
            np.testing.assert_array_equal(a, v)
        replay = True
    diagnostics = pd.concat([arviz_diagnostics(noisy, ids[kinds!=0], 'noisy_constraint'),
                            arviz_diagnostics(latent[:, :, monitor], monitor, 'latent_temperature')])
    diagnostics.to_csv(path.with_suffix('.diagnostics.csv'), index=False)
    seconds = perf_counter()-start
    finite = bool(np.isfinite(diagnostics[['rank_rhat', 'ess_bulk', 'ess_tail']]).all().all())
    summary = dict(seed=seed, samples_per_chain=samples, burn_in=burn, thin=thin,
        retries=int(retried), attempts=attempts, reference_replay_exact=replay,
        numerical_count=int(np.sum(kinds==0)), lower_count=int(np.sum(kinds==1)),
        upper_count=int(np.sum(kinds==-1)), observation_count=len(ids),
        classical_rhat_max=classic['rhat_max'], classical_censored_rhat_max=classic['selected_rhat_max'],
        noisy_classical_rhat_max=noisy_classic['rhat_max'],
        rank_rhat_max=float(diagnostics.rank_rhat.max()), ess_bulk_min=float(diagnostics.ess_bulk.min()),
        ess_tail_min=float(diagnostics.ess_tail.min()), diagnostics_finite=finite,
        seconds=seconds, process_peak_rss_MiB=rss(),
        prior_diagonal_sha256=digest(b['prediction_variance']), prior_mean_sha256=digest(b['prediction_mean']))
    summary['reliable_for_expansion'] = bool(finite and summary['rank_rhat_max']<=GATE['rank_rhat_max']
        and summary['ess_bulk_min']>=GATE['ess_bulk_min'] and summary['ess_tail_min']>=GATE['ess_tail_min']
        and max(classic['rhat_max'], classic['selected_rhat_max'], noisy_classic['rhat_max'])<=options['maximum_rhat']
        and seconds<=GATE['arm_seconds_max'] and rss()<=GATE['process_peak_rss_MiB_max'])
    np.savez_compressed(path, **dict(zip(SUFFIXES, prediction)), evaluation_chains=evaluation,
        monitored_chains=latent[:, :, monitor], monitored_ids=monitor, noisy_constraint_chains=noisy,
        constraint_ids=ids[kinds!=0], observation_ids=ids, likelihood_kinds=kinds,
        numerical_values=values)
    dump(path.with_suffix('.json'), summary)
    return prediction, summary


def metrics(d, pred, pool, added):
    mean, sd, low, high, draws = pred
    error = mean-d['truth']; score = empirical_crps(draws, d['truth'])
    w = d['area_weights']; mask = np.zeros(len(w), bool); mask[added] = True
    peak = np.zeros(len(w)); peak[np.argmax(d['truth'])] = 1.
    regions = {'unused_pool':w*pool, 'added_U':w*mask, 'remaining_unused':w*(pool&~mask),
        'current_censored':w*(d['camera']==d['ceiling']), 'hot1pct_area':d['hot_area_weights'],
        'whole_area':w, 'true_peak_location':peak}
    rows = []
    for region, weights in regions.items():
        avg = lambda x: float(np.sum(weights*x)/weights.sum())
        rows.append(dict(region=region, nodes=int(np.sum(weights>0)),
            rmse_K=np.sqrt(avg(error**2)), mae_K=avg(abs(error)), bias_K=avg(error), crps_K=avg(score),
            coverage95=avg((low<=d['truth'])&(d['truth']<=high)), width95_K=avg(high-low),
            posterior_sd_K=avg(sd), mean_K=avg(mean), truth_K=avg(d['truth']),
            lower95_K=avg(low), upper95_K=avg(high),
            peak_absolute_error_K=float(abs(mean.max()-d['truth'].max()))))
    return rows


def preflight(options):
    rows = []
    recorded = pd.read_csv(BASE/'verification.csv')
    for name in NAMES:
        base, added, pool = select_locations(name)
        with np.load(BASE/'by_trajectory'/name/'level_0.npz') as d:
            for label, extra in [('budget256', len(added)), ('all_omitted_NOT_RUN', int(pool.sum()))]:
                n = len(pool); m = len(base)+extra; k = int(np.sum(d['camera']==d['ceiling']))+extra
                # Arrays only, not a claim about full process RSS or solver workspaces.
                core = 8*(n*m+m*m+k*k+n*1600+4*400*n+4*400*k)/2**20
                baseline_seconds = float(recorded[(recorded.trajectory==name)&(recorded.level==0)&(recorded.model=='C')].model_seconds.iloc[0])
                rows.append(dict(trajectory=name, scope=label, prediction_nodes=n,
                    numerical_count=int(len(base)-np.sum(d['camera']==d['ceiling'])),
                    constrained_dimension=k, observations=m, core_arrays_MiB=core,
                    factorization_work_ratio_to_baseline=(m/len(base))**3,
                    constraint_matvec_work_ratio_to_baseline=(k/np.sum(d['camera']==d['ceiling']))**2,
                    saved_reference_model_seconds=baseline_seconds,
                    rough_scaling_seconds=baseline_seconds*max((m/len(base))**3,(k/np.sum(d['camera']==d['ceiling']))**2),
                    selected_lumped_area_fraction=float(d['area_weights'][added].sum()/d['area_weights'].sum()),
                    pool_area_fraction=float(d['area_weights'][pool].sum()/d['area_weights'].sum())))
    pd.DataFrame(rows).to_csv(OUT/'preflight.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['preflight', 'reference', 'secondary', 'report'], default='reference')
    args = parser.parse_args()
    for key in ['OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
        if os.environ.get(key)!='1':
            raise RuntimeError('Set '+key+'=1')
    OUT.mkdir(parents=True, exist_ok=True)
    config = ROOT/'outputs/by_experiment/49_matched_native_grid_gp/frozen_configuration.csv'
    options = pd.read_csv(config).iloc[0].to_dict()
    assert options==pd.read_csv(RETAINED/'frozen_configuration.csv').iloc[0].to_dict()
    assert options['current_noise_sd']==.5 and options['measurement_noise_sd']==.25
    if args.stage=='report':
        report(); return
    frozen = dict(inherited_settings=options, scope='two development cases; native nonadvective C only',
        selection='256-point deterministic Euclidean farthest-point coverage; bbox-center start; normalized distances rounded14digits; lowest node ID breaks ties',
        selection_seed=None, likelihood_noise_sd_K=.5, previous_inference_rerun=False,
        reference_repeats=[0, REPEAT_OFFSET], expansion_gate=GATE,
        posterior_representation='historical marginal draws; not full coherent latent fields',
        arviz='0.17.1; scipy.signal.gaussian compatibility alias only for package import',
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    if (OUT/'frozen_configuration.json').exists():
        assert json.loads((OUT/'frozen_configuration.json').read_text())==frozen
    else:
        dump(OUT/'frozen_configuration.json', frozen)
    if args.stage in ['preflight', 'reference']:
        preflight(options)
    if args.stage=='preflight':
        return
    if args.stage=='secondary':
        gate = json.loads((OUT/'reference_gate.json').read_text())
        if not gate['proceed_to_high_ceilings']:
            raise RuntimeError('Reference reliability gate failed: no higher-ceiling inference allowed')
    levels = [0] if args.stage=='reference' else [1, 2]
    allrows, checks, compatibility = [], [], []
    for name in NAMES:
        base, added, pool = select_locations(name)
        for level in levels:
            d, h, cfg, dt, prior = inputs(name, level, options)
            baseblocks = blocks(d, h, cfg, dt, base)
            extraids = np.union1d(base, added)
            extrablocks = blocks(d, h, cfg, dt, extraids)
            where = np.searchsorted(extraids, base)
            np.testing.assert_allclose(extrablocks['observed_covariance'][np.ix_(where, where)],
                                       baseblocks['observed_covariance'], rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(extrablocks['pred_observed_covariance'][:, where],
                                       baseblocks['pred_observed_covariance'], rtol=1e-13, atol=1e-13)
            monitor = np.union1d(extraids, spread_nodes(d['points'], np.ones(len(pool), bool), 64))
            repeats = [0, 1] if level==0 else [0]
            for repeat in repeats:
                for arm in (['C0', 'Cmask', 'Cnumeric'] if repeat==0 else ['C0', 'Cmask']):
                    path = OUT/'by_trajectory'/name/f'level_{level}_{arm}_repeat_{repeat}.npz'
                    ids, kinds, values = arm_layout(d, base, added, arm)
                    b = baseblocks if arm=='C0' else extrablocks
                    expected = None
                    if arm=='C0' and repeat==0:
                        if level==0:
                            expected = [d['C_'+s] for s in SUFFIXES]
                        else:
                            with np.load(RETAINED/'by_trajectory'/name/f'level_{level}.npz') as old:
                                expected = [old['retained_'+s] for s in SUFFIXES]
                    if path.exists():
                        with np.load(path) as cache:
                            pred = tuple(cache[s] for s in SUFFIXES)
                            np.testing.assert_array_equal(ids, cache['observation_ids'])
                        summary = json.loads(path.with_suffix('.json').read_text())
                    else:
                        print('START', name, level, arm, repeat, 'constraints', int(np.sum(kinds!=0)), flush=True)
                        pred, summary = run_arm(d, b, ids, kinds, values, options,
                            prior['current_seed']+repeat*REPEAT_OFFSET, monitor, path, expected)
                    common = dict(trajectory=name, level=level, ceiling_K=float(d['ceiling']), arm=arm, repeat=repeat)
                    allrows.extend(dict(**common, **row) for row in metrics(d, pred, pool, added))
                    checks.append(dict(**common, **{k:v for k,v in summary.items() if k!='attempts'}))
                    print('DONE', common, 'rank', summary['rank_rhat_max'], 'bulk/tail',
                          summary['ess_bulk_min'], summary['ess_tail_min'], 'seconds', summary['seconds'], flush=True)
                    if arm=='C0':
                        probs = ndtr((float(d['ceiling'])-pred[4][:, added])/options['current_noise_sd']).mean(axis=0)
                        compatibility.extend(dict(**common, node=int(node), baseline_non_saturation_probability=float(p))
                                             for node,p in zip(added, probs))
                    pd.DataFrame(allrows).to_csv(OUT/f'{args.stage}_metrics.csv', index=False)
                    pd.DataFrame(checks).to_csv(OUT/f'{args.stage}_convergence_runtime.csv', index=False)
                    pd.DataFrame(compatibility).to_csv(OUT/f'{args.stage}_flag_compatibility.csv', index=False)
    if args.stage=='reference':
        reliable = all(r['reliable_for_expansion'] for r in checks)
        dump(OUT/'reference_gate.json', dict(proceed_to_high_ceilings=reliable,
            reason='All reference arms/repeats must pass preset inference/runtime checks; effect need not be positive',
            failed=[{k:r[k] for k in ['trajectory','arm','repeat','rank_rhat_max','ess_bulk_min','ess_tail_min']}
                    for r in checks if not r['reliable_for_expansion']], thresholds=GATE))
    files = [Path(__file__), ROOT/'src/native_mask_pilot.py', ROOT/'src/truncated_gaussian.py',
             ROOT/'src/matched_projection.py', ROOT/'src/stochastic_heat_gp.py', config]
    dump(OUT/'source_fingerprints.json', {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    inputfiles = [p for name in NAMES for p in (BASE/'by_trajectory'/name).glob('level_*.npz')]
    inputfiles += [p for name in NAMES for p in (RETAINED/'by_trajectory'/name).glob('level_*.npz')]
    dump(OUT/'input_fingerprints.json', {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputfiles})
    report()


def report():
    frames = [pd.read_csv(p) for p in [OUT/'reference_metrics.csv', OUT/'secondary_metrics.csv'] if p.exists()]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT/'results.csv', index=False)
    fields = ['rmse_K','mae_K','bias_K','crps_K','coverage95','width95_K','posterior_sd_K']
    pairs = []
    for key, g in df.groupby(['trajectory','level','ceiling_K','repeat','region']):
        g = g.set_index('arm')
        for arm in ['Cmask','Cnumeric']:
            if arm in g.index:
                pairs.append(dict(zip(['trajectory','level','ceiling_K','repeat','region'],key),
                    contrast=arm+' minus C0', **{m:float(g.loc[arm,m]-g.loc['C0',m]) for m in fields}))
    paired = pd.DataFrame(pairs); paired.to_csv(OUT/'paired_comparisons.csv', index=False)
    repeats = []
    ref = paired[(paired.level==0)&(paired.contrast=='Cmask minus C0')]
    for key, g in ref.groupby(['trajectory','region']):
        if set(g.repeat)=={0,1}:
            for metric in fields:
                v = g.set_index('repeat')[metric]
                repeats.append(dict(trajectory=key[0],region=key[1],metric=metric,
                    paired_change_seed0=v[0],paired_change_seed1=v[1],
                    mean_paired_change=v.mean(),repeat_spread=abs(v[0]-v[1]),
                    same_sign=bool(v[0]*v[1]>0),
                    larger_than_repeat_spread=bool(abs(v.mean())>abs(v[0]-v[1]))))
    pd.DataFrame(repeats).to_csv(OUT/'repeat_seed_comparisons.csv',index=False)
    compact = df[(df.repeat==0)&df.region.isin(['unused_pool','added_U','remaining_unused','hot1pct_area'])]
    compact[['trajectory','level','arm','region',*fields]].to_csv(OUT/'compact_comparison.csv',index=False)
    plot_outputs(df)


def plot_outputs(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from matplotlib.colors import TwoSlopeNorm, Normalize
    for name, level in df[['trajectory','level']].drop_duplicates().itertuples(index=False, name=None):
        with np.load(BASE/'by_trajectory'/name/f'level_{level}.npz') as d:
            xy=d['points']; truth=d['truth']; ceiling=float(d['ceiling']); camera=d['camera']
            triangles=d['triangles']; physics=d['physics_mean']
        with np.load(OUT/'by_trajectory'/name/'selection.npz') as sel:
            added=sel['added_ids']; pool=sel['pool']
        arms={}
        for arm in ['C0','Cmask','Cnumeric']:
            with np.load(OUT/'by_trajectory'/name/f'level_{level}_{arm}_repeat_0.npz') as a:
                arms[arm]={k:a[k] for k in ['mean','sd','lower','upper']}
        tri=Triangulation(xy[:,0]*1000,xy[:,1]*1000,triangles)
        error_limit=max(max(abs(a['mean'][pool]-truth[pool])) for a in arms.values())
        sd_limit=max(max(a['sd'][pool]) for a in arms.values())
        fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        for col,(arm,a) in enumerate(arms.items()):
            err=np.where(pool,a['mean']-truth,np.nan); sd=np.where(pool,a['sd'],np.nan)
            # Face colors restrict this view to triangles entirely in U_pool.
            faces=np.all(pool[triangles],axis=1); faceerr=np.mean(err[triangles],axis=1)
            facesd=np.mean(sd[triangles],axis=1)
            shown=Triangulation(tri.x,tri.y,tri.triangles,mask=~faces)
            im=axes[0,col].tripcolor(shown,facecolors=faceerr,cmap='RdBu_r',norm=TwoSlopeNorm(0,-error_limit,error_limit))
            sm=axes[1,col].tripcolor(shown,facecolors=facesd,cmap='viridis',norm=Normalize(0,sd_limit))
            for row in [0,1]:
                axes[row,col].scatter(xy[added,0]*1000,xy[added,1]*1000,s=3,c='black',alpha=.5)
                axes[row,col].set(title=arm+(' signed error' if row==0 else ' posterior SD'),xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
        fig.colorbar(im,ax=list(axes[0]),label='Estimate - truth (K)',shrink=.75)
        fig.colorbar(sm,ax=list(axes[1]),label='Latent posterior SD (K)',shrink=.75)
        fig.suptitle(f'{name}, ceiling {ceiling:.3f} K | previously unused region\nShared scales; black dots: 256 added locations; blank: retained baseline region')
        fig.savefig(OUT/f'background_{name}_level_{level}.png',dpi=150);plt.close(fig)
        sections=pd.read_csv(ROOT/'outputs/by_experiment/49_matched_native_grid_gp/native_diagnostics'/f'cross_sections_{name}.csv')
        sections=sections[sections.model=='native_C']
        fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True,constrained_layout=True)
        for row,(section,nodes) in enumerate(sections.groupby('section',sort=False)):
            ids=nodes.node.to_numpy(int);x=nodes.along_mm.to_numpy()
            for col,(arm,a) in enumerate(arms.items()):
                ax=axes[row,col];ax.fill_between(x,a['lower'][ids],a['upper'][ids],alpha=.25,color='#0072B2')
                ax.plot(x,truth[ids],'k-',label='Native truth');ax.plot(x,physics[ids],'--',color='#D55E00',label='Frozen forecast')
                ax.plot(x,a['mean'][ids],color='#0072B2',label=arm)
                selected=np.isin(ids,added);ax.scatter(x[selected],a['mean'][ids[selected]],s=30,marker='x',color='black',label='Added U on section')
                ax.axhline(ceiling,ls=':',color='gray');ax.set(title=arm+' | '+section.replace('flux','gradient'),xlabel='Distance from true peak (mm)',ylabel='Temperature (K)')
                ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'{name} | ceiling {ceiling:.3f} K | same native section nodes and fixed predictive prior\nPointwise marginal intervals; connecting lines do not define off-mesh intervals')
        fig.savefig(OUT/f'cross_sections_{name}_level_{level}.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    main()
