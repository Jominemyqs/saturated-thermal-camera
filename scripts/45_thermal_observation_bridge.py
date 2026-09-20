from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ceiling_sweep import (SequentialAdvectiveConfig, lower_camera_ceiling,
    matched_censoring_observations, sequential_prior_blocks)
from src.observation_bridge import subset_observation_blocks, cooling_visibility
from src.pseudo_censoring_runner import _previous_prediction, _current_prediction
from src.thermal_posterior_physics import (prepare_trajectory, paired_camera_observations,
    posterior_physics_means, DEVELOPMENT_TRAJECTORIES)
from src.thermal_trajectory import trajectory_catalog
from src.metrics import empirical_crps

BASE = ROOT / 'outputs/by_experiment/38_pseudo_censoring_calibration'
OUT = ROOT / 'outputs/by_experiment/41_observation_bridge_and_smoothing_readiness'
METRICS = ['field_excess_rel_l2','overall_rmse_K','overall_crps_K','top_1pct_rmse_K',
    'top_1pct_bias_K','top_1pct_crps_K','top_1pct_coverage_95','top_1pct_width_95_K','peak_absolute_error_K']


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def metrics(pred, truth, ambient):
    mean, sd, lower, upper, draws = pred
    y = truth.ravel()
    e = mean-y
    top = np.argsort(y, kind='mergesort')[-max(1,int(.01*len(y))):]
    crps = empirical_crps(draws,y)
    return dict(field_excess_rel_l2=np.linalg.norm(e)/np.linalg.norm(y-ambient),
        overall_rmse_K=np.sqrt(np.mean(e**2)),overall_crps_K=np.mean(crps),
        top_1pct_rmse_K=np.sqrt(np.mean(e[top]**2)),top_1pct_bias_K=np.mean(e[top]),
        top_1pct_crps_K=np.mean(crps[top]),top_1pct_coverage_95=np.mean((lower[top]<=y[top])&(y[top]<=upper[top])),
        top_1pct_width_95_K=np.mean(upper[top]-lower[top]),peak_absolute_error_K=abs(mean.max()-y.max()))


def visibility(prepared, ceiling, index, options, output):
    # Calendar-position targets are fixed before looking at observations or truth.
    count = len(prepared.times)
    targets = sorted(set(max(1,int(f*(count-1))) for f in (.25,.5,.75)))
    rng = np.random.default_rng(int(options['seed'])+9_000_000+10_000*index)
    observed = np.minimum(prepared.history+rng.normal(0,options['measurement_noise_sd'],prepared.history.shape),ceiling)
    fixed_mask = np.zeros(prepared.truth.shape, dtype=bool)
    stride = int(options['observation_stride'])
    fixed_mask[::stride, ::stride] = True
    rows=[]
    for target in targets:
        for lag in sorted(set([1,2,4,8,16,32,64,128,count-1-target])):
            if target+lag>=count or lag<1:
                continue
            rows.append(dict(trajectory=prepared.name,target_frame=target,lag=lag,
                target_time_s=prepared.times[target],duration_s=prepared.times[target+lag]-prepared.times[target],
                ceiling_K=ceiling,**cooling_visibility(observed,ceiling,target,lag,options['measurement_noise_sd'],fixed_mask)))
    pd.DataFrame(rows).to_csv(output/f'{prepared.name}_visibility.csv',index=False)


def worker(payload):
    name,family,index,ceiling,options,output = payload
    output=Path(output)
    p,_ = prepare_trajectory(ROOT.parent/'heat_eq_laser_trajectories',name,
        nx=int(options['nx']),ny=int(options['ny']),heat_flux_cutoff=options['heat_flux_cutoff'])
    n=len(p.times)-1
    role='development' if name in DEVELOPMENT_TRAJECTORIES else 'heldout'
    if role=='development':
        visibility(p,ceiling,index,options,output)
    camera=paired_camera_observations(p,previous_index=n-1,current_index=n,threshold=ceiling,
        observation_stride=int(options['observation_stride']),noise_sd=options['measurement_noise_sd'],
        seed=int(options['seed'])+10_000*index)
    camera=lower_camera_ceiling(camera,ceiling)
    length=p.source_lengthscale*options['length_multiplier']
    previous,draws,prevcheck=_previous_prediction(p,frame=camera['frames'][0],fixed_mask=camera['fixed_observation_mask'],
        threshold=ceiling,signal_sd=options['signal_sd'],lengthscale=length,noise_sd=options['previous_noise_sd'],
        options=options,seed=int(options['seed'])+50_000*index)
    mean,_,_=posterior_physics_means(p,previous,previous_index=n-1,current_index=n,
        diffusivity=options['diffusivity'],cooling_rate=options['cooling_rate'],
        source_coupling=options['source_coupling'],source_flux_threshold=options['source_flux_threshold'])
    obs=matched_censoring_observations(p,frame=camera['frames'][1],fixed_mask=camera['fixed_observation_mask'],threshold=ceiling)
    config=SequentialAdvectiveConfig(signal_sd=options['signal_sd'],forcing_lengthscale=length*options['forcing_length_multiplier'],
        diffusivity=options['diffusivity'],cooling_rate=options['cooling_rate'],current_noise_sd=options['current_noise_sd'],quadrature_order=int(options['quadrature_order']))
    blocks,covcheck=sequential_prior_blocks(p,camera={**camera,'current':obs},previous_mean=previous,
        previous_draws=draws,mean_field=mean,displacement=None,config=config)
    included=(camera['fixed_observation_mask']|obs['saturated_full']).ravel()
    keep=camera['fixed_observation_mask'].ravel()[included]
    sparse,sparse_blocks=subset_observation_blocks(obs,blocks,keep)
    np.testing.assert_array_equal(sparse['y_obs'],camera['current']['y_obs'])
    np.testing.assert_array_equal(sparse['sat_mask'],camera['current']['sat_mask'])
    common=dict(trajectory=name,family=family,role=role,
        previous_draws_sha256=digest(draws),prior_mean_sha256=digest(mean),prior_variance_sha256=digest(blocks['prediction_variance']),
        camera_sha256=digest(camera['frames'][1]['clipped_full']),
        common_covariance_sha256=digest(sparse_blocks['observed_covariance'],sparse_blocks['pred_observed_covariance']),
        previous_rhat_max=prevcheck['rhat_max'],current_seed=int(options['seed'])+100_000*index,**covcheck)
    rows=[]
    checks=[]
    for label,o,b in [('sparse_stride5',sparse,sparse_blocks),('stride5_unsaturated_all_censored',obs,blocks)]:
        pred,check=_current_prediction(o,b,options=options,seed=common['current_seed'])
        rows.append(dict(trajectory=name,family=family,role=role,protocol=label,**metrics(pred,p.truth,p.ambient)))
        checks.append(dict(**common,protocol=label,n_observed=len(o['y_obs']),n_censored=int(o['sat_mask'].sum()),
            **{f'current_{k}':v for k,v in check.items()}))
        print(f'{name}: {label} complete',flush=True)
    saved=pd.read_csv(BASE/'results.csv')
    saved=saved[(saved.trajectory==name)&(saved.level_index==0)&(saved.method=='uncalibrated')]
    audit=[]
    if len(saved):
        for metric in METRICS:
            delta=rows[1][metric]-float(saved.iloc[0][metric])
            if abs(delta)>1e-8:
                raise AssertionError(f'Exp38 mismatch: {name} {metric} {delta}')
            audit.append(dict(trajectory=name,metric=metric,saved=saved.iloc[0][metric],reproduced=rows[1][metric],difference=delta))
    pd.DataFrame(rows).to_csv(output/f'{name}_results.csv',index=False)
    pd.DataFrame(checks).to_csv(output/f'{name}_checks.csv',index=False)
    pd.DataFrame(audit,columns=['trajectory','metric','saved','reproduced','difference']).to_csv(output/f'{name}_audit.csv',index=False)
    return name


def summarize(output):
    cache=output/'by_trajectory'
    results=pd.concat([pd.read_csv(p) for p in sorted(cache.glob('*_results.csv'))])
    checks=pd.concat([pd.read_csv(p) for p in sorted(cache.glob('*_checks.csv'))])
    audit_frames=[pd.read_csv(p) for p in sorted(cache.glob('*_audit.csv'))]
    audits=pd.concat([frame for frame in audit_frames if not frame.empty])
    results.to_csv(output/'results.csv',index=False)
    checks.to_csv(output/'implementation_checks.csv',index=False)
    audits.to_csv(output/'baseline_equality_audit.csv',index=False)
    held=results[results.role=='heldout']
    summary=held.groupby('protocol')[METRICS].mean()
    summary.to_csv(output/'heldout30_summary.csv')
    held.groupby(['family','protocol'])[METRICS].mean().to_csv(output/'family_summary.csv')
    paired=[]
    for family,frame in [('All',held),*held.groupby('family')]:
        a=frame[frame.protocol=='sparse_stride5'].set_index('trajectory')
        b=frame[frame.protocol!='sparse_stride5'].set_index('trajectory')
        for metric in METRICS:
            d=b[metric]-a[metric]
            paired.append(dict(family=family,metric=metric,mean_change=d.mean(),median_change=d.median(),
                n_decreased=int((d<0).sum()),n_increased=int((d>0).sum()),n_trajectories=len(d)))
    pd.DataFrame(paired).to_csv(output/'paired_comparisons.csv',index=False)
    old=pd.read_csv(ROOT/'outputs/by_experiment/37_canonical_converged_architectures/results.csv')
    old=old[(old.method=='Posterior physics mean + sequential ST') & old.trajectory.isin(held.trajectory)]
    new=held[held.protocol=='sparse_stride5']
    lineage=new.merge(old,on='trajectory',suffixes=('_bridge','_historical'))
    lineage.to_csv(output/'historical_sparse_lineage.csv',index=False)
    decomposition=[]
    for metric in METRICS:
        old_metric = 'top_1pct_interval_width_95_K' if metric=='top_1pct_width_95_K' else metric
        historical=float(old[old_metric].mean())
        sparse=float(summary.loc['sparse_stride5',metric])
        rich=float(summary.loc['stride5_unsaturated_all_censored',metric])
        decomposition.append(dict(metric=metric,historical_sparse=historical,controlled_sparse=sparse,controlled_all_censored=rich,
            observation_set_change=rich-sparse,remaining_protocol_change=sparse-historical,total_change=rich-historical))
    pd.DataFrame(decomposition).to_csv(output/'lineage_decomposition.csv',index=False)
    vis=pd.concat([pd.read_csv(p) for p in sorted(cache.glob('*_visibility.csv'))])
    vis.to_csv(output/'development_cooling_visibility.csv',index=False)
    vis.groupby(['trajectory','lag'])[['persistent_observable_fraction','persistent_observable_stride5_fraction',
        'ever_observable_fraction','endpoint_observable_fraction']].mean().to_csv(output/'cooling_family_lag_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,(name,frame) in zip(axes,vis.groupby('trajectory')):
        for target,part in frame.groupby('target_frame'):
            ax.plot(1000*part.duration_s,part.persistent_observable_fraction,'o-',label=f'Target {target}')
        ax.set(title=name,xlabel='Future duration (ms)',ylabel='Persistent newly observable fraction',ylim=(-.02,1.02))
        ax.legend(fontsize=8)
    fig.savefig(output/'cooling_visibility.png',dpi=180)
    plt.close(fig)
    lines=['# Observation bridge and smoothing readiness','',
        'Only the stationary sequential model is tested. One previous posterior and one prior are built per trajectory; sparse observations are exact subblocks of the richer set. No smoother or covariance change is implemented.',
        '','| Protocol | Field error | Overall CRPS | Top1 RMSE | Top1 CRPS | Coverage | Width |','|---|---:|---:|---:|---:|---:|---:|']
    for label,r in summary.iterrows():
        lines.append(f'| {label} | {r.field_excess_rel_l2:.6f} | {r.overall_crps_K:.6f} | {r.top_1pct_rmse_K:.6f} | {r.top_1pct_crps_K:.6f} | {r.top_1pct_coverage_95:.6f} | {r.top_1pct_width_95_K:.6f} |')
    lines += ['',f'Largest Experiment38 reproduction difference: {audits.difference.abs().max():.3g}.',
        '', 'Canonical observations: stride-5 unsaturated plus every observed-censored location (not all unsaturated camera pixels). Previous representation, physical coefficients, seeds, HMC convergence/retry settings and exact top-25 metric mask are frozen from Experiment38. Current likelihood sees current observations only.',
        '', 'Historical Experiment37 uses ESS instead of HMC for its current block sampler, a different draw pooling/mean estimator, and different retained sample settings. The controlled mask effect must be separated from its residual mismatch to the saved sparse row; see historical_sparse_lineage.csv.',
        '', 'Cooling readiness uses only three development paths. Targets are fixed at 25%, 50%, 75% of frame count, not selected using truth or performance. Noise is generated once per trajectory, and the inherited final-frame reference ceiling stays constant over time. Observed masks alone define censoring and visibility. Persistent visibility requires three consecutive observations more than two measurement SD below the ceiling. This guards against, but does not eliminate, noise flicker. All endpoint and ever-visible counts are also retained.',
        '', 'The stride5 visibility columns count only newly unsaturated locations that would actually enter the existing likelihood. Off-stride pixels can become observable in the camera but then drop out of the canonical inference subset. Camera-visible and assimilated information must not be conflated in a future smoothing test.',
        '', 'These preparation routines inherit simulator-based ambient and reference-ceiling provenance. Before a causal interior-frame experiment, ambient/source-width preparation needs an availability audit; no interior-frame GP was fitted here.',
        '', 'See SAMPLE_AUDIT.md for the joint-draw limitation. Do not feed current marginal draws into a spatial smoother as complete field samples.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    assert len(results)==66 and len(held)==60
    assert len(vis.trajectory.unique())==3
    for key in ['previous_draws_sha256','prior_mean_sha256','prior_variance_sha256','camera_sha256',
                'common_covariance_sha256','current_seed']:
        assert checks.groupby('trajectory')[key].nunique().max()==1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--readiness-only',action='store_true')
    args=parser.parse_args()
    cache=OUT/'by_trajectory'
    cache.mkdir(parents=True,exist_ok=True)
    options=pd.read_csv(BASE/'fixed_configuration.csv').iloc[0].to_dict()
    pd.DataFrame([options]).to_csv(OUT/'fixed_configuration.csv',index=False)
    catalog=trajectory_catalog(ROOT.parent/'heat_eq_laser_trajectories')
    assert len(catalog)==33 and sum(r.name in DEVELOPMENT_TRAJECTORIES for r in catalog)==3
    thresholds=pd.read_csv(ROOT/'outputs/by_experiment/30_final_architecture_comparison/sequential_rerun_results.csv').groupby('trajectory').threshold_K.first().to_dict()
    if args.readiness_only:
        for i,r in enumerate(catalog):
            if r.name in DEVELOPMENT_TRAJECTORIES:
                p,_=prepare_trajectory(ROOT.parent/'heat_eq_laser_trajectories',r.name,nx=int(options['nx']),
                    ny=int(options['ny']),heat_flux_cutoff=options['heat_flux_cutoff'])
                visibility(p,thresholds[r.name],i,options,cache)
        summarize(OUT)
        return
    paths=[Path(__file__),*sorted((ROOT/'src').glob('*.py')),BASE/'fixed_configuration.csv']
    (OUT/'source_fingerprints.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    payloads=[(r.name,r.family,i,thresholds[r.name],options,str(cache)) for i,r in enumerate(catalog)
        if not (cache/f'{r.name}_audit.csv').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(worker,p) for p in payloads]):
            print('Finished',future.result(),flush=True)
    summarize(OUT)


if __name__=='__main__':
    main()
