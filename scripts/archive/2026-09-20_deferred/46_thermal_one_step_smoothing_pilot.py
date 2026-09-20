from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=next(p for p in Path(__file__).resolve().parents if (p / 'src').is_dir())
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ceiling_sweep import (SequentialAdvectiveConfig, matched_censoring_observations,
    sequential_prior_blocks, observation_indices)
from src.censored_gp import cholesky_with_jitter
from src.metrics import empirical_crps
from src.one_step_smoothing import censored_log_likelihood, importance_weights, weighted_prediction
from src.pseudo_censoring_runner import _previous_prediction, _current_prediction
from src.stochastic_heat_gp import StochasticHeatConfig, finite_step_innovation_covariance, propagate_residual_draws
from src.thermal_posterior_physics import (PreparedTrajectory, DEVELOPMENT_TRAJECTORIES, trajectory_path,
    estimate_source_lengthscale, posterior_physics_means, current_source_field)
from src.thermal_trajectory import ThermalTrajectory, SurfaceGridProjector, trajectory_catalog
from src.observation_bridge import cooling_visibility

BASE=ROOT/'outputs/by_experiment/38_pseudo_censoring_calibration'
OUT=ROOT/'outputs/archive/2026-09-20_deferred/42_one_step_smoothing_pilot'


def camera_frame(p,observed,index,fixed,ceiling):
    values=observed[index]
    return dict(time_index=index,time=float(p.times[index]),clipped_full=values,
        saturated_full=values==ceiling,values=values[fixed],sat_mask=(values==ceiling)[fixed])


def load_development(name,index,ceiling,o):
    trajectory=ThermalTrajectory(trajectory_path(ROOT.parent/'heat_eq_laser_trajectories',name))
    times,raw=trajectory.read_history(time_indices=list(range(trajectory.n_times)),surface_only=True)
    _,flux=trajectory.read_history('HeatFluxZ',time_indices=list(range(trajectory.n_times)),surface_only=True)
    projector=SurfaceGridProjector(trajectory.surface_coordinates(),nx=int(o['nx']),ny=int(o['ny']))
    history=projector.project(raw)
    source=projector.project(flux)
    n=(len(times)-1)//2
    rng=np.random.default_rng(int(o['seed'])+9_000_000+10_000*index)
    observed=np.minimum(history+rng.normal(0,o['measurement_noise_sd'],history.shape),ceiling)
    # Only initial camera data and commanded source history through n set these.
    ambient=float(np.median(observed[0]))
    spacing=max(np.diff(projector.xs).mean(),np.diff(projector.ys).mean())
    length=max(estimate_source_lengthscale(source[:n+1],projector.xs,projector.ys),spacing)
    p=PreparedTrajectory(name,projector.xs,projector.ys,projector.grid_points,times,history,
        source,history[n],np.zeros(history[n].shape),ambient,length,o['signal_sd'])
    return p,observed,n


def score(draws,w,p,stage):
    y=p.truth.ravel()
    mean,sd,lower,upper,crps=weighted_prediction(draws,w,y)
    top=np.argsort(y,kind='mergesort')[-25:]
    e=mean-y
    row=dict(stage=stage,field_excess_rel_l2=np.linalg.norm(e)/np.linalg.norm(y-p.ambient),
        overall_rmse_K=np.sqrt(np.mean(e**2)),overall_empirical_crps_K=crps.mean(),
        top1_rmse_K=np.sqrt(np.mean(e[top]**2)),top1_bias_K=e[top].mean(),
        top1_empirical_crps_K=crps[top].mean(),top1_coverage=np.mean((lower[top]<=y[top])&(y[top]<=upper[top])),
        top1_width_K=np.mean(upper[top]-lower[top]),peak_error_K=abs(mean.max()-y.max()))
    return row,mean


def run(name,index,ceiling,o):
    p,observed,n=load_development(name,index,ceiling,o)
    fixed=np.zeros(p.truth.shape,bool)
    fixed[::int(o['observation_stride']),::int(o['observation_stride'])]=True
    prev_frame=camera_frame(p,observed,n-1,fixed,ceiling)
    now=camera_frame(p,observed,n,fixed,ceiling)
    future=camera_frame(p,observed,n+1,fixed,ceiling)
    obs=matched_censoring_observations(p,frame=now,fixed_mask=fixed,threshold=ceiling)
    next_obs=matched_censoring_observations(p,frame=future,fixed_mask=fixed,threshold=ceiling)
    previous,previous_draws,pc=_previous_prediction(p,frame=prev_frame,fixed_mask=fixed,threshold=ceiling,
        signal_sd=o['signal_sd'],lengthscale=p.source_lengthscale*o['length_multiplier'],
        noise_sd=o['previous_noise_sd'],options=o,seed=int(o['seed'])+50_000*index)
    mean,_,_=posterior_physics_means(p,previous,previous_index=n-1,current_index=n,
        diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],source_coupling=o['source_coupling'],
        source_flux_threshold=o['source_flux_threshold'])
    config=SequentialAdvectiveConfig(signal_sd=o['signal_sd'],forcing_lengthscale=p.source_lengthscale*o['length_multiplier']*o['forcing_length_multiplier'],
        diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],current_noise_sd=o['current_noise_sd'],quadrature_order=int(o['quadrature_order']))
    camera=dict(frames=[prev_frame,now],current=obs)
    blocks,bc=sequential_prior_blocks(p,camera=camera,previous_mean=previous,previous_draws=previous_draws,
        mean_field=mean,displacement=None,config=config)
    # Same prior, now requesting the complete prediction covariance.
    all_obs={**obs,'x_obs':obs['x_pred']}
    full,_=sequential_prior_blocks(p,camera={**camera,'current':all_obs},previous_mean=previous,
        previous_draws=previous_draws,mean_field=mean,displacement=None,config=config)
    k=full['observed_covariance']
    idx=observation_indices(p,obs['x_obs'])
    block_difference=float(np.max(abs(k[:,idx]-blocks['pred_observed_covariance'])))
    np.testing.assert_allclose(k[:,idx],blocks['pred_observed_covariance'],rtol=1e-10,atol=1e-10)
    seed=int(o['seed'])+100_000*index
    marginal,mc=_current_prediction(obs,blocks,options=o,seed=seed)
    joint,jc=_current_prediction(obs,{**blocks,'prediction_covariance':k},options=o,seed=seed)
    target_draws=joint[4]
    del full,k,previous_draws
    print(name,'joint target inference complete',flush=True)
    # Transition is the existing positive-excess heat/source forecast plus Q_dt.
    dt=float(p.times[n+1]-p.times[n])
    deterministic=p.ambient+propagate_residual_draws(np.maximum(target_draws.reshape(-1,*p.truth.shape)-p.ambient,0),
        dx=float(np.diff(p.xs).mean()),dy=float(np.diff(p.ys).mean()),time_step=dt,
        diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'])
    deterministic+=current_source_field(p,previous_index=n,current_index=n+1,source_coupling=o['source_coupling'],
        source_flux_threshold=o['source_flux_threshold'])
    future_indices=observation_indices(p,next_obs['x_obs'])
    forecast=deterministic.reshape(len(target_draws),-1)[:,future_indices]
    heat=StochasticHeatConfig(signal_sd=o['signal_sd'],forcing_lengthscale=config.forcing_lengthscale,
        diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],quadrature_order=int(o['quadrature_order']))
    q=finite_step_innovation_covariance(next_obs['x_obs'],next_obs['x_obs'],heat,dt)
    qfactor=cholesky_with_jitter(q,1e-7*max(float(np.max(np.diag(q))),1.))
    # Two independent four-path likelihood estimates per target particle.
    rng=np.random.default_rng(seed+8_000_000)
    loglik=[]
    for repeat in range(8):
        states=forecast+rng.normal(size=forecast.shape).dot(qfactor.T)
        loglik.append(censored_log_likelihood(states,next_obs['y_obs'],next_obs['sat_mask'],ceiling,o['current_noise_sd']))
    loglik=np.asarray(loglik)
    weighted=[]
    rows=[]
    diagnostics=[]
    for label,logs in [('repeat_A',loglik[:4]),('repeat_B',loglik[4:]),('pooled',loglik)]:
        weights,d=importance_weights(logsumexp(logs,axis=0)-np.log(len(logs)))
        _,pathd=importance_weights(logs.ravel())
        weighted.append(weights)
        row,_=score(target_draws,weights,p,'L1_'+label)
        rows.append(row)
        d.update(dict(trajectory=name,repeat=label,path_ess=pathd['ess'],path_max_weight=pathd['max_weight'],
            n_target_particles=len(weights),n_paths=len(logs)*len(weights)))
        for chain in range(int(o['current_chains'])):
            start=chain*int(o['evaluation_draws_per_chain'])
            d[f'target_chain_{chain}_weight']=float(weights[start:start+int(o['evaluation_draws_per_chain'])].sum())
        diagnostics.append(d)
    uniform=np.ones(len(target_draws))/len(target_draws)
    base_row,_=score(target_draws,uniform,p,'L0_joint')
    marginal_row,_=score(marginal[4],uniform,p,'L0_legacy_marginals')
    rows=[marginal_row,base_row,*rows]
    top=np.argsort(p.truth.ravel(),kind='mergesort')[-25:]
    repeat_error=np.sum((weighted[0]-weighted[1])[:,None]*target_draws,axis=0)
    repeat_top_rmse=float(np.sqrt(np.mean(repeat_error[top]**2)))
    repeat_global_rmse=float(np.sqrt(np.mean(repeat_error**2)))
    passed=all(d['ess']>=max(100,.1*len(target_draws)) and d['max_weight']<=.02 for d in diagnostics)
    passed=bool(passed and repeat_top_rmse<=.25 and repeat_global_rmse<=.1)
    checks=dict(trajectory=name,target_frame=n,time_s=float(p.times[n]),lag=1,lag_s=dt,
        ambient_from_initial_observation_K=p.ambient,source_lengthscale_past_only_m=p.source_lengthscale,
        prior_subblock_max_difference=block_difference,previous_rhat=pc['rhat_max'],joint_rhat=jc['rhat_max'],
        marginal_rhat=mc['rhat_max'],joint_vs_marginal_mean_rmse_K=float(np.sqrt(np.mean((joint[0]-marginal[0])**2))),
        joint_vs_marginal_mean_sd_mae_K=float(np.mean(abs(joint[1]-marginal[1]))),
        repeat_top1_mean_rmse_K=repeat_top_rmse,repeat_global_mean_rmse_K=repeat_global_rmse,
        repeat_weights_total_variation=float(.5*np.sum(abs(weighted[0]-weighted[1]))),
        reliability_gate_passed=passed,n_future_observations=len(next_obs['y_obs']),
        n_future_censored=int(next_obs['sat_mask'].sum()),
        current_camera_sha256=hashlib.sha256(observed[n].tobytes()).hexdigest(),
        **cooling_visibility(observed,ceiling,n,1,o['measurement_noise_sd'],fixed))
    out=OUT/name
    out.mkdir(exist_ok=True)
    for row in rows:
        row['trajectory']=name
        row['reliability_gate_passed']=passed
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'ess.csv',index=False)
    pd.DataFrame([checks]).to_csv(out/'checks.csv',index=False)
    np.savez_compressed(out/'pilot_particles.npz',target_draws=target_draws,loglik=loglik,
        pooled_weights=weighted[2],truth_evaluation_only=p.truth,ceiling=ceiling,
        target_observed=observed[n],future_observed=observed[n+1],target_frame=n)
    print(name,'gate',passed,'pooled ESS',diagnostics[-1]['ess'],flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    o=pd.read_csv(BASE/'fixed_configuration.csv').iloc[0].to_dict()
    pd.DataFrame([{**o,'target_rule':'midpoint','future_lag':1,'innovation_replicates':8,
        'gate_ess_min':100,'gate_ess_fraction_min':.1,'gate_max_weight':.02,
        'gate_repeat_top1_mean_rmse_K':.25,'gate_repeat_global_mean_rmse_K':.1}]).to_csv(OUT/'fixed_configuration.csv',index=False)
    thresholds=pd.read_csv(ROOT/'outputs/by_experiment/30_final_architecture_comparison/sequential_rerun_results.csv').groupby('trajectory').threshold_K.first().to_dict()
    catalog=trajectory_catalog(ROOT.parent/'heat_eq_laser_trajectories')
    for index,r in enumerate(catalog):
        if r.name in DEVELOPMENT_TRAJECTORIES and not (OUT/r.name/'pilot_particles.npz').exists():
            run(r.name,index,thresholds[r.name],o)
    for filename in ['metrics','ess','checks']:
        frame=pd.concat([pd.read_csv(OUT/name/f'{filename}.csv') for name in DEVELOPMENT_TRAJECTORIES])
        frame.to_csv(OUT/f'{filename}.csv',index=False)
    checks=pd.read_csv(OUT/'checks.csv')
    ess=pd.read_csv(OUT/'ess.csv')
    metrics=pd.read_csv(OUT/'metrics.csv')
    # Independent explicit-sum validation of saved scores (avoid BLAS vector-dot
    # floating-point status warnings on this platform).
    audits=[]
    for name in DEVELOPMENT_TRAJECTORIES:
        data=np.load(OUT/name/'pilot_particles.npz')
        check=checks[checks.trajectory==name].iloc[0]
        prepared=SimpleNamespace(truth=data['truth_evaluation_only'],ambient=check.ambient_from_initial_observation_K)
        for label,weights in [('L0_joint',np.ones(len(data['target_draws']))/len(data['target_draws'])),
                             ('L1_pooled',data['pooled_weights'])]:
            row,_=score(data['target_draws'],weights,prepared,label)
            saved=metrics[(metrics.trajectory==name)&(metrics.stage==label)].iloc[0]
            for key,value in row.items():
                if key=='stage':
                    continue
                difference=abs(value-float(saved[key]))
                assert np.isfinite(value) and difference<1e-9
                audits.append(dict(trajectory=name,stage=label,metric=key,difference=difference))
    pd.DataFrame(audits).to_csv(OUT/'saved_score_audit.csv',index=False)
    metrics.groupby('stage').mean(numeric_only=True).to_csv(OUT/'development_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    pooled=ess[ess.repeat=='pooled']
    axes[0].bar(pooled.trajectory,pooled.ess)
    axes[0].axhline(120,color='red',linestyle='--',label='ESS gate: 120 / 1200')
    axes[0].set_ylabel('Target-particle ESS')
    axes[0].legend()
    axes[1].bar(pooled.trajectory,pooled.max_weight)
    axes[1].axhline(.02,color='red',linestyle='--',label='Maximum-weight gate: 0.02')
    axes[1].set_ylabel('Maximum normalized weight')
    axes[1].legend()
    for ax in axes:
        ax.tick_params(axis='x',labelrotation=20,labelsize=8)
    fig.savefig(OUT/'ess_diagnostics.png',dpi=180)
    plt.close(fig)
    status=bool(checks.reliability_gate_passed.all())
    (OUT/'gate.json').write_text(json.dumps({'all_development_passed':status,
        'expanded_to_longer_lags':False,'evaluated_heldout':False},indent=2)+'\n')
    paths=[Path(__file__),*sorted((ROOT/'src').glob('*.py'))]
    (OUT/'source_fingerprints.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')


if __name__=='__main__':
    main()
