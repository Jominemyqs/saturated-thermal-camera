from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.ceiling_sweep import (artificial_ceiling_schedule,lower_camera_ceiling,
    matched_censoring_observations,SequentialAdvectiveConfig,sequential_prior_blocks,observation_indices)
from src.ceiling_model_comparison import rbf_blocks,evaluate_prediction
from src.pseudo_censoring_runner import _previous_prediction,_current_prediction
from src.thermal_posterior_physics import (DEVELOPMENT_TRAJECTORIES,prepare_trajectory,
    paired_camera_observations,posterior_physics_means)
from src.thermal_trajectory import trajectory_catalog

BASE=ROOT/'outputs/by_experiment/38_pseudo_censoring_calibration'
OUT=ROOT/'outputs/by_experiment/43_matched_ceiling_model_gap'
FRACTIONS=[.03,.05,.075,.10,.15,.20]
LABELS={'A':'Current-only ambient mean + RBF','B':'Posterior physics mean + RBF',
        'C':'Posterior physics mean + sequential stationary ST'}
METRICS=['field_excess_rel_l2','overall_rmse_K','top_1pct_rmse_K','overall_crps_K','top_1pct_crps_K',
    'censored_crps_K','top_1pct_bias_K','top_1pct_coverage_95','top_1pct_width_95_K','peak_absolute_error_K']


def sha(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def worker(payload):
    name,family,index,ceiling,o=payload
    folder=OUT/'by_trajectory'/name
    folder.mkdir(parents=True,exist_ok=True)
    if all((folder/f'level_{k}.json').exists() for k in range(6)):
        return name
    p,_=prepare_trajectory(ROOT.parent/'heat_eq_laser_trajectories',name,nx=int(o['nx']),ny=int(o['ny']),
        heat_flux_cutoff=o['heat_flux_cutoff'])
    n=len(p.times)-1
    reference=paired_camera_observations(p,previous_index=n-1,current_index=n,threshold=ceiling,
        observation_stride=int(o['observation_stride']),noise_sd=o['measurement_noise_sd'],seed=int(o['seed'])+10_000*index)
    schedule=artificial_ceiling_schedule(reference['frames'][1]['clipped_full'],ceiling,FRACTIONS)
    length=p.source_lengthscale*o['length_multiplier']
    saved=pd.read_csv(BASE/'results.csv')
    saved=saved[(saved.trajectory==name)&(saved.method=='uncalibrated')]
    region_saved=pd.read_csv(BASE/'region_results.csv')
    region_saved=region_saved[(region_saved.trajectory==name)&(region_saved.method=='uncalibrated')&(region_saved.region=='observed_censored')]
    config=SequentialAdvectiveConfig(signal_sd=o['signal_sd'],forcing_lengthscale=length*o['forcing_length_multiplier'],
        diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],current_noise_sd=o['current_noise_sd'],quadrature_order=int(o['quadrature_order']))
    for level in schedule:
        li=int(level['level_index'])
        path=folder/f'level_{li}.json'
        if path.exists():
            continue
        c=float(level['ceiling_K'])
        camera=lower_camera_ceiling(reference,c)
        obs=matched_censoring_observations(p,frame=camera['frames'][1],fixed_mask=camera['fixed_observation_mask'],threshold=c)
        previous,draws,prevcheck=_previous_prediction(p,frame=camera['frames'][0],fixed_mask=camera['fixed_observation_mask'],
            threshold=c,signal_sd=o['signal_sd'],lengthscale=length,noise_sd=o['previous_noise_sd'],
            options=o,seed=int(o['seed'])+50_000*index)
        mean,_,_=posterior_physics_means(p,previous,previous_index=n-1,current_index=n,diffusivity=o['diffusivity'],
            cooling_rate=o['cooling_rate'],source_coupling=o['source_coupling'],source_flux_threshold=o['source_flux_threshold'])
        b=rbf_blocks(p,obs,mean,o['signal_sd'],length,o['current_noise_sd'])
        a={**b,'prediction_mean':np.full(len(p.points),p.ambient),'observation_mean':np.full(len(obs['y_obs']),p.ambient)}
        cb,covcheck=sequential_prior_blocks(p,camera={**camera,'current':obs},previous_mean=previous,
            previous_draws=draws,mean_field=mean,displacement=None,config=config)
        np.testing.assert_array_equal(b['prediction_mean'],cb['prediction_mean'])
        rows=[]
        checks=[]
        audits=[]
        identity=dict(trajectory=name,family=family,role='development' if name in DEVELOPMENT_TRAJECTORIES else 'heldout',
            level_index=li,target_censored_fraction=FRACTIONS[li],ceiling_K=c,
            actual_censored_fraction=float(np.mean(obs['saturated_full'])))
        prevhash=sha(previous,draws)
        obshash=sha(obs['x_obs'],obs['y_obs'],obs['sat_mask'])
        for model,blocks in [('A',a),('B',b),('C',cb)]:
            seed=int(o['seed'])+100_000*index
            pred,check=_current_prediction(obs,blocks,options=o,seed=seed)
            metric=evaluate_prediction(pred,p.truth,p.ambient,obs['saturated_full'])
            rows.append({**identity,'model':model,'architecture':LABELS[model],**metric})
            checks.append({**identity,'model':model,'observation_sha256':obshash,
                'previous_posterior_sha256':prevhash if model!='A' else 'unused',
                'mean_sha256':sha(blocks['prediction_mean']),
                'covariance_sha256':sha(blocks['observed_covariance'],blocks['pred_observed_covariance'],blocks['prediction_variance']),
                'mean_prior_variance_K2':float(np.mean(blocks['prediction_variance'])),
                'n_observations':len(obs['y_obs']),'n_censored_observations':int(obs['sat_mask'].sum()),
                'current_seed':seed,'previous_rhat':prevcheck['rhat_max'] if model!='A' else None,
                **{f'current_{k}':v for k,v in check.items()},**covcheck})
            if model=='C':
                old=saved[saved.level_index==li]
                if len(old):
                    for key in METRICS:
                        if key=='censored_crps_K':
                            old_value=float(region_saved[region_saved.level_index==li].iloc[0].crps_K)
                        else:
                            old_value=float(old.iloc[0][key])
                        delta=metric[key]-old_value
                        assert abs(delta)<1e-8,(name,li,key,delta)
                        audits.append(dict(trajectory=name,level_index=li,metric=key,saved=old_value,reproduced=metric[key],difference=delta))
            print(f'{name}: level {li} {model} complete',flush=True)
        # Atomic per-level checkpoint, written only after all models and audit pass.
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(rows=rows,checks=checks,audits=audits),indent=2)+'\n')
        temporary.replace(path)
    return name


def summarize():
    cases=[json.loads(p.read_text()) for p in sorted((OUT/'by_trajectory').glob('*/level_*.json'))]
    results=pd.DataFrame([r for c in cases for r in c['rows']])
    checks=pd.DataFrame([r for c in cases for r in c['checks']])
    audits=pd.DataFrame([r for c in cases for r in c['audits']])
    assert len(results)==33*6*3 and len(audits)==30*6*len(METRICS)
    for _,g in checks.groupby(['trajectory','level_index']):
        assert g.observation_sha256.nunique()==1 and g.current_seed.nunique()==1
        assert g[g.model.isin(['B','C'])].previous_posterior_sha256.nunique()==1
        assert g[g.model.isin(['B','C'])].mean_sha256.nunique()==1
        assert g[g.model.isin(['A','B'])].covariance_sha256.nunique()==1
    results.to_csv(OUT/'results.csv',index=False)
    checks.to_csv(OUT/'implementation_checks.csv',index=False)
    audits.to_csv(OUT/'experiment38_equality_audit.csv',index=False)
    held=results[results.role=='heldout']
    summary=held.groupby(['level_index','model'])[METRICS+['actual_censored_fraction','ceiling_K']].mean().reset_index()
    summary.to_csv(OUT/'heldout30_summary.csv',index=False)
    held.groupby(['family','level_index','model'])[METRICS].mean().to_csv(OUT/'family_summary.csv')
    gaps=[]
    for (name,level),group in held.groupby(['trajectory','level_index']):
        indexed=group.set_index('model')
        for comparison,left,right in [('physics','A','B'),('sequential_covariance','B','C'),('total','A','C')]:
            for metric in METRICS:
                gaps.append(dict(trajectory=name,family=group.iloc[0].family,level_index=level,
                    actual_censored_fraction=group.iloc[0].actual_censored_fraction,
                    comparison=comparison,metric=metric,gap=float(indexed.loc[left,metric]-indexed.loc[right,metric])))
    gaps=pd.DataFrame(gaps)
    gaps.to_csv(OUT/'paired_gaps.csv',index=False)
    gaps.groupby(['level_index','comparison','metric']).gap.agg(['mean','median',lambda s:int((s>0).sum())]).rename(columns={'<lambda_0>':'positive_count'}).to_csv(OUT/'gap_summary.csv')
    endpoints=[]
    for family,frame in [('All',gaps),*gaps.groupby('family')]:
        for (comparison,metric),part in frame.groupby(['comparison','metric']):
            wide=part.pivot(index='trajectory',columns='level_index',values='gap')
            delta=(wide[5]-wide[0]).to_numpy()
            rng=np.random.default_rng(41043)
            boot=delta[rng.integers(0,len(delta),size=(4000,len(delta)))].mean(axis=1)
            endpoints.append(dict(family=family,comparison=comparison,metric=metric,reference_gap=wide[0].mean(),severe_gap=wide[5].mean(),
                mean_gap_growth=delta.mean(),median_gap_growth=np.median(delta),n_gap_increased=int((delta>0).sum()),n_trajectories=len(delta),
                bootstrap_ci_low=np.quantile(boot,.025),bootstrap_ci_high=np.quantile(boot,.975)))
    pd.DataFrame(endpoints).to_csv(OUT/'gap_growth.csv',index=False)
    plot_metrics=['field_excess_rel_l2','top_1pct_rmse_K','overall_crps_K','top_1pct_crps_K','censored_crps_K','top_1pct_bias_K']
    colors={'A':'#777777','B':'#0072B2','C':'#009E73'}
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ax,metric in zip(axes.ravel(),plot_metrics):
        for model,g in summary.groupby('model'):
            ax.plot(100*g.actual_censored_fraction,g[metric],'o-',label=model,color=colors[model])
        ax.set(title=metric.replace('_',' '),xlabel='Observed censored pixels (%)')
        ax.grid(alpha=.2)
    axes[0,0].legend(title='A: ambient RBF; B: physics RBF; C: physics ST',fontsize=8)
    fig.suptitle('Matched observations, frozen models, converged HMC: 30 held-out trajectories')
    fig.savefig(OUT/'performance_vs_censoring.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ax,metric in zip(axes.ravel(),plot_metrics):
        for comparison,g in gaps[gaps.metric==metric].groupby('comparison'):
            agg=g.groupby('level_index')[['actual_censored_fraction','gap']].mean()
            ax.plot(100*agg.actual_censored_fraction,agg.gap,'o-',label=comparison)
        ax.axhline(0,color='black',lw=.7)
        ax.set(title=metric.replace('_',' '),xlabel='Observed censored pixels (%)',ylabel='Paired gap')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('A-B: physics mean; B-C: covariance; A-C: total (positive favors right model for error/CRPS)')
    fig.savefig(OUT/'model_gaps_vs_censoring.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,metric in zip(axes.ravel(),['top_1pct_coverage_95','top_1pct_width_95_K','censored_coverage_95','censored_width_95_K']):
        for model,g in held.groupby('model'):
            agg=g.groupby('level_index')[[metric,'actual_censored_fraction']].mean()
            ax.plot(100*agg.actual_censored_fraction,agg[metric],'o-',color=colors[model],label=model)
        if 'coverage' in metric:
            ax.axhline(.95,color='black',ls='--',lw=.7)
        ax.set(title=metric.replace('_',' '),xlabel='Observed censored pixels (%)')
        ax.legend()
    fig.savefig(OUT/'coverage_and_width.png',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    o=pd.read_csv(BASE/'fixed_configuration.csv').iloc[0].to_dict()
    pd.DataFrame([{**o,'models':json.dumps(LABELS),'calibration_applied':False}]).to_csv(OUT/'fixed_configuration.csv',index=False)
    paths=[Path(__file__),*sorted((ROOT/'src').glob('*.py')),BASE/'fixed_configuration.csv']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest=OUT/'source_fingerprints.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==hashes,'Source changed: audit checkpoint compatibility before resuming'
    else:
        manifest.write_text(json.dumps(hashes,indent=2)+'\n')
    catalog=trajectory_catalog(ROOT.parent/'heat_eq_laser_trajectories')
    assert len(catalog)==33 and sum(r.name in DEVELOPMENT_TRAJECTORIES for r in catalog)==3
    thresholds=pd.read_csv(ROOT/'outputs/by_experiment/30_final_architecture_comparison/sequential_rerun_results.csv').groupby('trajectory').threshold_K.first().to_dict()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,(r.name,r.family,i,thresholds[r.name],o)) for i,r in enumerate(catalog)]
        for future in as_completed(futures):
            print('Finished',future.result(),flush=True)
    summarize()


if __name__=='__main__':
    main()
