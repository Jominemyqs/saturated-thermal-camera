from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.ceiling_sweep import (artificial_ceiling_schedule,lower_camera_ceiling,
    matched_censoring_observations,SequentialAdvectiveConfig,sequential_prior_blocks)
from src.ceiling_model_comparison import rbf_blocks,evaluate_prediction
from src.pseudo_censoring_runner import _previous_prediction,_current_prediction
from src.thermal_posterior_physics import (DEVELOPMENT_TRAJECTORIES,prepare_trajectory,
    paired_camera_observations,posterior_physics_means)
from src.thermal_trajectory import trajectory_catalog
from src.severity_diagnostics import (region_masks,mean_diagnostics,posterior_diagnostics,
    local_source_coordinates,directional_diagnostics)

BASE=ROOT/'outputs/by_experiment/43_matched_ceiling_model_gap'
OUT=ROOT/'outputs/by_experiment/45_censoring_severity_decomposition'
FRACTIONS=[.03,.05,.075,.1,.15,.2]


def sha(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def worker(payload):
    name,family,index,ceiling,o,levels=payload
    folder=OUT/'by_trajectory'/name
    folder.mkdir(parents=True,exist_ok=True)
    if all((folder/f'level_{li}.json').exists() for li in levels):
        return name
    p,_=prepare_trajectory(ROOT.parent/'heat_eq_laser_trajectories',name,nx=int(o['nx']),ny=int(o['ny']),heat_flux_cutoff=o['heat_flux_cutoff'])
    n=len(p.times)-1
    reference=paired_camera_observations(p,previous_index=n-1,current_index=n,threshold=ceiling,
        observation_stride=int(o['observation_stride']),noise_sd=o['measurement_noise_sd'],seed=int(o['seed'])+10_000*index)
    schedule=artificial_ceiling_schedule(reference['frames'][1]['clipped_full'],ceiling,FRACTIONS)
    length=p.source_lengthscale*o['length_multiplier']
    config=SequentialAdvectiveConfig(signal_sd=o['signal_sd'],forcing_lengthscale=length*o['forcing_length_multiplier'],
        diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],current_noise_sd=o['current_noise_sd'],quadrature_order=int(o['quadrature_order']))
    coordinates,direction_status=local_source_coordinates(p,o['source_flux_threshold'])
    for level in schedule:
        li=int(level['level_index'])
        if li not in levels or (folder/f'level_{li}.json').exists():
            continue
        c=float(level['ceiling_K'])
        camera=lower_camera_ceiling(reference,c)
        obs=matched_censoring_observations(p,frame=camera['frames'][1],fixed_mask=camera['fixed_observation_mask'],threshold=c)
        identity=dict(trajectory=name,family=family,role='development' if name in DEVELOPMENT_TRAJECTORIES else 'heldout',
            level_index=li,ceiling_K=c,actual_censored_fraction=float(np.mean(obs['saturated_full'])))
        previous,draws,pc=_previous_prediction(p,frame=camera['frames'][0],fixed_mask=camera['fixed_observation_mask'],threshold=c,
            signal_sd=o['signal_sd'],lengthscale=length,noise_sd=o['previous_noise_sd'],options=o,seed=int(o['seed'])+50_000*index)
        prior,_,_=posterior_physics_means(p,previous,previous_index=n-1,current_index=n,diffusivity=o['diffusivity'],
            cooling_rate=o['cooling_rate'],source_coupling=o['source_coupling'],source_flux_threshold=o['source_flux_threshold'])
        old=json.loads((BASE/'by_trajectory'/name/f'level_{li}.json').read_text())
        oldrows={r['model']:r for r in old['rows']}
        oldchecks={r['model']:r for r in old['checks']}
        common_hash=dict(observation_sha256=sha(obs['x_obs'],obs['y_obs'],obs['sat_mask']),
            previous_posterior_sha256=sha(previous,draws),mean_sha256=sha(prior.ravel()))
        for key,value in common_hash.items():
            assert value==oldchecks['B'][key],(name,li,key,'hash mismatch')
        assert c==oldrows['B']['ceiling_K']
        masks=region_masks(p.truth,obs['saturated_full'])
        means=[{**identity,'model':'shared_physics_prior','stage':'prior',**r} for r in mean_diagnostics(prior,p.truth,masks)]
        shape_rows=[]; checks=[]; audits=[]; current_draws={}
        b=rbf_blocks(p,obs,prior,o['signal_sd'],length,o['current_noise_sd'])
        cb,_=sequential_prior_blocks(p,camera={**camera,'current':obs},previous_mean=previous,previous_draws=draws,
            mean_field=prior,displacement=None,config=config)
        for model,blocks in [('B',b),('C',cb)]:
            covariance_hash=sha(blocks['observed_covariance'],blocks['pred_observed_covariance'],blocks['prediction_variance'])
            assert covariance_hash==oldchecks[model]['covariance_sha256'],(name,li,model,'covariance mismatch')
            pred,cc=_current_prediction(obs,blocks,options=o,seed=int(o['seed'])+100_000*index)
            metrics=evaluate_prediction(pred,p.truth,p.ambient,obs['saturated_full'])
            for key,value in metrics.items():
                delta=value-oldrows[model][key]
                assert abs(delta)<1e-8,(name,li,model,key,delta)
                audits.append({**identity,'model':model,'metric':key,'saved':oldrows[model][key],'reproduced':value,'difference':delta})
            means.extend({**identity,'model':model,'stage':'posterior',**r} for r in mean_diagnostics(pred[0],p.truth,masks))
            diagnostics,fields=posterior_diagnostics(pred,p.truth,masks)
            shape_rows.extend({**identity,'model':model,**r} for r in diagnostics)
            checks.append({**identity,'model':model,**common_hash,'covariance_sha256':covariance_hash,
                'previous_rhat':pc['rhat_max'],'current_rhat':cc['rhat_max'],
                'current_seed':int(o['seed'])+100_000*index,'direction_status':direction_status})
            current_draws.update({model+'_draws':pred[4],model+'_mean':pred[0],model+'_sd':pred[1],
                model+'_lower':pred[2],model+'_upper':pred[3],**{model+'_'+key:value for key,value in fields.items()}})
            print(name,'level',li,model,'exact replay passed',flush=True)
        path_rows=[{**identity,**r} for r in directional_diagnostics(prior,p.truth,coordinates,p.source_lengthscale)] if li in [0,3,5] else []
        arrays=dict(physics_prior=prior,previous_mean=previous,truth_evaluation_only=p.truth,
            top25_mask=masks['top_1pct'],observed_censored_mask=masks['observed_censored'],
            current_camera=camera['frames'][1]['clipped_full'],previous_camera=camera['frames'][0]['clipped_full'],
            observation_points=obs['x_obs'],observation_values=obs['y_obs'],observation_censored=obs['sat_mask'],
            grid_points=p.points,ceiling=c,**current_draws)
        if coordinates is not None:
            arrays.update({'path_'+k:v for k,v in coordinates.items()})
        # Marginal samples remain float64; no joint-field claim is made.
        temp=folder/f'level_{li}.tmp.npz'
        np.savez_compressed(temp,**arrays)
        temp.replace(folder/f'level_{li}.npz')
        temp=folder/f'level_{li}.tmp.json'
        temp.write_text(json.dumps(dict(means=means,posterior=shape_rows,checks=checks,audits=audits,directional=path_rows),indent=2))
        temp.replace(folder/f'level_{li}.json')
    return name


def summarize(require_complete=True):
    cases=[json.loads(p.read_text()) for p in sorted((OUT/'by_trajectory').glob('*/level_*.json'))]
    if require_complete:
        assert len(cases)==33*6
    frames={}
    for key in ['means','posterior','checks','audits','directional']:
        frame=pd.DataFrame([r for case in cases for r in case[key]])
        frame.to_csv(OUT/f'{key}.csv',index=False)
        frames[key]=frame
    for key in ['means','posterior']:
        held=frames[key][frames[key].role=='heldout']
        for groups,suffix in [(['level_index','model','region'],'heldout30'),(['family','level_index','model','region'],'family')]:
            held.groupby(groups).mean(numeric_only=True).to_csv(OUT/f'{key}_{suffix}_summary.csv')
    # Per-trajectory changes from reference preserve pairing.
    for key in ['means','posterior']:
        frame=frames[key]
        columns=frame.select_dtypes(include='number').columns.difference(['level_index','ceiling_K','actual_censored_fraction','n_pixels'])
        reference=frame[frame.level_index==0].set_index(['trajectory','model','region'])
        changes=[]
        for _,r in frame.iterrows():
            ref=reference.loc[(r.trajectory,r.model,r.region)]
            changes.append({k:r[k] for k in ['trajectory','family','role','level_index','model','region']} |
                {k+'_change':r[k]-ref[k] for k in columns})
        pd.DataFrame(changes).to_csv(OUT/f'{key}_paired_changes.csv',index=False)
    m=pd.read_csv(OUT/'means_heldout30_summary.csv'); p=pd.read_csv(OUT/'posterior_heldout30_summary.csv')
    colors={'B':'#0072B2','C':'#009E73','shared_physics_prior':'#666666'}
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    top=m[m.region=='top_1pct']; pt=p[p.region=='top_1pct']
    prior=top[top.model=='shared_physics_prior']
    axes[0,0].plot(prior.actual_censored_fraction*100,prior.bias_K,'o-',color=colors['shared_physics_prior'],label='Shared physics prior')
    for model in ['B','C']:
        r=top[top.model==model]; q=pt[pt.model==model]
        axes[0,1].plot(r.actual_censored_fraction*100,r.bias_K,'o-',color=colors[model],label=model)
        axes[0,2].plot(q.actual_censored_fraction*100,q.z_mean,'o-',color=colors[model],label=model)
        axes[1,0].plot(q.actual_censored_fraction*100,q.z_sd,'o-',color=colors[model],label=model)
        axes[1,1].plot(q.actual_censored_fraction*100,q.pit_upper_95_fraction,'o-',color=colors[model],label=model+' upper')
        axes[1,1].plot(q.actual_censored_fraction*100,q.pit_lower_05_fraction,'s--',color=colors[model],label=model+' lower')
        axes[1,2].plot(r.actual_censored_fraction*100,r.rmse_K,'o-',color=colors[model],label=model)
    axes[1,2].plot(prior.actual_censored_fraction*100,prior.rmse_K,':',color=colors['shared_physics_prior'],label='Prior')
    for ax,title in zip(axes.ravel(),['Physics-prior top-1% bias (K)','Posterior top-1% bias (K)',
        'Top-1% mean standardized error z','Top-1% SD of standardized error z',
        'Top-1% extreme posterior-quantile fractions','Top-1% RMSE (K)']):
        ax.set(title=title,xlabel='Observed censoring (%)'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    for ax in axes[0,:]: ax.axhline(0,color='black',lw=.6)
    axes[1,0].axhline(1,color='black',lw=.6,ls=':')
    fig.suptitle('Frozen Experiment 43: 30 held-out trajectories; descriptive hottest-tail diagnostics\nB: posterior physics + RBF; C: posterior physics + sequential ST; no refitting')
    fig.savefig(OUT/'severity_decomposition.png',dpi=180); plt.close(fig)
    path=frames['directional']
    if not path.empty:
        held=path[path.role=='heldout']
        agg=held.groupby(['level_index','axis','bin_center']).agg(bias_K=('bias_K','mean'),
            trajectories=('trajectory','nunique'),pixel_count=('n_pixels','sum')).reset_index()
        agg.to_csv(OUT/'directional_summary.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
        for ax,axis in zip(axes,['along','cross']):
            for level,label in [(0,'Reference'),(3,'10%'),(5,'20%')]:
                r=agg[(agg.axis==axis)&(agg.level_index==level)]
                ax.plot(r.bin_center,r.bias_K,'o-',label=label)
            ax.axhline(0,color='black',lw=.6); ax.axvline(0,color='gray',ls=':')
            ax.set(xlabel=axis+' distance / source lengthscale',ylabel='Physics-prior bias (K)')
            ax.legend(); ax.grid(alpha=.2)
        fig.suptitle('Source-relative physics residuals; transverse strip width +/-2 source lengthscales\nNegative along-coordinate = behind; equal trajectory weight within each populated bin')
        fig.savefig(OUT/'directional_physics_residual.png',dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--trajectory',default=None)
    parser.add_argument('--levels',type=int,nargs='+',default=list(range(6)))
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.summarize_only:
        summarize(); return
    o=pd.read_csv(BASE/'fixed_configuration.csv').iloc[0].to_dict()
    pd.DataFrame([o]).to_csv(OUT/'fixed_configuration.csv',index=False)
    files=[Path(__file__),*sorted((ROOT/'src').glob('*.py')),BASE/'fixed_configuration.csv']
    fingerprint={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest=OUT/'source_fingerprints.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==fingerprint,'Changed code: audit before resuming'
    else:
        manifest.write_text(json.dumps(fingerprint,indent=2))
    catalog=trajectory_catalog(ROOT.parent/'heat_eq_laser_trajectories')
    assert len(catalog)==33 and sum(r.name in DEVELOPMENT_TRAJECTORIES for r in catalog)==3
    thresholds=pd.read_csv(BASE/'results.csv'); thresholds=thresholds[thresholds.level_index==0].groupby('trajectory').ceiling_K.first().to_dict()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,(r.name,r.family,i,thresholds[r.name],o,args.levels)) for i,r in enumerate(catalog)
                 if args.trajectory is None or r.name==args.trajectory]
        for future in as_completed(futures): print('Completed',future.result(),flush=True)
    if args.trajectory is None and len(args.levels)==6:
        summarize()


if __name__=='__main__': main()
