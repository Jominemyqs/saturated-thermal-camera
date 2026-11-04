"""Output 61: matched current C updates after the previous-state unpinning audit."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.censored_gp import RBFConfig
from src.matched_projection import MatchedHeatOperator
from src.previous_unpinning_audit import noisy_precision
from src.unpinning_current_update import propagated_blocks
from src.stochastic_heat_gp import StochasticHeatConfig, finite_step_innovation_covariance
from src.native_ceiling_diagnostics import retained_reference_observations
from src.native_mask_pilot import sample_mixed_noisy_blocks, spread_nodes
from src.pseudo_censoring_runner import _current_prediction, prediction_from_draws
from src.mcmc_diagnostics import chain_diagnostics
from src.metrics import empirical_crps

SPEC=importlib.util.spec_from_file_location('mask_readonly_helpers',ROOT/'scripts/58_thermal_native_non_saturation_mask.py')
HELP=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(HELP)
BASE=ROOT/'outputs/by_experiment/52_native_higher_ceiling'
RET=ROOT/'outputs/by_experiment/53_native_observation_retention'
PREV=ROOT/'outputs/by_experiment/60_previous_unpinning_audit'
OUT=ROOT/'outputs/by_experiment/61_unpinning_current_update'
NAMES=['DiagonalScanPath_8','SpiralScanPath_13']
ARMS=['legacy_hybrid','all_readings_pinned','all_readings_latent']
GATE=dict(rank_rhat_max=1.01,ess_bulk_min=400.,ess_tail_min=400.,classical_rhat_max=1.05,
          arm_seconds_max=900.,peak_RSS_MiB_max=4096.)
SAMPLES,BURN,EVAL=800,300,300
REPEAT_OFFSET=7_000_019
SUFFIXES=['mean','sd','lower','upper','draws']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def load(path):
    with np.load(path) as d:return {k:d[k] for k in d.files}


def folder(name):return OUT/'by_trajectory'/name


def stem(name,level,arm,repeat):return folder(name)/f'level_{level}_{arm}_repeat_{repeat}'


def options():return pd.read_csv(BASE/'frozen_configuration.csv').iloc[0].to_dict()


def prepare():
    if OUT.exists():raise FileExistsError(f'Refusing to overwrite {OUT}')
    for name,h in json.loads((PREV/'source_fingerprints.json').read_text()).items():
        assert sha(ROOT/name)==h,name
    assert all(json.loads((PREV/'verification.json').read_text())['reference_gate'].values())
    o=options();meta=pd.read_csv(BASE/'verification.csv')
    times={d['trajectory']:d['dt_s'] for d in json.loads((ROOT/'outputs/by_experiment/56_within_step_integration/gradient_provenance.json').read_text())}
    assert len(set(times.values()))==1
    paths=[Path(__file__),ROOT/'src/unpinning_current_update.py',ROOT/'tests/test_unpinning_current_update.py',
           ROOT/'src/previous_unpinning_audit.py',ROOT/'src/previous_correlation_audit.py',
           ROOT/'src/native_mask_pilot.py',ROOT/'src/native_ceiling_diagnostics.py',
           ROOT/'src/matched_projection.py',ROOT/'src/stochastic_heat_gp.py',ROOT/'src/truncated_gaussian.py',
           ROOT/'src/dense_censored_gp.py',ROOT/'src/pseudo_censoring_runner.py',ROOT/'src/metrics.py',
           ROOT/'scripts/58_thermal_native_non_saturation_mask.py',BASE/'frozen_configuration.csv',
           BASE/'verification.csv',PREV/'verification.json',PREV/'frozen_configuration.json',
           ROOT/'outputs/by_experiment/56_within_step_integration/frozen_sections.csv',
           ROOT/'outputs/by_experiment/56_within_step_integration/gradient_provenance.json']
    for name in NAMES:
        for level in range(3):
            paths += [BASE/'by_trajectory'/name/f'level_{level}.npz',
                      PREV/'by_trajectory'/name/f'level_{level}_summaries.npz',
                      PREV/'by_trajectory'/name/f'level_{level}_attempt_0.npz']
            if level:paths.append(RET/'by_trajectory'/name/f'level_{level}.npz')
    OUT.mkdir(parents=True)
    dump(OUT/'source_and_input_fingerprints.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    dump(OUT/'frozen_configuration.json',dict(experiment=61,driver=str(Path(__file__).relative_to(ROOT)),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        arms=ARMS,trajectories=NAMES,ceilings=['reference',320,325],inherited_settings=o,
        current_observations='fixed numerical candidates UNION observed reference saturation; numerical when unsaturated, lower bound otherwise',
        current_noise_sd_K=.50,camera_noise_sd_K=.25,previous_noise_sd_K=.25,
        chains=4,samples_per_chain=SAMPLES,burn_in=BURN,thin=1,evaluation_per_chain=EVAL,
        retries=0,reference_repeats=2,high_ceiling_repeats=1,repeat_seed_offset=REPEAT_OFFSET,
        gate=GATE,high_ceiling_gate='All six reference arm/repeat fits per trajectory pass',
        previous_inference_rerun=False,covariance='Q + A [Cov(conditional means) + full correlated C] A^T; pinned bridge masks previous unsaturated coordinates',
        final_draws='pointwise marginal draws, NOT coherent field samples; no further propagation',
        source_change=False,mask_flags=False,parameter_fit=False,
        budget_note='New bounded task; does not change output-55 historical 300-second stop'))
    start=perf_counter()
    with np.load(BASE/'by_trajectory'/NAMES[0]/'level_0.npz') as d:
        xy=d['points'];triangles=d['triangles'];ambient=float(d['ambient'])
    n=len(xy);assert 10*n*n*8/2**20<GATE['peak_RSS_MiB_max']
    assert meta.lengthscale_m.nunique()==1
    cfg=RBFConfig(mean_temp=ambient,signal_sd=o['signal_sd'],lengthscale=float(meta.lengthscale_m.iloc[0]),noise_sd=o['previous_noise_sd'])
    p,v,inverse_error=noisy_precision(xy,cfg)
    physics=dict(dt=times[NAMES[0]],diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'])
    op=MatchedHeatOperator(SimpleNamespace(points=xy,triangles=triangles),
        np.linspace(xy[:,0].min(),xy[:,0].max(),int(o['nx'])),np.linspace(xy[:,1].min(),xy[:,1].max(),int(o['ny'])))
    chunks=[]
    for first in range(0,n,32):
        ids=np.arange(first,min(first+32,n));eye=np.zeros((len(ids),n));eye[np.arange(len(ids)),ids]=1
        chunks.append(csr_matrix(op.propagate(eye,'native',**physics).T))
    a=hstack(chunks,format='csr');del chunks,eye
    ap=a@p
    hcfg=StochasticHeatConfig(signal_sd=o['signal_sd'],forcing_lengthscale=cfg.lengthscale*o['forcing_length_multiplier'],
        diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],quadrature_order=int(o['quadrature_order']))
    qdiag=float(finite_step_innovation_covariance(xy[:1],xy[:1],hcfg,physics['dt'])[0,0])
    checks=[]
    for name in NAMES:
        folder(name).mkdir(parents=True)
        with np.load(BASE/'by_trajectory'/name/'level_0.npz') as d:
            refcam=d['camera'];refceiling=float(d['ceiling'])
        for level in range(3):
            with np.load(BASE/'by_trajectory'/name/f'level_{level}.npz') as f:
                d={k:f[k] for k in ['points','triangles','previous_draws','previous_mean','physics_mean','C_inherited_variance',
                   'C_prior_variance','camera','current_signal','numeric_candidates','ceiling','truth','area_weights','hot_area_weights','gradient_contribution']}
            prev=load(PREV/'by_trajectory'/name/f'level_{level}_summaries.npz')
            with np.load(PREV/'by_trajectory'/name/f'level_{level}_attempt_0.npz') as f:
                z=f['noisy_censored_chains'];s=f['censored_ids']
            np.testing.assert_array_equal(xy,d['points'])
            np.testing.assert_array_equal(d['gradient_contribution'],prev['gradient_contribution'])
            ids,numeric,sat=retained_reference_observations(refcam,refceiling,d['camera'],float(d['ceiling']),d['numeric_candidates'])
            if level:
                with np.load(RET/'by_trajectory'/name/f'level_{level}.npz') as f:np.testing.assert_array_equal(ids,f['retained_observation_ids'])
            else:
                with np.load(BASE/'by_trajectory'/name/'level_0.npz') as f:np.testing.assert_array_equal(ids,f['observation_ids'])
            qpo=finite_step_innovation_covariance(xy,xy[ids],hcfg,physics['dt'])
            qoo=finite_step_innovation_covariance(xy[ids],xy[ids],hcfg,physics['dt'])
            row=meta[(meta.trajectory==name)&(meta.level==level)&(meta.model=='C')].iloc[0]
            assert HELP.digest(d['previous_draws'])==row.previous_draws_sha256
            kinds=sat[ids].astype(int);values=np.where(kinds==0,d['camera'][ids],np.nan)
            case={k:d[k] for k in ['points','triangles','camera','ceiling','truth','area_weights','hot_area_weights','gradient_contribution']}
            case.update(observation_ids=ids,kinds=kinds,values=values,seed=int(row.current_seed))
            np.savez_compressed(folder(name)/f'case_{level}.npz',**case)
            for arm in ARMS:
                mean=prev['forecast_'+arm]
                if arm=='legacy_hybrid':
                    h=op.propagate(d['previous_draws']-d['previous_mean'],'native',**physics).T/np.sqrt(len(d['previous_draws'])-1)
                    oo=HELP.gram(h[ids],h[ids]);po=HELP.gram(h,h[ids]);diag=np.sum(h*h,axis=1)
                    np.testing.assert_array_equal(diag,d['C_inherited_variance'])
                    np.testing.assert_array_equal(mean,d['physics_mean']);del h
                else:
                    oo,po,diag=propagated_blocks(a,p,v,s,z,ids,pinned=arm=='all_readings_pinned',operator_precision=ap)
                diag_error=float(abs(diag-prev['inherited_variance_'+arm]).max())
                np.testing.assert_allclose(diag,prev['inherited_variance_'+arm],atol=1e-10,rtol=1e-10)
                oo+=qoo;po+=qpo;diag+=qdiag
                np.testing.assert_allclose(po[ids],oo,atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(np.diag(oo),diag[ids],atol=1e-10,rtol=1e-10)
                if arm=='legacy_hybrid':np.testing.assert_array_equal(diag,d['C_prior_variance'])
                np.savez_compressed(folder(name)/f'prior_{level}_{arm}.npz',prediction_mean=mean,observation_mean=mean[ids],
                    observed_covariance=oo,pred_observed_covariance=po,prediction_variance=diag)
                checks.append(dict(trajectory=name,level=level,arm=arm,previous_diagonal_max_error_K2=diag_error,
                    observations=len(ids),numeric=int((kinds==0).sum()),censored=int((kinds==1).sum()),
                    current_camera_sha256=HELP.digest(d['camera']),observation_ids_sha256=HELP.digest(ids),
                    source_sha256=HELP.digest(d['gradient_contribution']),Q_cross_sha256=HELP.digest(qpo),
                    mean_sha256=HELP.digest(mean),prior_diagonal_sha256=HELP.digest(diag),
                    prior_cross_sha256=HELP.digest(po),prior_observed_sha256=HELP.digest(oo)))
                del oo,po,diag
            del d,prev,z,qpo,qoo
            print('PREPARED',name,level,flush=True)
    pd.DataFrame(checks).to_csv(OUT/'prior_verification.csv',index=False)
    dump(OUT/'preparation.json',dict(seconds=perf_counter()-start,inverse_error=inverse_error,peak_RSS_MiB=HELP.rss(),
        previous_inference_rerun=False,full_covariance_blocks=True,native_nodes=n))
    assert HELP.rss()<=GATE['peak_RSS_MiB_max']


def replay(name,level):
    d=load(folder(name)/f'case_{level}.npz');b=load(folder(name)/f'prior_{level}_legacy_hybrid.npz')
    obs=dict(y_obs=d['camera'][d['observation_ids']],sat_mask=d['kinds']==1,
             saturated_full=d['camera']==d['ceiling'],threshold=float(d['ceiling']))
    start=perf_counter();pred,diagnostic=_current_prediction(obs,b,options=options(),seed=int(d['seed']))
    path=(BASE if level==0 else RET)/'by_trajectory'/name/f'level_{level}.npz'
    prefix='C_' if level==0 else 'retained_'
    with np.load(path) as f:
        for suffix,actual in zip(SUFFIXES,pred):np.testing.assert_array_equal(actual,f[prefix+suffix])
    dump(folder(name)/f'replay_{level}.json',dict(trajectory=name,level=level,exact=True,
        source=str(path.relative_to(ROOT)),seconds=perf_counter()-start,**diagnostic))
    print('EXACT REPLAY',name,level,flush=True)


def regions(d,name):
    w=d['area_weights'];ids=d['observation_ids'];unused=np.ones(len(w),bool);unused[ids]=False
    sat=d['camera']==d['ceiling']
    result=dict(whole_area=w,hot1pct_area=d['hot_area_weights'],observed_censored=w*sat,
                observed_unsaturated=w*~sat,current_unused=w*unused)
    sections=pd.read_csv(ROOT/'outputs/by_experiment/56_within_step_integration/frozen_sections.csv')
    for section,g in sections[sections.trajectory==name].groupby('section',sort=False):
        r=np.zeros_like(w);indices=g[g.shoulder].node.to_numpy(int);r[indices]=w[indices]
        result['shoulder_'+('along' if 'Along' in section else 'across')]=r
    return result


def metrics(d,pred,name):
    mean,sd,lo,hi,draws=pred;truth=d['truth'];err=mean-truth;crps=empirical_crps(draws,truth)
    peak=int(np.argmax(truth));rows=[]
    for region,w in regions(d,name).items():
        if w.sum()==0:continue
        avg=lambda a:float(np.sum(w*a)/w.sum())
        rows.append(dict(region=region,nodes=int((w>0).sum()),area_m2=float(w.sum()),
            rmse_K=np.sqrt(avg(err**2)),mae_K=avg(abs(err)),bias_K=avg(err),crps_K=avg(crps),
            coverage95=avg((lo<=truth)&(truth<=hi)),width95_K=avg(hi-lo),posterior_sd_K=avg(sd),
            true_peak_error_K=float(err[peak]),true_peak_lower_K=float(lo[peak]),true_peak_upper_K=float(hi[peak]),
            true_peak_K=float(truth[peak]),true_peak_covered=bool(lo[peak]<=truth[peak]<=hi[peak])))
    return rows


def fit(name,level,arm,repeat):
    path=stem(name,level,arm,repeat)
    if path.with_suffix('.npz').exists():raise FileExistsError(path)
    d=load(folder(name)/f'case_{level}.npz');b=load(folder(name)/f'prior_{level}_{arm}.npz')
    monitor=np.union1d(d['observation_ids'],spread_nodes(d['points'],np.ones(len(d['points']),bool),64))
    start=perf_counter();latent=[];noisy=[]
    for chain in range(4):
        pred,z=sample_mixed_noisy_blocks(ids=d['observation_ids'],kinds=d['kinds'],numerical_values=d['values'],
            thresholds=float(d['ceiling']),**b,noise_sd=.5,n_samples=SAMPLES,burn_in=BURN,thin=1,
            seed=int(d['seed'])+repeat*REPEAT_OFFSET+100000*chain)
        latent.append(pred[4]);noisy.append(z)
        print('CHAIN',name,level,arm,repeat,chain,flush=True)
    latent=np.asarray(latent);noisy=np.asarray(noisy)
    classic=chain_diagnostics(latent,selected_mask=d['camera']==d['ceiling']);nc=chain_diagnostics(noisy)
    diagnostics=pd.concat([HELP.arviz_diagnostics(noisy,d['observation_ids'][d['kinds']==1],'noisy_censored'),
        HELP.arviz_diagnostics(latent[:,:,monitor],monitor,'latent_temperature')],ignore_index=True)
    diagnostics.to_csv(path.with_suffix('.diagnostics.csv'),index=False)
    evaluation=latent[:,np.linspace(0,SAMPLES-1,EVAL,dtype=int),:]
    pred=prediction_from_draws(evaluation.reshape(-1,latent.shape[-1]))
    common=dict(trajectory=name,level=level,ceiling_K=float(d['ceiling']),arm=arm,repeat=repeat)
    result=metrics(d,pred,name)
    pd.DataFrame([dict(**common,**r) for r in result]).to_csv(path.with_suffix('.metrics.csv'),index=False)
    summary=dict(**common,seconds=perf_counter()-start,peak_RSS_MiB=HELP.rss(),
        rank_rhat_max=float(diagnostics.rank_rhat.max()),ess_bulk_min=float(diagnostics.ess_bulk.min()),
        ess_tail_min=float(diagnostics.ess_tail.min()),classical_rhat_max=max(classic['rhat_max'],nc['rhat_max']),
        finite=bool(np.isfinite(diagnostics[['rank_rhat','ess_bulk','ess_tail']]).all().all()),
        samples_per_chain=SAMPLES,burn_in=BURN,current_seed=int(d['seed'])+repeat*REPEAT_OFFSET,
        mean_sha256=HELP.digest(b['prediction_mean']),cross_sha256=HELP.digest(b['pred_observed_covariance']),
        obs_ids_sha256=HELP.digest(d['observation_ids']),noise_sd_K=.5)
    summary['passed']=bool(summary['finite'] and all(summary[k]<=GATE[k] for k in ['rank_rhat_max','classical_rhat_max'])
        and summary['ess_bulk_min']>=GATE['ess_bulk_min'] and summary['ess_tail_min']>=GATE['ess_tail_min']
        and summary['seconds']<=GATE['arm_seconds_max'] and summary['peak_RSS_MiB']<=GATE['peak_RSS_MiB_max'])
    np.savez_compressed(path.with_suffix('.npz'),**dict(zip(SUFFIXES,pred)),evaluation_chains=evaluation,
        monitored_chains=latent[:,:,monitor],monitor_ids=monitor,noisy_censored_chains=noisy,
        constrained_ids=d['observation_ids'][d['kinds']==1])
    dump(path.with_suffix('.json'),summary)
    print('FIT',summary,flush=True)


def collect():
    summaries=[];tables=[]
    for name in NAMES:
        for path in sorted(folder(name).glob('level_*_repeat_*.json')):
            summaries.append(json.loads(path.read_text()));tables.append(pd.read_csv(path.with_suffix('.metrics.csv')))
    pd.DataFrame(summaries).to_csv(OUT/'convergence_runtime.csv',index=False)
    data=pd.concat(tables,ignore_index=True);data.to_csv(OUT/'results.csv',index=False)
    paired=[]
    for keys,g in data.groupby(['trajectory','level','repeat','region']):
        g=g.set_index('arm')
        for before,after,label in [('legacy_hybrid','all_readings_pinned','conditioning'),
                                   ('all_readings_pinned','all_readings_latent','unpinning'),
                                   ('legacy_hybrid','all_readings_latent','combined')]:
            if before not in g.index or after not in g.index:continue
            paired.append(dict(zip(['trajectory','level','repeat','region'],keys),comparison=label,
                **{m:float(g.loc[after,m]-g.loc[before,m]) for m in ['rmse_K','bias_K','crps_K','coverage95','width95_K','posterior_sd_K','true_peak_error_K']}))
    pd.DataFrame(paired).to_csv(OUT/'paired_changes.csv',index=False)
    for name in NAMES:
        for level in range(3):
            if all(stem(name,level,arm,0).with_suffix('.npz').exists() for arm in ARMS):plot(name,level)
    return summaries


def plot(name,level):
    d=load(folder(name)/f'case_{level}.npz')
    sections=pd.read_csv(ROOT/'outputs/by_experiment/56_within_step_integration/frozen_sections.csv')
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True,constrained_layout=True)
    errors=[];sds=[]
    for col,arm in enumerate(ARMS):
        f=load(stem(name,level,arm,0).with_suffix('.npz'))
        with np.load(folder(name)/f'prior_{level}_{arm}.npz') as p:mean=p['prediction_mean']
        errors.append(f['mean']-d['truth']);sds.append(f['sd'])
        for row,(section,g) in enumerate(sections[sections.trajectory==name].groupby('section',sort=False)):
            g=g.sort_values('along_mm');ids=g.node.to_numpy(int);x=g.along_mm
            ax=axes[row,col];ax.plot(x,d['truth'][ids],'k',label='Native truth')
            ax.plot(x,mean[ids],'--',color='#D55E00',label='Physics forecast')
            ax.plot(x,f['mean'][ids],color='#0072B2',label='Current posterior')
            ax.fill_between(x,f['lower'][ids],f['upper'][ids],color='#0072B2',alpha=.2,label='Pointwise 95% interval')
            ax.axhline(float(d['ceiling']),color='#777777',ls=':')
            ax.set(title=arm.replace('_',' ')+' | '+('Along' if row==0 else 'Across'),
                xlabel='Distance from fixed current peak (mm)',ylabel='Temperature (K)');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'{name}; ceiling {float(d["ceiling"]):.3f} K\nSame retained current likelihood; previous-state representation changes. Marginal intervals only.')
    fig.savefig(OUT/f'cross_sections_{name}_level_{level}.png',dpi=150);plt.close(fig)
    if level==0:
        fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
        tri=mtri.Triangulation(d['points'][:,0]*1000,d['points'][:,1]*1000,d['triangles'])
        limit=max(np.max(abs(e)) for e in errors);sdmax=max(np.max(s) for s in sds)
        for col,arm in enumerate(ARMS):
            err=axes[0,col].tripcolor(tri,errors[col],cmap='RdBu_r',vmin=-limit,vmax=limit,shading='gouraud')
            sd=axes[1,col].tripcolor(tri,sds[col],cmap='viridis',vmin=0,vmax=sdmax,shading='gouraud')
            for ax in axes[:,col]:ax.set_aspect('equal');ax.set(xlabel='x (mm)',ylabel='y (mm)')
            axes[0,col].set_title(arm.replace('_',' '))
        fig.colorbar(err,ax=axes[0,:],label='Posterior mean minus truth (K)',shrink=.8)
        fig.colorbar(sd,ax=axes[1,:],label='Posterior marginal SD (K)',shrink=.8)
        fig.suptitle(f'{name}: shared-scale signed error and SD; reference ceiling')
        fig.savefig(OUT/f'signed_errors_{name}.png',dpi=150);plt.close(fig)


def run():
    start=perf_counter()
    def call(phase,name=None,level=None,arm=None,repeat=None,timeout=900):
        args=[sys.executable,str(Path(__file__)),'--phase',phase]
        for key,value in [('name',name),('level',level),('arm',arm),('repeat',repeat)]:
            if value is not None:args+=['--'+key,str(value)]
        subprocess.run(args,cwd=ROOT,check=True,timeout=timeout)
    call('prepare',timeout=1200)
    for name in NAMES:
        for level in range(3):call('replay',name,level)
    for name in NAMES:
        for repeat in range(2):
            for arm in ARMS:call('fit',name,0,arm,repeat)
    reference=[json.loads(stem(name,0,arm,repeat).with_suffix('.json').read_text())
               for name in NAMES for repeat in range(2) for arm in ARMS]
    passed=all(r['passed'] for r in reference)
    dump(OUT/'reference_gate.json',dict(passed=passed,gate=GATE,failed=[r for r in reference if not r['passed']]))
    if passed:
        for level in [1,2]:
            for name in NAMES:
                for arm in ARMS:call('fit',name,level,arm,0)
    summaries=collect()
    for name,h in json.loads((OUT/'source_and_input_fingerprints.json').read_text()).items():assert sha(ROOT/name)==h,name
    checks=pd.read_csv(OUT/'prior_verification.csv')
    for _,g in checks.groupby(['trajectory','level']):
        for col in ['current_camera_sha256','observation_ids_sha256','source_sha256','Q_cross_sha256']:assert g[col].nunique()==1,col
    dump(OUT/'verification.json',dict(completed_fits=len(summaries),reference_gate=passed,
        all_completed_fits_pass=all(s['passed'] for s in summaries),historical_replays_exact=6,
        previous_caches_unchanged=True,source_inputs_unchanged=True,full_prior_cross_covariance=True,
        current_likelihood_noise_and_observations_matched=True,previous_inference_rerun=False,
        total_wall_seconds=perf_counter()-start))


if __name__=='__main__':
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        if os.environ.get(key)!='1':raise RuntimeError('Set '+key+'=1')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['run','prepare','replay','fit','collect'],default='run')
    parser.add_argument('--name',choices=NAMES);parser.add_argument('--level',type=int,choices=[0,1,2])
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--repeat',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    if args.phase=='run':run()
    elif args.phase=='prepare':prepare()
    elif args.phase=='replay':replay(args.name,args.level)
    elif args.phase=='fit':fit(args.name,args.level,args.arm,args.repeat)
    else:collect()
