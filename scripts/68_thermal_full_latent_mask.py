"""Output 64: reference-only mask comparison under frozen full-latent C."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.latent_mask_audit import extend_blocks, additional_layout
from src.unpinning_current_update import propagated_blocks
from src.previous_unpinning_audit import noisy_precision
from src.censored_gp import RBFConfig
from src.matched_projection import MatchedHeatOperator
from src.stochastic_heat_gp import StochasticHeatConfig, finite_step_innovation_covariance

spec = importlib.util.spec_from_file_location('frozen61', ROOT/'scripts/65_thermal_unpinning_current_update.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
H = M.HELP
OUT = ROOT/'outputs/by_experiment/64_full_latent_mask'
OLD = ROOT/'outputs/by_experiment/54_native_non_saturation_mask'
PREVIOUS = 'all_readings_latent'
ARMS = ['C0', 'Cmask', 'Cnumeric']


def folder(name):return OUT/'by_trajectory'/name
def stem(name, arm, repeat):return folder(name)/f'{arm}_repeat_{repeat}'
def case(name):return M.load(M.folder(name)/'case_0.npz')
def prior(name):return M.load(M.folder(name)/f'prior_0_{PREVIOUS}.npz')
def selection(name):return M.load(folder(name)/'selection.npz')


def prepare():
    if OUT.exists():raise FileExistsError(OUT)
    start = perf_counter()
    manifest = json.loads((M.OUT/'source_and_input_fingerprints.json').read_text())
    for path, digest in manifest.items():assert M.sha(ROOT/path)==digest, path
    inherited = json.loads((M.OUT/'frozen_configuration.json').read_text())
    assert inherited['samples_per_chain']==M.SAMPLES and inherited['burn_in']==M.BURN
    assert inherited['gate']==M.GATE
    assert json.loads((M.OUT/'verification.json').read_text())['all_completed_fits_pass']
    H.arviz_diagnostics(np.random.default_rng(7).normal(size=(4,100,2)), np.arange(2), 'import_probe')
    import arviz, scipy
    paths = [Path(__file__), ROOT/'src/latent_mask_audit.py', ROOT/'tests/test_latent_mask_audit.py',
             ROOT/'scripts/65_thermal_unpinning_current_update.py', M.OUT/'frozen_configuration.json',
             M.OUT/'source_and_input_fingerprints.json', M.OUT/'verification.json',
             M.OUT/'prior_verification.csv', ROOT/'outputs/by_experiment/55_mask_reliability_and_ceiling/results.csv']
    paths += [ROOT/p for p in manifest if p.endswith('.py')]
    paths += [M.BASE/'frozen_configuration.csv', M.BASE/'verification.csv',
              ROOT/'outputs/by_experiment/56_within_step_integration/frozen_sections.csv',
              ROOT/'outputs/by_experiment/56_within_step_integration/gradient_provenance.json']
    for name in M.NAMES:
        paths += [OLD/'by_trajectory'/name/'selection.npz', OLD/'by_trajectory'/name/'selected_locations.csv',
                  M.folder(name)/f'prior_0_{PREVIOUS}.npz', M.PREV/'by_trajectory'/name/'level_0_attempt_0.npz',
                  M.PREV/'by_trajectory'/name/'level_0_summaries.npz', M.BASE/'by_trajectory'/name/'level_0.npz']
        for level in range(3):
            paths += [M.folder(name)/f'case_{level}.npz', M.stem(name,level,PREVIOUS,0).with_suffix('.npz')]
        paths.append(M.stem(name,0,PREVIOUS,1).with_suffix('.npz'))
    OUT.mkdir(parents=True)
    M.dump(OUT/'source_and_input_fingerprints.json', {str(p.relative_to(ROOT)):M.sha(p) for p in sorted(set(paths))})
    M.dump(OUT/'frozen_configuration.json', dict(experiment=64,driver=str(Path(__file__).relative_to(ROOT)),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inherited_output61=inherited, previous_representation=PREVIOUS, source_on=True,
        arms=ARMS, trajectories=M.NAMES, ceiling='reference only', additional_nodes=256,
        selection='Unchanged output54 cached geometry-selected IDs; no reselection',
        chains=4, samples_per_chain=M.SAMPLES,burn_in=M.BURN,thin=1,evaluation_per_chain=M.EVAL,
        paired_repeats={'C0':2,'Cmask':2,'Cnumeric':1},repeat_offset=M.REPEAT_OFFSET,
        gate=M.GATE,retries=0,current_noise_sd_K=.5,camera_noise_sd_K=.25,
        previous_inference_rerun=False,coefficient_fit=False,held_out_expansion=False,high_ceiling_inference=False,
        high_ceiling_diagnostic='Cached full-latent source-on marginal flag compatibility only',
        covariance='Q_dt + A [W Cov(Z_S) W.T + C] A.T; C=vI-v^2P, W=C[:,S]/v; pinned=False',
        budget='Prospective 900s per fit; output55 historical 300s decision unchanged',
        marginal_predictions_only=True,numpy=np.__version__,scipy=scipy.__version__,arviz=arviz.__version__))
    compatibility=[]; sizes=[]
    for name in M.NAMES:
        folder(name).mkdir(parents=True)
        for filename in ['selection.npz','selected_locations.csv']:
            shutil.copyfile(OLD/'by_trajectory'/name/filename,folder(name)/filename)
        d=case(name); sel=selection(name); u=sel['added_ids']; pool=sel['pool']
        np.testing.assert_array_equal(sel['baseline_ids'],d['observation_ids'])
        np.testing.assert_array_equal(sel['points'],d['points'])
        np.testing.assert_array_equal(sel['area_weights'],d['area_weights'])
        assert len(u)==256 and pool[u].all() and not pool[d['observation_ids']].any()
        assert len(d['points'])==len(u)+len(d['observation_ids'])+int((pool & ~np.isin(np.arange(len(pool)),u)).sum())
        assert (d['camera'][pool]<d['ceiling']).all()
        for level in range(3):
            other=M.load(M.folder(name)/f'case_{level}.npz')
            np.testing.assert_array_equal(other['points'],d['points'])
            assert (other['camera'][pool]<other['ceiling']).all()
            with np.load(M.stem(name,level,PREVIOUS,0).with_suffix('.npz')) as f:
                probs=ndtr((float(other['ceiling'])-f['draws'][:,u])/.5).mean(axis=0)
            compatibility += [dict(trajectory=name,level=level,ceiling_K=float(other['ceiling']),node=int(i),probability=float(p)) for i,p in zip(u,probs)]
        for arm in ARMS:
            n=len(pool); m=len(d['observation_ids'])+(256 if arm!='C0' else 0)
            k=int((d['kinds']==1).sum())+(256 if arm=='Cmask' else 0)
            sizes.append(dict(trajectory=name,arm=arm,prediction_nodes=n,observations=m,constraints=k,
                numeric=m-k,upper=256 if arm=='Cmask' else 0,lower=int((d['kinds']==1).sum()),
                covariance_blocks_MiB=8*(n*m+m*m+n)/2**20,
                retained_field_chains_MiB=8*4*M.SAMPLES*n/2**20,
                selected_area_fraction=float(d['area_weights'][u].sum()/d['area_weights'].sum()),
                pool_area_fraction=float(d['area_weights'][pool].sum()/d['area_weights'].sum())))
    pd.DataFrame(compatibility).to_csv(OUT/'cached_flag_compatibility.csv',index=False)
    pd.DataFrame(sizes).to_csv(OUT/'preflight.csv',index=False)
    print('CHECKED fixed selection and cached flag compatibility; constructing full-latent added covariance',flush=True)
    o=inherited['inherited_settings']; d=case(M.NAMES[0]); xy=d['points']; n=len(xy)
    meta=pd.read_csv(M.BASE/'verification.csv'); assert meta.lengthscale_m.nunique()==1
    with np.load(M.BASE/'by_trajectory'/M.NAMES[0]/'level_0.npz') as f:ambient=float(f['ambient'])
    cfg=RBFConfig(mean_temp=ambient,signal_sd=o['signal_sd'],lengthscale=float(meta.lengthscale_m.iloc[0]),noise_sd=o['previous_noise_sd'])
    precision,v,inverse_error=noisy_precision(xy,cfg)
    times={r['trajectory']:r['dt_s'] for r in json.loads((ROOT/'outputs/by_experiment/56_within_step_integration/gradient_provenance.json').read_text())}
    assert len(set(times.values()))==1
    physics=dict(dt=times[M.NAMES[0]],diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'])
    op=MatchedHeatOperator(SimpleNamespace(points=xy,triangles=d['triangles']),
        np.linspace(xy[:,0].min(),xy[:,0].max(),int(o['nx'])),np.linspace(xy[:,1].min(),xy[:,1].max(),int(o['ny'])))
    chunks=[]
    for first in range(0,n,32):
        ids=np.arange(first,min(first+32,n)); eye=np.zeros((len(ids),n)); eye[np.arange(len(ids)),ids]=1
        chunks.append(csr_matrix(op.propagate(eye,'native',**physics).T))
    a=hstack(chunks,format='csr');del chunks,eye
    ap=a@precision
    hcfg=StochasticHeatConfig(signal_sd=o['signal_sd'],forcing_lengthscale=cfg.lengthscale*o['forcing_length_multiplier'],
        diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],quadrature_order=int(o['quadrature_order']))
    qdiag=float(finite_step_innovation_covariance(xy[:1],xy[:1],hcfg,physics['dt'])[0,0])
    checks=[]
    for name in M.NAMES:
        d=case(name); b=prior(name); u=selection(name)['added_ids']
        np.testing.assert_array_equal(xy,d['points'])
        with np.load(M.PREV/'by_trajectory'/name/'level_0_attempt_0.npz') as f:
            z=f['noisy_censored_chains']; s=f['censored_ids']
        with np.load(M.PREV/'by_trajectory'/name/'level_0_summaries.npz') as f:
            np.testing.assert_array_equal(b['prediction_mean'],f['forecast_'+PREVIOUS])
            np.testing.assert_array_equal(d['gradient_contribution'],f['gradient_contribution'])
        _, po, diag=propagated_blocks(a,precision,v,s,z,u,pinned=False,operator_precision=ap)
        po+=finite_step_innovation_covariance(xy,xy[u],hcfg,physics['dt']);diag+=qdiag
        ids, extended, adjustment=extend_blocks(b,d['observation_ids'],u,po,diag)
        np.savez_compressed(folder(name)/'extended_prior.npz',**extended,observation_ids=ids)
        checks.append(dict(trajectory=name,added_nodes=len(u),overlap_reconciliation_max_K2=adjustment,
            recomputed_diagonal_max_error_K2=float(abs(diag-b['prediction_variance']).max()),
            mean_sha256=H.digest(b['prediction_mean']),diagonal_sha256=H.digest(b['prediction_variance']),
            baseline_cross_sha256=H.digest(b['pred_observed_covariance']),baseline_observed_sha256=H.digest(b['observed_covariance']),
            extended_cross_sha256=H.digest(extended['pred_observed_covariance']),
            previous_noisy_draws_sha256=H.digest(z),source_increment_sha256=H.digest(d['gradient_contribution']),
            added_ids_sha256=H.digest(u),base_ids_sha256=H.digest(d['observation_ids']),camera_sha256=H.digest(d['camera']),
            baseline_blocks_exact=True,full_conditional_residual_covariance=True))
        print('PREPARED',name,'max roundoff reconciliation',adjustment,flush=True)
        del po,diag,extended,z,b
    pd.DataFrame(checks).to_csv(OUT/'prior_verification.csv',index=False)
    M.dump(OUT/'preparation.json',dict(seconds=perf_counter()-start,peak_RSS_MiB=H.rss(),inverse_error=inverse_error))
    assert H.rss()<=M.GATE['peak_RSS_MiB_max']


def fit(name,arm,repeat):
    path=stem(name,arm,repeat)
    if path.with_suffix('.npz').exists() or path.with_suffix('.json').exists():raise FileExistsError(path)
    d=case(name); sel=selection(name); u=sel['added_ids']
    ids,kinds,values=additional_layout(d['observation_ids'],d['kinds'],d['values'],u,arm,
                                     readings=d['camera'][u] if arm=='Cnumeric' else None)
    b=prior(name) if arm=='C0' else M.load(folder(name)/'extended_prior.npz')
    if arm!='C0':np.testing.assert_array_equal(b.pop('observation_ids'),ids)
    monitor=np.union1d(np.union1d(d['observation_ids'],u), M.spread_nodes(d['points'],np.ones(len(d['points']),bool),64))
    start=perf_counter();latent=[];noisy=[]
    print('START',name,arm,repeat,'constraints',int((kinds!=0).sum()),flush=True)
    for chain in range(4):
        pred,z=M.sample_mixed_noisy_blocks(ids=ids,kinds=kinds,numerical_values=values,thresholds=float(d['ceiling']),
            **b,noise_sd=.5,n_samples=M.SAMPLES,burn_in=M.BURN,thin=1,seed=int(d['seed'])+repeat*M.REPEAT_OFFSET+100000*chain)
        latent.append(pred[4]);noisy.append(z)
        print('CHAIN',name,arm,repeat,chain,flush=True)
    latent=np.asarray(latent);noisy=np.asarray(noisy)
    evaluation=latent[:,np.linspace(0,M.SAMPLES-1,M.EVAL,dtype=int),:]
    pred=M.prediction_from_draws(evaluation.reshape(-1,latent.shape[-1]))
    replay=False
    if arm=='C0':
        with np.load(M.stem(name,0,PREVIOUS,repeat).with_suffix('.npz')) as old:
            for k,value in zip(M.SUFFIXES,pred):np.testing.assert_array_equal(value,old[k])
            np.testing.assert_array_equal(noisy,old['noisy_censored_chains'])
            np.testing.assert_array_equal(latent[:,:,old['monitor_ids']],old['monitored_chains'])
        replay=True
    diagnostics=pd.concat([H.arviz_diagnostics(noisy,ids[kinds!=0],'noisy_constraint'),
        H.arviz_diagnostics(latent[:,:,monitor],monitor,'latent_temperature')],ignore_index=True)
    classic=M.chain_diagnostics(latent,selected_mask=d['camera']==d['ceiling']); nc=M.chain_diagnostics(noisy)
    common=dict(trajectory=name,level=0,ceiling_K=float(d['ceiling']),arm=arm,repeat=repeat)
    rows=H.metrics(d,pred,sel['pool'],u);peak=int(np.argmax(d['truth']))
    for row in rows:
        row['true_peak_signed_error_K']=float(pred[0][peak]-d['truth'][peak])
        row['true_peak_covered']=bool(pred[2][peak]<=d['truth'][peak]<=pred[3][peak])
    pd.DataFrame([dict(**common,**r) for r in rows]).to_csv(path.with_suffix('.metrics.csv'),index=False)
    diagnostics.to_csv(path.with_suffix('.diagnostics.csv'),index=False)
    summary=dict(**common,seconds=perf_counter()-start,peak_RSS_MiB=H.rss(),rank_rhat_max=float(diagnostics.rank_rhat.max()),
        ess_bulk_min=float(diagnostics.ess_bulk.min()),ess_tail_min=float(diagnostics.ess_tail.min()),
        classical_rhat_max=max(classic['rhat_max'],classic['selected_rhat_max'],nc['rhat_max']),
        finite=bool(np.isfinite(diagnostics[['rank_rhat','ess_bulk','ess_tail']]).all().all()),
        baseline_replay_exact=replay,current_seed=int(d['seed'])+repeat*M.REPEAT_OFFSET,
        numeric=int((kinds==0).sum()),lower=int((kinds==1).sum()),upper=int((kinds==-1).sum()),
        prediction_mean_sha256=H.digest(b['prediction_mean']),prediction_variance_sha256=H.digest(b['prediction_variance']))
    summary['passed']=bool(summary['finite'] and summary['rank_rhat_max']<=M.GATE['rank_rhat_max']
        and summary['classical_rhat_max']<=M.GATE['classical_rhat_max'] and summary['ess_bulk_min']>=400
        and summary['ess_tail_min']>=400 and summary['seconds']<=900 and summary['peak_RSS_MiB']<=4096)
    np.savez_compressed(path.with_suffix('.npz'),**dict(zip(M.SUFFIXES,pred)),evaluation_chains=evaluation,
        monitored_chains=latent[:,:,monitor],monitor_ids=monitor,noisy_constraint_chains=noisy,
        constraint_ids=ids[kinds!=0],observation_ids=ids,likelihood_kinds=kinds,numerical_values=values)
    M.dump(path.with_suffix('.json'),summary)
    print('FIT',summary,flush=True)


def collect():
    metrics=[];summaries=[]
    for name in M.NAMES:
        for path in folder(name).glob('*_repeat_*.json'):
            summaries.append(json.loads(path.read_text()));metrics.append(pd.read_csv(path.with_suffix('.metrics.csv')))
    data=pd.concat(metrics,ignore_index=True);data.to_csv(OUT/'results.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'convergence_runtime.csv',index=False)
    fields=['rmse_K','crps_K','bias_K','coverage95','width95_K','posterior_sd_K','true_peak_signed_error_K']
    paired=[];repeats=[]
    for key,g in data.groupby(['trajectory','repeat','region']):
        g=g.set_index('arm')
        for arm in ['Cmask','Cnumeric']:
            if arm in g.index:
                paired.append(dict(zip(['trajectory','repeat','region'],key),arm=arm,
                    **{k:float(g.loc[arm,k]-g.loc['C0',k]) for k in fields}))
    changes=pd.DataFrame(paired);changes.to_csv(OUT/'paired_changes.csv',index=False)
    for key,g in changes[changes.arm=='Cmask'].groupby(['trajectory','region']):
        if set(g.repeat)=={0,1}:
            g=g.set_index('repeat')
            for metric in fields:
                a,b=float(g.loc[0,metric]),float(g.loc[1,metric])
                repeats.append(dict(trajectory=key[0],region=key[1],metric=metric,paired_seed0=a,paired_seed1=b,
                    same_sign=a*b>0,repeat_spread=abs(a-b),mean_paired_change=(a+b)/2))
    pd.DataFrame(repeats).to_csv(OUT/'repeat_seed_comparisons.csv',index=False)
    old=pd.read_csv(ROOT/'outputs/by_experiment/55_mask_reliability_and_ceiling/results.csv')
    comparison=[]
    for label,frame in [('old_hybrid_output55',old),('full_latent_output64',data)]:
        for name in M.NAMES:
            f=frame[(frame.trajectory==name)&(frame.repeat==0)&(frame.region=='unused_pool')].set_index('arm')
            comparison.append(dict(baseline=label,trajectory=name,baseline_rmse_K=f.loc['C0','rmse_K'],mask_rmse_K=f.loc['Cmask','rmse_K'],
                rmse_reduction_fraction=1-f.loc['Cmask','rmse_K']/f.loc['C0','rmse_K'],
                baseline_crps_K=f.loc['C0','crps_K'],mask_crps_K=f.loc['Cmask','crps_K']))
    pd.DataFrame(comparison).to_csv(OUT/'historical_effect_comparison.csv',index=False)
    for name in M.NAMES:plot(name)
    return summaries


def plot(name):
    d=case(name);sel=selection(name);u=sel['added_ids'];pool=sel['pool'];b=prior(name)
    predictions={a:M.load(stem(name,a,0).with_suffix('.npz')) for a in ARMS}
    xy=d['points'];tri=mtri.Triangulation(xy[:,0]*1000,xy[:,1]*1000,d['triangles'])
    tri.set_mask(~pool[d['triangles']].all(axis=1))
    limit=max(abs(p['mean'][pool]-d['truth'][pool]).max() for p in predictions.values())
    sdlimit=max(p['sd'][pool].max() for p in predictions.values())
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for col,(arm,p) in enumerate(predictions.items()):
        for row,values in enumerate([p['mean']-d['truth'],p['sd']]):
            im=axes[row,col].tripcolor(tri,values,cmap='RdBu_r' if row==0 else 'viridis',
                vmin=-limit if row==0 else 0,vmax=limit if row==0 else sdlimit,shading='gouraud')
            axes[row,col].scatter(xy[u,0]*1000,xy[u,1]*1000,s=3,c='k',alpha=.55)
            axes[row,col].set(title=arm+(' signed error' if row==0 else ' posterior SD'),xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
            if col==2:fig.colorbar(im,ax=list(axes[row]),label='K',shrink=.75)
    fig.suptitle(f'{name}; reference ceiling {float(d["ceiling"]):.3f} K; full-latent source-on C\nPreviously unused region; black dots: same 256 added locations; shared row scales')
    fig.savefig(OUT/f'background_{name}.png',dpi=150);plt.close(fig)
    sections=pd.read_csv(ROOT/'outputs/by_experiment/56_within_step_integration/frozen_sections.csv')
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True,constrained_layout=True)
    for col,(arm,p) in enumerate(predictions.items()):
        for row,(section,g) in enumerate(sections[sections.trajectory==name].groupby('section',sort=False)):
            g=g.sort_values('along_mm');ids=g.node.to_numpy(int);x=g.along_mm.to_numpy();ax=axes[row,col]
            ax.plot(x,d['truth'][ids],'k',label='Native truth')
            ax.plot(x,b['prediction_mean'][ids],'--',color='#D55E00',label='Frozen source-on forecast')
            ax.plot(x,p['mean'][ids],color='#0072B2',label=arm)
            ax.fill_between(x,p['lower'][ids],p['upper'][ids],alpha=.2,color='#0072B2',label='Pointwise 95% interval')
            chosen=np.isin(ids,u);ax.scatter(x[chosen],p['mean'][ids[chosen]],c='k',marker='x',label='Added U on section')
            ax.axhline(float(d['ceiling']),color='#777777',ls=':')
            ax.set(title=arm+(' | Along' if row==0 else ' | Across'),xlabel='Distance from fixed true peak (mm)',ylabel='Temperature (K)')
            ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'{name}; full-latent source-on C, reference {float(d["ceiling"]):.3f} K\nSame current baseline readings, same full prior; only information at U differs')
    fig.savefig(OUT/f'cross_sections_{name}.png',dpi=150);plt.close(fig)


def run():
    start=perf_counter()
    def call(phase,name=None,arm=None,repeat=None):
        args=[sys.executable,str(Path(__file__)),'--phase',phase]
        for k,v in [('name',name),('arm',arm),('repeat',repeat)]:
            if v is not None:args+=['--'+k,str(v)]
        subprocess.run(args,cwd=ROOT,check=True,timeout=900)
    call('prepare')
    # Replay all four paired controls before any additional-information arm.
    for name in M.NAMES:
        for repeat in range(2):call('fit',name,'C0',repeat)
    for name in M.NAMES:
        for repeat in range(2):call('fit',name,'Cmask',repeat)
        call('fit',name,'Cnumeric',0)
    summaries=collect()
    for path,h in json.loads((OUT/'source_and_input_fingerprints.json').read_text()).items():assert M.sha(ROOT/path)==h,path
    passed=all(s['passed'] for s in summaries)
    M.dump(OUT/'reference_gate.json',dict(passed=passed,thresholds=M.GATE,failed=[s for s in summaries if not s['passed']],
        automatic_expansion=False,note='Any future validation requires a separately frozen trajectory/ceiling plan; no budget sweep here'))
    M.dump(OUT/'verification.json',dict(completed_fits=len(summaries),all_fits_pass=passed,
        baseline_replays_exact=sum(s['baseline_replay_exact'] for s in summaries),same256=True,
        full_latent_covariance=True,previous_inference_rerun=False,source_on=True,
        historical_inputs_unchanged=True,fit_seconds=sum(s['seconds'] for s in summaries),
        total_wall_seconds=perf_counter()-start,held_out_inference=False,high_ceiling_inference=False))


if __name__=='__main__':
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        if os.environ.get(key)!='1':raise RuntimeError('Set '+key+'=1')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['run','prepare','fit','collect'],default='run')
    parser.add_argument('--name',choices=M.NAMES);parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--repeat',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    if args.phase=='run':run()
    elif args.phase=='prepare':prepare()
    elif args.phase=='fit':fit(args.name,args.arm,args.repeat)
    else:collect()
