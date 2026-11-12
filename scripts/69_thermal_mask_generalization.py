"""Output 65: prospectively selected held-out mask validation; no fitting."""
from pathlib import Path
from time import perf_counter
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, hstack
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mask_generalization import select_heldout, retained_layout, align_extended_prior, verify_prior_layout
from src.thermal_trajectory import ThermalTrajectory, trajectory_catalog, SurfaceGridProjector
from src.thermal_posterior_physics import DEVELOPMENT_TRAJECTORIES, estimate_source_lengthscale
from src.native_surface import extract_surface, hottest_area_weights
from src.native_mask_pilot import spread_nodes, sample_mixed_noisy_blocks
from src.previous_unpinning_audit import (noisy_precision, conditional_camera, latent_from_camera,
    conditional_covariance_block, mixture_moments)
from src.previous_correlation_audit import product, psd_factor
from src.truncated_gaussian import sample_lower_truncated_gaussian_hmc
from src.unpinning_current_update import propagated_blocks
from src.latent_mask_audit import additional_layout
from src.matched_projection import MatchedHeatOperator
from src.censored_gp import RBFConfig
from src.stochastic_heat_gp import StochasticHeatConfig, finite_step_innovation_covariance

spec=importlib.util.spec_from_file_location('frozen64',ROOT/'scripts/68_thermal_full_latent_mask.py')
F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
M,H=F.M,F.H
OUT=ROOT/'outputs/by_experiment/65_mask_generalization'
DATA=ROOT.parent/'heat_eq_laser_trajectories'
ARMS=['C0','Cmask','Cnumeric']


def folder(name,level=0):return OUT/'by_trajectory'/name/f'level_{level}'
def config():return json.loads((OUT/'frozen_configuration.json').read_text())
def load(path):return M.load(path)
def dump(path,data):return M.dump(path,data)
def passed(s):
    return bool(s['finite'] and s['rank_rhat_max']<=1.01 and s['classical_rhat_max']<=1.05
        and s['ess_bulk_min']>=400 and s['ess_tail_min']>=400 and s['seconds']<=900
        and s['peak_RSS_MiB']<=4096)


def freeze():
    if OUT.exists():raise FileExistsError(OUT)
    catalog=trajectory_catalog(DATA)
    names=select_heldout(catalog,DEVELOPMENT_TRAJECTORIES)
    assert names==['DiagonalScanPath_1','DiagonalScanPath_7','HorizontalScanPath_1',
                  'HorizontalScanPath_12','SpiralScanPath_1','SpiralScanPath_12']
    inherited=json.loads((F.OUT/'frozen_configuration.json').read_text())
    assert json.loads((F.OUT/'verification.json').read_text())['all_fits_pass']
    OUT.mkdir(parents=True)
    c=dict(experiment=65,driver=str(Path(__file__).relative_to(ROOT)),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        selection='Smallest and largest historical held-out run index per family; no outcomes used',
        trajectories=names,development_excluded=list(DEVELOPMENT_TRAJECTORIES),
        historical_heldout_not_pristine=True,mild_control=dict(trajectory=names[0],ceiling_K=320.),
        inherited_output64=inherited,source_on=True,previous_representation='all_readings_latent',
        additional_nodes=256,selection_rule='Unchanged deterministic physical-coordinate farthest-point rule',
        current_chains=4,current_samples=800,current_burn=300,evaluation_per_chain=300,
        repeats={'C0':2,'Cmask':2,'Cnumeric':1},mild_repeats={'C0':2,'Cmask':2,'Cnumeric':1},
        previous_attempts=[[800,300],[1600,600]],previous_case_budget_seconds=900,
        current_retries=0,gate=M.GATE,paired_repeat_offset=M.REPEAT_OFFSET,
        stop_rule='Stop a case after a failed previous/current gate; do not replace selected cases. Mild control only after all six reference cases pass.',
        source_or_kernel_fitting=False,coefficient_fitting=False,maximum_current_fits=35,
        runtime_estimate_minutes=[50,100],preparation_budget_seconds=900,
        preparation='Derive inherited ambient quantile and source-width rule without running unused diffusivity fits',
        input_limitation='Current simulator-derived HeatFluxZ and inherited simulation-derived preparation remain inputs; not camera-only deployment')
    dump(OUT/'frozen_configuration.json',c)
    records=[dict(trajectory=r.name,family=r.family,index=i,run_index=r.run_index,
                  historical_role='development' if r.name in DEVELOPMENT_TRAJECTORIES else 'heldout',
                  selected=r.name in names) for i,r in enumerate(catalog)]
    pd.DataFrame(records).to_csv(OUT/'prospective_selection.csv',index=False)
    paths=[Path(__file__),ROOT/'src/mask_generalization.py',ROOT/'tests/test_mask_generalization.py',
           F.OUT/'frozen_configuration.json',F.OUT/'verification.json',
           ROOT/'outputs/by_experiment/43_matched_ceiling_model_gap/results.csv']
    paths+=list((ROOT/'src').glob('*.py'))
    paths+=[ROOT/'scripts/68_thermal_full_latent_mask.py',ROOT/'scripts/65_thermal_unpinning_current_update.py',
            ROOT/'scripts/58_thermal_native_non_saturation_mask.py']
    dump(OUT/'source_fingerprints.json',{str(p.relative_to(ROOT)):M.sha(p) for p in paths})
    print('FROZEN',names,'mild',c['mild_control'],flush=True)


def prepare(name,level):
    """Construct native cameras and geometry before inspecting predictions."""
    start=perf_counter();c=config();o=c['inherited_output64']['inherited_output61']['inherited_settings']
    catalog=trajectory_catalog(DATA);index=next(i for i,r in enumerate(catalog) if r.name==name)
    t=ThermalTrajectory(catalog[index].xdmf_path);mesh=extract_surface(t);xy=mesh.points;n=len(xy)
    if 10*n*n*8/2**20>4096:raise RuntimeError('Dense preparation exceeds preflight memory budget')
    path=folder(name,level)
    if path.exists():raise FileExistsError(path)
    path.mkdir(parents=True)
    projector=SurfaceGridProjector(t.surface_coordinates(),nx=int(o['nx']),ny=int(o['ny']))
    # Same two derived quantities as prepare_trajectory, without its unused coefficient fit.
    _,history=t.read_history(time_indices=list(range(t.n_times)),surface_only=True)
    ambient=float(np.quantile(projector.project(history),.02));del history
    _,gradient_history=t.read_history('HeatFluxZ',time_indices=list(range(t.n_times)),surface_only=True)
    width=estimate_source_lengthscale(projector.project(gradient_history),projector.xs,projector.ys)
    del gradient_history
    length=max(width,float(np.mean(np.diff(projector.xs))),float(np.mean(np.diff(projector.ys))))*o['length_multiplier']
    truth=t.read_temperature(-1,surface_only=True);previous_truth=t.read_temperature(-2,surface_only=True)
    gradient=t.read_heat_flux_z(-1,surface_only=True)
    rng=np.random.default_rng(int(o['seed'])+10000*index)
    previous_signal=previous_truth+rng.normal(0,o['measurement_noise_sd'],n)
    current_signal=truth+rng.normal(0,o['measurement_noise_sd'],n)
    reference=pd.read_csv(ROOT/'outputs/by_experiment/43_matched_ceiling_model_gap/results.csv',
                          usecols=['trajectory','level_index','ceiling_K'])
    values=reference[(reference.trajectory==name)&(reference.level_index==0)].ceiling_K.unique()
    assert len(values)==1
    ref=float(values[0]);ceiling=ref if level==0 else float(c['mild_control']['ceiling_K'])
    assert ceiling>=ref
    gx,gy=np.meshgrid(projector.xs[::5],projector.ys[::5])
    sensors=cKDTree(xy).query(np.column_stack([gx.ravel(),gy.ravel()]))[1]
    candidates=np.zeros(n,bool);candidates[np.unique(sensors)]=True
    ref_sat=current_signal>=ref;pool=~(candidates|ref_sat)
    added=spread_nodes(xy,pool,256)
    assert len(added)==256 and (current_signal[pool]<ref).all()
    camera=np.minimum(current_signal,ceiling);previous_camera=np.minimum(previous_signal,ceiling)
    ids,kinds,readings=retained_layout(candidates,ref_sat,camera,ceiling)
    dt=float(t.times[-1]-t.times[-2]);source=o['source_coupling']*np.where(gradient>=o['source_flux_threshold'],gradient,0)*dt
    arrays=dict(points=xy,triangles=mesh.triangles,area_weights=mesh.weights,truth=truth,previous_truth=previous_truth,
        previous_signal=previous_signal,current_signal=current_signal,previous_camera=previous_camera,camera=camera,
        ceiling=ceiling,reference_ceiling=ref,observation_ids=ids,kinds=kinds,values=readings,
        numeric_candidates=candidates,pool=pool,added_ids=added,gradient_contribution=source,
        hot_area_weights=hottest_area_weights(truth,mesh.weights),ambient=ambient,dt=dt,lengthscale=length,
        previous_seed=int(o['seed'])+50000*index,seed=int(o['seed'])+100000*index)
    if level:
        original=load(folder(name)/'case.npz')
        for key in ['points','pool','added_ids','observation_ids','current_signal','previous_signal','hot_area_weights']:
            np.testing.assert_array_equal(arrays[key],original[key])
    np.savez_compressed(path/'case.npz',**arrays)
    pd.DataFrame(dict(node=added,x_m=xy[added,0],y_m=xy[added,1],area_m2=mesh.weights[added])).to_csv(path/'selected_locations.csv',index=False)
    fingerprints={str(t.xdmf_path):M.sha(t.xdmf_path)}
    for p in sorted(DATA.glob('*'+name+'.h5')):fingerprints[str(p)]=M.sha(p)
    dump(path/'input_fingerprints.json',fingerprints)
    dump(path/'preflight.json',dict(native_nodes=n,baseline_observations=len(ids),lower_constraints=int(kinds.sum()),
        mask_constraints=int(kinds.sum())+256,upper_constraints=256,previous_censored=int((previous_camera==ceiling).sum()),
        dense_precision_estimate_MiB=10*n*n*8/2**20,selected_area_fraction=float(mesh.weights[added].sum()/mesh.weights.sum()),
        unused_area_fraction=float(mesh.weights[pool].sum()/mesh.weights.sum()),seconds=perf_counter()-start,
        camera_seed=int(o['seed'])+10000*index,ambient_K=ambient,lengthscale_m=length,dt_s=dt))
    print('PREPARED cameras/IDs',name,level,'nodes',n,'constraints',int(kinds.sum()),flush=True)


def previous(name,level):
    start=perf_counter();c=config();o=c['inherited_output64']['inherited_output61']['inherited_settings']
    path=folder(name,level);d=load(path/'case.npz');xy=d['points'];n=len(xy)
    if (path/'previous.npz').exists():raise FileExistsError(path/'previous.npz')
    H.arviz_diagnostics(np.random.default_rng(7).normal(size=(4,100,2)),np.arange(2),'probe')
    cfg=RBFConfig(mean_temp=float(d['ambient']),signal_sd=o['signal_sd'],lengthscale=float(d['lengthscale']),noise_sd=o['previous_noise_sd'])
    precision,v,inverse_error=noisy_precision(xy,cfg)
    sat=d['previous_camera']==d['ceiling']
    ids,zm,zc,filled=conditional_camera(precision,d['previous_camera'],sat,float(d['ambient']))
    if not len(ids):raise RuntimeError('No censored predecessor nodes: stop, do not substitute a new path')
    base,w,cd=latent_from_camera(precision,v,filled,float(d['ambient']),ids)
    monitor=np.union1d(spread_nodes(xy,~sat,64),spread_nodes(xy,sat,min(64,len(ids))))
    cm=conditional_covariance_block(precision,v,monitor,monitor);factor,_=psd_factor(cm)
    timings=[]
    for attempt,(samples,burn) in enumerate(c['previous_attempts']):
        tick=perf_counter();chains=[];mon=[]
        for chain in range(4):
            z=sample_lower_truncated_gaussian_hmc(zm,zc,float(d['ceiling']),n_samples=samples,burn_in=burn,thin=1,
                seed=int(d['previous_seed'])+10000*chain+1000003,relative_jitter=cfg.relative_jitter)
            chains.append(z)
            conditional=base[monitor]+product(z-zm,w[monitor].T)
            noise=np.random.default_rng(int(d['previous_seed'])+7000001+10000*chain).normal(size=conditional.shape)
            mon.append(conditional+product(noise,factor.T))
            print('PREVIOUS chain',name,level,attempt,chain,flush=True)
        z=np.asarray(chains);mon=np.asarray(mon)
        diag=pd.concat([H.arviz_diagnostics(z,ids,'previous_noisy'),H.arviz_diagnostics(mon,monitor,'previous_latent')])
        classic=max(M.chain_diagnostics(z)['rhat_max'],M.chain_diagnostics(mon)['rhat_max'])
        timing=dict(attempt=attempt,samples=samples,burn=burn,seconds=perf_counter()-tick,
            rank_rhat_max=float(diag.rank_rhat.max()),ess_bulk_min=float(diag.ess_bulk.min()),
            ess_tail_min=float(diag.ess_tail.min()),classical_rhat_max=classic,peak_RSS_MiB=H.rss(),
            finite=bool(np.isfinite(diag[['rank_rhat','ess_bulk','ess_tail']]).all().all()))
        timing['passed']=passed(timing);timings.append(timing)
        diag.to_csv(path/f'previous_attempt_{attempt}.diagnostics.csv',index=False)
        np.savez_compressed(path/f'previous_attempt_{attempt}.npz',noisy_censored_chains=z,monitor_ids=monitor,monitored_chains=mon,censored_ids=ids)
        dump(path/'previous_sampling.json',timings)
        print('PREVIOUS diagnostics',name,level,timing,flush=True)
        if timing['passed']:break
    if not timings[-1]['passed']:raise RuntimeError('Previous gate failed; no current update')
    mean,variance,_=mixture_moments(base,w,z,zm,cd)
    np.savez_compressed(path/'previous.npz',mean=mean,variance=variance,noisy_censored_chains=z,censored_ids=ids)
    del w,base,chains,mon,factor,cm,variance
    from types import SimpleNamespace
    op=MatchedHeatOperator(SimpleNamespace(points=xy,triangles=d['triangles']),
        np.linspace(xy[:,0].min(),xy[:,0].max(),int(o['nx'])),np.linspace(xy[:,1].min(),xy[:,1].max(),int(o['ny'])))
    physics=dict(dt=float(d['dt']),diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'])
    chunks=[]
    for first in range(0,n,32):
        idx=np.arange(first,min(first+32,n));eye=np.zeros((len(idx),n));eye[np.arange(len(idx)),idx]=1
        chunks.append(csr_matrix(op.propagate(eye,'native',**physics).T))
    a=hstack(chunks,format='csr');del chunks,eye
    ap=a@precision
    forecast=float(d['ambient'])+op.propagate(mean-float(d['ambient']),'native',floor=True,**physics)+d['gradient_contribution']
    allids=np.r_[d['observation_ids'],d['added_ids']]
    oo,cross,pv=propagated_blocks(a,precision,v,ids,z,allids,pinned=False,operator_precision=ap)
    hcfg=StochasticHeatConfig(signal_sd=o['signal_sd'],forcing_lengthscale=float(d['lengthscale'])*o['forcing_length_multiplier'],
        diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],quadrature_order=int(o['quadrature_order']))
    cross+=finite_step_innovation_covariance(xy,xy[allids],hcfg,physics['dt'])
    oo=cross[allids].copy();pv+=float(finite_step_innovation_covariance(xy[:1],xy[:1],hcfg,physics['dt'])[0,0])
    np.testing.assert_allclose(np.diag(oo),pv[allids],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(oo,oo.T,rtol=1e-10,atol=1e-10)
    b=dict(prediction_mean=forecast,observation_mean=forecast[allids],observed_covariance=oo,
           pred_observed_covariance=cross,prediction_variance=pv)
    np.savez_compressed(path/'extended_prior.npz',**b)
    m=len(d['observation_ids'])
    b.update(observation_mean=forecast[d['observation_ids']],observed_covariance=oo[:m,:m],pred_observed_covariance=cross[:,:m])
    np.savez_compressed(path/'prior.npz',**b)
    dump(path/'prior_verification.json',dict(seconds=perf_counter()-start,peak_RSS_MiB=H.rss(),inverse_error=inverse_error,
        previous_draws_sha256=H.digest(z),prior_mean_sha256=H.digest(forecast),prior_diagonal_sha256=H.digest(pv),
        source_sha256=H.digest(d['gradient_contribution']),same_prior_all_arms=True,analytic_conditional_residual=True))
    assert H.rss()<=4096


def replay():
    name='DiagonalScanPath_8';prepare(name,0);previous(name,0)
    d=load(folder(name)/'case.npz');old=F.case(name)
    for key in ['points','camera','observation_ids','kinds','values','gradient_contribution','hot_area_weights']:
        np.testing.assert_array_equal(d[key],old[key])
    np.testing.assert_array_equal(d['added_ids'],F.selection(name)['added_ids'])
    newprev=load(folder(name)/'previous.npz')
    oldprev=load(M.PREV/'by_trajectory'/name/'level_0_attempt_0.npz')
    np.testing.assert_array_equal(newprev['noisy_censored_chains'],oldprev['noisy_censored_chains'])
    b=load(folder(name)/'prior.npz');oldb=F.prior(name);errors={}
    for key in b:
        errors[key]=float(abs(b[key]-oldb[key]).max())
        np.testing.assert_allclose(b[key],oldb[key],atol=1e-10,rtol=0)
    dump(OUT/'development_replay.json',dict(passed=True,camera_and_selection_exact=True,previous_draws_exact=True,
        prior_max_absolute_differences=errors,note='Development pipeline validation only, excluded from generalization results'))
    print('REPLAY verified',errors,flush=True)


def fit(name,level,arm,repeat):
    path=folder(name,level);stem=path/f'{arm}_repeat_{repeat}'
    if stem.with_suffix('.json').exists() or stem.with_suffix('.npz').exists():raise FileExistsError(stem)
    d=load(path/'case.npz');u=d['added_ids'];pool=d['pool']
    ids,kinds,values=additional_layout(d['observation_ids'],d['kinds'],d['values'],u,arm,
                                      readings=d['camera'][u] if arm=='Cnumeric' else None)
    b=load(path/('prior.npz' if arm=='C0' else 'extended_prior_aligned.npz'))
    if arm!='C0':np.testing.assert_array_equal(b.pop('observation_ids'),ids)
    verify_prior_layout(b,ids)
    monitor=np.union1d(np.union1d(d['observation_ids'],u),spread_nodes(d['points'],np.ones(len(pool),bool),64))
    start=perf_counter();latent=[];noisy=[]
    print('START',name,level,arm,repeat,'constraints',int((kinds!=0).sum()),flush=True)
    for chain in range(4):
        pred,z=sample_mixed_noisy_blocks(ids=ids,kinds=kinds,numerical_values=values,thresholds=float(d['ceiling']),
            **b,noise_sd=.5,n_samples=800,burn_in=300,thin=1,seed=int(d['seed'])+repeat*M.REPEAT_OFFSET+100000*chain)
        latent.append(pred[4]);noisy.append(z)
        print('CURRENT chain',name,level,arm,repeat,chain,flush=True)
    latent=np.asarray(latent);noisy=np.asarray(noisy)
    evaluation=latent[:,np.linspace(0,799,300,dtype=int)]
    pred=M.prediction_from_draws(evaluation.reshape(-1,latent.shape[-1]))
    pieces=[H.arviz_diagnostics(latent[:,:,monitor],monitor,'latent_temperature')]
    classic=M.chain_diagnostics(latent,selected_mask=d['camera']==d['ceiling'])
    classical=max(classic['rhat_max'],classic['selected_rhat_max'])
    if noisy.shape[-1]:
        pieces.append(H.arviz_diagnostics(noisy,ids[kinds!=0],'noisy_constraint'))
        classical=max(classical,M.chain_diagnostics(noisy)['rhat_max'])
    diagnostic=pd.concat(pieces,ignore_index=True)
    common=dict(trajectory=name,family=name.split('ScanPath')[0],level=level,ceiling_K=float(d['ceiling']),arm=arm,repeat=repeat)
    rows=H.metrics(d,pred,pool,u);peak=int(np.argmax(d['truth']))
    for row in rows:
        row.update(**common,true_peak_signed_error_K=float(pred[0][peak]-d['truth'][peak]),
                   true_peak_covered=bool(pred[2][peak]<=d['truth'][peak]<=pred[3][peak]))
    pd.DataFrame(rows).to_csv(stem.with_suffix('.metrics.csv'),index=False)
    diagnostic.to_csv(stem.with_suffix('.diagnostics.csv'),index=False)
    summary=dict(**common,seconds=perf_counter()-start,peak_RSS_MiB=H.rss(),rank_rhat_max=float(diagnostic.rank_rhat.max()),
        ess_bulk_min=float(diagnostic.ess_bulk.min()),ess_tail_min=float(diagnostic.ess_tail.min()),classical_rhat_max=classical,
        finite=bool(np.isfinite(diagnostic[['rank_rhat','ess_bulk','ess_tail']]).all().all()),
        numeric=int((kinds==0).sum()),lower=int((kinds==1).sum()),upper=int((kinds==-1).sum()),
        current_seed=int(d['seed'])+repeat*M.REPEAT_OFFSET,prior_mean_sha256=H.digest(b['prediction_mean']),
        prior_diagonal_sha256=H.digest(b['prediction_variance']))
    summary['passed']=passed(summary)
    np.savez_compressed(stem.with_suffix('.npz'),**dict(zip(M.SUFFIXES,pred)),evaluation_chains=evaluation,
        monitored_chains=latent[:,:,monitor],monitor_ids=monitor,noisy_constraint_chains=noisy,
        observation_ids=ids,likelihood_kinds=kinds,numerical_values=values)
    dump(stem.with_suffix('.json'),summary)
    if arm=='C0':
        probs=ndtr((float(d['ceiling'])-pred[4][:,u])/.5).mean(axis=0)
        pd.DataFrame(dict(node=u,probability=probs)).to_csv(path/f'flag_compatibility_repeat_{repeat}.csv',index=False)
    print('FIT',summary,flush=True)


def align(name,level):
    path=folder(name,level);d=load(path/'case.npz');base=load(path/'prior.npz')
    out=path/'extended_prior_aligned.npz'
    if out.exists():raise FileExistsError(out)
    ids,b,adjustment=align_extended_prior(base,d['observation_ids'],d['added_ids'],load(path/'extended_prior.npz'))
    verify_prior_layout(b,ids)
    np.savez_compressed(out,**b,observation_ids=ids)
    errors={}
    if name=='DiagonalScanPath_8':
        old=load(F.folder(name)/'extended_prior.npz')
        np.testing.assert_array_equal(ids,old.pop('observation_ids'))
        for key in b:
            errors[key]=float(abs(b[key]-old[key]).max())
            np.testing.assert_allclose(b[key],old[key],atol=1e-10,rtol=0)
    dump(path/'ordering_verification.json',dict(passed=True,overlap_roundoff_K2=adjustment,
         ids_sha256=H.digest(ids),output64_extended_max_errors=errors))
    print('ALIGNED and verified',name,level,errors,flush=True)


def resume():
    """Recover only ordering-affected arms; preserve all original attempts."""
    start=perf_counter();c=config();failures=[]
    def call(phase,name,level=0,arm=None,repeat=0):
        args=[sys.executable,str(Path(__file__)),'--phase',phase,'--name',name,'--level',str(level),'--repeat',str(repeat)]
        if arm:args+=['--arm',arm]
        subprocess.run(args,cwd=ROOT,check=True,timeout=900)
    call('align','DiagonalScanPath_8')
    source={str(p.relative_to(ROOT)):M.sha(p) for p in [Path(__file__),ROOT/'src/mask_generalization.py',ROOT/'tests/test_mask_generalization.py']}
    dump(OUT/'ordering_fix_fingerprints.json',source)
    call('fit','DiagonalScanPath_8',arm='Cmask')
    replay_new=load(folder('DiagonalScanPath_8')/'Cmask_repeat_0.npz')
    replay_old=load(F.stem('DiagonalScanPath_8','Cmask',0).with_suffix('.npz'))
    for key in M.SUFFIXES+['noisy_constraint_chains']:
        np.testing.assert_array_equal(replay_new[key],replay_old[key])
    dump(OUT/'corrected_mask_replay.json',dict(passed=True,posterior_and_constraint_draws_exact=True,
        development_only=True,excluded_from_generalization=True))
    okay=[]
    for name in c['trajectories']:
        path=folder(name)
        try:
            if name in ['DiagonalScanPath_1','HorizontalScanPath_1']:
                raise RuntimeError('Original predecessor-stage 900s stop preserved; not an ordering-affected fit')
            if not (path/'prior.npz').exists():call('previous',name)
            call('align',name)
            for arm,repeat in [('C0',0),('C0',1),('Cmask',0),('Cmask',1),('Cnumeric',0)]:
                result=path/f'{arm}_repeat_{repeat}.json'
                if not result.exists():call('fit',name,arm=arm,repeat=repeat)
                s=json.loads(result.read_text())
                if not s['passed']:raise RuntimeError('Current reliability gate failed: '+str(s))
            okay.append(True)
        except (subprocess.SubprocessError,RuntimeError) as e:
            okay.append(False);failures.append(dict(trajectory=name,level=0,error=str(e)))
            print('CASE STOP',failures[-1],flush=True)
        finally:
            dump(OUT/'corrected_stopping_decisions.json',failures);collect()
    summaries=collect()
    old=json.loads((OUT/'source_fingerprints.json').read_text())
    for p,h in old.items():
        if p not in source:assert M.sha(ROOT/p)==h,p
    for p,h in source.items():assert M.sha(ROOT/p)==h,p
    for p in (OUT/'by_trajectory').glob('*/level_*/input_fingerprints.json'):
        for f,h in json.loads(p.read_text()).items():assert M.sha(Path(f))==h,f
    dump(OUT/'verification.json',dict(completed_current_fits=len(summaries),reference_cases_passed=sum(okay),
        all_reference_cases_pass=all(okay),mild_control_completed=False,
        mild_control_reason='Predeclared all-six-reference reliability gate not met; no replacement case',
        failures=failures,correction_wall_seconds=perf_counter()-start,
        current_fit_seconds=sum(s['seconds'] for s in summaries),historical_inputs_unchanged=True,
        source_on=True,same256=True,no_fitting=True,ordering_bug_corrected=True,
        invalid_mask_timeouts_excluded=True))


def collect():
    c=config();rows=[];timings=[];pair=[]
    for name in c['trajectories']:
        for level in (0,1):
            path=folder(name,level)
            for p in sorted(path.glob('*_repeat_*.json')):
                timings.append(json.loads(p.read_text()));rows.append(pd.read_csv(p.with_suffix('.metrics.csv')))
    if not rows:return []
    results=pd.concat(rows,ignore_index=True);runtime=pd.DataFrame(timings)
    results.to_csv(OUT/'results.csv',index=False);runtime.to_csv(OUT/'convergence_runtime.csv',index=False)
    fields=['rmse_K','mae_K','bias_K','crps_K','coverage95','width95_K','posterior_sd_K','true_peak_signed_error_K']
    for (name,level,repeat,region),g in results.groupby(['trajectory','level','repeat','region']):
        g=g.set_index('arm')
        if 'C0' not in g.index:continue
        for arm in ['Cmask','Cnumeric']:
            if arm not in g.index:continue
            passed_pair=bool(runtime[(runtime.trajectory==name)&(runtime.level==level)&(runtime.repeat==repeat)&runtime.arm.isin(['C0',arm])].passed.all())
            pair.append(dict(trajectory=name,family=name.split('ScanPath')[0],level=level,repeat=repeat,region=region,arm=arm,reliable=passed_pair,
                **{key:float(g.loc[arm,key]-g.loc['C0',key]) for key in fields},
                rmse_reduction_percent=100*(1-float(g.loc[arm,'rmse_K']/g.loc['C0','rmse_K']))))
    pairs=pd.DataFrame(pair);pairs.to_csv(OUT/'paired_changes.csv',index=False)
    if len(pairs):
        pairs.groupby(['family','level','repeat','region','arm'])[fields+['rmse_reduction_percent']].mean().reset_index().to_csv(OUT/'family_summary.csv',index=False)
        view=pairs[(pairs.level==0)&(pairs.repeat==0)&(pairs.region=='unused_pool')]
        if len(view):
            fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
            for ax,key,label in zip(axes,['rmse_K','crps_K'],['RMSE change (K)','CRPS change (K)']):
                for j,arm in enumerate(['Cmask','Cnumeric']):
                    group=view[view.arm==arm].set_index('trajectory').reindex(c['trajectories'])
                    ax.bar(np.arange(6)+(j-.5)*.35,group[key],width=.35,label=arm)
                ax.axhline(0,color='black',lw=.7);ax.set_xticks(np.arange(6))
                ax.set_xticklabels([n.replace('ScanPath','') for n in c['trajectories']],rotation=30)
                ax.set_ylabel(label);ax.legend();ax.grid(axis='y',alpha=.2)
            fig.suptitle('Fixed previously unused pool; differences from C0; lower is better\nSix prospectively selected historical held-out cases, reference ceilings')
            fig.savefig(OUT/'generalization_changes.png',dpi=160);plt.close(fig)
    return timings


def run():
    start=perf_counter();freeze();c=config();failures=[]
    def call(phase,name=None,level=0,arm=None,repeat=0):
        args=[sys.executable,str(Path(__file__)),'--phase',phase,'--level',str(level),'--repeat',str(repeat)]
        if name:args+=['--name',name]
        if arm:args+=['--arm',arm]
        subprocess.run(args,cwd=ROOT,check=True,timeout=900)
    call('replay')
    # Freeze every case's U before fitting any held-out outcome.
    for name in c['trajectories']:call('prepare',name)
    call('prepare',c['mild_control']['trajectory'],1)
    def execute_case(name,level):
        try:
            call('previous',name,level)
            call('align',name,level)
            for arm,repeat in [('C0',0),('C0',1),('Cmask',0),('Cmask',1),('Cnumeric',0)]:
                call('fit',name,level,arm,repeat)
                s=json.loads((folder(name,level)/f'{arm}_repeat_{repeat}.json').read_text())
                if not s['passed']:raise RuntimeError('Current gate failed: '+str(s))
            return True
        except (subprocess.SubprocessError,RuntimeError) as e:
            failures.append(dict(trajectory=name,level=level,error=str(e)))
            dump(OUT/'stopping_decisions.json',failures)
            print('CASE STOP',failures[-1],flush=True)
            return False
        finally:collect()
    okay=[execute_case(name,0) for name in c['trajectories']]
    mild=False
    if all(okay):mild=execute_case(c['mild_control']['trajectory'],1)
    summaries=collect()
    for p,h in json.loads((OUT/'source_fingerprints.json').read_text()).items():assert M.sha(ROOT/p)==h,p
    for p in (OUT/'by_trajectory').glob('*/level_*/input_fingerprints.json'):
        for f,h in json.loads(p.read_text()).items():assert M.sha(Path(f))==h,f
    dump(OUT/'verification.json',dict(completed_current_fits=len(summaries),reference_cases_passed=sum(okay),
        all_reference_cases_pass=all(okay),mild_control_completed=mild,failures=failures,
        total_wall_seconds=perf_counter()-start,current_fit_seconds=sum(s['seconds'] for s in summaries),
        historical_inputs_unchanged=True,source_on=True,same256=True,no_fitting=True))


if __name__=='__main__':
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        if os.environ.get(key)!='1':raise RuntimeError('Set '+key+'=1')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['run','freeze','prepare','previous','replay','fit','collect','align','resume'],default='run')
    parser.add_argument('--name');parser.add_argument('--level',type=int,choices=[0,1],default=0)
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--repeat',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    if args.phase=='run':run()
    elif args.phase=='freeze':freeze()
    elif args.phase=='replay':replay()
    elif args.phase=='prepare':prepare(args.name,args.level)
    elif args.phase=='previous':previous(args.name,args.level)
    elif args.phase=='fit':fit(args.name,args.level,args.arm,args.repeat)
    elif args.phase=='align':align(args.name,args.level)
    elif args.phase=='resume':resume()
    else:collect()
