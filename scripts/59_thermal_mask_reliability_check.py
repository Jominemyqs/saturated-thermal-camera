"""Bounded sampling audit of experiment 54; no new model or location selection.

This audit harness reads the unchanged driver 58 to reuse its executed protocol
and reporting functions. It is not a replacement canonical experiment driver.
Each fit runs in a fresh process so peak RSS is attributable to that fit's job.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT/'outputs/by_experiment/54_native_non_saturation_mask'
OUT = ROOT/'outputs/by_experiment/55_mask_reliability_and_ceiling'
SPEC = importlib.util.spec_from_file_location('frozen_mask_protocol', ROOT/'scripts/58_thermal_native_non_saturation_mask.py')
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)
protocol.OUT = OUT
SAMPLES, BURN, EVALUATION = 800, 300, 300


def check_originals():
    for filename in ['source_fingerprints.json', 'input_fingerprints.json']:
        for name, digest in json.loads((PILOT/filename).read_text()).items():
            actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            if actual != digest:
                raise RuntimeError('Frozen experiment-54 input/source changed: '+name)


def selected(name):
    with np.load(PILOT/'by_trajectory'/name/'selection.npz') as s:
        return s['baseline_ids'], s['added_ids'], s['pool']


def options():
    return json.loads((PILOT/'frozen_configuration.json').read_text())['inherited_settings']


def prepare():
    check_originals()
    OUT.mkdir(parents=True, exist_ok=True)
    config = dict(parent='54_native_non_saturation_mask', samples_per_chain=SAMPLES,
        burn_in=BURN, thin=1, chains=4, evaluation_draws_per_chain=EVALUATION,
        additional_retry=False, location_budget=256, location_selection='reuse exact experiment-54 IDs',
        gate=protocol.GATE, classical_rhat_limit=options()['maximum_rhat'],
        reference_jobs='C0/Cmask at both paired seeds; Cnumeric at primary seed',
        secondary_jobs='C0/Cmask/Cnumeric at primary seed only; contingent on reference gate',
        changed='current chain length and burn-in only; unchanged model and evaluation draw count',
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    config_path=OUT/'frozen_configuration.json'
    if config_path.exists():
        assert json.loads(config_path.read_text())==config
    else:
        protocol.dump(config_path,config)
    compatibility=[]
    for name in protocol.NAMES:
        base,u,pool=selected(name)
        folder=OUT/'by_trajectory'/name
        folder.mkdir(parents=True,exist_ok=True)
        for filename in ['selected_locations.csv','selection.npz']:
            original=PILOT/'by_trajectory'/name/filename
            destination=folder/filename
            if destination.exists():
                assert destination.read_bytes()==original.read_bytes()
            else:
                destination.write_bytes(original.read_bytes())
        for level in [0,1,2]:
            with np.load(protocol.BASE/'by_trajectory'/name/f'level_{level}.npz') as d:
                assert np.all(d['camera'][pool]<d['ceiling'])
                ceiling=float(d['ceiling'])
                if level==0:
                    draws=d['C_draws'][:,u]
                else:
                    with np.load(protocol.RETAINED/'by_trajectory'/name/f'level_{level}.npz') as retained:
                        draws=retained['retained_draws'][:,u]
                probabilities=ndtr((ceiling-draws)/.5).mean(axis=0)
                compatibility.extend(dict(trajectory=name,level=level,ceiling_K=ceiling,node=int(node),
                    baseline_non_saturation_probability=float(probability),
                    source='historical short-chain baseline; descriptive, no new inference')
                    for node,probability in zip(u,probabilities))
    pd.DataFrame(compatibility).to_csv(OUT/'cached_baseline_flag_compatibility.csv',index=False)
    paths=[Path(__file__),ROOT/'scripts/58_thermal_native_non_saturation_mask.py',
           ROOT/'src/native_mask_pilot.py',PILOT/'source_fingerprints.json',PILOT/'input_fingerprints.json']
    protocol.dump(OUT/'source_fingerprints.json',{
        str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def long_fit(d,b,ids,kinds,values,seed,monitor,path):
    start=perf_counter()
    latent,noisy=[],[]
    for chain in range(4):
        prediction,z=protocol.sample_mixed_noisy_blocks(ids=ids,kinds=kinds,
            numerical_values=values,thresholds=float(d['ceiling']),**b,noise_sd=.5,
            n_samples=SAMPLES,burn_in=BURN,thin=1,seed=seed+100000*chain)
        latent.append(prediction[4]);noisy.append(z)
    latent=np.asarray(latent);noisy=np.asarray(noisy)
    classical=protocol.chain_diagnostics(latent,selected_mask=d['camera']==d['ceiling'])
    noisy_classical=protocol.chain_diagnostics(noisy)
    indices=np.linspace(0,SAMPLES-1,EVALUATION,dtype=int)
    evaluation=latent[:,indices,:]
    prediction=protocol.prediction_from_draws(evaluation.reshape(-1,latent.shape[-1]))
    diagnostics=pd.concat([
        protocol.arviz_diagnostics(noisy,ids[kinds!=0],'noisy_constraint'),
        protocol.arviz_diagnostics(latent[:,:,monitor],monitor,'latent_temperature')])
    diagnostics.to_csv(path.with_suffix('.diagnostics.csv'),index=False)
    summary=dict(seed=seed,samples_per_chain=SAMPLES,burn_in=BURN,thin=1,retries=0,
        numerical_count=int(np.sum(kinds==0)),lower_count=int(np.sum(kinds==1)),
        upper_count=int(np.sum(kinds==-1)),observation_count=len(ids),
        classical_rhat_max=classical['rhat_max'],classical_censored_rhat_max=classical['selected_rhat_max'],
        noisy_classical_rhat_max=noisy_classical['rhat_max'],
        rank_rhat_max=float(diagnostics.rank_rhat.max()),ess_bulk_min=float(diagnostics.ess_bulk.min()),
        ess_tail_min=float(diagnostics.ess_tail.min()),
        diagnostics_finite=bool(np.isfinite(diagnostics[['rank_rhat','ess_bulk','ess_tail']]).all().all()),
        seconds=perf_counter()-start,process_peak_rss_MiB=protocol.rss(),
        prior_diagonal_sha256=protocol.digest(b['prediction_variance']),
        prior_mean_sha256=protocol.digest(b['prediction_mean']))
    gate=protocol.GATE
    summary['reliable_for_expansion']=bool(summary['diagnostics_finite']
        and summary['rank_rhat_max']<=gate['rank_rhat_max']
        and summary['ess_bulk_min']>=gate['ess_bulk_min']
        and summary['ess_tail_min']>=gate['ess_tail_min']
        and max(classical['rhat_max'],classical['selected_rhat_max'],noisy_classical['rhat_max'])<=options()['maximum_rhat']
        and summary['seconds']<=gate['arm_seconds_max']
        and summary['process_peak_rss_MiB']<=gate['process_peak_rss_MiB_max'])
    np.savez_compressed(path,**dict(zip(protocol.SUFFIXES,prediction)),evaluation_chains=evaluation,
        monitored_chains=latent[:,:,monitor],monitored_ids=monitor,noisy_constraint_chains=noisy,
        constraint_ids=ids[kinds!=0],observation_ids=ids,likelihood_kinds=kinds,numerical_values=values)
    protocol.dump(path.with_suffix('.json'),summary)
    return prediction,summary


def worker(args):
    check_originals()
    if args.level>0 and not json.loads((OUT/'reference_gate.json').read_text())['proceed_to_high_ceilings']:
        raise RuntimeError('Reference gate failed: no higher-ceiling worker allowed')
    path=OUT/'by_trajectory'/args.name/f'level_{args.level}_{args.arm}_repeat_{args.repeat}.npz'
    if path.exists():
        raise RuntimeError('Refusing to overwrite completed fit: '+str(path))
    base,u,pool=selected(args.name)
    d,h,cfg,dt,prior=protocol.inputs(args.name,args.level,options())
    ids,kinds,values=protocol.arm_layout(d,base,u,args.arm)
    blocks=protocol.blocks(d,h,cfg,dt,ids)
    baseline=protocol.blocks(d,h,cfg,dt,base)
    where=np.searchsorted(ids,base)
    np.testing.assert_allclose(blocks['observed_covariance'][np.ix_(where,where)],
        baseline['observed_covariance'],rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(blocks['pred_observed_covariance'][:,where],
        baseline['pred_observed_covariance'],rtol=1e-13,atol=1e-13)
    if args.level==0:
        old=json.loads((PILOT/'by_trajectory'/args.name/'prior_level_0.json').read_text())
        for key in ['previous_draws_sha256','previous_mean_sha256','physics_mean_sha256',
                    'features_sha256','prior_variance_sha256','camera_sha256']:
            assert old[key]==prior[key],key
    else:
        # Verify the saved retained observations, not only their prose description.
        with np.load(protocol.RETAINED/'by_trajectory'/args.name/f'level_{args.level}.npz') as retained:
            np.testing.assert_array_equal(base,retained['retained_observation_ids'])
    monitor=np.union1d(np.union1d(base,u),protocol.spread_nodes(d['points'],np.ones(len(pool),bool),64))
    prediction,summary=long_fit(d,blocks,ids,kinds,values,
        prior['current_seed']+args.repeat*protocol.REPEAT_OFFSET,monitor,path)
    common=dict(trajectory=args.name,level=args.level,ceiling_K=float(d['ceiling']),arm=args.arm,repeat=args.repeat)
    pd.DataFrame([dict(**common,**row) for row in protocol.metrics(d,prediction,pool,u)]).to_csv(path.with_suffix('.metrics.csv'),index=False)
    if args.arm=='C0':
        p=ndtr((float(d['ceiling'])-prediction[4][:,u])/.5).mean(axis=0)
        pd.DataFrame(dict(node=u,baseline_non_saturation_probability=p)).to_csv(path.with_suffix('.compatibility.csv'),index=False)
    print('DONE',common,'rank',summary['rank_rhat_max'],'ESS',summary['ess_bulk_min'],summary['ess_tail_min'],
          'seconds',summary['seconds'],'RSS MiB',summary['process_peak_rss_MiB'],flush=True)


def collect(stage):
    rows=[];checks=[]
    for name in protocol.NAMES:
        for p in (OUT/'by_trajectory'/name).glob('level_*_*.metrics.csv'):
            df=pd.read_csv(p)
            if (stage=='reference') != (int(df.level.iloc[0])==0):
                continue
            rows.append(df)
            common=df[['trajectory','level','ceiling_K','arm','repeat']].iloc[0].to_dict()
            checks.append(dict(**common,**json.loads(p.with_name(p.name.replace('.metrics.csv','.json')).read_text())))
    pd.concat(rows,ignore_index=True).to_csv(OUT/f'{stage}_metrics.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/f'{stage}_convergence_runtime.csv',index=False)
    return checks


def run_stage(stage):
    if stage=='secondary':
        if not json.loads((OUT/'reference_gate.json').read_text())['proceed_to_high_ceilings']:
            raise RuntimeError('Reference gate failed: higher-ceiling fits forbidden')
    for name in protocol.NAMES:
        for level in ([0] if stage=='reference' else [1,2]):
            for repeat in ([0,1] if level==0 else [0]):
                for arm in (['C0','Cmask','Cnumeric'] if repeat==0 else ['C0','Cmask']):
                    p=OUT/'by_trajectory'/name/f'level_{level}_{arm}_repeat_{repeat}.npz'
                    if not p.exists():
                        print('START',name,level,arm,repeat,flush=True)
                        subprocess.run([sys.executable,__file__,'--stage','worker','--name',name,
                            '--level',str(level),'--arm',arm,'--repeat',str(repeat)],check=True,cwd=ROOT)
                    elif not p.with_suffix('.metrics.csv').exists():
                        raise RuntimeError('Partial cached fit requires inspection: '+str(p))
                    checks=collect(stage)
    if stage=='reference':
        assert len(checks)==10
        protocol.dump(OUT/'reference_gate.json',dict(proceed_to_high_ceilings=all(r['reliable_for_expansion'] for r in checks),
            thresholds=protocol.GATE,failed=[r for r in checks if not r['reliable_for_expansion']],
            decision='One bounded 800/300 check; no further retries or threshold changes'))
    protocol.report()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','reference','secondary','worker','report','all'],default='all')
    parser.add_argument('--name',choices=protocol.NAMES)
    parser.add_argument('--level',type=int,choices=[0,1,2])
    parser.add_argument('--arm',choices=['C0','Cmask','Cnumeric'])
    parser.add_argument('--repeat',type=int,choices=[0,1])
    args=parser.parse_args()
    for key in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']:
        assert os.environ.get(key)=='1',key
    if args.stage=='worker':
        worker(args);return
    if args.stage=='report':
        protocol.report();return
    prepare()
    if args.stage in ['reference','all']:
        run_stage('reference')
    if args.stage=='all' and not json.loads((OUT/'reference_gate.json').read_text())['proceed_to_high_ceilings']:
        print('STOP: reference gate failed; no high-ceiling updates',flush=True);return
    if args.stage in ['secondary','all']:
        run_stage('secondary')


if __name__=='__main__':
    main()
