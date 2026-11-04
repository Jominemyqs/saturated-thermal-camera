"""Output 60: all-camera previous-state conditioning and latent unpinning.

Previous-posterior/forecast audit only; no current GP, source or noise change.
"""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
import scipy.signal
from scipy.sparse import csr_matrix, hstack
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.censored_gp import RBFConfig
from src.matched_projection import MatchedHeatOperator
from src.native_mask_pilot import spread_nodes
from src.native_surface import hottest_area_weights
from src.previous_correlation_audit import product, psd_factor
from src.previous_unpinning_audit import (
    noisy_precision, conditional_camera, latent_from_camera,
    conditional_covariance_block, mixture_moments,
    propagated_conditional_residual, propagated_state_components,
)
from src.truncated_gaussian import sample_lower_truncated_gaussian_hmc
from src.mcmc_diagnostics import gelman_rubin
from src.within_step_integration import error_metrics

BASE = ROOT/'outputs/by_experiment/52_native_higher_ceiling'
CORR = ROOT/'outputs/by_experiment/59_previous_conditional_correlation_audit'
P56 = ROOT/'outputs/by_experiment/56_within_step_integration'
OUT = ROOT/'outputs/by_experiment/60_previous_unpinning_audit'
NAMES = ['DiagonalScanPath_8', 'SpiralScanPath_13']
GATE = dict(rank_rhat=1.01, ess_bulk=400., ess_tail=400.,
            case_seconds=900., peak_RSS_MiB=4096.)
ARMS = ['legacy_hybrid', 'correlation_control', 'all_readings_pinned', 'all_readings_latent']
COLORS = ['#555555', '#009E73', '#D55E00', '#0072B2']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def diagnostics(chains, ids, kind):
    import arviz as az
    data = az.from_dict(posterior={'v': chains})
    return pd.DataFrame(dict(node=ids, variable=kind, classical_rhat=gelman_rubin(chains),
        rank_rhat=az.rhat(data, method='rank')['v'].values,
        ess_bulk=az.ess(data, method='bulk')['v'].values,
        ess_tail=az.ess(data, method='tail')['v'].values))


def read_case(name, level):
    keys = ['points','triangles','area_weights','hot_area_weights','truth','previous_truth',
            'previous_camera','previous_mean','previous_draws','ceiling','numeric_candidates',
            'ambient','physics_mean','gradient_contribution','C_inherited_variance','C_prior_variance']
    with np.load(BASE/'by_trajectory'/name/f'level_{level}.npz') as f:
        c = {k: f[k] for k in keys}
    with np.load(CORR/'by_trajectory'/name/f'level_{level}.npz') as f:
        for key in ['shared_variance','joint_residual_variance','previous_marginal_variance',
                    'previous_censored_ids']:
            c['audit59_'+key] = f[key]
    return c


def regions_for(c, sections):
    w = c['area_weights']; sat = c['previous_camera'] == c['ceiling']
    regions = dict(whole_area=w, previous_unsaturated=w*~sat, previous_censored=w*sat,
                   current_hot1pct_area=c['hot_area_weights'],
                   previous_hot1pct_area=hottest_area_weights(c['previous_truth'], w))
    for section, group in sections.groupby('section', sort=False):
        weights = np.zeros_like(w)
        ids = group[group.shoulder].node.to_numpy(int)
        weights[ids] = w[ids]
        regions['shoulder_'+('along' if 'Along' in section else 'across')] = weights
    return regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args(); out = args.output
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        if os.environ.get(key) != '1': raise RuntimeError(f'Set {key}=1')
    if not hasattr(scipy.signal, 'gaussian'): scipy.signal.gaussian = scipy.signal.windows.gaussian
    import arviz as az
    start = perf_counter()
    o = pd.read_csv(BASE/'frozen_configuration.csv').iloc[0].to_dict()
    meta = pd.read_csv(BASE/'verification.csv')
    provenance = {r['trajectory']: r for r in json.loads((P56/'gradient_provenance.json').read_text())}
    sections = pd.read_csv(P56/'frozen_sections.csv')
    paths = [Path(__file__), ROOT/'src/previous_unpinning_audit.py', ROOT/'src/previous_correlation_audit.py',
             ROOT/'tests/test_previous_unpinning_audit.py', ROOT/'src/censored_gp.py',
             ROOT/'src/matched_projection.py', ROOT/'src/truncated_gaussian.py',
             ROOT/'src/native_surface.py', ROOT/'src/native_mask_pilot.py',
             BASE/'frozen_configuration.csv', BASE/'verification.csv',
             P56/'gradient_provenance.json', P56/'frozen_sections.csv', CORR/'frozen_configuration.json']
    for name in NAMES:
        for level in range(3):
            paths += [BASE/'by_trajectory'/name/f'level_{level}.npz',
                      CORR/'by_trajectory'/name/f'level_{level}.npz']
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    # The geometry, kernel, noise and timestep are shared, so C(T|all Z) is shared.
    first = read_case(NAMES[0], 0); xy = first['points']; n = len(xy)
    if 10*n*n*8/2**20 > GATE['peak_RSS_MiB']:
        raise RuntimeError('Conservative dense-matrix preflight exceeds memory budget')
    union = np.zeros(n, bool)
    for name in NAMES:
        for level in range(3):
            with np.load(BASE/'by_trajectory'/name/f'level_{level}.npz') as f:
                np.testing.assert_array_equal(f['points'], xy)
                union |= f['previous_camera'] == f['ceiling']
    union_ids = np.flatnonzero(union)
    assert meta.lengthscale_m.nunique() == 1
    assert len({p['dt_s'] for p in provenance.values()}) == 1
    physics = dict(dt=provenance[NAMES[0]]['dt_s'], diffusivity=o['diffusivity'], cooling_rate=o['cooling_rate'])
    cfg = RBFConfig(mean_temp=float(first['ambient']), signal_sd=o['signal_sd'],
                    lengthscale=float(meta.lengthscale_m.iloc[0]), noise_sd=o['previous_noise_sd'])
    out.mkdir(parents=True)
    write_json(out/'source_fingerprints.json', fingerprints)
    write_json(out/'frozen_configuration.json', dict(experiment=60,
        driver=str(Path(__file__).relative_to(ROOT)),
        revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        inherited_options=o, trajectories=NAMES, planned_ceilings=['reference',320,325],
        execution_order='Both reference cases first; higher ceilings only if both pass reliability/runtime',
        arms={
            'legacy_hybrid':'Cached sparse censored GP plus all unsaturated noisy values pinned',
            'correlation_control':'Output59 joint residual correction on censored block, same cached mean',
            'all_readings_pinned':'Condition censored GP on ALL unsaturated readings, then pin unsaturated latent values',
            'all_readings_latent':'SAME all-readings posterior as pinned bridge; retain full latent covariance and mean'},
        matched_information='All arms use the same camera values somewhere; likelihood use differs in the bridge, explicitly',
        latent_covariance='Analytic C=vI-v^2P; no independent residual approximation in propagation',
        gate=GATE, hmc_schedule=[dict(chains=4,retained=800,burn_in=300,thin=1),
                               dict(chains=4,retained=1600,burn_in=600,thin=1)],
        seed_rule='Inherited previous_seed + 10000*chain + 1000003; retry uses same chain seeds',
        residual_monitor_seed_offset=7000001, current_gp=False, parameter_fit=False,
        full_field_joint_residual_sampling=False, full_joint_covariance_retained_analytically=True,
        truth_use='Evaluation and fixed evaluation regions only; inherited simulator-derived preparation retained',
        numerical_prior_jitter=cfg.relative_jitter*cfg.signal_sd**2,
        dimension_preflight=dict(native_nodes=n, one_dense_matrix_MiB=n*n*8/2**20,
                                conservative_ten_matrices_MiB=10*n*n*8/2**20),
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,arviz=az.__version__,
                         PYTHONPATH=os.environ.get('PYTHONPATH'))))
    print('Preflight:', n, 'native nodes;', round(n*n*8/2**20,2), 'MiB per dense matrix', flush=True)
    setup_start = perf_counter()
    p, v, inverse_error = noisy_precision(xy, cfg)
    print('Full noisy precision ready:', round(perf_counter()-setup_start,2), 's', flush=True)
    mesh = SimpleNamespace(points=xy, triangles=first['triangles'])
    op = MatchedHeatOperator(mesh, np.linspace(xy[:,0].min(),xy[:,0].max(),int(o['nx'])),
                             np.linspace(xy[:,1].min(),xy[:,1].max(),int(o['ny'])))
    chunks=[]
    for index in range(0,n,32):
        ids=np.arange(index,min(index+32,n)); unit=np.zeros((len(ids),n)); unit[np.arange(len(ids)),ids]=1.
        chunks.append(csr_matrix(op.propagate(unit,'native',**physics).T))
    a=hstack(chunks,format='csr'); del chunks
    # Only EXACT zero entries are omitted, no magnitude-based sparsification.
    probe=np.random.default_rng(60).normal(size=(3,n))
    operator_error=float(abs(op.propagate(probe,'native',**physics)-(a@probe.T).T).max())
    assert operator_error < 1e-10
    full_r, ac_union = propagated_conditional_residual(p,v,a,union_ids)
    shared_setup=dict(seconds=perf_counter()-setup_start, precision_inverse_max_error=inverse_error,
        operator_max_error=operator_error, sparse_operator_nnz=a.nnz,
        sparse_operator_MiB=(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes)/2**20,
        precision_MiB=p.nbytes/2**20, ac_union_MiB=ac_union.nbytes/2**20,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    write_json(out/'shared_setup.json',shared_setup)
    if shared_setup['peak_RSS_MiB']>GATE['peak_RSS_MiB']:
        raise RuntimeError('Shared setup exceeded memory budget; do not expand')
    del first, probe
    metrics=[]; covariance_rows=[]; diagnostic_rows=[]; sampling_rows=[]; checks=[]; mcrows=[]; info_rows=[]
    reference_pass={}; completed=[]
    plan=[(name,0) for name in NAMES]+[(name,level) for level in (1,2) for name in NAMES]
    for name,level in plan:
        if level>0 and not all(reference_pass.get(x,False) for x in NAMES): break
        case_start=perf_counter(); c=read_case(name,level); ambient=float(c['ambient']); ceiling=float(c['ceiling'])
        common=dict(trajectory=name,level=level,ceiling_K=ceiling)
        oldmeta=meta[(meta.trajectory==name)&(meta.level==level)&(meta.model=='C')].iloc[0]
        assert digest(c['previous_draws'])==oldmeta.previous_draws_sha256
        sat=c['previous_camera']==ceiling; numerical=~sat
        ids, zmean, zcov, filled=conditional_camera(p,c['previous_camera'],sat,ambient)
        np.testing.assert_array_equal(ids,c['audit59_previous_censored_ids'])
        base,w,conditional_diag=latent_from_camera(p,v,filled,ambient,ids)
        monitor=np.unique(np.r_[spread_nodes(xy,numerical,64),spread_nodes(xy,sat,min(64,len(ids)))])
        cm=conditional_covariance_block(p,v,monitor,monitor)
        monitor_factor, factor_info=psd_factor(cm)
        folder=out/'by_trajectory'/name;folder.mkdir(parents=True,exist_ok=True)
        seed=int(oldmeta.previous_seed); accepted=False; final_diag=None
        for attempt,(samples,burn) in enumerate([(800,300),(1600,600)]):
            if perf_counter()-case_start>GATE['case_seconds']: break
            tick=perf_counter(); zchains=[]
            for chain in range(4):
                zchains.append(sample_lower_truncated_gaussian_hmc(zmean,zcov,ceiling,
                    n_samples=samples,burn_in=burn,thin=1,seed=seed+10000*chain+1000003,
                    relative_jitter=cfg.relative_jitter))
                print(name,level,'attempt',attempt,'chain',chain,'complete',flush=True)
            zchains=np.asarray(zchains)
            if (zchains<ceiling-1e-7).any(): raise ValueError('Infeasible noisy censoring sample')
            latent_monitor=[]
            for chain in range(4):
                conditional=base[monitor]+product(zchains[chain]-zmean,w[monitor].T)
                noise=np.random.default_rng(seed+7000001+10000*chain).normal(size=conditional.shape)
                latent_monitor.append(conditional+product(noise,monitor_factor.T))
            latent_monitor=np.asarray(latent_monitor)
            diagnostic=pd.concat([diagnostics(zchains,ids,'noisy_censored_signal'),
                                  diagnostics(latent_monitor,monitor,'joint_latent_monitor')],ignore_index=True)
            for k,value in dict(**common,attempt=attempt).items(): diagnostic[k]=value
            diagnostic_rows.append(diagnostic); final_diag=diagnostic
            finite=bool(np.isfinite(diagnostic[['rank_rhat','ess_bulk','ess_tail']]).all().all())
            accepted=bool(finite and diagnostic.rank_rhat.max()<=GATE['rank_rhat'] and
                          diagnostic.ess_bulk.min()>=GATE['ess_bulk'] and diagnostic.ess_tail.min()>=GATE['ess_tail'])
            timing=dict(**common,attempt=attempt,chains=4,samples_per_chain=samples,burn_in=burn,
                seconds=perf_counter()-tick,rank_rhat_max=float(diagnostic.rank_rhat.max()),
                classical_rhat_max=float(diagnostic.classical_rhat.max()),
                bulk_ess_min=float(diagnostic.ess_bulk.min()),tail_ess_min=float(diagnostic.ess_tail.min()),
                statistical_gate=accepted,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
            sampling_rows.append(timing)
            np.savez_compressed(folder/f'level_{level}_attempt_{attempt}.npz',
                noisy_censored_chains=zchains,joint_latent_monitor=latent_monitor,monitor_ids=monitor,
                censored_ids=ids,numerical_ids=np.flatnonzero(numerical),z_conditional_mean=zmean,
                z_conditional_covariance=zcov,latent_monitor_conditional_covariance=cm)
            pd.DataFrame(sampling_rows).to_csv(out/'sampling_runtime.csv',index=False)
            pd.concat(diagnostic_rows).to_csv(out/'sampling_diagnostics.csv',index=False)
            print('DIAGNOSTICS',common,timing,flush=True)
            if accepted: break
        if final_diag is None: raise RuntimeError('No sampling attempt within case budget')
        new_mean,new_var,new_shared=mixture_moments(base,w,zchains,zmean,conditional_diag)
        pinned=c['previous_camera'].copy();pinned[sat]=new_mean[sat]
        pinned_var=np.where(sat,new_var,0.)
        positions=np.searchsorted(union_ids,ids)
        np.testing.assert_array_equal(union_ids[positions],ids)
        components=propagated_state_components(a,p,v,ids,zchains,ac_union[:,positions],full_r)
        q=c['C_prior_variance']-c['C_inherited_variance']
        np.testing.assert_allclose(q,q[0],atol=1e-12,rtol=0)
        legacy_var=np.var(c['previous_draws'],axis=0,ddof=1)
        legacy_variance_error=float(abs(np.var((a@(c['previous_draws']-c['previous_mean']).T).T,
                                               axis=0,ddof=1)-c['C_inherited_variance']).max())
        assert legacy_variance_error<1e-9
        np.testing.assert_array_equal(c['previous_draws'][:,numerical],
                                      np.broadcast_to(c['previous_camera'][numerical],(len(c['previous_draws']),numerical.sum())))
        del c['previous_draws']
        corr_var=np.zeros(n);corr_var[sat]=c['audit59_previous_marginal_variance']
        means=dict(legacy_hybrid=c['previous_mean'],correlation_control=c['previous_mean'],
                   all_readings_pinned=pinned,all_readings_latent=new_mean)
        prev_vars=dict(legacy_hybrid=legacy_var,correlation_control=corr_var,
                       all_readings_pinned=pinned_var,all_readings_latent=new_var)
        inherited=dict(legacy_hybrid=c['C_inherited_variance'],
                        correlation_control=c['audit59_shared_variance']+c['audit59_joint_residual_variance'],
                        all_readings_pinned=components['censored'],all_readings_latent=components['total'])
        forecasts={arm:ambient+op.propagate(mean-ambient,'native',floor=True,**physics)+c['gradient_contribution']
                   for arm,mean in means.items()}
        np.testing.assert_array_equal(forecasts['legacy_hybrid'],c['physics_mean'])
        np.testing.assert_array_equal(new_mean[sat],pinned[sat])
        np.testing.assert_array_equal(new_var[sat],pinned_var[sat])
        selected=sections[sections.trajectory==name]; regions=regions_for(c,selected)
        for arm in ARMS:
            for stage,field,truth,variance in [('previous',means[arm],c['previous_truth'],prev_vars[arm]),
                                             ('forecast',forecasts[arm],c['truth'],q+inherited[arm])]:
                for region,weights in regions.items():
                    if weights.sum()<=0:continue
                    avg=lambda value:float(np.sum(weights*value)/weights.sum())
                    metrics.append(dict(**common,arm=arm,stage=stage,region=region,
                        nodes=int((weights>0).sum()),area_m2=float(weights.sum()),
                        **error_metrics(field,truth,weights),mean_sd_K=avg(np.sqrt(np.maximum(variance,0.))),
                        mean_variance_K2=avg(variance),
                        current_peak_signed_error_K=float((field-truth)[np.argmax(c['truth'])]),
                        stage_peak_signed_error_K=float((field-truth)[np.argmax(truth)])))
        for region,weights in regions.items():
            if weights.sum()<=0:continue
            avg=lambda value:float(np.sum(weights*value)/weights.sum())
            covariance_rows.append(dict(**common,region=region,
                **{key+'_K2':avg(value) for key,value in components.items()},Q_K2=avg(q),
                total_prior_variance_K2=avg(q+components['total']),
                prior_change_vs_matched_pinned_percent=100*avg(components['total']-components['censored'])/avg(q+components['censored'])))
        # Four independent HMC chain means quantify numerical sensitivity of the mean comparisons.
        for chain in range(4):
            lm=base+product(w,(zchains[chain].mean(axis=0)-zmean)[:,None]).ravel()
            pm=c['previous_camera'].copy();pm[sat]=lm[sat]
            for arm,previous in [('all_readings_pinned',pm),('all_readings_latent',lm)]:
                forecast=ambient+op.propagate(previous-ambient,'native',floor=True,**physics)+c['gradient_contribution']
                for region,weights in regions.items():
                    if weights.sum()<=0:continue
                    for stage,value,truth in [('previous',previous,c['previous_truth']),('forecast',forecast,c['truth'])]:
                        mcrows.append(dict(**common,chain=chain,arm=arm,stage=stage,region=region,
                                           **error_metrics(value,truth,weights)))
        np.savez_compressed(folder/f'level_{level}_summaries.npz',points=xy,triangles=c['triangles'],
            area_weights=c['area_weights'],hot_area_weights=c['hot_area_weights'],
            previous_truth=c['previous_truth'],truth=c['truth'],previous_camera=c['previous_camera'],
            ceiling=ceiling,numerical_ids=np.flatnonzero(numerical),censored_ids=ids,
            historical_likelihood_ids=np.flatnonzero(c['numeric_candidates']|sat),
            all_readings_likelihood_ids=np.arange(n),gradient_contribution=c['gradient_contribution'],
            conditional_latent_diagonal=conditional_diag,latent_mean_shared_variance=new_shared,
            **{'previous_mean_'+arm:value for arm,value in means.items()},
            **{'previous_variance_'+arm:value for arm,value in prev_vars.items()},
            **{'forecast_'+arm:value for arm,value in forecasts.items()},
            **{'inherited_variance_'+arm:value for arm,value in inherited.items()},
            **{'covariance_component_'+key:value for key,value in components.items()})
        elapsed=perf_counter()-case_start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        passed=bool(accepted and elapsed<=GATE['case_seconds'] and rss<=GATE['peak_RSS_MiB'])
        if level==0: reference_pass[name]=passed
        checks.append(dict(**common,statistical_gate=accepted,expansion_gate=passed,
            case_seconds=elapsed,peak_RSS_MiB=rss,legacy_mean_exact=True,
            legacy_inherited_max_abs_K2=legacy_variance_error,
            matched_censored_marginals_identical=True,source_identical=True,
            **factor_info))
        info_rows.append(dict(**common,native_nodes=n,all_readings_numerical=int(numerical.sum()),
            censored=int(sat.sum()),legacy_numerical_likelihood=int((c['numeric_candidates']&numerical).sum()),
            legacy_pinned_readings=int(numerical.sum()),newly_likelihood_conditioned_readings=int((~c['numeric_candidates']&numerical).sum())))
        for file,data in [('metrics',metrics),('propagated_covariance_components',covariance_rows),
                          ('chain_mean_sensitivity',mcrows),('verification_cases',checks),('information_sets',info_rows)]:
            pd.DataFrame(data).to_csv(out/f'{file}.csv',index=False)
        fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
        for col,(section,g) in enumerate(selected.groupby('section',sort=False)):
            g=g.sort_values('along_mm');nodes=g.node.to_numpy(int);x=g.along_mm.to_numpy()
            axes[0,col].plot(x,c['previous_truth'][nodes],'k',lw=2,label='Previous truth')
            axes[1,col].plot(x,c['truth'][nodes],'k',lw=2,label='Current truth')
            for arm,color in zip(ARMS,COLORS):
                if arm=='correlation_control':continue  # Mean identical to legacy, avoid obscuring it.
                axes[0,col].plot(x,means[arm][nodes],color=color,label=arm)
                axes[1,col].plot(x,forecasts[arm][nodes],color=color,label=arm)
            for ax in axes[:,col]:
                ax.axhline(ceiling,color='#999999',ls=':');ax.grid(alpha=.2)
                for lo,hi in [(-1.,-.25),(.25,1.)]:ax.axvspan(lo,hi,color='#999999',alpha=.1)
                ax.set_xlabel('Distance from fixed current peak (mm)');ax.set_ylabel('Temperature (K)')
            axes[0,col].set_title(section);axes[0,col].legend(fontsize=7)
        fig.suptitle(f'{name}; ceiling {ceiling:.3f} K\nPrevious latent mean and deterministic forecast; NO current GP update')
        fig.savefig(out/f'sections_{name}_level_{level}.png',dpi=150);plt.close(fig)
        completed.append(common)
        print('CASE COMPLETE',common,'seconds',round(elapsed,2),'gate',passed,flush=True)
        del c,base,w,zchains,new_shared,new_var,prev_vars,means,forecasts,inherited,components
    assert all(sha(ROOT/path)==h for path,h in fingerprints.items())
    write_json(out/'verification.json',dict(completed=completed,reference_gate=reference_pass,
        all_inputs_unchanged=True,current_gp_inference=False,full_latent_covariance_analytic=True,
        unsaturated_censored_cross_terms_preserved=True,source_noise_Q_unchanged=True,
        runtime_seconds=perf_counter()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        cautions=['All-readings bridge changes how already-used readings enter inference',
                  'Mean uses positive floor; covariance retains the inherited linear propagation approximation',
                  'Inherited ambient/kernel/source inputs remain simulator-derived',
                  'No current posterior performance or coverage claim']))


if __name__=='__main__':
    main()
