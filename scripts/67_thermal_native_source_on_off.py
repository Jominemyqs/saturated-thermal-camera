"""Output 63: source-on/off current updates with frozen output-61 full-latent C."""
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import argparse
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.source_off_audit import remove_current_increment
from src.matched_projection import MatchedHeatOperator

spec = importlib.util.spec_from_file_location('frozen_current_audit', ROOT/'scripts/65_thermal_unpinning_current_update.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
BASE = M.OUT
OUT = ROOT/'outputs/by_experiment/63_native_source_on_off'
PREVIOUS = 'all_readings_latent'
COVARIANCES = ['observed_covariance', 'pred_observed_covariance', 'prediction_variance']


def folder(name):
    return OUT/'by_trajectory'/name


def original(name, level, repeat=0):
    return M.stem(name, level, PREVIOUS, repeat)


def target(name, level, repeat=0):
    return folder(name)/f'level_{level}_source_off_repeat_{repeat}'


def case(name, level):
    return M.load(M.folder(name)/f'case_{level}.npz')


def prior(name, level):
    return M.load(M.folder(name)/f'prior_{level}_{PREVIOUS}.npz')


def repeats(level):
    return range(2 if level == 0 else 1)


def forecast_metrics(d, mean, name):
    error = mean-d['truth']; peak = int(np.argmax(d['truth'])); maximum = int(np.argmax(mean))
    rows = []
    for region, w in M.regions(d, name).items():
        if w.sum() == 0:
            continue
        avg = lambda x: float(np.sum(w*x)/w.sum())
        rows.append(dict(region=region, rmse_K=np.sqrt(avg(error**2)), mae_K=avg(abs(error)),
            bias_K=avg(error), true_peak_error_K=float(error[peak]), true_peak_absolute_error_K=float(abs(error[peak])),
            predicted_max_K=float(mean[maximum]), predicted_max_x_m=float(d['points'][maximum, 0]),
            predicted_max_y_m=float(d['points'][maximum, 1]),
            peak_location_distance_m=float(np.linalg.norm(d['points'][maximum]-d['points'][peak]))))
    return rows


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    for path, h in json.loads((BASE/'source_and_input_fingerprints.json').read_text()).items():
        assert M.sha(ROOT/path) == h, path
    inherited = json.loads((BASE/'frozen_configuration.json').read_text())
    assert inherited['samples_per_chain'] == M.SAMPLES and inherited['burn_in'] == M.BURN
    assert inherited['gate'] == M.GATE
    assert json.loads((BASE/'verification.json').read_text())['all_completed_fits_pass']
    # Ensure diagnostic packages work before any expensive sampling.
    M.HELP.arviz_diagnostics(np.random.default_rng(0).normal(size=(4, 100, 2)), np.arange(2), 'environment_probe')
    import arviz, scipy
    paths = [Path(__file__), ROOT/'src/source_off_audit.py', ROOT/'tests/test_source_off_audit.py',
        BASE/'frozen_configuration.json', BASE/'verification.json', BASE/'prior_verification.csv',
        BASE/'source_and_input_fingerprints.json',
        ROOT/'outputs/by_experiment/56_within_step_integration/gradient_provenance.json',
        ROOT/'outputs/by_experiment/56_within_step_integration/frozen_sections.csv']
    paths += [ROOT/p for p in json.loads((BASE/'source_and_input_fingerprints.json').read_text()) if p.endswith('.py')]
    paths.append(ROOT/'scripts/65_thermal_unpinning_current_update.py')
    for name in M.NAMES:
        for level in range(3):
            paths += [M.folder(name)/f'case_{level}.npz', M.folder(name)/f'prior_{level}_{PREVIOUS}.npz',
                M.PREV/'by_trajectory'/name/f'level_{level}_summaries.npz',
                M.BASE/'by_trajectory'/name/f'level_{level}.npz']
            for repeat in repeats(level):
                paths += [original(name, level, repeat).with_suffix(s) for s in ['.npz', '.json', '.metrics.csv']]
    OUT.mkdir(parents=True)
    M.dump(OUT/'source_and_input_fingerprints.json', {str(p.relative_to(ROOT)): M.sha(p) for p in sorted(set(paths))})
    M.dump(OUT/'frozen_configuration.json', dict(experiment=63, driver='scripts/67_thermal_native_source_on_off.py',
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        previous_representation=PREVIOUS, source_on='cached output61 full-latent C; replay each primary seed',
        source_off='prediction_mean minus cached gradient_contribution; corresponding observation means',
        inherited_output61=inherited, gate=M.GATE, chains=4, samples_per_chain=M.SAMPLES,
        burn_in=M.BURN, thin=1, evaluation_per_chain=M.EVAL, reference_repeats=2, high_ceiling_repeats=1,
        high_ceiling_gate='Both reference source-off seeds on both paths pass; all source-on controls already passed',
        retries=0, coefficient_fit=False, source_reinterpretation=False, previous_inference=False,
        non_saturation_flags=False, covariance_changed=False, current_noise_sd_K=.5,
        numpy=np.__version__, scipy=scipy.__version__, arviz=arviz.__version__,
        final_draws='Marginal-only for pointwise scoring; full covariance used for conditioning',
        limitation='Source-off is not gradient-independent: frozen previous/preparation/coefficients/Q unchanged'))
    times = {r['trajectory']: r['dt_s'] for r in json.loads((ROOT/'outputs/by_experiment/56_within_step_integration/gradient_provenance.json').read_text())}
    o = inherited['inherited_settings']; checks = []; forecasts = []
    oldchecks = pd.read_csv(BASE/'prior_verification.csv')
    for name in M.NAMES:
        folder(name).mkdir(parents=True)
        for level in range(3):
            d = case(name, level); on = prior(name, level)
            old = oldchecks[(oldchecks.trajectory == name)&(oldchecks.level == level)&(oldchecks.arm == PREVIOUS)].iloc[0]
            for key, column in [('prediction_mean','mean_sha256'),('prediction_variance','prior_diagonal_sha256'),
                ('pred_observed_covariance','prior_cross_sha256'),('observed_covariance','prior_observed_sha256')]:
                assert M.HELP.digest(on[key]) == old[column]
            off = remove_current_increment(on, d['observation_ids'], d['gradient_contribution'])
            with np.load(M.PREV/'by_trajectory'/name/f'level_{level}_summaries.npz') as f:
                previous = f['previous_mean_'+PREVIOUS]
                np.testing.assert_array_equal(f['forecast_'+PREVIOUS], on['prediction_mean'])
                np.testing.assert_array_equal(f['gradient_contribution'], d['gradient_contribution'])
            with np.load(M.BASE/'by_trajectory'/name/f'level_{level}.npz') as f:
                ambient = float(f['ambient'])
            xy = d['points']
            op = MatchedHeatOperator(SimpleNamespace(points=xy, triangles=d['triangles']),
                np.linspace(xy[:,0].min(), xy[:,0].max(), int(o['nx'])),
                np.linspace(xy[:,1].min(), xy[:,1].max(), int(o['ny'])))
            propagation = ambient+op.propagate(previous-ambient, 'native', dt=times[name],
                diffusivity=o['diffusivity'], cooling_rate=o['cooling_rate'], floor=True)
            np.testing.assert_allclose(off['prediction_mean'], propagation, rtol=0, atol=1e-12)
            for k in COVARIANCES:
                assert off[k] is on[k]
            peak = int(np.argmax(d['truth']))
            checks.append(dict(trajectory=name,level=level,ceiling_K=float(d['ceiling']),
                propagation_reconstruction_max_error_K=float(abs(propagation-off['prediction_mean']).max()),
                source_on_mean_sha256=M.HELP.digest(on['prediction_mean']), source_off_mean_sha256=M.HELP.digest(off['prediction_mean']),
                source_increment_sha256=M.HELP.digest(d['gradient_contribution']),
                observation_ids_sha256=M.HELP.digest(d['observation_ids']), camera_sha256=M.HELP.digest(d['camera']),
                covariance_identical=True, numeric=int((d['kinds']==0).sum()), censored=int((d['kinds']==1).sum()),
                noise_sd_K=.5, true_peak_increment_K=float(d['gradient_contribution'][peak]),
                **{k+'_sha256':M.HELP.digest(on[k]) for k in COVARIANCES}))
            np.savez_compressed(folder(name)/f'means_{level}.npz', source_on=on['prediction_mean'],
                source_off=off['prediction_mean'], increment=d['gradient_contribution'])
            for arm, b in [('source_on',on),('source_off',off)]:
                forecasts.extend([dict(trajectory=name,level=level,ceiling_K=float(d['ceiling']),arm=arm,**r)
                    for r in forecast_metrics(d,b['prediction_mean'],name)])
    pd.DataFrame(checks).to_csv(OUT/'equality_checks.csv',index=False)
    pd.DataFrame(forecasts).to_csv(OUT/'forecast_metrics.csv',index=False)
    print('PREPARED six matched priors; full covariance unchanged',flush=True)


def sample(name, level, repeat, replay=False):
    path = folder(name)/f'replay_{level}' if replay else target(name,level,repeat)
    if path.with_suffix('.json').exists() or path.with_suffix('.npz').exists():
        raise FileExistsError(path)
    d = case(name,level); b = prior(name,level)
    if not replay:
        b = remove_current_increment(b,d['observation_ids'],d['gradient_contribution'])
    monitor = np.union1d(d['observation_ids'], M.spread_nodes(d['points'], np.ones(len(d['points']),bool),64))
    start = perf_counter(); latent = []; noisy = []
    for chain in range(4):
        pred,z = M.sample_mixed_noisy_blocks(ids=d['observation_ids'],kinds=d['kinds'],numerical_values=d['values'],
            thresholds=float(d['ceiling']),**b,noise_sd=.5,n_samples=M.SAMPLES,burn_in=M.BURN,thin=1,
            seed=int(d['seed'])+repeat*M.REPEAT_OFFSET+100000*chain)
        latent.append(pred[4]);noisy.append(z)
        print('CHAIN',name,level,'on-replay' if replay else 'off',repeat,chain,flush=True)
    latent = np.asarray(latent);noisy = np.asarray(noisy)
    evaluation = latent[:,np.linspace(0,M.SAMPLES-1,M.EVAL,dtype=int),:]
    pred = M.prediction_from_draws(evaluation.reshape(-1,latent.shape[-1]))
    if replay:
        with np.load(original(name,level).with_suffix('.npz')) as old:
            for key, value in zip(M.SUFFIXES,pred):np.testing.assert_array_equal(value,old[key])
            np.testing.assert_array_equal(noisy,old['noisy_censored_chains'])
            np.testing.assert_array_equal(latent[:,:,monitor],old['monitored_chains'])
        M.dump(path.with_suffix('.json'),dict(trajectory=name,level=level,exact=True,seconds=perf_counter()-start))
        print('EXACT SOURCE-ON REPLAY',name,level,flush=True)
        return
    diagnostics = pd.concat([M.HELP.arviz_diagnostics(noisy,d['observation_ids'][d['kinds']==1],'noisy_censored'),
        M.HELP.arviz_diagnostics(latent[:,:,monitor],monitor,'latent_temperature')],ignore_index=True)
    classic = M.chain_diagnostics(latent,selected_mask=d['camera']==d['ceiling']); nc = M.chain_diagnostics(noisy)
    common = dict(trajectory=name,level=level,ceiling_K=float(d['ceiling']),arm='source_off',repeat=repeat)
    rows = M.metrics(d,pred,name);peak = int(np.argmax(d['truth']))
    for row in rows:
        row['true_peak_absolute_error_K'] = abs(row['true_peak_error_K'])
        row['current_update_at_true_peak_K'] = float(pred[0][peak]-b['prediction_mean'][peak])
    pd.DataFrame([dict(**common,**r) for r in rows]).to_csv(path.with_suffix('.metrics.csv'),index=False)
    diagnostics.to_csv(path.with_suffix('.diagnostics.csv'),index=False)
    summary = dict(**common,seconds=perf_counter()-start,peak_RSS_MiB=M.HELP.rss(),
        rank_rhat_max=float(diagnostics.rank_rhat.max()),ess_bulk_min=float(diagnostics.ess_bulk.min()),
        ess_tail_min=float(diagnostics.ess_tail.min()),classical_rhat_max=max(classic['rhat_max'],nc['rhat_max']),
        finite=bool(np.isfinite(diagnostics[['rank_rhat','ess_bulk','ess_tail']]).all().all()),
        current_seed=int(d['seed'])+repeat*M.REPEAT_OFFSET,noise_sd_K=.5,
        prior_cross_sha256=M.HELP.digest(b['pred_observed_covariance']),mean_sha256=M.HELP.digest(b['prediction_mean']))
    summary['passed'] = bool(summary['finite'] and all(summary[k]<=M.GATE[k] for k in ['rank_rhat_max','classical_rhat_max'])
        and summary['ess_bulk_min']>=400 and summary['ess_tail_min']>=400
        and summary['seconds']<=M.GATE['arm_seconds_max'] and summary['peak_RSS_MiB']<=M.GATE['peak_RSS_MiB_max'])
    np.savez_compressed(path.with_suffix('.npz'),**dict(zip(M.SUFFIXES,pred)),evaluation_chains=evaluation,
        monitored_chains=latent[:,:,monitor],monitor_ids=monitor,noisy_censored_chains=noisy,
        constrained_ids=d['observation_ids'][d['kinds']==1])
    M.dump(path.with_suffix('.json'),summary)
    print('FIT',summary,flush=True)


def collect():
    tables=[]; summaries=[]; changes=[];peakrows=[]
    for name in M.NAMES:
        for level in range(3):
            d=case(name,level);peak=int(np.argmax(d['truth']))
            with np.load(folder(name)/f'means_{level}.npz') as f:means={k:f[k] for k in ['source_on','source_off']}
            for repeat in repeats(level):
                new=target(name,level,repeat)
                if not new.with_suffix('.json').exists():continue
                for arm,path in [('source_on',original(name,level,repeat)),('source_off',new)]:
                    df=pd.read_csv(path.with_suffix('.metrics.csv'));df['arm']=arm
                    if arm=='source_on':
                        df['true_peak_absolute_error_K']=abs(df.true_peak_error_K)
                        with np.load(path.with_suffix('.npz')) as f:update=float(f['mean'][peak]-means[arm][peak])
                        df['current_update_at_true_peak_K']=update
                    tables.append(df)
                    s=json.loads(path.with_suffix('.json').read_text());s.update(arm=arm,reused=arm=='source_on');summaries.append(s)
                    r=df.iloc[0]
                    peakrows.append(dict(trajectory=name,level=level,repeat=repeat,arm=arm,
                        forecast_at_true_peak_K=float(means[arm][peak]),truth_peak_K=float(d['truth'][peak]),
                        posterior_at_true_peak_K=float(r.true_peak_error_K+d['truth'][peak]),
                        update_K=float(r.current_update_at_true_peak_K),covered=bool(r.true_peak_covered)))
            if new.with_suffix('.json').exists():plot(name,level,means)
    data=pd.concat(tables,ignore_index=True)
    data.to_csv(OUT/'posterior_metrics.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'convergence_runtime.csv',index=False)
    pd.DataFrame(peakrows).to_csv(OUT/'peak_recovery.csv',index=False)
    for keys,g in data.groupby(['trajectory','level','repeat','region']):
        g=g.set_index('arm')
        changes.append(dict(zip(['trajectory','level','repeat','region'],keys),
            **{k:float(g.loc['source_off',k]-g.loc['source_on',k]) for k in ['rmse_K','crps_K','bias_K','coverage95','width95_K','true_peak_error_K']}))
    pd.DataFrame(changes).to_csv(OUT/'paired_changes.csv',index=False)
    return summaries


def plot(name,level,means):
    d=case(name,level);sections=pd.read_csv(ROOT/'outputs/by_experiment/56_within_step_integration/frozen_sections.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True,constrained_layout=True)
    predictions=[]
    for col,(arm,path) in enumerate([('Source on',original(name,level)),('Source off',target(name,level))]):
        f=M.load(path.with_suffix('.npz'));predictions.append(f)
        for row,(section,g) in enumerate(sections[sections.trajectory==name].groupby('section',sort=False)):
            g=g.sort_values('along_mm');ids=g.node.to_numpy(int);x=g.along_mm.to_numpy();ax=axes[row,col]
            ax.plot(x,d['truth'][ids],'k',label='Native truth')
            ax.plot(x,means['source_on' if col==0 else 'source_off'][ids],'--',color='#D55E00',label='Physics forecast')
            ax.plot(x,f['mean'][ids],color='#0072B2',label='Current posterior')
            ax.fill_between(x,f['lower'][ids],f['upper'][ids],color='#0072B2',alpha=.2,label='Pointwise 95% interval')
            ax.axhline(float(d['ceiling']),color='#777777',ls=':',label='Camera ceiling')
            ax.set(title=arm+' | '+('Along' if row==0 else 'Across'),xlabel='Distance from fixed current peak (mm)',ylabel='Temperature (K)');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'{name}; ceiling {float(d["ceiling"]):.3f} K\nSame full-latent previous state, current observations and prior covariance; only explicit increment removed')
    fig.savefig(OUT/f'cross_sections_{name}_level_{level}.png',dpi=150);plt.close(fig)
    tri=mtri.Triangulation(d['points'][:,0]*1000,d['points'][:,1]*1000,d['triangles'])
    errors=[means['source_on']-d['truth'],means['source_off']-d['truth']]+[f['mean']-d['truth'] for f in predictions]
    limit=max(abs(e).max() for e in errors)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,e,label in zip(axes.flat,errors,['On forecast','Off forecast','On posterior','Off posterior']):
        im=ax.tripcolor(tri,e,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='gouraud')
        ax.set(title=label,xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
    fig.colorbar(im,ax=list(axes.flat),label='Estimate minus truth (K)',shrink=.8)
    fig.suptitle(f'{name}; ceiling {float(d["ceiling"]):.3f} K; shared error scale')
    fig.savefig(OUT/f'signed_errors_{name}_level_{level}.png',dpi=150);plt.close(fig)


def run():
    start=perf_counter()
    def call(phase,name=None,level=None,repeat=None):
        args=[sys.executable,str(Path(__file__)),'--phase',phase]
        for k,v in [('name',name),('level',level),('repeat',repeat)]:
            if v is not None:args+=['--'+k,str(v)]
        subprocess.run(args,cwd=ROOT,check=True,timeout=900)
    call('prepare')
    for name in M.NAMES:
        for level in range(3):call('replay',name,level,0)
    for name in M.NAMES:
        for repeat in range(2):call('fit',name,0,repeat)
    ref=[json.loads(target(n,0,r).with_suffix('.json').read_text()) for n in M.NAMES for r in range(2)]
    passed=all(s['passed'] for s in ref)
    M.dump(OUT/'reference_gate.json',dict(passed=passed,gate=M.GATE,failed=[s for s in ref if not s['passed']]))
    if passed:
        for name in M.NAMES:
            for level in [1,2]:call('fit',name,level,0)
    summaries=collect()
    for path,h in json.loads((OUT/'source_and_input_fingerprints.json').read_text()).items():assert M.sha(ROOT/path)==h,path
    M.dump(OUT/'verification.json',dict(completed_source_off_fits=sum(not s['reused'] for s in summaries),
        source_on_primary_replays_exact=6,reference_gate_pass=passed,all_fits_pass=all(s['passed'] for s in summaries),
        historical_inputs_unchanged=True,full_covariance_identical=True,previous_inference_rerun=False,
        source_off_fit_seconds=sum(s['seconds'] for s in summaries if not s['reused']),
        replay_seconds=sum(json.loads((folder(n)/f'replay_{l}.json').read_text())['seconds'] for n in M.NAMES for l in range(3)),
        total_wall_seconds=perf_counter()-start,
        timing_note='Fit timers exclude final compressed-cache writing; reused source-on timers are historical, not new runtime'))


if __name__=='__main__':
    for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        if os.environ.get(k)!='1':raise RuntimeError('Set '+k+'=1')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['run','prepare','replay','fit','collect'],default='run')
    parser.add_argument('--name',choices=M.NAMES);parser.add_argument('--level',type=int,choices=[0,1,2])
    parser.add_argument('--repeat',type=int,choices=[0,1],default=0)
    a=parser.parse_args()
    if a.phase=='run':run()
    elif a.phase=='prepare':prepare()
    elif a.phase=='replay':sample(a.name,a.level,a.repeat,replay=True)
    elif a.phase=='fit':sample(a.name,a.level,a.repeat)
    else:collect()
