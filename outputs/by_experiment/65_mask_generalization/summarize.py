"""Reporting only: read completed output-65 caches, never run inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
config=json.loads((OUT/'frozen_configuration.json').read_text())
results=pd.read_csv(OUT/'results.csv')
runtime=pd.read_csv(OUT/'convergence_runtime.csv')
paired=pd.read_csv(OUT/'paired_changes.csv')
status=[];previous=[];compatibility=[]
reasons={
    'DiagonalScanPath_1':'Initial rank R-hat 1.010955 > 1.01; longer retry hit the 900 s predecessor-stage watchdog.',
    'HorizontalScanPath_1':'Longer previous chains passed (rank R-hat 1.003918); prior construction then exceeded the combined 900 s stage watchdog.',
    'SpiralScanPath_12':'Previous reflected HMC raised RuntimeError: Reflected HMC produced an infeasible draw; no current fit.',
}
for name in config['trajectories']:
    p=OUT/'by_trajectory'/name/'level_0'
    g=runtime[runtime.trajectory==name]
    complete=len(g)==5 and g.passed.all()
    status.append(dict(trajectory=name,historical_role='heldout',current_fits=len(g),
                       complete_reliable_comparison=bool(complete),reason='Completed' if complete else reasons[name]))
    if (p/'previous_sampling.json').exists():
        previous.extend(dict(trajectory=name,**row) for row in json.loads((p/'previous_sampling.json').read_text()))
    for file in sorted(p.glob('flag_compatibility_repeat_*.csv')):
        values=pd.read_csv(file).probability
        compatibility.append(dict(trajectory=name,repeat=int(file.stem.rsplit('_',1)[1]),
            mean_probability=values.mean(),min_probability=values.min(),fraction_below_half=(values<.5).mean(),
            fraction_below_one_tenth=(values<.1).mean()))
pd.DataFrame(status).to_csv(OUT/'case_status.csv',index=False)
pd.DataFrame(previous).to_csv(OUT/'previous_convergence_runtime.csv',index=False)
pd.DataFrame(compatibility).to_csv(OUT/'flag_compatibility_summary.csv',index=False)
primary=results[(results.repeat==0)&(results.region=='unused_pool')]
primary.to_csv(OUT/'primary_unused_region_table.csv',index=False)
repeats=[]
for (name,region),g in paired[paired.arm=='Cmask'].groupby(['trajectory','region']):
    g=g.set_index('repeat')
    if set(g.index)!={0,1}:continue
    for metric in ['rmse_K','crps_K','coverage95','width95_K','rmse_reduction_percent']:
        a,b=float(g.loc[0,metric]),float(g.loc[1,metric])
        repeats.append(dict(trajectory=name,region=region,metric=metric,paired_seed0=a,paired_seed1=b,
                            between_repeat_difference=abs(a-b),same_sign=bool(a*b>0)))
pd.DataFrame(repeats).to_csv(OUT/'repeat_seed_comparisons.csv',index=False)
assert len(runtime)==15 and runtime.passed.all()
assert sum(r['complete_reliable_comparison'] for r in status)==3
assert not results.true_peak_covered.any()
for name,g in runtime.groupby('trajectory'):
    for field in ['prior_mean_sha256','prior_diagonal_sha256']:
        assert g[field].nunique()==1
    p=OUT/'by_trajectory'/name/'level_0'
    with np.load(p/'case.npz') as d, np.load(p/'prior.npz') as b, np.load(p/'extended_prior_aligned.npz') as e:
        ids=e['observation_ids']; old=d['observation_ids']; pos=np.searchsorted(ids,old)
        np.testing.assert_array_equal(e['prediction_mean'],b['prediction_mean'])
        np.testing.assert_array_equal(e['prediction_variance'],b['prediction_variance'])
        np.testing.assert_array_equal(e['pred_observed_covariance'][:,pos],b['pred_observed_covariance'])
        np.testing.assert_array_equal(e['observed_covariance'][np.ix_(pos,pos)],b['observed_covariance'])
        for arm in ['Cmask','Cnumeric']:
            with np.load(p/(arm+'_repeat_0.npz')) as f:
                np.testing.assert_array_equal(f['observation_ids'],ids)
                assert len(np.unique(ids))==len(ids)
                add=np.isin(ids,d['added_ids'])
                np.testing.assert_array_equal(f['likelihood_kinds'][add],-np.ones(256,int) if arm=='Cmask' else np.zeros(256,int))
                if arm=='Cmask':assert np.isnan(f['numerical_values'][add]).all()
                else:np.testing.assert_array_equal(f['numerical_values'][add],d['camera'][d['added_ids']])
valid=[r['trajectory'] for r in status if r['complete_reliable_comparison']]
fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
specs=[('unused_pool','rmse_K','Unused-pool RMSE change (K)'),
       ('remaining_unused','crps_K','Remaining-unused CRPS change (K)'),
       ('hot1pct_area','coverage95','Hot-area coverage change (percentage points)')]
for ax,(region,metric,label) in zip(axes,specs):
    for j,arm in enumerate(['Cmask','Cnumeric']):
        g=paired[(paired.arm==arm)&(paired.region==region)]
        factor=100 if metric=='coverage95' else 1
        center=[];lo=[];hi=[]
        for name in valid:
            values=g[g.trajectory==name][metric].to_numpy()*factor
            center.append(values.mean());lo.append(values.mean()-values.min());hi.append(values.max()-values.mean())
        ax.bar(np.arange(len(valid))+(j-.5)*.34,center,width=.34,label=arm,color=['#0072B2','#D55E00'][j],
               yerr=np.array([lo,hi]),capsize=3)
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(np.arange(len(valid)))
    ax.set_xticklabels([n.replace('ScanPath',' ') for n in valid],rotation=20)
    ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
axes[0].legend()
fig.suptitle('3 of 6 preselected historical held-out cases completed; reference ceilings only\nBars: paired changes from C0. Whiskers: range across two current seeds for mask, NOT confidence intervals.',fontsize=11)
fig.savefig(OUT/'validated_comparisons.png',dpi=160);plt.close(fig)
summary=dict(valid_cases=valid,incomplete_cases=[r for r in status if not r['complete_reliable_comparison']],
    valid_current_fits=len(runtime),all_valid_current_fits_pass=True,
    rank_rhat_max=float(runtime.rank_rhat_max.max()),bulk_ess_min=float(runtime.ess_bulk_min.min()),
    tail_ess_min=float(runtime.ess_tail_min.min()),peak_RSS_MiB=float(runtime.peak_RSS_MiB.max()),
    fit_seconds=float(runtime.seconds.sum()),per_arm_runtime_seconds={arm:dict(min=float(g.seconds.min()),max=float(g.seconds.max())) for arm,g in runtime.groupby('arm')},
    all_true_peaks_uncovered=True,baseline_extended_overlap_exact=True,likelihood_ID_alignment_verified=True,
    no_mask_arm_numerical_values_at_U=True,milder_ceiling_inference=False,
    elapsed_wall_from_initial_protocol_write_s=(OUT/'verification.json').stat().st_mtime-(OUT/'frozen_configuration.json').stat().st_mtime,
    wall_time_includes_invalid_attempts_and_correction=True)
(OUT/'report_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
