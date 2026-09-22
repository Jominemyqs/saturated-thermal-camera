"""Frozen forecast attribution only: no GP inference, fitting, or remedies."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.thermal_posterior_physics import prepare_trajectory, posterior_physics_means, DEVELOPMENT_TRAJECTORIES
from src.thermal_trajectory import trajectory_catalog

BASE = ROOT / 'outputs/by_experiment/45_censoring_severity_decomposition'
OUT = ROOT / 'outputs/by_experiment/46_oracle_forecast_decomposition'


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stats(e, mask):
    v = e[mask]
    return dict(n_pixels=int(mask.sum()), bias_K=float(v.mean()) if v.size else np.nan,
                rmse_K=float(np.sqrt(np.mean(v*v))) if v.size else np.nan,
                mae_K=float(np.mean(abs(v))) if v.size else np.nan)


def hottest(y):
    mask = np.zeros(y.size, bool)
    mask[np.argsort(y.ravel(), kind='mergesort')[-25:]] = True
    return mask


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for path, expected in json.loads((BASE/'source_fingerprints.json').read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    o = pd.read_csv(BASE/'fixed_configuration.csv').iloc[0].to_dict()
    pd.DataFrame([o]).to_csv(OUT/'fixed_configuration.csv', index=False)
    old = pd.read_csv(BASE/'means.csv').query("model=='shared_physics_prior'")
    checks = pd.read_csv(BASE/'checks.csv').query("model=='B'").set_index(['trajectory','level_index'])
    catalog = trajectory_catalog(ROOT.parent/'heat_eq_laser_trajectories')
    assert len(catalog)==33 and sum(r.name in DEVELOPMENT_TRAJECTORIES for r in catalog)==3
    rows=[]; gains=[]; status=[]; audits=[]; provenance={}
    for r in catalog:
        p, _ = prepare_trajectory(ROOT.parent/'heat_eq_laser_trajectories',r.name,
            nx=int(o['nx']),ny=int(o['ny']),heat_flux_cutoff=o['heat_flux_cutoff'])
        n=len(p.times)-1
        # The inherited preparer also calculates diagnostic fits; none are used.
        kw=dict(previous_index=n-1,current_index=n,diffusivity=o['diffusivity'],
            cooling_rate=o['cooling_rate'],source_coupling=o['source_coupling'],
            source_flux_threshold=o['source_flux_threshold'])
        def forecast(field):
            return posterior_physics_means(p,field,**kw)[0].ravel()
        prevtrue=p.history[n-1].ravel(); truth=p.truth.ravel()
        oracle=forecast(p.history[n-1]); physics=oracle-truth
        folder=OUT/'by_trajectory'/r.name
        folder.mkdir(parents=True,exist_ok=True)
        for li in range(6):
            cache=BASE/'by_trajectory'/r.name/f'level_{li}.npz'
            with np.load(cache) as d:
                prev=d['previous_mean']; baseline=forecast(prev)
                assert np.array_equal(baseline,d['physics_prior'].ravel())
                assert digest(baseline)==checks.loc[(r.name,li),'mean_sha256']
                assert np.array_equal(truth,d['truth_evaluation_only'].ravel())
                assert np.array_equal(hottest(truth),d['top25_mask'].ravel())
                assert np.array_equal(oracle,forecast(p.history[n-1]))
                c=float(d['ceiling']); previous_camera=d['previous_camera'].ravel()
                prevsat=previous_camera==c
                # Averaging thousands of identical pinned draws adds roundoff.
                pinned_delta=float(np.max(abs(prev.ravel()[~prevsat]-previous_camera[~prevsat])))
                assert pinned_delta<1e-9
                masks=dict(overall=np.ones(truth.size,bool),current_top25=d['top25_mask'].ravel(),
                    previous_top25=hottest(prevtrue),current_observed_censored=d['observed_censored_mask'].ravel())
                identity=dict(trajectory=r.name,family=r.family,role='development' if r.name in DEVELOPMENT_TRAJECTORIES else 'heldout',
                    level_index=li,ceiling_K=c,actual_censored_fraction=float(masks['current_observed_censored'].mean()))
                errors=dict(previous=prev.ravel()-prevtrue,state=baseline-oracle,physics=physics,total=baseline-truth)
                residual=float(np.max(abs(errors['total']-errors['state']-physics)))
                assert residual<1e-12
                metricdiff=0.
                for region,mask in masks.items():
                    for component,e in errors.items():
                        rows.append({**identity,'region':region,'component':component,**stats(e,mask)})
                    oldregion=dict(overall='overall',current_top25='top_1pct',current_observed_censored='observed_censored').get(region)
                    if oldregion:
                        saved=old[(old.trajectory==r.name)&(old.level_index==li)&(old.region==oldregion)].iloc[0]
                        s=stats(errors['total'],mask)
                        for key in ['bias_K','rmse_K']:
                            metricdiff=max(metricdiff,abs(s[key]-saved[key]))
                            assert abs(s[key]-saved[key])<1e-12
                    rm_prev=stats(errors['previous'],mask)['rmse_K']
                    mse_state=float(np.mean(errors['state'][mask]**2))
                    mse_phys=float(np.mean(physics[mask]**2))
                    cross=float(2*np.mean(errors['state'][mask]*physics[mask]))
                    mse_total=float(np.mean(errors['total'][mask]**2))
                    assert abs(mse_total-mse_state-mse_phys-cross)<1e-10
                    gains.append({**identity,'region':region,'gain_rmse':np.sqrt(mse_state)/rm_prev if rm_prev>1e-12 else np.nan,
                        'raw_rmse_K':rm_prev,'state_rmse_K':np.sqrt(mse_state),
                        'mse_total_K2':mse_total,'mse_state_K2':mse_state,'mse_physics_K2':mse_phys,'cross_term_K2':cross})
                    for label,selection in [('observed_censored',prevsat),('observed_unsaturated',~prevsat)]:
                        subset=mask&selection
                        status.append({**identity,'region':region,'previous_status':label,**stats(errors['previous'],subset),
                            'region_pixels':int(mask.sum()),'status_fraction':float(subset.sum()/mask.sum()),
                            'bias_contribution_K':float(errors['previous'][subset].sum()/mask.sum())})
                audits.append({**identity,'baseline_sha256':digest(baseline),'oracle_sha256':digest(oracle),
                    'decomposition_max_abs_K':residual,'baseline_metric_max_abs_K':metricdiff,
                    'unsaturated_pinned_verified':True,'pinned_mean_roundoff_K':pinned_delta})
                np.savez_compressed(folder/f'level_{li}.npz',previous_estimate=prev,previous_truth=prevtrue,
                    current_truth=truth,baseline_forecast=baseline,oracle_forecast=oracle,
                    **{'error_'+k:v for k,v in errors.items()},**{'mask_'+k:v for k,v in masks.items()},
                    previous_observed_censored=prevsat,ceiling=c)
            provenance[str(cache.relative_to(ROOT))]=dict(baseline_sha256=digest(baseline),previous_mean_sha256=digest(prev))
        print(r.name,'six exact baseline reproductions passed',flush=True)
    for key,data in [('results',rows),('amplification',gains),('previous_status',status),('audits',audits)]:
        f=pd.DataFrame(data); f.to_csv(OUT/f'{key}.csv',index=False)
        if key=='audits': continue
        h=f[f.role=='heldout']
        extra=['region']+(['component'] if key=='results' else ['previous_status'] if key=='previous_status' else [])
        for prefix,suffix in [([], 'heldout30'),(['family'],'family')]:
            h.groupby(prefix+['level_index']+extra).mean(numeric_only=True).to_csv(OUT/f'{key}_{suffix}_summary.csv')
        if key=='results':
            ref=f[f.level_index==0].set_index(['trajectory','region','component'])
            paired=[]
            for _,row in f.iterrows():
                rr=ref.loc[(row.trajectory,row.region,row.component)]
                paired.append({**{k:row[k] for k in ['trajectory','family','role','level_index','region','component']},
                    **{k+'_change':row[k]-rr[k] for k in ['bias_K','rmse_K','mae_K']}})
            pd.DataFrame(paired).to_csv(OUT/'paired_changes.csv',index=False)
    a=pd.DataFrame(audits)
    assert a.groupby('trajectory').oracle_sha256.nunique().eq(1).all()
    verification=dict(trajectories=33,development=3,heldout=30,cases=len(a),
        exact_baseline_arrays=True,oracle_identical_across_ceilings=True,unsaturated_pinned_verified=True,
        decomposition_max_abs_K=float(a.decomposition_max_abs_K.max()),baseline_metric_max_abs_K=float(a.baseline_metric_max_abs_K.max()),
        pinned_mean_roundoff_max_K=float(a.pinned_mean_roundoff_K.max()),no_gp_inference=True,no_refitting=True)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    files=[Path(__file__),ROOT/'src/thermal_posterior_physics.py',BASE/'fixed_configuration.csv']
    (OUT/'source_fingerprints.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
    plot()


def plot():
    f=pd.read_csv(OUT/'results_heldout30_summary.csv').query("region=='current_top25'")
    g=pd.read_csv(OUT/'amplification_heldout30_summary.csv').query("region=='current_top25'")
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors=dict(previous='#555555',total='#0072B2',state='#D55E00',physics='#009E73')
    names=dict(total='Baseline forecast',state='Propagated state contribution',physics='Oracle-state physics')
    prev=f[f.component=='previous']; x=prev.actual_censored_fraction*100
    axes[0,0].plot(x,prev.bias_K,'o-',label='Previous-state bias',color='#555555')
    axes[0,0].plot(x,prev.rmse_K,'s--',label='Previous-state RMSE',color='#CC79A7')
    for component in ['total','state','physics']:
        r=f[f.component==component]
        for ax,metric in [(axes[0,1],'bias_K'),(axes[1,0],'rmse_K')]:
            ax.plot(r.actual_censored_fraction*100,r[metric],'o-',color=colors[component],label=names[component])
    axes[1,1].plot(g.actual_censored_fraction*100,g.gain_rmse,'o-',color='#0072B2',label='Mean per-trajectory RMSE ratio')
    axes[1,1].axhline(1,color='gray',ls=':')
    for ax,title,ylabel in zip(axes.ravel(),['Previous-state error at current hot locations','Exact forecast bias decomposition','RMSEs are not additive','Local propagation ratio (not an operator norm)'],['Error (K)','Bias (K)','RMSE (K)','State RMSE / previous RMSE']):
        ax.set(title=title,xlabel='Observed current censoring (%)',ylabel=ylabel)
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0,0].axhline(0,color='black',lw=.5); axes[0,1].axhline(0,color='black',lw=.5)
    fig.suptitle('Oracle forecast attribution: 30 held-out trajectories, fixed current hottest 25 pixels\nFrozen physics and cached previous estimates; no current GP inference or tuning')
    fig.savefig(OUT/'oracle_decomposition.png',dpi=180); plt.close(fig)


if __name__=='__main__': main()
