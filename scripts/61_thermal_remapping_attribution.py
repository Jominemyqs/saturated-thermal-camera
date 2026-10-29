"""Output 57: remapping versus constant-gradient within-step evolution."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import hashlib
import json
import resource
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.matched_projection import MatchedHeatOperator
from src.within_step_integration import error_metrics, source_metrics
from src.remapping_attribution import constant_gradient_attribution

BASE=ROOT/'outputs/by_experiment/56_within_step_integration'
OUT=ROOT/'outputs/by_experiment/57_remapping_diffusion_attribution'
PRIMARY=['S0','S_map','S_evolved_GL4']
COLORS=['#666666','#0072B2','#D55E00']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    start=perf_counter()
    config=json.loads((BASE/'frozen_configuration.json').read_text())
    o=config['inherited_options']
    provenance={r['trajectory']:r for r in json.loads((BASE/'gradient_provenance.json').read_text())}
    sections=pd.read_csv(BASE/'frozen_sections.csv')
    fingerprint_paths=[BASE/'frozen_configuration.json',BASE/'gradient_provenance.json',BASE/'frozen_sections.csv',
        Path(__file__), ROOT/'src/remapping_attribution.py',ROOT/'src/matched_projection.py',
        ROOT/'src/within_step_integration.py',ROOT/'tests/test_remapping_attribution.py']
    fingerprints={str(p.relative_to(ROOT)):sha(p) for p in fingerprint_paths}
    write_json(OUT/'frozen_configuration.json',dict(
        experiment=57,driver=str(Path(__file__).relative_to(ROOT)),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        parent='Experiment 56; frozen 601x401 operator, evaluation sections, native area weights',
        trajectories=config['trajectories'],levels=config['levels'],inherited_options=o,
        source='constant thresholded CURRENT gradient; no previous gradient or temporal averaging',
        quadrature_orders=[2,4],primary_evolved='GL4',gp_inference=False,fitting=False,
        attribution='S_map-S0 and S_evolved-S_map; do not propagate an already remapped source',
        stopping='No GP or new model after this descriptive attribution; no new success threshold'))
    metrics=[]; source_rows=[]; point_rows=[]; quad=[]; checks=[]
    for name in config['trajectories']:
        cases=[]; src=None; oracle=None
        for level in range(3):
            path=BASE/'by_trajectory'/name/f'level_{level}.npz'
            fingerprints[str(path.relative_to(ROOT))]=sha(path)
            with np.load(path) as f: c={k:f[k] for k in f.files}
            q=c['current_thresholded_gradient']; dt=provenance[name]['dt_s']
            np.testing.assert_array_equal(q,np.where(c['current_gradient']>=o['source_flux_threshold'],c['current_gradient'],0))
            if src is None:
                mesh=SimpleNamespace(points=c['points'],triangles=c['triangles'])
                xs=np.linspace(mesh.points[:,0].min(),mesh.points[:,0].max(),int(o['nx']))
                ys=np.linspace(mesh.points[:,1].min(),mesh.points[:,1].max(),int(o['ny']))
                op=MatchedHeatOperator(mesh,xs,ys)
                def propagate(values,age):
                    return op.propagate(values,'native',dt=age,diffusivity=o['diffusivity'],
                                        cooling_rate=o['cooling_rate'],floor=False)
                src=constant_gradient_attribution(q,gamma=o['source_coupling'],dt=dt,propagate=propagate)
                for variant,s in src.items():
                    source_rows.append(dict(trajectory=name,variant=variant,**source_metrics(s,c['points'],c['area_weights'])))
                peak=int(np.argmax(c['truth']))
                point_rows.append(dict(trajectory=name,true_peak_node=peak,
                    **{v:float(s[peak]) for v,s in src.items()},
                    remapping_change_K=float((src['S_map']-src['S0'])[peak]),
                    evolution_after_remapping_change_K=float((src['S_evolved_GL4']-src['S_map'])[peak]),
                    total_change_K=float((src['S_evolved_GL4']-src['S0'])[peak])))
                np.testing.assert_allclose(src['S_evolved_GL4']-src['S0'],
                    (src['S_map']-src['S0'])+(src['S_evolved_GL4']-src['S_map']),rtol=0,atol=2e-14)
                original_q=q.copy(); original_points=c['points'].copy(); original_weights=c['area_weights'].copy()
            np.testing.assert_array_equal(q,original_q)
            np.testing.assert_array_equal(c['points'],original_points)
            np.testing.assert_array_equal(c['area_weights'],original_weights)
            np.testing.assert_array_equal(src['S0'],c['source_S0'])
            # Experiment 56's evolution-only row was trapezoidal, not this integral.
            old_trapezoid=o['source_coupling']*dt/2*(propagate(q,dt)+q)
            np.testing.assert_array_equal(old_trapezoid,c['source_S_evolve'])
            truth=c['truth']; weights=c['area_weights']; hot=c['hot_area_weights']
            peak=int(np.argmax(truth))
            regions=dict(whole_area=weights,hot1pct_area=hot,observed_censored=weights*(c['camera']==c['ceiling']))
            selected=sections[sections.trajectory==name]
            for section,g in selected.groupby('section',sort=False):
                ids=g[g.shoulder].node.to_numpy(int)
                w=np.zeros_like(weights); w[ids]=weights[ids]
                regions['shoulder_'+('across' if 'Across' in section else 'along')]=w
            fields={}
            for state in ['inferred','oracle']:
                propagation=c[state+'_propagation']
                fields[state]={v:propagation+s for v,s in src.items()}
                np.testing.assert_array_equal(fields[state]['S0'],c[state+'_S0'])
                if state=='oracle':
                    if oracle is None: oracle={k:v.copy() for k,v in fields[state].items()}
                    for v in src: np.testing.assert_array_equal(fields[state][v],oracle[v])
                for variant,field in fields[state].items():
                    for region,w in regions.items():
                        metrics.append(dict(trajectory=name,level=level,ceiling_K=float(c['ceiling']),predecessor=state,
                            variant=variant,region=region,**error_metrics(field,truth,w),
                            true_peak_signed_error_K=float(field[peak]-truth[peak]),
                            true_peak_absolute_error_K=float(abs(field[peak]-truth[peak]))))
                delta=src['S_evolved_GL2']-src['S_evolved_GL4']
                quad.append(dict(trajectory=name,level=level,predecessor=state,
                    max_source_difference_K=float(abs(delta).max()),
                    relative_source_L2=float(np.sqrt(np.sum(weights*delta**2)/np.sum(weights*src['S_evolved_GL4']**2))),
                    hot_rmse_difference_K=error_metrics(fields[state]['S_evolved_GL2'],truth,hot)['rmse_K']-
                                          error_metrics(fields[state]['S_evolved_GL4'],truth,hot)['rmse_K']))
            folder=OUT/'by_trajectory'/name; folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/f'level_{level}.npz',points=c['points'],triangles=c['triangles'],
                truth=truth,area_weights=weights,hot_area_weights=hot,ceiling=c['ceiling'],
                **{'source_'+k:v for k,v in src.items()},
                **{state+'_'+k:v for state,group in fields.items() for k,v in group.items()})
            checks.append(dict(trajectory=name,level=level,baseline_exact=True,old_evolution_trapezoid_exact=True,
                oracle_ceiling_invariant=True,dt_s=dt,source_identity_max_abs_K=float(abs(
                    src['S_evolved_GL4']-src['S0']-(src['S_map']-src['S0'])-(src['S_evolved_GL4']-src['S_map'])).max())))
            cases.append((level,c,fields))
        # One compact section figure per ceiling, including both predecessor controls.
        ylim=(min(float(c['truth'].min()) for _,c,_ in cases)-1,
              max(float(v.max()) for _,_,f in cases for g in f.values() for v in g.values())+1)
        for level,c,fields in cases:
            fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
            for col,(section,g) in enumerate(selected.groupby('section',sort=False)):
                g=g.sort_values('along_mm'); ids=g.node.to_numpy(int); x=g.along_mm.to_numpy()
                for row,state in enumerate(['inferred','oracle']):
                    ax=axes[row,col]; ax.plot(x,c['truth'][ids],'k-',lw=2,label='Truth')
                    for v,color in zip(PRIMARY,COLORS): ax.plot(x,fields[state][v][ids],color=color,label=v)
                    ax.axhline(float(c['ceiling']),color='#999999',ls=':',label='Ceiling')
                    for lo,hi in [(-1.,-.25),(.25,1.)]:ax.axvspan(lo,hi,color='#999999',alpha=.1)
                    ax.set_ylim(*ylim);ax.set_title(state+': '+section,fontsize=9)
                    ax.set_ylabel('Temperature (K)');ax.set_xlabel('Distance from fixed true peak (mm)')
                    ax.grid(alpha=.15);ax.legend(fontsize=8)
            fig.suptitle(f'{name}; c={float(c["ceiling"]):.2f} K\nConstant current gradient; no GP update; shaded fixed shoulders')
            fig.savefig(OUT/f'sections_{name}_level{level}.png',dpi=150);plt.close(fig)
        print(name,'completed',flush=True)
    m=pd.DataFrame(metrics)
    pairs=[]
    for (name,level,state,region),g in m.groupby(['trajectory','level','predecessor','region']):
        g=g.set_index('variant')
        for left,right,label in [('S0','S_map','remapping'),('S_map','S_evolved_GL4','evolution_after_remapping'),('S0','S_evolved_GL4','total')]:
            pairs.append(dict(trajectory=name,level=level,predecessor=state,region=region,comparison=label,
                **{k:float(g.loc[right,k]-g.loc[left,k]) for k in ['rmse_K','bias_K','mae_K','true_peak_signed_error_K']},
                mse_change_K2=float(g.loc[right,'rmse_K']**2-g.loc[left,'rmse_K']**2)))
    for stem,frame in [('metrics',m),('paired_changes',pd.DataFrame(pairs)),('source_components',pd.DataFrame(source_rows)),
                       ('true_peak_attribution',pd.DataFrame(point_rows)),('quadrature_checks',pd.DataFrame(quad)),
                       ('equality_checks',pd.DataFrame(checks))]:frame.to_csv(OUT/f'{stem}.csv',index=False)
    for p,h in fingerprints.items():assert sha(ROOT/p)==h,p
    write_json(OUT/'source_fingerprints.json',fingerprints)
    write_json(OUT/'verification.json',dict(baseline_exact=True,old_evolution_trapezoid_exact=True,
        prior_inputs_unchanged=True,oracle_ceiling_invariant=True,gp_inference=False,parameters_fitted=False,
        elapsed_seconds=perf_counter()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        limitation='Numerical evolution conditional on frozen interpolant/boundaries, not isolated 3D physical diffusion; no new spatial refinement.'))


if __name__=='__main__':main()
