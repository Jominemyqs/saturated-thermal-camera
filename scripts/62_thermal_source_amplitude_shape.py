"""Output 58: fixed-shape amplitude sweep and development-only oracle calibration."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import resource
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.source_amplitude_diagnostic import scale_diagnostic,select_shared_scale
from src.within_step_integration import native_source_contributions,error_metrics
from src.thermal_posterior_physics import DEVELOPMENT_TRAJECTORIES
from src.thermal_trajectory import ThermalTrajectory,trajectory_catalog,SurfaceGridProjector
from src.native_surface import extract_surface,hottest_area_weights
from src.matched_projection import MatchedHeatOperator

BASE=ROOT/'outputs/by_experiment/56_within_step_integration'
OUT=ROOT/'outputs/by_experiment/58_source_amplitude_shape'
DATA=ROOT.parent/'heat_eq_laser_trajectories'
GRID=[.6,.7,.8,.9,1.,1.1,1.2,1.3]
SHAPES={'endpoint':'source_S0','combined_GL4':'source_S_GL4'}
CASES=['DiagonalScanPath_8','SpiralScanPath_13']


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=False);start=perf_counter();fingerprints={}
    def record(p):fingerprints[str(p.relative_to(ROOT))]=sha(p)
    cfg=json.loads((BASE/'frozen_configuration.json').read_text());o=cfg['inherited_options']
    for p in [BASE/'frozen_configuration.json',BASE/'frozen_sections.csv',Path(__file__),
              ROOT/'src/source_amplitude_diagnostic.py',ROOT/'src/within_step_integration.py',
              ROOT/'src/matched_projection.py',ROOT/'src/thermal_trajectory.py',ROOT/'src/diffusion.py',
              ROOT/'src/thermal_posterior_physics.py',ROOT/'tests/test_source_amplitude_diagnostic.py']:record(p)
    config=dict(experiment=58,driver=str(Path(__file__).relative_to(ROOT)),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        kappa_grid=GRID,lambda_values=[0,1],shapes=SHAPES,
        shape_definition='GL4 is Experiment 56 combined time-interpolated gradient, NOT Experiment 57 constant-current integration',
        calibration_cases=list(DEVELOPMENT_TRAJECTORIES),
        calibration_objective='Mean of three trajectory-specific whole-field native-area MSEs; oracle previous state; separate kappa per shape',
        calibration_selection='Exhaustive specified grid; no initialization or optimizer; ties use first grid entry; no search expansion',
        validation_scope='30 held-out trajectories, ORACLE predecessor only; cached inferred-state assessment is development only',
        regional_optima='Unconstrained and bounded analytic LS, diagnostic only; not used to select shared scale',
        ambient_rule='Existing preparation rule: 2nd percentile of all-time 61x41 projected temperatures; reused dev CSV/cached ambient where available',
        provenance_caution='Simulator truth used in development fit and oracle evaluation; current gradient is simulator-derived; not self-supervised or deployment validation',
        inherited_parameters=o,gp_inference=False,original_gamma_modified=False,
        thresholds_and_source_shapes_fixed=True)
    dump(OUT/'frozen_configuration.json',config)
    sections=pd.read_csv(BASE/'frozen_sections.csv')
    cached={};sweep=[];optima=[];checks=[]
    def regions(c,name=None):
        w=c['area_weights'];peak=int(np.argmax(c['truth']));wp=np.zeros_like(w);wp[peak]=1
        distance=np.linalg.norm(c['points']-c['points'][peak],axis=1)*1000
        result=dict(whole_area=w,hot1pct_area=c['hot_area_weights'],true_peak=wp,
                    shoulder_annulus=w*((distance>=.25)&(distance<=1.)))
        if name in CASES:
            for label,g in sections[sections.trajectory==name].groupby('section',sort=False):
                ids=g[g.shoulder].node.to_numpy(int);weights=np.zeros_like(w);weights[ids]=w[ids]
                result['shoulder_'+('across' if 'Across' in label else 'along')]=weights
        return result
    for name in CASES:
        for level in range(3):
            path=BASE/'by_trajectory'/name/f'level_{level}.npz';record(path)
            with np.load(path) as f:c={k:f[k] for k in f.files}
            cached[name,level]=c
            for state in ['inferred','oracle']:
                base=c[state+'_propagation']
                for shape,key in SHAPES.items():
                    source=c[key]
                    np.testing.assert_array_equal(base+source,c[state+('_S0' if shape=='endpoint' else '_S_GL4')])
                    for region,w in regions(c,name).items():
                        identity=dict(trajectory=name,level=level,ceiling_K=float(c['ceiling']),predecessor=state,shape=shape,region=region)
                        optima.append(dict(**identity,**scale_diagnostic(base,source,c['truth'],w,GRID)))
                        for k in GRID:
                            forecast=base+k*source;peak=int(np.argmax(c['truth']))
                            sweep.append(dict(**identity,kappa=k,**error_metrics(forecast,c['truth'],w),
                                true_peak_signed_error_K=float(forecast[peak]-c['truth'][peak]),
                                true_peak_absolute_error_K=float(abs(forecast[peak]-c['truth'][peak]))))
            checks.append(dict(trajectory=name,level=level,kappa_one_endpoint_and_GL4_exact=True))
    pd.DataFrame(sweep).to_csv(OUT/'development_amplitude_sweep.csv',index=False)
    pd.DataFrame(optima).to_csv(OUT/'regional_optima.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'equality_checks.csv',index=False)
    catalog={r.name:r for r in trajectory_catalog(DATA)}
    assert len(catalog)==33 and all(n in catalog for n in DEVELOPMENT_TRAJECTORIES)
    heldout=sorted(set(catalog)-set(DEVELOPMENT_TRAJECTORIES));assert len(heldout)==30
    calibration_csv=ROOT/'outputs/by_experiment/20_physics_mean_propagation/development_physics_parameters.csv'
    record(calibration_csv);ambient_dev=pd.read_csv(calibration_csv).set_index('trajectory').ambient_temperature_K
    native={};input_rows=[]
    def prepare(name):
        if name in CASES:
            c=cached[name,0]
            np.testing.assert_equal(float(c['ambient']),float(ambient_dev[name]))
            return dict(points=c['points'],area_weights=c['area_weights'],truth=c['truth'],hot_area_weights=c['hot_area_weights'],
                        base=c['oracle_propagation'],endpoint=c['source_S0'],combined_GL4=c['source_S_GL4'],ambient=float(c['ambient']))
        t=ThermalTrajectory(catalog[name].xdmf_path);mesh=extract_surface(t)
        previous=t.read_temperature(-2,surface_only=True);truth=t.read_temperature(-1,surface_only=True)
        gp=t.read_heat_flux_z(-2,surface_only=True);gc=t.read_heat_flux_z(-1,surface_only=True)
        xs=np.linspace(mesh.points[:,0].min(),mesh.points[:,0].max(),int(o['nx']))
        ys=np.linspace(mesh.points[:,1].min(),mesh.points[:,1].max(),int(o['ny']))
        if name in DEVELOPMENT_TRAJECTORIES:ambient=float(ambient_dev[name]);ambient_source='saved development CSV'
        else:
            _,history=t.read_history(surface_only=True)
            projector=SurfaceGridProjector(mesh.points,nx=int(o['nx']),ny=int(o['ny']))
            ambient=float(np.quantile(projector.project(history),.02));del history
            ambient_source='inherited all-time projected-temperature 2% quantile; simulation-derived preparation'
        dt=float(t.times[-1]-t.times[-2]);physics=dict(dt=dt,diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'])
        op=MatchedHeatOperator(mesh,xs,ys)
        source=native_source_contributions(op,gp,gc,threshold=o['source_flux_threshold'],gamma=o['source_coupling'],**physics)
        base=ambient+op.propagate(previous-ambient,'native',floor=True,**physics)
        np.testing.assert_array_equal(base+source['S0'],op.mean(previous,gc,'native',ambient=ambient,
            source_coupling=o['source_coupling'],source_flux_threshold=o['source_flux_threshold'],**physics))
        result=dict(points=mesh.points,area_weights=mesh.weights,truth=truth,
                    hot_area_weights=hottest_area_weights(truth,mesh.weights),base=base,
                    endpoint=source['S0'],combined_GL4=source['S_GL4'],ambient=ambient)
        input_rows.append(dict(trajectory=name,role='development' if name in DEVELOPMENT_TRAJECTORIES else 'heldout',
            ambient_K=ambient,ambient_source=ambient_source,dt_s=dt,native_nodes=len(truth),
            xdmf=str(t.xdmf_path),xdmf_sha256=sha(t.xdmf_path),h5=str(t.h5_path),
            previous_temperature_sha256=digest(previous),truth_sha256=digest(truth),
            previous_gradient_sha256=digest(gp),current_gradient_sha256=digest(gc),area_weights_sha256=digest(mesh.weights)))
        return result
    # Freeze both selected scales BEFORE preparing any held-out native fields.
    for name in DEVELOPMENT_TRAJECTORIES:native[name]=prepare(name)
    selected={};objectives=[];dev_regional=[];shared_optima={}
    for name,c in native.items():
        for shape in SHAPES:
            for region,w in regions(c,name).items():
                dev_regional.append(dict(trajectory=name,shape=shape,region=region,
                    **scale_diagnostic(c['base'],c[shape],c['truth'],w,GRID)))
    pd.DataFrame(dev_regional).to_csv(OUT/'development_oracle_regional_optima.csv',index=False)
    for shape in SHAPES:
        fit_cases=[(native[n]['base'],native[n][shape],native[n]['truth'],native[n]['area_weights']) for n in DEVELOPMENT_TRAJECTORIES]
        k,losses=select_shared_scale(fit_cases,GRID);selected[shape]=k
        pooled=[np.concatenate([c[i] if i!=3 else c[i]/c[i].sum() for c in fit_cases]) for i in range(4)]
        shared_optima[shape]=float(scale_diagnostic(*pooled,GRID)['kappa_unconstrained'])
        objectives.extend(dict(shape=shape,kappa=kappa,development_equal_trajectory_MSE_K2=float(loss)) for kappa,loss in zip(GRID,losses))
    selection=dict(selected_kappa=selected,training_names=list(DEVELOPMENT_TRAJECTORIES),objective=config['calibration_objective'],
        heldout_names=heldout,heldout_data_loaded_before_selection=False,
        effective_gamma={s:float(o['source_coupling']*k) for s,k in selected.items()},
        original_gamma=float(o['source_coupling']),unconstrained_development_optimum_diagnostic=shared_optima,
        boundary_solution={s:k in (GRID[0],GRID[-1]) for s,k in selected.items()})
    dump(OUT/'frozen_selection.json',selection)
    selection_hash=sha(OUT/'frozen_selection.json')
    pd.DataFrame(objectives).to_csv(OUT/'development_selection_objective.csv',index=False)
    print('Frozen development scales:',selected,flush=True)
    arms={'A_endpoint_original':('endpoint',1.),'B_endpoint_calibrated':('endpoint',selected['endpoint']),
          'C_GL4_original':('combined_GL4',1.),'D_GL4_calibrated':('combined_GL4',selected['combined_GL4'])}
    rows=[]
    for name in list(DEVELOPMENT_TRAJECTORIES)+heldout:
        if name not in native:native[name]=prepare(name)
        c=native[name]
        for arm,(shape,k) in arms.items():
            field=c['base']+k*c[shape];peak=int(np.argmax(c['truth']))
            for region,w in regions(c).items():
                rows.append(dict(trajectory=name,family=name.split('Scan')[0],role='development' if name in DEVELOPMENT_TRAJECTORIES else 'heldout',
                    predecessor='oracle',arm=arm,kappa=k,region=region,**error_metrics(field,c['truth'],w),
                    true_peak_signed_error_K=float(field[peak]-c['truth'][peak]),
                    true_peak_absolute_error_K=float(abs(field[peak]-c['truth'][peak]))))
        folder=OUT/'oracle_inputs';folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/f'{name}.npz',**c)
        print(name,'oracle evaluation complete',flush=True)
    assert sha(OUT/'frozen_selection.json')==selection_hash
    result=pd.DataFrame(rows);result.to_csv(OUT/'oracle_33_results.csv',index=False)
    pd.DataFrame(input_rows).to_csv(OUT/'native_input_provenance.csv',index=False)
    held=result[result.role=='heldout'];numeric=['rmse_K','mae_K','bias_K','true_peak_signed_error_K','true_peak_absolute_error_K']
    held.groupby(['arm','region'])[numeric].mean().to_csv(OUT/'heldout30_oracle_summary.csv')
    held.groupby(['family','arm','region'])[numeric].mean().to_csv(OUT/'heldout30_oracle_family_summary.csv')
    paired=[]
    for region,g in held.groupby('region'):
        for left,right in [('A_endpoint_original','B_endpoint_calibrated'),('A_endpoint_original','C_GL4_original'),
                           ('C_GL4_original','D_GL4_calibrated'),('B_endpoint_calibrated','D_GL4_calibrated')]:
            a=g[g.arm==left].set_index('trajectory');b=g[g.arm==right].set_index('trajectory')
            for metric in ['rmse_K','mae_K','true_peak_absolute_error_K']:
                diff=b[metric]-a[metric]
                paired.append(dict(region=region,comparison=right+' minus '+left,metric=metric,
                    mean_change=float(diff.mean()),median_change=float(diff.median()),wins=int((diff<0).sum()),n=len(diff)))
    pd.DataFrame(paired).to_csv(OUT/'heldout30_oracle_paired.csv',index=False)
    # Apply the SAME development-oracle scales to every cached ceiling/state.
    applied=[]
    for (name,level),c in cached.items():
        for state in ['inferred','oracle']:
            for arm,(shape,k) in arms.items():
                field=c[state+'_propagation']+k*c[SHAPES[shape]];peak=int(np.argmax(c['truth']))
                for region,w in regions(c,name).items():
                    applied.append(dict(trajectory=name,level=level,ceiling_K=float(c['ceiling']),predecessor=state,arm=arm,kappa=k,
                        region=region,**error_metrics(field,c['truth'],w),true_peak_signed_error_K=float(field[peak]-c['truth'][peak])))
    pd.DataFrame(applied).to_csv(OUT/'frozen_scale_cached_cases.csv',index=False)
    sw=pd.DataFrame(sweep)
    for name in CASES:
        fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
        for i,shape in enumerate(SHAPES):
            for j,region in enumerate(['hot1pct_area','shoulder_across','whole_area']):
                ax=axes[i,j]
                for level,color in zip(range(3),['#0072B2','#D55E00','#009E73']):
                    g=sw[(sw.trajectory==name)&(sw['shape']==shape)&(sw.predecessor=='inferred')&(sw.region==region)&(sw.level==level)]
                    ax.plot(g.kappa,g.rmse_K,color=color,marker='.',label=f'c={g.ceiling_K.iloc[0]:.2f} K')
                g=sw[(sw.trajectory==name)&(sw['shape']==shape)&(sw.predecessor=='oracle')&(sw.region==region)&(sw.level==0)]
                ax.plot(g.kappa,g.rmse_K,'k--',label='Oracle predecessor')
                ax.axvline(1,color='#aaaaaa',ls=':');ax.axvline(selected[shape],color='black',ls=':',label='Frozen dev scale')
                ax.set_title(shape+': '+region,fontsize=10);ax.set_xlabel('kappa');ax.set_ylabel('RMSE (K)');ax.grid(alpha=.15);ax.legend(fontsize=7)
        fig.suptitle(name+': fixed-shape amplitude diagnostic; no GP inference')
        fig.savefig(OUT/f'amplitude_curves_{name}.png',dpi=150);plt.close(fig)
    for p,h in fingerprints.items():assert sha(ROOT/p)==h,p
    dump(OUT/'source_fingerprints.json',fingerprints)
    dump(OUT/'verification.json',dict(cached_kappa_one_exact=True,development_count=3,heldout_count=30,
        selection_saved_before_heldout_loading=True,selection_hash_unchanged=True,
        original_inputs_unchanged=True,no_previous_or_current_sampling=True,original_gamma_modified=False,
        new_calibration='Only experiment-local scalar kappa per shape, supervised with development simulator truth',
        elapsed_seconds=perf_counter()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)))


if __name__=='__main__':main()
