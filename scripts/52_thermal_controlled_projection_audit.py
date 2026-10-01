"""Projection-only oracle audit on three development paths. No GP inference."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.tri import Triangulation, LinearTriInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.native_surface import extract_surface, hottest_area_weights
from src.projection_audit import (area_weights, lift_grid, controlled_forecasts,
                                  error_summary, bridge_components)
from src.thermal_trajectory import ThermalTrajectory, SurfaceGridProjector
from src.thermal_posterior_physics import (DEVELOPMENT_TRAJECTORIES, prepare_trajectory,
                                          trajectory_path, posterior_physics_means)

BASE = ROOT / 'outputs/by_experiment/43_matched_ceiling_model_gap'
ORACLE = ROOT / 'outputs/by_experiment/46_oracle_forecast_decomposition'
OUT = ROOT / 'outputs/by_experiment/48_controlled_projection_audit'
DATA = ROOT.parent / 'heat_eq_laser_trajectories'


def native_evaluate(mesh, fields, xs, ys):
    tri = Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)
    finder = tri.get_trifinder()
    x, y = np.meshgrid(xs, ys)
    result = {}
    for key, field in fields.items():
        interpolated = LinearTriInterpolator(tri, field, trifinder=finder)(x, y)
        assert not np.ma.getmaskarray(interpolated).any()
        result[key] = np.asarray(interpolated)
    return result


def hot_weights(field, weights):
    return hottest_area_weights(field.ravel(), weights.ravel()).reshape(field.shape)


def centroid(weights, xs, ys):
    x, y = np.meshgrid(xs, ys)
    return np.array([np.sum(weights*x), np.sum(weights*y)]) / weights.sum()


def spatial_summary(name, mesh, prepared, native, projected, on_native, on_projected, xs, ys):
    rows = []
    w = area_weights(xs, ys)
    gridw = area_weights(prepared.xs, prepared.ys)
    gx, gy = np.meshgrid(prepared.xs, prepared.ys)
    gridxy = np.column_stack([gx.ravel(), gy.ravel()])
    for field, values in native.items():
        hn, hp = hot_weights(on_native[field], w), hot_weights(on_projected[field], w)
        cn, cp = centroid(hn, xs, ys), centroid(hp, xs, ys)
        pn = mesh.points[int(np.argmax(values))]
        pp = gridxy[int(np.argmax(projected[field]))]
        integral_n = float(np.sum(mesh.weights*values))
        integral_p = float(np.sum(gridw*projected[field]))
        rows.append(dict(trajectory=name,field=field,
            native_peak=float(values.max()),projected_peak=float(projected[field].max()),
            peak_change=float(projected[field].max()-values.max()),
            native_peak_x_mm=pn[0]*1000,native_peak_y_mm=pn[1]*1000,
            projected_peak_x_mm=pp[0]*1000,projected_peak_y_mm=pp[1]*1000,
            peak_location_shift_mm=float(np.linalg.norm(pp-pn)*1000),
            native_hot_centroid_x_mm=cn[0]*1000,native_hot_centroid_y_mm=cn[1]*1000,
            projected_hot_centroid_x_mm=cp[0]*1000,projected_hot_centroid_y_mm=cp[1]*1000,
            hot_centroid_shift_mm=float(np.linalg.norm(cp-cn)*1000),
            hot_area_overlap_iou=float(np.minimum(hn,hp).sum()/np.maximum(hn,hp).sum()),
            native_area_integral=integral_n,projected_area_integral=integral_p,
            area_mean_change=(integral_p-integral_n)/w.sum(),
            area_weighted_projection_rmse=float(np.sqrt(np.sum(w*(on_projected[field]-on_native[field])**2)/w.sum()))))
    return rows


def figures(name, arrays, xs, ys, ambient):
    extent = np.array([xs[0],xs[-1],ys[0],ys[-1]])*1000
    truth = arrays['native_current']
    scenarios = ['native_inputs','project_both_inputs','legacy_lifted']
    titles = ['Native inputs / fine operator','Projected inputs / SAME fine operator','Legacy coarse forecast / lifted']
    fields = [arrays[k] for k in scenarios]
    norm = PowerNorm(.35,vmin=ambient,vmax=max(truth.max(),*(f.max() for f in fields)))
    errors = [abs(f-truth) for f in fields]
    enorm = PowerNorm(.5,vmin=0,vmax=max(e.max() for e in errors))
    fig,axes = plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for j in range(3):
        im = axes[0,j].imshow(fields[j],origin='lower',extent=extent,norm=norm,cmap='inferno')
        axes[0,j].set_title(titles[j],fontsize=10)
        err = axes[1,j].imshow(errors[j],origin='lower',extent=extent,norm=enorm,cmap='magma')
        axes[1,j].set_title('Absolute error vs SAME native current field',fontsize=10)
    fig.colorbar(im,ax=list(axes[0]),label='Forecast (K); shared power scale',shrink=.8)
    fig.colorbar(err,ax=list(axes[1]),label='Absolute error (K); shared power scale',shrink=.8)
    for ax in axes.flat:
        ax.set(xlabel='x (mm)',ylabel='y (mm)')
    fig.suptitle(name+': deterministic oracle forecasts, no GP\nOnly the first two columns isolate input representation; third also changes discretization',fontsize=12)
    fig.savefig(OUT/f'forecast_error_maps_{name}.png',dpi=160)
    # A second view exposes localized peak errors without changing color limits.
    peak_y,peak_x=np.unravel_index(np.argmax(truth),truth.shape)
    for ax in axes.flat:
        ax.set_xlim(xs[peak_x]*1000-1.25,xs[peak_x]*1000+1.25)
        ax.set_ylim(ys[peak_y]*1000-1.25,ys[peak_y]*1000+1.25)
    fig.suptitle(name+': same forecasts and color scales, native-peak zoom\nBoth first columns use the same fine operator and reference; no GP inference',fontsize=12)
    fig.savefig(OUT/f'forecast_error_zoom_{name}.png',dpi=160)
    plt.close(fig)

    components = ['previous_projection_effect','source_projection_effect','input_projection','reference_projection','coarse_discretization']
    fig,axes = plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    im = axes[0,0].imshow(truth,origin='lower',extent=extent,
        norm=PowerNorm(.35,vmin=ambient,vmax=truth.max()),cmap='inferno')
    axes[0,0].set_title('Native current reference',fontsize=10)
    fig.colorbar(im,ax=axes[0,0],label='K',shrink=.8)
    limit=max(float(abs(arrays[k]).max()) for k in components)
    for ax,key in zip(list(axes.flat)[1:],components):
        im=ax.imshow(arrays[key],origin='lower',extent=extent,vmin=-limit,vmax=limit,cmap='RdBu_r')
        ax.set_title('Coarse-pipeline bridge' if key=='coarse_discretization' else key.replace('_',' '),fontsize=10)
    fig.colorbar(im,ax=list(axes.flat)[1:],label='Signed contribution (K)',shrink=.8)
    for ax in axes.flat:
        ax.set(xlabel='x (mm)',ylabel='y (mm)')
    fig.suptitle(name+': exact bias bridge; contributions add pixelwise',fontsize=12)
    fig.savefig(OUT/f'projection_contributions_{name}.png',dpi=160)
    plt.close(fig)


def main():
    start=perf_counter()
    OUT.mkdir(parents=True,exist_ok=True)
    for filename,expected in json.loads((ORACLE/'source_fingerprints.json').read_text()).items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==expected,filename
    config=pd.read_csv(BASE/'fixed_configuration.csv').iloc[0].to_dict()
    pd.DataFrame([config]).to_csv(OUT/'frozen_configuration.csv',index=False)
    results=[]; spatial=[]; bridges=[]; refinement=[]; checks=[]
    for name in DEVELOPMENT_TRAJECTORIES:
        t=ThermalTrajectory(trajectory_path(DATA,name)); mesh=extract_surface(t)
        p,_=prepare_trajectory(DATA,name,nx=int(config['nx']),ny=int(config['ny']),heat_flux_cutoff=config['heat_flux_cutoff'])
        n=len(p.times)-1; dt=float(p.times[n]-p.times[n-1])
        raw={key:t.read_field(field,index,surface_only=True) for key,field,index in
             [('previous','Temperature',-2),('current','Temperature',-1),
              ('previous_flux','HeatFluxZ',-2),('flux','HeatFluxZ',-1)]}
        projector=SurfaceGridProjector(mesh.points,nx=len(p.xs),ny=len(p.ys))
        projected={key:projector.project(v) for key,v in raw.items()}
        np.testing.assert_array_equal(projected['previous'],p.history[-2])
        np.testing.assert_array_equal(projected['current'],p.history[-1])
        np.testing.assert_array_equal(projected['flux'],p.heat_flux[-1])
        legacy=posterior_physics_means(p,p.history[-2],previous_index=n-1,current_index=n,
            diffusivity=config['diffusivity'],cooling_rate=config['cooling_rate'],
            source_coupling=config['source_coupling'],source_flux_threshold=config['source_flux_threshold'])[0]
        cache=ORACLE/'by_trajectory'/name/'level_0.npz'
        with np.load(cache) as old:
            np.testing.assert_array_equal(legacy.ravel(),old['oracle_forecast'].ravel())
            legacy_top25_bias=float(np.mean((legacy.ravel()-p.truth.ravel())[old['mask_current_top25']]))
        # Common evaluation locations remain fixed across numerical refinements.
        ex=np.linspace(p.xs[0],p.xs[-1],601); ey=np.linspace(p.ys[0],p.ys[-1],401)
        w=area_weights(ex,ey)
        runs=[]; peaks=[]
        for factor in [1,2,4]:
            xs=np.linspace(ex[0],ex[-1],600*factor+1)
            ys=np.linspace(ey[0],ey[-1],400*factor+1)
            native=native_evaluate(mesh,raw,xs,ys)
            lifted={key:lift_grid(v,p.xs,p.ys,xs,ys) for key,v in projected.items()}
            f=controlled_forecasts(native['previous'],native['flux'],lifted['previous'],lifted['flux'],
                xs,ys,p.ambient,dt,config)
            peaks.append({key:float(v.max()) for key,v in f.items()})
            runs.append({key:v[::factor,::factor].copy() for key,v in f.items()})
            if factor==1:
                evaluation_native=native; evaluation_projected=lifted
        hot=hot_weights(evaluation_native['current'],w)
        masks={'whole_area':w,'fixed_native_hot1pct_area':hot}
        for fine in [1,2]:
            for key in runs[0]:
                for region,weights in masks.items():
                    refinement.append(dict(trajectory=name,quantity=key,region=region,
                        coarse_dx_um=25/2**(fine-1),fine_dx_um=25/2**fine,
                        full_grid_maximum_change_K=peaks[fine][key]-peaks[fine-1][key],
                        **error_summary(runs[fine][key]-runs[fine-1][key],weights)))
        f=runs[-1]; truth=evaluation_native['current']; ptruth=evaluation_projected['current']
        legacy_lifted=lift_grid(legacy,p.xs,p.ys,ex,ey)
        components=bridge_components(f['native_inputs'],f['project_both_inputs'],truth,ptruth,legacy_lifted)
        identity=components['legacy_error']-sum(components[k] for k in components if k!='legacy_error')
        input_identity=components['input_projection']-f['previous_projection_effect']-f['source_projection_effect']
        assert abs(identity).max()<1e-11 and abs(input_identity).max()<1e-11
        for key in ['native_inputs','project_previous_only','project_source_only','project_both_inputs']:
            for region,weights in masks.items():
                results.append(dict(trajectory=name,forecast=key,reference='native',region=region,
                    forecast_peak_K=peaks[-1][key],peak_error_K=peaks[-1][key]-float(raw['current'].max()),
                    **error_summary(f[key]-truth,weights)))
        for key,forecast,reference,ref in [('project_both_vs_projected',f['project_both_inputs'],'projected',ptruth),
                                         ('legacy_vs_native',legacy_lifted,'native',truth),
                                         ('legacy_vs_projected',legacy_lifted,'projected',ptruth)]:
            for region,weights in masks.items():
                forecast_peak=peaks[-1]['project_both_inputs'] if key=='project_both_vs_projected' else float(legacy.max())
                reference_peak=float(raw['current'].max()) if reference=='native' else float(projected['current'].max())
                results.append(dict(trajectory=name,forecast=key,reference=reference,region=region,
                    forecast_peak_K=forecast_peak,peak_error_K=forecast_peak-reference_peak,
                    **error_summary(forecast-ref,weights)))
        for key,error in {**components,'previous_projection_effect':f['previous_projection_effect'],
                          'source_projection_effect':f['source_projection_effect']}.items():
            for region,weights in masks.items():
                bridges.append(dict(trajectory=name,component=key,region=region,**error_summary(error,weights)))
        spatial.extend(spatial_summary(name,mesh,p,raw,projected,evaluation_native,evaluation_projected,ex,ey))
        arrays={**f,**components,'native_current':truth,'projected_current':ptruth,'legacy_lifted':legacy_lifted}
        assert all(np.isfinite(v).all() for v in arrays.values())
        folder=OUT/'by_trajectory'/name; folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'projection_audit.npz',xs=ex,ys=ey,weights=w,hot_area_weights=hot,**arrays)
        figures(name,arrays,ex,ey,p.ambient)
        checks.append(dict(trajectory=name,legacy_array_exact=True,projection_arrays_exact=True,
            legacy_top25_oracle_bias_K=legacy_top25_bias,
            legacy_oracle_sha256=hashlib.sha256(legacy.ravel().tobytes()).hexdigest(),
            oracle_cache_file_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
            bridge_max_abs_K=float(abs(identity).max()),input_split_max_abs_K=float(abs(input_identity).max())))
        print(name,'projection and oracle baseline exact; decomposition verified',flush=True)
    for filename,rows in [('field_comparison',spatial),('forecast_metrics',results),('bias_decomposition',bridges),
                          ('resolution_check',refinement),('baseline_checks',checks)]:
        pd.DataFrame(rows).to_csv(OUT/f'{filename}.csv',index=False)
    files=[Path(__file__),ROOT/'src/projection_audit.py',ROOT/'src/native_surface.py',
           ROOT/'src/thermal_posterior_physics.py',ROOT/'src/thermal_trajectory.py',BASE/'fixed_configuration.csv']
    (OUT/'source_fingerprints.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
    (OUT/'verification.json').write_text(json.dumps(dict(development_trajectories=DEVELOPMENT_TRAJECTORIES,
        no_gp_inference=True,no_refitting=True,no_native_fem_propagation=True,
        numerical_grids=[[601,401],[1201,801],[2401,1601]],evaluation_grid=[601,401],
        comparison='same fine-grid operator, frozen parameters, fixed native hot-area mask',
        runtime_seconds=perf_counter()-start),indent=2))


if __name__=='__main__':
    main()
