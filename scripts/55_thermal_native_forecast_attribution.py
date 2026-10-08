"""Cached-state native forecast attribution: no GP inference and no fitting."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.thermal_trajectory import ThermalTrajectory,trajectory_catalog
from src.native_surface import extract_surface,hottest_area_weights
from src.matched_projection import MatchedHeatOperator
from src.forecast_attribution import forecast_components,component_metrics

BASE=ROOT/'outputs/by_experiment/49_matched_native_grid_gp'
OUT=ROOT/'outputs/by_experiment/51_native_forecast_attribution'
DATA=ROOT.parent/'heat_eq_laser_trajectories'
NAMES=['DiagonalScanPath_8','SpiralScanPath_13']


def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def plot_sections(name,sections,arrays,ceiling):
    fig,axes=plt.subplots(4,2,figsize=(14,13),constrained_layout=True,sharex='col')
    tempkeys=['previous_inferred','previous_truth','base_without_source','oracle_without_source','base_forecast','oracle_forecast','truth']
    ids=np.unique(np.concatenate([s.node.to_numpy(int) for s in sections]))
    ymin=min(float(arrays[k][ids].min()) for k in tempkeys)-2
    ymax=max(float(arrays[k][ids].max()) for k in tempkeys)+2
    for col,section in enumerate(sections):
        nodes=section.node.to_numpy(int);x=section.along_mm.to_numpy()
        panels=[
            [('previous_truth','Previous truth','black','-'),('previous_inferred','Previous inferred','#0072B2','--')],
            [('truth','Current truth','black','-'),('base_without_source','Propagated inferred + ambient','#0072B2','-'),('oracle_without_source','Propagated oracle + ambient','#009E73','--')],
            [('truth','Current truth','black','-'),('base_forecast','Complete inferred-state forecast','#0072B2','-'),('oracle_forecast','Complete oracle-state forecast','#009E73','--')],
            [('source','Added numerical source s','#D55E00','-'),('base_needed','Needed with inferred state','#0072B2','-'),('oracle_needed','Needed with oracle state','#009E73','--')]]
        titles=['1. Is the shoulder missing in the predecessor?', '2. Propagation only: no source added',
                '3. Complete forecast: same source in both', '4. Added source versus residual needed to match truth']
        for row,series in enumerate(panels):
            ax=axes[row,col]
            for key,label,color,ls in series:
                ax.plot(x,arrays[key][nodes],label=label,color=color,ls=ls,lw=1.6)
            if row<3:
                ax.set_ylim(ymin,ymax);ax.axhline(ceiling,color='#999999',ls=':',lw=.8,label='Camera ceiling')
                ax.set_ylabel('Temperature (K)')
            else:
                ax.axhline(0,color='black',lw=.6);ax.set_ylabel('Temperature contribution (K)')
            ax.set_title(titles[row],fontsize=11);ax.grid(alpha=.15);ax.legend(fontsize=8)
            if row==3:ax.set_xlabel('Distance from current true peak (mm)')
        axes[0,col].set_title(titles[0]+'\n'+section.section.iloc[0],fontsize=11)
    fig.suptitle(name+': exact cached native sections; non-advective numerical attribution\nNeeded residual is not true heating. Curves connect saved native nodes; no new inference.',fontsize=12)
    fig.savefig(OUT/f'component_sections_{name}.png',dpi=160);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    o=pd.read_csv(BASE/'frozen_configuration.csv').iloc[0].to_dict()
    baseline=pd.read_csv(BASE/'matching_checks.csv')
    catalog={r.name:r for r in trajectory_catalog(DATA)}
    allmetrics=[];budgets=[];checks=[];pointrows=[];sectionrows=[];fingerprints={}
    start=perf_counter()
    for name in NAMES:
        tick=perf_counter();t=ThermalTrajectory(catalog[name].xdmf_path);mesh=extract_surface(t)
        path=BASE/'by_trajectory'/name/'matched.npz'
        with np.load(path) as f:
            c={k:f[k] for k in ['points','triangles','area_weights','truth','previous_mean','previous_camera','camera',
                               'ceiling','ambient','observation_ids','native_physics_mean']}
        fingerprints[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        ref=baseline.query('trajectory==@name and representation=="native" and model=="B"').iloc[0]
        for values,key in [(c['previous_mean'],'previous_mean_sha256'),(c['previous_camera'],'previous_camera_sha256'),
                           (c['camera'],'current_camera_sha256'),(c['observation_ids'],'observation_ids_sha256')]:
            assert digest(values)==ref[key],key
        np.testing.assert_array_equal(mesh.points,c['points']);np.testing.assert_array_equal(mesh.weights,c['area_weights'])
        previous_truth=t.read_temperature(-2,surface_only=True);truth=t.read_temperature(-1,surface_only=True)
        np.testing.assert_array_equal(truth,c['truth'])
        flux=t.read_heat_flux_z(-1,surface_only=True)
        xs=np.linspace(mesh.points[:,0].min(),mesh.points[:,0].max(),int(o['nx']))
        ys=np.linspace(mesh.points[:,1].min(),mesh.points[:,1].max(),int(o['ny']))
        op=MatchedHeatOperator(mesh,xs,ys)
        physics=dict(dt=float(t.times[-1]-t.times[-2]),diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'])
        args=dict(ambient=float(c['ambient']),source_coupling=o['source_coupling'],source_flux_threshold=o['source_flux_threshold'],**physics)
        a=forecast_components(op,c['previous_mean'],previous_truth,truth,flux,**args)
        np.testing.assert_array_equal(a['base_forecast'],c['native_physics_mean'])
        weights=mesh.weights;hot=hottest_area_weights(truth,weights)
        assert digest(hot)==ref.hot_area_sha256
        regions=dict(whole_area=weights,current_hot1pct_area=hot,previous_hot1pct_area=hottest_area_weights(previous_truth,weights),
                     observed_censored=weights*(c['camera']==c['ceiling']),source_active=weights*(flux>=o['source_flux_threshold']))
        for region,w in regions.items():
            rows,budget=component_metrics(a,w)
            allmetrics.extend(dict(trajectory=name,region=region,**row) for row in rows)
            budgets.append(dict(trajectory=name,region=region,area_fraction=float(w.sum()/weights.sum()),**budget))
        sections_path=BASE/'native_diagnostics'/f'cross_sections_{name}.csv'
        sections=pd.read_csv(sections_path);fixed=[]
        for section,group in sections.groupby('section',sort=False):
            b=group[group.model=='native_B'];other=group[group.model=='native_C']
            np.testing.assert_array_equal(b.node.to_numpy(),other.node.to_numpy())
            nodes=b.node.to_numpy(int)
            np.testing.assert_allclose(mesh.points[nodes]*1000,b[['x_mm','y_mm']].to_numpy(),rtol=0,atol=1e-12)
            np.testing.assert_allclose(a['base_forecast'][nodes],b.physics_K.to_numpy(),rtol=0,atol=1e-12)
            fixed.append(b)
            for idx,row in b.iterrows():
                node=int(row.node)
                sectionrows.append(dict(trajectory=name,section=section,node=node,along_mm=row.along_mm,
                    normal_offset_mm=row.normal_offset_mm,**{k:v[node] for k,v in a.items()}))
        plot_sections(name,fixed,a,float(c['ceiling']))
        for label,node in [('true_current_peak',int(np.argmax(truth))),('baseline_forecast_peak',int(np.argmax(a['base_forecast']))),
                           ('oracle_forecast_peak',int(np.argmax(a['oracle_forecast']))),('numerical_source_peak',int(np.argmax(a['source'])))]:
            pointrows.append(dict(trajectory=name,location=label,node=node,x_mm=mesh.points[node,0]*1000,
                                 y_mm=mesh.points[node,1]*1000,**{k:v[node] for k,v in a.items()}))
        checks.append(dict(trajectory=name,cached_forecast_max_abs=float(abs(a['base_forecast']-c['native_physics_mean']).max()),
            source_budget_max_abs=float(abs(a['base_error']-(a['source']-a['base_needed'])).max()),
            oracle_budget_max_abs=float(abs(a['oracle_error']-(a['source']-a['oracle_needed'])).max()),
            state_physics_identity_max_abs=float(abs(a['base_error']-a['propagated_state_error']-a['oracle_error']).max()),
            previous_mean_sha256=digest(c['previous_mean']),current_camera_sha256=digest(c['camera']),
            source_sha256=digest(a['source']),hot_area_sha256=digest(hot),
            seconds=perf_counter()-tick,**args))
        dest=OUT/'by_trajectory'/name;dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'components.npz',**a,points=mesh.points,triangles=mesh.triangles,weights=weights,
                            hot_area_weights=hot,flux=flux,ceiling=c['ceiling'],ambient=c['ambient'])
        fingerprints[str(sections_path.relative_to(ROOT))]=hashlib.sha256(sections_path.read_bytes()).hexdigest()
        print(name,'cached forecast reproduced exactly',flush=True)
    for file,rows in [('region_metrics',allmetrics),('region_budgets',budgets),('point_components',pointrows),
                      ('section_components',sectionrows),('equality_checks',checks)]:
        pd.DataFrame(rows).to_csv(OUT/(file+'.csv'),index=False)
    for p in [Path(__file__),ROOT/'src/forecast_attribution.py',ROOT/'src/matched_projection.py',BASE/'frozen_configuration.csv']:
        fingerprints[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'source_fingerprints.json').write_text(json.dumps(fingerprints,indent=2))
    (OUT/'verification.json').write_text(json.dumps(dict(paths=NAMES,gp_inference=False,parameter_fitting=False,
        cached_forecast_exact=True,source_identical_for_oracle=True,cross_section_node_ids_reused=True,
        truth_usage='oracle/evaluation only; no inference or input selection changed',
        operator='unchanged native-input map: P1 lift, floor, Gaussian diffusion/cooling, zero extension, bilinear node evaluation',
        runtime_seconds=perf_counter()-start),indent=2))


if __name__=='__main__':main()
