"""Native surface pilot and area-weighted projection audit; frozen parameters."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import resource
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import PowerNorm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.native_surface import extract_surface,propagate,physics_forecast,hottest_area_weights,quadrature
from src.thermal_trajectory import ThermalTrajectory,trajectory_catalog
from src.thermal_posterior_physics import DEVELOPMENT_TRAJECTORIES,prepare_trajectory,trajectory_path
from src.censored_gp import RBFConfig,rbf_covariance,sample_multiple_chains
from src.mcmc_diagnostics import chain_diagnostics
from src.pseudo_censoring_runner import _current_prediction
from src.stochastic_heat_gp import StochasticHeatConfig,finite_step_innovation_covariance
from src.metrics import empirical_crps

BASE=ROOT/'outputs/by_experiment/43_matched_ceiling_model_gap'
OUT=ROOT/'outputs/by_experiment/47_native_mesh_pilot'
DATA=ROOT.parent/'heat_eq_laser_trajectories'


def grid_weights(xs,ys):
    wx=np.ones(len(xs));wy=np.ones(len(ys));wx[[0,-1]]=.5;wy[[0,-1]]=.5
    return np.outer(wy,wx)*np.mean(np.diff(xs))*np.mean(np.diff(ys))


def on_points(field,p,points):
    return RegularGridInterpolator((p.ys,p.xs),field,bounds_error=True)(points[:,[1,0]])


def compare_fields(t,p,mesh):
    points,weights,bary=quadrature(mesh,refinement=2)
    check_points,check_weights,check_bary=quadrature(mesh,refinement=1)
    area=weights.sum(); gw=grid_weights(p.xs,p.ys)
    rows=[]
    for index in [-2,-1]:
        for field in ['Temperature','HeatFluxZ']:
            native=t.read_field(field,index,surface_only=True)
            projected=p.history[index] if field=='Temperature' else p.heat_flux[index]
            nq=np.einsum('qv,tv->tq',bary,native[mesh.triangles]).ravel()
            pq=on_points(projected,p,points); err=pq-nq
            hot=hottest_area_weights(nq,weights)
            native_integral=float(np.sum(mesh.weights*native)); projected_integral=float(np.sum(gw*projected))
            check_native=np.einsum('qv,tv->tq',check_bary,native[mesh.triangles]).ravel()
            check_error=on_points(projected,p,check_points)-check_native
            rmse=float(np.sqrt(np.sum(weights*err**2)/area))
            coarse_rmse=float(np.sqrt(np.sum(check_weights*check_error**2)/area))
            assert np.isfinite(err).all()
            rows.append(dict(trajectory=t.name,frame='previous' if index==-2 else 'current',field=field,
                time_s=float(t.times[index]),native_nodes=len(native),grid_nodes=projected.size,area_m2=area,
                native_peak=float(native.max()),projected_peak=float(projected.max()),
                peak_change=float(projected.max()-native.max()),
                native_area_mean=native_integral/area,projected_area_mean=projected_integral/area,
                native_integral=native_integral,projected_integral=projected_integral,
                projection_bias=float(np.sum(weights*err)/area),projection_rmse=rmse,
                projection_mae=float(np.sum(weights*abs(err))/area),
                native_hot1pct_projection_bias=float(np.sum(hot*err)/hot.sum()),
                native_hot1pct_projection_rmse=float(np.sqrt(np.sum(hot*err**2)/hot.sum())),
                quadrature_relative_rmse_change=abs(rmse-coarse_rmse)/max(rmse,1e-15),
                quadrature_points=len(points)))
    # Display interpolated comparisons only; quantitative integration is separate.
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    tri=Triangulation(mesh.points[:,0]*1000,mesh.points[:,1]*1000,mesh.triangles)
    for row,field in enumerate(['Temperature','HeatFluxZ']):
        native=t.read_field(field,-1,surface_only=True)
        projected=p.history[-1] if row==0 else p.heat_flux[-1]
        back=on_points(projected,p,mesh.points); error=back-native
        scale=1 if row==0 else 1e-3; unit='K' if row==0 else '1000 stored flux units'
        lo=min(native.min(),projected.min())*scale;hi=max(native.max(),projected.max())*scale
        norm=PowerNorm(gamma=.35 if row==0 else 1.,vmin=lo,vmax=hi)
        axes[row,0].tripcolor(tri,native*scale,shading='gouraud',norm=norm,cmap='inferno')
        im=axes[row,1].imshow(projected*scale,origin='lower',extent=np.array([p.xs[0],p.xs[-1],p.ys[0],p.ys[-1]])*1000,norm=norm,cmap='inferno')
        limit=max(abs(error).max()*scale,1e-10)
        diff=axes[row,2].tripcolor(tri,error*scale,shading='gouraud',vmin=-limit,vmax=limit,cmap='RdBu_r')
        fig.colorbar(im,ax=axes[row,:2],label=unit,shrink=.75)
        fig.colorbar(diff,ax=axes[row,2],label='Projected - native ('+unit+')',shrink=.75)
        for ax,title in zip(axes[row],['Native surface','Projected 61 x 41','Projected - native at native nodes']):
            ax.set(xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
            ax.set_title(field+'\n'+title,fontsize=10)
    fig.suptitle(t.name+': same current field, no GP inference\nTemperature: shared full-range power scale (gamma=0.35) to reveal the tail',fontsize=12)
    fig.savefig(OUT/f'field_comparison_{t.name}.png',dpi=160);plt.close(fig)
    return rows


def observations(points,values,mask,ceiling,prediction_points):
    ids=np.flatnonzero(mask);sat=values>=ceiling
    return dict(x_obs=points[ids],y_obs=values[ids],sat_mask=sat[ids],threshold=ceiling,
        x_pred=prediction_points,saturated_full=sat),ids


def previous_native(points,values,fixed,ceiling,ambient,length,o,seed):
    sat=values>=ceiling
    obs,_=observations(points,values,fixed|sat,ceiling,points[sat])
    config=RBFConfig(mean_temp=ambient,signal_sd=o['signal_sd'],lengthscale=length,noise_sd=o['previous_noise_sd'])
    last=None
    for retry in [False,True]:
        prefix='previous_retry_' if retry else 'previous_'
        samples=int(o[prefix+'samples_per_chain']);chains=int(o['previous_chains'])
        pred=sample_multiple_chains(obs,config,n_chains=chains,samples_per_chain=samples,
            burn_in=int(o[prefix+'burn_in']),thin=int(o[prefix+'thin']),seed=seed,truncated_sampler=o['truncated_sampler'])
        last=chain_diagnostics(pred[4].reshape(chains,samples,int(sat.sum())))
        if last['rhat_max']<=o['maximum_rhat']:
            draws=np.repeat(values[None],chains*samples,axis=0)
            draws[:,sat]=pred[4]
            return draws.mean(0),draws,{**last,'retried':retry,'retained_draws':len(draws)}
    raise RuntimeError('Native previous posterior did not pass frozen convergence gate: '+str(last))


def score(pred,truth,weights,ambient,saturated):
    mu,sd,lo,hi,draws=pred;err=mu-truth;crps=empirical_crps(draws,truth)
    rows=[]
    for region,w in [('whole_surface',weights),('native_area_hot1pct',hottest_area_weights(truth,weights)),('observed_censored',weights*saturated)]:
        total=w.sum()
        rows.append(dict(region=region,area_m2=float(total),effective_nodes=int(np.count_nonzero(w)),
            bias_K=float(np.sum(w*err)/total),rmse_K=float(np.sqrt(np.sum(w*err**2)/total)),mae_K=float(np.sum(w*abs(err))/total),
            crps_K=float(np.sum(w*crps)/total),coverage95=float(np.sum(w*((lo<=truth)&(truth<=hi)))/total),width95_K=float(np.sum(w*(hi-lo))/total),
            excess_relative_L2=float(np.sqrt(np.sum(w*err**2)/np.sum(w*(truth-ambient)**2))),
            peak_absolute_error_K=float(abs(mu.max()-truth.max()))))
    return rows


def reconstruction_figure(mesh,truth,camera,previous,mean,arrays,ambient):
    tri=Triangulation(mesh.points[:,0]*1000,mesh.points[:,1]*1000,mesh.triangles)
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    fields=[truth,camera,previous,mean,arrays['native_B_mean'],arrays['native_C_mean']]
    titles=['Native truth','Censored camera','Previous hybrid posterior estimate','Native FEM physics forecast',
            'Native B: physics + RBF','Native C: sequential covariance']
    norm=PowerNorm(.35,vmin=ambient,vmax=float(truth.max()))
    for ax,field,title in zip(axes.flat,fields,titles):
        im=ax.tripcolor(tri,field,shading='gouraud',norm=norm,cmap='inferno')
        ax.set_title(title,fontsize=10)
    fig.colorbar(im,ax=list(axes[0]),label='Temperature (K); all six temperature panels',shrink=.75,extend='both')
    errors=[abs(arrays[key+'_mean']-truth) for key in ['native_B','native_C']]
    limit=max(float(e.max()) for e in errors)
    for ax,error,title in zip(list(axes.flat)[6:],errors,['Native B: absolute error','Native C: absolute error']):
        im=ax.tripcolor(tri,error,shading='gouraud',vmin=0,vmax=limit,cmap='magma')
        ax.set_title(title,fontsize=10)
    fig.colorbar(im,ax=list(axes[1]),label='Absolute error (K); both error panels',shrink=.75)
    for ax in axes.flat:
        ax.set(aspect='equal',xlabel='x (mm)',ylabel='y (mm)')
    fig.suptitle('DiagonalScanPath_8: native-surface feasibility pilot\nNo temperature/source projection for inference; not a controlled native-versus-grid comparison',fontsize=12)
    fig.savefig(OUT/'native_reconstruction_DiagonalScanPath_8.png',dpi=160)
    plt.close(fig)


def native_pilot(t,p,mesh,o):
    start=perf_counter();n=t.n_times-1;xy=mesh.points;N=len(xy)
    dt=float(t.times[n]-t.times[n-1]);truth=t.read_temperature(n,surface_only=True)
    prevtruth=t.read_temperature(n-1,surface_only=True);flux=t.read_heat_flux_z(n,surface_only=True)
    ref=pd.read_csv(BASE/'results.csv');c=float(ref[(ref.trajectory==t.name)&(ref.level_index==0)].ceiling_K.iloc[0])
    # Geometry-only mapping to approximate the same numerical sensor locations.
    X,Y=np.meshgrid(p.xs[::5],p.ys[::5]);targets=np.column_stack([X.ravel(),Y.ravel()])
    dist,ids=cKDTree(xy).query(targets);fixed=np.zeros(N,bool);fixed[np.unique(ids)]=True
    index=7 # DiagonalScanPath_8 has the canonical trajectory index 7.
    rng=np.random.default_rng(int(o['seed'])+10000*index)
    previous_camera=np.minimum(prevtruth+rng.normal(0,o['measurement_noise_sd'],N),c)
    current_camera=np.minimum(truth+rng.normal(0,o['measurement_noise_sd'],N),c)
    prevpoints=np.column_stack([xy,np.full(N,t.times[n-1])]);points=np.column_stack([xy,np.full(N,t.times[n])])
    length=p.source_lengthscale*o['length_multiplier']
    tick=perf_counter()
    previous,draws,pc=previous_native(prevpoints,previous_camera,fixed,c,p.ambient,length,o,int(o['seed'])+50000*index)
    previous_seconds=perf_counter()-tick;tick=perf_counter()
    mean=physics_forecast(mesh,previous,flux,ambient=p.ambient,dt=dt,diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],
        source_coupling=o['source_coupling'],source_flux_threshold=o['source_flux_threshold'])
    oracle=physics_forecast(mesh,prevtruth,flux,ambient=p.ambient,dt=dt,diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],
        source_coupling=o['source_coupling'],source_flux_threshold=o['source_flux_threshold'])
    mean_seconds=perf_counter()-tick
    current_sat=current_camera==c
    obs,obsids=observations(points,current_camera,fixed|current_sat,c,points)
    cfg=RBFConfig(mean_temp=p.ambient,signal_sd=o['signal_sd'],lengthscale=length,noise_sd=o['current_noise_sd'])
    heat=StochasticHeatConfig(signal_sd=o['signal_sd'],forcing_lengthscale=length*o['forcing_length_multiplier'],diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],quadrature_order=int(o['quadrature_order']))
    allrows=[];times=[];arrays={}
    for model in ['native_B','native_C']:
        tick=perf_counter();obsxy=points[obsids]
        if model=='native_B':
            Koo=rbf_covariance(obsxy,obsxy,cfg);Kpo=rbf_covariance(points,obsxy,cfg);diag=np.full(N,o['signal_sd']**2)
        else:
            propagated=propagate(mesh,draws-previous[None],dt=dt,diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'])
            features=propagated.T/np.sqrt(len(draws)-1)
            fobs=features[obsids]
            Koo=finite_step_innovation_covariance(obsxy,obsxy,heat,dt)+fobs@fobs.T
            Kpo=finite_step_innovation_covariance(points,obsxy,heat,dt)+features@fobs.T
            # Independently check a block without BLAS and reject non-finite output.
            check_pred=np.argsort(np.sum(features**2,axis=1))[-12:]
            check_obs=np.argsort(np.sum(fobs**2,axis=1))[-12:]
            reference=np.einsum('ik,jk->ij',features[check_pred],fobs[check_obs],optimize=False)
            reference+=finite_step_innovation_covariance(points[check_pred],obsxy[check_obs],heat,dt)
            np.testing.assert_allclose(Kpo[np.ix_(check_pred,check_obs)],reference,rtol=1e-12,atol=1e-12)
            variance=float(finite_step_innovation_covariance(points[:1],points[:1],heat,dt)[0,0])
            diag=variance+np.sum(features**2,axis=1)
            del propagated,features,fobs
        blocks=dict(prediction_mean=mean,observation_mean=mean[obsids],observed_covariance=Koo,pred_observed_covariance=Kpo,prediction_variance=diag)
        assert all(np.isfinite(v).all() for v in blocks.values())
        block_seconds=perf_counter()-tick;tick=perf_counter()
        pred,cc=_current_prediction(obs,blocks,options=o,seed=int(o['seed'])+100000*index)
        assert all(np.isfinite(v).all() for v in pred)
        inference_seconds=perf_counter()-tick
        allrows.extend(dict(model=model,**row) for row in score(pred,truth,mesh.weights,p.ambient,current_sat))
        times.append(dict(model=model,previous_seconds=previous_seconds,mean_seconds=mean_seconds,covariance_seconds=block_seconds,current_seconds=inference_seconds,
            current_rhat=cc['rhat_max'],previous_rhat=pc['rhat_max'],current_retried=cc['current_retried'],previous_retried=pc['retried']))
        arrays.update({model+'_mean':pred[0],model+'_sd':pred[1],model+'_lower':pred[2],model+'_upper':pred[3],model+'_draws':pred[4]})
        print(model,'native inference completed; Rhat',cc['rhat_max'],flush=True)
        del pred,blocks,Koo,Kpo
    folder=OUT/'by_trajectory'/t.name;folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'native_pilot.npz',points=xy,triangles=mesh.triangles,area_weights=mesh.weights,
        previous_mean=previous,physics_mean=mean,oracle_mean=oracle,truth=truth,previous_truth=prevtruth,
        previous_camera=previous_camera,current_camera=current_camera,observation_node_ids=obsids,**arrays)
    reconstruction_figure(mesh,truth,current_camera,previous,mean,arrays,p.ambient)
    pd.DataFrame(allrows).to_csv(OUT/'native_pilot_metrics.csv',index=False)
    pd.DataFrame(times).to_csv(OUT/'native_pilot_runtime.csv',index=False)
    record=dict(trajectory=t.name,nodes=N,triangles=len(mesh.triangles),boundary_nodes=int(mesh.boundary.sum()),area_m2=float(mesh.weights.sum()),
        threshold_K=c,previous_censored=int((previous_camera==c).sum()),current_censored=int(current_sat.sum()),
        current_node_censored_fraction=float(current_sat.mean()),current_area_censored_fraction=float(np.sum(mesh.weights*current_sat)/mesh.weights.sum()),
        fixed_numeric_candidates=int(fixed.sum()),current_numeric=int((fixed&~current_sat).sum()),current_observations=len(obsids),
        nearest_sensor_max_distance_m=float(dist.max()),previous_draws=len(draws),previous_posterior=pc,
        diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],ambient=p.ambient,lengthscale_m=length,
        elapsed_total_seconds=perf_counter()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
        boundary_assumption='Zero temperature excess on outer surface boundary during propagation; source added afterward.',
        residual_innovation='Inherited free-space coordinate heat innovation, not an FEM-boundary-consistent SPDE innovation.',
        previous_state_representation='Hybrid marginal censored draws; unsaturated noisy values pinned, not a coherent full latent posterior.',
        native_vs_grid_comparison='Not matched observation realizations or discretization; feasibility pilot, not an architecture winner.')
    (OUT/'native_pilot_configuration.json').write_text(json.dumps(record,indent=2))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=ROOT/'outputs/by_experiment/45_censoring_severity_decomposition/source_fingerprints.json'
    for path,value in json.loads(old.read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value,path
    o=pd.read_csv(BASE/'fixed_configuration.csv').iloc[0].to_dict()
    pd.DataFrame([o]).to_csv(OUT/'frozen_configuration.csv',index=False)
    records=trajectory_catalog(DATA)
    assert len(records)==33
    assert sum(r.name in DEVELOPMENT_TRAJECTORIES for r in records)==3
    inventory=[]
    for r in records:
        t=ThermalTrajectory(r.xdmf_path); mesh=extract_surface(t)
        for index in [-2,-1]:
            for field in ['Temperature','HeatFluxZ']:
                values=t.read_field(field,index,surface_only=True)
                assert values.shape==(len(mesh.points),) and np.isfinite(values).all()
        inventory.append(dict(trajectory=r.name,split='development' if r.name in DEVELOPMENT_TRAJECTORIES else 'heldout',
            native_surface_nodes=len(mesh.points),surface_triangles=len(mesh.triangles),area_m2=float(mesh.weights.sum()),
            coordinates_sha256=hashlib.sha256(mesh.points.tobytes()).hexdigest(),
            triangles_sha256=hashlib.sha256(mesh.triangles.tobytes()).hexdigest()))
    pd.DataFrame(inventory).to_csv(OUT/'native_inventory.csv',index=False)
    print('All 33 native surfaces and final two temperature/source frames load; split 3 development / 30 held-out',flush=True)
    rows=[];pilot=None;meshchecks=[]
    for name in DEVELOPMENT_TRAJECTORIES:
        t=ThermalTrajectory(trajectory_path(DATA,name))
        mesh=extract_surface(t)
        assert abs(mesh.weights.sum()-.015*.010)<1e-14
        assert np.max(abs(mesh.stiffness@np.ones(len(mesh.points))))<1e-10
        p,_=prepare_trajectory(DATA,name,nx=int(o['nx']),ny=int(o['ny']),heat_flux_cutoff=o['heat_flux_cutoff'])
        rows.extend(compare_fields(t,p,mesh))
        meshchecks.append(dict(trajectory=name,nodes=len(mesh.points),triangles=len(mesh.triangles),area_m2=float(mesh.weights.sum()),boundary_nodes=int(mesh.boundary.sum()),
            stiffness_constant_residual=float(np.max(abs(mesh.stiffness@np.ones(len(mesh.points)))))))
        if name=='DiagonalScanPath_8':pilot=(t,p,mesh)
        print(name,'field comparison complete',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'native_projected_fields.csv',index=False)
    pd.DataFrame(meshchecks).to_csv(OUT/'mesh_checks.csv',index=False)
    native_pilot(*pilot,o)
    paths=[Path(__file__),ROOT/'src/native_surface.py',BASE/'fixed_configuration.csv']
    (OUT/'source_fingerprints.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},indent=2))


if __name__=='__main__':main()
