"""Two-path matched native/projected-input B/C pilot; all camera data shared."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import hashlib
import json
import sys
import resource

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.linalg.blas import dgemm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import PowerNorm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.native_surface import extract_surface, hottest_area_weights
from src.matched_projection import MatchedHeatOperator, weighted_metrics
from src.thermal_trajectory import ThermalTrajectory, trajectory_catalog
from src.thermal_posterior_physics import prepare_trajectory
from src.pseudo_censoring_runner import _previous_prediction, _current_prediction
from src.censored_gp import RBFConfig, rbf_covariance
from src.stochastic_heat_gp import StochasticHeatConfig, finite_step_innovation_covariance

BASE=ROOT/'outputs/by_experiment/43_matched_ceiling_model_gap'
OUT=ROOT/'outputs/by_experiment/49_matched_native_grid_gp'
DATA=ROOT.parent/'heat_eq_laser_trajectories'
NAMES=['DiagonalScanPath_8','SpiralScanPath_13']


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def gram(a,b):
    return dgemm(alpha=1.,a=np.asfortranarray(a),b=np.asfortranarray(b),trans_b=True)


def plot_reconstructions(name,mesh,truth,ambient,arrays):
    keys=['native_B','projected_B','native_C','projected_C']
    tri=Triangulation(mesh.points[:,0]*1000,mesh.points[:,1]*1000,mesh.triangles)
    fig,axes=plt.subplots(2,4,figsize=(15,7),constrained_layout=True)
    norm=PowerNorm(.35,vmin=ambient,vmax=float(truth.max()))
    errors=[abs(arrays[k+'_mean']-truth) for k in keys]
    enorm=PowerNorm(.5,vmin=0,vmax=max(float(e.max()) for e in errors))
    for i,k in enumerate(keys):
        im=axes[0,i].tripcolor(tri,arrays[k+'_mean'],shading='gouraud',cmap='inferno',norm=norm)
        axes[0,i].set_title(k.replace('_',' input + '),fontsize=11)
        er=axes[1,i].tripcolor(tri,errors[i],shading='gouraud',cmap='magma',norm=enorm)
        axes[1,i].set_title('Absolute error vs native truth',fontsize=10)
    fig.colorbar(im,ax=list(axes[0]),label='Temperature (K); shared power scale',shrink=.8,extend='both')
    fig.colorbar(er,ax=list(axes[1]),label='Absolute error (K); shared power scale',shrink=.8)
    for ax in axes.flat:
        ax.set(xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
    fig.suptitle(name+': matched camera, previous posterior, evaluation and heat operator\nB = RBF residual; C = propagated uncertainty + heat innovation',fontsize=12)
    fig.savefig(OUT/f'reconstruction_{name}.png',dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    previous_results=pd.read_csv(OUT/'results.csv') if (OUT/'results.csv').exists() else None
    fingerprint=ROOT/'outputs/by_experiment/48_controlled_projection_audit/source_fingerprints.json'
    for path,value in json.loads(fingerprint.read_text()).items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value,path
    o=pd.read_csv(BASE/'fixed_configuration.csv').iloc[0].to_dict()
    pd.DataFrame([o]).to_csv(OUT/'frozen_configuration.csv',index=False)
    catalog=trajectory_catalog(DATA); assert len(catalog)==33
    reference=pd.read_csv(BASE/'results.csv')
    results=[]; checks=[]; runtimes=[]; refinement=[]
    for index,r in enumerate(catalog):
        if r.name not in NAMES: continue
        start=perf_counter()
        t=ThermalTrajectory(r.xdmf_path); mesh=extract_surface(t); xy=mesh.points; N=len(xy)
        p,_=prepare_trajectory(DATA,r.name,nx=int(o['nx']),ny=int(o['ny']),heat_flux_cutoff=o['heat_flux_cutoff'])
        truth=t.read_temperature(-1,surface_only=True); previous_truth=t.read_temperature(-2,surface_only=True)
        flux=t.read_heat_flux_z(-1,surface_only=True)
        ceiling=float(reference[(reference.trajectory==r.name)&(reference.level_index==0)].ceiling_K.iloc[0])
        rng=np.random.default_rng(int(o['seed'])+10000*index)
        previous_camera=np.minimum(previous_truth+rng.normal(0,o['measurement_noise_sd'],N),ceiling)
        camera=np.minimum(truth+rng.normal(0,o['measurement_noise_sd'],N),ceiling)
        gx,gy=np.meshgrid(p.xs[::5],p.ys[::5])
        distance,sensors=cKDTree(xy).query(np.column_stack([gx.ravel(),gy.ravel()]))
        fixed=np.zeros(N,bool); fixed[np.unique(sensors)]=True
        points=np.column_stack([xy,np.full(N,t.times[-1])])
        saturated=camera==ceiling; obsids=np.flatnonzero(fixed|saturated)
        obs=dict(x_obs=points[obsids],y_obs=camera[obsids],sat_mask=saturated[obsids],
                 saturated_full=saturated,x_pred=points,threshold=ceiling)
        length=p.source_lengthscale*o['length_multiplier']
        previous_seed=int(o['seed'])+50000*index; current_seed=int(o['seed'])+100000*index
        # This adapter changes geometry only; retain the existing hybrid inference.
        proxy=SimpleNamespace(points=xy,ambient=p.ambient,history=previous_truth.reshape(1,1,N))
        frame=dict(time_index=0,time=float(t.times[-2]),clipped_full=previous_camera[None],
                   saturated_full=(previous_camera==ceiling)[None])
        tick=perf_counter()
        previous,draws,pc=_previous_prediction(proxy,frame=frame,fixed_mask=fixed[None],threshold=ceiling,
            signal_sd=o['signal_sd'],lengthscale=length,noise_sd=o['previous_noise_sd'],options=o,seed=previous_seed)
        previous=previous.ravel(); draws=draws[:,0,:]
        previous_seconds=perf_counter()-tick
        if r.name=='DiagonalScanPath_8':
            with np.load(ROOT/'outputs/by_experiment/47_native_mesh_pilot/by_trajectory'/r.name/'native_pilot.npz') as old:
                np.testing.assert_array_equal(previous_camera,old['previous_camera'])
                np.testing.assert_array_equal(camera,old['current_camera'])
                np.testing.assert_array_equal(previous,old['previous_mean'])
        op=MatchedHeatOperator(mesh,p.xs,p.ys)
        finer=MatchedHeatOperator(mesh,p.xs,p.ys,nx=1201,ny=801)
        physics=dict(dt=float(t.times[-1]-t.times[-2]),diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'])
        meanargs=dict(ambient=p.ambient,source_coupling=o['source_coupling'],source_flux_threshold=o['source_flux_threshold'],**physics)
        rbfcfg=RBFConfig(mean_temp=p.ambient,signal_sd=o['signal_sd'],lengthscale=length,noise_sd=o['current_noise_sd'])
        hcfg=StochasticHeatConfig(signal_sd=o['signal_sd'],forcing_lengthscale=length*o['forcing_length_multiplier'],
            diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],quadrature_order=int(o['quadrature_order']))
        rbfoo=rbf_covariance(points[obsids],points[obsids],rbfcfg)
        rbfpo=rbf_covariance(points,points[obsids],rbfcfg)
        qoo=finite_step_innovation_covariance(points[obsids],points[obsids],hcfg,physics['dt'])
        qpo=finite_step_innovation_covariance(points,points[obsids],hcfg,physics['dt'])
        qdiag=float(finite_step_innovation_covariance(points[:1],points[:1],hcfg,physics['dt'])[0,0])
        shared=dict(trajectory=r.name,previous_draws_sha256=digest(draws),previous_mean_sha256=digest(previous),
            coordinates_sha256=digest(points),
            previous_camera_sha256=digest(previous_camera),current_camera_sha256=digest(camera),
            observation_ids_sha256=digest(obsids),likelihood_y_sha256=digest(obs['y_obs']),
            censor_mask_sha256=digest(saturated),rbf_oo_sha256=digest(rbfoo),rbf_po_sha256=digest(rbfpo),
            Q_oo_sha256=digest(qoo),Q_po_sha256=digest(qpo),
            hot_area_sha256=digest(hottest_area_weights(truth,mesh.weights)),
            previous_seed=previous_seed,current_seed=current_seed,ceiling_K=ceiling,
            nodes=N,numeric_count=int(np.sum(fixed&~saturated)),censored_count=int(saturated.sum()),
            area_censored_fraction=float(np.sum(mesh.weights*saturated)/mesh.weights.sum()),
            previous_rhat=pc['rhat_max'],previous_retried=pc['previous_retried'],
            previous_seconds=previous_seconds,sensor_max_distance_mm=float(distance.max()*1000))
        arrays={}; check_draw_ids=np.linspace(0,len(draws)-1,8,dtype=int)
        for representation in ['native','projected']:
            mean=op.mean(previous,flux,representation,**meanargs)
            fine_mean=finer.mean(previous,flux,representation,**meanargs)
            delta=fine_mean-mean; hw=hottest_area_weights(truth,mesh.weights)
            rmse=float(np.sqrt(np.sum(hw*delta**2)/hw.sum()))
            assert rmse<.05,('Heat resolution needs revisiting',r.name,representation,rmse)
            # Compare the same selected draws, rather than different covariance samples.
            small=draws[check_draw_ids]-previous
            fine_prop=finer.propagate(small,representation,**physics)
            coarse_prop=op.propagate(small,representation,**physics)
            refinement.append(dict(trajectory=r.name,representation=representation,
                mean_hot_rmse_change_K=rmse,mean_max_change_K=float(abs(delta).max()),
                propagated_draw_rmse_change_K=float(np.sqrt(np.mean((fine_prop-coarse_prop)**2)))))
            arrays[representation+'_physics_mean']=mean
            for model in ['B','C']:
                tick=perf_counter(); key=representation+'_'+model
                if model=='B':
                    koo,kpo=rbfoo,rbfpo
                    diag=np.full(N,o['signal_sd']**2)
                    inherited=np.zeros(N)
                else:
                    propagated=op.propagate(draws-previous,representation,**physics)
                    features=propagated.T/np.sqrt(len(draws)-1)
                    foo=features[obsids]
                    koo=qoo+gram(foo,foo); kpo=qpo+gram(features,foo)
                    inherited=np.sum(features**2,axis=1); diag=qdiag+inherited
                    selected=np.argsort(inherited)[-8:]; so=np.argsort(inherited[obsids])[-8:]
                    expected=qpo[np.ix_(selected,so)]+np.einsum('ik,jk->ij',features[selected],foo[so],optimize=False)
                    np.testing.assert_allclose(kpo[np.ix_(selected,so)],expected,atol=1e-12,rtol=1e-12)
                    del features,foo,propagated
                covariance_seconds=perf_counter()-tick
                blocks=dict(prediction_mean=mean,observation_mean=mean[obsids],observed_covariance=koo,
                            pred_observed_covariance=kpo,prediction_variance=diag)
                assert all(np.isfinite(a).all() for a in blocks.values())
                tick=perf_counter()
                pred,cc=_current_prediction(obs,blocks,options=o,seed=current_seed)
                inference_seconds=perf_counter()-tick
                for values,check in [(draws,'previous_draws_sha256'),(previous,'previous_mean_sha256'),
                                     (obs['y_obs'],'likelihood_y_sha256'),(camera,'current_camera_sha256'),
                                     (saturated,'censor_mask_sha256'),(obsids,'observation_ids_sha256'),
                                     (points,'coordinates_sha256'),(rbfoo,'rbf_oo_sha256'),
                                     (rbfpo,'rbf_po_sha256'),(qoo,'Q_oo_sha256'),(qpo,'Q_po_sha256')]:
                    assert digest(values)==shared[check],check
                results.extend(dict(trajectory=r.name,representation=representation,model=model,
                                    **row) for row in weighted_metrics(pred,truth,mesh.weights,p.ambient,saturated))
                checks.append(dict(**shared,representation=representation,model=model,
                    prior_mean_sha256=digest(mean),prior_diagonal_sha256=digest(diag),
                    current_rhat=cc['rhat_max'],current_censored_rhat=cc['selected_rhat_max'],
                    current_retried=cc['current_retried'],
                    evaluation_draws=len(pred[4]),mean_prior_variance=float(np.sum(mesh.weights*diag)/mesh.weights.sum()),
                    mean_inherited_variance=float(np.sum(mesh.weights*inherited)/mesh.weights.sum())))
                runtimes.append(dict(trajectory=r.name,model=key,previous_seconds_shared=previous_seconds,
                    covariance_seconds=covariance_seconds,current_seconds=inference_seconds))
                for suffix,value in zip(['mean','sd','lower','upper','draws'],pred): arrays[key+'_'+suffix]=value
                print(r.name,key,'done; Rhat',cc['rhat_max'],flush=True)
        folder=OUT/'by_trajectory'/r.name; folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'matched.npz',points=xy,triangles=mesh.triangles,area_weights=mesh.weights,
            truth=truth,previous_mean=previous,previous_camera=previous_camera,camera=camera,ceiling=ceiling,
            observation_ids=obsids,ambient=p.ambient,**arrays)
        plot_reconstructions(r.name,mesh,truth,p.ambient,arrays)
        path_seconds=perf_counter()-start
        for row in runtimes:
            if row['trajectory']==r.name: row['path_total_seconds']=path_seconds
        print(r.name,'total seconds',path_seconds,flush=True)
        for file,rows in [('results',results),('matching_checks',checks),('runtime',runtimes),('resolution_checks',refinement)]:
            pd.DataFrame(rows).to_csv(OUT/(file+'.csv'),index=False)
    d=pd.DataFrame(results)
    replay_difference=None
    if previous_results is not None and len(previous_results)==len(d):
        indexcols=['trajectory','representation','model','region']
        a=previous_results.set_index(indexcols).sort_index()
        b=d.set_index(indexcols).sort_index()
        np.testing.assert_allclose(a.to_numpy(),b.to_numpy(),atol=1e-12,rtol=1e-12)
        replay_difference=float(np.max(abs(a.to_numpy()-b.to_numpy())))
    metrics=['relative_L2','rmse_K','mae_K','bias_K','crps_K','coverage95','width95_K','peak_absolute_error_K']
    paired=[]
    for (trajectory,model,region),group in d.groupby(['trajectory','model','region']):
        native=group[group.representation=='native'].iloc[0]; projected=group[group.representation=='projected'].iloc[0]
        paired.append(dict(trajectory=trajectory,model=model,region=region,
            **{metric:float(projected[metric]-native[metric]) for metric in metrics}))
    pd.DataFrame(paired).to_csv(OUT/'paired_projected_minus_native.csv',index=False)
    verify=pd.DataFrame(checks)
    sharedcols=[c for c in verify if c.endswith('sha256') and not c.startswith('prior_')]+['current_seed','previous_seed','ceiling_K']
    for name,group in verify.groupby('trajectory'):
        assert all(group[c].nunique()==1 for c in sharedcols),(name,'unmatched configuration')
    files=[Path(__file__),ROOT/'src/matched_projection.py',ROOT/'src/pseudo_censoring_runner.py',
           ROOT/'src/thermal_posterior_physics.py',BASE/'fixed_configuration.csv']
    (OUT/'source_fingerprints.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
    (OUT/'verification.json').write_text(json.dumps(dict(paths=NAMES,previous_posterior_inferred_once=True,
        final_likelihood_current_only=True,shared_camera_masks_seeds_kernels_verified=True,
        no_refit=True,no_FEM_operator=True,common_grid=[601,401],resolution_check_grid=[1201,801],
        previous_representation='Hybrid marginal draws with unsaturated noisy values pinned',
        comparison='projection of previous state/samples/source on or off; both fits/evaluation use native coordinates',
        replay_metric_max_abs=replay_difference,
        peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20),indent=2))


if __name__=='__main__': main()
