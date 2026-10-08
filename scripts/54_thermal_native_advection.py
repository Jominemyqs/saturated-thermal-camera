"""Native B/C signed-error diagnostics and isolated two-path advection pilot."""
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
import hashlib
import json
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.linalg.blas import dgemm
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.native_surface import extract_surface, hottest_area_weights
from src.native_diagnostics import source_centroid, write_diagnostics, advection_comparison
from src.matched_projection import MatchedHeatOperator, weighted_metrics
from src.thermal_trajectory import ThermalTrajectory, trajectory_catalog
from src.thermal_posterior_physics import prepare_trajectory
from src.pseudo_censoring_runner import _previous_prediction, _current_prediction
from src.censored_gp import RBFConfig, rbf_covariance
from src.stochastic_heat_gp import StochasticHeatConfig, finite_step_innovation_covariance

BASE=ROOT/'outputs/by_experiment/49_matched_native_grid_gp'
OUT=ROOT/'outputs/by_experiment/50_native_advection'
DATA=ROOT.parent/'heat_eq_laser_trajectories'
NAMES=['DiagonalScanPath_8','SpiralScanPath_13']
VARIANTS=[('B',False,False),('B_mean_advection',True,False),
          ('C',False,False),('C_mean_advection',True,False),
          ('C_uncertainty_advection',False,True),('C_both_advection',True,True)]


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def gram(a,b):
    return dgemm(alpha=1.,a=np.asfortranarray(a),b=np.asfortranarray(b),trans_b=True)


def main(diagnostics_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    o=pd.read_csv(BASE/'frozen_configuration.csv').iloc[0].to_dict()
    baseline_checks=pd.read_csv(BASE/'matching_checks.csv')
    catalog=trajectory_catalog(DATA)
    paths={r.name:(i,r) for i,r in enumerate(catalog) if r.name in NAMES}
    cases={}; limit=0.
    for name in NAMES:
        _,r=paths[name]; t=ThermalTrajectory(r.xdmf_path);mesh=extract_surface(t)
        with np.load(BASE/'by_trajectory'/name/'matched.npz') as f:
            cached={k:f[k] for k in f.files}
        flux=t.read_heat_flux_z(-1,surface_only=True)
        prevflux=t.read_heat_flux_z(-2,surface_only=True)
        s0=source_centroid(mesh.points,mesh.weights,prevflux,o['source_flux_threshold'])
        s1=source_centroid(mesh.points,mesh.weights,flux,o['source_flux_threshold'])
        displacement=s1-s0
        assert np.linalg.norm(displacement)>0
        cases[name]=(t,mesh,cached,flux,displacement)
        limit=max(limit,*[float(abs(cached[k]-cached['truth']).max()) for k in
                        ['native_physics_mean','native_B_mean','native_C_mean']])
    signrows=[]
    for name,(_,mesh,cached,_,d) in cases.items():
        signrows.extend(write_diagnostics(BASE/'native_diagnostics',name,cached,d,error_limit=limit))
    pd.DataFrame(signrows).to_csv(BASE/'native_diagnostics/signed_error_summary.csv',index=False)
    if diagnostics_only: return
    results=[]; checks=[]; resolutions=[]; timings=[]; deltas=[]
    for name in NAMES:
        start=perf_counter(); index,r=paths[name]
        t,mesh,cached,flux,d=cases[name]; xy=mesh.points; N=len(xy)
        p,_=prepare_trajectory(DATA,name,nx=int(o['nx']),ny=int(o['ny']),heat_flux_cutoff=o['heat_flux_cutoff'])
        reference=baseline_checks.query('trajectory==@name and representation=="native" and model=="B"').iloc[0]
        ceiling=float(cached['ceiling']); obsids=cached['observation_ids']
        points=np.column_stack([xy,np.full(N,t.times[-1])]);camera=cached['camera']
        sat=camera==ceiling; prevcamera=cached['previous_camera']
        obs=dict(x_obs=points[obsids],y_obs=camera[obsids],sat_mask=sat[obsids],
                 saturated_full=sat,x_pred=points,threshold=ceiling)
        # The previous helper pins unsaturated measurements; retain the original mask argument.
        from scipy.spatial import cKDTree
        gx,gy=np.meshgrid(p.xs[::5],p.ys[::5])
        _,sensors=cKDTree(xy).query(np.column_stack([gx.ravel(),gy.ravel()]))
        fixed=np.zeros(N,bool);fixed[np.unique(sensors)]=True
        np.testing.assert_array_equal(obsids,np.flatnonzero(fixed|sat))
        proxy=SimpleNamespace(points=xy,ambient=p.ambient,history=t.read_temperature(-2,surface_only=True).reshape(1,1,N))
        frame=dict(time_index=0,time=float(t.times[-2]),clipped_full=prevcamera[None],saturated_full=(prevcamera==ceiling)[None])
        length=p.source_lengthscale*o['length_multiplier']
        previous,draws,pc=_previous_prediction(proxy,frame=frame,fixed_mask=fixed[None],threshold=ceiling,
            signal_sd=o['signal_sd'],lengthscale=length,noise_sd=o['previous_noise_sd'],options=o,seed=int(reference.previous_seed))
        previous=previous.ravel();draws=draws[:,0,:]
        assert digest(previous)==reference.previous_mean_sha256
        assert digest(draws)==reference.previous_draws_sha256
        for a,key in [(camera,'current_camera_sha256'),(prevcamera,'previous_camera_sha256'),
                      (obs['y_obs'],'likelihood_y_sha256'),(points,'coordinates_sha256')]:
            assert digest(a)==reference[key]
        op=MatchedHeatOperator(mesh,p.xs,p.ys)
        kw=dict(dt=float(t.times[-1]-t.times[-2]),diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'])
        mk=dict(ambient=p.ambient,source_coupling=o['source_coupling'],source_flux_threshold=o['source_flux_threshold'],**kw)
        means={False:op.mean(previous,flux,'native',**mk),True:op.mean(previous,flux,'native',displacement=d,**mk)}
        np.testing.assert_array_equal(means[False],cached['native_physics_mean'])
        finer=MatchedHeatOperator(mesh,p.xs,p.ys,nx=1201,ny=801)
        fine=finer.mean(previous,flux,'native',displacement=d,**mk)
        hot=hottest_area_weights(cached['truth'],mesh.weights)
        dr=fine-means[True]; err=float(np.sqrt(np.sum(hot*dr**2)/hot.sum()))
        assert err<.05,('Transport resolution',err)
        resolutions.append(dict(trajectory=name,advective_mean_hot_rmse_change_K=err,max_change_K=float(abs(dr).max())))
        del finer
        rbfcfg=RBFConfig(mean_temp=p.ambient,signal_sd=o['signal_sd'],lengthscale=length,noise_sd=o['current_noise_sd'])
        rbfoo=rbf_covariance(points[obsids],points[obsids],rbfcfg);rbfpo=rbf_covariance(points,points[obsids],rbfcfg)
        cfg=StochasticHeatConfig(signal_sd=o['signal_sd'],forcing_lengthscale=length*o['forcing_length_multiplier'],
            diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'],quadrature_order=int(o['quadrature_order']))
        qoo=finite_step_innovation_covariance(points[obsids],points[obsids],cfg,kw['dt'])
        qpo=finite_step_innovation_covariance(points,points[obsids],cfg,kw['dt'])
        qdiag=float(finite_step_innovation_covariance(points[:1],points[:1],cfg,kw['dt'])[0,0])
        for a,key in [(rbfoo,'rbf_oo_sha256'),(rbfpo,'rbf_po_sha256'),(qoo,'Q_oo_sha256'),(qpo,'Q_po_sha256')]:
            assert digest(a)==reference[key]
        # Equal-time homogeneous innovation is unchanged by uniform translation.
        shifted=points.copy();shifted[:,:2]+=d
        np.testing.assert_allclose(finite_step_innovation_covariance(shifted[:10],shifted[:10],cfg,kw['dt']),
                                   finite_step_innovation_covariance(points[:10],points[:10],cfg,kw['dt']),atol=1e-12)
        covariances={}
        for transport in [False,True]:
            propagated=op.propagate(draws-previous,'native',displacement=d if transport else None,**kw)
            features=propagated.T/np.sqrt(len(draws)-1)
            covariances[transport]=(qoo+gram(features[obsids],features[obsids]),
                                   qpo+gram(features,features[obsids]),qdiag+np.sum(features**2,axis=1))
        arrays={};eqmax=0.
        for label,meanadv,covadv in VARIANTS:
            tick=perf_counter();mean=means[meanadv]
            if label.startswith('B'):
                koo,kpo,diag=rbfoo,rbfpo,np.full(N,o['signal_sd']**2)
            else: koo,kpo,diag=covariances[covadv]
            blocks=dict(prediction_mean=mean,observation_mean=mean[obsids],observed_covariance=koo,
                        pred_observed_covariance=kpo,prediction_variance=diag)
            pred,cc=_current_prediction(obs,blocks,options=o,seed=int(reference.current_seed))
            if label in ['B','C']:
                for value,suffix in zip(pred,['mean','sd','lower','upper','draws']):
                    delta=float(abs(value-cached['native_'+label+'_'+suffix]).max())
                    eqmax=max(eqmax,delta)
                    np.testing.assert_allclose(value,cached['native_'+label+'_'+suffix],atol=1e-10,rtol=1e-12)
            for a,key in [(draws,'previous_draws_sha256'),(previous,'previous_mean_sha256'),
                          (obs['y_obs'],'likelihood_y_sha256'),(qoo,'Q_oo_sha256'),(qpo,'Q_po_sha256'),
                          (rbfoo,'rbf_oo_sha256'),(rbfpo,'rbf_po_sha256')]: assert digest(a)==reference[key]
            results.extend(dict(trajectory=name,model=label,mean_advection=meanadv,uncertainty_advection=covadv,**m)
                           for m in weighted_metrics(pred,cached['truth'],mesh.weights,p.ambient,sat))
            checks.append(dict(trajectory=name,model=label,rhat_max=cc['rhat_max'],previous_rhat=pc['rhat_max'],
                current_retried=cc['current_retried'],previous_draws_sha256=digest(draws),current_y_sha256=digest(obs['y_obs']),
                current_seed=int(reference.current_seed),mean_sha256=digest(mean),covariance_sha256=digest(koo),
                baseline_prediction_max_abs=eqmax,displacement_x_mm=d[0]*1000,displacement_y_mm=d[1]*1000))
            timings.append(dict(trajectory=name,model=label,current_seconds=perf_counter()-tick))
            for suffix,value in zip(['mean','sd','lower','upper','draws'],pred):arrays[label+'_'+suffix]=value
            print(name,label,'done Rhat',cc['rhat_max'],flush=True)
        dest=OUT/'by_trajectory'/name;dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'advection.npz',**arrays,physics_mean=means[False],advective_physics_mean=means[True],displacement=d)
        for row in timings:
            if row['trajectory']==name:row['path_total_seconds']=perf_counter()-start
        for file,rows in [('results',results),('matching_checks',checks),('resolution_checks',resolutions),('runtime',timings)]:
            pd.DataFrame(rows).to_csv(OUT/(file+'.csv'),index=False)
    df=pd.DataFrame(results)
    for (name,region),g in df.groupby(['trajectory','region']):
        for _,row in g.iterrows():
            if row.model in ['B','C']:continue
            base=g[g.model==row.model[0]].iloc[0]
            deltas.append(dict(trajectory=name,region=region,model=row.model,control=base.model,
                **{k:row[k]-base[k] for k in ['rmse_K','crps_K','bias_K','coverage95','width95_K','peak_absolute_error_K']}))
    pd.DataFrame(deltas).to_csv(OUT/'paired_changes.csv',index=False)
    advection_comparison(OUT)
    (OUT/'verification.json').write_text(json.dumps(dict(baseline_reproduced=True,previous_draws_shared=True,
        current_only_likelihood=True,no_refit=True,native_inputs_only=True,paths=NAMES,
        innovation_Q_unchanged_under_homogeneous_advection=True,
        source_definition_unconfirmed=True,previous_representation='hybrid marginal samples, not full joint posterior'),indent=2))
    files=[Path(__file__),ROOT/'src/matched_projection.py',ROOT/'src/native_diagnostics.py',ROOT/'src/pseudo_censoring_runner.py']
    (OUT/'source_fingerprints.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--diagnostics-only',action='store_true')
    main(parser.parse_args().diagnostics_only)
