"""Shared scoring and RBF blocks for a matched ceiling-response comparison."""
from __future__ import annotations
import numpy as np
from src.censored_gp import RBFConfig, rbf_covariance
from src.ceiling_sweep import observation_indices
from src.metrics import empirical_crps


def rbf_blocks(prepared, observations, mean_field, signal_sd, lengthscale, noise_sd):
    config=RBFConfig(mean_temp=prepared.ambient,signal_sd=signal_sd,lengthscale=lengthscale,noise_sd=noise_sd)
    pred=np.asarray(observations['x_pred'])
    obs=np.asarray(observations['x_obs'])
    mean=np.asarray(mean_field).ravel()
    return dict(prediction_mean=mean,observation_mean=mean[observation_indices(prepared,obs)],
        observed_covariance=rbf_covariance(obs,obs,config),
        pred_observed_covariance=rbf_covariance(pred,obs,config),
        prediction_variance=np.full(len(pred),signal_sd**2))


def evaluate_prediction(prediction,truth,ambient,observed_censored):
    mean,sd,lower,upper,draws=prediction
    y=np.asarray(truth).ravel()
    top=np.zeros(len(y),bool)
    top[np.argsort(y,kind='mergesort')[-max(1,int(.01*len(y))):]]=True
    e=mean-y
    crps=empirical_crps(draws,y)
    result=dict(field_excess_rel_l2=float(np.linalg.norm(e)/np.linalg.norm(y-ambient)),
        peak_absolute_error_K=float(abs(mean.max()-y.max())))
    for region,mask in [('overall',np.ones(len(y),bool)),('top_1pct',top),('censored',np.asarray(observed_censored).ravel())]:
        result.update({f'{region}_n_pixels':int(mask.sum()),
            f'{region}_rmse_K':float(np.sqrt(np.mean(e[mask]**2))),
            f'{region}_mae_K':float(np.mean(abs(e[mask]))),f'{region}_bias_K':float(np.mean(e[mask])),
            f'{region}_crps_K':float(np.mean(crps[mask])),
            f'{region}_coverage_95':float(np.mean((lower[mask]<=y[mask])&(y[mask]<=upper[mask]))),
            f'{region}_width_95_K':float(np.mean(upper[mask]-lower[mask])),
            f'{region}_sd_K':float(np.mean(sd[mask]))})
    return result
