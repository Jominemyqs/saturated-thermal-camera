"""Numerical attribution of an unchanged native-input deterministic forecast."""
import numpy as np


def forecast_components(operator, previous, previous_truth, truth, flux, *,
                        ambient, source_coupling, source_flux_threshold, **physics):
    previous=np.asarray(previous,dtype=float)
    previous_truth=np.asarray(previous_truth,dtype=float)
    truth=np.asarray(truth,dtype=float)
    if previous.shape!=previous_truth.shape or previous.shape!=truth.shape:
        raise ValueError('Temperature fields must use identical nodes')
    args=dict(source_coupling=source_coupling,source_flux_threshold=source_flux_threshold,**physics)
    # Zero excess and zero ambient isolate the existing additive source exactly.
    source=operator.mean(np.zeros_like(previous),flux,'native',ambient=0.,**args)
    out=dict(previous_inferred=previous,previous_truth=previous_truth,truth=truth,
             previous_error=previous-previous_truth,source=source)
    for label,state in [('base',previous),('oracle',previous_truth)]:
        propagated=operator.propagate(state-ambient,'native',floor=True,**physics)
        without_source=ambient+propagated
        full=operator.mean(state,flux,'native',ambient=ambient,**args)
        np.testing.assert_array_equal(full,without_source+source)
        needed=truth-without_source
        error=full-truth
        np.testing.assert_allclose(error,source-needed,rtol=0,atol=2e-12)
        out.update({label+'_propagated_excess':propagated,label+'_without_source':without_source,
                    label+'_forecast':full,label+'_needed':needed,label+'_error':error,
                    label+'_propagation_change':without_source-state})
    out['propagated_state_error']=out['base_without_source']-out['oracle_without_source']
    np.testing.assert_allclose(out['base_error'],out['propagated_state_error']+out['oracle_error'],rtol=0,atol=2e-12)
    return out


def component_metrics(arrays, weights):
    weights=np.asarray(weights,dtype=float)
    if np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Positive region area is required')
    def average(v):return float(np.sum(weights*v)/weights.sum())
    rows=[]
    for key in ['previous_error','propagated_state_error','base_error','oracle_error']:
        err=arrays[key]
        rows.append(dict(component=key,bias_K=average(err),rmse_K=np.sqrt(average(err**2)),
                         mae_K=average(abs(err)),positive_error_area_fraction=average(err>0)))
    budget={key+'_mean_K':average(arrays[key]) for key in
            ['source','base_needed','oracle_needed','base_propagated_excess','oracle_propagated_excess',
             'base_propagation_change','oracle_propagation_change']}
    return rows,budget
