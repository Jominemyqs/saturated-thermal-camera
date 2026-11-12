"""Evaluation-only Gaussian predictive-prior summaries for Output 68.

These are not deterministic-forecast intervals or current posteriors. The mean
changes, but the saved predictive-prior diagonal is held fixed in each pair.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm


def gaussian_crps(mean, sd, truth):
    mean, sd, truth = np.broadcast_arrays(mean, sd, truth)
    if np.any(sd <= 0) or not all(np.isfinite(a).all() for a in [mean, sd, truth]):
        raise ValueError('Finite arrays and positive SD required')
    z = (truth-mean)/sd
    return sd*(z*(2*ndtr(z)-1)+2*np.exp(-.5*z*z)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('response_pilot_readonly', root/'scripts/72_thermal_physical_source_response.py')
    pilot = importlib.util.module_from_spec(spec); spec.loader.exec_module(pilot)
    rows, integral_checks, remaining = [], [], []
    for name in pilot.M.NAMES:
        audit = pilot.M.load(pilot.FEM/'by_trajectory'/name/'surface_comparison.npz')
        for level in range(3):
            d = pilot.S.case(name, level)
            means = pilot.M.load(pilot.folder(name)/f'means_{level}.npz')
            prior = pilot.S.prior(name, level)
            sd = np.sqrt(prior['prediction_variance']); half = norm.ppf(.975)*sd
            peak = int(d['truth'].argmax())
            state = means['propagation_only']-audit['old_off']
            propagation = audit['propagation_discrepancy']
            replay = audit['replay_discrepancy']
            independent_difference = means['physical_increment']-audit['fem_response']
            total = means['physical']-d['truth']
            np.testing.assert_allclose(total, state+propagation+replay+independent_difference,
                                       rtol=0, atol=1e-10)
            for region, w in pilot.M.regions(d, name).items():
                if w.sum() == 0: continue
                for term, error in [('inferred_surface_state', state), ('surface_propagation_discrepancy', propagation),
                                    ('FEM_replay_discrepancy', replay), ('independent_solve_roundoff', independent_difference),
                                    ('remaining_forecast_error', total)]:
                    remaining.append(dict(trajectory=name, level=level, region=region, term=term,
                        bias_K=float(np.sum(w*error)/w.sum()), rmse_K=float(np.sqrt(np.sum(w*error**2)/w.sum())),
                        true_peak_error_K=float(error[peak])))
            for arm in ['baseline', 'physical']:
                mean = means[arm]; err = mean-d['truth']; lo = mean-half; hi = mean+half
                crps = gaussian_crps(mean, sd, d['truth'])
                for region, w in pilot.M.regions(d, name).items():
                    if w.sum() == 0: continue
                    avg = lambda a: float(np.sum(w*a)/w.sum())
                    rows.append(dict(trajectory=name, level=level, arm=arm, region=region,
                        distribution='Gaussian predictive prior BEFORE current observations',
                        rmse_K=np.sqrt(avg(err**2)), mae_K=avg(abs(err)), bias_K=avg(err),
                        crps_K=avg(crps), coverage95=avg((lo <= d['truth']) & (d['truth'] <= hi)),
                        width95_K=avg(hi-lo), predictive_sd_K=avg(sd), true_peak_error_K=float(err[peak]),
                        true_peak_lower_K=float(lo[peak]), true_peak_upper_K=float(hi[peak]),
                        true_peak_covered=bool(lo[peak] <= d['truth'][peak] <= hi[peak])))
            if level == 0:
                saved = pd.read_csv(pilot.OUT/'increment_metrics.csv')
                for arm, key in [('baseline', 'old_increment'), ('physical', 'physical_increment')]:
                    # Explicit reduction checks the BLAS dot-product warnings seen at validation.
                    value = float(np.sum(d['area_weights']*means[key]))
                    original = saved[(saved.trajectory == name) & (saved.arm == arm)].area_integral_K_m2.iloc[0]
                    np.testing.assert_allclose(value, original, rtol=1e-14, atol=0)
                    integral_checks.append(dict(trajectory=name, arm=arm, explicit_sum_K_m2=value,
                                               dot_product_abs_difference=float(abs(value-original))))
    pd.DataFrame(rows).to_csv(pilot.OUT/'gaussian_predictive_prior_metrics.csv', index=False)
    pd.DataFrame(remaining).to_csv(pilot.OUT/'remaining_error_decomposition.csv', index=False)
    paths = [Path(__file__), root/'tests/test_physical_response_reporting.py']
    for name in pilot.M.NAMES:
        paths += [pilot.folder(name)/'response.npz', pilot.folder(name)/'response.json']
    pilot.M.dump(pilot.OUT/'reporting_verification.json', dict(integral_checks=integral_checks,
        interpretation='No uncertainty assigned to deterministic/oracle forecasts; Gaussian prior intervals reported separately.',
        files={str(p.resolve()): pilot.M.sha(p) for p in paths}))


if __name__ == '__main__': main()
