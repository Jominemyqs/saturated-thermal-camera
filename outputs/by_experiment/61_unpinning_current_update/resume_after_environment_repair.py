"""Resume unchanged driver 65 after temporary diagnostic packages disappeared.

No completed fit is replaced; the unfinished fit uses its original seed/settings.
"""
from pathlib import Path
from time import perf_counter, time
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
SPEC=importlib.util.spec_from_file_location('unchanged_current_audit',ROOT/'scripts/65_thermal_unpinning_current_update.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def main():
    start=perf_counter()
    for path,h in json.loads((M.OUT/'source_and_input_fingerprints.json').read_text()).items():
        assert M.sha(ROOT/path)==h,path
    before={str(p.relative_to(ROOT)):M.sha(p) for name in M.NAMES
            for p in M.folder(name).glob('level_*_repeat_*.*')}
    config=json.loads((M.OUT/'frozen_configuration.json').read_text())
    assert config['gate']==M.GATE and config['samples_per_chain']==M.SAMPLES and config['burn_in']==M.BURN
    # Verify the restored diagnostic version against two already saved fits.
    diagnostic_checks=[]
    for arm in ['legacy_hybrid','all_readings_pinned']:
        path=M.stem(M.NAMES[0],0,arm,0)
        with np.load(path.with_suffix('.npz')) as f:
            new=pd.concat([M.HELP.arviz_diagnostics(f['noisy_censored_chains'],f['constrained_ids'],'noisy_censored'),
                M.HELP.arviz_diagnostics(f['monitored_chains'],f['monitor_ids'],'latent_temperature')],ignore_index=True)
        old=pd.read_csv(path.with_suffix('.diagnostics.csv'))
        for col in ['rank_rhat','ess_bulk','ess_tail','mean_mcse_K']:
            np.testing.assert_allclose(new[col],old[col],rtol=1e-12,atol=1e-11)
        diagnostic_checks.append(str(path.relative_to(ROOT)))
    import arviz, xarray, xarray_einstats
    M.dump(M.OUT/'environment_recovery.json',dict(
        cause='Temporary ArviZ package lost its __init__.py and became an empty namespace package; from_dict was unavailable',
        failed_case=dict(trajectory='SpiralScanPath_13',level=0,arm='all_readings_pinned',repeat=0),
        failed_attempt='Four chains completed but diagnostics failed before fit artifacts were saved',
        action='Restore identical diagnostic versions in ignored repo tmp/thermal_diagnostics; rerun unfinished fit with SAME seed/settings',
        package_versions=dict(arviz=arviz.__version__,xarray=xarray.__version__,xarray_einstats=xarray_einstats.__version__),
        PYTHONPATH=os.environ.get('PYTHONPATH'),MPLCONFIGDIR=os.environ.get('MPLCONFIGDIR'),
        completed_artifacts_before_resume=before,diagnostic_replay_checks=diagnostic_checks,
        scientific_driver_unchanged=True,gate_unchanged=True,
        recovery_script_sha256=M.sha(Path(__file__))))
    for name in M.NAMES:
        for level in range(3):assert json.loads((M.folder(name)/f'replay_{level}.json').read_text())['exact']
    def call(name,level,arm,repeat):
        path=M.stem(name,level,arm,repeat)
        if path.with_suffix('.json').exists():
            assert all(path.with_suffix(suffix).exists() for suffix in ['.npz','.metrics.csv','.diagnostics.csv'])
            return
        assert not path.with_suffix('.npz').exists(),'Unexpected partial artifact; inspect before resuming'
        subprocess.run([sys.executable,str(ROOT/'scripts/65_thermal_unpinning_current_update.py'),
            '--phase','fit','--name',name,'--level',str(level),'--arm',arm,'--repeat',str(repeat)],
            cwd=ROOT,check=True,timeout=900)
    for name in M.NAMES:
        for repeat in range(2):
            for arm in M.ARMS:call(name,0,arm,repeat)
    reference=[json.loads(M.stem(name,0,arm,repeat).with_suffix('.json').read_text())
        for name in M.NAMES for repeat in range(2) for arm in M.ARMS]
    passed=all(r['passed'] for r in reference)
    M.dump(M.OUT/'reference_gate.json',dict(passed=passed,gate=M.GATE,failed=[r for r in reference if not r['passed']]))
    if passed:
        for level in [1,2]:
            for name in M.NAMES:
                for arm in M.ARMS:call(name,level,arm,0)
    summaries=M.collect()
    for path,h in before.items():assert M.sha(ROOT/path)==h,path
    for path,h in json.loads((M.OUT/'source_and_input_fingerprints.json').read_text()).items():assert M.sha(ROOT/path)==h,path
    checks=pd.read_csv(M.OUT/'prior_verification.csv')
    for _,g in checks.groupby(['trajectory','level']):
        for col in ['current_camera_sha256','observation_ids_sha256','source_sha256','Q_cross_sha256']:assert g[col].nunique()==1,col
    replays=[json.loads((M.folder(name)/f'replay_{level}.json').read_text()) for name in M.NAMES for level in range(3)]
    M.dump(M.OUT/'verification.json',dict(completed_fits=len(summaries),reference_gate=passed,
        all_completed_fits_pass=all(s['passed'] for s in summaries),historical_replays_exact=6,
        previous_caches_unchanged=True,source_inputs_unchanged=True,full_prior_cross_covariance=True,
        current_likelihood_noise_and_observations_matched=True,previous_inference_rerun=False,
        previously_completed_artifacts_preserved=True,environment_recovery='environment_recovery.json',
        recovery_wall_seconds=perf_counter()-start,
        elapsed_since_configuration_write_seconds=time()-(M.OUT/'frozen_configuration.json').stat().st_mtime,
        completed_fit_seconds=sum(s['seconds'] for s in summaries),historical_replay_seconds=sum(r['seconds'] for r in replays),
        timing_note='Completed fit/preparation/replay timings exclude the failed unsaved fit and environment repair; elapsed wall time includes interruptions'))


if __name__=='__main__':main()
