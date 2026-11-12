"""Output 68: prescribed boundary response replaces ONLY the current increment."""
from pathlib import Path
from time import perf_counter
import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.physical_source_response import replace_increment, COVARIANCE_KEYS, validate_geometry
from src.thermal_trajectory import ThermalTrajectory
from src.native_surface import extract_surface
from src.prescribed_laser_audit import NotebookLaser

SPEC = importlib.util.spec_from_file_location('frozen_source_off_audit', ROOT/'scripts/67_thermal_native_source_on_off.py')
S = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(S)
M = S.M
OUT = ROOT/'outputs/by_experiment/68_physical_source_response'
FEM = ROOT/'outputs/by_experiment/67_one_step_fem_response'
AUDIT = ROOT/'outputs/by_experiment/66_prescribed_laser_input_audit'
SOURCE_OFF = ROOT/'outputs/by_experiment/63_native_source_on_off'


def folder(name): return OUT/'by_trajectory'/name
def target(name, level, repeat=0): return folder(name)/f'level_{level}_physical_repeat_{repeat}'
def response(name): return M.load(folder(name)/'response.npz')['surface_response']
def candidate(name, level):
    d = S.case(name, level)
    return replace_increment(S.prior(name, level), d['observation_ids'], d['gradient_contribution'], response(name))


def prepare():
    if OUT.exists(): raise FileExistsError(OUT)
    assert json.loads((FEM/'verification.json').read_text())['decomposition_completed']
    assert json.loads((M.OUT/'verification.json').read_text())['all_completed_fits_pass']
    historical = json.loads((M.OUT/'source_and_input_fingerprints.json').read_text())
    for path, digest in historical.items(): assert M.sha(ROOT/path) == digest, path
    prior_config = json.loads((M.OUT/'frozen_configuration.json').read_text())
    assert prior_config['gate'] == M.GATE
    assert prior_config['samples_per_chain'] == M.SAMPLES and prior_config['burn_in'] == M.BURN
    M.HELP.arviz_diagnostics(np.random.default_rng(42).normal(size=(4, 100, 2)), np.arange(2), 'environment_probe')
    import arviz, scipy
    OUT.mkdir()
    paths = [ROOT/p for p in historical] + [Path(__file__), ROOT/'src/physical_source_response.py',
        ROOT/'tests/test_physical_source_response.py', ROOT/'src/fem_replay.py',
        ROOT/'src/source_off_audit.py', ROOT/'src/prescribed_laser_audit.py',
        ROOT/'scripts/65_thermal_unpinning_current_update.py', ROOT/'scripts/67_thermal_native_source_on_off.py',
        M.OUT/'frozen_configuration.json', M.OUT/'verification.json', M.OUT/'prior_verification.csv',
        FEM/'verification.json', FEM/'fingerprints.json', FEM/'environment_explicit.txt']
    # Check the original FEM audit's source and data fingerprints, without reading fields.
    for path, digest in json.loads((FEM/'fingerprints.json').read_text())['files'].items():
        assert M.sha(path) == digest, path
        paths.append(Path(path))
    for name in M.NAMES:
        folder(name).mkdir(parents=True)
        t = ThermalTrajectory(ROOT.parent/'heat_eq_laser_trajectories'/f'Copy of {name}.xdmf')
        mesh = extract_surface(t)
        data = dict(points=t.geometry(), cells=t.topology(), surface_ids=t.surface_node_indices(),
                    time=t.times[-1], previous_time=t.times[-2])
        validate_geometry(data)
        family, interval = name.split('ScanPath_')
        notebook = ROOT.parent/(family+'ScanPath.ipynb')
        laser = NotebookLaser(notebook, family, int(interval))
        np.testing.assert_allclose(t.times, laser.times, atol=1e-14, rtol=0)
        np.savez_compressed(folder(name)/'geometry.npz', **data)
        d = S.case(name, 0)
        np.testing.assert_array_equal(mesh.points, d['points'])
        np.testing.assert_array_equal(mesh.triangles, d['triangles'])
        np.testing.assert_array_equal(mesh.weights, d['area_weights'])
        paths += [notebook, t.xdmf_path, t.h5_path, FEM/'by_trajectory'/name/'surface_comparison.npz',
                  FEM/'by_trajectory'/name/'off.npz', folder(name)/'geometry.npz']
        for level in range(3):
            paths += [M.folder(name)/f'case_{level}.npz', M.folder(name)/f'prior_{level}_all_readings_latent.npz',
                      SOURCE_OFF/'by_trajectory'/name/f'means_{level}.npz', AUDIT/f'oracle_{name}_level{level}.npz']
            for repeat in S.repeats(level):
                paths += [S.original(name, level, repeat).with_suffix(ext) for ext in ['.npz', '.json', '.metrics.csv']]
    M.dump(OUT/'fingerprints.json', dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
        files={str(p.resolve()): M.sha(p) for p in dict.fromkeys(paths)}))
    M.dump(OUT/'frozen_configuration.json', dict(experiment=68, driver=Path(__file__).name,
        names=M.NAMES, ceilings=['reference', 320, 325], previous_representation='all_readings_latent',
        baseline='Output 61 source-on; Output 63 propagation-only arrays used as equality controls',
        candidate='old propagated predecessor + independent FEM boundary response; no old gradient increment',
        independent_response='(M/dt + K)b = f(g_current), zero bottom increment, zero initial increment',
        response_inputs=['volume mesh', 'surface node IDs', 'timestamps', 'prescribed notebook g and material/BC'],
        response_validation_atol_K=1e-8, linear_residual_max=1e-10,
        oracle_identity='new oracle error = output67 propagation discrepancy + replay discrepancy',
        scope='Six paired cases, two reference seeds, one preset primary seed at 320/325 K; no expansion',
        reference_repeats=2, high_ceiling_repeats=1, chains=4, samples_per_chain=M.SAMPLES,
        burn_in=M.BURN, thin=1, evaluation_per_chain=M.EVAL, repeat_seed_offset=M.REPEAT_OFFSET,
        retries=0, current_noise_sd_K=.5, camera_noise_sd_K=.25, gate=M.GATE,
        execution_gate='Numerical checks before ALL current updates; no deterministic-improvement gate. '
                       'Each fit independently flagged by unchanged diagnostics and 900s process budget; no retuning.',
        previous_inference_rerun=False, covariance_changed=False, mask=False, fitting=False,
        baseline_replay='All six primary fits; repeat-1 baseline reused from verified Output61 cache',
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__, arviz=arviz.__version__,
        limitations=['Linear constant-material FEM response only', 'Old surface propagation/covariance retained',
                     'Complete inherited pipeline not claimed camera-only', 'Final draws are pointwise marginals']))
    print('PREPARED geometry-only FEM inputs and frozen controls', flush=True)


def validate():
    checks, rows, components = [], [], []
    prior_checks = pd.read_csv(M.OUT/'prior_verification.csv')
    for name in M.NAMES:
        report = json.loads((folder(name)/'response.json').read_text())
        assert report['passed'] and report['seconds'] <= 900 and report['peak_RSS_MiB'] <= 4096
        raw = M.load(folder(name)/'response.npz'); physical = raw['surface_response']
        saved = M.load(FEM/'by_trajectory'/name/'surface_comparison.npz')
        # Evaluation begins here, after the independent worker has finished.
        np.testing.assert_allclose(physical, saved['fem_response'], rtol=0, atol=1e-8)
        np.testing.assert_allclose(raw['volume_response'], M.load(FEM/'by_trajectory'/name/'off.npz')['response'], rtol=0, atol=1e-8)
        for level in range(3):
            d = S.case(name, level); old = S.prior(name, level); new = candidate(name, level)
            oldoff = M.load(SOURCE_OFF/'by_trajectory'/name/f'means_{level}.npz')['source_off']
            np.testing.assert_array_equal(oldoff, old['prediction_mean']-d['gradient_contribution'])
            np.testing.assert_array_equal(saved['points'][:, :2], d['points'])
            np.testing.assert_array_equal(saved['truth'], d['truth'])
            np.testing.assert_array_equal(saved['area_weights'], d['area_weights'])
            np.testing.assert_array_equal(new['prediction_mean'], oldoff+physical)
            np.testing.assert_array_equal(new['observation_mean'], new['prediction_mean'][d['observation_ids']])
            control = prior_checks[(prior_checks.trajectory == name) & (prior_checks.level == level)
                                   & (prior_checks.arm == 'all_readings_latent')].iloc[0]
            hashes = {}
            for key, column in [('observed_covariance', 'prior_observed_sha256'),
                                ('pred_observed_covariance', 'prior_cross_sha256'),
                                ('prediction_variance', 'prior_diagonal_sha256')]:
                assert M.HELP.digest(old[key]) == control[column]
                np.testing.assert_array_equal(old[key], new[key])
                hashes[key] = M.HELP.digest(new[key])
            oracle = M.load(AUDIT/f'oracle_{name}_level{level}.npz')
            oracle_new = oracle['oracle_off']+physical
            oracle_error = oracle_new-d['truth']
            np.testing.assert_allclose(oracle_error, saved['propagation_discrepancy']+saved['replay_discrepancy'], rtol=0, atol=1e-8)
            np.testing.assert_allclose(oracle_error, saved['propagation_discrepancy'], rtol=0, atol=1e-8)
            np.savez_compressed(folder(name)/f'means_{level}.npz', baseline=old['prediction_mean'],
                physical=new['prediction_mean'], propagation_only=oldoff, physical_increment=physical,
                old_increment=d['gradient_contribution'], oracle_baseline=oracle['oracle_on'], oracle_physical=oracle_new)
            checks.append(dict(trajectory=name, level=level, response_sha256=M.HELP.digest(physical),
                response_vs67_max_abs_K=float(abs(physical-saved['fem_response']).max()),
                oracle_vs67_propagation_max_abs_K=float(abs(oracle_error-saved['propagation_discrepancy']).max()),
                covariance_identical=True, observation_mean_consistent=True, **hashes,
                observation_ids_sha256=M.HELP.digest(d['observation_ids']), camera_sha256=M.HELP.digest(d['camera'])))
            for state, forecasts in [('inferred', [old['prediction_mean'], new['prediction_mean']]),
                                     ('oracle_surface', [oracle['oracle_on'], oracle_new])]:
                for arm, mean in zip(['baseline', 'physical'], forecasts):
                    rows += [dict(trajectory=name, level=level, ceiling_K=float(d['ceiling']),
                                  predecessor=state, arm=arm, **r) for r in S.forecast_metrics(d, mean, name)]
            if level == 0:
                for arm, increment in [('baseline', d['gradient_contribution']), ('physical', physical)]:
                    peak = int(d['truth'].argmax()); maximum = int(increment.argmax())
                    components.append(dict(trajectory=name, arm=arm, maximum_K=float(increment[maximum]),
                        maximum_node=maximum, at_true_peak_K=float(increment[peak]),
                        area_integral_K_m2=float(d['area_weights']@increment)))
    table = pd.DataFrame(checks)
    assert (table.groupby('trajectory').response_sha256.nunique() == 1).all()
    table.to_csv(OUT/'equality_checks.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT/'forecast_metrics.csv', index=False)
    pd.DataFrame(components).to_csv(OUT/'increment_metrics.csv', index=False)
    M.dump(OUT/'numerical_gate.json', dict(passed=True, independent_response_verified=True,
        ceiling_independent=True, oracle_identity_verified=True, covariance_identical=True))
    print('NUMERICAL GATE PASSED', flush=True)


def sample(name, level, repeat=0, replay=False):
    assert json.loads((OUT/'numerical_gate.json').read_text())['passed']
    path = folder(name)/f'replay_{level}' if replay else target(name, level, repeat)
    if path.with_suffix('.json').exists() or path.with_suffix('.npz').exists(): raise FileExistsError(path)
    d = S.case(name, level); b = S.prior(name, level) if replay else candidate(name, level)
    monitor = np.union1d(d['observation_ids'], M.spread_nodes(d['points'], np.ones(len(d['points']), bool), 64))
    start = perf_counter(); latent = []; noisy = []
    for chain in range(4):
        pred, z = M.sample_mixed_noisy_blocks(ids=d['observation_ids'], kinds=d['kinds'], numerical_values=d['values'],
            thresholds=float(d['ceiling']), **b, noise_sd=.5, n_samples=M.SAMPLES, burn_in=M.BURN, thin=1,
            seed=int(d['seed'])+repeat*M.REPEAT_OFFSET+100000*chain)
        latent.append(pred[4]); noisy.append(z)
        print('CHAIN', name, level, repeat, chain, 'replay' if replay else 'physical', flush=True)
    latent = np.asarray(latent); noisy = np.asarray(noisy)
    evaluation = latent[:, np.linspace(0, M.SAMPLES-1, M.EVAL, dtype=int), :]
    pred = M.prediction_from_draws(evaluation.reshape(-1, latent.shape[-1]))
    if replay:
        old = M.load(S.original(name, level).with_suffix('.npz'))
        for key, value in zip(M.SUFFIXES, pred): np.testing.assert_array_equal(value, old[key])
        np.testing.assert_array_equal(noisy, old['noisy_censored_chains'])
        np.testing.assert_array_equal(latent[:, :, monitor], old['monitored_chains'])
        M.dump(path.with_suffix('.json'), dict(exact=True, seconds=perf_counter()-start))
        return
    diagnostics = pd.concat([M.HELP.arviz_diagnostics(noisy, d['observation_ids'][d['kinds'] == 1], 'noisy_censored'),
        M.HELP.arviz_diagnostics(latent[:, :, monitor], monitor, 'latent_temperature')], ignore_index=True)
    classic = M.chain_diagnostics(latent, selected_mask=d['camera'] == d['ceiling'])
    nc = M.chain_diagnostics(noisy)
    common = dict(trajectory=name, level=level, ceiling_K=float(d['ceiling']), arm='physical', repeat=repeat)
    pd.DataFrame([dict(**common, **r) for r in M.metrics(d, pred, name)]).to_csv(path.with_suffix('.metrics.csv'), index=False)
    diagnostics.to_csv(path.with_suffix('.diagnostics.csv'), index=False)
    summary = dict(**common, seconds=perf_counter()-start, peak_RSS_MiB=M.HELP.rss(),
        rank_rhat_max=float(diagnostics.rank_rhat.max()), ess_bulk_min=float(diagnostics.ess_bulk.min()),
        ess_tail_min=float(diagnostics.ess_tail.min()), classical_rhat_max=max(classic['rhat_max'], nc['rhat_max']),
        finite=bool(np.isfinite(diagnostics[['rank_rhat', 'ess_bulk', 'ess_tail']]).all().all()),
        current_seed=int(d['seed'])+repeat*M.REPEAT_OFFSET, constraints=noisy.shape[-1],
        noise_sd_K=.5, prior_cross_sha256=M.HELP.digest(b['pred_observed_covariance']))
    summary['passed'] = bool(summary['finite'] and all(summary[k] <= M.GATE[k] for k in ['rank_rhat_max', 'classical_rhat_max'])
        and summary['ess_bulk_min'] >= 400 and summary['ess_tail_min'] >= 400
        and summary['seconds'] <= 900 and summary['peak_RSS_MiB'] <= 4096)
    np.savez_compressed(path.with_suffix('.npz'), **dict(zip(M.SUFFIXES, pred)), evaluation_chains=evaluation,
        monitored_chains=latent[:, :, monitor], monitor_ids=monitor, noisy_censored_chains=noisy,
        constrained_ids=d['observation_ids'][d['kinds'] == 1])
    M.dump(path.with_suffix('.json'), summary)
    print('FIT', summary, flush=True)


def launch(phase, name, level, repeat=0):
    log = folder(name)/f'{phase}_{level}_{repeat}.log'
    command = [sys.executable, str(Path(__file__).resolve()), '--phase', phase, '--name', name,
               '--level', str(level), '--repeat', str(repeat)]
    start = perf_counter()
    with log.open('w') as f:
        process = subprocess.Popen(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=900); timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            code = None; timed_out = True
    result = dict(phase=phase, trajectory=name, level=level, repeat=repeat, returncode=code,
                  timed_out=timed_out, wall_seconds=perf_counter()-start, log=str(log.relative_to(OUT)))
    print('PROCESS', result, flush=True)
    return result


def run_gp():
    assert json.loads((OUT/'numerical_gate.json').read_text())['passed']
    start = perf_counter(); executions = []
    for name in M.NAMES:
        for level in range(3):
            result = launch('replay', name, level); executions.append(result)
            M.dump(OUT/'execution_status.json', executions)
            if result['returncode'] != 0: raise RuntimeError('Baseline replay failed; stop before candidate fits')
    for name in M.NAMES:
        for level in range(3):
            for repeat in S.repeats(level):
                executions.append(launch('fit', name, level, repeat))
                M.dump(OUT/'execution_status.json', executions)
    M.dump(OUT/'run_timing.json', dict(seconds=perf_counter()-start))
    collect()


def plot(name, level):
    d = S.case(name, level); means = M.load(folder(name)/f'means_{level}.npz')
    sections = pd.read_csv(ROOT/'outputs/by_experiment/56_within_step_integration/frozen_sections.csv')
    predictions = [M.load(S.original(name, level).with_suffix('.npz')), M.load(target(name, level).with_suffix('.npz'))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, constrained_layout=True)
    for col, (arm, pred) in enumerate(zip(['baseline', 'physical'], predictions)):
        for row, (section, g) in enumerate(sections[sections.trajectory == name].groupby('section', sort=False)):
            g = g.sort_values('along_mm'); ids = g.node.to_numpy(int); x = g.along_mm.to_numpy(); ax = axes[row, col]
            ax.plot(x, d['truth'][ids], 'k', label='Native truth')
            ax.plot(x, means[arm][ids], '--', color='#D55E00', label='Forecast')
            ax.plot(x, pred['mean'][ids], color='#0072B2', label='Posterior')
            ax.fill_between(x, pred['lower'][ids], pred['upper'][ids], color='#0072B2', alpha=.2, label='Pointwise 95%')
            ax.axhline(float(d['ceiling']), ls=':', color='#777777', label='Ceiling')
            ax.set(title=arm+' | '+('Along' if row == 0 else 'Across'), xlabel='Distance from fixed peak (mm)', ylabel='K')
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{name}; ceiling {float(d["ceiling"]):.3f} K\nOnly current mean increment replaced; primary seed; pointwise marginal intervals')
    fig.savefig(OUT/f'cross_sections_{name}_level_{level}.png', dpi=150); plt.close(fig)
    errors = [means[a]-d['truth'] for a in ['baseline', 'physical']]+[p['mean']-d['truth'] for p in predictions]
    tri = mtri.Triangulation(d['points'][:, 0]*1000, d['points'][:, 1]*1000, d['triangles'])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    limit = max(abs(e).max() for e in errors)
    for ax, err, label in zip(axes.flat, errors, ['Old forecast', 'Physical forecast', 'Old posterior', 'Physical posterior']):
        im = ax.tripcolor(tri, err, shading='gouraud', cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set(title=label, xlabel='x (mm)', ylabel='y (mm)', aspect='equal')
    fig.colorbar(im, ax=list(axes.flat), label='Estimate minus truth (K)', shrink=.8)
    fig.suptitle(f'{name}; ceiling {float(d["ceiling"]):.3f} K; shared scale')
    fig.savefig(OUT/f'signed_errors_{name}_level_{level}.png', dpi=150); plt.close(fig)


def collect():
    tables, summaries = [], []
    for name in M.NAMES:
        for level in range(3):
            for repeat in S.repeats(level):
                path = target(name, level, repeat)
                if not path.with_suffix('.json').exists(): continue
                for arm, source in [('baseline', S.original(name, level, repeat)), ('physical', path)]:
                    s = json.loads(source.with_suffix('.json').read_text()); s.update(arm=arm, reused=arm == 'baseline')
                    summaries.append(s)
                    df = pd.read_csv(source.with_suffix('.metrics.csv')); df['arm'] = arm; df['reliable'] = s['passed']
                    tables.append(df)
            if target(name, level).with_suffix('.npz').exists(): plot(name, level)
    if not tables: return
    data = pd.concat(tables, ignore_index=True)
    data.to_csv(OUT/'posterior_metrics.csv', index=False)
    pd.DataFrame(summaries).to_csv(OUT/'convergence_runtime.csv', index=False)
    differences = []
    for keys, group in data.groupby(['trajectory', 'level', 'repeat', 'region']):
        g = group.set_index('arm')
        differences.append(dict(zip(['trajectory', 'level', 'repeat', 'region'], keys),
            pair_reliable=bool(g.reliable.all()), **{k:float(g.loc['physical', k]-g.loc['baseline', k])
                for k in ['rmse_K', 'crps_K', 'bias_K', 'coverage95', 'width95_K', 'true_peak_error_K']}))
    pd.DataFrame(differences).to_csv(OUT/'paired_changes.csv', index=False)
    primary = data[(data.repeat == 0) & (data.region == 'hot1pct_area')]
    primary.to_csv(OUT/'compact_hot_area.csv', index=False)
    repeats = data[(data.level == 0)].pivot_table(index=['trajectory', 'arm', 'region'], columns='repeat',
        values=['rmse_K', 'crps_K', 'true_peak_error_K', 'coverage95'])
    repeats.to_csv(OUT/'reference_repeat_comparison.csv')
    for path, digest in json.loads((OUT/'fingerprints.json').read_text())['files'].items():
        assert M.sha(path) == digest, path
    M.dump(OUT/'verification.json', dict(numerical_checks_pass=True,
        baseline_primary_replays_exact=sum((folder(n)/f'replay_{l}.json').exists() for n in M.NAMES for l in range(3)),
        candidate_fits_completed=sum(not s['reused'] for s in summaries),
        all_completed_fits_pass=all(s['passed'] for s in summaries), historical_inputs_unchanged=True,
        covariance_changed=False, previous_inference_rerun=False, source_independent_of_temperature=True,
        oracle_truth_evaluation_only=True, candidate_fit_seconds=sum(s['seconds'] for s in summaries if not s['reused']),
        runtime_note='Fit times exclude compressed-cache write; baseline fit timings are historical. Process wall times in execution_status.json.'))


if __name__ == '__main__':
    for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        if os.environ.get(k) != '1': raise RuntimeError('Set '+k+'=1')
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['prepare', 'validate', 'run-gp', 'replay', 'fit', 'collect'], required=True)
    p.add_argument('--name', choices=M.NAMES); p.add_argument('--level', type=int, choices=[0, 1, 2])
    p.add_argument('--repeat', type=int, choices=[0, 1], default=0)
    a = p.parse_args()
    if a.phase == 'prepare': prepare()
    elif a.phase == 'validate': validate()
    elif a.phase == 'run-gp': run_gp()
    elif a.phase == 'collect': collect()
    else: sample(a.name, a.level, a.repeat, replay=a.phase == 'replay')
