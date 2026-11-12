"""Two original-FEM transitions, with a prospectively frozen source-on gate."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.thermal_trajectory import ThermalTrajectory
from src.native_surface import extract_surface, hottest_area_weights
from src.prescribed_laser_audit import NotebookLaser
from src.fem_replay import volume_weights, error_metrics, replay_passes, decomposition, build_problem, solve_problem

OUT = ROOT/'outputs/by_experiment/67_one_step_fem_response'
BASE66 = ROOT/'outputs/by_experiment/66_prescribed_laser_input_audit'
DATA = ROOT.parent/'heat_eq_laser_trajectories'
PREFIX = ROOT/'tmp/fem_env'
MAMBA = Path.home()/'miniforge3/bin/mamba'
NAMES = ['DiagonalScanPath_8', 'SpiralScanPath_13']
GATE = dict(rmse_K=1e-4, max_abs_K=1e-3, relative_linear_residual=1e-10,
            stage_seconds=900, peak_rss_MiB=4096)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def load(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def child_environment():
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               XDG_CACHE_HOME=str(ROOT/'tmp/fem_cache'), MPLCONFIGDIR=str(ROOT/'tmp/matplotlib'),
               PYTHONPYCACHEPREFIX=str(ROOT/'tmp/fem_pycache'),
               DIJITSO_CACHE_DIR=str(ROOT/'tmp/fem_dijitso'))
    return env


def peak_rss():
    scale = 1024**2 if platform.system() == 'Darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/scale


def worker(name, stage):
    import dolfin as df
    start = perf_counter()
    folder = OUT/'by_trajectory'/name
    data = load(folder/'input.npz')
    family, interval = name.split('ScanPath_')
    laser = NotebookLaser(ROOT.parent/(family+'ScanPath.ipynb'), family, int(interval))
    df.parameters['form_compiler']['cache_dir'] = str(ROOT/'tmp/fem_ffc')
    problem = build_problem(data['points'], data['cells'], data['previous'], laser, float(data['time']))
    surface = data['surface_ids']
    record = dict(trajectory=name, stage=stage, dolfin_version=df.__version__,
        current_time_s=float(data['time']), previous_time_s=float(data['previous_time']),
        solver_dt_s=laser.parameters['dt'], material_parameters={k: laser.parameters[k] for k in ['rho', 'Cp', 'k', 'T_0']},
        form_compiler_parameters={key: df.parameters['form_compiler'][key]
                                  for key in df.parameters['form_compiler'].keys()},
        dofs=problem['V'].dim(), cells=len(data['cells']), top_nodes=len(surface),
        top_area_m2=problem['top_area_m2'], bottom_area_m2=problem['bottom_area_m2'],
        assembled_volume_m3=problem['assembled_volume_m3'], volume_sum_rtol=problem['volume_sum_rtol'],
        assembled_boundary_input_W=problem['assembled_boundary_input_W'],
        input_sha256=sha(folder/'input.npz'), default_linear_algebra_backend=str(df.parameters['linear_algebra_backend']))
    result, residual = solve_problem(problem, stage)
    record['relative_linear_residual'] = residual
    arrays = dict(temperature=result, dof_to_original_node=problem['dof_nodes'])
    if stage == 'on':
        metrics = dict(volume=error_metrics(result, data['truth'], data['volume_weights'], laser.parameters['T_0']),
                       surface=error_metrics(result[surface], data['truth'][surface], data['area_weights'], laser.parameters['T_0']))
        peak = surface[int(np.argmax(data['truth'][surface]))]
        record.update(metrics=metrics, true_peak_error_K=float(result[peak]-data['truth'][peak]),
                      numerical_gate=replay_passes(metrics, GATE) and residual <= GATE['relative_linear_residual'])
    else:
        on = load(folder/'on.npz')['temperature']
        # Independent zero-bottom response solve verifies that on-off is the load response.
        response, response_residual = solve_problem(problem, 'response')
        record.update(response_relative_linear_residual=response_residual,
                      superposition_max_abs_K=float(abs(on-result-response).max()))
        np.testing.assert_allclose(on-result, response, rtol=0, atol=1e-8)
        arrays.update(response=response)
        record['numerical_gate'] = (residual <= GATE['relative_linear_residual']
                                    and response_residual <= GATE['relative_linear_residual'])
    record.update(runtime_seconds=perf_counter()-start, peak_rss_MiB=peak_rss())
    record['passed'] = bool(record['numerical_gate'] and record['runtime_seconds'] <= GATE['stage_seconds']
                          and record['peak_rss_MiB'] <= GATE['peak_rss_MiB'])
    np.savez_compressed(folder/(stage+'.npz'), **arrays)
    write_json(folder/(stage+'.json'), record)
    print(json.dumps(record, indent=2))


def launch(name, stage):
    log = OUT/'by_trajectory'/name/(stage+'.log')
    command = [str(MAMBA), 'run', '-p', str(PREFIX), 'python', str(Path(__file__).resolve()),
               '--worker', name, '--stage', stage]
    start = perf_counter()
    with log.open('w') as f:
        process = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdout=f,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=GATE['stage_seconds'])
            result = dict(returncode=returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            result = dict(returncode=None, timed_out=True)
    result.update(trajectory=name, stage=stage, wall_seconds=perf_counter()-start,
                  log=str(log.relative_to(OUT)))
    path = OUT/'by_trajectory'/name/(stage+'.json')
    result['passed'] = bool(result['returncode'] == 0 and path.exists() and json.loads(path.read_text())['passed'])
    return result


def summarize(decomposed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    metrics, peaks, stages = [], [], []
    for name in NAMES:
        folder = OUT/'by_trajectory'/name
        data = load(folder/'input.npz'); ids = data['surface_ids']; truth = data['truth'][ids]
        for stage in ['on', 'off']:
            path = folder/(stage+'.json')
            if path.exists():
                report = json.loads(path.read_text())
                stages.append({k: report[k] for k in ['trajectory', 'stage', 'runtime_seconds', 'peak_rss_MiB',
                    'relative_linear_residual', 'assembled_boundary_input_W', 'passed']})
        onpath = folder/'on.npz'
        if not onpath.exists():
            continue
        on = load(onpath)['temperature'][ids]
        fields = dict(fem_on=on, truth=truth, replay_discrepancy=on-truth)
        if decomposed:
            off = load(folder/'off.npz')['temperature'][ids]
            old = load(BASE66/f'oracle_{name}_level0.npz')
            np.testing.assert_array_equal(old['truth'], truth)
            fields.update(decomposition(old['oracle_off'], old['source'], off, on, truth))
            fields.update(fem_off=off, old_off=old['oracle_off'], old_increment=old['source'], old_on=old['oracle_on'])
        w = data['area_weights']; hot = hottest_area_weights(truth, w)
        for region, weights in [('whole_area', w), ('hot1pct_area', hot)]:
            for key, value in fields.items():
                if key in ['truth', 'old_increment', 'fem_response']:
                    continue
                error = value-truth if key in ['fem_on', 'fem_off', 'old_off', 'old_on'] else value
                metrics.append(dict(trajectory=name, region=region, term=key,
                    bias_K=float(np.sum(weights*error)/weights.sum()),
                    rmse_K=float(np.sqrt(np.sum(weights*error**2)/weights.sum()))))
        peak = int(truth.argmax())
        peaks.append(dict(trajectory=name, peak_node=peak, **{key: float(value[peak]) for key, value in fields.items()}))
        np.savez_compressed(folder/'surface_comparison.npz', **fields, points=data['points'][ids],
                            triangles=data['triangles'], area_weights=w)
        if decomposed:
            sections = pd.read_csv(ROOT/f'outputs/by_experiment/49_matched_native_grid_gp/native_diagnostics/cross_sections_{name}.csv')
            sections = sections[sections.model == 'native_C']
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
            for col, (label, section) in enumerate(sections.groupby('section', sort=False)):
                nodes = section.node.to_numpy(int); x = section.along_mm.to_numpy()
                for key in ['truth', 'fem_on', 'fem_off', 'old_off', 'old_on']:
                    axes[0, col].plot(x, fields[key][nodes], label=key)
                for key in ['old_increment', 'fem_response', 'propagation_discrepancy', 'increment_discrepancy']:
                    axes[1, col].plot(x, fields[key][nodes], label=key)
                for row in range(2):
                    axes[row, col].set(xlabel='Distance from true peak (mm)', ylabel='K', title=label)
                    axes[row, col].legend(fontsize=8); axes[row, col].grid(alpha=.15)
            fig.suptitle(name+': one-step full-volume oracle diagnostic; no GP\nHistorical section nodes retained')
            fig.savefig(OUT/f'sections_{name}.png', dpi=150, bbox_inches='tight'); plt.close(fig)
            keys = ['propagation_discrepancy', 'increment_discrepancy', 'total_error']
        else:
            keys = ['replay_discrepancy']
        fig, axes = plt.subplots(1, len(keys), figsize=(5*len(keys), 4), constrained_layout=True, squeeze=False)
        xy = data['points'][ids, :2]*1000
        tri = Triangulation(xy[:, 0], xy[:, 1], data['triangles'])
        scale = max(max(abs(fields[key])) for key in keys) or 1e-12
        for ax, key in zip(axes[0], keys):
            im = ax.tripcolor(tri, fields[key], shading='gouraud', cmap='RdBu_r', vmin=-scale, vmax=scale)
            ax.set(title=key, xlabel='x (mm)', ylabel='y (mm)', aspect='equal')
            fig.colorbar(im, ax=ax, label='K')
        fig.suptitle(name+' | matched surface nodes')
        fig.savefig(OUT/f'discrepancies_{name}.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    pd.DataFrame(metrics).to_csv(OUT/'surface_metrics.csv', index=False)
    pd.DataFrame(peaks).to_csv(OUT/'true_peak_decomposition.csv', index=False)
    pd.DataFrame(stages).to_csv(OUT/'runtime_and_solver.csv', index=False)


def main():
    if OUT.exists():
        raise FileExistsError('Refusing to overwrite an existing experiment')
    if not (PREFIX/'bin/python').exists():
        raise FileNotFoundError('Prepare the isolated legacy FEniCS environment first')
    OUT.mkdir()
    start = perf_counter()
    configuration = dict(experiment=67, driver=Path(__file__).name, names=NAMES,
        scope='Last stored transition only; source-on gate for BOTH cases before any source-off',
        gate=GATE, units='K; physical volume/area weights; nodal maximum',
        gate_note='No relaxation after seeing replay error', coefficient_fit=False, gp=False,
        solver='Supplied weak form, legacy DOLFIN solve defaults, UserExpression degree=2',
        quadrature='Original FFC automatic estimate, not tuned against saved truth',
        previous_state='True full-volume predecessor; not a deployable state estimate',
        source='Original prescribed g evaluated at current saved time; no gradient in FEM RHS')
    write_json(OUT/'frozen_configuration.json', configuration)
    paths = [Path(__file__).resolve(), ROOT/'src/fem_replay.py', ROOT/'src/prescribed_laser_audit.py',
             ROOT/'src/thermal_trajectory.py', ROOT/'src/native_surface.py',
             ROOT/'tests/test_fem_replay.py', BASE66/'frozen_configuration.json']
    for name in NAMES:
        folder = OUT/'by_trajectory'/name; folder.mkdir(parents=True)
        family, interval = name.split('ScanPath_')
        notebook = ROOT.parent/(family+'ScanPath.ipynb')
        laser = NotebookLaser(notebook, family, int(interval))
        t = ThermalTrajectory(DATA/f'Copy of {name}.xdmf'); mesh = extract_surface(t)
        np.testing.assert_allclose(laser.times, t.times, atol=1e-14, rtol=0)
        xyz, cells = t.geometry(), t.topology()
        previous, truth = t.read_temperature(-2), t.read_temperature(-1)
        np.testing.assert_allclose(previous[np.isclose(xyz[:, 2], xyz[:, 2].min())], laser.parameters['T_0'], atol=1e-10, rtol=0)
        np.savez_compressed(folder/'input.npz', points=xyz, cells=cells, previous=previous, truth=truth,
                            time=t.times[-1], previous_time=t.times[-2], surface_ids=t.surface_node_indices(),
                            triangles=mesh.triangles, area_weights=mesh.weights, volume_weights=volume_weights(xyz, cells))
        paths.extend([notebook, t.xdmf_path, t.h5_path, BASE66/f'oracle_{name}_level0.npz',
            ROOT/f'outputs/by_experiment/49_matched_native_grid_gp/native_diagnostics/cross_sections_{name}.csv'])
    fingerprints = {str(p): sha(p) for p in paths}
    write_json(OUT/'fingerprints.json', dict(revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'],cwd=ROOT,text=True).strip(),
               git_status=subprocess.check_output(['git', 'status', '--short'],cwd=ROOT,text=True), files=fingerprints))
    explicit = subprocess.check_output([str(MAMBA), 'list', '-p', str(PREFIX), '--explicit'], env=child_environment(), text=True)
    (OUT/'environment_explicit.txt').write_text(explicit)
    executions = []
    for name in NAMES:
        executions.append(launch(name, 'on'))
        write_json(OUT/'execution_status.json', executions)
    gate_passed = all(e['passed'] for e in executions)
    if gate_passed:
        for name in NAMES:
            executions.append(launch(name, 'off'))
            write_json(OUT/'execution_status.json', executions)
    decomposed = gate_passed and all(e['passed'] for e in executions)
    summarize(decomposed)
    for path, digest in fingerprints.items():
        assert sha(path) == digest, path
    write_json(OUT/'verification.json', dict(source_on_gate_passed=gate_passed,
        decomposition_completed=decomposed, inputs_unchanged=True, executions=executions,
        total_wall_seconds=perf_counter()-start, no_GP=True, no_fitting=True, historical_baselines_unchanged=True))
    print(json.dumps(dict(source_on_gate_passed=gate_passed, decomposition_completed=decomposed, output=str(OUT)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=NAMES)
    parser.add_argument('--stage', choices=['on', 'off'])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.stage)
    else:
        main()
