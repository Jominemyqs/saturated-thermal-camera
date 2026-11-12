"""Bounded notebook/input equivalence and cached oracle reconciliation; no fitting."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prescribed_laser_audit import NotebookLaser, flux_metrics
from src.thermal_trajectory import ThermalTrajectory
from src.native_surface import extract_surface, hottest_area_weights, quadrature
from src.matched_projection import MatchedHeatOperator

BASE = ROOT / 'outputs/by_experiment'
OUT = BASE / '66_prescribed_laser_input_audit'
DATA = ROOT.parent / 'heat_eq_laser_trajectories'
NAMES = ['DiagonalScanPath_8', 'SpiralScanPath_13']
FILES = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = path.resolve()
    FILES[str(path)] = dict(sha256=sha(path), bytes=path.stat().st_size)
    return path


def load(path):
    with np.load(record(path)) as f:
        return {k: f[k] for k in f.files}


def save_json(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2)+'\n')


def plot_flux(name, mesh, frames):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    tri = Triangulation(mesh.points[:, 0]*1000, mesh.points[:, 1]*1000, mesh.triangles)
    scale = max(max(f['g'].max(), f['kg'].max()) for f in frames)/1e6
    difference = max(abs(f['kg']-f['g']).max() for f in frames)/1e6
    centers = np.array([f['focus'] for f in frames])*1000
    for row, frame in enumerate(frames):
        for col, (values, title) in enumerate([
            (frame['g'], 'Prescribed g'), (frame['kg'], 'k * exported gradient'),
            (frame['kg']-frame['g'], 'Exported minus prescribed')]):
            ax = axes[row, col]
            im = ax.tripcolor(tri, values/1e6, shading='gouraud',
                              cmap='RdBu_r' if col == 2 else 'viridis',
                              vmin=-difference if col == 2 else 0,
                              vmax=difference if col == 2 else scale)
            ax.plot(*(frame['focus']*1000), 'kx', ms=6)
            ax.set(xlim=(centers[:, 0].min()-.5, centers[:, 0].max()+.5),
                   ylim=(centers[:, 1].min()-.5, centers[:, 1].max()+.5),
                   xlabel='x (mm)', ylabel='y (mm)', aspect='equal',
                   title=f"{title}\nt={frame['time']:.4f} s")
            fig.colorbar(im, ax=ax, label='MW / m2', shrink=.75)
    fig.suptitle(name+': raw native-node flux comparison; x marks prescribed focus\n'
                 'Same scales across frames; no cutoff, gamma, fitted alignment or normalization')
    fig.savefig(OUT/f'flux_comparison_{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def reconcile(name, mesh, trajectory, options, oracle_rows, checks):
    old = load(BASE/f'51_native_forecast_attribution/by_trajectory/{name}/components.npz')
    truth = trajectory.read_temperature(-1, surface_only=True)
    previous_truth = trajectory.read_temperature(-2, surface_only=True)
    gradient = trajectory.read_heat_flux_z(-1, surface_only=True)
    for key, array in [('points', mesh.points), ('triangles', mesh.triangles),
                       ('weights', mesh.weights), ('truth', truth),
                       ('previous_truth', previous_truth), ('flux', gradient)]:
        np.testing.assert_array_equal(old[key], array)
    op = MatchedHeatOperator(mesh,
        np.linspace(mesh.points[:, 0].min(), mesh.points[:, 0].max(), int(options['nx'])),
        np.linspace(mesh.points[:, 1].min(), mesh.points[:, 1].max(), int(options['ny'])))
    physics = dict(dt=float(trajectory.times[-1]-trajectory.times[-2]),
                   diffusivity=options['diffusivity'], cooling_rate=options['cooling_rate'])
    source = options['source_coupling']*np.where(
        gradient >= options['source_flux_threshold'], gradient, 0)*physics['dt']
    np.testing.assert_array_equal(source, old['source'])
    reference_arrays = None
    for level in range(3):
        c52 = load(BASE/f'52_native_higher_ceiling/by_trajectory/{name}/level_{level}.npz')
        c60 = load(BASE/f'60_previous_unpinning_audit/by_trajectory/{name}/level_{level}_summaries.npz')
        c63 = load(BASE/f'63_native_source_on_off/by_trajectory/{name}/means_{level}.npz')
        ambient = float(c52['ambient'])
        np.testing.assert_array_equal(c60['points'], mesh.points)
        np.testing.assert_array_equal(c60['area_weights'], mesh.weights)
        np.testing.assert_array_equal(c60['truth'], truth)
        np.testing.assert_array_equal(c60['previous_truth'], previous_truth)
        arrays = dict(truth=truth, source=source)
        for state, values in [('oracle', previous_truth), ('latest_inferred', c60['previous_mean_all_readings_latent'])]:
            off = ambient+op.propagate(values-ambient, 'native', floor=True, **physics)
            on = off+source
            needed = truth-off
            np.testing.assert_allclose(on-truth, source-needed, atol=1e-12, rtol=0)
            arrays.update({state+'_off': off, state+'_on': on, state+'_needed': needed})
            for mode, field in [('off', off), ('on', on)]:
                for region, w in [('whole_area', mesh.weights), ('hot1pct_area', hottest_area_weights(truth, mesh.weights))]:
                    error = field-truth
                    peak = int(truth.argmax())
                    oracle_rows.append(dict(trajectory=name, level=level, ceiling_K=float(c52['ceiling']),
                        predecessor=state, increment=mode, region=region,
                        rmse_K=float(np.sqrt(np.sum(w*error**2)/w.sum())), bias_K=float(np.sum(w*error)/w.sum()),
                        true_peak_K=float(truth[peak]), forecast_at_true_peak_K=float(field[peak]),
                        true_peak_error_K=float(error[peak]), increment_at_true_peak_K=float(source[peak]),
                        needed_at_true_peak_K=float(needed[peak])))
        for key, oldkey in [('oracle_off', 'oracle_without_source'), ('oracle_on', 'oracle_forecast'),
                            ('oracle_needed', 'oracle_needed')]:
            np.testing.assert_array_equal(arrays[key], old[oldkey])
        for key, cached in [('latest_inferred_off', c63['source_off']),
                             ('latest_inferred_on', c63['source_on'])]:
            np.testing.assert_allclose(arrays[key], cached, atol=1e-12, rtol=0)
        checks.append(dict(trajectory=name, level=level, oracle51_exact=True,
            source51_exact=True, latest63_on_max_abs=float(abs(arrays['latest_inferred_on']-c63['source_on']).max()),
            latest63_off_max_abs=float(abs(arrays['latest_inferred_off']-c63['source_off']).max()),
            operator_grid_shape=list(op.shape), **physics))
        np.savez_compressed(OUT/f'oracle_{name}_level{level}.npz', **arrays)
        if level == 0:
            reference_arrays = arrays
    path = record(BASE/f'49_matched_native_grid_gp/native_diagnostics/cross_sections_{name}.csv')
    sections = pd.read_csv(path)
    sections = sections[sections.model == 'native_C']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    labels = dict(truth='Current truth', oracle_off='Oracle: propagation only',
                  oracle_on='Oracle: complete forecast', latest_inferred_off='Latest inferred: propagation only',
                  latest_inferred_on='Latest inferred: complete forecast', source='Added gradient increment',
                  oracle_needed='Required residual: oracle', latest_inferred_needed='Required residual: inferred')
    for col, (label, section) in enumerate(sections.groupby('section', sort=False)):
        ids = section.node.to_numpy(int); x = section.along_mm.to_numpy()
        for key, style in [('truth', 'k-'), ('oracle_off', 'g--'), ('oracle_on', 'g-'),
                           ('latest_inferred_off', 'b--'), ('latest_inferred_on', 'b-')]:
            axes[0, col].plot(x, reference_arrays[key][ids], style, label=labels[key])
        for key, style in [('source', 'r-'), ('oracle_needed', 'g--'), ('latest_inferred_needed', 'b--')]:
            axes[1, col].plot(x, reference_arrays[key][ids], style, label=labels[key])
        for row in range(2):
            axes[row, col].set(xlabel='Distance from true peak (mm)', ylabel='K', title=label)
            axes[row, col].grid(alpha=.15); axes[row, col].legend(fontsize=8)
        axes[1, col].axhline(0, color='grey', lw=.5)
    for row in range(2):
        limits = [ax.get_ylim() for ax in axes[row]]
        for ax in axes[row]:
            ax.set_ylim(min(a[0] for a in limits), max(a[1] for a in limits))
    fig.suptitle(name+': latest full-latent inferred predecessor (reference) versus oracle\n'
                 'Deterministic only; required residual is NOT heating; historical section directions retained')
    fig.savefig(OUT/f'oracle_sections_{name}.png', dpi=150); plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError('Refusing to overwrite an existing audit')
    start = perf_counter()
    status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for path in sorted({*ROOT.glob('src/*.py'), *ROOT.glob('scripts/*.py'), *ROOT.glob('tests/*.py')}):
        record(path)
    for family in ['Diagonal', 'Spiral', 'Horizontal', 'Vertical']:
        record(ROOT.parent/(family+'ScanPath.ipynb'))
    options = json.loads(record(BASE/'61_unpinning_current_update/frozen_configuration.json').read_text())['inherited_settings']
    OUT.mkdir()
    rows, frames_metadata, integral_rows, provenance, oracle_rows, checks = [], [], [], [], [], []
    clocks = []
    for name in NAMES:
        family, interval = name.split('ScanPath_')
        notebook = ROOT.parent/(family+'ScanPath.ipynb')
        laser = NotebookLaser(notebook, family, int(interval))
        for cell in [3, 5, 6]:
            source = laser.sources[cell]
            (OUT/f'{family}_cell{cell}_source.txt').write_text(source)
        trajectory = ThermalTrajectory(record(DATA/f'Copy of {name}.xdmf'))
        record(trajectory.h5_path)
        mesh = extract_surface(trajectory)
        assert laser.frame_count == trajectory.n_times
        np.testing.assert_allclose(laser.times, trajectory.times, atol=1e-14, rtol=0)
        np.testing.assert_allclose(trajectory.geometry().min(0), [-.0075, -.005, -.001], atol=1e-15)
        np.testing.assert_allclose(trajectory.geometry().max(0), [.0075, .005, .001], atol=1e-15)
        clocks.append(dict(trajectory=name, frames=laser.frame_count, last_time_s=float(trajectory.times[-1]),
            maximum_clock_difference_s=float(abs(laser.times-trajectory.times).max()),
            solver_dt_s=laser.parameters['dt'], exports_per_step=1,
            top_nodes=len(mesh.points), top_triangles=len(mesh.triangles), area_m2=float(mesh.weights.sum()),
            path_points=len(laser.path), path_length_m=float(laser.distance)))
        units = dict(dt='s', velocity='m/s', sigma='m', power='W', rho='kg/m3', Cp='J/(kg K)',
                     k='W/(m K)', T_0='K', length_x='m', length_y='m', width='m', depth='m', height='m', dt_path='rad (path discretization)')
        for key in units:
            if key not in laser.parameters:
                continue
            cell, first, last = laser.locations[key]
            provenance.append(dict(trajectory=name, quantity=key, value=laser.parameters[key], units=units[key],
                independent_of_current_temperature=True, source=f'{notebook.name}:cell{cell}:L{first}-{last}',
                dataset_link='Adrienne Nov 5 attribution; complete saved-clock and geometry-bounds checks; no FEM reproduction',
                qualification='SI interpretation; dataset lacks explicit unit attributes'))
        frames = []
        for index in [trajectory.n_times-2, trajectory.n_times-1]:
            time = float(trajectory.times[index]); focus = laser.focus(time)
            g = laser.evaluate(mesh.points, time)
            gradient = trajectory.read_heat_flux_z(index, surface_only=True)
            kg = laser.parameters['k']*gradient
            near = np.linalg.norm(mesh.points-focus, axis=1) <= 3*laser.parameters['sigma']
            for region, weights in [('whole_surface', mesh.weights), ('within_3sigma_of_prescribed_focus', mesh.weights*near)]:
                rows.append(dict(trajectory=name, frame=index, time_s=time, region=region,
                                 **flux_metrics(g, kg, weights)))
            nodeg = int(g.argmax()); nodekg = int(kg.argmax())
            frames_metadata.append(dict(trajectory=name, frame=index, time_s=time,
                focus_x_m=float(focus[0]), focus_y_m=float(focus[1]),
                prescribed_native_max_W_m2=float(g[nodeg]), exported_native_max_W_m2=float(kg[nodekg]),
                peak_ratio=float(kg[nodekg]/g[nodeg]),
                native_peak_separation_um=float(np.linalg.norm(mesh.points[nodeg]-mesh.points[nodekg])*1e6),
                exported_peak_to_prescribed_focus_um=float(np.linalg.norm(mesh.points[nodekg]-focus)*1e6),
                prescribed_focus_to_nearest_node_um=float(np.linalg.norm(mesh.points[nodeg]-focus)*1e6),
                prescribed_peak_node=nodeg, exported_peak_node=nodekg))
            for refinement in [0, 1, 2, 3]:
                points, weights, bary = quadrature(mesh, refinement=refinement)
                gq = laser.evaluate(points, time)
                kgq = np.einsum('qv,tv->tq', bary, kg[mesh.triangles]).ravel()
                integral_rows.append(dict(trajectory=name, frame=index, refinement=refinement,
                    quadrature_points=len(points), prescribed_integral_W=float(np.sum(weights*gq)),
                    exported_P1_integral_W=float(np.sum(weights*kgq))))
            frames.append(dict(g=g, kg=kg, focus=focus, time=time))
            np.savez_compressed(OUT/f'flux_{name}_frame{index}.npz', points=mesh.points,
                triangles=mesh.triangles, area_weights=mesh.weights, raw_gradient=gradient,
                prescribed_g=g, conductivity_gradient=kg, focus=focus, time=time)
        plot_flux(name, mesh, frames)
        reconcile(name, mesh, trajectory, options, oracle_rows, checks)
    for filename, data in [('flux_metrics.csv', rows), ('flux_peaks.csv', frames_metadata),
                           ('flux_integral_quadrature.csv', integral_rows), ('provenance_parameters.csv', provenance),
                           ('oracle_forecast_metrics.csv', oracle_rows), ('oracle_equality_checks.csv', checks)]:
        pd.DataFrame(data).to_csv(OUT/filename, index=False)
    save_json('clock_checks.json', clocks)
    save_json('frozen_configuration.json', dict(output=66, driver=Path(__file__).name,
        names=NAMES, frames='last two saved frames', near_source_radius='3 * notebook sigma',
        surrogate_parameters={k: options[k] for k in ['diffusivity', 'cooling_rate', 'source_coupling', 'source_flux_threshold', 'nx', 'ny']},
        internal_propagation_grid=[601, 401], fit=False, gp=False, fem=False,
        raw_flux_comparison='g versus k*HeatFluxZ; no threshold, gamma, alignment or renormalization',
        oracle_reconciliation='Latest 61/63 coefficients and 60 full-latent means; cached51 oracle checked at all three ceilings'))
    for path, info in FILES.items():
        assert sha(Path(path)) == info['sha256'], path
    save_json('fingerprints.json', dict(revision=revision, git_status_before=status, files=FILES))
    save_json('verification.json', dict(inputs_and_existing_sources_unchanged=True,
        recorded_files=len(FILES), complete_clock_checks=clocks, oracle_checks=checks,
        runtime_seconds=perf_counter()-start, gp_inference=False, fem_execution=False, fitting=False,
        qualification='Notebook path/material/export provenance corroborated, not bitwise FEM reproduction'))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(frames_metadata).to_string(index=False))
    print('Audit output:', OUT)


if __name__ == '__main__':
    main()
