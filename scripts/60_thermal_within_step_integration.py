"""Experiment/output 56: deterministic within-step gradient integration gate."""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import platform
import resource
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.thermal_trajectory import ThermalTrajectory, trajectory_catalog
from src.native_surface import extract_surface, hottest_area_weights
from src.matched_projection import MatchedHeatOperator
from src.within_step_integration import (
    native_source_contributions, threshold_endpoints, error_metrics, source_metrics,
)

BASE = ROOT/'outputs/by_experiment'
OUT = BASE/'56_within_step_integration'
DATA = ROOT.parent/'heat_eq_laser_trajectories'
NAMES = ['DiagonalScanPath_8', 'SpiralScanPath_13']
VARIANTS = ['S0', 'S_time', 'S_evolve', 'S_both', 'S_GL2', 'S_GL4']
COLORS = ['#666666', '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#882255']
GATE = dict(hot_rmse_relative_improvement=.05, peak_absolute_improvement_K=1.,
            shoulder_allowed_relative_increase=.05, shoulder_allowed_absolute_increase_K=.10,
            whole_allowed_relative_increase=.02, whole_allowed_absolute_increase_K=.02,
            oracle_hot_allowed_relative_increase=.02, oracle_hot_allowed_absolute_increase_K=.10,
            oracle_peak_allowed_increase_K=.25,
            quadrature_relative_source_L2=.01, quadrature_hot_rmse_difference_K=.10,
            refinement_relative_source_L2=.01, refinement_hot_rmse_difference_K=.10)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sections_for(name):
    path = BASE/'49_matched_native_grid_gp/native_diagnostics'/f'cross_sections_{name}.csv'
    frame = pd.read_csv(path)
    sections = []
    for _, group in frame.groupby('section', sort=False):
        b = group[group.model == 'native_B'].sort_values('along_mm')
        c = group[group.model == 'native_C'].sort_values('along_mm')
        np.testing.assert_array_equal(b.node.to_numpy(), c.node.to_numpy())
        sections.append(b)
    return sections, path


def plot_case(name, level, c, fields, sources, sections, temp_limits, error_limit):
    ceiling = float(c['ceiling'])
    for predecessor in ('inferred', 'oracle'):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for col, section in enumerate(sections):
            ids = section.node.to_numpy(int); x = section.along_mm.to_numpy()
            ax = axes[0, col]
            ax.plot(x, c['truth'][ids], 'k-', label='Current truth', lw=2)
            ax.plot(x, fields[predecessor]['propagation'][ids], color='#aaaaaa',
                    ls=':', label='Propagation only', lw=2)
            for variant, color in zip(VARIANTS, COLORS):
                ax.plot(x, fields[predecessor][variant][ids], color=color,
                        label=variant, ls='--' if variant.startswith('S_GL') else '-', lw=1.5)
                axes[1, col].plot(x, (fields[predecessor][variant]-c['truth'])[ids],
                                  color=color, label=variant, lw=1.5)
            ax.axhline(ceiling, color='black', ls=':', label=f'Ceiling {ceiling:.2f} K')
            ax.set_ylim(*temp_limits); ax.set_ylabel('Temperature (K)')
            ax.set_title(section.section.iloc[0], fontsize=10)
            axes[1, col].set_ylim(-error_limit, error_limit)
            axes[1, col].set_ylabel('Forecast minus truth (K)')
            axes[1, col].set_xlabel('Distance from fixed current true peak (mm)')
            for a in axes[:, col]:
                for lo, hi in [(-1., -.25), (.25, 1.)]:
                    a.axvspan(lo, hi, color='#999999', alpha=.10)
                a.grid(alpha=.15); a.axhline(0, color='black', lw=.5)
            ax.legend(fontsize=7, ncol=2)
        fig.suptitle(f'{name}: {predecessor} predecessor; ceiling {ceiling:.2f} K\n'
                     'Same target frame; deterministic means only. Shading: fixed shoulders.')
        fig.savefig(OUT/f'cross_sections_{name}_level{level}_{predecessor}.png', dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        tri = Triangulation(c['points'][:, 0]*1000, c['points'][:, 1]*1000, c['triangles'])
        peak = c['points'][int(np.argmax(c['truth']))]*1000
        for ax, variant in zip(axes.flat, VARIANTS):
            im = ax.tripcolor(tri, fields[predecessor][variant]-c['truth'],
                              shading='gouraud', cmap='RdBu_r', vmin=-error_limit, vmax=error_limit)
            ax.set_title(variant); ax.set_aspect('equal')
            ax.set_xlim(peak[0]-2, peak[0]+2); ax.set_ylim(peak[1]-2, peak[1]+2)
            ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Forecast minus truth (K)')
        fig.suptitle(f'{name}: {predecessor}; c={ceiling:.2f} K; fixed shared error scale\n'
                     'Native-node values rendered on mesh; zoom within 2 mm of true peak')
        fig.savefig(OUT/f'error_maps_{name}_level{level}_{predecessor}.png', dpi=130)
        plt.close(fig)


def plot_sources(name, sources, sections):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for col, section in enumerate(sections):
        ids = section.node.to_numpy(int); x = section.along_mm.to_numpy()
        for variant, color in zip(VARIANTS, COLORS):
            axes[0, col].plot(x, sources[variant][ids], label=variant, color=color)
            axes[1, col].plot(x, (sources[variant]-sources['S0'])[ids], color=color)
        axes[0, col].set_title(section.section.iloc[0], fontsize=10)
        axes[0, col].set_ylabel('Numerical increment (K)'); axes[0, col].legend(fontsize=8)
        axes[1, col].set_ylabel('Increment minus S0 (K)')
        axes[1, col].set_xlabel('Distance from fixed true peak (mm)')
        for ax in axes[:, col]: ax.grid(alpha=.15); ax.axhline(0, color='black', lw=.5)
    for row in axes:
        low = min(a.get_ylim()[0] for a in row); high = max(a.get_ylim()[1] for a in row)
        for a in row: a.set_ylim(low, high)
    fig.suptitle(name+': source contribution only; identical at all three ceilings\n'
                 'Temperature increments, not physical heat/energy measurements')
    fig.savefig(OUT/f'source_sections_{name}.png', dpi=160); plt.close(fig)


def gate_table(metrics, quad, refinement):
    rows = []
    for (name, level, state), group in metrics.groupby(['trajectory', 'level', 'predecessor']):
        baseline = group[group.variant == 'S0'].set_index('region')
        for variant in VARIANTS[1:]:
            v = group[group.variant == variant].set_index('region')
            hot, b = v.loc['hot1pct_area'], baseline.loc['hot1pct_area']
            if state == 'inferred':
                hot_ok = hot.rmse_K <= b.rmse_K*(1-GATE['hot_rmse_relative_improvement'])
                peak_ok = hot.true_peak_absolute_error_K <= b.true_peak_absolute_error_K-GATE['peak_absolute_improvement_K']
            else:
                hot_ok = hot.rmse_K <= b.rmse_K+max(GATE['oracle_hot_allowed_absolute_increase_K'], GATE['oracle_hot_allowed_relative_increase']*b.rmse_K)
                peak_ok = hot.true_peak_absolute_error_K <= b.true_peak_absolute_error_K+GATE['oracle_peak_allowed_increase_K']
            shoulder_ok = all(v.loc[r].rmse_K <= baseline.loc[r].rmse_K+max(
                GATE['shoulder_allowed_absolute_increase_K'], GATE['shoulder_allowed_relative_increase']*baseline.loc[r].rmse_K)
                for r in v.index if r.startswith('shoulder_'))
            whole_ok = v.loc['whole_area'].rmse_K <= baseline.loc['whole_area'].rmse_K+max(
                GATE['whole_allowed_absolute_increase_K'], GATE['whole_allowed_relative_increase']*baseline.loc['whole_area'].rmse_K)
            q = quad[(quad.trajectory == name)&(quad.level == level)&(quad.predecessor == state)]
            pair = 'S_GL2_vs_S_GL4' if variant.startswith('S_GL') else 'S_both_vs_S_GL4'
            q = q[q.comparison == pair].iloc[0]
            quadrature_ok = (variant in ('S_time', 'S_evolve') or
                (q.relative_source_L2 <= GATE['quadrature_relative_source_L2'] and
                 abs(q.hot_rmse_difference_K) <= GATE['quadrature_hot_rmse_difference_K']))
            r = refinement[(refinement.trajectory == name)&(refinement.level == level)&
                           (refinement.predecessor == state)&(refinement.variant == variant)]
            refinement_ok = bool(((r.relative_source_L2 <= GATE['refinement_relative_source_L2']) &
                                  (abs(r.hot_rmse_difference_K) <= GATE['refinement_hot_rmse_difference_K'])).all())
            rows.append(dict(trajectory=name, level=level, predecessor=state, variant=variant,
                             hot_pass=bool(hot_ok), peak_pass=bool(peak_ok), shoulder_pass=bool(shoulder_ok),
                             whole_pass=bool(whole_ok), quadrature_pass=bool(quadrature_ok),
                             refinement_pass=refinement_ok,
                             case_pass=bool(hot_ok and peak_ok and shoulder_ok and whole_ok and quadrature_ok and refinement_ok)))
    return pd.DataFrame(rows)


def main():
    # Refuse to overwrite a scientific run, including a partially completed run.
    OUT.mkdir(parents=True, exist_ok=False)
    start = perf_counter(); fingerprints = {}; checks = []; provenance = []
    def fingerprint(path):
        fingerprints[str(path.relative_to(ROOT))] = file_digest(path)
    for name in ('49_matched_native_grid_gp', '52_native_higher_ceiling', '53_native_observation_retention'):
        fingerprint(BASE/name/'frozen_configuration.csv')
    o = pd.read_csv(BASE/'52_native_higher_ceiling/frozen_configuration.csv').iloc[0].to_dict()
    for name in ('49_matched_native_grid_gp', '53_native_observation_retention'):
        assert o == pd.read_csv(BASE/name/'frozen_configuration.csv').iloc[0].to_dict()
    frozen = dict(experiment=56, driver=str(Path(__file__).relative_to(ROOT)),
        revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        scope='deterministic only until gate; two development cases; no held-out inference',
        trajectories=NAMES, levels=['cached_reference', '320 K', '325 K'],
        inherited_options=o, effective_evaluation='native physical-area weights; exact 1% with fractional last node',
        metadata_caveat='Inherited top-25/grid and calibration-target CSV labels are historical; not executed here.',
        fine_grids=[[601,401], [1201,801], [1801,1201]],
        refinement_scope='source contribution only; predecessor contribution held at frozen 601x401',
        shoulder_definition='saved native sections; 0.25 <= abs(along_mm) <= 1.0; area-weighted section-node diagnostics',
        gate=GATE, gate_scope='every inferred case must improve; oracle cases must not materially deteriorate; all six cases, fixed tolerances',
        quadrature='linearly interpolate individually thresholded endpoints; GL2 and GL4; no unseen snapshot assumption validation',
        units=dict(coordinates='m (dataset convention)', time='s', temperature='K',
                   HeatFluxZ='K/m per supplied derivative export; no independent HDF unit confirmation',
                   gamma='empirical; m/s if stated coordinate/time units hold', integral='K*m^2, NOT energy'),
        gp_inference=False, parameters_refitted=False, python=sys.version, numpy=np.__version__)
    write_json(OUT/'frozen_configuration.json', frozen)
    for path in [Path(__file__), ROOT/'src/within_step_integration.py', ROOT/'src/matched_projection.py',
                 ROOT/'src/forecast_attribution.py', ROOT/'src/thermal_trajectory.py', ROOT/'src/native_surface.py',
                 ROOT/'tests/test_within_step_integration.py',
                 ROOT/'scripts/55_thermal_native_forecast_attribution.py', ROOT/'scripts/56_thermal_native_higher_ceiling.py',
                 ROOT/'scripts/57_thermal_native_observation_retention.py']:
        fingerprint(path)
    catalog = {r.name:r for r in trajectory_catalog(DATA)}
    assert len(catalog) == 33
    metrics = []; source_rows = []; quadrature = []; refinement = []; sections_saved = []
    runtime = []
    for name in NAMES:
        tick = perf_counter()
        trajectory = ThermalTrajectory(catalog[name].xdmf_path); mesh = extract_surface(trajectory)
        gp = trajectory.read_heat_flux_z(-2, surface_only=True)
        gc = trajectory.read_heat_flux_z(-1, surface_only=True)
        previous_truth = trajectory.read_temperature(-2, surface_only=True)
        truth = trajectory.read_temperature(-1, surface_only=True)
        dt = float(trajectory.times[-1]-trajectory.times[-2]); assert dt > 0
        physics = dict(dt=dt, diffusivity=o['diffusivity'], cooling_rate=o['cooling_rate'])
        xs = np.linspace(mesh.points[:,0].min(), mesh.points[:,0].max(), int(o['nx']))
        ys = np.linspace(mesh.points[:,1].min(), mesh.points[:,1].max(), int(o['ny']))
        op = MatchedHeatOperator(mesh, xs, ys)
        args = dict(threshold=o['source_flux_threshold'], gamma=o['source_coupling'], **physics)
        sources = native_source_contributions(op, gp, gc, **args)
        qp, qc = threshold_endpoints(gp, gc, o['source_flux_threshold'])
        refined = {}
        for nx, ny in [(1201,801), (1801,1201)]:
            operator = MatchedHeatOperator(mesh, xs, ys, nx=nx, ny=ny)
            refined[nx] = native_source_contributions(operator, gp, gc, **args)
            del operator
        source_metrics_here = {}
        for variant, source in sources.items():
            row = dict(trajectory=name, variant=variant, **source_metrics(source, mesh.points, mesh.weights))
            source_rows.append(row); source_metrics_here[variant] = row
        source_rows.extend(dict(trajectory=name, variant=f'{variant}_grid{nx}', **source_metrics(s, mesh.points, mesh.weights))
                           for nx, group in refined.items() for variant,s in group.items())
        sections, spath = sections_for(name); fingerprint(spath)
        for section in sections:
            ids = section.node.to_numpy(int)
            np.testing.assert_allclose(mesh.points[ids]*1000, section[['x_mm','y_mm']].to_numpy(), rtol=0, atol=1e-12)
            for row in section.itertuples():
                sections_saved.append(dict(trajectory=name, section=row.section, node=row.node, along_mm=row.along_mm,
                                           shoulder=bool(.25 <= abs(row.along_mm) <= 1.0)))
        path51 = BASE/'51_native_forecast_attribution/by_trajectory'/name/'components.npz'
        fingerprint(path51)
        with np.load(path51) as f: old51 = {k:f[k] for k in ('source','base_forecast','oracle_forecast')}
        np.testing.assert_array_equal(sources['S0'], old51['source'])
        cases = []
        reference_oracle = None
        for level in range(3):
            path = BASE/'52_native_higher_ceiling/by_trajectory'/name/f'level_{level}.npz'
            fingerprint(path)
            with np.load(path) as f:
                c = {k:f[k] for k in ['points','triangles','area_weights','truth','previous_truth','previous_mean',
                    'physics_mean','gradient_contribution','ambient','ceiling','camera','previous_camera',
                    'hot_area_weights','numeric_candidates','current_signal','previous_signal','observation_ids']}
                previous_draws_hash = digest(f['previous_draws'])
                covariance_hash = digest(f['C_prior_variance'])
            for a,b in [(mesh.points,c['points']), (mesh.triangles,c['triangles']), (mesh.weights,c['area_weights']),
                        (truth,c['truth']), (previous_truth,c['previous_truth']), (sources['S0'],c['gradient_contribution']),
                        (hottest_area_weights(truth,mesh.weights),c['hot_area_weights'])]:
                np.testing.assert_array_equal(a,b)
            ambient = float(c['ambient']); peak = int(np.argmax(truth))
            fields = {}
            regions = dict(whole_area=mesh.weights, hot1pct_area=c['hot_area_weights'],
                           observed_censored=mesh.weights*(c['camera']==c['ceiling']))
            for index, section in enumerate(sections):
                selected = section[(abs(section.along_mm)>=.25)&(abs(section.along_mm)<=1.)].node.to_numpy(int)
                mask = np.zeros(len(truth),bool); mask[selected] = True
                assert mask.sum() > 2
                regions['shoulder_'+('across' if 'Across' in section.section.iloc[0] else 'along')] = mesh.weights*mask
            for state, previous in [('inferred',c['previous_mean']), ('oracle',previous_truth)]:
                propagation = ambient+op.propagate(previous-ambient, 'native', floor=True, **physics)
                fields[state] = dict(propagation=propagation, **{v:propagation+s for v,s in sources.items()})
                original = op.mean(previous, gc, 'native', ambient=ambient, source_coupling=o['source_coupling'],
                                   source_flux_threshold=o['source_flux_threshold'], **physics)
                np.testing.assert_array_equal(fields[state]['S0'], original)
                if state == 'inferred': np.testing.assert_array_equal(original,c['physics_mean'])
                if state == 'oracle':
                    if reference_oracle is None: reference_oracle = original.copy()
                    np.testing.assert_array_equal(original,reference_oracle)
                    np.testing.assert_array_equal(original,old51['oracle_forecast'])
                if level == 0 and state == 'inferred': np.testing.assert_array_equal(original,old51['base_forecast'])
                for variant, forecast in fields[state].items():
                    if variant == 'propagation': continue
                    for region, weights in regions.items():
                        metrics.append(dict(trajectory=name, level=level, ceiling_K=float(c['ceiling']),
                            predecessor=state, variant=variant, region=region, nodes=int(np.count_nonzero(weights)),
                            area_m2=float(weights.sum()), **error_metrics(forecast, truth, weights),
                            true_peak_signed_error_K=float(forecast[peak]-truth[peak]),
                            true_peak_absolute_error_K=float(abs(forecast[peak]-truth[peak])),
                            forecast_max_K=float(forecast.max()), forecast_max_node=int(np.argmax(forecast)),
                            forecast_max_x_mm=float(mesh.points[np.argmax(forecast),0]*1000),
                            forecast_max_y_mm=float(mesh.points[np.argmax(forecast),1]*1000)))
                for a,b in [('S_both','S_GL2'),('S_both','S_GL4'),('S_GL2','S_GL4')]:
                    difference = sources[a]-sources[b]
                    quadrature.append(dict(trajectory=name, level=level, predecessor=state, comparison=a+'_vs_'+b,
                        relative_source_L2=float(np.sqrt(np.sum(mesh.weights*difference**2)/np.sum(mesh.weights*sources[b]**2))),
                        max_source_difference_K=float(abs(difference).max()),
                        source_maximum_difference_K=float(sources[a].max()-sources[b].max()),
                        hot_rmse_difference_K=error_metrics(fields[state][a],truth,c['hot_area_weights'])['rmse_K']-
                                              error_metrics(fields[state][b],truth,c['hot_area_weights'])['rmse_K'],
                        whole_rmse_difference_K=error_metrics(fields[state][a],truth,mesh.weights)['rmse_K']-
                                                error_metrics(fields[state][b],truth,mesh.weights)['rmse_K']))
                for nx, group in refined.items():
                    for variant,s in group.items():
                        delta = s-sources[variant]
                        refinement.append(dict(trajectory=name, level=level, predecessor=state, fine_nx=nx, variant=variant,
                            relative_source_L2=float(np.sqrt(np.sum(mesh.weights*delta**2)/np.sum(mesh.weights*s**2))),
                            max_source_difference_K=float(abs(delta).max()),
                            hot_rmse_difference_K=error_metrics(propagation+s,truth,c['hot_area_weights'])['rmse_K']-
                                error_metrics(fields[state][variant],truth,c['hot_area_weights'])['rmse_K']))
            dest = OUT/'by_trajectory'/name; dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/f'level_{level}.npz', **c, previous_gradient=gp, current_gradient=gc,
                previous_thresholded_gradient=qp,current_thresholded_gradient=qc,
                **{'source_'+k:v for k,v in sources.items()},
                **{state+'_'+k:v for state,group in fields.items() for k,v in group.items()})
            checks.append(dict(trajectory=name,level=level,dt_s=dt,native_nodes=len(truth),
                baseline_max_abs_difference_K=float(abs(fields['inferred']['S0']-c['physics_mean']).max()),
                source_max_abs_difference_K=float(abs(sources['S0']-c['gradient_contribution']).max()),
                points_sha256=digest(mesh.points),triangles_sha256=digest(mesh.triangles),area_sha256=digest(mesh.weights),
                hot_area_sha256=digest(c['hot_area_weights']),camera_sha256=digest(c['camera']),
                previous_camera_sha256=digest(c['previous_camera']),previous_mean_sha256=digest(c['previous_mean']),
                previous_draws_sha256=previous_draws_hash,prior_variance_sha256=covariance_hash,
                cached_observation_ids_sha256=digest(c['observation_ids'])))
            cases.append((level,c,fields))
        # The numerical lift/evaluation is not an exact native-node identity.
        roundtrip = op.propagate(qc, 'native',dt=0.,diffusivity=o['diffusivity'],cooling_rate=o['cooling_rate'])
        provenance.append(dict(trajectory=name,xdmf=str(trajectory.xdmf_path),xdmf_sha256=file_digest(trajectory.xdmf_path),
            h5=str(trajectory.h5_path),previous_time_s=float(trajectory.times[-2]),current_time_s=float(trajectory.times[-1]),
            dt_s=dt,previous_gradient_sha256=digest(gp),current_gradient_sha256=digest(gc),
            surface_node_indices_sha256=digest(trajectory.surface_node_indices()),
            previous_gradient_dataset=trajectory._field_datasets['HeatFluxZ'][-2],
            current_gradient_dataset=trajectory._field_datasets['HeatFluxZ'][-1],
            zero_age_resampling_increment_max_difference_K=float(abs((roundtrip-qc)*o['source_coupling']*dt).max()),
            coordinate_min_m=mesh.points.min(axis=0).tolist(),coordinate_max_m=mesh.points.max(axis=0).tolist()))
        numerical_seconds = perf_counter()-tick
        allfields = [v for _,_,f in cases for group in f.values() for v in group.values()]
        temp_limits = (min(float(truth.min()),min(float(v.min()) for v in allfields))-1,
                       max(float(truth.max()),max(float(v.max()) for v in allfields))+1)
        error_limit = max(float(abs(v-truth).max()) for v in allfields)
        plot_sources(name,sources,sections)
        for level,c,fields in cases: plot_case(name,level,c,fields,sources,sections,temp_limits,error_limit)
        runtime.append(dict(trajectory=name,numerical_and_cache_seconds=numerical_seconds,
                            total_with_plots_seconds=perf_counter()-tick))
        print(name, runtime[-1], flush=True)
    m, q, r = pd.DataFrame(metrics), pd.DataFrame(quadrature), pd.DataFrame(refinement)
    gate = gate_table(m,q,r)
    passed = gate.groupby('variant').case_pass.all()
    candidates = passed[passed].index.tolist()
    for filename, frame in [('deterministic_metrics',m), ('source_component_comparison',pd.DataFrame(source_rows)),
                            ('quadrature_checks',q), ('spatial_refinement',r), ('gate_checks',gate),
                            ('frozen_sections',pd.DataFrame(sections_saved)), ('runtime',pd.DataFrame(runtime)),
                            ('equality_checks',pd.DataFrame(checks))]:
        frame.to_csv(OUT/f'{filename}.csv',index=False)
    m.pivot(index=['trajectory','level','ceiling_K','variant','region'],columns='predecessor',
            values=['rmse_K','bias_K','mae_K','true_peak_signed_error_K']).to_csv(OUT/'oracle_vs_inferred.csv')
    write_json(OUT/'gradient_provenance.json',provenance)
    write_json(OUT/'source_fingerprints.json',fingerprints)
    verification = dict(baseline_exact=True,source_exact=True,oracle_ceiling_invariant=True,
        all33_catalogued=len(catalog),development_cases_only=NAMES,geometry_and_hot_weights_unchanged=True,
        previous_inference_rerun=False,current_gp_inference=False,parameters_fitted=False,
        stage1_gate_pass=bool(candidates),eligible_candidates=candidates,
        runtime_seconds=perf_counter()-start,
        peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        units_provenance='supplied notebook derivative export; HDF data values not independently unit-labelled',
        numerical_limitation='positive-age native P1 lift/fine-grid/bilinear map; zero-age endpoint used directly; not exact native semigroup',
        next_step='Review deterministic gate before any GP inference' if candidates else 'STOP after Stage 1; no GP inference')
    write_json(OUT/'verification.json',verification)
    print(json.dumps(verification,indent=2),flush=True)


if __name__ == '__main__':
    main()
