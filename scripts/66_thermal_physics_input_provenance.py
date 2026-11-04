"""Read-only input audit: no GP, calibration, propagation or laser reconstruction."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.thermal_trajectory import ThermalTrajectory
from src.native_surface import extract_surface

OUT = ROOT / 'outputs/by_experiment/62_physics_input_provenance'
DATA = ROOT.parent / 'heat_eq_laser_trajectories'
NAMES = ['DiagonalScanPath_8', 'SpiralScanPath_13']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    if OUT.exists():
        raise FileExistsError('Refusing to overwrite an existing provenance audit')
    before = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    sources = sorted({*ROOT.glob('src/**/*.py'), *ROOT.glob('scripts/**/*.py'),
                      *ROOT.glob('tests/**/*.py')})
    inputs = [ROOT / p for p in [
        'EXPERIMENT_PROTOCOL.md', 'EXPERIMENT_INVENTORY.md',
        'outputs/archive/2026-09-20_deferred/relocation_manifest.json',
        'outputs/by_experiment/24_posterior_physics_mean_advection/development_physics_parameters.csv',
        'outputs/by_experiment/24_posterior_physics_mean_advection/source_coupling_calibration.csv',
        'outputs/by_experiment/49_matched_native_grid_gp/frozen_configuration.csv',
        'outputs/by_experiment/52_native_higher_ceiling/frozen_configuration.csv',
        'outputs/by_experiment/52_native_higher_ceiling/PHYSICS_PROVENANCE.md',
        'outputs/by_experiment/56_within_step_integration/gradient_provenance.json',
        'outputs/by_experiment/60_previous_unpinning_audit/frozen_configuration.json',
        'outputs/by_experiment/61_unpinning_current_update/frozen_configuration.json',
        'outputs/by_experiment/61_unpinning_current_update/verification.json',
        'docs/adrienne_review/parameter_inventory.csv']]
    for name in NAMES:
        inputs.extend([DATA / f'Copy of {name}.xdmf', DATA / f'Copy of {name}.h5',
            ROOT / f'outputs/by_experiment/52_native_higher_ceiling/by_trajectory/{name}/level_0.npz'])
    thesis = Path.home() / 'Downloads/MasterThesis_Trentini.pdf'
    inputs.append(thesis)
    duplicate = thesis.with_name('MasterThesis_Trentini (1).pdf')
    if duplicate.exists():
        inputs.append(duplicate)
    fingerprints = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size)
                    for p in sources + inputs}
    OUT.mkdir(parents=True)
    provenance = json.loads((ROOT / 'outputs/by_experiment/56_within_step_integration/gradient_provenance.json').read_text())
    metadata = []
    for name in NAMES:
        t = ThermalTrajectory(DATA / f'Copy of {name}.xdmf')
        xyz = t.geometry(); cells = t.topology(); ids = t.surface_node_indices()
        mesh = extract_surface(t)
        evidence = next(r for r in provenance if r['trajectory'] == name)
        assert sha(t.xdmf_path) == evidence['xdmf_sha256']
        assert digest(ids) == evidence['surface_node_indices_sha256']
        root = ET.parse(t.xdmf_path).getroot()
        tags = lambda tag: [e for e in root.iter() if e.tag.rsplit('}', 1)[-1] == tag]
        grids = [e for e in tags('Grid') if e.find('Time') is not None]
        attribute_keys = {}; attribute_objects = [0]; object_count = [0]
        with h5py.File(t.h5_path, 'r') as handle:
            root_attrs = {k: np.asarray(v).tolist() for k, v in handle.attrs.items()}
            def visit(path, obj):
                object_count[0] += 1
                if obj.attrs:
                    attribute_objects[0] += 1
                for k, v in obj.attrs.items():
                    attribute_keys.setdefault(k, set()).add(json.dumps(np.asarray(v).tolist()))
            handle.visititems(visit)
        intervals = np.diff(t.times)
        frames = []
        for i in [t.n_times - 2, t.n_times - 1]:
            temp = t.read_temperature(i, surface_only=True)
            grad = t.read_heat_flux_z(i, surface_only=True)
            previous = i == t.n_times - 2
            prefix = 'previous' if previous else 'current'
            assert digest(grad) == evidence[prefix + '_gradient_sha256']
            assert t.times[i] == evidence[prefix + '_time_s']
            frames.append(dict(index_zero_based=i, time_value=float(t.times[i]),
                time_unit='seconds in downstream code; no unit declaration in XDMF',
                temperature_dataset=t._field_datasets['Temperature'][i],
                gradient_dataset=t._field_datasets['HeatFluxZ'][i],
                surface_temperature_sha256=digest(temp), surface_gradient_sha256=digest(grad),
                finite=bool(np.isfinite(temp).all() and np.isfinite(grad).all()),
                time_attribute=grids[i].find('Time').attrib,
                fields=[a.attrib for a in grids[i].findall('Attribute')]))
        with np.load(ROOT / f'outputs/by_experiment/52_native_higher_ceiling/by_trajectory/{name}/level_0.npz') as cache:
            np.testing.assert_array_equal(cache['points'], mesh.points)
            np.testing.assert_array_equal(cache['triangles'], mesh.triangles)
            np.testing.assert_array_equal(cache['area_weights'], mesh.weights)
            np.testing.assert_array_equal(cache['truth'], t.read_temperature(-1, surface_only=True))
            np.testing.assert_array_equal(cache['previous_truth'], t.read_temperature(-2, surface_only=True))
            ambient = float(cache['ambient'])
        metadata.append(dict(trajectory=name, stored_frames=t.n_times,
            first_saved_time=float(t.times[0]), last_saved_time=float(t.times[-1]),
            saved_interval_min=float(intervals.min()), saved_interval_max=float(intervals.max()),
            final_saved_interval=float(intervals[-1]), solver_timestep='unresolved',
            node_count=len(xyz), tetrahedra=len(cells), surface_nodes=len(ids),
            surface_triangles=len(mesh.triangles), coordinate_bounds=[xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
            surface_area_in_stored_coordinate_units_squared=float(mesh.weights.sum()),
            surface_indices_sha256=digest(ids), geometry_sha256=digest(xyz), topology_sha256=digest(cells),
            top_geometric_outward_normal=[0, 0, 1],
            normal_note='Geometric max-z exterior; generator facet marker and flux convention unresolved',
            fields=list(t.field_names), hdf_root_attributes=root_attrs, hdf_object_count=object_count[0],
            hdf_objects_with_attributes=attribute_objects[0],
            hdf_distinct_attribute_values={k: sorted(v) for k, v in attribute_keys.items()},
            xdmf_information=[dict(attributes=e.attrib, text=e.text) for e in tags('Information')],
            last_two_frames=frames, cache52_geometry_area_truth_exact=True,
            output56_gradient_hashes_exact=True, cached_surrogate_ambient_K=ambient))
    with (ROOT / 'outputs/by_experiment/52_native_higher_ceiling/frozen_configuration.csv').open() as f:
        cfg = next(csv.DictReader(f))
    save('data_metadata.json', metadata)
    save('fingerprints.json', dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status_before=before, files=fingerprints))
    save('audit_configuration.json', dict(date='2026-11-04', output_number=62,
        driver='scripts/66_thermal_physics_input_provenance.py', scope='two paths, final two saved frames; inspection only',
        retained_surrogate_parameters={k: cfg[k] for k in ['diffusivity','cooling_rate','source_coupling','source_flux_threshold','heat_flux_cutoff']},
        generator_code_confirmed=False, reconstruction_gate='STOP: exact generator, trajectory, power/width/absorption/switch and sign/material provenance unresolved',
        prescribed_flux_reconstructed=False, conductivity_gradient_compared=False,
        gp_inference=False, fitting=False, physics_propagation=False, baseline_changed=False,
        unit_note='Raw coordinates/times lack explicit SI-unit attributes; units are supplied-context/downstream conventions'))
    for path, record in fingerprints.items():
        assert sha(Path(path)) == record['sha256'], path
    save('verification.json', dict(all_fingerprinted_inputs_unchanged=True,
        source_files_checked=len(sources), input_artifacts_checked=len(inputs),
        native_cache_equality_cases=2, recorded_gradient_hashes_verified=4,
        source_comparison_performed=False, reason='Missing independently specified matching generator inputs'))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
