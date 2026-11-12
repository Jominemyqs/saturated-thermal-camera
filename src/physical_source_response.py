"""Independent linear FEM boundary response and mean-only replacement.

The response worker accepts geometry/time only, never a thermal-state cache.
Its zero predecessor is a placeholder for the existing assembly interface;
only the source load and homogeneous bottom condition enter the solve.
"""
import argparse
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

GEOMETRY_KEYS = {'points', 'cells', 'surface_ids', 'time', 'previous_time'}
COVARIANCE_KEYS = ('observed_covariance', 'pred_observed_covariance', 'prediction_variance')


def validate_geometry(data):
    if set(data) != GEOMETRY_KEYS:
        raise ValueError('Response input must contain ONLY geometry and timestamps')
    xyz, cells, ids = data['points'], data['cells'], data['surface_ids']
    if xyz.ndim != 2 or xyz.shape[1] != 3 or cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError('Expected tetrahedral volume geometry')
    if not np.isfinite(xyz).all() or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid geometry or surface IDs')
    if cells.min() < 0 or cells.max() >= len(xyz) or ids.min() < 0 or ids.max() >= len(xyz):
        raise ValueError('Invalid connectivity')
    if not float(data['time']) > float(data['previous_time']):
        raise ValueError('Invalid timestamps')


def solve_independent_response(data, laser):
    from src.fem_replay import build_problem, solve_problem
    validate_geometry(data)
    np.testing.assert_allclose(float(data['time']-data['previous_time']),
                               laser.parameters['dt'], rtol=0, atol=1e-14)
    problem = build_problem(data['points'], data['cells'], np.zeros(len(data['points'])),
                            laser, float(data['time']))
    response, residual = solve_problem(problem, 'response')
    return response, residual, problem


def replace_increment(blocks, observation_ids, old_increment, response):
    from src.source_off_audit import remove_current_increment
    result = remove_current_increment(blocks, observation_ids, old_increment)
    response = np.asarray(response)
    if response.shape != result['prediction_mean'].shape or not np.isfinite(response).all():
        raise ValueError('Invalid physical response')
    result['prediction_mean'] = result['prediction_mean'] + response
    result['observation_mean'] = result['prediction_mean'][observation_ids]
    for key in COVARIANCE_KEYS:
        assert result[key] is blocks[key]
    return result


def worker(geometry, notebook, family, interval, output):
    import dolfin as df
    from src.prescribed_laser_audit import NotebookLaser
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    if output.with_suffix('.npz').exists() or output.with_suffix('.json').exists():
        raise FileExistsError(output)
    start = perf_counter()
    with np.load(geometry) as f:
        data = dict(f)
    laser = NotebookLaser(Path(notebook), family, interval)
    df.parameters['form_compiler']['cache_dir'] = str(root/'tmp/fem_ffc')
    response, residual, problem = solve_independent_response(data, laser)
    record = dict(relative_linear_residual=residual, seconds=perf_counter()-start,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        passed=bool(residual <= 1e-10), dolfin_version=df.__version__,
        input_keys=sorted(data), temperature_fields_read=False, gradient_read=False,
        initial_increment=0, bottom_increment=0, material_parameters=laser.parameters,
        current_time_s=float(data['time']), dt_s=laser.parameters['dt'],
        volume_nodes=len(data['points']), cells=len(data['cells']),
        surface_nodes=len(data['surface_ids']), assembled_input_W=problem['assembled_boundary_input_W'])
    np.savez_compressed(output.with_suffix('.npz'), volume_response=response,
                        surface_response=response[data['surface_ids']], surface_ids=data['surface_ids'])
    output.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('geometry'); p.add_argument('notebook'); p.add_argument('family')
    p.add_argument('interval', type=int); p.add_argument('output')
    a = p.parse_args()
    worker(a.geometry, a.notebook, a.family, a.interval, a.output)
