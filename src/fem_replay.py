"""One-step legacy-DOLFIN replay utilities; no GP or reduced-model changes."""
import numpy as np
from scipy.spatial import cKDTree


def coordinate_map(original, target, tolerance=1e-12):
    """Return original-node IDs in target order; reject non-bijective recovery."""
    original, target = np.asarray(original), np.asarray(target)
    if original.shape != target.shape or not np.isfinite(target).all():
        raise ValueError('Coordinate arrays must be finite and have identical shapes')
    distances, ids = cKDTree(original).query(target)
    if distances.max() > tolerance or len(np.unique(ids)) != len(ids):
        raise ValueError('Coordinates do not define a one-to-one node/DOF mapping')
    return ids


def volume_weights(points, cells):
    vertices = np.asarray(points)[cells]
    volumes = abs(np.linalg.det(vertices[:, 1:]-vertices[:, :1]))/6
    if np.any(volumes <= 0):
        raise ValueError('Degenerate tetrahedron')
    weights = np.zeros(len(points))
    np.add.at(weights, np.asarray(cells).ravel(), np.repeat(volumes/4, 4))
    return weights


def error_metrics(prediction, truth, weights, ambient):
    error = np.asarray(prediction)-truth
    w = np.asarray(weights)
    denom = np.sum(w*(truth-ambient)**2)
    return dict(rmse_K=float(np.sqrt(np.sum(w*error**2)/w.sum())),
                bias_K=float(np.sum(w*error)/w.sum()),
                max_abs_K=float(abs(error).max()),
                relative_excess_L2=float(np.sqrt(np.sum(w*error**2)/denom)) if denom > 0 else None)


def replay_passes(metrics, gate):
    return all(np.isfinite(metrics[region][key]) and metrics[region][key] <= gate[key]
               for region in ['volume', 'surface'] for key in ['rmse_K', 'max_abs_K'])


def decomposition(old_off, old_increment, fem_off, fem_on, truth):
    response = fem_on-fem_off
    propagation_error = old_off-fem_off
    response_error = old_increment-response
    replay_error = fem_on-truth
    total = old_off+old_increment-truth
    np.testing.assert_allclose(total, propagation_error+response_error+replay_error,
                               atol=2e-12, rtol=0)
    return dict(fem_response=response, propagation_discrepancy=propagation_error,
                increment_discrepancy=response_error, replay_discrepancy=replay_error,
                total_error=total)


def build_problem(points, cells, previous, laser, time):
    """Recreate the supplied P1 weak form on the saved mesh, in serial."""
    import dolfin as df
    if df.MPI.size(df.MPI.comm_world) != 1:
        raise RuntimeError('This bounded nodal-mapping audit requires serial execution')
    mesh = df.Mesh()
    editor = df.MeshEditor()
    editor.open(mesh, 'tetrahedron', 3, 3)
    editor.init_vertices(len(points)); editor.init_cells(len(cells))
    for i, point in enumerate(points):
        editor.add_vertex(i, point)
    for i, cell in enumerate(cells):
        editor.add_cell(i, np.asarray(cell, dtype=np.uintp))
    editor.close()
    vertex_nodes = coordinate_map(points, mesh.coordinates())
    actual = np.sort(vertex_nodes[mesh.cells()], axis=1)
    expected = np.sort(cells, axis=1)
    canonical = lambda x: x[np.lexsort(x.T[::-1])]
    np.testing.assert_array_equal(canonical(actual), canonical(expected))
    V = df.FunctionSpace(mesh, 'P', 1)
    dof_nodes = coordinate_map(points, V.tabulate_dof_coordinates().reshape((-1, 3)))
    prev = df.Function(V)
    prev.vector().set_local(previous[dof_nodes]); prev.vector().apply('insert')
    np.testing.assert_array_equal(prev.compute_vertex_values(mesh), previous[vertex_nodes])
    p = laser.parameters
    zmin, zmax = points[:, 2].min(), points[:, 2].max()
    bottom = df.AutoSubDomain(lambda x, on_boundary: df.near(x[2], zmin) and on_boundary)
    top = df.AutoSubDomain(lambda x, on_boundary: df.near(x[2], zmax) and on_boundary)
    boundaries = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundaries.set_all(0); bottom.mark(boundaries, 1); top.mark(boundaries, 2)
    ds = df.Measure('ds', domain=mesh, subdomain_data=boundaries)
    bc = df.DirichletBC(V, df.Constant(p['T_0']), bottom)
    bc_zero = df.DirichletBC(V, df.Constant(0.), bottom)
    expression = laser.expression(time)

    class PrescribedExpression(df.UserExpression):
        def eval(self, values, x):
            expression.eval(values, x)

        def value_shape(self):
            return ()

    g = PrescribedExpression(degree=2)
    u, v = df.TrialFunction(V), df.TestFunction(V)
    dx = df.Measure('dx', domain=mesh)
    a = (p['rho']*p['Cp']/p['dt'])*u*v*dx + df.dot(p['k']*df.grad(u), df.grad(v))*dx
    old_load = (p['rho']*p['Cp']/p['dt'])*prev*v*dx
    source_load = g*v*ds(2)
    top_area = float(df.assemble(df.Constant(1.)*ds(2)))
    bottom_area = float(df.assemble(df.Constant(1.)*ds(1)))
    extent = np.ptp(points, axis=0)
    np.testing.assert_allclose([top_area, bottom_area], extent[0]*extent[1], atol=1e-14, rtol=0)
    assembled_volume = float(df.assemble(df.Constant(1.)*dx))
    # Serial element accumulation has an O(n_cells * eps) rounding bound.
    volume_sum_rtol = 2*len(cells)*np.finfo(float).eps
    np.testing.assert_allclose(assembled_volume, np.prod(extent), rtol=volume_sum_rtol, atol=0)
    return dict(df=df, mesh=mesh, V=V, a=a, old_load=old_load, source_load=source_load,
                bc=bc, bc_zero=bc_zero, dof_nodes=dof_nodes, vertex_nodes=vertex_nodes,
                top_area_m2=top_area, bottom_area_m2=bottom_area,
                assembled_volume_m3=assembled_volume, volume_sum_rtol=volume_sum_rtol,
                assembled_boundary_input_W=float(df.assemble(g*ds(2))))


def solve_problem(problem, mode):
    df, a = problem['df'], problem['a']
    if mode == 'on':
        load = problem['old_load']+problem['source_load']; bc = problem['bc']
    elif mode == 'off':
        load = problem['old_load']; bc = problem['bc']
    elif mode == 'response':
        load = problem['source_load']; bc = problem['bc_zero']
    else:
        raise ValueError(mode)
    result = df.Function(problem['V'])
    # Keep notebook solve defaults; do not tune solver settings to match truth.
    df.solve(a == load, result, bc)
    matrix, rhs = df.assemble(a), df.assemble(load)
    bc.apply(matrix, rhs)
    residual = matrix*result.vector()-rhs
    relative_residual = float(residual.norm('l2')/max(rhs.norm('l2'), np.finfo(float).tiny))
    nodal = np.empty(problem['V'].dim())
    nodal[problem['dof_nodes']] = result.vector().get_local()
    return nodal, relative_residual
