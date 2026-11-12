import unittest
import importlib.util
import numpy as np
from src.fem_replay import coordinate_map, volume_weights, error_metrics, replay_passes, decomposition, build_problem, solve_problem


class FEMReplayTests(unittest.TestCase):
    def test_coordinate_permutation_and_rejections(self):
        points = np.eye(3)
        np.testing.assert_array_equal(coordinate_map(points, points[[2, 0, 1]]), [2, 0, 1])
        for bad in [points[[0, 0, 2]], points+.01]:
            with self.assertRaises(ValueError):
                coordinate_map(points, bad)

    def test_tetrahedral_weights_orientation(self):
        p = np.vstack([np.zeros(3), np.eye(3)])
        for c in [[[0, 1, 2, 3]], [[0, 2, 1, 3]]]:
            np.testing.assert_allclose(volume_weights(p, np.array(c)), np.full(4, 1/24))

    def test_gate_and_bookkeeping(self):
        truth = np.array([300., 310.]); w = np.array([1., 3.])
        m = error_metrics(truth+.01, truth, w, 293.15)
        self.assertAlmostEqual(m['rmse_K'], .01)
        gate = dict(rmse_K=1e-4, max_abs_K=1e-3)
        self.assertFalse(replay_passes(dict(volume=m, surface=m), gate))
        m = error_metrics(truth, truth, w, 293.15)
        self.assertTrue(replay_passes(dict(volume=m, surface=m), gate))
        d = decomposition(truth-5, np.array([8., 9.]), truth-3, truth+.01, truth)
        np.testing.assert_allclose(d['total_error'], [3., 4.])


@unittest.skipUnless(importlib.util.find_spec('dolfin'), 'Requires isolated legacy FEniCS environment')
class FEMOperatorTests(unittest.TestCase):
    def test_constant_state_and_linear_boundary_response(self):
        import dolfin as df
        from pathlib import Path
        df.parameters['form_compiler']['cache_dir'] = str(Path(__file__).resolve().parents[1]/'tmp/fem_ffc')

        class Laser:
            parameters = dict(T_0=293.15, rho=2., Cp=3., k=4., dt=.1)

            def expression(self, time):
                class Flux:
                    def eval(self, values, x):
                        values[0] = 1.
                return Flux()

        mesh = df.UnitCubeMesh(2, 2, 2)
        points, cells = mesh.coordinates().copy(), mesh.cells().copy()
        previous = np.full(len(points), 293.15)
        problem = build_problem(points, cells, previous, Laser(), .1)
        off, off_residual = solve_problem(problem, 'off')
        on, on_residual = solve_problem(problem, 'on')
        response, response_residual = solve_problem(problem, 'response')
        np.testing.assert_allclose(off, previous, atol=1e-10, rtol=0)
        np.testing.assert_allclose(on-off, response, atol=1e-10, rtol=0)
        self.assertGreater(response.max(), 0.)
        np.testing.assert_allclose(response[points[:, 2] == 0], 0., atol=1e-12)
        self.assertAlmostEqual(problem['assembled_boundary_input_W'], 1.)
        self.assertLess(max(off_residual, on_residual, response_residual), 1e-10)


if __name__ == '__main__':
    unittest.main()
