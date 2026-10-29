"""Native adapter tests, including the frozen map's non-identity limitation."""
import unittest
import numpy as np
from scipy.spatial import Delaunay
from src.native_surface import surface_mesh
from src.matched_projection import MatchedHeatOperator
from src.within_step_integration import native_source_contributions


class WithinStepOperatorTests(unittest.TestCase):
    def setUp(self):
        xs = np.linspace(0, 1, 9)
        x, y = np.meshgrid(xs, xs)
        points = np.column_stack([x.ravel(), y.ravel()])
        mesh = surface_mesh(points, Delaunay(points).simplices)
        self.op = MatchedHeatOperator(mesh, xs, xs, nx=9, ny=9)
        self.q = np.arange(len(points), dtype=float)+10000

    def test_identity_no_diffusion_or_cooling_on_matching_nodes(self):
        sources = native_source_contributions(self.op, self.q, self.q,
            threshold=10000, gamma=.17, dt=.0005, diffusivity=0., cooling_rate=0.)
        for s in sources.values():
            np.testing.assert_allclose(s, .17*self.q*.0005, rtol=1e-14)

    def test_pure_cooling_constant_native_field(self):
        q = np.ones(len(self.q))*20000
        beta, dt = 12., .0005
        sources = native_source_contributions(self.op, q, q,
            threshold=10000, gamma=.17, dt=dt, diffusivity=0., cooling_rate=beta)
        exact = .17*q*(-np.expm1(-beta*dt))/beta
        np.testing.assert_allclose(sources['S_GL4'], exact, rtol=1e-13)

    def test_unfloored_source_map_is_linear(self):
        kw = dict(dt=.0005, diffusivity=3e-6, cooling_rate=12., floor=False)
        a = self.op.propagate(self.q, 'native', **kw)
        b = self.op.propagate(-self.q, 'native', **kw)
        np.testing.assert_allclose(a, -b, atol=1e-13)

    def test_irregular_zero_age_map_is_not_identity(self):
        # This records a pre-existing interpolation effect, not a new remedy.
        mesh = surface_mesh(np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.413,.617]]),
                            np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]]))
        xs = np.linspace(0, 1, 5)
        op = MatchedHeatOperator(mesh, xs, xs, nx=21, ny=21)
        pulse = np.array([0.,0.,0.,0.,1.])
        mapped = op.propagate(pulse, 'native', dt=0., diffusivity=0., cooling_rate=0.)
        self.assertGreater(np.max(abs(mapped-pulse)), 1e-3)
        self.assertLess(mapped[-1], pulse[-1])


if __name__ == '__main__':
    unittest.main()
