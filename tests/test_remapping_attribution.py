import unittest
import numpy as np
from src.remapping_attribution import constant_gradient_attribution


class RemappingAttributionTests(unittest.TestCase):
    def test_no_evolution_leaves_only_one_remapping(self):
        q = np.array([3., 10.])
        mapping = np.array([[.8,.2],[.3,.7]])
        calls = []
        def propagate(values, age):
            calls.append((values.copy(), age))
            return mapping@values
        s = constant_gradient_attribution(q, gamma=.17, dt=.0005, propagate=propagate)
        for key in ['S_map', 'S_evolved_GL2', 'S_evolved_GL4']:
            np.testing.assert_allclose(s[key], .17*.0005*(mapping@q), rtol=1e-14)
        for values, age in calls:
            np.testing.assert_array_equal(values, q)
            self.assertTrue(0 <= age <= .0005)

    def test_cooling_integral_with_nonidentity_map(self):
        q = np.array([3., -2.]); mapping = np.array([[.8,.2],[.3,.7]])
        for beta in [0., 1e-12, 12.]:
            dt=.0005
            s=constant_gradient_attribution(q,gamma=.17,dt=dt,
                propagate=lambda v,t:(mapping@v)*np.exp(-beta*t))
            exact=.17*(mapping@q)*(dt if beta==0 else -np.expm1(-beta*dt)/beta)
            np.testing.assert_allclose(s['S_evolved_GL4'],exact,rtol=1e-13)
            total=s['S_evolved_GL4']-s['S0']
            np.testing.assert_allclose(total,(s['S_map']-s['S0'])+(s['S_evolved_GL4']-s['S_map']),atol=1e-15)

    def test_identity_and_zero_source(self):
        for q in [np.zeros(4), np.arange(4.)]:
            s=constant_gradient_attribution(q,gamma=.17,dt=.0005,propagate=lambda v,t:v)
            for value in s.values():
                np.testing.assert_allclose(value,.17*q*.0005,rtol=1e-14)


if __name__ == '__main__': unittest.main()
