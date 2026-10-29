import unittest
import numpy as np
from src.within_step_integration import (
    source_contributions, threshold_endpoints, native_source_contributions,
)


class WithinStepTests(unittest.TestCase):
    def test_threshold_individual_snapshots(self):
        qp, qc = threshold_endpoints(np.array([9., 10., -4.]),
                                    np.array([11., 0., 12.]), 10.)
        np.testing.assert_array_equal(qp, [0, 10, 0])
        np.testing.assert_array_equal(qc, [11, 0, 12])

    def test_zero_gradient(self):
        result = source_contributions(np.zeros(4), np.zeros(4), gamma=.17,
                                      dt=.0005, evolve=lambda q, t: q*np.exp(-12*t))
        for value in result.values():
            np.testing.assert_array_equal(value, np.zeros(4))

    def test_identity_constant(self):
        q = np.array([0., -3., 12000.])
        result = source_contributions(q, q, gamma=.17, dt=.0005,
                                      evolve=lambda q, t: q)
        for value in result.values():
            np.testing.assert_allclose(value, .17*q*.0005, rtol=1e-14)

    def test_identity_linear_time(self):
        qp, qc = np.array([-3., 10.]), np.array([8., -2.])
        result = source_contributions(qp, qc, gamma=2., dt=.3, evolve=lambda q,t: q)
        exact = 2*.3*(qp+qc)/2
        for key in ('S_time', 'S_both', 'S_GL2', 'S_GL4'):
            np.testing.assert_allclose(result[key], exact, atol=1e-14)

    def test_pure_cooling_and_zero_limit(self):
        q = np.array([3., 7.])
        for beta in (0., 1e-12, 12.422901570529978):
            dt = .0005
            result = source_contributions(q, q, gamma=.17, dt=dt,
                                          evolve=lambda q,t: q*np.exp(-beta*t))
            exact = .17*q*(dt if beta == 0 else -np.expm1(-beta*dt)/beta)
            np.testing.assert_allclose(result['S_GL4'], exact, rtol=1e-13)
            np.testing.assert_allclose(result['S_GL2'], exact, rtol=1e-11)
            # Trapezoidal integration is an approximation, not an exact solver.
            np.testing.assert_allclose(result['S_both'], exact, rtol=4e-6)

    def test_native_adapter_no_floor_or_rethreshold(self):
        class Operator:
            def __init__(self): self.calls = []
            def propagate(self, q, representation, **kwargs):
                self.calls.append((q.copy(), representation, kwargs))
                return q*.8
        op = Operator()
        native_source_contributions(op, np.array([10000., 0]), np.array([0., 20000.]),
                                    threshold=10000, gamma=.17, dt=.0005,
                                    diffusivity=3e-6, cooling_rate=12.)
        self.assertEqual(len(op.calls), 8)
        self.assertTrue(all(not k['floor'] and r == 'native' for _,r,k in op.calls))
        self.assertTrue(any(np.any((q > 0) & (q < 10000)) for q,_,_ in op.calls))
        self.assertTrue(all(0 <= k['dt'] <= .0005 for _,_,k in op.calls))


if __name__ == '__main__':
    unittest.main()
