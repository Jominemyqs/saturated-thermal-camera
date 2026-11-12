from pathlib import Path

import numpy as np
import unittest

from src.prescribed_laser_audit import NotebookLaser, flux_metrics


def check_notebook_expression_and_clock(family, interval, frames, end):
    path = Path(__file__).resolve().parents[2] / (family+'ScanPath.ipynb')
    if not path.exists():
        raise unittest.SkipTest('Locally supplied generator notebook not installed')
    laser = NotebookLaser(path, family, interval)
    assert laser.frame_count == frames
    assert abs(laser.times[-1]-end) < 1e-14
    p = laser.parameters
    for t in [0., laser.times[0], laser.times[-2], laser.times[-1]]:
        focus = laser.focus(t)
        points = focus + np.array([[0, 0], [p['sigma'], 0], [0, 2*p['sigma']]])
        expected = p['power']/(2*np.pi*p['sigma']**2)*np.exp(-np.array([0., .5, 2.]))
        np.testing.assert_allclose(laser.evaluate(points, t), expected, rtol=1e-12)
        np.testing.assert_allclose(focus, np.clip(focus, laser.path.min(0), laser.path.max(0)), atol=1e-15)
    assert p['k'] == 15.11
    assert p['dt'] == .0005


class PrescribedLaserTests(unittest.TestCase):
    def test_diagonal_expression_and_clock(self):
        check_notebook_expression_and_clock('Diagonal', 8, 355, .1775)

    def test_spiral_expression_and_clock(self):
        check_notebook_expression_and_clock('Spiral', 13, 544, .272)

    def test_weighted_flux_comparison(self):
        g = np.array([1., 3.]); w = np.array([2., 1.])
        result = flux_metrics(g, 2*g, w)
        self.assertEqual(result['relative_L2'], 1.)
        self.assertEqual(result['integral_ratio'], 2.)
        self.assertEqual(result['bias_W_m2'], 5/3)
        self.assertEqual(flux_metrics(g, g, w)['rmse_W_m2'], 0.)
        with self.assertRaises(ValueError):
            flux_metrics(g, g, w*0)


if __name__ == '__main__':
    unittest.main()
