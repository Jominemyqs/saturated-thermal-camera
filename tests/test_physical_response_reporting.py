import unittest
import numpy as np
from src.physical_response_reporting import gaussian_crps


class GaussianPriorMetricsTests(unittest.TestCase):
    def test_at_mean(self):
        expected = (np.sqrt(2)-1)/np.sqrt(np.pi)
        np.testing.assert_allclose(gaussian_crps(0., 1., 0.), expected, rtol=1e-14)

    def test_translation_and_scaling(self):
        base = gaussian_crps(np.array([0., 1.]), 2., 3.)
        np.testing.assert_allclose(gaussian_crps(np.array([0., 1.])+300, 2., 303.), base)
        np.testing.assert_allclose(gaussian_crps(np.array([0., 2.]), 4., 6.), 2*base)

    def test_invalid_sd(self):
        with self.assertRaises(ValueError): gaussian_crps(0., 0., 1.)


if __name__ == '__main__': unittest.main()
