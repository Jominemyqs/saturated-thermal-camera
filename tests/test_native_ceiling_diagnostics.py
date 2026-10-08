import unittest
import numpy as np
from src.native_ceiling_diagnostics import camera_at_ceiling, uncertainty_distance, retained_reference_observations


class NativeCeilingTests(unittest.TestCase):
    def test_reference_retention_releases_values_without_dropping_nodes(self):
        signal = np.array([294., 297., 310., 330.])
        candidates = np.array([True, False, False, False])
        reference = np.minimum(signal, 296.)
        ids, numeric, saturated = retained_reference_observations(reference, 296.,
            np.minimum(signal, 320.), 320., candidates)
        np.testing.assert_array_equal(ids, [0, 1, 2, 3])
        np.testing.assert_array_equal(numeric, [True, True, True, False])
        self.assertFalse(np.any(numeric & saturated))
        refids, _, _ = retained_reference_observations(reference, 296., reference, 296., candidates)
        np.testing.assert_array_equal(ids, refids)
        with self.assertRaises(ValueError):
            retained_reference_observations(reference, 296., np.minimum(signal+1, 320.), 320., candidates)
        with self.assertRaises(ValueError):
            retained_reference_observations(reference, 296., reference, 295., candidates)

    def test_higher_ceiling_uses_original_signal(self):
        z = np.array([295., 310., 330.])
        low = camera_at_ceiling(z, z, 296., np.ones(3, bool))
        high = camera_at_ceiling(z, z, 320., np.ones(3, bool))
        np.testing.assert_array_equal(high[1], [295., 310., 320.])
        self.assertFalse(np.array_equal(high[1], np.minimum(low[1], 320.)))
        np.testing.assert_array_equal(high[3], [True, True, False])
        np.testing.assert_array_equal(high[4], [0, 1, 2])

    def test_distance_ignores_omitted_unsaturated_nodes(self):
        points = np.array([[0., 0.], [.0009, 0.], [.001, 0.]])
        truth = np.ones(3)
        pred = (truth, truth, truth-1, truth+1)
        distance, rows = uncertainty_distance(points, np.array([True, False, False]),
            np.array([False, False, True]), np.ones(3)*1e-6, truth, pred)
        self.assertAlmostEqual(distance[2], 1.)
        self.assertEqual(rows[0]['nodes'], 1)
        self.assertEqual(rows[0]['coverage95'], 1.)
        with self.assertRaises(ValueError):
            uncertainty_distance(points, np.zeros(3, bool), np.ones(3, bool), truth, truth, pred)


if __name__ == '__main__':
    unittest.main()
