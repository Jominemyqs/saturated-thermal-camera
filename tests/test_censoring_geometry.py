import unittest

import numpy as np

from src.censored_gp import RBFConfig
from src.censoring_geometry import (
    global_to_local_edges,
    local_ordering_edges,
    observed_mask_geometry,
    randomize_edge_orientations,
)
from src.preference_censored_gp import (
    preference_log_likelihood,
    sample_preference_gp_multiple_chains,
)


class ObservedMaskGeometryTest(unittest.TestCase):
    def test_square_depth_starts_at_zero_and_increases_inward(self):
        mask = np.ones((5, 5), dtype=bool)

        labels, depth, components = observed_mask_geometry(mask)

        self.assertTrue(np.all(labels == 1))
        self.assertEqual(depth[0, 0], 0.0)
        self.assertEqual(depth[1, 1], 1.0)
        self.assertEqual(depth[2, 2], 2.0)
        self.assertEqual(components[0]["component_size"], 25)

    def test_edges_are_local_same_component_and_increasing_depth(self):
        mask = np.zeros((5, 7), dtype=bool)
        mask[:, :5] = True
        mask[0, 6] = True
        labels, depth, _ = observed_mask_geometry(mask)

        edges = local_ordering_edges(mask, labels, depth)
        flat_depth = depth.ravel()
        flat_labels = labels.ravel()

        self.assertTrue(np.all(flat_depth[edges[:, 1]] > flat_depth[edges[:, 0]]))
        self.assertTrue(np.all(flat_labels[edges[:, 1]] == flat_labels[edges[:, 0]]))
        self.assertFalse(np.any(edges == np.ravel_multi_index((0, 6), mask.shape)))

    def test_random_control_preserves_undirected_edges(self):
        edges = np.array([[1, 2], [3, 4], [5, 6]])
        randomized = randomize_edge_orientations(edges, seed=3)

        expected = {tuple(sorted(edge)) for edge in edges}
        actual = {tuple(sorted(edge)) for edge in randomized}

        self.assertEqual(actual, expected)
        self.assertEqual(len(randomized), len(edges))

    def test_global_edges_map_to_selected_order(self):
        edges = np.array([[7, 11], [11, 14]])
        selected = np.array([4, 7, 11, 14])

        mapped = global_to_local_edges(edges, selected)

        np.testing.assert_array_equal(mapped, [[1, 2], [2, 3]])


class PreferenceLikelihoodTest(unittest.TestCase):
    def test_preferred_order_has_larger_likelihood(self):
        pairs = np.array([[0, 1]])
        ordered = preference_log_likelihood(
            np.array([1.1, 2.0]),
            threshold=1.0,
            noise_sd=0.2,
            preference_pairs=pairs,
            tau=0.5,
        )
        reversed_order = preference_log_likelihood(
            np.array([2.0, 1.1]),
            threshold=1.0,
            noise_sd=0.2,
            preference_pairs=pairs,
            tau=0.5,
        )

        self.assertGreater(ordered, reversed_order)

    def test_multiple_chains_pool_requested_number_of_draws(self):
        points = np.column_stack(
            [np.arange(3, dtype=float), np.zeros(3), np.zeros(3)]
        )
        observations = {
            "x_obs": points[[0, 1]],
            "y_obs": np.array([0.1, 1.0]),
            "sat_mask": np.array([False, True]),
            "x_pred": points,
            "threshold": 1.0,
        }
        config = RBFConfig(
            mean_temp=0.0,
            signal_sd=1.0,
            lengthscale=1.0,
            noise_sd=0.2,
        )

        prediction, diagnostics = sample_preference_gp_multiple_chains(
            observations,
            config,
            preference_pairs=np.empty((0, 2), dtype=int),
            tau=0.5,
            n_chains=2,
            samples_per_chain=12,
            burn_in=15,
            thin=1,
            seed=7,
        )

        self.assertEqual(prediction[0].shape, (3,))
        self.assertEqual(prediction[4].shape, (24, 3))
        self.assertTrue(np.all(np.isfinite(prediction[0])))
        self.assertEqual(diagnostics["pooled_samples"], 24)
        self.assertEqual(diagnostics["n_saturated"], 1)


if __name__ == "__main__":
    unittest.main()
