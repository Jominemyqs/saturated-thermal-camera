import unittest
import numpy as np
from src.observation_bridge import subset_observation_blocks, cooling_visibility
from src.dense_censored_gp import sample_censored_gaussian_blocks


class BridgeTests(unittest.TestCase):
    def test_block_sampler_is_marginal_not_joint(self):
        # With an independent observed coordinate, two correlated targets should
        # retain covariance 0.8. The legacy block API cannot represent this term.
        obs = dict(y_obs=np.zeros(1), sat_mask=np.zeros(1, bool), threshold=3.)
        p = sample_censored_gaussian_blocks(obs, prediction_mean=np.zeros(2),
            observation_mean=np.zeros(1), observed_covariance=np.ones((1,1)),
            pred_observed_covariance=np.zeros((2,1)), prediction_variance=np.ones(2),
            noise_sd=.25,n_samples=30000,burn_in=0,thin=1,seed=41)
        covariance = np.cov(p[4],rowvar=False)
        np.testing.assert_allclose(np.diag(covariance),1.,atol=.03)
        self.assertLess(abs(covariance[0,1]),.03)

    def test_subsetting(self):
        obs = dict(y_obs=np.arange(3), sat_mask=np.array([0,1,1], bool), x_obs=np.arange(6).reshape(3,2))
        k = np.arange(9).reshape(3,3)
        blocks = dict(observation_mean=np.arange(3), observed_covariance=k,
                      pred_observed_covariance=k, prediction_variance=np.ones(3), prediction_mean=np.zeros(3))
        a,b = subset_observation_blocks(obs, blocks, [True,False,True])
        np.testing.assert_array_equal(b['observed_covariance'], k[np.ix_([0,2],[0,2])])
        self.assertIs(b['prediction_variance'], blocks['prediction_variance'])
        np.testing.assert_array_equal(a['y_obs'], [0,2])

    def test_flicker_is_not_persistent(self):
        y = np.array([[10,10,8],[9,10,7],[10,9,7],[10,10,7]])
        r = cooling_visibility(y, 10, 0, 3, .25)
        self.assertEqual(r['ever_observable_count'], 2)
        self.assertEqual(r['persistent_observable_count'], 0)

    def test_persistent_and_initial_selection(self):
        y = np.array([[10,8],[9,7],[9,7],[9,7]])
        r = cooling_visibility(y, 10, 0, 3, .25)
        self.assertEqual(r['persistent_observable_count'], 1)
        self.assertEqual(r['target_censored_count'], 1)
        r = cooling_visibility(y, 10, 0, 3, .25, np.array([False,True]))
        self.assertEqual(r['persistent_observable_stride5_count'], 0)
