import unittest
import numpy as np

from src.latent_mask_audit import extend_blocks, additional_layout
from src.unpinning_current_update import propagated_blocks


class LatentMaskAuditTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(71)
        x = rng.normal(size=(6, 6)); k = x@x.T+np.eye(6)
        v = .25; p = np.linalg.inv(k+v*np.eye(6))
        a = rng.normal(size=(6, 6)); z = rng.normal(size=(4, 80, 2))
        self.old = np.array([0, 3]); self.add = np.array([1, 5])
        self.args = (a, p, v, np.array([2, 4]), z)
        oo, po, diag = propagated_blocks(*self.args, self.old, pinned=False)
        mean = np.arange(6.)
        self.base = dict(prediction_mean=mean, observation_mean=mean[self.old],
                         observed_covariance=oo, pred_observed_covariance=po, prediction_variance=diag)

    def test_full_latent_extension_matches_direct_joint_blocks(self):
        _, cross, diag = propagated_blocks(*self.args, self.add, pinned=False)
        ids, b, change = extend_blocks(self.base, self.old, self.add, cross, diag)
        oo, po, var = propagated_blocks(*self.args, ids, pinned=False)
        np.testing.assert_allclose(b['observed_covariance'], oo, atol=1e-12)
        np.testing.assert_allclose(b['pred_observed_covariance'], po, atol=1e-12)
        np.testing.assert_array_equal(b['prediction_variance'], var)
        np.testing.assert_array_equal(b['pred_observed_covariance'][:, np.searchsorted(ids,self.old)],
                                      self.base['pred_observed_covariance'])
        self.assertLess(change, 1e-12)

    def test_reject_hybrid_covariance_substitution(self):
        _, cross, diag = propagated_blocks(*self.args, self.add, pinned=True)
        with self.assertRaises(AssertionError):
            extend_blocks(self.base, self.old, self.add, cross, diag)

    def test_no_additions_preserve_arrays(self):
        ids, b, change = extend_blocks(self.base, self.old, [], None, None)
        np.testing.assert_array_equal(ids, self.old)
        for key in b:self.assertIs(b[key], self.base[key])
        self.assertEqual(change, 0.)

    def test_disjoint_likelihood_and_no_mask_numerical_readings(self):
        args = (self.old, np.array([0, 1]), np.array([3., np.nan]), self.add)
        ids, kinds, values = additional_layout(*args, 'Cmask')
        j = np.searchsorted(ids, self.add)
        np.testing.assert_array_equal(kinds[j], [-1, -1])
        self.assertTrue(np.isnan(values[j]).all())
        with self.assertRaises(ValueError):additional_layout(*args, 'Cmask', readings=np.ones(2))
        ids, kinds, values = additional_layout(*args, 'Cnumeric', readings=np.array([2.,4.]))
        np.testing.assert_array_equal(kinds[np.searchsorted(ids,self.add)], [0,0])
        with self.assertRaises(ValueError):additional_layout(self.old,[0,1],[3.,np.nan],[0],'Cmask')


if __name__ == '__main__':unittest.main()
