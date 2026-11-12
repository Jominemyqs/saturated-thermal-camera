import unittest
from types import SimpleNamespace
import numpy as np
from src.mask_generalization import select_heldout, retained_layout, align_extended_prior, verify_prior_layout
from src.latent_mask_audit import additional_layout
from src.native_mask_pilot import spread_nodes


class GeneralizationTests(unittest.TestCase):
    def test_selection_ignores_order_and_development(self):
        records = [SimpleNamespace(family=f, run_index=i, name=f+'_'+str(i))
                   for f in ('Diagonal','Horizontal','Spiral') for i in range(1,5)]
        dev = [f+'_4' for f in ('Diagonal','Horizontal','Spiral')]
        expected = [f+'_'+str(i) for f in ('Diagonal','Horizontal','Spiral') for i in (1,3)]
        self.assertEqual(select_heldout(records, dev), expected)
        self.assertEqual(select_heldout(records[::-1], dev), expected)

    def test_retention_and_disjoint_flags(self):
        candidates = np.array([1,0,0,0,0], bool)
        refsat = np.array([0,1,1,0,0], bool)
        ids,kinds,values = retained_layout(candidates,refsat,np.array([2,5,8,1,1]),8)
        np.testing.assert_array_equal(ids,[0,1,2])
        np.testing.assert_array_equal(kinds,[0,0,1])
        np.testing.assert_array_equal(values[:2],[2,5])
        self.assertTrue(np.isnan(values[2]))
        pool = ~(candidates|refsat)
        u = spread_nodes(np.column_stack([np.arange(5),np.zeros(5)]),pool,256)
        self.assertFalse(np.isin(u,ids).any())

    def test_extended_blocks_match_likelihood_order(self):
        rng=np.random.default_rng(5);a=rng.normal(size=(6,6));k=a@a.T+np.eye(6)
        mean=np.arange(6.)+290.;old=np.array([0,3,5]);u=np.array([1,2]);order=np.r_[old,u]
        base=dict(prediction_mean=mean,observation_mean=mean[old],prediction_variance=np.diag(k),
                  observed_covariance=k[np.ix_(old,old)],pred_observed_covariance=k[:,old])
        wrong=dict(base,observation_mean=mean[order],observed_covariance=k[np.ix_(order,order)],pred_observed_covariance=k[:,order])
        ids,kinds,values=additional_layout(old,np.zeros(3,int),mean[old],u,'Cmask')
        with self.assertRaises(AssertionError):verify_prior_layout(wrong,ids)
        fixed_ids,fixed,_=align_extended_prior(base,old,u,wrong)
        np.testing.assert_array_equal(fixed_ids,ids)
        verify_prior_layout(fixed,ids)
        np.testing.assert_array_equal(fixed['pred_observed_covariance'],k[:,ids])
        np.testing.assert_array_equal(fixed['observed_covariance'],k[np.ix_(ids,ids)])
        np.testing.assert_array_equal(ids[kinds==-1],u)


if __name__ == '__main__':
    unittest.main()
