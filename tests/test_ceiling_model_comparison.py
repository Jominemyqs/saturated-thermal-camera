import unittest
from types import SimpleNamespace
import numpy as np
from src.ceiling_model_comparison import rbf_blocks,evaluate_prediction


class ComparisonTests(unittest.TestCase):
    def test_mean_only_rbf_change(self):
        p=SimpleNamespace(ambient=0.,xs=np.array([0.,1.]),ys=np.array([0.,1.]))
        points=np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.],[1.,1.,1.]])
        obs=dict(x_pred=points,x_obs=points[[0,3]])
        a=rbf_blocks(p,obs,np.zeros((2,2)),2.,1.,.5)
        b=rbf_blocks(p,obs,np.ones((2,2)),2.,1.,.5)
        for key in ['observed_covariance','pred_observed_covariance','prediction_variance']:
            np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(b['observation_mean'],[1.,1.])

    def test_fixed_top_mask_and_crps(self):
        truth=np.arange(100.).reshape(10,10)
        mean=truth.ravel()
        draws=np.stack([mean-1,mean+1])
        prediction=(mean,np.ones(100),mean-1,mean+1,draws)
        scored=evaluate_prediction(prediction,truth,0.,truth>=80)
        self.assertEqual(scored['top_1pct_n_pixels'],1)
        self.assertEqual(scored['field_excess_rel_l2'],0.)
        self.assertEqual(scored['overall_crps_K'],0.)
