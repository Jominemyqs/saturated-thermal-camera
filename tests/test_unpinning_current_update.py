import unittest
import numpy as np
from scipy.sparse import csr_matrix

from src.censored_gp import RBFConfig, rbf_covariance
from src.previous_unpinning_audit import noisy_precision
from src.unpinning_current_update import propagated_blocks
from src.native_mask_pilot import sample_mixed_noisy_blocks
from src.dense_censored_gp import sample_censored_gaussian_blocks


class CurrentUnpinningTests(unittest.TestCase):
    def setUp(self):
        self.xy=np.column_stack([np.arange(6)*.2,np.zeros(6)])
        self.cfg=RBFConfig(mean_temp=3.,signal_sd=1.5,lengthscale=.4,noise_sd=.25)
        self.p,self.v,_=noisy_precision(self.xy,self.cfg)
        self.s=np.array([2,3]);self.obs=np.array([0,2,4])
        rng=np.random.default_rng(165)
        self.a=rng.normal(size=(6,6));self.z=rng.normal(size=(4,50,2))
        c=self.v*np.eye(6)-self.v**2*self.p
        w=c[:,self.s]/self.v
        self.sigma=w@np.cov(self.z.reshape(-1,2).T)@w.T+c

    def test_full_and_pinned_match_explicit_joint_covariance(self):
        for pinned in [False,True]:
            a=self.a.copy()
            if pinned:a[:,[0,1,4,5]]=0
            expected=a@self.sigma@a.T
            oo,po,diag=propagated_blocks(self.a,self.p,self.v,self.s,self.z,self.obs,pinned=pinned)
            np.testing.assert_allclose(oo,expected[np.ix_(self.obs,self.obs)],atol=1e-11)
            np.testing.assert_allclose(po,expected[:,self.obs],atol=1e-11)
            np.testing.assert_allclose(diag,np.diag(expected),atol=1e-11)

    def test_sparse_path_matches_dense(self):
        for pinned in [False,True]:
            dense=propagated_blocks(self.a,self.p,self.v,self.s,self.z,self.obs,pinned=pinned)
            sparse=propagated_blocks(csr_matrix(self.a),self.p,self.v,self.s,self.z,self.obs,
                                    pinned=pinned,operator_precision=self.a@self.p)
            for d,s in zip(dense,sparse):np.testing.assert_allclose(d,s,atol=1e-11)

    def test_cross_covariance_is_not_diagonal_replacement(self):
        _,po,_=propagated_blocks(self.a,self.p,self.v,self.s,self.z,self.obs,pinned=False)
        wrong=(self.a@np.diag(np.diag(self.sigma))@self.a.T)[:,self.obs]
        self.assertGreater(np.max(abs(po-wrong)),.01)

    def test_lower_only_current_sampler_reproduces_historical_path(self):
        k=rbf_covariance(self.xy,self.xy,self.cfg)
        ids=np.array([0,2,4]);sat=np.array([False,True,False]); y=np.array([3.1,3.5,3.2])
        blocks=dict(prediction_mean=np.full(6,3.),observation_mean=np.full(3,3.),
                    observed_covariance=k[np.ix_(ids,ids)],pred_observed_covariance=k[:,ids],
                    prediction_variance=np.diag(k))
        settings=dict(noise_sd=.5,n_samples=80,burn_in=30,thin=1,seed=44)
        expected=sample_censored_gaussian_blocks(dict(y_obs=y,sat_mask=sat,threshold=3.5),
                    **blocks,**settings,truncated_sampler='hmc')
        values=y.copy();values[sat]=np.nan
        actual,_=sample_mixed_noisy_blocks(ids=ids,kinds=sat.astype(int),numerical_values=values,
                    thresholds=3.5,**blocks,**settings)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
