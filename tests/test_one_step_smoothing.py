import unittest
import numpy as np
from scipy.stats import norm
from src.one_step_smoothing import censored_log_likelihood, importance_weights, weighted_prediction
from src.dense_censored_gp import sample_censored_gaussian_blocks


class SmoothingTests(unittest.TestCase):
    def test_likelihood(self):
        got=censored_log_likelihood(np.array([[1.,3.]]),np.array([0.,2.]),[False,True],2.,1.)
        np.testing.assert_allclose(got,norm.logpdf(0,loc=1)+norm.logcdf(1))

    def test_weights(self):
        w,d=importance_weights(np.zeros(100))
        self.assertAlmostEqual(d['ess'],100)
        _,d=importance_weights(np.r_[0.,np.full(99,-1000.)])
        self.assertEqual(d['ess'],1.)

    def test_scalar_stochastic_smoothing_against_gaussian_solution(self):
        rng=np.random.default_rng(12)
        original=rng.normal(size=80000)
        future=original+.5*rng.normal(size=len(original))
        logs=censored_log_likelihood(future[:,None],np.array([1.]),[False],3.,.5)
        w,_=importance_weights(logs)
        mean=float(np.sum(w*original))
        variance=float(np.sum(w*(original-mean)**2))
        self.assertAlmostEqual(mean,2/3,delta=.02)
        self.assertAlmostEqual(variance,1/3,delta=.02)

    def test_weighted_crps(self):
        x=np.array([[0.,2.],[3.,4.],[6.,8.]])
        w=np.array([.2,.3,.5])
        y=np.array([1.,5.])
        expected=w@abs(x-y)-.5*np.einsum('i,j,ijk->k',w,w,abs(x[:,None,:]-x[None,:,:]))
        np.testing.assert_allclose(weighted_prediction(x,w,y)[4],expected)

    def test_joint_opt_in_preserves_marginal_parameters(self):
        obs=dict(y_obs=np.zeros(1),sat_mask=np.zeros(1,bool),threshold=3.)
        args=dict(prediction_mean=np.zeros(2),observation_mean=np.zeros(1),observed_covariance=np.ones((1,1)),
            pred_observed_covariance=np.zeros((2,1)),prediction_variance=np.ones(2),noise_sd=.25,
            n_samples=30000,burn_in=0,thin=1,seed=41)
        old=sample_censored_gaussian_blocks(obs,**args)
        new=sample_censored_gaussian_blocks(obs,**args,prediction_covariance=np.array([[1.,.8],[.8,1.]]))
        np.testing.assert_array_equal(old[0],new[0])
        np.testing.assert_array_equal(old[1],new[1])
        self.assertAlmostEqual(np.cov(new[4],rowvar=False)[0,1],.8,delta=.03)
