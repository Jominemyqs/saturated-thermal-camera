import unittest
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix

from src.censored_gp import RBFConfig, rbf_covariance
from src.previous_correlation_audit import product
from src.previous_unpinning_audit import (
    noisy_precision, conditional_camera, latent_from_camera,
    conditional_covariance_block, mixture_moments,
    propagated_conditional_residual, propagated_state_components,
)


class PreviousUnpinningTests(unittest.TestCase):
    def setUp(self):
        self.points = np.column_stack([np.arange(6)*.2, np.zeros(6)])
        self.cfg = RBFConfig(mean_temp=3., signal_sd=1.5, lengthscale=.4, noise_sd=.25)
        self.k = rbf_covariance(self.points, self.points, self.cfg)
        self.p, self.v, _ = noisy_precision(self.points, self.cfg, block_size=2)
        self.camera = np.array([2.9, 3.2, 4., 4., 3.4, 3.1])
        self.sat = np.array([False, False, True, True, False, False])

    def test_precision_conditioning_equals_direct_schur_complement(self):
        ids, mu, cov, _ = conditional_camera(self.p, self.camera, self.sat, 3.)
        u = np.flatnonzero(~self.sat); v = self.k + self.v*np.eye(6)
        f = cho_factor(v[np.ix_(u, u)], lower=True)
        vsu = v[np.ix_(ids, u)]
        expected = 3. + vsu.dot(cho_solve(f, self.camera[u]-3.))
        expected_cov = v[np.ix_(ids, ids)]-vsu.dot(cho_solve(f, vsu.T))
        np.testing.assert_allclose(mu, expected, atol=1e-12)
        np.testing.assert_allclose(cov, expected_cov, atol=1e-12)

    def test_saturated_camera_magnitudes_do_not_enter_conditioning(self):
        first = conditional_camera(self.p, self.camera, self.sat, 3.)
        y = self.camera.copy(); y[self.sat] = np.nan
        second = conditional_camera(self.p, y, self.sat, 3.)
        for a, b in zip(first, second): np.testing.assert_array_equal(a, b)

    def test_conditional_latent_covariance_and_cross_terms(self):
        ids, _, _, z = conditional_camera(self.p, self.camera, self.sat, 3.)
        m, w, diag = latent_from_camera(self.p, self.v, z, 3., ids)
        expected_mean = 3.+self.k.dot(self.p.dot(z-3.))
        c = self.k-self.k.dot(self.p.dot(self.k))
        np.testing.assert_allclose(m, expected_mean, atol=1e-12)
        np.testing.assert_allclose(diag, np.diag(c), atol=1e-12)
        np.testing.assert_allclose(w, self.k.dot(self.p)[:, ids], atol=1e-12)
        np.testing.assert_allclose(conditional_covariance_block(self.p, self.v, [0, 2], [2, 4]),
                                   c[np.ix_([0, 2], [2, 4])], atol=1e-12)

    def test_propagation_includes_unsaturated_censored_cross_covariance(self):
        rng = np.random.default_rng(71)
        ids, mu, cov, z = conditional_camera(self.p, self.camera, self.sat, 3.)
        draws = rng.multivariate_normal(mu, cov, size=200)
        a = rng.normal(size=(4, 6))
        rv, ac = propagated_conditional_residual(self.p, self.v, a, ids, block_size=2)
        result = propagated_state_components(a, self.p, self.v, ids, draws, ac, rv)
        sr, sc = propagated_conditional_residual(self.p, self.v, csr_matrix(a), ids, block_size=2)
        sparse_result = propagated_state_components(csr_matrix(a), self.p, self.v, ids, draws, sc, sr)
        for key in result:
            np.testing.assert_allclose(sparse_result[key], result[key], atol=1e-12)
        _, w, _ = latent_from_camera(self.p, self.v, z, 3., ids)
        c = self.v*np.eye(6)-self.v**2*self.p
        sigma = product(product(w, np.cov(draws.T)), w.T)+c
        expected = np.diag(product(product(a, sigma), a.T))
        np.testing.assert_allclose(result['total'], expected, atol=1e-11)
        np.testing.assert_allclose(result['censored'],
            np.diag(a[:, ids].dot(sigma[np.ix_(ids, ids)]).dot(a[:, ids].T)), atol=1e-11)
        self.assertGreater(abs(result['cross']).max(), 1e-4)

    def test_mixture_and_pinned_bridge_have_same_censored_marginals(self):
        ids, mu, cov, z = conditional_camera(self.p, self.camera, self.sat, 3.)
        base, w, diag = latent_from_camera(self.p, self.v, z, 3., ids)
        draws = np.random.default_rng(12).multivariate_normal(mu, cov, size=(4, 80))
        m, variance, _ = mixture_moments(base, w, draws, mu, diag, block_size=2)
        m_s, var_s, _ = mixture_moments(base[ids], w[ids], draws, mu, diag[ids])
        np.testing.assert_allclose(m[ids], m_s, atol=1e-12)
        np.testing.assert_allclose(variance[ids], var_s, atol=1e-12)
        self.assertTrue(np.all(variance[~self.sat] > 0))

    def test_all_numerical_conditioning_and_small_noise_limit(self):
        ids, mu, cov, z = conditional_camera(self.p, self.camera, np.zeros(6, bool), 3.)
        self.assertEqual(len(ids), 0); self.assertEqual(cov.shape, (0, 0))
        m, _, diagonal = latent_from_camera(self.p, self.v, z, 3., ids)
        np.testing.assert_allclose(m, 3.+self.k.dot(self.p.dot(z-3.)), atol=1e-12)
        self.assertTrue(np.all(diagonal > 0))
        cfg = RBFConfig(mean_temp=3., signal_sd=1.5, lengthscale=.4,
                        noise_sd=1e-5, relative_jitter=0.)
        p, v, _ = noisy_precision(self.points, cfg)
        m, _, diagonal = latent_from_camera(p, v, z, 3., ids)
        np.testing.assert_allclose(m, z, atol=1e-7)
        self.assertLess(diagonal.max(), 1.1e-10)


if __name__ == '__main__':
    unittest.main()
