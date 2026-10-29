import unittest
import numpy as np
from src.source_amplitude_diagnostic import scale_diagnostic,select_shared_scale


class AmplitudeTests(unittest.TestCase):
    def test_closed_form_and_grid(self):
        s=np.array([1.,2.,3.]); b=np.ones(3)*10; t=b+.8*s
        d=scale_diagnostic(b,s,t,np.array([1.,2.,4.]),[.6,.8,1.,1.2])
        self.assertAlmostEqual(d['kappa_unconstrained'],.8)
        self.assertEqual(d['kappa_grid'],.8)

    def test_unidentifiable_and_boundary(self):
        z=np.zeros(3); o=np.ones(3)
        self.assertFalse(scale_diagnostic(z,z,o,o,[.6,1.3])['identifiable'])
        d=scale_diagnostic(z,o,2*o,o,[.6,1.3])
        self.assertTrue(d['optimum_outside_grid']);self.assertEqual(d['kappa_bounded'],1.3)

    def test_equal_trajectory_not_node_weight(self):
        cases=[(np.zeros(1),np.ones(1),np.ones(1)*.6,np.ones(1)),
               (np.zeros(100),np.ones(100),np.ones(100)*1.2,np.ones(100))]
        k,_=select_shared_scale(cases,[.6,.9,1.2]);self.assertEqual(k,.9)

    def test_scale_commutes_with_linear_propagation(self):
        a=np.array([[.6,.2],[.3,.5]]);q=np.array([4.,8.]);k=.8
        np.testing.assert_allclose(a@(k*q),k*(a@q),atol=1e-15)


if __name__=='__main__':unittest.main()
