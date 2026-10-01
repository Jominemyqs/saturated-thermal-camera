import unittest
import numpy as np
from src.native_surface import surface_mesh,propagate,hottest_area_weights,quadrature


class NativeSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.points=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.4,.6]])
        self.tri=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        self.mesh=surface_mesh(self.points,self.tri)

    def test_area_and_linear_integral(self):
        self.assertAlmostEqual(self.mesh.weights.sum(),1.)
        self.assertAlmostEqual(self.mesh.weights@self.points[:,0],.5)
        np.testing.assert_allclose(self.mesh.stiffness.toarray(),self.mesh.stiffness.toarray().T)
        np.testing.assert_allclose(self.mesh.stiffness@np.ones(5),0,atol=1e-14)
        q,w,_=quadrature(self.mesh)
        self.assertAlmostEqual(w.sum(),1.)
        self.assertAlmostEqual(w@q[:,1],.5)
        q,w,_=quadrature(self.mesh,refinement=2)
        self.assertAlmostEqual(w.sum(),1.)
        self.assertAlmostEqual(np.sum(w*q[:,1]),.5)

    def test_heat_decay_and_linear_residuals(self):
        x=np.array([0.,0.,0.,0.,2.])
        kw=dict(dt=.1,diffusivity=.01,cooling_rate=.2)
        expected=2*np.exp(-.1*(.01*self.mesh.stiffness[4,4]/self.mesh.weights[4]+.2))
        y=propagate(self.mesh,x,**kw)
        self.assertAlmostEqual(y[4],expected)
        np.testing.assert_array_equal(y[:4],0.)
        both=propagate(self.mesh,np.stack([x,-x]),**kw)
        np.testing.assert_allclose(both[0],-both[1])
        self.assertLess(np.sum(self.mesh.weights*y*y),np.sum(self.mesh.weights*x*x))

    def test_orientation_and_area_hot_region(self):
        reversed_mesh=surface_mesh(self.points,self.tri[:,::-1])
        np.testing.assert_allclose(self.mesh.stiffness.toarray(),reversed_mesh.stiffness.toarray())
        h=hottest_area_weights(np.arange(5),self.mesh.weights,.01)
        self.assertAlmostEqual(h.sum(),.01)
        self.assertGreater(h[4],0.)
        self.assertEqual(np.count_nonzero(h),1)


if __name__=='__main__':unittest.main()
