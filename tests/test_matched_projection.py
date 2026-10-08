import unittest
import numpy as np
from src.native_surface import surface_mesh
from src.matched_projection import bilinear_matrix, triangle_matrix, MatchedHeatOperator


class MatchedProjectionTests(unittest.TestCase):
    def setUp(self):
        self.mesh = surface_mesh(np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.4,.6]]),
                                 np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]]))

    def test_coordinate_interpolation(self):
        p = np.array([[.1,.2],[.8,.7],[1.,1.]])
        a = triangle_matrix(self.mesh, p)
        np.testing.assert_allclose(a@self.mesh.points, p)
        xs = np.array([0.,.3,1.]); ys = np.array([0.,.5,1.])
        x,y = np.meshgrid(xs,ys)
        b = bilinear_matrix(xs,ys,p)
        np.testing.assert_allclose(b@(x+2*y+x*y).ravel(),p[:,0]+2*p[:,1]+p[:,0]*p[:,1])

    def test_same_representation_and_batch_linearity(self):
        op = MatchedHeatOperator(self.mesh,np.linspace(0,1,5),np.linspace(0,1,5),nx=21,ny=21)
        values = self.mesh.points[:,0]+2*self.mesh.points[:,1]
        kw = dict(dt=.01,diffusivity=.02,cooling_rate=.1)
        a = op.propagate(values,'native',**kw)
        b = op.propagate(values,'projected',**kw)
        np.testing.assert_allclose(a,b,atol=1e-14)
        draws = op.propagate(np.stack([values,-values]),'native',batch_size=1,**kw)
        np.testing.assert_allclose(draws,np.stack([a,-a]),atol=1e-14)

    def test_transport_sign_and_unchanged_source(self):
        from scipy.spatial import Delaunay
        x,y=np.meshgrid(np.linspace(0,1,11),np.linspace(0,1,11))
        points=np.column_stack([x.ravel(),y.ravel()])
        mesh=surface_mesh(points,Delaunay(points).simplices)
        op=MatchedHeatOperator(mesh,np.linspace(0,1,11),np.linspace(0,1,11),nx=11,ny=11)
        pulse=np.zeros(len(points)); pulse[5*11+4]=1.
        kw=dict(dt=1.,diffusivity=0.,cooling_rate=0.)
        shifted=op.propagate(pulse,'native',displacement=np.array([.2,-.1]),**kw)
        self.assertEqual(np.argmax(shifted),4*11+6)
        self.assertAlmostEqual(shifted.max(),1.)
        np.testing.assert_allclose(op.propagate(pulse,'native',displacement=np.zeros(2),**kw),pulse,atol=1e-14)
        source=np.zeros(len(points));source[2*11+2]=3.
        mean=op.mean(pulse+10,source,'native',ambient=10,source_coupling=1.,
                     source_flux_threshold=1.,displacement=np.array([.2,-.1]),**kw)
        np.testing.assert_allclose(mean,10+shifted+source,atol=1e-14)


if __name__=='__main__': unittest.main()
