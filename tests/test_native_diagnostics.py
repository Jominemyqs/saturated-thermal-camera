"""Geometry-only diagnostic selection and native-area source centroids."""
import unittest
import numpy as np
from src.native_diagnostics import source_centroid, section_nodes


class NativeDiagnosticTests(unittest.TestCase):
    def test_area_weighted_centroid(self):
        points=np.array([[0.,0.],[1.,0.],[0.,1.]])
        weights=np.array([1.,3.,100.]);flux=np.array([2.,2.,.5])
        np.testing.assert_allclose(source_centroid(points,weights,flux,1.),[.75,0.])
        with self.assertRaises(ValueError):source_centroid(points,weights,flux,10.)

    def test_sections_use_native_nodes(self):
        x,y=np.meshgrid(np.linspace(-1,1,11),np.linspace(-1,1,11))
        points=np.column_stack([x.ravel(),y.ravel()])
        ids,along,offset=section_nodes(points,np.zeros(2),np.array([1.,0.]),half_length=1.)
        self.assertEqual(len(ids),11)
        np.testing.assert_allclose(points[ids,0],along)
        np.testing.assert_allclose(offset,0.)
        self.assertTrue(np.all(np.diff(along)>0))


if __name__=='__main__':unittest.main()
