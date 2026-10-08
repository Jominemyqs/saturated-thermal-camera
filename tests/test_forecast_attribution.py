import unittest
import numpy as np
from src.native_surface import surface_mesh
from src.matched_projection import MatchedHeatOperator
from src.forecast_attribution import forecast_components, component_metrics


class ForecastAttributionTests(unittest.TestCase):
    def test_exact_budget_with_floor_and_cutoff(self):
        mesh=surface_mesh(np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.4,.6]]),
                          np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]]))
        op=MatchedHeatOperator(mesh,np.linspace(0,1,5),np.linspace(0,1,5),nx=21,ny=21)
        previous=np.array([8.,11.,12.,10.,13.]);prevtruth=previous+1
        truth=previous+2;flux=np.array([-3.,1.,2.,0.,5.])
        a=forecast_components(op,previous,prevtruth,truth,flux,ambient=10.,source_coupling=.4,
                              source_flux_threshold=2.,dt=.1,diffusivity=.02,cooling_rate=.1)
        np.testing.assert_allclose(a['source'],[0,0,.08,0,.2])
        np.testing.assert_allclose(a['base_error'],a['source']-a['base_needed'],atol=1e-12)
        np.testing.assert_allclose(a['base_error'],a['propagated_state_error']+a['oracle_error'],atol=1e-12)
        self.assertTrue(np.all(a['base_propagated_excess']>=0))
        rows,budget=component_metrics(a,mesh.weights)
        np.testing.assert_allclose(rows[2]['bias_K'],budget['source_mean_K']-budget['base_needed_mean_K'],atol=1e-12)
        with self.assertRaises(ValueError):component_metrics(a,np.zeros(5))


if __name__=='__main__':unittest.main()
