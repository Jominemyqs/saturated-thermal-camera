import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/50_thermal_oracle_forecast_decomposition.py'
spec=importlib.util.spec_from_file_location('oracle_diagnostic',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OracleDiagnosticTest(unittest.TestCase):
    def test_exact_top25_and_empty_status(self):
        y=np.arange(100,dtype=float)
        mask=module.hottest(y)
        self.assertEqual(mask.sum(),25)
        self.assertTrue(np.all(mask[75:]))
        result=module.stats(y,np.zeros(100,bool))
        self.assertEqual(result['n_pixels'],0)
        self.assertTrue(np.isnan(result['bias_K']))

    def test_nonlinear_floor_still_has_exact_decomposition(self):
        previous=np.array([-1.,2.,4.]); estimate=np.array([-3.,1.,7.])
        truth=np.array([1.,4.,8.])
        forecast=lambda x: .8*np.maximum(x,0)+1
        state=forecast(estimate)-forecast(previous)
        physics=forecast(previous)-truth
        total=forecast(estimate)-truth
        np.testing.assert_allclose(total,state+physics,atol=1e-14)
        self.assertAlmostEqual(np.mean(total**2),np.mean(state**2)+np.mean(physics**2)+2*np.mean(state*physics))
        self.assertFalse(np.allclose(state,.8*(estimate-previous)))


if __name__=='__main__': unittest.main()
