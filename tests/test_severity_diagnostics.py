import unittest
import numpy as np
from src.severity_diagnostics import region_masks, mean_diagnostics, posterior_diagnostics


class SeverityTests(unittest.TestCase):
    def test_top_mask_and_peak_definitions(self):
        truth=np.arange(2501.)
        masks=region_masks(truth,truth>2400)
        self.assertEqual(masks['top_1pct'].sum(),25)
        rows=mean_diagnostics(truth+2,truth,masks)
        for row in rows:
            self.assertEqual(row['bias_K'],2)
            self.assertEqual(row['peak_signed_error_K'],2)

    def test_z_moments_pit_and_zero_sd(self):
        draws=np.array([[0.,0.,5.],[1.,2.,5.],[2.,4.,5.]])
        mu=draws.mean(0); sd=draws.std(0,ddof=1)
        truth=np.array([2.,0.,7.])
        masks={'overall':np.ones(3,bool)}
        rows,fields=posterior_diagnostics((mu,sd,draws.min(0),draws.max(0),draws),truth,masks)
        self.assertEqual(rows[0]['n_invalid_sd'],1)
        self.assertTrue(np.isnan(fields['z'][2]))
        np.testing.assert_allclose(fields['pit'],[1,1/3,1])
        r=rows[0]
        self.assertAlmostEqual(r['z_rms']**2,r['z_sd']**2+r['z_mean']**2)
