import unittest

from src.thermal_plotting import TAIL_EXCESS_MIN_K, tail_temperature_norm


class ThermalPlottingTest(unittest.TestCase):
    def test_tail_norm_uses_ambient_offset_and_camera_ceiling(self):
        norm = tail_temperature_norm(300.0, 325.0)

        self.assertEqual(norm.vmin, 300.0 + TAIL_EXCESS_MIN_K)
        self.assertEqual(norm.vmax, 325.0)
        self.assertEqual(norm.gamma, 0.60)

    def test_tail_norm_keeps_a_nonzero_range_for_low_ceiling(self):
        norm = tail_temperature_norm(300.0, 300.1)

        self.assertEqual(norm.vmin, 300.0 + TAIL_EXCESS_MIN_K)
        self.assertEqual(norm.vmax, norm.vmin + 0.5)


if __name__ == "__main__":
    unittest.main()
