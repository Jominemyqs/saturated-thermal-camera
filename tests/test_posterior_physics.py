from __future__ import annotations

import unittest

import numpy as np

from src.censored_gp import RBFConfig, rbf_covariance
from src.thermal_posterior_physics import translate_field


class TranslationConventionTest(unittest.TestCase):
    def test_positive_x_displacement_moves_peak_to_positive_x(self):
        field = np.zeros((7, 7))
        field[3, 3] = 1.0
        translated = translate_field(
            field,
            displacement=np.array([0.25, 0.0]),
            dx=0.25,
            dy=0.25,
        )
        self.assertEqual(np.unravel_index(np.argmax(translated), translated.shape), (3, 4))

    def test_positive_y_displacement_moves_peak_to_positive_y(self):
        field = np.zeros((7, 7))
        field[3, 3] = 1.0
        translated = translate_field(
            field,
            displacement=np.array([0.0, 0.25]),
            dx=0.25,
            dy=0.25,
        )
        self.assertEqual(np.unravel_index(np.argmax(translated), translated.shape), (4, 3))


class FixedResidualTest(unittest.TestCase):
    def test_mean_function_does_not_change_rbf_covariance(self):
        points = np.array([[0.0, 0.0], [0.5, 0.25], [1.0, -0.25]])
        first = RBFConfig(
            mean_temp=0.0,
            mean_function=lambda values: np.zeros(len(values)),
            signal_sd=2.0,
            lengthscale=0.7,
            noise_sd=0.2,
        )
        second = RBFConfig(
            mean_temp=0.0,
            mean_function=lambda values: np.ones(len(values)),
            signal_sd=2.0,
            lengthscale=0.7,
            noise_sd=0.2,
        )
        np.testing.assert_array_equal(
            rbf_covariance(points, points, first),
            rbf_covariance(points, points, second),
        )


if __name__ == "__main__":
    unittest.main()
