import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

from src.physical_source_response import validate_geometry, solve_independent_response, replace_increment


class ResponseTests(unittest.TestCase):
    def geometry(self):
        return dict(points=np.eye(4, 3), cells=np.array([[0, 1, 2, 3]]),
                    surface_ids=np.array([0, 1, 2]), time=np.array(.001), previous_time=np.array(.0005))

    def blocks(self):
        return dict(prediction_mean=np.array([300., 310., 320.]), observation_mean=np.array([300., 320.]),
                    observed_covariance=np.eye(2), pred_observed_covariance=np.ones((3, 2)),
                    prediction_variance=np.ones(3))

    def test_no_thermal_input_allowed(self):
        for key in ['previous', 'truth', 'HeatFluxZ', 'temperature']:
            data = self.geometry(); data[key] = np.zeros(4)
            with self.assertRaises(ValueError): validate_geometry(data)

    def test_zero_initial_and_response_load_only(self):
        data = self.geometry(); problem = object()
        with patch('src.fem_replay.build_problem', return_value=problem) as build, \
             patch('src.fem_replay.solve_problem', return_value=(np.ones(4), 0.)) as solve:
            solve_independent_response(data, SimpleNamespace(parameters={'dt': .0005}))
            np.testing.assert_array_equal(build.call_args.args[2], np.zeros(4))
            solve.assert_called_once_with(problem, 'response')

    def test_mean_only_and_observation_consistency(self):
        b = self.blocks(); old = b['prediction_mean'].copy()
        new = replace_increment(b, np.array([0, 2]), np.array([1., 2., 3.]), np.array([4., 5., 6.]))
        np.testing.assert_array_equal(new['prediction_mean'], old+3)
        np.testing.assert_array_equal(new['observation_mean'], new['prediction_mean'][[0, 2]])
        np.testing.assert_array_equal(b['prediction_mean'], old)
        for key in ['observed_covariance', 'pred_observed_covariance', 'prediction_variance']:
            self.assertIs(new[key], b[key])

    def test_identity_response(self):
        b = self.blocks(); inc = np.ones(3)
        new = replace_increment(b, np.array([0, 2]), inc, inc)
        np.testing.assert_array_equal(new['prediction_mean'], b['prediction_mean'])

    def test_invalid_response_rejected(self):
        for response in [np.ones(2), np.full(3, np.nan)]:
            with self.assertRaises(ValueError):
                replace_increment(self.blocks(), np.array([0, 2]), np.ones(3), response)


if __name__ == '__main__': unittest.main()
