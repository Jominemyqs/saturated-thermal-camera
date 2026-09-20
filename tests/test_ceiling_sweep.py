from __future__ import annotations

import unittest

import numpy as np

from src.ceiling_sweep import artificial_ceiling_schedule, lower_camera_ceiling


def example_camera() -> dict[str, object]:
    fixed = np.array([[True, False], [False, True]])
    frames = []
    for time, values in (
        (0.0, np.array([[1.0, 2.0], [3.0, 4.0]])),
        (1.0, np.array([[2.0, 3.0], [4.0, 5.0]])),
    ):
        frames.append(
            {
                "time": time,
                "time_index": int(time),
                "points": np.array([[0.0, 0.0, time], [1.0, 1.0, time]]),
                "values": values[fixed],
                "sat_mask": values[fixed] >= 5.0,
                "clipped_full": values,
                "saturated_full": values >= 5.0,
            }
        )
    return {
        "frames": frames,
        "fixed_observation_mask": fixed,
        "current": {
            "x_obs": frames[1]["points"],
            "y_obs": frames[1]["values"],
            "sat_mask": frames[1]["sat_mask"],
            "x_pred": frames[1]["points"],
            "threshold": 5.0,
            "clipped_full": frames[1]["clipped_full"],
        },
    }


class CeilingSweepTest(unittest.TestCase):
    def test_artificial_ceiling_schedule_uses_observations_and_is_monotone(self) -> None:
        observed = np.arange(1.0, 101.0)
        schedule = artificial_ceiling_schedule(observed, 100.0, [0.01, 0.1, 0.2])
        ceilings = np.array([row["ceiling_K"] for row in schedule], dtype=float)
        self.assertEqual(ceilings[0], 100.0)
        self.assertTrue(np.all(np.diff(ceilings) < 0.0))
        self.assertEqual(schedule[1]["ceiling_source"], "reference-observation quantile")
        self.assertTrue(np.isclose(np.mean(observed >= ceilings[2]), 0.2))

    def test_lower_camera_ceiling_only_reclips_reference_observation(self) -> None:
        camera = example_camera()
        lowered = lower_camera_ceiling(camera, 3.5)
        expected = np.minimum(camera["frames"][1]["clipped_full"], 3.5)
        self.assertTrue(np.array_equal(lowered["current"]["clipped_full"], expected))
        self.assertTrue(
            np.array_equal(lowered["current"]["saturated_full"], expected >= 3.5)
        )
        self.assertTrue(
            np.array_equal(
                lowered["current"]["y_obs"],
                expected[[[True, False], [False, True]]],
            )
        )
        self.assertEqual(np.max(camera["frames"][1]["clipped_full"]), 5.0)

    def test_lower_camera_ceiling_rejects_higher_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            lower_camera_ceiling(example_camera(), 5.1)
