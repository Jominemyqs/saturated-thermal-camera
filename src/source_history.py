"""Source-history means using the inherited discrete thermal forecast."""
from __future__ import annotations

import numpy as np

from src.thermal_posterior_physics import current_source_field, diffuse_and_cool


def source_history_mean(prepared, initial_mean, *, start_index, current_index,
                        diffusivity, cooling_rate, source_coupling,
                        source_flux_threshold):
    if not 0 <= start_index < current_index < len(prepared.times):
        raise ValueError("History must start before the target within the recorded times")
    state = np.asarray(initial_mean, dtype=float)
    for index in range(start_index + 1, current_index + 1):
        background = diffuse_and_cool(
            prepared, state, previous_index=index - 1, current_index=index,
            diffusivity=diffusivity, cooling_rate=cooling_rate,
        )
        source = current_source_field(
            prepared, previous_index=index - 1, current_index=index,
            source_coupling=source_coupling,
            source_flux_threshold=source_flux_threshold,
        )
        state = prepared.ambient + background + source
    return state
