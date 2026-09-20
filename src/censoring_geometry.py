from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure, label


FOUR_NEIGHBOR_STRUCTURE = generate_binary_structure(2, 1)


def observed_mask_geometry(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    """Return four-connected labels and erosion depth from an observed mask.

    Boundary pixels have depth zero. Each subsequent four-neighbor erosion
    layer increases depth by one camera-grid pixel. Only ``mask`` is used.
    """
    observed = np.asarray(mask, dtype=bool)
    if observed.ndim != 2:
        raise ValueError("The observed censoring mask must be two-dimensional")
    labels, n_components = label(observed, structure=FOUR_NEIGHBOR_STRUCTURE)
    depth = np.full(observed.shape, np.nan)
    components: list[dict[str, int]] = []
    for component_id in range(1, n_components + 1):
        component = labels == component_id
        remaining = component.copy()
        layer = 0
        while np.any(remaining):
            boundary = remaining & ~binary_erosion(
                remaining,
                structure=FOUR_NEIGHBOR_STRUCTURE,
                border_value=0,
            )
            depth[boundary] = float(layer)
            remaining[boundary] = False
            layer += 1
        components.append(
            {
                "component_id": component_id,
                "component_size": int(np.sum(component)),
                "maximum_depth_pixels": layer - 1,
            }
        )
    if np.any(observed & ~np.isfinite(depth)):
        raise AssertionError("Every observed-censored pixel must receive a depth")
    return labels, depth, components


def local_ordering_edges(
    mask: np.ndarray,
    labels: np.ndarray,
    depth: np.ndarray,
) -> np.ndarray:
    """Orient same-component four-neighbor edges from shallow to deep."""
    observed = np.asarray(mask, dtype=bool)
    component_labels = np.asarray(labels, dtype=int)
    depths = np.asarray(depth, dtype=float)
    if observed.shape != component_labels.shape or observed.shape != depths.shape:
        raise ValueError("Mask, labels, and depth must have matching shapes")
    edges: list[tuple[int, int]] = []
    n_rows, n_columns = observed.shape
    for row in range(n_rows):
        for column in range(n_columns):
            if not observed[row, column]:
                continue
            current = np.ravel_multi_index((row, column), observed.shape)
            for other_row, other_column in ((row + 1, column), (row, column + 1)):
                if other_row >= n_rows or other_column >= n_columns:
                    continue
                if not observed[other_row, other_column]:
                    continue
                if component_labels[other_row, other_column] != component_labels[row, column]:
                    continue
                other = np.ravel_multi_index(
                    (other_row, other_column), observed.shape
                )
                current_depth = depths[row, column]
                other_depth = depths[other_row, other_column]
                if other_depth > current_depth:
                    edges.append((current, other))
                elif current_depth > other_depth:
                    edges.append((other, current))
    if not edges:
        return np.empty((0, 2), dtype=int)
    return np.asarray(edges, dtype=int)


def randomize_edge_orientations(
    edges: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Randomly orient the exact undirected local edges used by geometry."""
    pairs = np.asarray(edges, dtype=int)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("Edges must have shape (n_edges, 2)")
    randomized = pairs.copy()
    flips = np.random.default_rng(seed).random(len(randomized)) < 0.5
    randomized[flips] = randomized[flips, ::-1]
    return randomized


def global_to_local_edges(edges: np.ndarray, selected_indices: np.ndarray) -> np.ndarray:
    """Map full-grid edge indices into an ordered selected-pixel index set."""
    pairs = np.asarray(edges, dtype=int)
    selected = np.asarray(selected_indices, dtype=int).reshape(-1)
    if pairs.size == 0:
        return np.empty((0, 2), dtype=int)
    lookup = {int(index): position for position, index in enumerate(selected)}
    try:
        return np.asarray(
            [(lookup[int(first)], lookup[int(second)]) for first, second in pairs],
            dtype=int,
        )
    except KeyError as error:
        raise ValueError("Every edge endpoint must belong to the selected pixels") from error
