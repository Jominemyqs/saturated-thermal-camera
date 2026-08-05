from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial import Delaunay


_TRAJECTORY_PATTERN = re.compile(
    r"(?:Copy of )?(Diagonal|Horizontal|Spiral)ScanPath_(\d+)\.xdmf$"
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


def _hdf_reference(data_item: ET.Element) -> tuple[str, str]:
    text = (data_item.text or "").strip()
    if ":" not in text:
        raise ValueError(f"Invalid HDF reference {text!r}")
    filename, dataset = text.split(":", 1)
    return filename, dataset


def _as_node_coordinates(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional coordinate array, got {array.shape}")
    if array.shape[1] == 3:
        return np.asarray(array, dtype=float)
    if array.shape[0] == 3:
        return np.asarray(array.T, dtype=float)
    raise ValueError(f"Could not identify the XYZ axis in coordinate array {array.shape}")


def _as_connectivity(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional connectivity array, got {array.shape}")
    if array.shape[1] in {3, 4}:
        return np.asarray(array, dtype=np.int64)
    if array.shape[0] in {3, 4}:
        return np.asarray(array.T, dtype=np.int64)
    raise ValueError(f"Could not identify the cell axis in connectivity array {array.shape}")


@dataclass(frozen=True)
class TrajectoryRecord:
    family: str
    run_index: int
    xdmf_path: Path
    h5_path: Path
    n_times: int
    start_time: float
    end_time: float

    @property
    def name(self) -> str:
        return f"{self.family}ScanPath_{self.run_index}"


class ThermalTrajectory:
    """Lazy reader for one XDMF/HDF5 thermal trajectory pair."""

    def __init__(self, xdmf_path: str | Path):
        self.xdmf_path = Path(xdmf_path).expanduser().resolve()
        if not self.xdmf_path.is_file():
            raise FileNotFoundError(self.xdmf_path)

        match = _TRAJECTORY_PATTERN.search(self.xdmf_path.name)
        if match is None:
            raise ValueError(f"Unrecognized trajectory filename {self.xdmf_path.name!r}")
        self.family = match.group(1)
        self.run_index = int(match.group(2))

        root = ET.parse(self.xdmf_path).getroot()
        grids = [element for element in root.iter() if _local_name(element.tag) == "Grid"]
        timed_grids = [grid for grid in grids if _direct_child(grid, "Time") is not None]
        if not timed_grids:
            raise ValueError(f"No time-indexed grids found in {self.xdmf_path}")

        topology = next(
            (element for element in timed_grids[0] if _local_name(element.tag) == "Topology"),
            None,
        )
        geometry = next(
            (element for element in timed_grids[0] if _local_name(element.tag) == "Geometry"),
            None,
        )
        if topology is None or geometry is None:
            raise ValueError(f"First time grid in {self.xdmf_path} has no mesh")

        topology_item = next(
            (element for element in topology.iter() if _local_name(element.tag) == "DataItem"),
            None,
        )
        geometry_item = next(
            (element for element in geometry.iter() if _local_name(element.tag) == "DataItem"),
            None,
        )
        if topology_item is None or geometry_item is None:
            raise ValueError(f"Mesh references are missing in {self.xdmf_path}")

        topology_file, self.topology_dataset = _hdf_reference(topology_item)
        geometry_file, self.geometry_dataset = _hdf_reference(geometry_item)
        if topology_file != geometry_file:
            raise ValueError("Topology and geometry refer to different HDF5 files")

        self.h5_path = (self.xdmf_path.parent / geometry_file).resolve()
        if not self.h5_path.is_file():
            raise FileNotFoundError(
                f"{self.xdmf_path.name} refers to missing HDF5 file {self.h5_path}"
            )

        self.times: np.ndarray
        self._field_datasets: dict[str, tuple[str, ...]]
        times: list[float] = []
        fields: dict[str, list[str]] = {}

        for grid in timed_grids:
            time_element = _direct_child(grid, "Time")
            assert time_element is not None
            times.append(float(time_element.attrib["Value"]))

            for attribute in grid:
                if _local_name(attribute.tag) != "Attribute":
                    continue
                name = attribute.attrib.get("Name")
                data_item = next(
                    (
                        element
                        for element in attribute.iter()
                        if _local_name(element.tag) == "DataItem"
                    ),
                    None,
                )
                if not name or data_item is None:
                    continue
                filename, dataset = _hdf_reference(data_item)
                if (self.xdmf_path.parent / filename).resolve() != self.h5_path:
                    raise ValueError(f"Field {name!r} refers to a different HDF5 file")
                fields.setdefault(name, []).append(dataset)

        self.times = np.asarray(times, dtype=float)
        self._field_datasets = {name: tuple(paths) for name, paths in fields.items()}
        for name, paths in self._field_datasets.items():
            if len(paths) != len(self.times):
                raise ValueError(
                    f"Field {name!r} has {len(paths)} datasets for {len(self.times)} times"
                )

        self._geometry_cache: np.ndarray | None = None
        self._surface_indices_cache: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"{self.family}ScanPath_{self.run_index}"

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self._field_datasets)

    @property
    def n_times(self) -> int:
        return len(self.times)

    def geometry(self) -> np.ndarray:
        if self._geometry_cache is None:
            with h5py.File(self.h5_path, "r") as handle:
                self._geometry_cache = _as_node_coordinates(handle[self.geometry_dataset][...])
        return self._geometry_cache.copy()

    def topology(self) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as handle:
            return _as_connectivity(handle[self.topology_dataset][...])

    def surface_node_indices(self) -> np.ndarray:
        if self._surface_indices_cache is None:
            geometry = self.geometry()
            z = geometry[:, 2]
            tolerance = max(np.ptp(z) * 1e-10, np.finfo(float).eps * 100.0)
            self._surface_indices_cache = np.flatnonzero(
                np.isclose(z, np.max(z), rtol=0.0, atol=tolerance)
            )
        return self._surface_indices_cache.copy()

    def surface_coordinates(self) -> np.ndarray:
        geometry = self.geometry()
        return geometry[self.surface_node_indices(), :2]

    def read_field(
        self,
        name: str,
        time_index: int,
        *,
        surface_only: bool = False,
    ) -> np.ndarray:
        if name not in self._field_datasets:
            raise KeyError(f"Unknown field {name!r}; available fields: {self.field_names}")
        index = int(time_index)
        if index < 0:
            index += self.n_times
        if index < 0 or index >= self.n_times:
            raise IndexError(f"Time index {time_index} is outside [0, {self.n_times})")
        dataset = self._field_datasets[name][index]
        with h5py.File(self.h5_path, "r") as handle:
            values = np.asarray(handle[dataset][...], dtype=float).reshape(-1)
        if surface_only:
            values = values[self.surface_node_indices()]
        return values

    def read_temperature(self, time_index: int, *, surface_only: bool = False) -> np.ndarray:
        return self.read_field("Temperature", time_index, surface_only=surface_only)

    def read_heat_flux_z(self, time_index: int, *, surface_only: bool = False) -> np.ndarray:
        return self.read_field("HeatFluxZ", time_index, surface_only=surface_only)

    def read_history(
        self,
        name: str = "Temperature",
        *,
        time_indices: np.ndarray | list[int] | None = None,
        surface_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if name not in self._field_datasets:
            raise KeyError(f"Unknown field {name!r}; available fields: {self.field_names}")
        if time_indices is None:
            indices = np.arange(self.n_times, dtype=int)
        else:
            indices = np.asarray(time_indices, dtype=int)
            indices = np.where(indices < 0, indices + self.n_times, indices)
        if np.any((indices < 0) | (indices >= self.n_times)):
            raise IndexError("At least one requested time index is outside the trajectory")

        surface_indices = self.surface_node_indices() if surface_only else None
        history: list[np.ndarray] = []
        with h5py.File(self.h5_path, "r") as handle:
            for index in indices:
                values = np.asarray(
                    handle[self._field_datasets[name][int(index)]][...],
                    dtype=float,
                ).reshape(-1)
                if surface_indices is not None:
                    values = values[surface_indices]
                history.append(values)
        return self.times[indices], np.stack(history)

    def record(self) -> TrajectoryRecord:
        return TrajectoryRecord(
            family=self.family,
            run_index=self.run_index,
            xdmf_path=self.xdmf_path,
            h5_path=self.h5_path,
            n_times=self.n_times,
            start_time=float(self.times[0]),
            end_time=float(self.times[-1]),
        )


class SurfaceGridProjector:
    """Precomputed linear interpolation from surface nodes to a regular grid."""

    def __init__(
        self,
        surface_points: np.ndarray,
        *,
        nx: int = 31,
        ny: int = 21,
    ):
        points = np.asarray(surface_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Expected surface points with shape (n, 2), got {points.shape}")
        if nx < 2 or ny < 2:
            raise ValueError("Regular grid dimensions must both be at least two")

        self.surface_points = points
        self.xs = np.linspace(float(points[:, 0].min()), float(points[:, 0].max()), nx)
        self.ys = np.linspace(float(points[:, 1].min()), float(points[:, 1].max()), ny)
        grid_x, grid_y = np.meshgrid(self.xs, self.ys)
        self.grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        self.shape = (ny, nx)

        triangulation = Delaunay(points)
        simplex = triangulation.find_simplex(self.grid_points)
        if np.any(simplex < 0):
            count = int(np.sum(simplex < 0))
            raise ValueError(f"{count} regular-grid points lie outside the surface convex hull")

        transform = triangulation.transform[simplex]
        delta = self.grid_points - transform[:, 2]
        barycentric = np.einsum("nij,nj->ni", transform[:, :2], delta)
        self.weights = np.column_stack(
            [barycentric, 1.0 - np.sum(barycentric, axis=1)]
        )
        self.vertices = triangulation.simplices[simplex]

    def project(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != len(self.surface_points):
            raise ValueError(
                f"Last array dimension must be {len(self.surface_points)}, got {array.shape}"
            )
        gathered = array[..., self.vertices]
        projected = np.sum(gathered * self.weights, axis=-1)
        return projected.reshape(array.shape[:-1] + self.shape)


def trajectory_catalog(dataset_dir: str | Path) -> list[TrajectoryRecord]:
    directory = Path(dataset_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    records = [
        ThermalTrajectory(path).record()
        for path in sorted(directory.glob("*.xdmf"))
        if _TRAJECTORY_PATTERN.search(path.name)
    ]
    family_order = {"Diagonal": 0, "Horizontal": 1, "Spiral": 2}
    return sorted(records, key=lambda record: (family_order[record.family], record.run_index))
