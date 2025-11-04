"""
Velocity model abstractions, IO helpers, and high-level convenience routines.
"""

# Imports
import json
import pickle
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from .maps import Dimensionality, VelocityMap


class VelocityModel(ABC):
    """
    Abstract base class for velocity models of any dimensionality.

    The base exposes convenient accessors for metadata, geometry, and data
    serialisation while keeping the heavy lifting in the VelocityMap structure.
    """

    def __init__(
        self,
        velmap: VelocityMap,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Store the map and associated metadata used by the velocity model.

        Args:
            velmap: Backing :class:`VelocityMap` containing the velocity grid.
            metadata: Arbitrary metadata describing the model provenance.
        """
        self.velmap = velmap
        self.metadata = metadata
    # end def __init__

    # region PROPERTIES

    @property
    def ndim(self) -> int:
        """
        Return the number of spatial dimensions.
        """
        mapping = {
            Dimensionality.DIM_1D: 1,
            Dimensionality.DIM_2D: 2,
            Dimensionality.DIM_3D: 3,
        }
        try:
            return mapping[self.velmap.ndim]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Unsupported dimensionality: {self.velmap.ndim}"
            ) from exc
        # end try
    # end def ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Return the grid shape.
        """
        return self.velmap.shape
    # end def shape

    @property
    def grid_spacing(self) -> Tuple[float, ...]:
        """
        Return the sampling interval for each dimension.
        """
        dim = self.velmap.ndim
        if dim == Dimensionality.DIM_1D:
            return (self.velmap.dz,)
        # end if

        if dim == Dimensionality.DIM_2D:
            return self.velmap.dx, self.velmap.dz
        # end if

        if dim == Dimensionality.DIM_3D:
            return self.velmap.dx, self.velmap.dy, self.velmap.dz
        # end if

        raise ValueError(f"Unsupported dimensionality: {dim}")  # pragma: no cover
    # end def grid_spacing

    @property
    def origin(self) -> Tuple[float, ...]:
        """
        Return the origin coordinates (always zero for now).
        """
        return tuple(0.0 for _ in range(self.ndim))
    # end def origin

    @property
    def extent(self) -> Tuple[Tuple[float, float], ...]:
        """
        Return coordinate extents for each dimension.
        """
        extents: List[Tuple[float, float]] = []
        for idx in range(self.ndim):
            start = self.origin[idx]
            stop = start + self.grid_spacing[idx] * self.shape[idx]
            extents.append((start, stop))
        # end for
        return tuple(extents)
    # end def extent

    @property
    def min_velocity(self) -> float:
        """Return the minimum velocity contained in the grid."""
        return float(np.min(self.velmap.data))
    # end def min_velocity

    @property
    def max_velocity(self) -> float:
        """Return the maximum velocity contained in the grid."""
        return float(np.max(self.velmap.data))
    # end def max_velocity

    @property
    def mean_velocity(self) -> float:
        """Return the mean velocity across the grid."""
        return float(np.mean(self.velmap.data))
    # end def mean_velocity

    @property
    def std_velocity(self) -> float:
        """Return the velocity standard deviation."""
        return float(np.std(self.velmap.data))
    # end def std_velocity

    # endregion PROPERTIES

    # region EXPORT

    def as_numpy(self) -> np.ndarray:
        """Expose the underlying velocity array as a NumPy array."""
        return self.velmap.data
    # end def as_numpy

    def as_list(self) -> List[Any]:
        """Expose the velocity data as a nested Python list."""
        return self.velmap.data.tolist()
    # end def as_list

    def as_torch(self) -> "torch.Tensor":
        """Return the velocity data as a PyTorch tensor."""
        return torch.from_numpy(self.velmap.data)
    # end def as_torch

    # endregion EXPORT

    # region PUBLIC

    def get_coordinates(self, dimension: int) -> np.ndarray:
        """Return the coordinate vector for the requested dimension."""
        if dimension < 0 or dimension >= self.ndim:
            raise ValueError(f"Dimension {dimension} is out of range for {self.ndim}D model")
        # end if
        start = self.origin[dimension]
        step = self.grid_spacing[dimension]
        size = self.shape[dimension]
        return np.linspace(start, start + (size - 1) * step, size, dtype=np.float32)
    # end def get_coordinates

    def save(
            self,
            path: Union[str, Path],
            file_format: str = "numpy"
    ) -> None:
        """
        Persist the model to disk.

        Supported formats: ``numpy`` (compressed npz), ``json`` and ``pickle``.
        """
        path = Path(path)
        if file_format == "numpy":
            payload = {
                "data": self.velmap.data,
                "dim": self.velmap.ndim,
                "dx": self.velmap.dx,
                "dy": self.velmap.dy,
                "dz": self.velmap.dz,
                "metadata": self.metadata,
            }
            np.savez_compressed(path, **payload)
            return
        # end if
        if file_format == "json":
            payload = {
                "data": self.velmap.data.tolist(),
                "dim": self.velmap.ndim,
                "dx": self.velmap.dx,
                "dy": self.velmap.dy,
                "dz": self.velmap.dz,
                "metadata": self.metadata,
            }
            with open(path, "w") as fh:
                json.dump(payload, fh)
            # end with
            return
        # end if
        if file_format == "pickle":
            with open(path, "wb") as fh:
                pickle.dump(self, fh)
            # end with
            return
        # end if
        raise ValueError(f"Unsupported format: {file_format}")
    # end def save

    @classmethod
    def load(
            cls,
            path: Union[str, Path]
    ) -> "VelocityModel":
        """
        Load a velocity model from disk and return the appropriate subclass.
        """
        resolved = Path(path)
        suffix = resolved.suffix.lower()

        # File format
        if suffix == ".pkl":
            with open(resolved, "rb") as fh:
                model = pickle.load(fh)
            # end with
            if isinstance(model, VelocityModel):
                return model
            # end if
            raise ValueError(f"File does not contain a valid VelocityModel: {resolved}")
        # end if

        if suffix == ".json":
            with open(resolved, "r") as fh:
                payload = json.load(fh)
            # end with

            required = {"data", "dim", "dx", "dz"}
            if not required.issubset(payload):
                raise ValueError(f"Invalid velocity model JSON payload: {resolved}")
            # end if

            return _factory_from_components(
                data=np.array(payload["data"], dtype=np.float32),
                dim=Dimensionality(payload["dim"]),
                dx=float(payload["dx"]),
                dz=float(payload["dz"]),
                dy=payload.get("dy"),
                metadata=_coerce_metadata(payload.get("metadata")),
            )
        # end if

        if suffix == ".npz":
            try:
                with np.load(resolved, allow_pickle=True) as npz:
                    data = np.array(npz["data"], dtype=np.float32)
                    if "dim" in npz and npz["dim"]:
                        dim = Dimensionality(str(npz["dim"].item()))
                    else:
                        dim = _infer_dimensionality_from_ndim(data.ndim)
                    # end if

                    dx = float(npz["dx"].item()) if "dx" in npz else 1.0
                    dz = float(npz["dz"].item()) if "dz" in npz else 1.0
                    raw_dy = npz["dy"] if "dy" in npz else None
                    dy = _coerce_optional_float(raw_dy)
                    metadata_raw = npz["metadata"] if "metadata" in npz else {}
                    metadata = _coerce_metadata(metadata_raw)
                # end with
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Failed to load npz file: {exc}") from exc
            # end try

            return _factory_from_components(
                data=data,
                dim=dim,
                dx=dx,
                dz=dz,
                dy=dy,
                metadata=metadata,
            )
        # end if

        if suffix == ".npy":
            try:
                data = np.load(resolved)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Failed to load numpy file: {exc}") from exc
            # end try

            dim = _infer_dimensionality_from_ndim(data.ndim)
            dy = 1.0 if dim == Dimensionality.DIM_3D else None
            return _factory_from_components(
                data=np.array(data, dtype=np.float32),
                dim=dim,
                dx=1.0,
                dz=1.0,
                dy=dy,
                metadata={},
            )
        # end if

        # Fall back to probing multiple formats when the extension is unknown.
        for loader_suffix in (".pkl", ".json", ".npz", ".npy"):
            try:
                return cls.load(resolved.with_suffix(loader_suffix))
            except FileNotFoundError:
                continue
            except ValueError:
                continue
            # end try
        # end for

        raise ValueError(f"Unsupported file format: {resolved}")
    # end def load

    # endregion PUBLIC

    # region OVERRIDE

    def __str__(self) -> str:  # pragma: no cover - human readable
        return (
            f"{self.__class__.__name__}(shape={self.shape}, "
            f"grid_spacing={self.grid_spacing}, origin={self.origin}, "
            f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}])"
        )
    # end def __str__

    # endregion OVERRIDE

# end class VelocityModel


def _coerce_optional_float(value: Any) -> Optional[float]:
    """
    Convert optional numeric inputs stored in numpy to Python floats.
    """
    if value is None:
        return None
    # end if

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.size == 1:
            value = value.item()
        # end if
    # end if

    if value in ("None", ""):
        return None
    # end if

    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    # end try
# end def _coerce_optional_float


def _coerce_metadata(value: Any) -> Dict[str, Any]:
    """
    Ensure metadata payloads coming from numpy arrays are dictionaries.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray) and value.size == 1:
        extracted = value.item()
        if isinstance(extracted, dict):
            return extracted
        # end if
    # end if
    return {}
# end def _coerce_metadata


def _infer_dimensionality_from_ndim(ndim: int) -> Dimensionality:
    """
    Turn an array rank into a :class:`Dimensionality` value.
    """
    mapping = {
        1: Dimensionality.DIM_1D,
        2: Dimensionality.DIM_2D,
        3: Dimensionality.DIM_3D,
    }
    try:
        return mapping[ndim]
    except KeyError as exc:
        raise ValueError(f"Unsupported number of dimensions: {ndim}") from exc
    # end try
# end def _infer_dimensionality_from_ndim


def _factory_from_components(
    *,
    data: np.ndarray,
    dim: Dimensionality,
    dx: float,
    dz: float,
    dy: Optional[float],
    metadata: Optional[Dict[str, Any]],
) -> VelocityModel:
    """
    Build the appropriate VelocityModel subclass from raw components.
    """
    if dim == Dimensionality.DIM_1D:
        velocity_map = VelocityMap.from_1d_array(data, dz)
        return VelocityModel1D(velmap=velocity_map, metadata=metadata or {})
    # end if
    if dim == Dimensionality.DIM_2D:
        velocity_map = VelocityMap.from_2d_array(data, dx, dz)
        return VelocityModel2D(velmap=velocity_map, metadata=metadata or {})
    # end if
    if dim == Dimensionality.DIM_3D:
        if dy is None:
            raise ValueError("3D velocity model requires 'dy' spacing information")
        # end if
        velocity_map = VelocityMap.from_3d_array(data, dx, dy, dz)
        return VelocityModel3D(velmap=velocity_map, metadata=metadata or {})
    # end if
    raise ValueError(f"Unsupported dimensionality: {dim}")
# end def _factory_from_components


class VelocityModel1D(VelocityModel):
    """
    Concrete implementation for 1D models.
    """

    @classmethod
    def from_array(
        cls,
        velocity_data: np.ndarray,
        grid_spacing: Optional[Union[float, Sequence[float]]] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "VelocityModel1D":
        array = np.asarray(velocity_data, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"Expected 1D array, got {array.ndim}D")
        # end if
        dz = float(grid_spacing) if grid_spacing is not None else 1.0
        vmap = VelocityMap.from_1d_array(array, dz)
        return cls(velmap=vmap, metadata=metadata or {})
    # end def from_array

    @property
    def nz(self) -> int:
        """Return the number of samples along the depth axis."""
        return self.shape[0]
    # end def nz

    @property
    def dz(self) -> float:
        """Return the depth sampling interval."""
        return self.grid_spacing[0]
    # end def dz

    @property
    def z_extent(self) -> Tuple[float, float]:
        """Return the depth extent covered by the model."""
        return self.extent[0]
    # end def z_extent
# end class VelocityModel1D


class VelocityModel2D(VelocityModel):
    """Concrete implementation for 2D models."""

    @classmethod
    def from_array(
        cls,
        velocity_data: np.ndarray,
        grid_spacing: Union[float, Sequence[float]],
        *,
        metadata: Dict[str, Any],
    ) -> "VelocityModel2D":
        array = np.asarray(velocity_data, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D array, got {array.ndim}D")
        # end if

        if isinstance(grid_spacing, (int, float)):
            dx = dz = float(grid_spacing)
        else:
            if len(grid_spacing) != 2:
                raise ValueError(f"Expected 2 grid spacing values, got {len(grid_spacing)}")
            # end if
            dx, dz = float(grid_spacing[0]), float(grid_spacing[1])
        # end if
        vmap = VelocityMap.from_2d_array(array, dx, dz)
        return cls(velmap=vmap, metadata=metadata or {})
    # end def from_array

    @property
    def nz(self) -> int:
        """Return the number of samples along the vertical axis."""
        return self.shape[0]
    # end def nz

    @property
    def nx(self) -> int:
        """Return the number of samples along the horizontal axis."""
        return self.shape[1]
    # end def nx

    @property
    def dz(self) -> float:
        """Return the vertical sampling interval."""
        return self.grid_spacing[1]
    # end def dz

    @property
    def dx(self) -> float:
        """Return the horizontal sampling interval."""
        return self.grid_spacing[0]
    # end def dx
# end class VelocityModel2D


class VelocityModel3D(VelocityModel):
    """Concrete implementation for 3D models."""

    @classmethod
    def from_array(
        cls,
        velocity_data: np.ndarray,
        grid_spacing: Optional[Union[float, Sequence[float]]] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "VelocityModel3D":
        array = np.asarray(velocity_data, dtype=np.float32)
        if array.ndim != 3:
            raise ValueError(f"Expected 3D array, got {array.ndim}D")
        # end if
        if grid_spacing is None:
            dx = dy = dz = 1.0
        elif isinstance(grid_spacing, (int, float)):
            dx = dy = dz = float(grid_spacing)
        else:
            if len(grid_spacing) != 3:
                raise ValueError(f"Expected 3 grid spacing values, got {len(grid_spacing)}")
            # end if
            dx, dy, dz = (
                float(grid_spacing[0]),
                float(grid_spacing[1]),
                float(grid_spacing[2]),
            )
        vmap = VelocityMap.from_3d_array(array, dx, dy, dz)
        return cls(velmap=vmap, metadata=metadata or {})
    # end def from_array

    @property
    def nz(self) -> int:
        """Return the number of samples along the depth axis."""
        return self.shape[2]
    # end def nz

    @property
    def ny(self) -> int:
        """Return the number of samples along the inline axis."""
        return self.shape[1]
    # end def ny

    @property
    def nx(self) -> int:
        """Return the number of samples along the crossline axis."""
        return self.shape[0]
    # end def nx

    @property
    def dz(self) -> float:
        """Return the depth sampling interval."""
        return self.grid_spacing[2]
    # end def dz

    @property
    def dy(self) -> float:
        """Return the inline sampling interval."""
        return self.grid_spacing[1]
    # end def dy

    @property
    def dx(self) -> float:
        """Return the crossline sampling interval."""
        return self.grid_spacing[0]
    # end def dx
# end class VelocityModel3D


VelocityModelBase = VelocityModel


def load_velocity_model(path: Union[str, Path]) -> VelocityModel:
    """
    Load a velocity model from disk.
    """
    return VelocityModel.load(path)
# end def load_velocity_model


def save_velocity_models(
        models: Iterable[VelocityModel],
        path: Union[str, Path],
        file_format: str = "pickle"
) -> None:
    """
    Persist a collection of velocity models.
    """
    if file_format != "pickle":
        raise ValueError("Only 'pickle' format is supported when saving collections")
    # end if
    path = Path(path)
    with open(path, "wb") as fh:
        pickle.dump(list(models), fh)
    # end with
# end def save_velocity_models


def load_velocity_models(path: Union[str, Path]) -> List[VelocityModel]:
    """
    Load multiple velocity models stored with :func:`save_velocity_models`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # end if

    with open(path, "rb") as fh:
        try:
            models = pickle.load(fh)
        except (pickle.UnpicklingError, EOFError) as exc:  # pragma: no cover
            raise ValueError(f"Error loading velocity models: {exc}") from exc
        # end try
    # end with

    if not isinstance(models, list) or not all(isinstance(m, VelocityModel) for m in models):
        raise ValueError("File does not contain a valid list of VelocityModel objects")
    # end if

    return models
# end def load_velocity_models


def create_velocity_model(
    data: Union[np.ndarray, Sequence[Any]],
    *,
    grid_spacing: Optional[Union[float, Sequence[float]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> VelocityModel:
    """
    Create a velocity model from an array-like payload.

    The dimensionality is inferred from the array rank.  This acts as a single
    entry point for callers that do not want to reason about the specialised
    subclasses.
    """
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        return VelocityModel1D.from_array(
            velocity_data=array,
            grid_spacing=grid_spacing,
            metadata=metadata
        )
    # end if

    if array.ndim == 2:
        return VelocityModel2D.from_array(
            array,
            grid_spacing,
            metadata=metadata
        )
    # end if

    if array.ndim == 3:
        return VelocityModel3D.from_array(
            array,
            grid_spacing,
            metadata=metadata
        )
    # end if

    raise ValueError("Velocity models must be 1D, 2D, or 3D")
    # end if
# end def generate


def as_map(
    data: Union[np.ndarray, Sequence[Any]],
    *,
    grid_spacing: Optional[Union[float, Sequence[float]]] = None,
) -> VelocityMap:
    """
    Convenience helper returning a :class:`VelocityMap` from array-like input.
    """
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        dz = float(grid_spacing) if grid_spacing is not None else 1.0
        return VelocityMap.from_1d_array(array, dz)
    # end if

    if array.ndim == 2:
        if grid_spacing is None:
            dx = dz = 1.0
        elif isinstance(grid_spacing, (int, float)):
            dx = dz = float(grid_spacing)
        else:
            if len(grid_spacing) != 2:
                raise ValueError(f"Expected 2 grid spacing values, got {len(grid_spacing)}")
            # end if
            dx, dz = float(grid_spacing[0]), float(grid_spacing[1])
        return VelocityMap.from_2d_array(array, dx, dz)
    # end if

    if array.ndim == 3:
        if grid_spacing is None:
            dx = dy = dz = 1.0
        elif isinstance(grid_spacing, (int, float)):
            dx = dy = dz = float(grid_spacing)
        else:
            if len(grid_spacing) != 3:
                raise ValueError(f"Expected 3 grid spacing values, got {len(grid_spacing)}")
            # end if
            dx, dy, dz = (
                float(grid_spacing[0]),
                float(grid_spacing[1]),
                float(grid_spacing[2]),
            )
        # end if
        return VelocityMap.from_3d_array(array, dx, dy, dz)
    # end if
    raise ValueError("Velocity maps must be 1D, 2D, or 3D")
# end def as_map


def is_velocity_model(obj: Any) -> bool:
    """Return ``True`` if *obj* looks like a velocity model instance."""
    return isinstance(obj, VelocityModel)
# end def is_velocity_model


def open_models(path: Union[str, Path]) -> List[VelocityModel]:
    """Alias for :func:`load_velocity_models` with a friendlier name."""
    return load_velocity_models(path)
# end def open_models
