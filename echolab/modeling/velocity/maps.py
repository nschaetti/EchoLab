"""
Velocity map primitives and IO helpers.
"""

from __future__ import annotations

import pickle
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np


class Dimensionality(str, Enum):
    """
    Supported velocity map dimensionalities.
    """
    DIM_1D = "1D"
    DIM_2D = "2D"
    DIM_3D = "3D"
# end class Dimensionality


class VelocityMap:
    """
    Pydantic model describing a velocity map and its sampling resolution.

    The map stores both the velocity values and the sampling interval for each
    axis so that downstream consumers can reconstruct physical coordinates.
    """

    def __init__(
            self,
            data: np.ndarray,
            dim: Dimensionality,
            dz: float,
            dx: Optional[float] = None,
            dy: Optional[float] = None,
    ):
        self.data = data
        self.ndim = dim
        self.dz = dz
        self.dx = dx
        self.dy = dy
    # end __init__

    # region PROPERTIES

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the velocity array shape."""
        return self.data.shape
    # end def shape

    # endregion PROPERTIES

    # region CLASS_METHODS

    @classmethod
    def from_1d_array(cls, data: np.ndarray, dz: float) -> "VelocityMap":
        """Build a 1D velocity map from a numpy vector."""
        return cls(
            data=np.asarray(data, dtype=np.float32),
            dim=Dimensionality.DIM_1D,
            dx=0.0,
            dz=float(dz),
        )
    # end def from_1d_array

    @classmethod
    def from_2d_array(cls, data: np.ndarray, dx: float, dz: float) -> "VelocityMap":
        """Build a 2D velocity map from a numpy matrix."""
        return cls(
            data=np.asarray(data, dtype=np.float32),
            dim=Dimensionality.DIM_2D,
            dx=float(dx),
            dz=float(dz),
        )
    # end def from_2d_array

    @classmethod
    def from_3d_array(
        cls, data: np.ndarray, dx: float, dy: float, dz: float
    ) -> "VelocityMap":
        """Build a 3D velocity map from a numpy volume."""
        return cls(
            data=np.asarray(data, dtype=np.float32),
            dim=Dimensionality.DIM_3D,
            dx=float(dx),
            dy=float(dy),
            dz=float(dz),
        )
    # end def from_3d_array

    # endregion CLASS_METHODS

    # region OVERRIDE

    def __str__(self) -> str:
        components = [
            f"{self.ndim}",
            f"shape={'x'.join(str(dim) for dim in self.shape)}",
            f"range=[{float(np.min(self.data)):.2f}, {float(np.max(self.data)):.2f}]",
            f"mean={float(np.mean(self.data)):.2f}",
            f"dx={self.dx:.2f}",
            f"dz={self.dz:.2f}",
        ]

        if self.ndim == Dimensionality.DIM_3D:
            components.append(f"dy={float(self.dy):.2f}")
        # end if

        return f"VelocityMap({', '.join(components)})"
    # end def __str__

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        if self.ndim == Dimensionality.DIM_1D:
            return (
                "VelocityMap.from_1d_array("
                f"np.array({self.data.tolist()}), dz={self.dz})"
            )
        # end if
        if self.ndim == Dimensionality.DIM_2D:
            return (
                "VelocityMap.from_2d_array("
                f"np.array({self.data.tolist()}), dx={self.dx}, dz={self.dz})"
            )
        # end if
        return (
            "VelocityMap.from_3d_array("
            f"np.array({self.data.tolist()}), dx={self.dx}, "
            f"dy={self.dy}, dz={self.dz})"
        )
    # end def __repr__

    # endregion OVERRIDE

# end class VelocityMap


def save_velocity_maps(
    velocity_maps: Iterable[VelocityMap], file_path: Union[str, Path]
) -> None:
    """
    Serialise a sequence of velocity maps to disk using pickle.
    """
    file_path = Path(file_path)
    with open(file_path, "wb") as fh:
        pickle.dump(list(velocity_maps), fh)
    # end with
# end def save_velocity_maps


def load_velocity_maps(file_path: Union[str, Path]) -> List[VelocityMap]:
    """
    Load a list of velocity maps from disk.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    # end if
    with open(file_path, "rb") as fh:
        try:
            maps = pickle.load(fh)
        except (pickle.UnpicklingError, EOFError) as exc:  # pragma: no cover
            raise ValueError(f"Error loading velocity maps: {exc}") from exc
        # end try
    # end with

    if not isinstance(maps, list) or not all(isinstance(vm, VelocityMap) for vm in maps):
        raise ValueError("File does not contain a valid list of VelocityMap objects")
    # end if
    return maps
# end def load_velocity_maps

