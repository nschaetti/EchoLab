"""
Velocity model classes for echolab.

This module provides classes for handling velocity models in different dimensions
(1D, 2D, 3D) with support for various data formats and serialization methods.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import pickle

import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class VelocityModel(ABC):
    """
    Abstract base class for velocity models.
    
    This class defines the interface for velocity models of different dimensions.
    It provides methods for accessing the velocity data in different formats,
    getting metadata about the model, and serialization.
    """
    
    def __init__(
        self,
        velocity_data: np.ndarray,
        grid_spacing: Union[float, List[float], Tuple[float, ...]] = 1.0,
        origin: Optional[Union[float, List[float], Tuple[float, ...]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a velocity model.
        
        Args:
            velocity_data: Velocity data as a numpy array.
            grid_spacing: Grid spacing in each dimension. If a single value is provided,
                it will be used for all dimensions.
            origin: Origin coordinates for each dimension. If None, defaults to zeros.
            metadata: Additional metadata for the velocity model.
        """
        self._velocity_data = np.asarray(velocity_data, dtype=np.float32)
        
        # Set grid spacing
        if isinstance(grid_spacing, (int, float)):
            self._grid_spacing = tuple([float(grid_spacing)] * self.ndim)
        else:
            if len(grid_spacing) != self.ndim:
                raise ValueError(f"Expected {self.ndim} grid spacing values, got {len(grid_spacing)}")
            self._grid_spacing = tuple(float(x) for x in grid_spacing)
        
        # Set origin
        if origin is None:
            self._origin = tuple([0.0] * self.ndim)
        else:
            if isinstance(origin, (int, float)):
                self._origin = tuple([float(origin)] * self.ndim)
            else:
                if len(origin) != self.ndim:
                    raise ValueError(f"Expected {self.ndim} origin values, got {len(origin)}")
                self._origin = tuple(float(x) for x in origin)
        
        # Set metadata
        self._metadata = metadata or {}
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the velocity model."""
        return self._velocity_data.ndim
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the velocity model."""
        return self._velocity_data.shape
    
    @property
    def grid_spacing(self) -> Tuple[float, ...]:
        """Get the grid spacing in each dimension."""
        return self._grid_spacing
    
    @property
    def origin(self) -> Tuple[float, ...]:
        """Get the origin coordinates in each dimension."""
        return self._origin
    
    @property
    def extent(self) -> Tuple[float, ...]:
        """
        Get the extent of the velocity model in each dimension.
        
        Returns:
            A tuple of (min_x, max_x, min_y, max_y, ...) for each dimension.
        """
        result = []
        for i in range(self.ndim):
            min_val = self._origin[i]
            max_val = self._origin[i] + (self.shape[i] - 1) * self._grid_spacing[i]
            result.extend([min_val, max_val])
        return tuple(result)
    
    @property
    def min_velocity(self) -> float:
        """Get the minimum velocity value."""
        return float(np.min(self._velocity_data))
    
    @property
    def max_velocity(self) -> float:
        """Get the maximum velocity value."""
        return float(np.max(self._velocity_data))
    
    @property
    def mean_velocity(self) -> float:
        """Get the mean velocity value."""
        return float(np.mean(self._velocity_data))
    
    @property
    def std_velocity(self) -> float:
        """Get the standard deviation of velocity values."""
        return float(np.std(self._velocity_data))
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata dictionary."""
        return self._metadata
    
    def as_numpy(self) -> np.ndarray:
        """Get the velocity data as a numpy array."""
        return self._velocity_data
    
    def as_list(self) -> List:
        """Get the velocity data as a nested list."""
        return self._velocity_data.tolist()
    
    def as_torch(self) -> Any:
        """Get the velocity data as a PyTorch tensor."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Install it with 'pip install torch'.")
        return torch.from_numpy(self._velocity_data)
    
    def get_coordinates(self, dimension: int) -> np.ndarray:
        """
        Get the coordinates along a specific dimension.
        
        Args:
            dimension: The dimension index (0 for x, 1 for y, etc.).
            
        Returns:
            An array of coordinates along the specified dimension.
        """
        if dimension < 0 or dimension >= self.ndim:
            raise ValueError(f"Dimension {dimension} out of bounds for {self.ndim}D model")
        
        return np.arange(self.shape[dimension]) * self._grid_spacing[dimension] + self._origin[dimension]
    
    def save(self, path: Union[str, Path], format: str = "numpy") -> None:
        """
        Save the velocity model to a file.
        
        Args:
            path: Path to save the file.
            format: Format to save the file in. Options are "numpy", "pickle", or "json".
        """
        path = Path(path)
        
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "numpy":
            # Save as .npz file with metadata
            save_dict = {
                "velocity_data": self._velocity_data,
                "grid_spacing": self._grid_spacing,
                "origin": self._origin,
                "metadata": self._metadata,
                "class_name": self.__class__.__name__
            }
            np.savez(path, **save_dict)
        
        elif format == "pickle":
            # Save as pickle file
            with open(path, "wb") as f:
                pickle.dump(self, f)
        
        elif format == "json":
            # Save as JSON file
            save_dict = {
                "velocity_data": self._velocity_data.tolist(),
                "grid_spacing": self._grid_spacing,
                "origin": self._origin,
                "metadata": self._metadata,
                "class_name": self.__class__.__name__
            }
            with open(path, "w") as f:
                json.dump(save_dict, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> VelocityModel:
        """
        Load a velocity model from a file.
        
        Args:
            path: Path to the file.
            
        Returns:
            A VelocityModel instance.
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Determine the file format based on extension
        if path.suffix == ".npz":
            # Load from .npz file
            with np.load(path, allow_pickle=True) as data:
                velocity_data = data["velocity_data"]
                grid_spacing = data["grid_spacing"].item() if "grid_spacing" in data else 1.0
                origin = data["origin"].item() if "origin" in data else None
                metadata = data["metadata"].item() if "metadata" in data else None
                class_name = data["class_name"].item() if "class_name" in data else None
            
            # Create the appropriate class instance
            if class_name == "VelocityModel1D":
                from .velocity_model import VelocityModel1D
                return VelocityModel1D(velocity_data, grid_spacing, origin, metadata)
            elif class_name == "VelocityModel2D":
                from .velocity_model import VelocityModel2D
                return VelocityModel2D(velocity_data, grid_spacing, origin, metadata)
            elif class_name == "VelocityModel3D":
                from .velocity_model import VelocityModel3D
                return VelocityModel3D(velocity_data, grid_spacing, origin, metadata)
            else:
                # Determine the class based on the dimensions
                ndim = velocity_data.ndim
                if ndim == 1:
                    from .velocity_model import VelocityModel1D
                    return VelocityModel1D(velocity_data, grid_spacing, origin, metadata)
                elif ndim == 2:
                    from .velocity_model import VelocityModel2D
                    return VelocityModel2D(velocity_data, grid_spacing, origin, metadata)
                elif ndim == 3:
                    from .velocity_model import VelocityModel3D
                    return VelocityModel3D(velocity_data, grid_spacing, origin, metadata)
                else:
                    raise ValueError(f"Unsupported number of dimensions: {ndim}")
        
        elif path.suffix == ".pkl":
            # Load from pickle file
            with open(path, "rb") as f:
                return pickle.load(f)
        
        elif path.suffix == ".json":
            # Load from JSON file
            with open(path, "r") as f:
                data = json.load(f)
            
            velocity_data = np.array(data["velocity_data"])
            grid_spacing = data.get("grid_spacing", 1.0)
            origin = data.get("origin", None)
            metadata = data.get("metadata", None)
            class_name = data.get("class_name", None)
            
            # Create the appropriate class instance
            if class_name == "VelocityModel1D":
                from .velocity_model import VelocityModel1D
                return VelocityModel1D(velocity_data, grid_spacing, origin, metadata)
            elif class_name == "VelocityModel2D":
                from .velocity_model import VelocityModel2D
                return VelocityModel2D(velocity_data, grid_spacing, origin, metadata)
            elif class_name == "VelocityModel3D":
                from .velocity_model import VelocityModel3D
                return VelocityModel3D(velocity_data, grid_spacing, origin, metadata)
            else:
                # Determine the class based on the dimensions
                ndim = velocity_data.ndim
                if ndim == 1:
                    from .velocity_model import VelocityModel1D
                    return VelocityModel1D(velocity_data, grid_spacing, origin, metadata)
                elif ndim == 2:
                    from .velocity_model import VelocityModel2D
                    return VelocityModel2D(velocity_data, grid_spacing, origin, metadata)
                elif ndim == 3:
                    from .velocity_model import VelocityModel3D
                    return VelocityModel3D(velocity_data, grid_spacing, origin, metadata)
                else:
                    raise ValueError(f"Unsupported number of dimensions: {ndim}")
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")


class VelocityModel1D(VelocityModel):
    """
    1D velocity model class.
    
    This class represents a 1D velocity model, typically used for depth-dependent
    velocity profiles.
    """
    
    def __init__(
        self,
        velocity_data: np.ndarray,
        grid_spacing: Union[float, List[float], Tuple[float]] = 1.0,
        origin: Optional[Union[float, List[float], Tuple[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a 1D velocity model.
        
        Args:
            velocity_data: 1D array of velocity values.
            grid_spacing: Grid spacing in the z-dimension.
            origin: Origin coordinate in the z-dimension.
            metadata: Additional metadata for the velocity model.
        """
        # Ensure the data is 1D
        velocity_data = np.asarray(velocity_data)
        if velocity_data.ndim != 1:
            if velocity_data.ndim > 1 and np.prod(velocity_data.shape[1:]) == 1:
                # Reshape to 1D if all other dimensions are singleton
                velocity_data = velocity_data.reshape(-1)
            else:
                raise ValueError(f"Expected 1D array, got shape {velocity_data.shape}")
        
        super().__init__(velocity_data, grid_spacing, origin, metadata)
    
    @property
    def nz(self) -> int:
        """Get the number of grid points in the z-dimension."""
        return self.shape[0]
    
    @property
    def dz(self) -> float:
        """Get the grid spacing in the z-dimension."""
        return self.grid_spacing[0]
    
    @property
    def z_origin(self) -> float:
        """Get the origin coordinate in the z-dimension."""
        return self.origin[0]
    
    @property
    def z(self) -> np.ndarray:
        """Get the z-coordinates."""
        return self.get_coordinates(0)
    
    @property
    def z_extent(self) -> Tuple[float, float]:
        """Get the extent in the z-dimension."""
        return (self.extent[0], self.extent[1])


class VelocityModel2D(VelocityModel):
    """
    2D velocity model class.
    
    This class represents a 2D velocity model, typically used for cross-sections
    with x (horizontal) and z (depth) dimensions.
    """
    
    def __init__(
        self,
        velocity_data: np.ndarray,
        grid_spacing: Union[float, List[float], Tuple[float, float]] = 1.0,
        origin: Optional[Union[float, List[float], Tuple[float, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a 2D velocity model.
        
        Args:
            velocity_data: 2D array of velocity values with shape (nz, nx).
            grid_spacing: Grid spacing in the z and x dimensions.
            origin: Origin coordinates in the z and x dimensions.
            metadata: Additional metadata for the velocity model.
        """
        # Ensure the data is 2D
        velocity_data = np.asarray(velocity_data)
        if velocity_data.ndim != 2:
            if velocity_data.ndim > 2 and np.prod(velocity_data.shape[2:]) == 1:
                # Reshape to 2D if all other dimensions are singleton
                velocity_data = velocity_data.reshape(velocity_data.shape[0], velocity_data.shape[1])
            else:
                raise ValueError(f"Expected 2D array, got shape {velocity_data.shape}")
        
        super().__init__(velocity_data, grid_spacing, origin, metadata)
    
    @property
    def nz(self) -> int:
        """Get the number of grid points in the z-dimension."""
        return self.shape[0]
    
    @property
    def nx(self) -> int:
        """Get the number of grid points in the x-dimension."""
        return self.shape[1]
    
    @property
    def dz(self) -> float:
        """Get the grid spacing in the z-dimension."""
        return self.grid_spacing[0]
    
    @property
    def dx(self) -> float:
        """Get the grid spacing in the x-dimension."""
        return self.grid_spacing[1]
    
    @property
    def z_origin(self) -> float:
        """Get the origin coordinate in the z-dimension."""
        return self.origin[0]
    
    @property
    def x_origin(self) -> float:
        """Get the origin coordinate in the x-dimension."""
        return self.origin[1]
    
    @property
    def z(self) -> np.ndarray:
        """Get the z-coordinates."""
        return self.get_coordinates(0)
    
    @property
    def x(self) -> np.ndarray:
        """Get the x-coordinates."""
        return self.get_coordinates(1)
    
    @property
    def z_extent(self) -> Tuple[float, float]:
        """Get the extent in the z-dimension."""
        return (self.extent[0], self.extent[1])
    
    @property
    def x_extent(self) -> Tuple[float, float]:
        """Get the extent in the x-dimension."""
        return (self.extent[2], self.extent[3])


class VelocityModel3D(VelocityModel):
    """
    3D velocity model class.
    
    This class represents a 3D velocity model with x, y, and z dimensions.
    """
    
    def __init__(
        self,
        velocity_data: np.ndarray,
        grid_spacing: Union[float, List[float], Tuple[float, float, float]] = 1.0,
        origin: Optional[Union[float, List[float], Tuple[float, float, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a 3D velocity model.
        
        Args:
            velocity_data: 3D array of velocity values with shape (nz, ny, nx).
            grid_spacing: Grid spacing in the z, y, and x dimensions.
            origin: Origin coordinates in the z, y, and x dimensions.
            metadata: Additional metadata for the velocity model.
        """
        # Ensure the data is 3D
        velocity_data = np.asarray(velocity_data)
        if velocity_data.ndim != 3:
            if velocity_data.ndim > 3 and np.prod(velocity_data.shape[3:]) == 1:
                # Reshape to 3D if all other dimensions are singleton
                velocity_data = velocity_data.reshape(
                    velocity_data.shape[0], velocity_data.shape[1], velocity_data.shape[2]
                )
            else:
                raise ValueError(f"Expected 3D array, got shape {velocity_data.shape}")
        
        super().__init__(velocity_data, grid_spacing, origin, metadata)
    
    @property
    def nz(self) -> int:
        """Get the number of grid points in the z-dimension."""
        return self.shape[0]
    
    @property
    def ny(self) -> int:
        """Get the number of grid points in the y-dimension."""
        return self.shape[1]
    
    @property
    def nx(self) -> int:
        """Get the number of grid points in the x-dimension."""
        return self.shape[2]
    
    @property
    def dz(self) -> float:
        """Get the grid spacing in the z-dimension."""
        return self.grid_spacing[0]
    
    @property
    def dy(self) -> float:
        """Get the grid spacing in the y-dimension."""
        return self.grid_spacing[1]
    
    @property
    def dx(self) -> float:
        """Get the grid spacing in the x-dimension."""
        return self.grid_spacing[2]
    
    @property
    def z_origin(self) -> float:
        """Get the origin coordinate in the z-dimension."""
        return self.origin[0]
    
    @property
    def y_origin(self) -> float:
        """Get the origin coordinate in the y-dimension."""
        return self.origin[1]
    
    @property
    def x_origin(self) -> float:
        """Get the origin coordinate in the x-dimension."""
        return self.origin[2]
    
    @property
    def z(self) -> np.ndarray:
        """Get the z-coordinates."""
        return self.get_coordinates(0)
    
    @property
    def y(self) -> np.ndarray:
        """Get the y-coordinates."""
        return self.get_coordinates(1)
    
    @property
    def x(self) -> np.ndarray:
        """Get the x-coordinates."""
        return self.get_coordinates(2)
    
    @property
    def z_extent(self) -> Tuple[float, float]:
        """Get the extent in the z-dimension."""
        return (self.extent[0], self.extent[1])
    
    @property
    def y_extent(self) -> Tuple[float, float]:
        """Get the extent in the y-dimension."""
        return (self.extent[2], self.extent[3])
    
    @property
    def x_extent(self) -> Tuple[float, float]:
        """Get the extent in the x-dimension."""
        return (self.extent[4], self.extent[5])


def load_velocity_model(path: Union[str, Path]) -> VelocityModel:
    """
    Load a velocity model from a file.
    
    This is a convenience function that calls VelocityModel.load().
    
    Args:
        path: Path to the file.
        
    Returns:
        A VelocityModel instance.
    """
    return VelocityModel.load(path)