"""
Classes and utilities for representing and serializing velocity models.

This module provides a unified interface for working with velocity models,
including both basic implementations and Pydantic-based implementations.
"""

from __future__ import annotations

import pickle
import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, ClassVar

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

# Part 1: Basic Dimensionality and VelocityMap classes
# (from velocity_map.py)

class Dimensionality(Enum):
    """Enum representing the dimensionality of a velocity model."""
    DIM_1D = "1D"
    DIM_2D = "2D"
    DIM_3D = "3D"


class VelocityMap:
    """
    Class representing a velocity model with specific dimensionality.
    
    Attributes:
        data (np.ndarray): The velocity data.
        dimensionality (Dimensionality): The dimensionality of the model (1D, 2D, or 3D).
        dx (float): Grid spacing in x direction.
        dy (float, optional): Grid spacing in y direction (for 3D models).
        dz (float): Grid spacing in z direction.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        dimensionality: Dimensionality,
        dx: float,
        dz: float,
        dy: Optional[float] = None,
    ):
        """
        Initialize a VelocityMap.
        
        Args:
            data (np.ndarray): The velocity data.
            dimensionality (Dimensionality): The dimensionality of the model.
            dx (float): Grid spacing in x direction.
            dz (float): Grid spacing in z direction.
            dy (float, optional): Grid spacing in y direction (for 3D models).
        
        Raises:
            ValueError: If the data shape doesn't match the specified dimensionality.
        """
        self.data = data
        self.dimensionality = dimensionality
        self.dx = dx
        self.dz = dz
        self.dy = dy
        
        # Validate data shape matches dimensionality
        if dimensionality == Dimensionality.DIM_1D and len(data.shape) != 1:
            raise ValueError("1D velocity map should have 1D data array")
        elif dimensionality == Dimensionality.DIM_2D and len(data.shape) != 2:
            raise ValueError("2D velocity map should have 2D data array")
        elif dimensionality == Dimensionality.DIM_3D:
            if len(data.shape) != 3:
                raise ValueError("3D velocity map should have 3D data array")
            if dy is None:
                raise ValueError("3D velocity map requires dy spacing parameter")
    
    @property
    def shape(self):
        """Return the shape of the velocity data."""
        return self.data.shape
        
    def __str__(self):
        """
        Return a string representation of the VelocityMap.
        
        Returns:
            str: A human-readable string describing the velocity map.
        """
        shape_str = "x".join(str(dim) for dim in self.shape)
        min_val = np.min(self.data)
        max_val = np.max(self.data)
        mean_val = np.mean(self.data)
        
        result = f"VelocityMap({self.dimensionality.value}, shape={shape_str}, "
        result += f"range=[{min_val:.2f}, {max_val:.2f}], mean={mean_val:.2f}, "
        result += f"dx={self.dx:.2f}, dz={self.dz:.2f}"
        
        if self.dimensionality == Dimensionality.DIM_3D:
            result += f", dy={self.dy:.2f}"
            
        result += ")"
        return result
    
    def __repr__(self):
        """
        Return a string representation that can be used to recreate the object.
        
        Returns:
            str: A string representation that can be used to recreate the object.
        """
        if self.dimensionality == Dimensionality.DIM_1D:
            return f"VelocityMap.from_1d_array(np.array({self.data.tolist()}), dz={self.dz})"
        elif self.dimensionality == Dimensionality.DIM_2D:
            return f"VelocityMap.from_2d_array(np.array({self.data.tolist()}), dx={self.dx}, dz={self.dz})"
        else:  # DIM_3D
            return f"VelocityMap.from_3d_array(np.array({self.data.tolist()}), dx={self.dx}, dy={self.dy}, dz={self.dz})"
    
    @classmethod
    def from_2d_array(cls, data: np.ndarray, dx: float, dz: float) -> VelocityMap:
        """
        Create a 2D VelocityMap from a 2D numpy array.
        
        Args:
            data (np.ndarray): 2D array of velocity data.
            dx (float): Grid spacing in x direction.
            dz (float): Grid spacing in z direction.
            
        Returns:
            VelocityMap: A new 2D VelocityMap instance.
        """
        return cls(data, Dimensionality.DIM_2D, dx, dz)
    
    @classmethod
    def from_1d_array(cls, data: np.ndarray, dz: float) -> VelocityMap:
        """
        Create a 1D VelocityMap from a 1D numpy array.
        
        Args:
            data (np.ndarray): 1D array of velocity data.
            dz (float): Grid spacing in z direction.
            
        Returns:
            VelocityMap: A new 1D VelocityMap instance.
        """
        return cls(data, Dimensionality.DIM_1D, 0.0, dz)
    
    @classmethod
    def from_3d_array(cls, data: np.ndarray, dx: float, dy: float, dz: float) -> VelocityMap:
        """
        Create a 3D VelocityMap from a 3D numpy array.
        
        Args:
            data (np.ndarray): 3D array of velocity data.
            dx (float): Grid spacing in x direction.
            dy (float): Grid spacing in y direction.
            dz (float): Grid spacing in z direction.
            
        Returns:
            VelocityMap: A new 3D VelocityMap instance.
        """
        return cls(data, Dimensionality.DIM_3D, dx, dz, dy)


def save_velocity_maps(velocity_maps: List[VelocityMap], file_path: Union[str, Path]) -> None:
    """
    Save a list of VelocityMap objects to a file using pickle serialization.
    
    Args:
        velocity_maps (List[VelocityMap]): List of VelocityMap objects to save.
        file_path (Union[str, Path]): Path where the file will be saved.
    """
    file_path = Path(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(velocity_maps, f)


def load_velocity_maps(file_path: Union[str, Path]) -> List[VelocityMap]:
    """
    Load a list of VelocityMap objects from a file.
    
    Args:
        file_path (Union[str, Path]): Path to the file containing serialized VelocityMap objects.
        
    Returns:
        List[VelocityMap]: The loaded list of VelocityMap objects.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file doesn't contain a valid list of VelocityMap objects.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        try:
            velocity_maps = pickle.load(f)
            if not isinstance(velocity_maps, list) or not all(isinstance(vm, VelocityMap) for vm in velocity_maps):
                raise ValueError("File does not contain a valid list of VelocityMap objects")
            return velocity_maps
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Error loading velocity maps: {e}")


# Part 2: Pydantic Dimensionality and VelocityMap classes
# (from velocity_map_pydantic.py)

class DimensionalityPydantic(str, Enum):
    """Enum representing the dimensionality of a velocity model."""
    DIM_1D = "1D"
    DIM_2D = "2D"
    DIM_3D = "3D"


class VelocityMapPydantic(BaseModel):
    """
    Pydantic model representing a velocity model with specific dimensionality.
    
    Attributes:
        data (np.ndarray): The velocity data.
        dimensionality (Dimensionality): The dimensionality of the model (1D, 2D, or 3D).
        dx (float): Grid spacing in x direction.
        dy (float, optional): Grid spacing in y direction (for 3D models).
        dz (float): Grid spacing in z direction.
    """
    data: np.ndarray
    dimensionality: DimensionalityPydantic
    dx: float
    dz: float
    dy: Optional[float] = None
    
    # Configure pydantic model
    model_config = {
        "arbitrary_types_allowed": True,  # Allow numpy arrays
    }
    
    @model_validator(mode='after')
    def validate_data_shape(self) -> 'VelocityMapPydantic':
        """Validate that the data shape matches the specified dimensionality."""
        if self.dimensionality == DimensionalityPydantic.DIM_1D and len(self.data.shape) != 1:
            raise ValueError("1D velocity map should have 1D data array")
        elif self.dimensionality == DimensionalityPydantic.DIM_2D and len(self.data.shape) != 2:
            raise ValueError("2D velocity map should have 2D data array")
        elif self.dimensionality == DimensionalityPydantic.DIM_3D:
            if len(self.data.shape) != 3:
                raise ValueError("3D velocity map should have 3D data array")
            if self.dy is None:
                raise ValueError("3D velocity map requires dy spacing parameter")
        return self
    
    @property
    def shape(self):
        """Return the shape of the velocity data."""
        return self.data.shape
    
    def __str__(self):
        """
        Return a string representation of the VelocityMap.
        
        Returns:
            str: A human-readable string describing the velocity map.
        """
        shape_str = "x".join(str(dim) for dim in self.shape)
        min_val = np.min(self.data)
        max_val = np.max(self.data)
        mean_val = np.mean(self.data)
        
        result = f"VelocityMapPydantic({self.dimensionality}, shape={shape_str}, "
        result += f"range=[{min_val:.2f}, {max_val:.2f}], mean={mean_val:.2f}, "
        result += f"dx={self.dx:.2f}, dz={self.dz:.2f}"
        
        if self.dimensionality == DimensionalityPydantic.DIM_3D:
            result += f", dy={self.dy:.2f}"
            
        result += ")"
        return result
    
    def __repr__(self):
        """
        Return a string representation that can be used to recreate the object.
        
        Returns:
            str: A string representation that can be used to recreate the object.
        """
        if self.dimensionality == DimensionalityPydantic.DIM_1D:
            return f"VelocityMapPydantic.from_1d_array(np.array({self.data.tolist()}), dz={self.dz})"
        elif self.dimensionality == DimensionalityPydantic.DIM_2D:
            return f"VelocityMapPydantic.from_2d_array(np.array({self.data.tolist()}), dx={self.dx}, dz={self.dz})"
        else:  # DIM_3D
            return f"VelocityMapPydantic.from_3d_array(np.array({self.data.tolist()}), dx={self.dx}, dy={self.dy}, dz={self.dz})"
    
    @classmethod
    def from_1d_array(cls, data: np.ndarray, dz: float) -> 'VelocityMapPydantic':
        """
        Create a 1D VelocityMap from a 1D numpy array.
        
        Args:
            data (np.ndarray): 1D array of velocity data.
            dz (float): Grid spacing in z direction.
            
        Returns:
            VelocityMapPydantic: A new 1D VelocityMapPydantic instance.
        """
        return cls(data=data, dimensionality=DimensionalityPydantic.DIM_1D, dx=0.0, dz=dz)
    
    @classmethod
    def from_2d_array(cls, data: np.ndarray, dx: float, dz: float) -> 'VelocityMapPydantic':
        """
        Create a 2D VelocityMap from a 2D numpy array.
        
        Args:
            data (np.ndarray): 2D array of velocity data.
            dx (float): Grid spacing in x direction.
            dz (float): Grid spacing in z direction.
            
        Returns:
            VelocityMapPydantic: A new 2D VelocityMapPydantic instance.
        """
        return cls(data=data, dimensionality=DimensionalityPydantic.DIM_2D, dx=dx, dz=dz)
    
    @classmethod
    def from_3d_array(cls, data: np.ndarray, dx: float, dy: float, dz: float) -> 'VelocityMapPydantic':
        """
        Create a 3D VelocityMap from a 3D numpy array.
        
        Args:
            data (np.ndarray): 3D array of velocity data.
            dx (float): Grid spacing in x direction.
            dy (float): Grid spacing in y direction.
            dz (float): Grid spacing in z direction.
            
        Returns:
            VelocityMapPydantic: A new 3D VelocityMapPydantic instance.
        """
        return cls(data=data, dimensionality=DimensionalityPydantic.DIM_3D, dx=dx, dz=dz, dy=dy)


def save_velocity_maps_pydantic(velocity_maps: List[VelocityMapPydantic], file_path: Union[str, Path]) -> None:
    """
    Save a list of VelocityMapPydantic objects to a file using pickle serialization.
    
    Args:
        velocity_maps (List[VelocityMapPydantic]): List of VelocityMapPydantic objects to save.
        file_path (Union[str, Path]): Path where the file will be saved.
    """
    file_path = Path(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(velocity_maps, f)


def load_velocity_maps_pydantic(file_path: Union[str, Path]) -> List[VelocityMapPydantic]:
    """
    Load a list of VelocityMapPydantic objects from a file.
    
    Args:
        file_path (Union[str, Path]): Path to the file containing serialized VelocityMapPydantic objects.
        
    Returns:
        List[VelocityMapPydantic]: The loaded list of VelocityMapPydantic objects.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file doesn't contain a valid list of VelocityMapPydantic objects.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        try:
            velocity_maps = pickle.load(f)
            if not isinstance(velocity_maps, list) or not all(isinstance(vm, VelocityMapPydantic) for vm in velocity_maps):
                raise ValueError("File does not contain a valid list of VelocityMapPydantic objects")
            return velocity_maps
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Error loading velocity maps: {e}")


# Part 3: VelocityModel abstract base class
# (from velocity_model.py)

class VelocityModel(ABC):
    """
    Abstract base class for velocity models.
    
    This class provides a common interface for working with velocity models
    of different dimensionalities (1D, 2D, 3D).
    
    Attributes:
        _velocity_data (np.ndarray): The velocity data.
        _grid_spacing (Tuple[float, ...]): Grid spacing in each dimension.
        _origin (Tuple[float, ...]): Origin coordinates in each dimension.
        _metadata (Dict[str, Any]): Additional metadata about the model.
    """
    
    def __init__(
        self,
        velocity_data: Union[np.ndarray, VelocityMap],
        grid_spacing: Union[float, List[float], Tuple[float, ...]],
        origin: Optional[Union[float, List[float], Tuple[float, ...]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a VelocityModel.
        
        Args:
            velocity_data: The velocity data as a numpy array or VelocityMap.
            grid_spacing: Grid spacing in each dimension.
            origin: Origin coordinates in each dimension.
            metadata: Additional metadata about the model.
        """
        # Extract data from VelocityMap if provided
        if isinstance(velocity_data, VelocityMap):
            self._velocity_data = velocity_data.data
            # Use grid spacing from VelocityMap if not provided
            if grid_spacing is None:
                if velocity_data.dimensionality == Dimensionality.DIM_1D:
                    grid_spacing = (velocity_data.dz,)
                elif velocity_data.dimensionality == Dimensionality.DIM_2D:
                    grid_spacing = (velocity_data.dz, velocity_data.dx)
                else:  # DIM_3D
                    grid_spacing = (velocity_data.dz, velocity_data.dy, velocity_data.dx)
        else:
            self._velocity_data = velocity_data
        
        # Convert grid_spacing to tuple
        if isinstance(grid_spacing, (int, float)):
            self._grid_spacing = (float(grid_spacing),) * len(self._velocity_data.shape)
        elif isinstance(grid_spacing, (list, tuple)):
            if len(grid_spacing) != len(self._velocity_data.shape):
                raise ValueError(f"Grid spacing dimensions ({len(grid_spacing)}) don't match data dimensions ({len(self._velocity_data.shape)})")
            self._grid_spacing = tuple(float(gs) for gs in grid_spacing)
        else:
            raise TypeError(f"Unsupported grid_spacing type: {type(grid_spacing)}")
        
        # Set origin
        if origin is None:
            self._origin = (0.0,) * len(self._velocity_data.shape)
        elif isinstance(origin, (int, float)):
            self._origin = (float(origin),) * len(self._velocity_data.shape)
        elif isinstance(origin, (list, tuple)):
            if len(origin) != len(self._velocity_data.shape):
                raise ValueError(f"Origin dimensions ({len(origin)}) don't match data dimensions ({len(self._velocity_data.shape)})")
            self._origin = tuple(float(o) for o in origin)
        else:
            raise TypeError(f"Unsupported origin type: {type(origin)}")
        
        # Set metadata
        self._metadata = metadata or {}
    
    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of the velocity model.
        
        Returns:
            int: The number of dimensions (1, 2, or 3).
        """
        return len(self._velocity_data.shape)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the velocity data.
        
        Returns:
            Tuple[int, ...]: The shape of the velocity data.
        """
        return self._velocity_data.shape
    
    @property
    def grid_spacing(self) -> Tuple[float, ...]:
        """
        Get the grid spacing in each dimension.
        
        Returns:
            Tuple[float, ...]: The grid spacing in each dimension.
        """
        return self._grid_spacing
    
    @property
    def origin(self) -> Tuple[float, ...]:
        """
        Get the origin coordinates in each dimension.
        
        Returns:
            Tuple[float, ...]: The origin coordinates in each dimension.
        """
        return self._origin
    
    @property
    def extent(self) -> Tuple[Tuple[float, float], ...]:
        """
        Get the extent of the model in each dimension.
        
        Returns:
            Tuple[Tuple[float, float], ...]: The extent (min, max) in each dimension.
        """
        extents = []
        for i, (origin, size, spacing) in enumerate(zip(self._origin, self._velocity_data.shape, self._grid_spacing)):
            min_val = origin
            max_val = origin + (size - 1) * spacing
            extents.append((min_val, max_val))
        return tuple(extents)
    
    @property
    def min_velocity(self) -> float:
        """
        Get the minimum velocity value in the model.
        
        Returns:
            float: The minimum velocity value.
        """
        return float(np.min(self._velocity_data))
    
    @property
    def max_velocity(self) -> float:
        """
        Get the maximum velocity value in the model.
        
        Returns:
            float: The maximum velocity value.
        """
        return float(np.max(self._velocity_data))
    
    @property
    def mean_velocity(self) -> float:
        """
        Get the mean velocity value in the model.
        
        Returns:
            float: The mean velocity value.
        """
        return float(np.mean(self._velocity_data))
    
    @property
    def std_velocity(self) -> float:
        """
        Get the standard deviation of velocity values in the model.
        
        Returns:
            float: The standard deviation of velocity values.
        """
        return float(np.std(self._velocity_data))
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get the metadata associated with the model.
        
        Returns:
            Dict[str, Any]: The metadata dictionary.
        """
        return self._metadata
    
    def __str__(self) -> str:
        """
        Return a string representation of the velocity model.
        
        Returns:
            str: A human-readable string describing the velocity model.
        """
        shape_str = "x".join(str(dim) for dim in self.shape)
        result = f"{self.__class__.__name__}(shape={shape_str}, "
        result += f"range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
        result += f"mean={self.mean_velocity:.2f}, std={self.std_velocity:.2f})"
        return result
    
    def __repr__(self) -> str:
        """
        Return a string representation that can be used to recreate the object.
        
        Returns:
            str: A string representation that can be used to recreate the object.
        """
        # This is a simplified representation, actual recreation would require more details
        return f"{self.__class__.__name__}(velocity_data=<array of shape {self.shape}>, grid_spacing={self.grid_spacing}, origin={self.origin})"
    
    def as_numpy(self) -> np.ndarray:
        """
        Get the velocity data as a numpy array.
        
        Returns:
            np.ndarray: The velocity data.
        """
        return self._velocity_data
    
    def as_list(self) -> List:
        """
        Get the velocity data as a nested list.
        
        Returns:
            List: The velocity data as a nested list.
        """
        return self._velocity_data.tolist()
    
    def as_torch(self):
        """
        Get the velocity data as a PyTorch tensor.
        
        Returns:
            torch.Tensor: The velocity data as a PyTorch tensor.
            
        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install it with 'pip install torch'.")
        import torch
        return torch.from_numpy(self._velocity_data)
    
    def get_coordinates(self, dimension: int) -> np.ndarray:
        """
        Get the coordinates along a specific dimension.
        
        Args:
            dimension: The dimension index (0 for z, 1 for y in 3D or x in 2D, 2 for x in 3D).
            
        Returns:
            np.ndarray: The coordinates along the specified dimension.
            
        Raises:
            ValueError: If the dimension is out of range.
        """
        if dimension < 0 or dimension >= self.ndim:
            raise ValueError(f"Dimension {dimension} is out of range for a {self.ndim}D model")
        
        size = self.shape[dimension]
        origin = self.origin[dimension]
        spacing = self.grid_spacing[dimension]
        
        return np.linspace(origin, origin + (size - 1) * spacing, size)
    
    def save(self, path: Union[str, Path], format: str = "numpy") -> None:
        """
        Save the velocity model to a file.
        
        Args:
            path: Path where the file will be saved.
            format: Format to use for saving. Options are "numpy", "pickle", or "json".
            
        Raises:
            ValueError: If the format is not supported.
        """
        path = Path(path)
        
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "numpy":
            # Save as .npz file with metadata
            metadata_copy = dict(self._metadata)
            metadata_copy.update({
                "grid_spacing": self._grid_spacing,
                "origin": self._origin,
                "class_name": self.__class__.__name__,
            })
            np.savez(
                path,
                velocity_data=self._velocity_data,
                metadata=np.array([json.dumps(metadata_copy)], dtype=object)
            )
        elif format == "pickle":
            # Save as pickle file
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        elif format == "json":
            # Save as JSON file (limited to 1D and 2D models due to JSON limitations)
            if self.ndim > 2:
                raise ValueError("JSON format is only supported for 1D and 2D models")
            
            data = {
                "velocity_data": self._velocity_data.tolist(),
                "grid_spacing": self._grid_spacing,
                "origin": self._origin,
                "metadata": self._metadata,
                "class_name": self.__class__.__name__,
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'VelocityModel':
        """
        Load a velocity model from a file.
        
        Args:
            path: Path to the file containing the velocity model.
            
        Returns:
            VelocityModel: The loaded velocity model.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is not supported or the file is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Try to determine the format from the file extension
        suffix = path.suffix.lower()
        
        if suffix == ".npz":
            # Load from .npz file
            try:
                with np.load(path, allow_pickle=True) as data:
                    velocity_data = data["velocity_data"]
                    metadata_str = data["metadata"][0]
                    metadata = json.loads(metadata_str)
                    
                    # Extract grid_spacing and origin from metadata
                    grid_spacing = tuple(metadata.pop("grid_spacing"))
                    origin = tuple(metadata.pop("origin"))
                    class_name = metadata.pop("class_name", None)
                    
                    # Create the appropriate class based on dimensionality
                    if class_name == "VelocityModel1D" or (class_name is None and len(velocity_data.shape) == 1):
                        from .velocity_model import VelocityModel1D
                        return VelocityModel1D(velocity_data, grid_spacing, origin, metadata)
                    elif class_name == "VelocityModel2D" or (class_name is None and len(velocity_data.shape) == 2):
                        from .velocity_model import VelocityModel2D
                        return VelocityModel2D(velocity_data, grid_spacing, origin, metadata)
                    elif class_name == "VelocityModel3D" or (class_name is None and len(velocity_data.shape) == 3):
                        from .velocity_model import VelocityModel3D
                        return VelocityModel3D(velocity_data, grid_spacing, origin, metadata)
                    else:
                        raise ValueError(f"Unsupported class name: {class_name}")
            except Exception as e:
                raise ValueError(f"Error loading velocity model from .npz file: {e}")
        
        elif suffix == ".pkl" or suffix == ".pickle":
            # Load from pickle file
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                    if not isinstance(model, VelocityModel):
                        raise ValueError("File does not contain a valid VelocityModel object")
                    return model
            except Exception as e:
                raise ValueError(f"Error loading velocity model from pickle file: {e}")
        
        elif suffix == ".json":
            # Load from JSON file
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                    velocity_data = np.array(data["velocity_data"])
                    grid_spacing = tuple(data["grid_spacing"])
                    origin = tuple(data["origin"])
                    metadata = data.get("metadata", {})
                    class_name = data.get("class_name")
                    
                    # Create the appropriate class based on dimensionality
                    if class_name == "VelocityModel1D" or (class_name is None and len(velocity_data.shape) == 1):
                        from .velocity_model import VelocityModel1D
                        return VelocityModel1D(velocity_data, grid_spacing, origin, metadata)
                    elif class_name == "VelocityModel2D" or (class_name is None and len(velocity_data.shape) == 2):
                        from .velocity_model import VelocityModel2D
                        return VelocityModel2D(velocity_data, grid_spacing, origin, metadata)
                    else:
                        raise ValueError(f"Unsupported class name or dimensionality: {class_name}")
            except Exception as e:
                raise ValueError(f"Error loading velocity model from JSON file: {e}")
        
        else:
            # Try each format in turn
            for load_format in ["npz", "pickle", "json"]:
                try:
                    if load_format == "npz":
                        return cls.load(path.with_suffix(".npz"))
                    elif load_format == "pickle":
                        return cls.load(path.with_suffix(".pkl"))
                    elif load_format == "json":
                        return cls.load(path.with_suffix(".json"))
                except (FileNotFoundError, ValueError):
                    continue
            
            raise ValueError(f"Unsupported file format: {suffix}")


# Part 6: Pydantic VelocityModel classes
# (from velocity_model_pydantic.py)

class VelocityModelBase(BaseModel, ABC):
    """
    Abstract base class for Pydantic velocity models.
    
    This class provides a common interface for working with velocity models
    of different dimensionalities (1D, 2D, 3D) using Pydantic for validation.
    
    Attributes:
        velocity_map (VelocityMapPydantic): The velocity map.
        metadata (Dict[str, Any]): Additional metadata about the model.
    """
    velocity_map: VelocityMapPydantic
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Configure pydantic model
    model_config = {
        "arbitrary_types_allowed": True,  # Allow numpy arrays
    }
    
    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of the velocity model.
        
        Returns:
            int: The number of dimensions (1, 2, or 3).
        """
        if self.velocity_map.dimensionality == DimensionalityPydantic.DIM_1D:
            return 1
        elif self.velocity_map.dimensionality == DimensionalityPydantic.DIM_2D:
            return 2
        elif self.velocity_map.dimensionality == DimensionalityPydantic.DIM_3D:
            return 3
        else:
            raise ValueError(f"Unsupported dimensionality: {self.velocity_map.dimensionality}")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the velocity data.
        
        Returns:
            Tuple[int, ...]: The shape of the velocity data.
        """
        return self.velocity_map.shape
    
    @property
    def grid_spacing(self) -> Tuple[float, ...]:
        """
        Get the grid spacing in each dimension.
        
        Returns:
            Tuple[float, ...]: The grid spacing in each dimension.
        """
        if self.velocity_map.dimensionality == DimensionalityPydantic.DIM_1D:
            return (self.velocity_map.dz,)
        elif self.velocity_map.dimensionality == DimensionalityPydantic.DIM_2D:
            return (self.velocity_map.dz, self.velocity_map.dx)
        elif self.velocity_map.dimensionality == DimensionalityPydantic.DIM_3D:
            return (self.velocity_map.dz, self.velocity_map.dy, self.velocity_map.dx)
        else:
            raise ValueError(f"Unsupported dimensionality: {self.velocity_map.dimensionality}")
    
    @property
    def origin(self) -> Tuple[float, ...]:
        """
        Get the origin coordinates in each dimension.
        
        Returns:
            Tuple[float, ...]: The origin coordinates in each dimension.
        """
        # Default origin is (0, 0, 0) or (0, 0) or (0,) depending on dimensionality
        return (0.0,) * self.ndim
    
    @property
    def extent(self) -> Tuple[Tuple[float, float], ...]:
        """
        Get the extent of the model in each dimension.
        
        Returns:
            Tuple[Tuple[float, float], ...]: The extent (min, max) in each dimension.
        """
        extents = []
        for i, (origin, size, spacing) in enumerate(zip(self.origin, self.shape, self.grid_spacing)):
            min_val = origin
            max_val = origin + (size - 1) * spacing
            extents.append((min_val, max_val))
        return tuple(extents)
    
    @property
    def min_velocity(self) -> float:
        """
        Get the minimum velocity value in the model.
        
        Returns:
            float: The minimum velocity value.
        """
        return float(np.min(self.velocity_map.data))
    
    @property
    def max_velocity(self) -> float:
        """
        Get the maximum velocity value in the model.
        
        Returns:
            float: The maximum velocity value.
        """
        return float(np.max(self.velocity_map.data))
    
    @property
    def mean_velocity(self) -> float:
        """
        Get the mean velocity value in the model.
        
        Returns:
            float: The mean velocity value.
        """
        return float(np.mean(self.velocity_map.data))
    
    @property
    def std_velocity(self) -> float:
        """
        Get the standard deviation of velocity values in the model.
        
        Returns:
            float: The standard deviation of velocity values.
        """
        return float(np.std(self.velocity_map.data))
    
    def as_numpy(self) -> np.ndarray:
        """
        Get the velocity data as a numpy array.
        
        Returns:
            np.ndarray: The velocity data.
        """
        return self.velocity_map.data
    
    def as_list(self) -> List:
        """
        Get the velocity data as a nested list.
        
        Returns:
            List: The velocity data as a nested list.
        """
        return self.velocity_map.data.tolist()
    
    def as_torch(self) -> Any:
        """
        Get the velocity data as a PyTorch tensor.
        
        Returns:
            torch.Tensor: The velocity data as a PyTorch tensor.
            
        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install it with 'pip install torch'.")
        
        import torch
        return torch.from_numpy(self.velocity_map.data)
    
    def get_coordinates(self, dimension: int) -> np.ndarray:
        """
        Get the coordinates along a specific dimension.
        
        Args:
            dimension: The dimension index (0 for z, 1 for y in 3D or x in 2D, 2 for x in 3D).
            
        Returns:
            np.ndarray: The coordinates along the specified dimension.
            
        Raises:
            ValueError: If the dimension is out of range.
        """
        if dimension < 0 or dimension >= self.ndim:
            raise ValueError(f"Dimension {dimension} is out of range for {self.ndim}D model")
        
        size = self.shape[dimension]
        spacing = self.grid_spacing[dimension]
        origin = self.origin[dimension]
        
        return np.linspace(origin, origin + (size - 1) * spacing, size)
    
    def save(self, path: Union[str, Path], format: str = "numpy") -> None:
        """
        Save the velocity model to a file.
        
        Args:
            path: Path to save the model.
            format: Format to save the model in ('numpy', 'pickle', or 'json').
            
        Raises:
            ValueError: If the format is not supported.
        """
        path = Path(path)
        
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in the specified format
        if format.lower() == "numpy":
            # Save as .npz file
            np.savez(
                path.with_suffix(".npz"),
                velocity_data=self.velocity_map.data,
                dimensionality=self.velocity_map.dimensionality.value,
                dx=self.velocity_map.dx,
                dy=self.velocity_map.dy,
                dz=self.velocity_map.dz,
                metadata=json.dumps(self.metadata),
            )
        elif format.lower() == "pickle":
            # Save as .pkl file
            with open(path.with_suffix(".pkl"), 'wb') as f:
                pickle.dump(self, f)
        elif format.lower() == "json":
            # Save as .json file
            with open(path.with_suffix(".json"), 'w') as f:
                json.dump(
                    {
                        "velocity_data": self.velocity_map.data.tolist(),
                        "dimensionality": self.velocity_map.dimensionality.value,
                        "dx": self.velocity_map.dx,
                        "dy": self.velocity_map.dy,
                        "dz": self.velocity_map.dz,
                        "metadata": self.metadata,
                    },
                    f,
                    indent=2,
                )
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'VelocityModelBase':
        """
        Load a velocity model from a file.
        
        Args:
            path: Path to the file.
            
        Returns:
            VelocityModelBase: The loaded velocity model.
            
        Raises:
            ValueError: If the file format is not supported or the file is invalid.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix == ".npz":
            # Load from .npz file
            try:
                with np.load(path, allow_pickle=True) as data:
                    velocity_data = data["velocity_data"]
                    dimensionality = data["dimensionality"].item()
                    dx = data["dx"].item()
                    dy = data.get("dy", None)
                    if dy is not None:
                        dy = dy.item()
                    dz = data["dz"].item()
                    metadata = json.loads(data["metadata"].item())
                    
                    # Create velocity map
                    velocity_map = VelocityMapPydantic(
                        data=velocity_data,
                        dimensionality=DimensionalityPydantic(dimensionality),
                        dx=dx,
                        dy=dy,
                        dz=dz,
                    )
                    
                    # Create appropriate model based on dimensionality
                    if dimensionality == DimensionalityPydantic.DIM_1D.value:
                        from .velocity_model_pydantic import VelocityModel1D
                        return VelocityModel1D(velocity_map=velocity_map, metadata=metadata)
                    elif dimensionality == DimensionalityPydantic.DIM_2D.value:
                        from .velocity_model_pydantic import VelocityModel2D
                        return VelocityModel2D(velocity_map=velocity_map, metadata=metadata)
                    elif dimensionality == DimensionalityPydantic.DIM_3D.value:
                        from .velocity_model_pydantic import VelocityModel3D
                        return VelocityModel3D(velocity_map=velocity_map, metadata=metadata)
                    else:
                        raise ValueError(f"Unsupported dimensionality: {dimensionality}")
            except Exception as e:
                raise ValueError(f"Error loading velocity model from NPZ file: {e}")
        elif suffix == ".pkl":
            # Load from .pkl file
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                    if not isinstance(model, VelocityModelBase):
                        raise ValueError("File does not contain a valid VelocityModelBase object")
                    return model
            except Exception as e:
                raise ValueError(f"Error loading velocity model from pickle file: {e}")
        elif suffix == ".json":
            # Load from .json file
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    velocity_data = np.array(data["velocity_data"])
                    dimensionality = data["dimensionality"]
                    dx = data["dx"]
                    dy = data.get("dy")
                    dz = data["dz"]
                    metadata = data.get("metadata", {})
                    
                    # Create velocity map
                    velocity_map = VelocityMapPydantic(
                        data=velocity_data,
                        dimensionality=DimensionalityPydantic(dimensionality),
                        dx=dx,
                        dy=dy,
                        dz=dz,
                    )
                    
                    # Create appropriate model based on dimensionality
                    if dimensionality == DimensionalityPydantic.DIM_1D.value:
                        from .velocity_model_pydantic import VelocityModel1D
                        return VelocityModel1D(velocity_map=velocity_map, metadata=metadata)
                    elif dimensionality == DimensionalityPydantic.DIM_2D.value:
                        from .velocity_model_pydantic import VelocityModel2D
                        return VelocityModel2D(velocity_map=velocity_map, metadata=metadata)
                    elif dimensionality == DimensionalityPydantic.DIM_3D.value:
                        from .velocity_model_pydantic import VelocityModel3D
                        return VelocityModel3D(velocity_map=velocity_map, metadata=metadata)
                    else:
                        raise ValueError(f"Unsupported dimensionality: {dimensionality}")
            except Exception as e:
                raise ValueError(f"Error loading velocity model from JSON file: {e}")
        else:
            # Try to infer format from file extension
            # Try each format in turn
            for load_format in ["npz", "pickle", "json"]:
                try:
                    if load_format == "npz":
                        return cls.load(path.with_suffix(".npz"))
                    elif load_format == "pickle":
                        return cls.load(path.with_suffix(".pkl"))
                    elif load_format == "json":
                        return cls.load(path.with_suffix(".json"))
                except (FileNotFoundError, ValueError):
                    continue
            
            raise ValueError(f"Unsupported file format: {suffix}")

# Part 7: Concrete Pydantic VelocityModel implementations
# (from velocity_model_pydantic.py)

class VelocityModel1DPydantic(VelocityModelBase):
    """
    Pydantic class representing a 1D velocity model.
    
    A 1D velocity model represents velocity as a function of depth (z).
    
    Attributes:
        velocity_map (VelocityMapPydantic): The velocity map with dimensionality DIM_1D.
        metadata (Dict[str, Any]): Additional metadata about the model.
    """
    
    @model_validator(mode='after')
    def validate_dimensionality(self) -> 'VelocityModel1DPydantic':
        """Validate that the velocity map has the correct dimensionality."""
        if self.velocity_map.dimensionality != DimensionalityPydantic.DIM_1D:
            raise ValueError("VelocityMap must have dimensionality DIM_1D for VelocityModel1D")
        return self
    
    @classmethod
    def from_array(
        cls,
        velocity_data: np.ndarray,
        grid_spacing: Optional[Union[float, List[float], Tuple[float]]] = None,
        origin: Optional[Union[float, List[float], Tuple[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'VelocityModel1DPydantic':
        """
        Create a 1D velocity model from a numpy array.
        
        Args:
            velocity_data: The velocity data as a 1D numpy array.
            grid_spacing: Grid spacing in z direction.
            origin: Origin coordinate in z direction.
            metadata: Additional metadata about the model.
            
        Returns:
            VelocityModel1DPydantic: A new 1D velocity model.
            
        Raises:
            ValueError: If the velocity data is not 1D.
        """
        # Ensure data is 1D
        if not isinstance(velocity_data, np.ndarray):
            velocity_data = np.array(velocity_data)
        
        if len(velocity_data.shape) != 1:
            raise ValueError(f"Velocity data must be 1D, got shape {velocity_data.shape}")
        
        # Use default grid spacing if not provided
        if grid_spacing is None:
            grid_spacing = 1.0
        
        # Convert to tuple if needed
        if isinstance(grid_spacing, (list, tuple)):
            if len(grid_spacing) != 1:
                raise ValueError(f"Grid spacing must have 1 element for 1D model, got {len(grid_spacing)}")
            dz = grid_spacing[0]
        else:
            dz = float(grid_spacing)
        
        # Create velocity map
        velocity_map = VelocityMapPydantic(
            data=velocity_data,
            dimensionality=DimensionalityPydantic.DIM_1D,
            dx=0.0,
            dz=dz,
        )
        
        # Create model
        return cls(velocity_map=velocity_map, metadata=metadata or {})
    
    @property
    def nz(self) -> int:
        """
        Get the number of grid points in z direction.
        
        Returns:
            int: The number of grid points in z direction.
        """
        return self.shape[0]
    
    @property
    def dz(self) -> float:
        """
        Get the grid spacing in z direction.
        
        Returns:
            float: The grid spacing in z direction.
        """
        return self.grid_spacing[0]
    
    @property
    def z_origin(self) -> float:
        """
        Get the origin coordinate in z direction.
        
        Returns:
            float: The origin coordinate in z direction.
        """
        return self.origin[0]
    
    @property
    def z(self) -> np.ndarray:
        """
        Get the z coordinates.
        
        Returns:
            np.ndarray: The z coordinates.
        """
        return self.get_coordinates(0)
    
    @property
    def z_extent(self) -> Tuple[float, float]:
        """
        Get the extent of the model in z direction.
        
        Returns:
            Tuple[float, float]: The extent (min, max) in z direction.
        """
        return self.extent[0]
    
    def __str__(self) -> str:
        """
        Return a string representation of the 1D velocity model.
        
        Returns:
            str: A human-readable string describing the velocity model.
        """
        result = f"VelocityModel1DPydantic(nz={self.nz}, dz={self.dz:.2f}, "
        result += f"z_extent=[{self.z_extent[0]:.2f}, {self.z_extent[1]:.2f}], "
        result += f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
        result += f"mean_velocity={self.mean_velocity:.2f})"
        return result
    
    def __repr__(self) -> str:
        """
        Return a string representation that can be used to recreate the object.
        
        Returns:
            str: A string representation that can be used to recreate the object.
        """
        return (
            f"VelocityModel1DPydantic.from_array(velocity_data=np.array({self.velocity_map.data.tolist()}), "
            f"grid_spacing={self.dz}, origin={self.z_origin})"
        )


class VelocityModel2DPydantic(VelocityModelBase):
    """
    Pydantic class representing a 2D velocity model.
    
    A 2D velocity model represents velocity as a function of depth (z) and distance (x).
    
    Attributes:
        velocity_map (VelocityMapPydantic): The velocity map with dimensionality DIM_2D.
        metadata (Dict[str, Any]): Additional metadata about the model.
    """
    
    @model_validator(mode='after')
    def validate_dimensionality(self) -> 'VelocityModel2DPydantic':
        """Validate that the velocity map has the correct dimensionality."""
        if self.velocity_map.dimensionality != DimensionalityPydantic.DIM_2D:
            raise ValueError("VelocityMap must have dimensionality DIM_2D for VelocityModel2D")
        return self
    
    @classmethod
    def from_array(
        cls,
        velocity_data: np.ndarray,
        grid_spacing: Optional[Union[float, List[float], Tuple[float, float]]] = None,
        origin: Optional[Union[float, List[float], Tuple[float, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'VelocityModel2DPydantic':
        """
        Create a 2D velocity model from a numpy array.
        
        Args:
            velocity_data: The velocity data as a 2D numpy array.
            grid_spacing: Grid spacing in z and x directions.
            origin: Origin coordinates in z and x directions.
            metadata: Additional metadata about the model.
            
        Returns:
            VelocityModel2DPydantic: A new 2D velocity model.
            
        Raises:
            ValueError: If the velocity data is not 2D.
        """
        # Ensure data is 2D
        if not isinstance(velocity_data, np.ndarray):
            velocity_data = np.array(velocity_data)
        
        if len(velocity_data.shape) != 2:
            raise ValueError(f"Velocity data must be 2D, got shape {velocity_data.shape}")
        
        # Use default grid spacing if not provided
        if grid_spacing is None:
            grid_spacing = (1.0, 1.0)
        
        # Convert to tuple if needed
        if isinstance(grid_spacing, (list, tuple)):
            if len(grid_spacing) == 1:
                dz = dx = grid_spacing[0]
            elif len(grid_spacing) == 2:
                dz, dx = grid_spacing
            else:
                raise ValueError(f"Grid spacing must have 1 or 2 elements for 2D model, got {len(grid_spacing)}")
        else:
            dz = dx = float(grid_spacing)
        
        # Create velocity map
        velocity_map = VelocityMapPydantic(
            data=velocity_data,
            dimensionality=DimensionalityPydantic.DIM_2D,
            dx=dx,
            dz=dz,
        )
        
        # Create model
        return cls(velocity_map=velocity_map, metadata=metadata or {})
    
    @property
    def nz(self) -> int:
        """
        Get the number of grid points in z direction.
        
        Returns:
            int: The number of grid points in z direction.
        """
        return self.shape[0]
    
    @property
    def nx(self) -> int:
        """
        Get the number of grid points in x direction.
        
        Returns:
            int: The number of grid points in x direction.
        """
        return self.shape[1]
    
    @property
    def dz(self) -> float:
        """
        Get the grid spacing in z direction.
        
        Returns:
            float: The grid spacing in z direction.
        """
        return self.grid_spacing[0]
    
    @property
    def dx(self) -> float:
        """
        Get the grid spacing in x direction.
        
        Returns:
            float: The grid spacing in x direction.
        """
        return self.grid_spacing[1]
    
    @property
    def z_origin(self) -> float:
        """
        Get the origin coordinate in z direction.
        
        Returns:
            float: The origin coordinate in z direction.
        """
        return self.origin[0]
    
    @property
    def x_origin(self) -> float:
        """
        Get the origin coordinate in x direction.
        
        Returns:
            float: The origin coordinate in x direction.
        """
        return self.origin[1]
    
    @property
    def z(self) -> np.ndarray:
        """
        Get the z coordinates.
        
        Returns:
            np.ndarray: The z coordinates.
        """
        return self.get_coordinates(0)
    
    @property
    def x(self) -> np.ndarray:
        """
        Get the x coordinates.
        
        Returns:
            np.ndarray: The x coordinates.
        """
        return self.get_coordinates(1)
    
    @property
    def z_extent(self) -> Tuple[float, float]:
        """
        Get the extent of the model in z direction.
        
        Returns:
            Tuple[float, float]: The extent (min, max) in z direction.
        """
        return self.extent[0]
    
    @property
    def x_extent(self) -> Tuple[float, float]:
        """
        Get the extent of the model in x direction.
        
        Returns:
            Tuple[float, float]: The extent (min, max) in x direction.
        """
        return self.extent[1]
    
    def __str__(self) -> str:
        """
        Return a string representation of the 2D velocity model.
        
        Returns:
            str: A human-readable string describing the velocity model.
        """
        result = f"VelocityModel2DPydantic(nz={self.nz}, nx={self.nx}, "
        result += f"dz={self.dz:.2f}, dx={self.dx:.2f}, "
        result += f"z_extent=[{self.z_extent[0]:.2f}, {self.z_extent[1]:.2f}], "
        result += f"x_extent=[{self.x_extent[0]:.2f}, {self.x_extent[1]:.2f}], "
        result += f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
        result += f"mean_velocity={self.mean_velocity:.2f})"
        return result
    
    def __repr__(self) -> str:
        """
        Return a string representation that can be used to recreate the object.
        
        Returns:
            str: A string representation that can be used to recreate the object.
        """
        return (
            f"VelocityModel2DPydantic.from_array(velocity_data=<array of shape {self.shape}>, "
            f"grid_spacing=({self.dz}, {self.dx}), origin=({self.z_origin}, {self.x_origin}))"
        )


class VelocityModel3DPydantic(VelocityModelBase):
    """
    Pydantic class representing a 3D velocity model.
    
    A 3D velocity model represents velocity as a function of depth (z), y-distance (y), and x-distance (x).
    
    Attributes:
        velocity_map (VelocityMapPydantic): The velocity map with dimensionality DIM_3D.
        metadata (Dict[str, Any]): Additional metadata about the model.
    """
    
    @model_validator(mode='after')
    def validate_dimensionality(self) -> 'VelocityModel3DPydantic':
        """Validate that the velocity map has the correct dimensionality."""
        if self.velocity_map.dimensionality != DimensionalityPydantic.DIM_3D:
            raise ValueError("VelocityMap must have dimensionality DIM_3D for VelocityModel3D")
        return self
    
    @classmethod
    def from_array(
        cls,
        velocity_data: np.ndarray,
        grid_spacing: Optional[Union[float, List[float], Tuple[float, float, float]]] = None,
        origin: Optional[Union[float, List[float], Tuple[float, float, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'VelocityModel3DPydantic':
        """
        Create a 3D velocity model from a numpy array.
        
        Args:
            velocity_data: The velocity data as a 3D numpy array.
            grid_spacing: Grid spacing in z, y, and x directions.
            origin: Origin coordinates in z, y, and x directions.
            metadata: Additional metadata about the model.
            
        Returns:
            VelocityModel3DPydantic: A new 3D velocity model.
            
        Raises:
            ValueError: If the velocity data is not 3D.
        """
        # Ensure data is 3D
        if not isinstance(velocity_data, np.ndarray):
            velocity_data = np.array(velocity_data)
        
        if len(velocity_data.shape) != 3:
            raise ValueError(f"Velocity data must be 3D, got shape {velocity_data.shape}")
        
        # Use default grid spacing if not provided
        if grid_spacing is None:
            grid_spacing = (1.0, 1.0, 1.0)
        
        # Convert to tuple if needed
        if isinstance(grid_spacing, (list, tuple)):
            if len(grid_spacing) == 1:
                dz = dy = dx = grid_spacing[0]
            elif len(grid_spacing) == 3:
                dz, dy, dx = grid_spacing
            else:
                raise ValueError(f"Grid spacing must have 1 or 3 elements for 3D model, got {len(grid_spacing)}")
        else:
            dz = dy = dx = float(grid_spacing)
        
        # Create velocity map
        velocity_map = VelocityMapPydantic(
            data=velocity_data,
            dimensionality=DimensionalityPydantic.DIM_3D,
            dx=dx,
            dy=dy,
            dz=dz,
        )
        
        # Create model
        return cls(velocity_map=velocity_map, metadata=metadata or {})


# Part 4: Concrete VelocityModel implementations
# (from velocity_model.py)

class VelocityModel1D(VelocityModel):
    """
    Class representing a 1D velocity model.
    
    A 1D velocity model represents velocity as a function of depth (z).
    
    Attributes:
        _velocity_data (np.ndarray): 1D array of velocity data.
        _grid_spacing (Tuple[float]): Grid spacing in z direction.
        _origin (Tuple[float]): Origin coordinate in z direction.
        _metadata (Dict[str, Any]): Additional metadata about the model.
    """
    
    def __init__(
        self,
        velocity_data: Union[np.ndarray, VelocityMap],
        grid_spacing: Optional[Union[float, List[float], Tuple[float]]] = None,
        origin: Optional[Union[float, List[float], Tuple[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a 1D velocity model.
        
        Args:
            velocity_data: The velocity data as a 1D numpy array or VelocityMap.
            grid_spacing: Grid spacing in z direction.
            origin: Origin coordinate in z direction.
            metadata: Additional metadata about the model.
            
        Raises:
            ValueError: If the velocity data is not 1D.
        """
        # Extract data from VelocityMap if provided
        if isinstance(velocity_data, VelocityMap):
            if velocity_data.dimensionality != Dimensionality.DIM_1D:
                raise ValueError("VelocityMap must have dimensionality DIM_1D for VelocityModel1D")
            
            # Use grid spacing from VelocityMap if not provided
            if grid_spacing is None:
                grid_spacing = velocity_data.dz
        
        # Convert to numpy array if needed
        if not isinstance(velocity_data, np.ndarray):
            velocity_data = np.array(velocity_data)
        
        # Ensure data is 1D
        if len(velocity_data.shape) != 1:
            raise ValueError(f"Velocity data must be 1D, got shape {velocity_data.shape}")
        
        # Initialize base class
        super().__init__(velocity_data, grid_spacing, origin, metadata)
    
    @property
    def nz(self) -> int:
        """
        Get the number of grid points in z direction.
        
        Returns:
            int: The number of grid points in z direction.
        """
        return self.shape[0]
    
    @property
    def dz(self) -> float:
        """
        Get the grid spacing in z direction.
        
        Returns:
            float: The grid spacing in z direction.
        """
        return self.grid_spacing[0]
    
    @property
    def z_origin(self) -> float:
        """
        Get the origin coordinate in z direction.
        
        Returns:
            float: The origin coordinate in z direction.
        """
        return self.origin[0]
    
    @property
    def z(self) -> np.ndarray:
        """
        Get the z coordinates.
        
        Returns:
            np.ndarray: The z coordinates.
        """
        return self.get_coordinates(0)
    
    @property
    def z_extent(self) -> Tuple[float, float]:
        """
        Get the extent of the model in z direction.
        
        Returns:
            Tuple[float, float]: The extent (min, max) in z direction.
        """
        return self.extent[0]
    
    def __str__(self) -> str:
        """
        Return a string representation of the 1D velocity model.
        
        Returns:
            str: A human-readable string describing the velocity model.
        """
        result = f"VelocityModel1D(nz={self.nz}, dz={self.dz:.2f}, "
        result += f"z_extent=[{self.z_extent[0]:.2f}, {self.z_extent[1]:.2f}], "
        result += f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
        result += f"mean_velocity={self.mean_velocity:.2f})"
        return result
    
    def __repr__(self) -> str:
        """
        Return a string representation that can be used to recreate the object.
        
        Returns:
            str: A string representation that can be used to recreate the object.
        """
        return (
            f"VelocityModel1D(velocity_data=np.array({self._velocity_data.tolist()}), "
            f"grid_spacing={self.dz}, origin={self.z_origin})"
        )


class VelocityModel2D(VelocityModel):
    """
    Class representing a 2D velocity model.
    
    A 2D velocity model represents velocity as a function of depth (z) and distance (x).
    
    Attributes:
        _velocity_data (np.ndarray): 2D array of velocity data.
        _grid_spacing (Tuple[float, float]): Grid spacing in z and x directions.
        _origin (Tuple[float, float]): Origin coordinates in z and x directions.
        _metadata (Dict[str, Any]): Additional metadata about the model.
    """
    
    def __init__(
        self,
        velocity_data: Union[np.ndarray, VelocityMap],
        grid_spacing: Optional[Union[float, List[float], Tuple[float, float]]] = None,
        origin: Optional[Union[float, List[float], Tuple[float, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a 2D velocity model.
        
        Args:
            velocity_data: The velocity data as a 2D numpy array or VelocityMap.
            grid_spacing: Grid spacing in z and x directions.
            origin: Origin coordinates in z and x directions.
            metadata: Additional metadata about the model.
            
        Raises:
            ValueError: If the velocity data is not 2D.
        """
        # Extract data from VelocityMap if provided
        if isinstance(velocity_data, VelocityMap):
            if velocity_data.dimensionality != Dimensionality.DIM_2D:
                raise ValueError("VelocityMap must have dimensionality DIM_2D for VelocityModel2D")
            
            # Use grid spacing from VelocityMap if not provided
            if grid_spacing is None:
                grid_spacing = (velocity_data.dz, velocity_data.dx)
        
        # Convert to numpy array if needed
        if not isinstance(velocity_data, np.ndarray):
            velocity_data = np.array(velocity_data)
        
        # Ensure data is 2D
        if len(velocity_data.shape) != 2:
            raise ValueError(f"Velocity data must be 2D, got shape {velocity_data.shape}")
        
        # Initialize base class
        super().__init__(velocity_data, grid_spacing, origin, metadata)
    
    @property
    def nz(self) -> int:
        """
        Get the number of grid points in z direction.
        
        Returns:
            int: The number of grid points in z direction.
        """
        return self.shape[0]
    
    @property
    def nx(self) -> int:
        """
        Get the number of grid points in x direction.
        
        Returns:
            int: The number of grid points in x direction.
        """
        return self.shape[1]
    
    @property
    def dz(self) -> float:
        """
        Get the grid spacing in z direction.
        
        Returns:
            float: The grid spacing in z direction.
        """
        return self.grid_spacing[0]
    
    @property
    def dx(self) -> float:
        """
        Get the grid spacing in x direction.
        
        Returns:
            float: The grid spacing in x direction.
        """
        return self.grid_spacing[1]
    
    @property
    def z_origin(self) -> float:
        """
        Get the origin coordinate in z direction.
        
        Returns:
            float: The origin coordinate in z direction.
        """
        return self.origin[0]
    
    @property
    def x_origin(self) -> float:
        """
        Get the origin coordinate in x direction.
        
        Returns:
            float: The origin coordinate in x direction.
        """
        return self.origin[1]
    
    @property
    def z(self) -> np.ndarray:
        """
        Get the z coordinates.
        
        Returns:
            np.ndarray: The z coordinates.
        """
        return self.get_coordinates(0)
    
    @property
    def x(self) -> np.ndarray:
        """
        Get the x coordinates.
        
        Returns:
            np.ndarray: The x coordinates.
        """
        return self.get_coordinates(1)
    
    @property
    def z_extent(self) -> Tuple[float, float]:
        """
        Get the extent of the model in z direction.
        
        Returns:
            Tuple[float, float]: The extent (min, max) in z direction.
        """
        return self.extent[0]
    
    @property
    def x_extent(self) -> Tuple[float, float]:
        """
        Get the extent of the model in x direction.
        
        Returns:
            Tuple[float, float]: The extent (min, max) in x direction.
        """
        return self.extent[1]
    
    def __str__(self) -> str:
        """
        Return a string representation of the 2D velocity model.
        
        Returns:
            str: A human-readable string describing the velocity model.
        """
        result = f"VelocityModel2D(nz={self.nz}, nx={self.nx}, "
        result += f"dz={self.dz:.2f}, dx={self.dx:.2f}, "
        result += f"z_extent=[{self.z_extent[0]:.2f}, {self.z_extent[1]:.2f}], "
        result += f"x_extent=[{self.x_extent[0]:.2f}, {self.x_extent[1]:.2f}], "
        result += f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
        result += f"mean_velocity={self.mean_velocity:.2f})"
        return result
    
    def __repr__(self) -> str:
        """
        Return a string representation that can be used to recreate the object.
        
        Returns:
            str: A string representation that can be used to recreate the object.
        """
        return (
            f"VelocityModel2D(velocity_data=<array of shape {self.shape}>, "
            f"grid_spacing=({self.dz}, {self.dx}), origin=({self.z_origin}, {self.x_origin}))"
        )


class VelocityModel3D(VelocityModel):
    """
    Class representing a 3D velocity model.
    
    A 3D velocity model represents velocity as a function of depth (z), y-distance (y), and x-distance (x).
    
    Attributes:
        _velocity_data (np.ndarray): 3D array of velocity data.
        _grid_spacing (Tuple[float, float, float]): Grid spacing in z, y, and x directions.
        _origin (Tuple[float, float, float]): Origin coordinates in z, y, and x directions.
        _metadata (Dict[str, Any]): Additional metadata about the model.
    """
    
    def __init__(
        self,
        velocity_data: Union[np.ndarray, VelocityMap],
        grid_spacing: Optional[Union[float, List[float], Tuple[float, float, float]]] = None,
        origin: Optional[Union[float, List[float], Tuple[float, float, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a 3D velocity model.
        
        Args:
            velocity_data: The velocity data as a 3D numpy array or VelocityMap.
            grid_spacing: Grid spacing in z, y, and x directions.
            origin: Origin coordinates in z, y, and x directions.
            metadata: Additional metadata about the model.
            
        Raises:
            ValueError: If the velocity data is not 3D.
        """
        # Extract data from VelocityMap if provided
        if isinstance(velocity_data, VelocityMap):
            if velocity_data.dimensionality != Dimensionality.DIM_3D:
                raise ValueError("VelocityMap must have dimensionality DIM_3D for VelocityModel3D")
            
            # Use grid spacing from VelocityMap if not provided
            if grid_spacing is None:
                grid_spacing = (velocity_data.dz, velocity_data.dy, velocity_data.dx)
        
        # Convert to numpy array if needed
        if not isinstance(velocity_data, np.ndarray):
            velocity_data = np.array(velocity_data)
        
        # Ensure data is 3D
        if len(velocity_data.shape) != 3:
            raise ValueError(f"Velocity data must be 3D, got shape {velocity_data.shape}")
        
        # Initialize base class
        super().__init__(velocity_data, grid_spacing, origin, metadata)
    
    @property
    def nz(self) -> int:
        """
        Get the number of grid points in z direction.
        
        Returns:
            int: The number of grid points in z direction.
        """
        return self.shape[0]
    
    @property
    def ny(self) -> int:
        """
        Get the number of grid points in y direction.
        
        Returns:
            int: The number of grid points in y direction.
        """
        return self.shape[1]
    
    @property
    def nx(self) -> int:
        """
        Get the number of grid points in x direction.
        
        Returns:
            int: The number of grid points in x direction.
        """
        return self.shape[2]
    
    @property
    def dz(self) -> float:
        """
        Get the grid spacing in z direction.
        
        Returns:
            float: The grid spacing in z direction.
        """
        return self.grid_spacing[0]
    
    @property
    def dy(self) -> float:
        """
        Get the grid spacing in y direction.
        
        Returns:
            float: The grid spacing in y direction.
        """
        return self.grid_spacing[1]
    
    @property
    def dx(self) -> float:
        """
        Get the grid spacing in x direction.
        
        Returns:
            float: The grid spacing in x direction.
        """
        return self.grid_spacing[2]
    
    @property
    def z_origin(self) -> float:
        """
        Get the origin coordinate in z direction.
        
        Returns:
            float: The origin coordinate in z direction.
        """
        return self.origin[0]
    
    @property
    def y_origin(self) -> float:
        """
        Get the origin coordinate in y direction.
        
        Returns:
            float: The origin coordinate in y direction.
        """
        return self.origin[1]
    
    @property
    def x_origin(self) -> float:
        """
        Get the origin coordinate in x direction.
        
        Returns:
            float: The origin coordinate in x direction.
        """
        return self.origin[2]
    
    @property
    def z(self) -> np.ndarray:
        """
        Get the z coordinates.
        
        Returns:
            np.ndarray: The z coordinates.
        """
        return self.get_coordinates(0)
    
    @property
    def y(self) -> np.ndarray:
        """
        Get the y coordinates.
        
        Returns:
            np.ndarray: The y coordinates.
        """
        return self.get_coordinates(1)
    
    @property
    def x(self) -> np.ndarray:
        """
        Get the x coordinates.
        
        Returns:
            np.ndarray: The x coordinates.
        """
        return self.get_coordinates(2)
    
    @property
    def z_extent(self) -> Tuple[float, float]:
        """
        Get the extent of the model in z direction.
        
        Returns:
            Tuple[float, float]: The extent (min, max) in z direction.
        """
        return self.extent[0]
    
    @property
    def y_extent(self) -> Tuple[float, float]:
        """
        Get the extent of the model in y direction.
        
        Returns:
            Tuple[float, float]: The extent (min, max) in y direction.
        """
        return self.extent[1]
    
    @property
    def x_extent(self) -> Tuple[float, float]:
        """
        Get the extent of the model in x direction.
        
        Returns:
            Tuple[float, float]: The extent (min, max) in x direction.
        """
        return self.extent[2]
    
    def __str__(self) -> str:
        """
        Return a string representation of the 3D velocity model.
        
        Returns:
            str: A human-readable string describing the velocity model.
        """
        result = f"VelocityModel3D(nz={self.nz}, ny={self.ny}, nx={self.nx}, "
        result += f"dz={self.dz:.2f}, dy={self.dy:.2f}, dx={self.dx:.2f}, "
        result += f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
        result += f"mean_velocity={self.mean_velocity:.2f})"
        return result
    
    def __repr__(self) -> str:
        """
        Return a string representation that can be used to recreate the object.
        
        Returns:
            str: A string representation that can be used to recreate the object.
        """
        return (
            f"VelocityModel3D(velocity_data=<array of shape {self.shape}>, "
            f"grid_spacing=({self.dz}, {self.dy}, {self.dx}), "
            f"origin=({self.z_origin}, {self.y_origin}, {self.x_origin}))"
        )


# Part 8: Facade for unified API
# (from velocity_models.py)

# Type aliases for backward compatibility
VelocityModelPydantic = VelocityModelBase
VelocityModel1D_Pydantic = VelocityModel1DPydantic
VelocityModel2D_Pydantic = VelocityModel2DPydantic
VelocityModel3D_Pydantic = VelocityModel3DPydantic

# Re-export functions with the same interface
def load_velocity_model_pydantic(path: Union[str, Path]) -> VelocityModelBase:
    """
    Load a velocity model from a file using Pydantic implementation.
    
    Args:
        path: Path to the file containing the velocity model.
        
    Returns:
        VelocityModelBase: The loaded velocity model.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is not supported or the file is invalid.
    """
    return VelocityModelBase.load(path)


def save_velocity_models_pydantic(models: List[VelocityModelBase], path: Union[str, Path], format: str = "pickle") -> None:
    """
    Save a list of velocity models to a file using Pydantic implementation.
    
    Args:
        models: List of velocity models to save.
        path: Path where the file will be saved.
        format: Format to use for saving. Only "pickle" is supported for multiple models.
        
    Raises:
        ValueError: If the format is not supported.
    """
    path = Path(path)
    
    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "pickle":
        # Save as pickle file
        with open(path, 'wb') as f:
            pickle.dump(models, f)
    else:
        raise ValueError(f"Unsupported format for saving multiple models: {format}. Use 'pickle' instead.")


def load_velocity_models_pydantic(path: Union[str, Path]) -> List[VelocityModelBase]:
    """
    Load a list of velocity models from a file using Pydantic implementation.
    
    Args:
        path: Path to the file containing the velocity models.
        
    Returns:
        List[VelocityModelBase]: The loaded list of velocity models.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file doesn't contain a valid list of velocity models.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            models = pickle.load(f)
            if not isinstance(models, list) or not all(isinstance(model, VelocityModelBase) for model in models):
                raise ValueError("File does not contain a valid list of VelocityModelBase objects")
            return models
    except Exception as e:
        raise ValueError(f"Error loading velocity models: {e}")


# Export unified API
__all__ = [
    # Basic classes
    "Dimensionality",
    "VelocityMap",
    "save_velocity_maps",
    "load_velocity_maps",
    
    # Pydantic classes
    "DimensionalityPydantic",
    "VelocityMapPydantic",
    "save_velocity_maps_pydantic",
    "load_velocity_maps_pydantic",
    
    # Basic VelocityModel classes
    "VelocityModel",
    "VelocityModel1D",
    "VelocityModel2D",
    "VelocityModel3D",
    "load_velocity_model",
    "save_velocity_models",
    "load_velocity_models",
    
    # Pydantic VelocityModel classes
    "VelocityModelBase",
    "VelocityModel1DPydantic",
    "VelocityModel2DPydantic",
    "VelocityModel3DPydantic",
    "load_velocity_model_pydantic",
    "save_velocity_models_pydantic",
    "load_velocity_models_pydantic",
]


# Part 5: Utility functions for loading and saving velocity models
# (from velocity_model.py)

def load_velocity_model(path: Union[str, Path]) -> VelocityModel:
    """
    Load a velocity model from a file.
    
    Args:
        path: Path to the file containing the velocity model.
        
    Returns:
        VelocityModel: The loaded velocity model.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is not supported or the file is invalid.
    """
    return VelocityModel.load(path)


def save_velocity_models(models: List[VelocityModel], path: Union[str, Path], format: str = "pickle") -> None:
    """
    Save a list of velocity models to a file.
    
    Args:
        models: List of velocity models to save.
        path: Path where the file will be saved.
        format: Format to use for saving. Only "pickle" is supported for multiple models.
        
    Raises:
        ValueError: If the format is not supported.
    """
    path = Path(path)
    
    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "pickle":
        # Save as pickle file
        with open(path, 'wb') as f:
            pickle.dump(models, f)
    else:
        raise ValueError(f"Unsupported format for saving multiple models: {format}. Use 'pickle' instead.")


def load_velocity_models(path: Union[str, Path]) -> List[VelocityModel]:
    """
    Load a list of velocity models from a file.
    
    Args:
        path: Path to the file containing the velocity models.
        
    Returns:
        List[VelocityModel]: The loaded list of velocity models.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file doesn't contain a valid list of velocity models.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            models = pickle.load(f)
            if not isinstance(models, list) or not all(isinstance(model, VelocityModel) for model in models):
                raise ValueError("File does not contain a valid list of VelocityModel objects")
            return models
    except Exception as e:
        raise ValueError(f"Error loading velocity models: {e}")


# Part 6: Pydantic VelocityModel classes
# (from velocity_model_pydantic.py)

class VelocityModelBase(BaseModel, ABC):
    """
    Abstract base class for Pydantic velocity models.
    
    This class provides a common interface for working with velocity models
    of different dimensionalities (1D, 2D, 3D) using Pydantic for validation.
    
    Attributes:
        velocity_map (VelocityMapPydantic): The velocity map.
        metadata (Dict[str, Any]): Additional metadata about the model.
    """
    velocity_map: VelocityMapPydantic
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Configure pydantic model
    model_config = {
        "arbitrary_types_allowed": True,  # Allow numpy arrays
    }
    
    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of the velocity model.
        
        Returns:
            int: The number of dimensions (1, 2, or 3).
        """
        if self.velocity_map.dimensionality == DimensionalityPydantic.DIM_1D:
            return 1
        elif self.velocity_map.dimensionality == DimensionalityPydantic.DIM_2D:
            return 2
        elif self.velocity_map.dimensionality == DimensionalityPydantic.DIM_3D:
            return 3
        else:
            raise ValueError(f"Unsupported dimensionality: {self.velocity_map.dimensionality}")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the velocity data.
        
        Returns:
            Tuple[int, ...]: The shape of the velocity data.
        """
        return self.velocity_map.shape
    
    @property
    def grid_spacing(self) -> Tuple[float, ...]:
        """
        Get the grid spacing in each dimension.
        
        Returns:
            Tuple[float, ...]: The grid spacing in each dimension.
        """
        if self.velocity_map.dimensionality == DimensionalityPydantic.DIM_1D:
            return (self.velocity_map.dz,)
        elif self.velocity_map.dimensionality == DimensionalityPydantic.DIM_2D:
            return (self.velocity_map.dz, self.velocity_map.dx)
        elif self.velocity_map.dimensionality == DimensionalityPydantic.DIM_3D:
            return (self.velocity_map.dz, self.velocity_map.dy, self.velocity_map.dx)
        else:
            raise ValueError(f"Unsupported dimensionality: {self.velocity_map.dimensionality}")
    
    @property
    def origin(self) -> Tuple[float, ...]:
        """
        Get the origin coordinates in each dimension.
        
        Returns:
            Tuple[float, ...]: The origin coordinates in each dimension.
        """
        # Default origin is (0, 0, 0) or (0, 0) or (0,) depending on dimensionality
        return (0.0,) * self.ndim
    
    @property
    def extent(self) -> Tuple[Tuple[float, float], ...]:
        """
        Get the extent of the model in each dimension.
        
        Returns:
            Tuple[Tuple[float, float], ...]: The extent (min, max) in each dimension.
        """
        extents = []
        for i, (origin, size, spacing) in enumerate(zip(self.origin, self.shape, self.grid_spacing)):
            min_val = origin
            max_val = origin + (size - 1) * spacing
            extents.append((min_val, max_val))
        return tuple(extents)
    
    @property
    def min_velocity(self) -> float:
        """
        Get the minimum velocity value in the model.
        
        Returns:
            float: The minimum velocity value.
        """
        return float(np.min(self.velocity_map.data))
    
    @property
    def max_velocity(self) -> float:
        """
        Get the maximum velocity value in the model.
        
        Returns:
            float: The maximum velocity value.
        """
        return float(np.max(self.velocity_map.data))
    
    @property
    def mean_velocity(self) -> float:
        """
        Get the mean velocity value in the model.
        
        Returns:
            float: The mean velocity value.
        """
        return float(np.mean(self.velocity_map.data))
    
    @property
    def std_velocity(self) -> float:
        """
        Get the standard deviation of velocity values in the model.
        
        Returns:
            float: The standard deviation of velocity values.
        """
        return float(np.std(self.velocity_map.data))
    
    def as_numpy(self) -> np.ndarray:
        """
        Get the velocity data as a numpy array.
        
        Returns:
            np.ndarray: The velocity data.
        """
        return self.velocity_map.data
    
    def as_list(self) -> List:
        """
        Get the velocity data as a nested list.
        
        Returns:
            List: The velocity data as a nested list.
        """
        return self.velocity_map.data.tolist()
    
    def as_torch(self):
        """
        Get the velocity data as a PyTorch tensor.
        
        Returns:
            torch.Tensor: The velocity data as a PyTorch tensor.
            
        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install it with 'pip install torch'.")
        import torch
        return torch.from_numpy(self.velocity_map.data)
    
    def get_coordinates(self, dimension: int) -> np.ndarray:
        """
        Get the coordinates along a specific dimension.
        
        Args:
            dimension: The dimension index (0 for z, 1 for y in 3D or x in 2D, 2 for x in 3D).
            
        Returns:
            np.ndarray: The coordinates along the specified dimension.
            
        Raises:
            ValueError: If the dimension is out of range.
        """
        if dimension < 0 or dimension >= self.ndim:
            raise ValueError(f"Dimension {dimension} is out of range for a {self.ndim}D model")
        
        size = self.shape[dimension]
        origin = self.origin[dimension]
        spacing = self.grid_spacing[dimension]
        
        return np.linspace(origin, origin + (size - 1) * spacing, size)
    
    def save(self, path: Union[str, Path], format: str = "numpy") -> None:
        """
        Save the velocity model to a file.
        
        Args:
            path: Path where the file will be saved.
            format: Format to use for saving. Options are "numpy", "pickle", or "json".
            
        Raises:
            ValueError: If the format is not supported.
        """
        path = Path(path)
        
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "numpy":
            # Save as .npz file with metadata
            metadata_copy = dict(self.metadata)
            metadata_copy.update({
                "grid_spacing": self.grid_spacing,
                "origin": self.origin,
                "class_name": self.__class__.__name__,
            })
            np.savez(
                path,
                velocity_data=self.velocity_map.data,
                metadata=np.array([json.dumps(metadata_copy)], dtype=object)
            )
        elif format == "pickle":
            # Save as pickle file
            with open(path, 'wb') as f:
                pickle.dump(self, f)
        elif format == "json":
            # Save as JSON file (limited to 1D and 2D models due to JSON limitations)
            if self.ndim > 2:
                raise ValueError("JSON format is only supported for 1D and 2D models")
            
            data = {
                "velocity_data": self.velocity_map.data.tolist(),
                "grid_spacing": self.grid_spacing,
                "origin": self.origin,
                "metadata": self.metadata,
                "class_name": self.__class__.__name__,
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'VelocityModelBase':
        """
        Load a velocity model from a file.
        
        Args:
            path: Path to the file containing the velocity model.
            
        Returns:
            VelocityModelBase: The loaded velocity model.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is not supported or the file is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Try to determine the format from the file extension
        suffix = path.suffix.lower()
        
        if suffix == ".npz":
            # Load from .npz file
            try:
                with np.load(path, allow_pickle=True) as data:
                    velocity_data = data["velocity_data"]
                    metadata_str = data["metadata"][0]
                    metadata = json.loads(metadata_str)
                    
                    # Extract grid_spacing and origin from metadata
                    grid_spacing = tuple(metadata.pop("grid_spacing", (1.0,) * len(velocity_data.shape)))
                    origin = tuple(metadata.pop("origin", (0.0,) * len(velocity_data.shape)))
                    class_name = metadata.pop("class_name", None)
                    
                    # Create the appropriate class based on dimensionality
                    if class_name == "VelocityModel1D" or (class_name is None and len(velocity_data.shape) == 1):
                        from .velocity_model_pydantic import VelocityModel1D
                        return VelocityModel1D.from_array(velocity_data, grid_spacing, origin, metadata)
                    elif class_name == "VelocityModel2D" or (class_name is None and len(velocity_data.shape) == 2):
                        from .velocity_model_pydantic import VelocityModel2D
                        return VelocityModel2D.from_array(velocity_data, grid_spacing, origin, metadata)
                    elif class_name == "VelocityModel3D" or (class_name is None and len(velocity_data.shape) == 3):
                        from .velocity_model_pydantic import VelocityModel3D
                        return VelocityModel3D.from_array(velocity_data, grid_spacing, origin, metadata)
                    else:
                        raise ValueError(f"Unsupported class name: {class_name}")
            except Exception as e:
                raise ValueError(f"Error loading velocity model from .npz file: {e}")
        
        elif suffix == ".pkl" or suffix == ".pickle":
            # Load from pickle file
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                    if not isinstance(model, VelocityModelBase):
                        raise ValueError("File does not contain a valid VelocityModelBase object")
                    return model
            except Exception as e:
                raise ValueError(f"Error loading velocity model from pickle file: {e}")
        
        elif suffix == ".json":
            # Load from JSON file
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                    velocity_data = np.array(data["velocity_data"])
                    grid_spacing = tuple(data.get("grid_spacing", (1.0,) * len(velocity_data.shape)))
                    origin = tuple(data.get("origin", (0.0,) * len(velocity_data.shape)))
                    metadata = data.get("metadata", {})
                    class_name = data.get("class_name")
                    
                    # Create the appropriate class based on dimensionality
                    if class_name == "VelocityModel1D" or (class_name is None and len(velocity_data.shape) == 1):
                        from .velocity_model_pydantic import VelocityModel1D
                        return VelocityModel1D.from_array(velocity_data, grid_spacing, origin, metadata)
                    elif class_name == "VelocityModel2D" or (class_name is None and len(velocity_data.shape) == 2):
                        from .velocity_model_pydantic import VelocityModel2D
                        return VelocityModel2D.from_array(velocity_data, grid_spacing, origin, metadata)
                    else:
                        raise ValueError(f"Unsupported class name or dimensionality: {class_name}")
            except Exception as e:
                raise ValueError(f"Error loading velocity model from JSON file: {e}")
        
        else:
            # Try each format in turn
            for load_format in ["npz", "pickle", "json"]:
                try:
                    if load_format == "npz":
                        return cls.load(path.with_suffix(".npz"))
                    elif load_format == "pickle":
                        return cls.load(path.with_suffix(".pkl"))
                    elif load_format == "json":
                        return cls.load(path.with_suffix(".json"))
                except (FileNotFoundError, ValueError):
                    continue
            
            raise ValueError(f"Unsupported file format: {suffix}")