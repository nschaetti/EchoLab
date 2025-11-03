"""
Classes and utilities for representing and serializing velocity models using pydantic.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, ClassVar
import json
import pickle

import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from .velocity_map_pydantic import VelocityMap, Dimensionality


class VelocityModelBase(BaseModel, ABC):
    """
    Abstract base class for velocity models using pydantic for validation.
    
    This class defines the interface for velocity models of different dimensions.
    It provides methods for accessing the velocity data in different formats,
    getting metadata about the model, and serialization.
    
    This class uses VelocityMap internally to store and manage velocity data.
    """
    velocity_map: VelocityMap
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Configure pydantic model
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow numpy arrays
    )
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the velocity model."""
        if self.velocity_map.dimensionality == Dimensionality.DIM_1D:
            return 1
        elif self.velocity_map.dimensionality == Dimensionality.DIM_2D:
            return 2
        elif self.velocity_map.dimensionality == Dimensionality.DIM_3D:
            return 3
        else:
            raise ValueError(f"Unsupported dimensionality: {self.velocity_map.dimensionality}")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the velocity model."""
        return self.velocity_map.shape
    
    @property
    def grid_spacing(self) -> Tuple[float, ...]:
        """Get the grid spacing in each dimension."""
        if self.velocity_map.dimensionality == Dimensionality.DIM_1D:
            return (self.velocity_map.dz,)
        elif self.velocity_map.dimensionality == Dimensionality.DIM_2D:
            return (self.velocity_map.dx, self.velocity_map.dz)
        elif self.velocity_map.dimensionality == Dimensionality.DIM_3D:
            return (self.velocity_map.dx, self.velocity_map.dy, self.velocity_map.dz)
        else:
            raise ValueError(f"Unsupported dimensionality: {self.velocity_map.dimensionality}")
    
    @property
    def origin(self) -> Tuple[float, ...]:
        """Get the origin coordinates in each dimension."""
        # VelocityMap doesn't store origin, so we use the default (0.0)
        return tuple([0.0] * self.ndim)
    
    @property
    def extent(self) -> Tuple[Tuple[float, float], ...]:
        """Get the extent of the velocity model in each dimension."""
        origin = self.origin
        shape = self.shape
        grid_spacing = self.grid_spacing
        
        extents = []
        for i in range(self.ndim):
            start = origin[i]
            end = start + shape[i] * grid_spacing[i]
            extents.append((start, end))
            
        return tuple(extents)
    
    @property
    def min_velocity(self) -> float:
        """Get the minimum velocity value."""
        return float(np.min(self.velocity_map.data))
    
    @property
    def max_velocity(self) -> float:
        """Get the maximum velocity value."""
        return float(np.max(self.velocity_map.data))
    
    @property
    def mean_velocity(self) -> float:
        """Get the mean velocity value."""
        return float(np.mean(self.velocity_map.data))
    
    @property
    def std_velocity(self) -> float:
        """Get the standard deviation of velocity values."""
        return float(np.std(self.velocity_map.data))
    
    def as_numpy(self) -> np.ndarray:
        """Get the velocity data as a numpy array."""
        return self.velocity_map.data
    
    def as_list(self) -> List:
        """Get the velocity data as a nested list."""
        return self.velocity_map.data.tolist()
    
    def as_torch(self) -> 'torch.Tensor':
        """Get the velocity data as a PyTorch tensor."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Install it with 'pip install torch'.")
        return torch.from_numpy(self.velocity_map.data)
    
    def get_coordinates(self, dimension: int) -> np.ndarray:
        """
        Get the coordinates along a specific dimension.
        
        Args:
            dimension: The dimension index (0 for x, 1 for y, 2 for z).
            
        Returns:
            np.ndarray: Array of coordinates along the specified dimension.
            
        Raises:
            ValueError: If the dimension is out of range.
        """
        if dimension < 0 or dimension >= self.ndim:
            raise ValueError(f"Dimension {dimension} is out of range for {self.ndim}D model")
        
        origin = self.origin[dimension]
        grid_spacing = self.grid_spacing[dimension]
        size = self.shape[dimension]
        
        return np.linspace(origin, origin + (size - 1) * grid_spacing, size)
    
    def save(self, path: Union[str, Path], format: str = "numpy") -> None:
        """
        Save the velocity model to a file.
        
        Args:
            path: Path where the file will be saved.
            format: Format to use for saving. Options are "numpy", "json", or "pickle".
            
        Raises:
            ValueError: If the format is not supported.
        """
        path = Path(path)
        
        if format == "numpy":
            # Save the data and metadata in a compressed .npz file
            # to preserve grid spacing information
            metadata = {
                "dimensionality": self.velocity_map.dimensionality.value,
                "dx": self.velocity_map.dx,
                "dy": self.velocity_map.dy,
                "dz": self.velocity_map.dz,
                "metadata": self.metadata
            }
            np.savez_compressed(path, data=self.velocity_map.data, **metadata)
        elif format == "json":
            with open(path, 'w') as f:
                json.dump({
                    "data": self.velocity_map.data.tolist(),
                    "dimensionality": self.velocity_map.dimensionality.value,
                    "dx": self.velocity_map.dx,
                    "dy": self.velocity_map.dy,
                    "dz": self.velocity_map.dz,
                    "metadata": self.metadata
                }, f)
        elif format == "pickle":
            with open(path, 'wb') as f:
                pickle.dump(self, f)
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
        
        # Check if the file exists with the given path
        if not path.exists():
            # Try adding extensions if the file doesn't exist
            if path.with_suffix('.npz').exists():
                path = path.with_suffix('.npz')
            elif path.with_suffix('.npy').exists():
                path = path.with_suffix('.npy')
            elif path.with_suffix('.json').exists():
                path = path.with_suffix('.json')
            elif path.with_suffix('.pkl').exists():
                path = path.with_suffix('.pkl')
            else:
                raise FileNotFoundError(f"File not found: {path}")
        
        # Determine the file format based on extension
        suffix = path.suffix.lower()
        
        # Try to load based on file extension
        if suffix == '.pkl':
            # Load as pickle
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                    if isinstance(model, VelocityModelBase):
                        return model
                    else:
                        raise ValueError(f"File does not contain a valid VelocityModel: {path}")
            except Exception as e:
                raise ValueError(f"Failed to load pickle file: {e}")
        
        elif suffix == '.json':
            # Load as JSON
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                    if isinstance(data, dict) and "data" in data and "dimensionality" in data:
                        velocity_array = np.array(data["data"], dtype=np.float32)
                        dimensionality = data["dimensionality"]
                        dx = data.get("dx", 1.0)
                        dy = data.get("dy")
                        dz = data.get("dz", 1.0)
                        metadata = data.get("metadata", {})
                        
                        if dimensionality == "1D":
                            velocity_map = VelocityMap.from_1d_array(velocity_array, dz)
                            return VelocityModel1D(velocity_map=velocity_map, metadata=metadata)
                        elif dimensionality == "2D":
                            velocity_map = VelocityMap.from_2d_array(velocity_array, dx, dz)
                            return VelocityModel2D(velocity_map=velocity_map, metadata=metadata)
                        elif dimensionality == "3D":
                            if dy is None:
                                raise ValueError("3D velocity model requires dy spacing parameter")
                            velocity_map = VelocityMap.from_3d_array(velocity_array, dx, dy, dz)
                            return VelocityModel3D(velocity_map=velocity_map, metadata=metadata)
                        else:
                            raise ValueError(f"Unsupported dimensionality: {dimensionality}")
                    else:
                        raise ValueError("Invalid JSON format for VelocityModel")
            except Exception as e:
                raise ValueError(f"Failed to load JSON file: {e}")
        
        elif suffix == '.npz':
            # Load as numpy compressed with metadata
            try:
                with np.load(path, allow_pickle=True) as data:
                    velocity_array = data['data']
                    ndim = velocity_array.ndim
                    
                    # Extract metadata if available
                    dimensionality = str(data.get('dimensionality', ''))
                    dx = float(data.get('dx', 1.0))
                    dy = data.get('dy', None)
                    if dy is not None and dy != 'None':
                        try:
                            dy = float(dy)
                        except (ValueError, TypeError):
                            dy = None
                    else:
                        dy = None
                    dz = float(data.get('dz', 1.0))
                    metadata = data.get('metadata', {})
                    if isinstance(metadata, np.ndarray) and metadata.size == 1:
                        metadata = metadata.item() or {}
                    
                    if ndim == 1 or dimensionality == '1D':
                        velocity_map = VelocityMap.from_1d_array(velocity_array, dz)
                        return VelocityModel1D(velocity_map=velocity_map, metadata=metadata)
                    elif ndim == 2 or dimensionality == '2D':
                        velocity_map = VelocityMap.from_2d_array(velocity_array, dx, dz)
                        return VelocityModel2D(velocity_map=velocity_map, metadata=metadata)
                    elif ndim == 3 or dimensionality == '3D':
                        if dy is None:
                            dy = 1.0  # Default value if not provided
                        velocity_map = VelocityMap.from_3d_array(velocity_array, dx, dy, dz)
                        return VelocityModel3D(velocity_map=velocity_map, metadata=metadata)
                    else:
                        raise ValueError(f"Unsupported number of dimensions: {ndim}")
            except Exception as e:
                raise ValueError(f"Failed to load npz file: {e}")
                
        elif suffix == '.npy':
            # Load as numpy without metadata
            try:
                velocity_array = np.load(path)
                ndim = velocity_array.ndim
                
                if ndim == 1:
                    velocity_map = VelocityMap.from_1d_array(velocity_array, 1.0)
                    return VelocityModel1D(velocity_map=velocity_map)
                elif ndim == 2:
                    velocity_map = VelocityMap.from_2d_array(velocity_array, 1.0, 1.0)
                    return VelocityModel2D(velocity_map=velocity_map)
                elif ndim == 3:
                    velocity_map = VelocityMap.from_3d_array(velocity_array, 1.0, 1.0, 1.0)
                    return VelocityModel3D(velocity_map=velocity_map)
                else:
                    raise ValueError(f"Unsupported number of dimensions: {ndim}")
            except Exception as e:
                raise ValueError(f"Failed to load numpy file: {e}")
        
        else:
            # Try all formats if extension is not recognized
            # Try to load as pickle first
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                    if isinstance(model, VelocityModelBase):
                        return model
            except:
                pass
            
            # Try to load as JSON
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                    if isinstance(data, dict) and "data" in data and "dimensionality" in data:
                        velocity_array = np.array(data["data"], dtype=np.float32)
                        dimensionality = data["dimensionality"]
                        dx = data.get("dx", 1.0)
                        dy = data.get("dy")
                        dz = data.get("dz", 1.0)
                        metadata = data.get("metadata", {})
                        
                        if dimensionality == "1D":
                            velocity_map = VelocityMap.from_1d_array(velocity_array, dz)
                            return VelocityModel1D(velocity_map=velocity_map, metadata=metadata)
                        elif dimensionality == "2D":
                            velocity_map = VelocityMap.from_2d_array(velocity_array, dx, dz)
                            return VelocityModel2D(velocity_map=velocity_map, metadata=metadata)
                        elif dimensionality == "3D":
                            if dy is None:
                                raise ValueError("3D velocity model requires dy spacing parameter")
                            velocity_map = VelocityMap.from_3d_array(velocity_array, dx, dy, dz)
                            return VelocityModel3D(velocity_map=velocity_map, metadata=metadata)
                        else:
                            raise ValueError(f"Unsupported dimensionality: {dimensionality}")
                    else:
                        raise ValueError("Invalid JSON format for VelocityModel")
            except:
                pass
            
            # Try to load as numpy
            try:
                velocity_array = np.load(path)
                ndim = velocity_array.ndim
                
                if ndim == 1:
                    velocity_map = VelocityMap.from_1d_array(velocity_array, 1.0)
                    return VelocityModel1D(velocity_map=velocity_map)
                elif ndim == 2:
                    velocity_map = VelocityMap.from_2d_array(velocity_array, 1.0, 1.0)
                    return VelocityModel2D(velocity_map=velocity_map)
                elif ndim == 3:
                    velocity_map = VelocityMap.from_3d_array(velocity_array, 1.0, 1.0, 1.0)
                    return VelocityModel3D(velocity_map=velocity_map)
                else:
                    raise ValueError(f"Unsupported number of dimensions: {ndim}")
            except:
                pass
            
            # If all attempts fail
            raise ValueError(f"Failed to load velocity model from {path}: Unsupported file format")


class VelocityModel1D(VelocityModelBase):
    """
    Class representing a 1D velocity model.
    
    This class provides methods specific to 1D velocity models, such as
    accessing the z-dimension properties.
    """
    
    @model_validator(mode='after')
    def validate_dimensionality(self) -> 'VelocityModel1D':
        """Validate that the velocity map has the correct dimensionality."""
        if self.velocity_map.dimensionality != Dimensionality.DIM_1D:
            raise ValueError("VelocityModel1D requires a 1D velocity map")
        return self
    
    @classmethod
    def from_array(cls, 
                  velocity_data: np.ndarray, 
                  grid_spacing: Optional[Union[float, List[float], Tuple[float]]] = None,
                  origin: Optional[Union[float, List[float], Tuple[float]]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> 'VelocityModel1D':
        """
        Create a 1D velocity model from a numpy array.
        
        Args:
            velocity_data: 1D array of velocity data.
            grid_spacing: Grid spacing in z direction. Defaults to 1.0.
            origin: Origin coordinate in z direction. Ignored (always 0.0).
            metadata: Additional metadata for the velocity model.
            
        Returns:
            VelocityModel1D: A new 1D velocity model instance.
            
        Raises:
            ValueError: If the velocity data is not 1D.
        """
        velocity_array = np.asarray(velocity_data, dtype=np.float32)
        if velocity_array.ndim != 1:
            raise ValueError(f"Expected 1D array, got {velocity_array.ndim}D")
        
        dz = float(grid_spacing) if grid_spacing is not None else 1.0
        velocity_map = VelocityMap.from_1d_array(velocity_array, dz)
        
        return cls(velocity_map=velocity_map, metadata=metadata or {})
    
    @property
    def nz(self) -> int:
        """Get the number of grid points in the z direction."""
        return self.shape[0]
    
    @property
    def dz(self) -> float:
        """Get the grid spacing in the z direction."""
        return self.grid_spacing[0]
    
    @property
    def z_origin(self) -> float:
        """Get the origin coordinate in the z direction."""
        return self.origin[0]
    
    @property
    def z(self) -> np.ndarray:
        """Get the z coordinates."""
        return self.get_coordinates(0)
    
    @property
    def z_extent(self) -> Tuple[float, float]:
        """Get the extent of the model in the z direction."""
        return self.extent[0]
    
    def __str__(self) -> str:
        """
        Return a string representation of the velocity model.
        
        Returns:
            str: A human-readable string describing the velocity model.
        """
        return (
            f"VelocityModel1D(shape=({self.nz},), "
            f"range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
            f"mean={self.mean_velocity:.2f}, "
            f"dz={self.dz:.2f})"
        )
    
    def __repr__(self) -> str:
        """
        Return a string representation that can be used to recreate the object.
        
        Returns:
            str: A string representation that can be used to recreate the object.
        """
        return (
            f"VelocityModel1D.from_array(\n"
            f"    velocity_data=np.array({self.as_list()}),\n"
            f"    grid_spacing={self.dz},\n"
            f"    metadata={self.metadata}\n"
            f")"
        )


class VelocityModel2D(VelocityModelBase):
    """
    Class representing a 2D velocity model.
    
    This class provides methods specific to 2D velocity models, such as
    accessing the x and z dimension properties.
    """
    
    @model_validator(mode='after')
    def validate_dimensionality(self) -> 'VelocityModel2D':
        """Validate that the velocity map has the correct dimensionality."""
        if self.velocity_map.dimensionality != Dimensionality.DIM_2D:
            raise ValueError("VelocityModel2D requires a 2D velocity map")
        return self
    
    @classmethod
    def from_array(cls, 
                  velocity_data: np.ndarray, 
                  grid_spacing: Optional[Union[float, List[float], Tuple[float, float]]] = None,
                  origin: Optional[Union[float, List[float], Tuple[float, float]]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> 'VelocityModel2D':
        """
        Create a 2D velocity model from a numpy array.
        
        Args:
            velocity_data: 2D array of velocity data.
            grid_spacing: Grid spacing in x and z directions. If a single value is provided,
                it will be used for both dimensions. Defaults to (1.0, 1.0).
            origin: Origin coordinates in x and z directions. Ignored (always 0.0).
            metadata: Additional metadata for the velocity model.
            
        Returns:
            VelocityModel2D: A new 2D velocity model instance.
            
        Raises:
            ValueError: If the velocity data is not 2D.
        """
        velocity_array = np.asarray(velocity_data, dtype=np.float32)
        if velocity_array.ndim != 2:
            raise ValueError(f"Expected 2D array, got {velocity_array.ndim}D")
        
        if grid_spacing is None:
            dx, dz = 1.0, 1.0
        elif isinstance(grid_spacing, (int, float)):
            dx = dz = float(grid_spacing)
        else:
            if len(grid_spacing) != 2:
                raise ValueError(f"Expected 2 grid spacing values, got {len(grid_spacing)}")
            dx, dz = float(grid_spacing[0]), float(grid_spacing[1])
        
        velocity_map = VelocityMap.from_2d_array(velocity_array, dx, dz)
        
        return cls(velocity_map=velocity_map, metadata=metadata or {})
    
    @property
    def nz(self) -> int:
        """Get the number of grid points in the z direction."""
        return self.shape[0]
    
    @property
    def nx(self) -> int:
        """Get the number of grid points in the x direction."""
        return self.shape[1]
    
    @property
    def dz(self) -> float:
        """Get the grid spacing in the z direction."""
        return self.grid_spacing[1]
    
    @property
    def dx(self) -> float:
        """Get the grid spacing in the x direction."""
        return self.grid_spacing[0]
    
    @property
    def z_origin(self) -> float:
        """Get the origin coordinate in the z direction."""
        return self.origin[1]
    
    @property
    def x_origin(self) -> float:
        """Get the origin coordinate in the x direction."""
        return self.origin[0]
    
    @property
    def z(self) -> np.ndarray:
        """Get the z coordinates."""
        return self.get_coordinates(1)
    
    @property
    def x(self) -> np.ndarray:
        """Get the x coordinates."""
        return self.get_coordinates(0)
    
    @property
    def z_extent(self) -> Tuple[float, float]:
        """Get the extent of the model in the z direction."""
        return self.extent[1]
    
    @property
    def x_extent(self) -> Tuple[float, float]:
        """Get the extent of the model in the x direction."""
        return self.extent[0]
    
    def __str__(self) -> str:
        """
        Return a string representation of the velocity model.
        
        Returns:
            str: A human-readable string describing the velocity model.
        """
        return (
            f"VelocityModel2D(shape=({self.nz}, {self.nx}), "
            f"range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
            f"mean={self.mean_velocity:.2f}, "
            f"dx={self.dx:.2f}, dz={self.dz:.2f})"
        )
    
    def __repr__(self) -> str:
        """
        Return a string representation that can be used to recreate the object.
        
        Returns:
            str: A string representation that can be used to recreate the object.
        """
        return (
            f"VelocityModel2D.from_array(\n"
            f"    velocity_data=np.array({self.as_list()}),\n"
            f"    grid_spacing=({self.dx}, {self.dz}),\n"
            f"    metadata={self.metadata}\n"
            f")"
        )


class VelocityModel3D(VelocityModelBase):
    """
    Class representing a 3D velocity model.
    
    This class provides methods specific to 3D velocity models, such as
    accessing the x, y, and z dimension properties.
    """
    
    @model_validator(mode='after')
    def validate_dimensionality(self) -> 'VelocityModel3D':
        """Validate that the velocity map has the correct dimensionality."""
        if self.velocity_map.dimensionality != Dimensionality.DIM_3D:
            raise ValueError("VelocityModel3D requires a 3D velocity map")
        return self
    
    @classmethod
    def from_array(cls, 
                  velocity_data: np.ndarray, 
                  grid_spacing: Optional[Union[float, List[float], Tuple[float, float, float]]] = None,
                  origin: Optional[Union[float, List[float], Tuple[float, float, float]]] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> 'VelocityModel3D':
        """
        Create a 3D velocity model from a numpy array.
        
        Args:
            velocity_data: 3D array of velocity data.
            grid_spacing: Grid spacing in x, y, and z directions. If a single value is provided,
                it will be used for all dimensions. Defaults to (1.0, 1.0, 1.0).
            origin: Origin coordinates in x, y, and z directions. Ignored (always 0.0).
            metadata: Additional metadata for the velocity model.
            
        Returns:
            VelocityModel3D: A new 3D velocity model instance.
            
        Raises:
            ValueError: If the velocity data is not 3D.
        """
        velocity_array = np.asarray(velocity_data, dtype=np.float32)
        if velocity_array.ndim != 3:
            raise ValueError(f"Expected 3D array, got {velocity_array.ndim}D")
        
        if grid_spacing is None:
            dx, dy, dz = 1.0, 1.0, 1.0
        elif isinstance(grid_spacing, (int, float)):
            dx = dy = dz = float(grid_spacing)
        else:
            if len(grid_spacing) != 3:
                raise ValueError(f"Expected 3 grid spacing values, got {len(grid_spacing)}")
            dx, dy, dz = float(grid_spacing[0]), float(grid_spacing[1]), float(grid_spacing[2])
        
        velocity_map = VelocityMap.from_3d_array(velocity_array, dx, dy, dz)
        
        return cls(velocity_map=velocity_map, metadata=metadata or {})
    
    @property
    def nz(self) -> int:
        """Get the number of grid points in the z direction."""
        return self.shape[2]
    
    @property
    def ny(self) -> int:
        """Get the number of grid points in the y direction."""
        return self.shape[1]
    
    @property
    def nx(self) -> int:
        """Get the number of grid points in the x direction."""
        return self.shape[0]
    
    @property
    def dz(self) -> float:
        """Get the grid spacing in the z direction."""
        return self.grid_spacing[2]
    
    @property
    def dy(self) -> float:
        """Get the grid spacing in the y direction."""
        return self.grid_spacing[1]
    
    @property
    def dx(self) -> float:
        """Get the grid spacing in the x direction."""
        return self.grid_spacing[0]
    
    @property
    def z_origin(self) -> float:
        """Get the origin coordinate in the z direction."""
        return self.origin[2]
    
    @property
    def y_origin(self) -> float:
        """Get the origin coordinate in the y direction."""
        return self.origin[1]
    
    @property
    def x_origin(self) -> float:
        """Get the origin coordinate in the x direction."""
        return self.origin[0]
    
    @property
    def z(self) -> np.ndarray:
        """Get the z coordinates."""
        return self.get_coordinates(2)
    
    @property
    def y(self) -> np.ndarray:
        """Get the y coordinates."""
        return self.get_coordinates(1)
    
    @property
    def x(self) -> np.ndarray:
        """Get the x coordinates."""
        return self.get_coordinates(0)
    
    @property
    def z_extent(self) -> Tuple[float, float]:
        """Get the extent of the model in the z direction."""
        return self.extent[2]
    
    @property
    def y_extent(self) -> Tuple[float, float]:
        """Get the extent of the model in the y direction."""
        return self.extent[1]
    
    @property
    def x_extent(self) -> Tuple[float, float]:
        """Get the extent of the model in the x direction."""
        return self.extent[0]
    
    def __str__(self) -> str:
        """
        Return a string representation of the velocity model.
        
        Returns:
            str: A human-readable string describing the velocity model.
        """
        return (
            f"VelocityModel3D(shape=({self.nz}, {self.ny}, {self.nx}), "
            f"range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
            f"mean={self.mean_velocity:.2f}, "
            f"dx={self.dx:.2f}, dy={self.dy:.2f}, dz={self.dz:.2f})"
        )
    
    def __repr__(self) -> str:
        """
        Return a string representation that can be used to recreate the object.
        
        Returns:
            str: A string representation that can be used to recreate the object.
        """
        return (
            f"VelocityModel3D.from_array(\n"
            f"    velocity_data=np.array({self.as_list()}),\n"
            f"    grid_spacing=({self.dx}, {self.dy}, {self.dz}),\n"
            f"    metadata={self.metadata}\n"
            f")"
        )


def load_velocity_model(path: Union[str, Path]) -> VelocityModelBase:
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
    return VelocityModelBase.load(path)


def save_velocity_models(models: List[VelocityModelBase], path: Union[str, Path], format: str = "pickle") -> None:
    """
    Save a list of velocity models to a file.
    
    Args:
        models: List of velocity models to save.
        path: Path where the file will be saved.
        format: Format to use for saving. Only "pickle" is supported for multiple models.
        
    Raises:
        ValueError: If the format is not supported.
    """
    if format != "pickle":
        raise ValueError("Only 'pickle' format is supported for saving multiple models")
    
    path = Path(path)
    with open(path, 'wb') as f:
        pickle.dump(models, f)


def load_velocity_models(path: Union[str, Path]) -> List[VelocityModelBase]:
    """
    Load a list of velocity models from a file.
    
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
    
    with open(path, 'rb') as f:
        try:
            models = pickle.load(f)
            if not isinstance(models, list) or not all(isinstance(model, VelocityModelBase) for model in models):
                raise ValueError("File does not contain a valid list of velocity models")
            return models
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Error loading velocity models: {e}")