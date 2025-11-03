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

from .velocity_map import Dimensionality, VelocityMap, save_velocity_maps, load_velocity_maps


class VelocityModel(ABC):
    """
    Abstract base class for velocity models.
    
    This class defines the interface for velocity models of different dimensions.
    It provides methods for accessing the velocity data in different formats,
    getting metadata about the model, and serialization.
    
    This class uses VelocityMap internally to store and manage velocity data.
    """
    
    def __init__(
        self,
        velocity_data: Union[np.ndarray, VelocityMap],
        grid_spacing: Union[float, List[float], Tuple[float, ...]],
        origin: Optional[Union[float, List[float], Tuple[float, ...]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a velocity model.
        
        Args:
            velocity_data: Velocity data as a numpy array or VelocityMap.
            grid_spacing: Grid spacing in each dimension. If a single value is provided,
                it will be used for all dimensions. This is a required parameter when
                velocity_data is a numpy array. Ignored if velocity_data is a VelocityMap.
            origin: Origin coordinates for each dimension. If None, defaults to zeros.
                Ignored if velocity_data is a VelocityMap.
            metadata: Additional metadata for the velocity model.
                
        Raises:
            ValueError: If grid_spacing is not provided when velocity_data is a numpy array.
        """
        # Handle VelocityMap input
        if isinstance(velocity_data, VelocityMap):
            self._velocity_map = velocity_data
            self._metadata = metadata or {}
            return
            
        # Handle numpy array input for backward compatibility
        velocity_array = np.asarray(velocity_data, dtype=np.float32)
        ndim = velocity_array.ndim
        
        # Set grid spacing
        if isinstance(grid_spacing, (int, float)):
            grid_spacing_tuple = tuple([float(grid_spacing)] * ndim)
        else:
            if len(grid_spacing) != ndim:
                raise ValueError(f"Expected {ndim} grid spacing values, got {len(grid_spacing)}")
            grid_spacing_tuple = tuple(float(x) for x in grid_spacing)
        
        # Set origin
        if origin is None:
            origin_tuple = tuple([0.0] * ndim)
        else:
            if isinstance(origin, (int, float)):
                origin_tuple = tuple([float(origin)] * ndim)
            else:
                if len(origin) != ndim:
                    raise ValueError(f"Expected {ndim} origin values, got {len(origin)}")
                origin_tuple = tuple(float(x) for x in origin)
        
        # Create VelocityMap based on dimensionality
        if ndim == 1:
            self._velocity_map = VelocityMap.from_1d_array(velocity_array, grid_spacing_tuple[0])
        elif ndim == 2:
            self._velocity_map = VelocityMap.from_2d_array(velocity_array, grid_spacing_tuple[0], grid_spacing_tuple[1])
        elif ndim == 3:
            self._velocity_map = VelocityMap.from_3d_array(velocity_array, grid_spacing_tuple[0], grid_spacing_tuple[1], grid_spacing_tuple[2])
        else:
            raise ValueError(f"Unsupported number of dimensions: {ndim}")
        
        # Set metadata
        self._metadata = metadata or {}
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the velocity model."""
        if self._velocity_map.dimensionality == Dimensionality.DIM_1D:
            return 1
        elif self._velocity_map.dimensionality == Dimensionality.DIM_2D:
            return 2
        elif self._velocity_map.dimensionality == Dimensionality.DIM_3D:
            return 3
        else:
            raise ValueError(f"Unsupported dimensionality: {self._velocity_map.dimensionality}")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the velocity model."""
        return self._velocity_map.shape
    
    @property
    def grid_spacing(self) -> Tuple[float, ...]:
        """Get the grid spacing in each dimension."""
        if self._velocity_map.dimensionality == Dimensionality.DIM_1D:
            return (self._velocity_map.dz,)
        elif self._velocity_map.dimensionality == Dimensionality.DIM_2D:
            return (self._velocity_map.dx, self._velocity_map.dz)
        elif self._velocity_map.dimensionality == Dimensionality.DIM_3D:
            return (self._velocity_map.dx, self._velocity_map.dy, self._velocity_map.dz)
        else:
            raise ValueError(f"Unsupported dimensionality: {self._velocity_map.dimensionality}")
    
    @property
    def origin(self) -> Tuple[float, ...]:
        """Get the origin coordinates in each dimension."""
        # VelocityMap doesn't store origin, so we use the default (0.0)
        return tuple([0.0] * self.ndim)
    
    @property
    def extent(self) -> Tuple[float, ...]:
        """
        Get the extent of the velocity model in each dimension.
        
        Returns:
            A tuple of (min_x, max_x, min_y, max_y, ...) for each dimension.
        """
        result = []
        for i in range(self.ndim):
            min_val = self.origin[i]
            max_val = self.origin[i] + (self.shape[i] - 1) * self.grid_spacing[i]
            result.extend([min_val, max_val])
        return tuple(result)
    
    @property
    def min_velocity(self) -> float:
        """Get the minimum velocity value."""
        return float(np.min(self._velocity_map.data))
    
    @property
    def max_velocity(self) -> float:
        """Get the maximum velocity value."""
        return float(np.max(self._velocity_map.data))
    
    @property
    def mean_velocity(self) -> float:
        """Get the mean velocity value."""
        return float(np.mean(self._velocity_map.data))
    
    @property
    def std_velocity(self) -> float:
        """Get the standard deviation of velocity values."""
        return float(np.std(self._velocity_map.data))
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata dictionary."""
        return self._metadata
    
    def __str__(self) -> str:
        """
        Return a string representation of the velocity model.
        
        Returns:
            A string with basic information about the velocity model.
        """
        return (f"{self.__class__.__name__}(shape={self.shape}, "
                f"grid_spacing={self.grid_spacing}, origin={self.origin}, "
                f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}])")
    
    def __repr__(self) -> str:
        """
        Return a detailed string representation of the velocity model.
        
        Returns:
            A string with detailed information about the velocity model.
        """
        return (f"{self.__class__.__name__}(shape={self.shape}, "
                f"grid_spacing={self.grid_spacing}, origin={self.origin}, "
                f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
                f"mean_velocity={self.mean_velocity:.2f}, std_velocity={self.std_velocity:.2f}, "
                f"metadata={self.metadata})")
    
    def as_numpy(self) -> np.ndarray:
        """Get the velocity data as a numpy array."""
        return self._velocity_map.data
    
    def as_list(self) -> List:
        """Get the velocity data as a nested list."""
        return self._velocity_map.data.tolist()
    
    def as_torch(self) -> Any:
        """Get the velocity data as a PyTorch tensor."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Install it with 'pip install torch'.")
        return torch.from_numpy(self._velocity_map.data)
    
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
        
        return np.arange(self.shape[dimension]) * self.grid_spacing[dimension] + self.origin[dimension]
    
    def save(self, path: Union[str, Path], format: str = "numpy") -> None:
        """
        Save the velocity model to a file.
        
        Args:
            path: Path to save the file.
            format: Format to save the file in. Options are "numpy", "pickle", "json", or "velocity_map".
        """
        path = Path(path)
        
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "velocity_map":
            # Save using VelocityMap serialization
            # This is a new format that saves the VelocityMap directly
            with open(path, "wb") as f:
                pickle.dump(self._velocity_map, f)
        
        elif format == "numpy":
            # Save as .npz file with metadata
            save_dict = {
                "velocity_data": self._velocity_map.data,
                "grid_spacing": self.grid_spacing,
                "origin": self.origin,
                "metadata": self._metadata,
                "class_name": self.__class__.__name__,
                "dimensionality": str(self._velocity_map.dimensionality.value)
            }
            np.savez(path, **save_dict)
        
        elif format == "pickle":
            # Save as pickle file
            with open(path, "wb") as f:
                pickle.dump(self, f)
        
        elif format == "json":
            # Save as JSON file
            save_dict = {
                "velocity_data": self._velocity_map.data.tolist(),
                "grid_spacing": self.grid_spacing,
                "origin": self.origin,
                "metadata": self._metadata,
                "class_name": self.__class__.__name__,
                "dimensionality": str(self._velocity_map.dimensionality.value)
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
        
        # Try to load as a VelocityMap first
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
                if isinstance(obj, VelocityMap):
                    # Create the appropriate class instance based on dimensionality
                    if obj.dimensionality == Dimensionality.DIM_1D:
                        from .velocity_model import VelocityModel1D
                        return VelocityModel1D(obj)
                    elif obj.dimensionality == Dimensionality.DIM_2D:
                        from .velocity_model import VelocityModel2D
                        return VelocityModel2D(obj)
                    elif obj.dimensionality == Dimensionality.DIM_3D:
                        from .velocity_model import VelocityModel3D
                        return VelocityModel3D(obj)
                    else:
                        raise ValueError(f"Unsupported dimensionality: {obj.dimensionality}")
                elif isinstance(obj, VelocityModel):
                    # If it's already a VelocityModel, return it directly
                    return obj
        except:
            # If loading as VelocityMap fails, continue with other formats
            pass
        
        # Determine the file format based on extension
        if path.suffix == ".npz":
            # Load from .npz file
            with np.load(path, allow_pickle=True) as data:
                velocity_data = data["velocity_data"]
                grid_spacing = data["grid_spacing"].item() if "grid_spacing" in data else 1.0
                origin = data["origin"].item() if "origin" in data else None
                metadata = data["metadata"].item() if "metadata" in data else None
                class_name = data["class_name"].item() if "class_name" in data else None
                dimensionality_str = data["dimensionality"].item() if "dimensionality" in data else None
                
                # If dimensionality is provided, create a VelocityMap first
                if dimensionality_str:
                    if dimensionality_str == "1D":
                        velocity_map = VelocityMap.from_1d_array(velocity_data, grid_spacing[0])
                    elif dimensionality_str == "2D":
                        velocity_map = VelocityMap.from_2d_array(velocity_data, grid_spacing[0], grid_spacing[1])
                    elif dimensionality_str == "3D":
                        velocity_map = VelocityMap.from_3d_array(velocity_data, grid_spacing[0], grid_spacing[1], grid_spacing[2])
                    else:
                        velocity_map = None
                        
                    if velocity_map:
                        # Create the appropriate class instance
                        if class_name == "VelocityModel1D":
                            from .velocity_model import VelocityModel1D
                            model = VelocityModel1D(velocity_map)
                        elif class_name == "VelocityModel2D":
                            from .velocity_model import VelocityModel2D
                            model = VelocityModel2D(velocity_map)
                        elif class_name == "VelocityModel3D":
                            from .velocity_model import VelocityModel3D
                            model = VelocityModel3D(velocity_map)
                        else:
                            model = None
                            
                        if model:
                            model._metadata = metadata or {}
                            return model
            
            # Fall back to legacy loading if VelocityMap creation failed
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
            dimensionality_str = data.get("dimensionality", None)
            
            # If dimensionality is provided, create a VelocityMap first
            if dimensionality_str:
                if dimensionality_str == "1D":
                    velocity_map = VelocityMap.from_1d_array(velocity_data, grid_spacing[0])
                elif dimensionality_str == "2D":
                    velocity_map = VelocityMap.from_2d_array(velocity_data, grid_spacing[0], grid_spacing[1])
                elif dimensionality_str == "3D":
                    velocity_map = VelocityMap.from_3d_array(velocity_data, grid_spacing[0], grid_spacing[1], grid_spacing[2])
                else:
                    velocity_map = None
                    
                if velocity_map:
                    # Create the appropriate class instance
                    if class_name == "VelocityModel1D":
                        from .velocity_model import VelocityModel1D
                        model = VelocityModel1D(velocity_map)
                    elif class_name == "VelocityModel2D":
                        from .velocity_model import VelocityModel2D
                        model = VelocityModel2D(velocity_map)
                    elif class_name == "VelocityModel3D":
                        from .velocity_model import VelocityModel3D
                        model = VelocityModel3D(velocity_map)
                    else:
                        model = None
                        
                    if model:
                        model._metadata = metadata or {}
                        return model
            
            # Fall back to legacy loading if VelocityMap creation failed
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
        velocity_data: Union[np.ndarray, VelocityMap],
        grid_spacing: Optional[Union[float, List[float], Tuple[float]]] = None,
        origin: Optional[Union[float, List[float], Tuple[float]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a 1D velocity model.
        
        Args:
            velocity_data: 1D array of velocity values or a VelocityMap.
            grid_spacing: Grid spacing in the z-dimension. Required if velocity_data is a numpy array.
                Ignored if velocity_data is a VelocityMap.
            origin: Origin coordinate in the z-dimension. Ignored if velocity_data is a VelocityMap.
            metadata: Additional metadata for the velocity model.
            
        Raises:
            ValueError: If grid_spacing is not provided when velocity_data is a numpy array.
        """
        # Handle VelocityMap input
        if isinstance(velocity_data, VelocityMap):
            super().__init__(velocity_data, (1.0,), origin, metadata)  # grid_spacing is ignored for VelocityMap
            return
            
        # Handle numpy array input
        if grid_spacing is None:
            raise ValueError("grid_spacing is required when velocity_data is a numpy array")
            
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
    
    def __str__(self) -> str:
        """
        Return a string representation of the 1D velocity model.
        
        Returns:
            A string with basic information about the 1D velocity model.
        """
        return (f"{self.__class__.__name__}(nz={self.nz}, dz={self.dz}, "
                f"z_origin={self.z_origin}, "
                f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}])")
    
    def __repr__(self) -> str:
        """
        Return a detailed string representation of the 1D velocity model.
        
        Returns:
            A string with detailed information about the 1D velocity model.
        """
        return (f"{self.__class__.__name__}(nz={self.nz}, dz={self.dz}, "
                f"z_origin={self.z_origin}, z_extent={self.z_extent}, "
                f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
                f"mean_velocity={self.mean_velocity:.2f}, std_velocity={self.std_velocity:.2f}, "
                f"metadata={self.metadata})")


class VelocityModel2D(VelocityModel):
    """
    2D velocity model class.
    
    This class represents a 2D velocity model, typically used for cross-sections
    with x (horizontal) and z (depth) dimensions.
    """
    
    def __init__(
        self,
        velocity_data: Union[np.ndarray, VelocityMap],
        grid_spacing: Optional[Union[float, List[float], Tuple[float, float]]] = None,
        origin: Optional[Union[float, List[float], Tuple[float, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a 2D velocity model.
        
        Args:
            velocity_data: 2D array of velocity values with shape (nz, nx) or a VelocityMap.
            grid_spacing: Grid spacing in the z and x dimensions. Required if velocity_data is a numpy array.
                Ignored if velocity_data is a VelocityMap.
            origin: Origin coordinates in the z and x dimensions. Ignored if velocity_data is a VelocityMap.
            metadata: Additional metadata for the velocity model.
            
        Raises:
            ValueError: If grid_spacing is not provided when velocity_data is a numpy array.
        """
        # Handle VelocityMap input
        if isinstance(velocity_data, VelocityMap):
            super().__init__(velocity_data, (1.0, 1.0), origin, metadata)  # grid_spacing is ignored for VelocityMap
            return
            
        # Handle numpy array input
        if grid_spacing is None:
            raise ValueError("grid_spacing is required when velocity_data is a numpy array")
            
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
    
    def __str__(self) -> str:
        """
        Return a string representation of the 2D velocity model.
        
        Returns:
            A string with basic information about the 2D velocity model.
        """
        return (f"{self.__class__.__name__}(nz={self.nz}, nx={self.nx}, "
                f"dz={self.dz}, dx={self.dx}, "
                f"z_origin={self.z_origin}, x_origin={self.x_origin}, "
                f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}])")
    
    def __repr__(self) -> str:
        """
        Return a detailed string representation of the 2D velocity model.
        
        Returns:
            A string with detailed information about the 2D velocity model.
        """
        return (f"{self.__class__.__name__}(nz={self.nz}, nx={self.nx}, "
                f"dz={self.dz}, dx={self.dx}, "
                f"z_origin={self.z_origin}, x_origin={self.x_origin}, "
                f"z_extent={self.z_extent}, x_extent={self.x_extent}, "
                f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
                f"mean_velocity={self.mean_velocity:.2f}, std_velocity={self.std_velocity:.2f}, "
                f"metadata={self.metadata})")


class VelocityModel3D(VelocityModel):
    """
    3D velocity model class.
    
    This class represents a 3D velocity model with x, y, and z dimensions.
    """
    
    def __init__(
        self,
        velocity_data: Union[np.ndarray, VelocityMap],
        grid_spacing: Optional[Union[float, List[float], Tuple[float, float, float]]] = None,
        origin: Optional[Union[float, List[float], Tuple[float, float, float]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a 3D velocity model.
        
        Args:
            velocity_data: 3D array of velocity values with shape (nz, ny, nx) or a VelocityMap.
            grid_spacing: Grid spacing in the z, y, and x dimensions. Required if velocity_data is a numpy array.
                Ignored if velocity_data is a VelocityMap.
            origin: Origin coordinates in the z, y, and x dimensions. Ignored if velocity_data is a VelocityMap.
            metadata: Additional metadata for the velocity model.
            
        Raises:
            ValueError: If grid_spacing is not provided when velocity_data is a numpy array.
        """
        # Handle VelocityMap input
        if isinstance(velocity_data, VelocityMap):
            super().__init__(velocity_data, (1.0, 1.0, 1.0), origin, metadata)  # grid_spacing is ignored for VelocityMap
            return
            
        # Handle numpy array input
        if grid_spacing is None:
            raise ValueError("grid_spacing is required when velocity_data is a numpy array")
            
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
    
    def __str__(self) -> str:
        """
        Return a string representation of the 3D velocity model.
        
        Returns:
            A string with basic information about the 3D velocity model.
        """
        return (f"{self.__class__.__name__}(nz={self.nz}, ny={self.ny}, nx={self.nx}, "
                f"dz={self.dz}, dy={self.dy}, dx={self.dx}, "
                f"z_origin={self.z_origin}, y_origin={self.y_origin}, x_origin={self.x_origin}, "
                f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}])")
    
    def __repr__(self) -> str:
        """
        Return a detailed string representation of the 3D velocity model.
        
        Returns:
            A string with detailed information about the 3D velocity model.
        """
        return (f"{self.__class__.__name__}(nz={self.nz}, ny={self.ny}, nx={self.nx}, "
                f"dz={self.dz}, dy={self.dy}, dx={self.dx}, "
                f"z_origin={self.z_origin}, y_origin={self.y_origin}, x_origin={self.x_origin}, "
                f"z_extent={self.z_extent}, y_extent={self.y_extent}, x_extent={self.x_extent}, "
                f"velocity_range=[{self.min_velocity:.2f}, {self.max_velocity:.2f}], "
                f"mean_velocity={self.mean_velocity:.2f}, std_velocity={self.std_velocity:.2f}, "
                f"metadata={self.metadata})")


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


def save_velocity_models(models: List[VelocityModel], path: Union[str, Path], format: str = "pickle") -> None:
    """
    Save a list of velocity models to a file.
    
    Args:
        models: List of VelocityModel objects to save.
        path: Path to save the file.
        format: Format to save the file in. Currently only "pickle" is supported.
            This ensures the complete VelocityModel objects with all parameters are serialized.
    """
    path = Path(path)
    
    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "pickle":
        # Save as pickle file with complete VelocityModel objects
        with open(path, "wb") as f:
            pickle.dump(models, f)
        # end with
    else:
        raise ValueError(f"Unsupported format: {format}")
    # end if
# end save_velocity_models


def load_velocity_models(path: Union[str, Path]) -> List[VelocityModel]:
    """
    Load a list of velocity models from a file.
    
    Args:
        path: Path to the file.
        
    Returns:
        A list of VelocityModel instances with all parameters.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Load the complete VelocityModel objects
    try:
        with open(path, "rb") as f:
            models = pickle.load(f)
            if isinstance(models, list) and all(isinstance(model, VelocityModel) for model in models):
                return models
            else:
                raise ValueError(f"File does not contain a list of VelocityModel objects: {path}")
    except Exception as e:
        raise ValueError(f"Failed to load velocity models from {path}: {str(e)}")