"""
Classes and utilities for representing and serializing velocity models using pydantic.
"""

from __future__ import annotations

import pickle
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, ClassVar

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class Dimensionality(str, Enum):
    """Enum representing the dimensionality of a velocity model."""
    DIM_1D = "1D"
    DIM_2D = "2D"
    DIM_3D = "3D"


class VelocityMap(BaseModel):
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
    dimensionality: Dimensionality
    dx: float
    dz: float
    dy: Optional[float] = None
    
    # Configure pydantic model
    model_config = {
        "arbitrary_types_allowed": True,  # Allow numpy arrays
    }
    
    @model_validator(mode='after')
    def validate_data_shape(self) -> 'VelocityMap':
        """Validate that the data shape matches the specified dimensionality."""
        if self.dimensionality == Dimensionality.DIM_1D and len(self.data.shape) != 1:
            raise ValueError("1D velocity map should have 1D data array")
        elif self.dimensionality == Dimensionality.DIM_2D and len(self.data.shape) != 2:
            raise ValueError("2D velocity map should have 2D data array")
        elif self.dimensionality == Dimensionality.DIM_3D:
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
        
        result = f"VelocityMap({self.dimensionality}, shape={shape_str}, "
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
    def from_1d_array(cls, data: np.ndarray, dz: float) -> VelocityMap:
        """
        Create a 1D VelocityMap from a 1D numpy array.
        
        Args:
            data (np.ndarray): 1D array of velocity data.
            dz (float): Grid spacing in z direction.
            
        Returns:
            VelocityMap: A new 1D VelocityMap instance.
        """
        return cls(data=data, dimensionality=Dimensionality.DIM_1D, dx=0.0, dz=dz)
    
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
        return cls(data=data, dimensionality=Dimensionality.DIM_2D, dx=dx, dz=dz)
    
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
        return cls(data=data, dimensionality=Dimensionality.DIM_3D, dx=dx, dz=dz, dy=dy)


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