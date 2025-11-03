"""
Velocity model classes using pydantic for validation and data management.

This module provides a unified interface for working with velocity models,
using pydantic for validation and data management.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import pydantic implementations
from .velocity_map_pydantic import (
    VelocityMap, 
    Dimensionality,
    save_velocity_maps as save_velocity_maps_pydantic,
    load_velocity_maps as load_velocity_maps_pydantic
)

from .velocity_model_pydantic import (
    VelocityModelBase,
    VelocityModel1D,
    VelocityModel2D,
    VelocityModel3D,
    load_velocity_model as load_velocity_model_pydantic,
    save_velocity_models as save_velocity_models_pydantic,
    load_velocity_models as load_velocity_models_pydantic
)

# Re-export the classes and functions
__all__ = [
    'VelocityMap',
    'Dimensionality',
    'VelocityModel',
    'VelocityModel1D',
    'VelocityModel2D',
    'VelocityModel3D',
    'load_velocity_model',
    'save_velocity_models',
    'load_velocity_models',
    'save_velocity_maps',
    'load_velocity_maps'
]

# Type alias for backward compatibility
VelocityModel = VelocityModelBase

# Re-export functions with the same interface
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
    return load_velocity_model_pydantic(path)


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
    save_velocity_models_pydantic(models, path, format)


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
    return load_velocity_models_pydantic(path)


def save_velocity_maps(velocity_maps: List[VelocityMap], file_path: Union[str, Path]) -> None:
    """
    Save a list of VelocityMap objects to a file using pickle serialization.
    
    Args:
        velocity_maps (List[VelocityMap]): List of VelocityMap objects to save.
        file_path (Union[str, Path]): Path where the file will be saved.
    """
    save_velocity_maps_pydantic(velocity_maps, file_path)


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
    return load_velocity_maps_pydantic(file_path)