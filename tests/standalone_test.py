#!/usr/bin/env python
"""
Standalone test for the VelocityModel classes.

This script includes a copy of the VelocityModel classes for testing purposes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import pickle

# Define the VelocityModel classes for testing
class VelocityModel:
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


def load_velocity_model(path: Union[str, Path]) -> VelocityModel:
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
            
            # Handle grid_spacing
            if "grid_spacing" in data:
                grid_spacing_data = data["grid_spacing"]
                if grid_spacing_data.size == 1:
                    grid_spacing = float(grid_spacing_data)
                else:
                    grid_spacing = tuple(float(x) for x in grid_spacing_data)
            else:
                grid_spacing = 1.0
            
            # Handle origin
            if "origin" in data:
                origin_data = data["origin"]
                if origin_data.size == 1:
                    origin = float(origin_data)
                else:
                    origin = tuple(float(x) for x in origin_data)
            else:
                origin = None
            
            # Handle metadata and class_name
            metadata = data["metadata"].item() if "metadata" in data else None
            class_name = data["class_name"].item() if "class_name" in data else None
        
        # Create the appropriate class instance
        if class_name == "VelocityModel2D":
            return VelocityModel2D(velocity_data, grid_spacing, origin, metadata)
        else:
            # Determine the class based on the dimensions
            ndim = velocity_data.ndim
            if ndim == 2:
                return VelocityModel2D(velocity_data, grid_spacing, origin, metadata)
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
        if class_name == "VelocityModel2D":
            return VelocityModel2D(velocity_data, grid_spacing, origin, metadata)
        else:
            # Determine the class based on the dimensions
            ndim = velocity_data.ndim
            if ndim == 2:
                return VelocityModel2D(velocity_data, grid_spacing, origin, metadata)
            else:
                raise ValueError(f"Unsupported number of dimensions: {ndim}")
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def test_velocity_model_2d():
    """Test the VelocityModel2D class."""
    print("\n=== Testing VelocityModel2D ===")
    
    # Create a simple 2D velocity model
    nz, nx = 100, 200
    data = np.zeros((nz, nx))
    
    # Create a layered model
    for i in range(nz):
        if i < 20:
            data[i, :] = 1500
        elif i < 40:
            data[i, :] = 2000
        elif i < 60:
            data[i, :] = 2500
        elif i < 80:
            data[i, :] = 3000
        else:
            data[i, :] = 3500
    
    # Add a fault
    for i in range(nz):
        for j in range(nx):
            if j > nx // 2 + i // 2:
                data[i, j] += 500
    
    dx, dz = 10.0, 5.0
    
    # Create the model
    model = VelocityModel2D(data, (dz, dx))
    
    # Test basic properties
    print(f"Shape: {model.shape}")
    print(f"Grid spacing: {model.grid_spacing}")
    print(f"Origin: {model.origin}")
    print(f"Extent: {model.extent}")
    print(f"Min velocity: {model.min_velocity}")
    print(f"Max velocity: {model.max_velocity}")
    print(f"Mean velocity: {model.mean_velocity}")
    print(f"Std velocity: {model.std_velocity}")
    
    # Test dimension-specific properties
    print(f"nz, nx: {model.nz}, {model.nx}")
    print(f"dz, dx: {model.dz}, {model.dx}")
    print(f"z_origin, x_origin: {model.z_origin}, {model.x_origin}")
    print(f"z_extent, x_extent: {model.z_extent}, {model.x_extent}")
    
    # Test data access methods
    np_data = model.as_numpy()
    list_data = model.as_list()
    
    print(f"Numpy data shape: {np_data.shape}")
    print(f"List data dimensions: {len(list_data)} x {len(list_data[0])}")
    
    # Test serialization
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    
    # Save in different formats
    model.save(test_dir / "model_2d.npz", format="numpy")
    model.save(test_dir / "model_2d.pkl", format="pickle")
    model.save(test_dir / "model_2d.json", format="json")
    
    # Load from saved files
    model_npz = load_velocity_model(test_dir / "model_2d.npz")
    model_pkl = load_velocity_model(test_dir / "model_2d.pkl")
    model_json = load_velocity_model(test_dir / "model_2d.json")
    
    print(f"Loaded model (npz) shape: {model_npz.shape}")
    print(f"Loaded model (pkl) shape: {model_pkl.shape}")
    print(f"Loaded model (json) shape: {model_json.shape}")
    
    # Plot the model
    plt.figure(figsize=(10, 6))
    plt.imshow(
        model.as_numpy(),
        extent=[
            model.x_extent[0],
            model.x_extent[1],
            model.z_extent[1],
            model.z_extent[0]
        ],
        aspect="auto",
        cmap="viridis"
    )
    plt.colorbar(label="Velocity (m/s)")
    plt.title("2D Velocity Model")
    plt.xlabel("Distance (m)")
    plt.ylabel("Depth (m)")
    plt.savefig(test_dir / "model_2d.png")
    plt.close()
    
    print("VelocityModel2D test completed successfully")


def main():
    """Run the test."""
    print("Testing VelocityModel2D class...")
    
    # Create test output directory
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)
    
    # Run test
    test_velocity_model_2d()
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()