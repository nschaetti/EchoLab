"""
Test script for the pydantic-based velocity model implementation.
"""

import numpy as np
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import echolab
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the pydantic-based implementation
from echolab.modeling.velocity_models import (
    VelocityMap,
    Dimensionality,
    VelocityModel1D,
    VelocityModel2D,
    VelocityModel3D,
    load_velocity_model,
    save_velocity_models,
    load_velocity_models
)

def test_velocity_map():
    """Test VelocityMap creation and validation."""
    print("Testing VelocityMap...")
    
    # Create 1D velocity map
    data_1d = np.linspace(1500, 3000, 100).astype(np.float32)
    vmap_1d = VelocityMap.from_1d_array(data_1d, dz=10.0)
    print(f"1D VelocityMap: {vmap_1d}")
    
    # Create 2D velocity map
    data_2d = np.ones((50, 100), dtype=np.float32) * 2000
    vmap_2d = VelocityMap.from_2d_array(data_2d, dx=5.0, dz=10.0)
    print(f"2D VelocityMap: {vmap_2d}")
    
    # Create 3D velocity map
    data_3d = np.ones((20, 30, 40), dtype=np.float32) * 2500
    vmap_3d = VelocityMap.from_3d_array(data_3d, dx=5.0, dy=7.5, dz=10.0)
    print(f"3D VelocityMap: {vmap_3d}")
    
    # Test validation
    try:
        # This should fail because 3D velocity map requires dy parameter
        VelocityMap(
            data=data_3d,
            dimensionality=Dimensionality.DIM_3D,
            dx=5.0,
            dz=10.0,
            dy=None
        )
        print("ERROR: Validation failed to catch missing dy parameter")
    except ValueError as e:
        print(f"Validation correctly caught error: {e}")
    
    print("VelocityMap tests completed successfully\n")


def test_velocity_model_1d():
    """Test VelocityModel1D creation and methods."""
    print("Testing VelocityModel1D...")
    
    # Create 1D velocity model
    data_1d = np.linspace(1500, 3000, 100).astype(np.float32)
    model_1d = VelocityModel1D.from_array(data_1d, grid_spacing=10.0)
    
    # Test properties
    print(f"1D Model: {model_1d}")
    print(f"Shape: {model_1d.shape}")
    print(f"Grid spacing: {model_1d.grid_spacing}")
    print(f"Origin: {model_1d.origin}")
    print(f"Extent: {model_1d.extent}")
    print(f"Min velocity: {model_1d.min_velocity}")
    print(f"Max velocity: {model_1d.max_velocity}")
    print(f"Mean velocity: {model_1d.mean_velocity}")
    
    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save and load with different formats
        formats = [
            ("numpy", "npz"),
            ("json", "json"),
            ("pickle", "pkl")
        ]
        for fmt, ext in formats:
            save_path = Path(tmpdir) / f"model_1d.{ext}"
            model_1d.save(save_path, format=fmt)
            loaded_model = load_velocity_model(save_path)
            print(f"Loaded 1D model ({fmt}): {loaded_model}")
            
            # Verify data is the same
            np.testing.assert_array_equal(model_1d.as_numpy(), loaded_model.as_numpy())
            assert model_1d.grid_spacing == loaded_model.grid_spacing
    
    print("VelocityModel1D tests completed successfully\n")


def test_velocity_model_2d():
    """Test VelocityModel2D creation and methods."""
    print("Testing VelocityModel2D...")
    
    # Create 2D velocity model
    data_2d = np.ones((50, 100), dtype=np.float32) * 2000
    # Add a velocity anomaly
    data_2d[20:30, 40:60] = 3000
    model_2d = VelocityModel2D.from_array(data_2d, grid_spacing=(5.0, 10.0))
    
    # Test properties
    print(f"2D Model: {model_2d}")
    print(f"Shape: {model_2d.shape}")
    print(f"Grid spacing: {model_2d.grid_spacing}")
    print(f"Origin: {model_2d.origin}")
    print(f"Extent: {model_2d.extent}")
    print(f"Min velocity: {model_2d.min_velocity}")
    print(f"Max velocity: {model_2d.max_velocity}")
    print(f"Mean velocity: {model_2d.mean_velocity}")
    
    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save and load with different formats
        formats = [
            ("numpy", "npz"),
            ("json", "json"),
            ("pickle", "pkl")
        ]
        for fmt, ext in formats:
            save_path = Path(tmpdir) / f"model_2d.{ext}"
            model_2d.save(save_path, format=fmt)
            loaded_model = load_velocity_model(save_path)
            print(f"Loaded 2D model ({fmt}): {loaded_model}")
            
            # Verify data is the same
            np.testing.assert_array_equal(model_2d.as_numpy(), loaded_model.as_numpy())
            assert model_2d.grid_spacing == loaded_model.grid_spacing
    
    print("VelocityModel2D tests completed successfully\n")


def test_velocity_model_3d():
    """Test VelocityModel3D creation and methods."""
    print("Testing VelocityModel3D...")
    
    # Create 3D velocity model
    data_3d = np.ones((20, 30, 40), dtype=np.float32) * 2500
    # Add a velocity anomaly
    data_3d[5:15, 10:20, 15:25] = 3500
    model_3d = VelocityModel3D.from_array(data_3d, grid_spacing=(5.0, 7.5, 10.0))
    
    # Test properties
    print(f"3D Model: {model_3d}")
    print(f"Shape: {model_3d.shape}")
    print(f"Grid spacing: {model_3d.grid_spacing}")
    print(f"Origin: {model_3d.origin}")
    print(f"Extent: {model_3d.extent}")
    print(f"Min velocity: {model_3d.min_velocity}")
    print(f"Max velocity: {model_3d.max_velocity}")
    print(f"Mean velocity: {model_3d.mean_velocity}")
    
    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save and load with different formats
        formats = [
            ("numpy", "npz"),
            ("json", "json"),
            ("pickle", "pkl")
        ]
        for fmt, ext in formats:
            save_path = Path(tmpdir) / f"model_3d.{ext}"
            model_3d.save(save_path, format=fmt)
            loaded_model = load_velocity_model(save_path)
            print(f"Loaded 3D model ({fmt}): {loaded_model}")
            
            # Verify data is the same
            np.testing.assert_array_equal(model_3d.as_numpy(), loaded_model.as_numpy())
            assert model_3d.grid_spacing == loaded_model.grid_spacing
    
    print("VelocityModel3D tests completed successfully\n")


def test_multiple_models():
    """Test saving and loading multiple models."""
    print("Testing multiple models serialization...")
    
    # Create models of different dimensions
    data_1d = np.linspace(1500, 3000, 100).astype(np.float32)
    model_1d = VelocityModel1D.from_array(data_1d, grid_spacing=10.0)
    
    data_2d = np.ones((50, 100), dtype=np.float32) * 2000
    model_2d = VelocityModel2D.from_array(data_2d, grid_spacing=(5.0, 10.0))
    
    data_3d = np.ones((20, 30, 40), dtype=np.float32) * 2500
    model_3d = VelocityModel3D.from_array(data_3d, grid_spacing=(5.0, 7.5, 10.0))
    
    models = [model_1d, model_2d, model_3d]
    
    # Test serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "models.pickle"
        save_velocity_models(models, save_path)
        loaded_models = load_velocity_models(save_path)
        
        print(f"Loaded {len(loaded_models)} models")
        for i, model in enumerate(loaded_models):
            print(f"Model {i}: {model}")
            
            # Verify data is the same
            np.testing.assert_array_equal(models[i].as_numpy(), loaded_models[i].as_numpy())
            assert models[i].grid_spacing == loaded_models[i].grid_spacing
    
    print("Multiple models serialization tests completed successfully\n")


if __name__ == "__main__":
    print("Testing pydantic-based velocity model implementation\n")
    
    test_velocity_map()
    test_velocity_model_1d()
    test_velocity_model_2d()
    test_velocity_model_3d()
    test_multiple_models()
    
    print("All tests completed successfully!")