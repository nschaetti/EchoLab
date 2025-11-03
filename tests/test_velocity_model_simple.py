#!/usr/bin/env python
"""
Simple test script for the VelocityModel classes.

This script directly imports and tests the VelocityModel classes without
depending on the entire echolab package.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Direct import of the velocity_model module
from echolab.modeling.velocity_model import (
    VelocityModel,
    VelocityModel1D,
    VelocityModel2D,
    VelocityModel3D,
    load_velocity_model
)

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
    
    try:
        torch_data = model.as_torch()
        print(f"Torch data shape: {torch_data.shape}")
    except ImportError:
        print("PyTorch not available")
    
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