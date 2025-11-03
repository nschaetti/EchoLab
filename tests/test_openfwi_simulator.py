#!/usr/bin/env python3
"""
Test script for the OpenFWI simulator.

This script tests the OpenFWI simulator with a Ricker wave source.
"""

import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import echolab
sys.path.insert(0, str(Path(__file__).parent.parent))

from echolab.simulators.openfwi import run_openfwi_simulation, plot_openfwi_results, animate_openfwi_wavefields


def create_test_velocity_model(nx=100, nz=100, v_min=1500, v_max=3500):
    """
    Create a simple test velocity model.
    
    Args:
        nx (int): Number of grid points in x direction.
        nz (int): Number of grid points in z direction.
        v_min (float): Minimum velocity.
        v_max (float): Maximum velocity.
        
    Returns:
        np.ndarray: Velocity model with shape (1, nz, nx).
    """
    # Create a simple layered model
    velocity_model = np.ones((nz, nx)) * v_min
    
    # Add a high-velocity layer
    velocity_model[nz//2:, :] = v_max
    
    # Add a circular anomaly
    center_x, center_z = nx//2, nz//3
    radius = min(nx, nz) // 8
    
    for i in range(nz):
        for j in range(nx):
            if (i - center_z)**2 + (j - center_x)**2 < radius**2:
                velocity_model[i, j] = (v_min + v_max) / 2
    
    # Reshape to (1, nz, nx) to match the expected format
    return np.expand_dims(velocity_model, axis=0)


def create_test_config():
    """
    Create a simple test configuration.
    
    Returns:
        Path: Path to the configuration file.
    """
    config_dir = Path(__file__).parent / "test_configs"
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / "openfwi_test_config.yaml"
    
    config_content = """
dx: 10.0
dz: 10.0
sx: 500.0
sz: 200.0
nt: 500
dt: 0.001
freq: 25.0
nbc: 20
gx_start: 0
gx_end: 99
gx_step: 1
gz_value: 0.0
snapshot_stride: 5
capture_wavefield: true
"""
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    return config_path


def test_openfwi_simulator():
    """
    Test the OpenFWI simulator.
    """
    print("Creating test velocity model...")
    velocity_model = create_test_velocity_model()
    
    print("Creating test configuration...")
    config_path = create_test_config()
    
    # Save the velocity model to a temporary file
    model_path = Path(__file__).parent / "test_configs" / "openfwi_test_model.npy"
    np.save(model_path, velocity_model)
    
    print("Running OpenFWI simulation...")
    results = run_openfwi_simulation(
        models_path=model_path,
        model_index=0,
        config_path=config_path
    )
    
    # Create output directory
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    print("Plotting results...")
    plot_openfwi_results(results, output_dir, show=False)
    
    print("Animating wavefields...")
    animate_openfwi_wavefields(results, output_dir, fps=10)
    
    print("Test completed successfully!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    test_openfwi_simulator()