#!/usr/bin/env python3
"""
Test script for the OpenANFWI simulator.

This script tests the OpenANFWI simulator with different noise sources:
- Ricker wavelet
- Random Ricker wavelet
- Brown noise
- Pink noise
- Composite noise (combination of the above)
"""

import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import echolab
sys.path.insert(0, str(Path(__file__).parent.parent))

from echolab.simulators.openanfwi import (
    load_config,
    prepare_wave_simulation,
    run_simulation,
    plot_velocity
)


def create_test_velocity_model(nx=100, nz=100, v_min=1500, v_max=3500):
    """
    Create a simple test velocity model.
    
    Args:
        nx (int): Number of grid points in x direction.
        nz (int): Number of grid points in z direction.
        v_min (float): Minimum velocity.
        v_max (float): Maximum velocity.
        
    Returns:
        np.ndarray: Velocity model with shape (nz, nx).
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
    
    return velocity_model


def create_test_config(source_type="ricker"):
    """
    Create a simple test configuration.
    
    Args:
        source_type (str): Type of noise source to use.
            Options: "ricker", "random_ricker", "brown", "pink", "composite".
            
    Returns:
        Path: Path to the configuration file.
    """
    config_dir = Path(__file__).parent / "test_configs"
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / f"openanfwi_test_config_{source_type}.yaml"
    
    # Base configuration
    config_content = f"""
nx: 100
nz: 100
dx: 10.0
dz: 10.0
v_const: 2000.0

source:
  ix: 50
  iz: 30
  frequency: 25.0
  amplitude: 1.0
  delay: 0.0
  type: {source_type}
  min_frequency: 10.0
  max_frequency: 50.0
  weights: [0.3, 0.3, 0.4]

simulation:
  nt: 500
  dt: 0.001
  output_interval: 10
  damping: 0.0
  save_snapshots: true
  snapshot_dir: "{str(Path(__file__).parent / 'test_outputs' / f'openanfwi_{source_type}')}"
  seed: 42

plot:
  title: "OpenANFWI Test - {source_type.capitalize()} Source"
  colormap: "seismic"
  show_colorbar: true
  grid_visible: true
"""
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    return config_path


def test_openanfwi_simulator(source_type="ricker", visualize=True):
    """
    Test the OpenANFWI simulator with a specific noise source.
    
    Args:
        source_type (str): Type of noise source to use.
            Options: "ricker", "random_ricker", "brown", "pink", "composite".
        visualize (bool): Whether to visualize the simulation.
    """
    print(f"Testing OpenANFWI simulator with {source_type} source...")
    
    # Create test configuration
    config_path = create_test_config(source_type)
    
    # Load configuration and prepare simulation
    config, velocity, x_grid, z_grid = prepare_wave_simulation(config_path)
    
    # Plot velocity model
    if visualize:
        plot_velocity(velocity, config)
    
    # Run simulation
    final_pressure = run_simulation(config, velocity, visualize=visualize)
    
    print(f"Test completed for {source_type} source!")
    print(f"Results saved to {config['simulation']['snapshot_dir']}")
    
    return final_pressure


def test_all_noise_sources(visualize=True):
    """
    Test the OpenANFWI simulator with all noise sources.
    
    Args:
        visualize (bool): Whether to visualize the simulations.
    """
    # Create output directory
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # Test each noise source
    noise_sources = ["ricker", "random_ricker", "brown", "pink", "composite"]
    results = {}
    
    for source_type in noise_sources:
        results[source_type] = test_openanfwi_simulator(source_type, visualize)
    
    # Compare results
    if visualize:
        plt.figure(figsize=(15, 10))
        
        for i, (source_type, pressure) in enumerate(results.items()):
            plt.subplot(2, 3, i+1)
            plt.imshow(pressure, cmap="seismic")
            plt.title(f"{source_type.capitalize()} Source")
            plt.colorbar(label="Pressure")
        
        plt.tight_layout()
        plt.savefig(output_dir / "openanfwi_comparison.png", dpi=150)
        plt.show()
    
    print("All tests completed successfully!")


if __name__ == "__main__":
    test_all_noise_sources()