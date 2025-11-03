#!/usr/bin/env python
"""
Test script for the OpenFWI command with progress bar and on-the-fly visualization.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the repository root to the Python path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from echolab.cli import main

def generate_test_model():
    """Generate a simple test velocity model."""
    # Create output directory if it doesn't exist
    os.makedirs("../test_output", exist_ok=True)
    
    # Generate a simple layered velocity model
    nx, nz = 70, 70
    n_models = 1
    model = np.ones((n_models, nz, nx), dtype=np.float32) * 2000  # Base velocity
    
    # Add layers
    model[:, :20, :] = 1500  # Top layer
    model[:, 20:40, :] = 2500  # Middle layer
    model[:, 40:, :] = 3500  # Bottom layer
    
    # Save the model
    model_path = Path("../test_output/test_model.npy")
    np.save(model_path, model)
    
    return model_path

def test_openfwi_with_progress_bar(model_path, config_path):
    """Test the OpenFWI command with progress bar and without animation."""
    # Define test arguments
    test_args = [
        "openfwi",
        "--models", str(model_path),
        "--model-i", "0",
        "--config", str(config_path),
        "--output", "test_output",
        "--no-animate"
    ]
    
    # Run the command
    print("Testing OpenFWI with progress bar and without animation...")
    main(test_args)
    
    print("\n" + "="*50 + "\n")

def test_openfwi_with_animation(model_path, config_path):
    """Test the OpenFWI command with progress bar and with animation."""
    # Define test arguments
    test_args = [
        "openfwi",
        "--models", str(model_path),
        "--model-i", "0",
        "--config", str(config_path),
        "--output", "test_output",
        "--animate"
    ]
    
    # Run the command
    print("Testing OpenFWI with progress bar and with animation...")
    main(test_args)
    
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("../test_output", exist_ok=True)
    
    # Generate a test model
    print("Generating test velocity model...")
    model_path = generate_test_model()
    print(f"Test model saved to: {model_path}")
    
    # Define the config path
    config_path = Path("../test_output/test_config.yaml")
    print(f"Using config file: {config_path}")
    
    # Run tests
    test_openfwi_with_progress_bar(model_path, config_path)
    test_openfwi_with_animation(model_path, config_path)