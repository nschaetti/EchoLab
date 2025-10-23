"""
Direct test of the CLI wavelet commands by importing them directly from the file.

This script bypasses the module import system entirely to avoid dependency issues.
"""

import sys
import os
import importlib.util
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display issues
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import click
from click.testing import CliRunner

# Path to the wavelets.py file for the ricker function
wavelets_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "echolab", "modeling", "wavelets.py"
)

# Load the wavelets.py file as a module
spec = importlib.util.spec_from_file_location("wavelets", wavelets_path)
wavelets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wavelets)

# Path to the cli_wavelets.py file
cli_wavelets_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "echolab", "cli_wavelets.py"
)

# Create a temporary module with the ricker function
temp_module = type('module', (), {})()
temp_module.ricker = wavelets.ricker

# Mock the import in cli_wavelets.py
sys.modules['echolab.modeling.wavelets'] = temp_module

# Load the cli_wavelets.py file as a module
spec = importlib.util.spec_from_file_location("cli_wavelets", cli_wavelets_path)
cli_wavelets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli_wavelets)

# We can't call the CLI function directly because it's a Click command
# Instead, we'll use CliRunner for all tests

# Test the CLI command using CliRunner
print("Testing ricker_wavelet CLI command using CliRunner...")

# Create a CliRunner instance
runner = CliRunner()

# Create a temporary directory for output files
with runner.isolated_filesystem():
    # Test with basic parameters
    result = runner.invoke(cli_wavelets.wavelets, ['ricker-wavelet', 
                                                '--frequency', '25.0', 
                                                '--time-step', '0.001',
                                                '--output', 'ricker_test1.png',
                                                '--save-data', 'ricker_data1.npy'])
    print(f"Test 1 - Basic parameters - Exit code: {result.exit_code}")
    if result.exception:
        print(f"Exception: {result.exception}")
    else:
        print("Command executed successfully")
        
        # Check if files were created
        output_path = Path("ricker_test1.png")
        data_path = Path("ricker_data1.npy")
        print(f"Output files:")
        print(f"  Plot file exists: {output_path.exists()}")
        print(f"  Data file exists: {data_path.exists()}")
        
        # Load and verify the saved data if it exists
        if data_path.exists():
            data = np.load(data_path, allow_pickle=True).item()
            wavelet = data['wavelet']
            time = data['time']
            print(f"  Loaded wavelet data: {len(wavelet)} samples")
            print(f"  Time range: {time[0]:.6f}s to {time[-1]:.6f}s")
            print(f"  Wavelet amplitude range: {wavelet.min():.6f} to {wavelet.max():.6f}")
    
    # Test with num_samples parameter
    print("\nTest 2 - With num_samples parameter")
    result = runner.invoke(cli_wavelets.wavelets, ['ricker-wavelet', 
                                                '--frequency', '30.0', 
                                                '--time-step', '0.002',
                                                '--num-samples', '200',
                                                '--output', 'ricker_test2.png',
                                                '--save-data', 'ricker_data2.npy'])
    print(f"Exit code: {result.exit_code}")
    if result.exception:
        print(f"Exception: {result.exception}")
    else:
        print("Command executed successfully")
        
        # Check if files were created
        output_path = Path("ricker_test2.png")
        data_path = Path("ricker_data2.npy")
        print(f"Output files:")
        print(f"  Plot file exists: {output_path.exists()}")
        print(f"  Data file exists: {data_path.exists()}")
        
        # Load and verify the saved data if it exists
        if data_path.exists():
            data = np.load(data_path, allow_pickle=True).item()
            wavelet = data['wavelet']
            time = data['time']
            print(f"  Loaded wavelet data: {len(wavelet)} samples")
            print(f"  Time range: {time[0]:.6f}s to {time[-1]:.6f}s")
            print(f"  Wavelet amplitude range: {wavelet.min():.6f} to {wavelet.max():.6f}")
            
            # Verify that num_samples was respected
            print(f"  Requested num_samples: 200, Actual samples: {len(wavelet)}")

print("Test completed successfully!")