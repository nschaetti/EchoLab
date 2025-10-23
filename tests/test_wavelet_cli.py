"""
Test script for the wavelet CLI commands.

This script directly imports and calls the ricker_wavelet function from cli_wavelets.py
to verify it works correctly without running the full CLI.
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from echolab.modeling.wavelets import ricker
from echolab.cli_wavelets import ricker_wavelet

# Test parameters
frequency = 25.0  # Hz
time_step = 0.001  # seconds
output_path = Path("ricker_test.png")
data_path = Path("ricker_data.npy")

# Call the CLI function directly
print(f"Generating Ricker wavelet with frequency={frequency}Hz, time_step={time_step}s")
ricker_wavelet(
    frequency=frequency,
    time_step=time_step,
    num_samples=None,
    output=output_path,
    save_data=data_path
)

# Verify the output files were created
print(f"Checking if output files were created:")
print(f"  Plot file: {output_path.exists()}")
print(f"  Data file: {data_path.exists()}")

# Load and verify the saved data
if data_path.exists():
    data = np.load(data_path, allow_pickle=True).item()
    wavelet = data['wavelet']
    time = data['time']
    print(f"Loaded wavelet data: {len(wavelet)} samples")
    print(f"Time range: {time[0]:.6f}s to {time[-1]:.6f}s")
    print(f"Wavelet amplitude range: {wavelet.min():.6f} to {wavelet.max():.6f}")

print("Test completed successfully!")