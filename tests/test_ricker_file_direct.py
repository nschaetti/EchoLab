"""
Direct test of the Ricker wavelet function by importing it directly from the file.

This script bypasses the module import system entirely to avoid dependency issues.
"""

import sys
import os
import importlib.util
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Path to the wavelets.py file
wavelets_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "echolab", "modeling", "wavelets.py"
)

# Load the wavelets.py file as a module
spec = importlib.util.spec_from_file_location("wavelets", wavelets_path)
wavelets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wavelets)

# Now we can access the ricker function directly
ricker = wavelets.ricker

# Test parameters
frequency = 25.0  # Hz
time_step = 0.001  # seconds
output_path = Path("../ricker_file_direct_test.png")

# Generate the Ricker wavelet
print(f"Generating Ricker wavelet with frequency={frequency}Hz, time_step={time_step}s")
wavelet, time_vector = ricker(
    frequency=frequency,
    time_step=time_step,
    num_samples=None,
)

# Print some information about the wavelet
print(f"Generated wavelet with {len(wavelet)} samples")
print(f"Time range: {time_vector[0]:.6f}s to {time_vector[-1]:.6f}s")
print(f"Wavelet amplitude range: {wavelet.min():.6f} to {wavelet.max():.6f}")

# Create and save the plot
plt.figure(figsize=(10, 6))
plt.plot(time_vector, wavelet)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title(f"Ricker Wavelet (f={frequency} Hz, dt={time_step} s)")
plt.grid(True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to {output_path}")

print("Test completed successfully!")