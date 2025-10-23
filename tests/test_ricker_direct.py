"""
Direct test of the Ricker wavelet function.

This script directly imports and tests the ricker function from wavelets.py
without going through the full module import chain to avoid dependency issues.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add the directory containing the wavelets.py file to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the ricker function directly from the file
from echolab.modeling.wavelets import ricker

# Test parameters
frequency = 25.0  # Hz
time_step = 0.001  # seconds
output_path = Path("ricker_direct_test.png")

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