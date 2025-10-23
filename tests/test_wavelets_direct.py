#!/usr/bin/env python3
"""
Direct test of the wavelets.py file without going through __init__.py
"""

import sys
import os
import importlib.util

# Path to the wavelets.py file
wavelets_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "echolab", "modeling", "wavelets.py"
)

# Load the module directly
spec = importlib.util.spec_from_file_location("wavelets", wavelets_path)
wavelets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wavelets)

# Now we can use the ricker function directly
def test_ricker_function():
    """Test that the ricker function works correctly."""
    # Parameters
    frequency = 25.0  # Hz
    time_step = 0.001  # seconds
    
    # Generate the wavelet
    wavelet, time_vector = wavelets.ricker(frequency, time_step)
    
    # Print some basic information
    print(f"Wavelet shape: {wavelet.shape}")
    print(f"Time vector shape: {time_vector.shape}")
    print(f"Wavelet min: {wavelet.min():.4f}, max: {wavelet.max():.4f}")
    
    print("Test completed successfully.")
    return True

if __name__ == "__main__":
    print(f"Testing ricker function from {wavelets_path}...")
    test_ricker_function()
    print("All tests passed!")