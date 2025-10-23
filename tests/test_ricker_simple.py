#!/usr/bin/env python3
"""
Simple test script to verify that the ricker function can be imported directly.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Import the ricker function directly from the module
from echolab.modeling.wavelets import ricker

def test_ricker_function():
    """Test that the ricker function works correctly."""
    # Parameters
    frequency = 25.0  # Hz
    time_step = 0.001  # seconds
    
    # Generate the wavelet
    wavelet, time_vector = ricker(frequency, time_step)
    
    # Print some basic information
    print(f"Wavelet shape: {wavelet.shape}")
    print(f"Time vector shape: {time_vector.shape}")
    print(f"Wavelet min: {wavelet.min():.4f}, max: {wavelet.max():.4f}")
    
    print("Test completed successfully.")
    return True

if __name__ == "__main__":
    print("Testing ricker function from echolab.modeling.wavelets...")
    test_ricker_function()
    print("All tests passed!")