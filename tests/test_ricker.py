#!/usr/bin/env python3
"""
Simple test script to verify that the ricker function can be imported from its new location.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import the ricker function from the new location
from echolab.modeling import ricker

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
    
    # Plot the wavelet
    plt.figure(figsize=(10, 6))
    plt.plot(time_vector, wavelet)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Ricker Wavelet (25 Hz)')
    plt.grid(True)
    plt.savefig('ricker_wavelet_test.png')
    plt.close()
    
    print("Test completed successfully. Plot saved to 'ricker_wavelet_test.png'")
    return True

if __name__ == "__main__":
    test_ricker_function()