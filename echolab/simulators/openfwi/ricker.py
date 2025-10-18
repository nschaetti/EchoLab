
"""
Ricker Wavelet Generator for Seismic Modeling.

This module provides a function to generate a Ricker wavelet, which is commonly
used as a source signal in seismic modeling and processing. The Ricker wavelet
(also known as the Mexican hat wavelet) is the negative normalized second
derivative of a Gaussian function.
"""

import numpy as np


def ricker(frequency, time_step, num_samples=None):
    """
    Generate a Ricker wavelet (Mexican hat wavelet).

    A Ricker wavelet is commonly used in seismic modeling to represent
    a seismic source. It is the negative normalized second derivative of a
    Gaussian function and has a characteristic shape with a main peak and
    two smaller side lobes.

    Args:
        frequency (float): Central frequency of the wavelet in Hz.
            This determines the width of the main peak.
        time_step (float): Sampling interval in seconds.
            This is the time between consecutive samples.
        num_samples (int, optional): Total number of samples to generate.
            If specified, the wavelet will be padded with zeros to reach this length.
            If None, the function will calculate an appropriate length.

    Returns:
        tuple: A tuple containing:
            - wavelet (numpy.ndarray): The Ricker wavelet samples
            - time_vector (numpy.ndarray): The corresponding time values in seconds

    Raises:
        ValueError: If num_samples is smaller than the calculated wavelet length

    Example:
        >>> wavelet, time = ricker(25.0, 0.001)  # 25 Hz wavelet, 1ms sampling
        >>> plt.plot(time, wavelet)
        >>> plt.xlabel('Time (s)')
        >>> plt.ylabel('Amplitude')
        >>> plt.title('Ricker Wavelet (25 Hz)')
    """
    # Calculate the number of samples needed for the wavelet
    # The factor 2.2 ensures the wavelet is fully captured
    wavelet_length = int(2.2 / frequency / time_step)

    # Ensure the wavelet length is odd (for symmetry around the central peak)
    wavelet_length = 2 * (wavelet_length // 2) + 1

    # Calculate the center point index
    center_index = wavelet_length // 2

    # Create a column vector with indices [1...wavelet_length]
    k = np.arange(1, wavelet_length + 1).reshape(-1, 1)

    # Calculate the wavelet parameters
    alpha = (center_index - k + 1) * frequency * time_step * np.pi
    beta = alpha ** 2

    # Generate the Ricker wavelet using the formula: (1-2β)e^(-β)
    # This is the analytical expression for the negative normalized
    # second derivative of a Gaussian function
    wavelet_raw = (1.0 - 2.0 * beta) * np.exp(-beta)

    # Handle the case when a specific number of samples is requested
    if num_samples is not None:
        if num_samples < len(wavelet_raw):
            raise ValueError("num_samples is smaller than the required wavelet length!")

        # Create a zero-padded array of the requested length
        wavelet = np.zeros((num_samples, 1))

        # Copy the raw wavelet into the beginning of the padded array
        wavelet[:len(wavelet_raw)] = wavelet_raw
    else:
        wavelet = wavelet_raw

    # Create a time vector corresponding to the wavelet samples
    wavelet_length = len(wavelet)
    time_vector = np.arange(wavelet_length) * time_step

    # Return the flattened wavelet and the time vector
    return wavelet.flatten(), time_vector
