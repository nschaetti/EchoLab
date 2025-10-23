"""
Wavelet Generators for Seismic Modeling.

This module provides functions to generate various wavelets commonly used
in seismic modeling and processing, such as the Ricker wavelet (Mexican hat wavelet).
"""

from typing import Optional
import numpy as np
from rich.console import Console


console = Console()


def ricker(
        frequency: float,
        time_step: float,
        num_samples: Optional[int] = None,
        log: bool = False,
):
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
        log (bool, optional): If True, the wavelet will be logged.

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
    # Print
    if log:
        console.print(f"[green]Generating Ricker wavelet[/]")
        console.print(f"[yellow]Frequency: [/] {frequency}")
        console.print(f"[yellow]Time step: [/] {time_step}")
        console.print(f"[yellow]Number of samples: [/] {num_samples}")
    # end if

    # Calculate the number of samples needed for the wavelet
    # The factor 2.2 ensures the wavelet is fully captured
    # wavelet_length = int(2.2 / frequency / time_step)
    wavelet_length = (1.0 / frequency) / time_step * 2.2

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
        # end if

        # Create a zero-padded array of the requested length
        wavelet = np.zeros((num_samples, 1))

        # Copy the raw wavelet into the beginning of the padded array
        wavelet[:len(wavelet_raw)] = wavelet_raw
    else:
        wavelet = wavelet_raw
    # end if

    # Create a time vector corresponding to the wavelet samples
    wavelet_length = len(wavelet)
    time_vector = np.arange(wavelet_length) * time_step

    # Return the flattened wavelet and the time vector
    return wavelet.flatten(), time_vector
# end def ricker

