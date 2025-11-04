"""
Noise Source Generators for Acoustic Field Simulation.

This module provides classes to generate various types of noise sources
commonly used in acoustic field simulation, such as Ricker wavelets,
brown noise, and pink noise.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from scipy import signal


class NoiseSource(ABC):
    """
    Abstract base class for noise sources.
    
    This class defines the interface for noise sources used in acoustic field simulation.
    """
    
    @abstractmethod
    def generate(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a noise source signal.
        
        Args:
            **kwargs: Additional arguments for the noise generation.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: The generated signal and corresponding time vector.
        """
        pass


class RickerWavelet(NoiseSource):
    """
    Ricker wavelet (Mexican hat wavelet) noise source.
    
    A Ricker wavelet is commonly used in seismic modeling to represent
    a seismic source. It is the negative normalized second derivative of a
    Gaussian function and has a characteristic shape with a main peak and
    two smaller side lobes.
    """
    
    def __init__(self, frequency: float):
        """
        Initialize a Ricker wavelet noise source.
        
        Args:
            frequency (float): Central frequency of the wavelet in Hz.
                This determines the width of the main peak.
        """
        self.frequency = frequency
    
    def generate(
        self,
        time_step: float,
        num_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a Ricker wavelet.
        
        Args:
            time_step (float): Sampling interval in seconds.
                This is the time between consecutive samples.
            num_samples (int, optional): Total number of samples to generate.
                If specified, the wavelet will be padded with zeros to reach this length.
                If None, the function will calculate an appropriate length.
                
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - wavelet (numpy.ndarray): The Ricker wavelet samples
                - time_vector (numpy.ndarray): The corresponding time values in seconds
                
        Raises:
            ValueError: If num_samples is smaller than the calculated wavelet length
        """
        # Calculate the number of samples needed for the wavelet
        # The factor 2.2 ensures the wavelet is fully captured
        wavelet_length = (1.0 / self.frequency) / time_step * 2.2
        
        # Ensure the wavelet length is odd (for symmetry around the central peak)
        wavelet_length = 2 * (wavelet_length // 2) + 1
        
        # Calculate the center point index
        center_index = wavelet_length // 2
        
        # Create a column vector with indices [1...wavelet_length]
        k = np.arange(1, wavelet_length + 1).reshape(-1, 1)
        
        # Calculate the wavelet parameters
        alpha = (center_index - k + 1) * self.frequency * time_step * np.pi
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


class RandomRickerWavelet(NoiseSource):
    """
    Random Ricker wavelet noise source.
    
    This class generates Ricker wavelets with random frequencies within a specified range.
    """
    
    def __init__(self, min_frequency: float, max_frequency: float):
        """
        Initialize a random Ricker wavelet noise source.
        
        Args:
            min_frequency (float): Minimum central frequency of the wavelet in Hz.
            max_frequency (float): Maximum central frequency of the wavelet in Hz.
        """
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
    
    def generate(
        self,
        time_step: float,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a random Ricker wavelet.
        
        Args:
            time_step (float): Sampling interval in seconds.
            num_samples (int, optional): Total number of samples to generate.
            seed (int, optional): Random seed for reproducibility.
                
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - wavelet (numpy.ndarray): The Ricker wavelet samples
                - time_vector (numpy.ndarray): The corresponding time values in seconds
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Generate a random frequency within the specified range
        frequency = np.random.uniform(self.min_frequency, self.max_frequency)
        
        # Create a Ricker wavelet with the random frequency
        ricker = RickerWavelet(frequency)
        return ricker.generate(time_step, num_samples)


class BrownNoise(NoiseSource):
    """
    Brown noise (Brownian noise) source.
    
    Brown noise is a type of noise where the power spectral density decreases
    with frequency according to 1/f^2. It is also known as random walk noise
    or Brownian noise.
    """
    
    def generate(
        self,
        time_step: float,
        num_samples: int,
        amplitude: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate brown noise.
        
        Args:
            time_step (float): Sampling interval in seconds.
            num_samples (int): Number of samples to generate.
            amplitude (float, optional): Amplitude scaling factor. Defaults to 1.0.
            seed (int, optional): Random seed for reproducibility.
                
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - noise (numpy.ndarray): The brown noise samples
                - time_vector (numpy.ndarray): The corresponding time values in seconds
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Generate white noise
        white_noise = np.random.normal(0, 1, num_samples)
        
        # Integrate white noise to get brown noise (cumulative sum)
        brown_noise = np.cumsum(white_noise)
        
        # Normalize and scale
        brown_noise = brown_noise / np.max(np.abs(brown_noise)) * amplitude
        
        # Create time vector
        time_vector = np.arange(num_samples) * time_step
        
        return brown_noise, time_vector


class PinkNoise(NoiseSource):
    """
    Pink noise source.
    
    Pink noise is a type of noise where the power spectral density decreases
    with frequency according to 1/f. It is also known as 1/f noise or flicker noise.
    """
    
    def generate(
        self,
        time_step: float,
        num_samples: int,
        amplitude: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate pink noise.
        
        Args:
            time_step (float): Sampling interval in seconds.
            num_samples (int): Number of samples to generate.
            amplitude (float, optional): Amplitude scaling factor. Defaults to 1.0.
            seed (int, optional): Random seed for reproducibility.
                
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - noise (numpy.ndarray): The pink noise samples
                - time_vector (numpy.ndarray): The corresponding time values in seconds
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Generate white noise in frequency domain
        white_noise_fft = np.random.normal(0, 1, num_samples // 2 + 1) + 1j * np.random.normal(0, 1, num_samples // 2 + 1)
        
        # Create 1/f filter
        frequencies = np.fft.rfftfreq(num_samples, time_step)
        # Avoid division by zero at DC
        frequencies[0] = 1e-6
        
        # Apply 1/f filter
        pink_noise_fft = white_noise_fft / np.sqrt(frequencies)
        
        # Convert back to time domain
        pink_noise = np.fft.irfft(pink_noise_fft, n=num_samples)
        
        # Normalize and scale
        pink_noise = pink_noise / np.max(np.abs(pink_noise)) * amplitude
        
        # Create time vector
        time_vector = np.arange(num_samples) * time_step
        
        return pink_noise, time_vector


class CompositeNoiseSource(NoiseSource):
    """
    Composite noise source that combines multiple noise sources.
    
    This class allows combining multiple noise sources with different weights
    to create a composite noise source.
    """
    
    def __init__(self, sources: list, weights: Optional[list] = None):
        """
        Initialize a composite noise source.
        
        Args:
            sources (list): List of NoiseSource objects.
            weights (list, optional): List of weights for each source.
                If None, equal weights are used.
        """
        self.sources = sources
        
        if weights is None:
            self.weights = [1.0] * len(sources)
        else:
            if len(weights) != len(sources):
                raise ValueError("Number of weights must match number of sources")
            self.weights = weights
    
    def generate(
        self,
        time_step: float,
        num_samples: int,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a composite noise signal.
        
        Args:
            time_step (float): Sampling interval in seconds.
            num_samples (int): Number of samples to generate.
            seed (int, optional): Base random seed for reproducibility.
                Each source will get a different seed derived from this base seed.
                
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - noise (numpy.ndarray): The composite noise samples
                - time_vector (numpy.ndarray): The corresponding time values in seconds
        """
        # Initialize the composite signal
        composite_signal = np.zeros(num_samples)
        time_vector = np.arange(num_samples) * time_step
        
        # Generate and combine signals from each source
        for i, (source, weight) in enumerate(zip(self.sources, self.weights)):
            # Use a different seed for each source if a base seed is provided
            source_seed = None if seed is None else seed + i
            
            # Generate the signal from this source
            signal, _ = source.create_velocity_model(time_step, num_samples, seed=source_seed)
            
            # Ensure the signal has the right length
            if len(signal) < num_samples:
                padded_signal = np.zeros(num_samples)
                padded_signal[:len(signal)] = signal
                signal = padded_signal
            elif len(signal) > num_samples:
                signal = signal[:num_samples]
            
            # Add the weighted signal to the composite
            composite_signal += weight * signal
        
        # Normalize the composite signal
        if np.max(np.abs(composite_signal)) > 0:
            composite_signal = composite_signal / np.max(np.abs(composite_signal))
        
        return composite_signal, time_vector