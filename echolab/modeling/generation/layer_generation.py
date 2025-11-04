"""
Functions for generating synthetic layered velocity models.

This module provides functions for generating various types of synthetic velocity models
including layered models, fault models, dome models, and perlin noise-based models.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from noise import pnoise2


def layered_model(
    nx: int = 100,
    nz: int = 100,
    num_layers: int = 5,
    min_velocity: float = 1500.0,
    max_velocity: float = 4500.0,
    smoothing: float = 0.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a layered velocity model with horizontal layers.
    
    Args:
        nx: Number of grid points in x direction
        nz: Number of grid points in z direction
        num_layers: Number of layers in the model
        min_velocity: Minimum velocity value (m/s)
        max_velocity: Maximum velocity value (m/s)
        smoothing: Amount of smoothing to apply (0.0 = no smoothing)
        random_seed: Random seed for reproducibility
        
    Returns:
        2D numpy array representing the velocity model
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    # end if
    
    # Initialize velocity model
    model = np.zeros((nz, nx))
    
    # Generate layer boundaries
    layer_boundaries = np.sort(np.random.choice(range(1, nz), num_layers - 1, replace=False))
    layer_boundaries = np.concatenate(([0], layer_boundaries, [nz]))
    
    # Generate velocities for each layer (increasing with depth)
    velocities = np.linspace(min_velocity, max_velocity, num_layers)
    np.random.shuffle(velocities)  # Shuffle to avoid strictly increasing velocity with depth
    
    # Fill the model with layer velocities
    for i in range(num_layers):
        start_z = layer_boundaries[i]
        end_z = layer_boundaries[i + 1]
        model[start_z:end_z, :] = velocities[i]
    # end for
    
    # Apply smoothing if requested
    if smoothing > 0:
        from scipy.ndimage import gaussian_filter
        model = gaussian_filter(model, sigma=smoothing)
    # end if
    
    return model
# end def layered_model


def random_fault_model(
    nx: int = 100,
    nz: int = 100,
    num_layers: int = 5,
    num_faults: int = 2,
    min_velocity: float = 1500.0,
    max_velocity: float = 4500.0,
    max_fault_throw: Optional[int] = None,
    smoothing: float = 0.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a layered velocity model with random faults.
    
    Args:
        nx: Number of grid points in x direction
        nz: Number of grid points in z direction
        num_layers: Number of layers in the model
        num_faults: Number of faults to include
        min_velocity: Minimum velocity value (m/s)
        max_velocity: Maximum velocity value (m/s)
        max_fault_throw: Maximum displacement of layers across faults
        smoothing: Amount of smoothing to apply (0.0 = no smoothing)
        random_seed: Random seed for reproducibility
        
    Returns:
        2D numpy array representing the velocity model
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    # end if
    
    # Set default max_fault_throw if not provided
    if max_fault_throw is None:
        max_fault_throw = nz // 5
    # end if
    
    # Generate a layered model first
    model = layered_model(
        nx=nx,
        nz=nz,
        num_layers=num_layers,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
        smoothing=0.0  # No smoothing yet
    )
    
    # Generate fault locations
    fault_locations = np.sort(np.random.choice(range(1, nx - 1), num_faults, replace=False))
    
    # Apply faults
    for fault_loc in fault_locations:
        # Random fault throw (displacement)
        throw = np.random.randint(1, max_fault_throw + 1)
        
        # Randomly decide if throw is up or down
        if np.random.random() < 0.5:
            throw = -throw
        # end if
        
        # Apply the fault - shift the right side of the model
        if throw > 0:
            model[throw:, fault_loc:] = model[:-throw, fault_loc:]
            # Fill the gap with the last layer
            for x in range(fault_loc, nx):
                model[:throw, x] = model[throw, x]
            # end for
        else:
            throw = abs(throw)
            model[:-throw, fault_loc:] = model[throw:, fault_loc:]
            # Fill the gap with the last layer
            for x in range(fault_loc, nx):
                model[-throw:, x] = model[-throw-1, x]
            # end for
        # end if
    # end for
    
    # Apply smoothing if requested
    if smoothing > 0:
        from scipy.ndimage import gaussian_filter
        model = gaussian_filter(model, sigma=smoothing)
    # end if
    
    return model
# end def random_fault_model


def dome_model(
    nx: int = 100,
    nz: int = 100,
    num_layers: int = 5,
    dome_height: Optional[int] = None,
    dome_width: Optional[int] = None,
    dome_center: Optional[int] = None,
    min_velocity: float = 1500.0,
    max_velocity: float = 4500.0,
    smoothing: float = 0.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a velocity model with a dome structure.
    
    Args:
        nx: Number of grid points in x direction
        nz: Number of grid points in z direction
        num_layers: Number of layers in the model
        dome_height: Height of the dome in grid points
        dome_width: Width of the dome in grid points
        dome_center: X-coordinate of the dome center
        min_velocity: Minimum velocity value (m/s)
        max_velocity: Maximum velocity value (m/s)
        smoothing: Amount of smoothing to apply (0.0 = no smoothing)
        random_seed: Random seed for reproducibility
        
    Returns:
        2D numpy array representing the velocity model
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    # end if
    
    # Set default parameters if not provided
    if dome_height is None:
        dome_height = nz // 3
    # end if
    
    if dome_width is None:
        dome_width = nx // 2
    # end if
    
    if dome_center is None:
        dome_center = nx // 2
    # end if
    
    # Generate a layered model first
    model = layered_model(
        nx=nx,
        nz=nz,
        num_layers=num_layers,
        min_velocity=min_velocity,
        max_velocity=max_velocity,
        smoothing=0.0  # No smoothing yet
    )
    
    # Create a dome shape
    x = np.arange(nx)
    dome = dome_height * np.exp(-((x - dome_center) ** 2) / (2 * (dome_width / 4) ** 2))
    dome = dome.astype(int)
    
    # Apply the dome deformation to the model
    new_model = np.zeros_like(model)
    
    for i in range(nx):
        shift = dome[i]
        if shift > 0:
            # Shift layers upward
            new_model[:-shift, i] = model[shift:, i]
            # Fill the bottom with the last layer
            new_model[-shift:, i] = model[-1, i]
        # end if
    # end for
    
    # Apply smoothing if requested
    if smoothing > 0:
        from scipy.ndimage import gaussian_filter
        new_model = gaussian_filter(new_model, sigma=smoothing)
    # end if
    
    return new_model
# end def dome_model


def generate_perlin_noise_2d(
    shape: Tuple[int, int],
    scale: float = 10.0,
    octaves: int = 6,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a 2D Perlin noise array.
    
    Args:
        shape: Shape of the output array (height, width)
        scale: Scale of the noise (higher = more zoomed out)
        octaves: Number of octaves (detail levels)
        persistence: How much each octave contributes to the overall shape
        lacunarity: How much detail is added at each octave
        seed: Random seed
        
    Returns:
        2D numpy array of Perlin noise values between -1 and 1
    """
    if not NOISE_AVAILABLE:
        raise ImportError("The 'noise' package is required for Perlin noise generation. "
                         "Install it with 'pip install noise'.")
    # end if
    
    height, width = shape
    
    # Initialize the noise array
    noise = np.zeros(shape)
    
    # Set seed if provided
    if seed is not None:
        import random
        random.seed(seed)
        seed_value = random.randint(0, 100)
    else:
        seed_value = 0
    # end if
    
    # Generate the noise
    for i in range(height):
        for j in range(width):
            noise[i][j] = pnoise2(
                i / scale,
                j / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=seed_value
            )
        # end for
    # end for
    
    return noise
# end def generate_perlin_noise_2d


def perlin_threshold_model(
    nx: int = 100,
    nz: int = 100,
    min_velocity: float = 1500.0,
    max_velocity: float = 4500.0,
    num_thresholds: int = 4,
    scale: float = 10.0,
    octaves: int = 6,
    smoothing: float = 0.0,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a velocity model using Perlin noise with thresholds.
    
    Args:
        nx: Number of grid points in x direction
        nz: Number of grid points in z direction
        min_velocity: Minimum velocity value (m/s)
        max_velocity: Maximum velocity value (m/s)
        num_thresholds: Number of velocity regions
        scale: Scale of the Perlin noise
        octaves: Number of octaves for Perlin noise
        smoothing: Amount of smoothing to apply (0.0 = no smoothing)
        random_seed: Random seed for reproducibility
        
    Returns:
        2D numpy array representing the velocity model
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    # end if
    
    # Generate Perlin noise
    noise = generate_perlin_noise_2d(
        shape=(nz, nx),
        scale=scale,
        octaves=octaves,
        seed=random_seed
    )
    
    # Normalize noise to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    # Create thresholds
    thresholds = np.linspace(0, 1, num_thresholds + 1)
    
    # Create velocity values for each region
    velocities = np.linspace(min_velocity, max_velocity, num_thresholds)
    
    # Initialize model
    model = np.zeros((nz, nx))
    
    # Assign velocities based on thresholds
    for i in range(num_thresholds):
        mask = (noise >= thresholds[i]) & (noise < thresholds[i + 1])
        model[mask] = velocities[i]
    # end for
    
    # Apply smoothing if requested
    if smoothing > 0:
        from scipy.ndimage import gaussian_filter
        model = gaussian_filter(model, sigma=smoothing)
    # end if
    
    return model
# end def perlin_threshold_model