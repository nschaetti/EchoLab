"""
Functions for generating synthetic velocity models.

This module provides functions for generating various types of synthetic velocity models
for seismic modeling and inversion.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import matplotlib.pyplot as plt

from .layer_generation import (
    layered_model,
    random_fault_model,
    dome_model,
    perlin_threshold_model,
)


def generate_models(
    num_models: int,
    output_dir: Union[str, Path],
    model_type: str = "layered",
    nx: int = 100,
    nz: int = 100,
    dx: float = 10.0,
    dz: float = 10.0,
    min_velocity: float = 1500.0,
    max_velocity: float = 4500.0,
    num_layers: Optional[int] = None,
    fault_probability: float = 0.5,
    dome_probability: float = 0.3,
    perlin_probability: float = 0.2,
    random_seed: Optional[int] = None,
    plot_models: bool = True,
    **kwargs: Any
) -> List[np.ndarray]:
    """
    Generate a specified number of synthetic velocity models.
    
    Args:
        num_models: Number of models to generate
        output_dir: Directory to save the generated models
        model_type: Type of model to generate ('layered', 'fault', 'dome', 'perlin', or 'mixed')
        nx: Number of grid points in x direction
        nz: Number of grid points in z direction
        dx: Grid spacing in x direction (meters)
        dz: Grid spacing in z direction (meters)
        min_velocity: Minimum velocity value (m/s)
        max_velocity: Maximum velocity value (m/s)
        num_layers: Number of layers for layered models
        fault_probability: Probability of generating a fault model when model_type is 'mixed'
        dome_probability: Probability of generating a dome model when model_type is 'mixed'
        perlin_probability: Probability of generating a perlin model when model_type is 'mixed'
        random_seed: Random seed for reproducibility
        plot_models: Whether to plot the generated models
        **kwargs: Additional keyword arguments for specific model generators
        
    Returns:
        List of generated velocity models as numpy arrays
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    # end if
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize list to store generated models
    models = []
    
    # Generate models
    for i in range(num_models):
        # For mixed model type, randomly select a model type based on probabilities
        if model_type == "mixed":
            rand_val = np.random.random()
            if rand_val < fault_probability:
                current_model_type = "fault"
            elif rand_val < fault_probability + dome_probability:
                current_model_type = "dome"
            elif rand_val < fault_probability + dome_probability + perlin_probability:
                current_model_type = "perlin"
            else:
                current_model_type = "layered"
            # end if
        else:
            current_model_type = model_type
        # end if
        
        # Generate model based on selected type
        if current_model_type == "layered":
            n_layers = num_layers if num_layers is not None else np.random.randint(3, 10)
            model = layered_model(
                nx=nx,
                nz=nz,
                num_layers=n_layers,
                min_velocity=min_velocity,
                max_velocity=max_velocity,
                **kwargs
            )
        elif current_model_type == "fault":
            model = random_fault_model(
                nx=nx,
                nz=nz,
                min_velocity=min_velocity,
                max_velocity=max_velocity,
                **kwargs
            )
        elif current_model_type == "dome":
            model = dome_model(
                nx=nx,
                nz=nz,
                min_velocity=min_velocity,
                max_velocity=max_velocity,
                **kwargs
            )
        elif current_model_type == "perlin":
            model = perlin_threshold_model(
                nx=nx,
                nz=nz,
                min_velocity=min_velocity,
                max_velocity=max_velocity,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {current_model_type}")
        # end if
        
        # Save and optionally plot the model
        save_and_plot(
            model=model,
            filename=output_dir / f"model_{i:04d}.npy",
            dx=dx,
            dz=dz,
            plot=plot_models
        )
        
        models.append(model)
    # end for
    
    return models
# end def generate_models


def save_and_plot(
    model: np.ndarray,
    filename: Union[str, Path],
    dx: float = 10.0,
    dz: float = 10.0,
    plot: bool = True
) -> None:
    """
    Save a velocity model to a file and optionally plot it.
    
    Args:
        model: Velocity model as a 2D numpy array
        filename: Path to save the model
        dx: Grid spacing in x direction (meters)
        dz: Grid spacing in z direction (meters)
        plot: Whether to plot the model
    """
    # Save model
    np.save(filename, model)
    
    # Plot model if requested
    if plot:
        plt.figure(figsize=(10, 6))
        extent = [0, model.shape[1] * dx, model.shape[0] * dz, 0]
        plt.imshow(model, cmap='viridis', extent=extent)
        plt.colorbar(label='Velocity (m/s)')
        plt.xlabel('Distance (m)')
        plt.ylabel('Depth (m)')
        plt.title(f'Velocity Model: {Path(filename).stem}')
        
        # Save plot with same name but different extension
        plot_filename = str(filename).replace('.npy', '.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
    # end if
# end def save_and_plot