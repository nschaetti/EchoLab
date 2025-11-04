"""
High-level helpers to generate collections of synthetic velocity models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from skimage import filters
from rich.console import Console

from .data_augmentation import (
    AddCrossLine,
    AddInclusion,
    AddPerlinNoise,
    RandomDisplacementField,
    RandomPiecewiseAffine,
    RandomRotation,
    RandomStretch,
    RandomSwirl,
)
from .layer_generation import (
    dome_model,
    layered_model,
    perlin_threshold_model,
    random_fault_model,
)
from ..validation import is_valid_velmap
from ..velocity import (
    VelocityModel1D,
    VelocityModel2D,
    VelocityModel3D,
    create_velocity_model
)



console = Console()


def synthesize_velocity_models(
        n_models: int,
        rng: np.random.Generator,
        config: Dict[str, Any],
        verbose: bool = False,
) -> List[Union[VelocityModel1D, VelocityModel2D, VelocityModel3D]]:
    """
    Generate a batch of velocity models using configurable primitives.
    
    Args:
        n_models (int): Number of models to generate.
        rng: Random number generator.
        config: Model generation configuration.
        verbose: Whether to print verbose output.
        
    Returns:
        List of VelocityMap objects.
        
    Raises:
        NotImplementedError: If 1D or 3D dimensionality is requested (currently only 2D is implemented).
    """
    # Shortcuts
    model_types = config['model_types']['available_types']
    model_probs = config['model_types']['selection_probabilities']
    model_params = config['model_parameters']
    transforms = config['transformations']['available_transforms']
    transform_probs = config['transformations']['application_probabilities']
    transform_params = config['transformation_parameters']

    # Sigma blur param
    sigma_blur = config['smoothing_sigma']

    # Grid configuration
    grid_conf = config['grid']
    dx = grid_conf['spacing_x']
    dz = grid_conf['spacing_z']
    nx = grid_conf['points_x']
    nz = grid_conf['points_z']

    # Validation
    velocity_val = config['validation']['velocity']
    quality_val = config['validation']['quality']

    # Classes for each velmap of vel map
    model_classes: Dict[str, Any] = {
        "layered": layered_model,
        "fault": random_fault_model,
        "dome": dome_model,
        "perlin": perlin_threshold_model,
    }

    # Transformation classes
    transform_classes: Dict[str, Any] = {
        "rotate": RandomRotation,
        "stretch": RandomStretch,
        "inclusion": AddInclusion,
        "perlin_noise": AddPerlinNoise,
        "add_crossline": AddCrossLine,
        "swirl": RandomSwirl,
        "piecewise_affine": RandomPiecewiseAffine,
        "displacement_field": RandomDisplacementField,
    }

    # Create transformers
    transforms_list: Dict[str, Any] = {
        name: transform_classes[name](**transform_params.get(name, {}), rng=rng)
        for name in transforms
    }

    # Model weights
    model_weights: List[float] = [model_probs[m] for m in model_types]
    transform_weights: List[float] = [transform_probs.get(t, 0.0) for t in transforms]

    # Keep and count generated models
    velocity_models = []
    n_generated: int = 0

    # Generate each velmap
    while n_generated < n_models:
        # Choose a velmap type randomly
        model_name = rng.choice(model_types, p=model_weights)

        # Generate the velocity map
        velmap = model_classes[model_name](
            nz=nz,
            nx=nx,
            rng=rng,
            **model_params.get(model_name),
        )

        # Apply transform randomly
        for name, prob in zip(transforms, transform_weights):
            if rng.random() < prob:
                velmap = transforms_list[name](velmap)
            # end if
        # end for

        # Check if the velmap is valid
        if not is_valid_velmap(
            velmap=velmap,
            min_v=velocity_val['min_velocity'],
            max_v=velocity_val['max_velocity'],
            zero_thresh=quality_val['zero_threshold'],
            unique_thresh=quality_val['unique_threshold'],
            entropy_thresh=quality_val['entropy_threshold'],
            verbose=verbose,
        ):
            if verbose:
                console.print(f"[bold red]WARNING[/]: [x] velocity map {n_generated} invalid, skipped.")
            # end if
            continue
        # end if

        # Blur velocity map
        velmap = filters.gaussian(
            image=velmap,
            sigma=sigma_blur,
            preserve_range=True
        )

        # Create
        velocity_model = create_velocity_model(
            data=velmap,
            grid_spacing=(dx, dz),
            metadata={
                "index": n_generated,
                "generated_with": "generate_models",
                "model_type": model_name,
                "model_blur_sigma": sigma_blur,
                "generation_date": str(np.datetime64('now'))
            },
        )
        
        velocity_models.append(velocity_model)
        n_generated += 1
    # end while

    return velocity_models
# end generate_models


def save_and_plot(
    model: np.ndarray,
    filename: Union[str, Path],
    dx: float = 10.0,
    dz: float = 10.0,
    plot: bool = True
) -> None:
    """
    Save a velocity model to a NumPy file and optionally export a preview plot.
    
    Args:
        model: Velocity model as a 2D numpy array.
        filename: Path (with .npy extension) where the model will be stored.
        dx: Grid spacing in the x direction (meters).
        dz: Grid spacing in the z direction (meters).
        plot: Whether to write a PNG preview alongside the array.
    """
    from matplotlib import pyplot as plt  # Local import to avoid hard dependency when unused

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.save(filename, model)

    if not plot:
        return

    plt.figure(figsize=(10, 6))
    extent = [0, model.shape[1] * dx, model.shape[0] * dz, 0]
    plt.imshow(model, cmap="viridis", extent=extent)
    plt.colorbar(label="Velocity (m/s)")
    plt.xlabel("Distance (m)")
    plt.ylabel("Depth (m)")
    plt.title(f"Velocity Model: {filename.stem}")
    plot_filename = filename.with_suffix(".png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
# end def save_and_plot
