"""
High-level helpers to generate collections of synthetic velocity models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from skimage import filters

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
from .validation import is_valid_model
from .velocity_map import Dimensionality, VelocityMap, save_velocity_maps, load_velocity_maps
from .velocity_model import VelocityModel1D, VelocityModel2D, VelocityModel3D, save_velocity_models


def generate_models(
    rng: np.random.Generator,
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    model_types: List[str],
    model_probs: Dict[str, float],
    model_params: Dict[str, Dict[str, Any]],
    model_blur_sigma: float,
    transform_types: List[str],
    transform_probs: Dict[str, float],
    transform_params: Dict[str, Dict[str, Any]],
    n_models: int,
    min_v: float,
    max_v: float,
    unique_thresh: float,
    entropy_thresh: float,
    zero_thresh: float,
    dimensionality: Dimensionality = Dimensionality.DIM_2D,
    ny: Optional[int] = None,
    dy: Optional[float] = None,
    verbose: bool = False,
) -> List[Union[VelocityModel1D, VelocityModel2D, VelocityModel3D]]:
    """
    Generate a batch of velocity models using configurable primitives.
    
    Args:
        rng: Random number generator.
        nx: Number of grid points in x direction.
        nz: Number of grid points in z direction.
        dx: Grid spacing in x direction.
        dz: Grid spacing in z direction.
        model_types: List of model types to generate.
        model_probs: Probability of selecting each model type.
        model_params: Parameters for each model type.
        model_blur_sigma: Sigma for Gaussian blur applied to models.
        transform_types: List of transforms to apply.
        transform_probs: Probability of applying each transform.
        transform_params: Parameters for each transform.
        n_models: Number of models to generate.
        min_v: Minimum valid velocity value.
        max_v: Maximum valid velocity value.
        unique_thresh: Threshold for uniqueness validation.
        entropy_thresh: Threshold for entropy validation.
        zero_thresh: Threshold for zero validation.
        dimensionality: Dimensionality of the models to generate (1D, 2D, or 3D).
        ny: Number of grid points in y direction (for 3D models).
        dy: Grid spacing in y direction (for 3D models).
        verbose: Whether to print verbose output.
        
    Returns:
        List of VelocityMap objects.
        
    Raises:
        NotImplementedError: If 1D or 3D dimensionality is requested (currently only 2D is implemented).
    """
    
    # Check dimensionality
    if dimensionality == Dimensionality.DIM_1D:
        raise NotImplementedError("1D velocity model generation is not yet implemented")
    elif dimensionality == Dimensionality.DIM_3D:
        raise NotImplementedError("3D velocity model generation is not yet implemented")
    elif dimensionality != Dimensionality.DIM_2D:
        raise ValueError(f"Unsupported dimensionality: {dimensionality}")
    model_classes: Dict[str, Any] = {
        "layered": layered_model,
        "fault": random_fault_model,
        "dome": dome_model,
        "perlin": perlin_threshold_model,
    }

    model_generators: Dict[str, Any] = {
        name: (
            lambda name=name: lambda nz, nx, dx, dz: model_classes[name](
                nz, nx, dx, dz, **model_params.get(name, {}), rng=rng
            )
        )()
        for name in model_types
    }

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

    transforms_list: Dict[str, Any] = {
        name: transform_classes[name](**transform_params.get(name, {}), rng=rng)
        for name in transform_types
    }

    model_weights: List[float] = [model_probs[m] for m in model_types]
    transform_weights: List[float] = [transform_probs.get(t, 0.0) for t in transform_types]

    velocity_models = []
    n_generated: int = 0

    while n_generated < n_models:
        model_name = rng.choice(model_types, p=model_weights)
        model = model_generators[model_name](nz=nz, nx=nx, dx=dx, dz=dz)

        for name, prob in zip(transform_types, transform_weights):
            if rng.random() < prob:
                model = transforms_list[name](model)
            # end if
        # end for

        if not is_valid_model(
            model,
            min_v=min_v,
            max_v=max_v,
            zero_thresh=zero_thresh,
            unique_thresh=unique_thresh,
            entropy_thresh=entropy_thresh,
            verbose=verbose,
        ):
            if verbose:
                print(f"[x] Model {n_generated} invalid, skipped.")
            continue
        # end if

        model = filters.gaussian(model, sigma=model_blur_sigma, preserve_range=True)
        
        # Create VelocityMap object based on dimensionality
        if dimensionality == Dimensionality.DIM_1D:
            velocity_map = VelocityMap.from_1d_array(model, dz)
            velocity_model = VelocityModel1D(velocity_map)
        elif dimensionality == Dimensionality.DIM_2D:
            velocity_map = VelocityMap.from_2d_array(model, dx, dz)
            velocity_model = VelocityModel2D(velocity_map)
        elif dimensionality == Dimensionality.DIM_3D:
            velocity_map = VelocityMap.from_3d_array(model, dx, dy, dz)
            velocity_model = VelocityModel3D(velocity_map)
        else:
            # This should never happen due to the dimensionality check at the beginning
            raise NotImplementedError(f"Unsupported dimensionality: {dimensionality}")
        
        # Add metadata about the generation process
        velocity_model._metadata.update({
            "generated_with": "generate_models",
            "model_type": model_name,
            "model_blur_sigma": model_blur_sigma,
            "generation_date": str(np.datetime64('now'))
        })
        
        velocity_models.append(velocity_model)
        n_generated += 1
    # end while

    return velocity_models
# end generate_models
