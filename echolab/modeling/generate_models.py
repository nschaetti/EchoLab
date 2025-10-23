"""
High-level helpers to generate collections of synthetic velocity models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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
    verbose: bool = False,
) -> np.ndarray:
    """
    Generate a batch of velocity models using configurable primitives.
    """
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

    models: List[np.ndarray] = []
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
        models.append(model)
        n_generated += 1
    # end while

    return np.stack(models)
# end generate_models
