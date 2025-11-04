"""
Model generation utilities for creating synthetic velocity models.

This subpackage contains tools for generating synthetic velocity models,
including layer-based models, fault models, dome models, and more.
It also includes data augmentation utilities for transforming models.
"""

# Export from generate_models
from .generate_models import (
    synthesize_velocity_models,
    save_and_plot,
) # end from generate_models

# Export from layer_generation
from .layer_generation import (
    layered_model,
    random_fault_model,
    dome_model,
    perlin_threshold_model,
    generate_perlin_noise_2d,
) # end from layer_generation

# Export from data_augmentation
from .data_augmentation import (
    RandomRotation,
    RandomStretch,
    AddInclusion,
    AddPerlinNoise,
    AddCrossLine,
    RandomSwirl,
    RandomPiecewiseAffine,
    RandomDisplacementField,
    apply_random_transformations,
    RandomTransformer,
    crop_and_resize,
) # end from data_augmentation

__all__ = [
    # From generate_models
    "synthesize_velocity_models",
    "save_and_plot",
    
    # From layer_generation
    "layered_model",
    "random_fault_model",
    "dome_model",
    "perlin_threshold_model",
    "generate_perlin_noise_2d",
    
    # From data_augmentation
    "RandomRotation",
    "RandomStretch",
    "AddInclusion",
    "AddPerlinNoise",
    "AddCrossLine",
    "RandomSwirl",
    "RandomPiecewiseAffine",
    "RandomDisplacementField",
    "apply_random_transformations",
    "RandomTransformer",
    "crop_and_resize",
] # end __all__