"""
Model generation utilities for creating synthetic velocity models.

This subpackage contains tools for generating synthetic velocity models,
including layer-based models, fault models, dome models, and more.
It also includes data augmentation utilities for transforming models.
"""

# Re-export from generate_models
from ..generate_models import generate_models

# Re-export from layer_generation
from ..layer_generation import (
    layered_model,
    random_fault_model,
    dome_model,
    perlin_threshold_model,
    generate_perlin_noise_2d,
    save_and_plot,
)

# Re-export from data_augmentation
from ..data_augmentation import (
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
    RandomStretchZoom,
    RandomDisplacement,
    RandomNoise,
    RandomGaussianBlur,
    RandomBrightnessContrast,
)

# Re-export from model_generation_config
from ..model_generation_config import (
    ModelGenerationConfig,
    ValidationConfig,
    GridConfig,
    LayeredModelParams,
    FaultModelParams,
    DomeModelParams,
    PerlinModelParams,
    ModelParams,
    RotateTransformParams,
    StretchTransformParams,
    InclusionTransformParams,
    PerlinNoiseTransformParams,
    CrosslineTransformParams,
    SwirlTransformParams,
    PiecewiseAffineTransformParams,
    DisplacementFieldTransformParams,
    TransformParams,
)

__all__ = [
    # From generate_models
    "generate_models",
    
    # From layer_generation
    "layered_model",
    "random_fault_model",
    "dome_model",
    "perlin_threshold_model",
    "generate_perlin_noise_2d",
    "save_and_plot",
    
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
    "RandomStretchZoom",
    "RandomDisplacement",
    "RandomNoise",
    "RandomGaussianBlur",
    "RandomBrightnessContrast",
    
    # From model_generation_config
    "ModelGenerationConfig",
    "ValidationConfig",
    "GridConfig",
    "LayeredModelParams",
    "FaultModelParams",
    "DomeModelParams",
    "PerlinModelParams",
    "ModelParams",
    "RotateTransformParams",
    "StretchTransformParams",
    "InclusionTransformParams",
    "PerlinNoiseTransformParams",
    "CrosslineTransformParams",
    "SwirlTransformParams",
    "PiecewiseAffineTransformParams",
    "DisplacementFieldTransformParams",
    "TransformParams",
]