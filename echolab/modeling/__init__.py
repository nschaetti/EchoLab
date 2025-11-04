"""Synthetic velocity model generation utilities and seismic modeling tools."""

# Import from generation subpackage
from .generation import (
    generate_models,
    layered_model,
    random_fault_model,
    dome_model,
    perlin_threshold_model,
    generate_perlin_noise_2d,
    save_and_plot,
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
)

from .validation import entropy_score, is_valid_model
from .wavelets import ricker

# Import from velocity modules
from .velocity_map import (
    Dimensionality,
    VelocityMap,
    save_velocity_maps,
    load_velocity_maps,
) # end from velocity_map

from .velocity_model import (
    VelocityModel,
    VelocityModel1D,
    VelocityModel2D,
    VelocityModel3D,
    load_velocity_model,
) # end from velocity_model

from .velocity_models import (
    save_velocity_models,
    load_velocity_models,
) # end from velocity_models

# Import acoustic field simulation classes
from .acoustic_field import (
    PressureField,
    AcousticPressureField,
    strip_absorbing_boundary,
)

# Import noise source classes
from .noise import (
    NoiseSource,
    RickerWavelet,
    RandomRickerWavelet,
    BrownNoise,
    PinkNoise,
    CompositeNoiseSource,
)

# Import simulator classes
from .simulator import (
    Simulator,
    OpenFWISimulator,
    OpenANFWISimulator,
)

__all__ = [
    # Velocity model generation
    "generate_models",
    "layered_model",
    "random_fault_model",
    "dome_model",
    "perlin_threshold_model",
    "generate_perlin_noise_2d",
    "save_and_plot",
    
    # Velocity maps and models
    "Dimensionality",
    "VelocityMap",
    "save_velocity_maps",
    "load_velocity_maps",
    "save_velocity_models",
    "load_velocity_models",
    
    # Data augmentation
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
    
    # Validation
    "entropy_score",
    "is_valid_model",
    
    # Wavelets
    "ricker",
    
    # Acoustic field simulation
    "PressureField",
    "AcousticPressureField",
    "strip_absorbing_boundary",
    
    # Noise sources
    "NoiseSource",
    "RickerWavelet",
    "RandomRickerWavelet",
    "BrownNoise",
    "PinkNoise",
    "CompositeNoiseSource",
    
    # Simulators
    "Simulator",
    "OpenFWISimulator",
    "OpenANFWISimulator",
    
    # Velocity models
    "VelocityModel",
    "VelocityModel1D",
    "VelocityModel2D",
    "VelocityModel3D",
    "load_velocity_model",
]

