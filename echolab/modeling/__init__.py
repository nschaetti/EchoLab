"""Synthetic velocity model generation utilities and seismic modeling tools."""

from .generate_models import generate_models
from .layer_generation import (
    layered_model,
    random_fault_model,
    dome_model,
    perlin_threshold_model,
    generate_perlin_noise_2d,
    save_and_plot,
)
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
)
from .validation import entropy_score, is_valid_model
from .wavelets import ricker

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

# Import velocity model classes
from .velocity_model import (
    VelocityModel,
    VelocityModel1D,
    VelocityModel2D,
    VelocityModel3D,
    load_velocity_model,
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

