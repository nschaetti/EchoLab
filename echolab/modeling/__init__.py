"""Synthetic velocity model generation utilities and seismic modeling tools."""

# Import from generation subpackage
from .generation import (
    synthesize_velocity_models,
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

from .validation import entropy_score, is_valid_velmap
from .wavelets import ricker

# Import from velocity modules
from .velocity import (
    Dimensionality,
    VelocityMap,
    VelocityModel,
    VelocityModel1D,
    VelocityModel2D,
    VelocityModel3D,
    VelocityModelBase,
    load_velocity_model,
    load_velocity_models,
    save_velocity_models,
    save_velocity_maps,
    load_velocity_maps,
    create_velocity_model,
    as_map,
    is_velocity_model,
    open_models,
) # end from velocity

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
    "synthesize_velocity_models",
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
    "open_models",
    "create_velocity_model",
    "as_map",
    "is_velocity_model",
    
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
    "is_valid_velmap",
    
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
    "VelocityModelBase",
    "VelocityModel1D",
    "VelocityModel2D",
    "VelocityModel3D",
    "load_velocity_model",
]
