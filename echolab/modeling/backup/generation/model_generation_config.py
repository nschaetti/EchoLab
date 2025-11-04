"""
Class for model generation configuration using Pydantic for validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import yaml

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from ..velocity import Dimensionality


class ValidationConfig(BaseModel):
    """
    Configuration for model validation parameters.
    """
    min_v: float = Field(1000, description="Minimum velocity value")
    max_v: float = Field(5000, description="Maximum velocity value")
    zero_thresh: float = Field(0.01, description="Threshold for zero values")
    unique_thresh: float = Field(0.85, description="Threshold for uniqueness")
    entropy_thresh: float = Field(0.2, description="Threshold for entropy")


class GridConfig(BaseModel):
    """
    Configuration for grid parameters.
    """
    nx: int = Field(..., description="Number of grid points in x direction", gt=0)
    nz: int = Field(..., description="Number of grid points in z direction", gt=0)
    dx: float = Field(..., description="Grid spacing in x direction", gt=0)
    dz: float = Field(..., description="Grid spacing in z direction", gt=0)
    ny: Optional[int] = Field(None, description="Number of grid points in y direction", gt=0)
    dy: Optional[float] = Field(None, description="Grid spacing in y direction", gt=0)


class LayeredModelParams(BaseModel):
    """
    Parameters for layered velocity models.
    """
    n_layers_range: Tuple[int, int] = Field(..., description="Range for number of layers")
    v_range: Tuple[float, float] = Field(..., description="Range for velocity values")
    angle: float = Field(0.0, description="Maximum angle for layers")

    @field_validator('n_layers_range')
    def validate_n_layers_range(cls, v):
        if len(v) != 2 or v[0] < 1 or v[0] > v[1]:
            raise ValueError("n_layers_range must be a tuple of two positive integers with min <= max")
        return v

    @field_validator('v_range')
    def validate_v_range(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[0] > v[1]:
            raise ValueError("v_range must be a tuple of two positive values with min <= max")
        return v


class FaultModelParams(BaseModel):
    """
    Parameters for fault velocity models.
    """
    v_range: Tuple[float, float] = Field(..., description="Range for base velocity values")
    delta_v_range: Tuple[float, float] = Field(..., description="Range for velocity differences")
    slope_range: Tuple[float, float] = Field(..., description="Range for fault slopes")
    offset_range: Tuple[int, int] = Field(..., description="Range for fault offsets")
    sigma: float = Field(0.75, description="Smoothing sigma")

    @field_validator('v_range', 'delta_v_range', 'slope_range')
    def validate_ranges(cls, v):
        if len(v) != 2 or v[0] < 0 or v[0] > v[1]:
            raise ValueError("Range must be a tuple of two non-negative values with min <= max")
        return v

    @field_validator('offset_range')
    def validate_offset_range(cls, v):
        if len(v) != 2 or v[0] > v[1]:
            raise ValueError("offset_range must be a tuple of two integers with min <= max")
        return v


class DomeModelParams(BaseModel):
    """
    Parameters for dome velocity models.
    """
    center_range: Tuple[float, float] = Field(..., description="Range for dome center position")
    radius_range: Tuple[float, float] = Field(..., description="Range for dome radius")
    v_high_range: Tuple[float, float] = Field(..., description="Range for high velocity values")
    v_low_range: Tuple[float, float] = Field(..., description="Range for low velocity values")
    sigma: float = Field(0.75, description="Smoothing sigma")

    @field_validator('center_range', 'radius_range')
    def validate_ranges(cls, v):
        if len(v) != 2 or v[0] < 0 or v[0] > v[1]:
            raise ValueError("Range must be a tuple of two non-negative values with min <= max")
        return v

    @field_validator('v_high_range', 'v_low_range')
    def validate_velocity_ranges(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[0] > v[1]:
            raise ValueError("Velocity range must be a tuple of two positive values with min <= max")
        return v


class PerlinModelParams(BaseModel):
    """
    Parameters for Perlin noise velocity models.
    """
    scale_range: Tuple[float, float] = Field(..., description="Range for noise scale")
    v_range: Tuple[float, float] = Field(..., description="Range for velocity values")
    contrast_range: Tuple[float, float] = Field(..., description="Range for contrast values")
    n_level_range: Tuple[int, int] = Field(..., description="Range for number of octave levels")
    sigma: float = Field(0.75, description="Smoothing sigma")

    @field_validator('scale_range', 'contrast_range')
    def validate_ranges(cls, v):
        if len(v) != 2 or v[0] < 0 or v[0] > v[1]:
            raise ValueError("Range must be a tuple of two non-negative values with min <= max")
        return v

    @field_validator('v_range')
    def validate_v_range(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[0] > v[1]:
            raise ValueError("v_range must be a tuple of two positive values with min <= max")
        return v

    @field_validator('n_level_range')
    def validate_n_level_range(cls, v):
        if len(v) != 2 or v[0] < 1 or v[0] > v[1]:
            raise ValueError("n_level_range must be a tuple of two positive integers with min <= max")
        return v


class ModelParams(BaseModel):
    """
    Configuration for model parameters.
    """
    # Grid parameters
    nx: int = Field(..., description="Number of grid points in x direction", gt=0)
    nz: int = Field(..., description="Number of grid points in z direction", gt=0)
    dx: float = Field(..., description="Grid spacing in x direction", gt=0)
    dz: float = Field(..., description="Grid spacing in z direction", gt=0)
    sigma: float = Field(0.75, description="Global smoothing sigma")
    
    # Model-specific parameters
    layered: LayeredModelParams = Field(..., description="Parameters for layered models")
    fault: FaultModelParams = Field(..., description="Parameters for fault models")
    dome: DomeModelParams = Field(..., description="Parameters for dome models")
    perlin: PerlinModelParams = Field(..., description="Parameters for Perlin noise models")


class RotateTransformParams(BaseModel):
    """
    Parameters for rotation transformation.
    """
    angle_range: Tuple[float, float] = Field(..., description="Range for rotation angles")

    @field_validator('angle_range')
    def validate_angle_range(cls, v):
        if len(v) != 2 or v[0] > v[1]:
            raise ValueError("angle_range must be a tuple of two values with min <= max")
        return v


class StretchTransformParams(BaseModel):
    """
    Parameters for stretch transformation.
    """
    scale_range: Tuple[float, float] = Field(..., description="Range for stretch scale factors")

    @field_validator('scale_range')
    def validate_scale_range(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[0] > v[1]:
            raise ValueError("scale_range must be a tuple of two positive values with min <= max")
        return v


class InclusionTransformParams(BaseModel):
    """
    Parameters for inclusion transformation.
    """
    radius_range: Tuple[float, float] = Field(..., description="Range for inclusion radii")
    velocity_range: Tuple[float, float] = Field(..., description="Range for inclusion velocities")
    n_inclusions_range: Tuple[int, int] = Field(..., description="Range for number of inclusions")

    @field_validator('radius_range')
    def validate_radius_range(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[0] > v[1]:
            raise ValueError("radius_range must be a tuple of two positive values with min <= max")
        return v

    @field_validator('velocity_range')
    def validate_velocity_range(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[0] > v[1]:
            raise ValueError("velocity_range must be a tuple of two positive values with min <= max")
        return v

    @field_validator('n_inclusions_range')
    def validate_n_inclusions_range(cls, v):
        if len(v) != 2 or v[0] < 1 or v[0] > v[1]:
            raise ValueError("n_inclusions_range must be a tuple of two positive integers with min <= max")
        return v


class PerlinNoiseTransformParams(BaseModel):
    """
    Parameters for Perlin noise transformation.
    """
    scale_range: Tuple[float, float] = Field(..., description="Range for noise scale")
    amplitude: float = Field(..., description="Amplitude of the noise")
    n_levels: List[float] = Field(..., description="Octave levels for the noise")

    @field_validator('scale_range')
    def validate_scale_range(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[0] > v[1]:
            raise ValueError("scale_range must be a tuple of two positive values with min <= max")
        return v


class CrosslineTransformParams(BaseModel):
    """
    Parameters for crossline transformation.
    """
    thickness_range: Tuple[float, float] = Field(..., description="Range for crossline thickness")
    velocity_range: Tuple[float, float] = Field(..., description="Range for crossline velocities")

    @field_validator('thickness_range')
    def validate_thickness_range(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[0] > v[1]:
            raise ValueError("thickness_range must be a tuple of two positive values with min <= max")
        return v

    @field_validator('velocity_range')
    def validate_velocity_range(cls, v):
        if len(v) != 2 or v[0] <= 0 or v[0] > v[1]:
            raise ValueError("velocity_range must be a tuple of two positive values with min <= max")
        return v


class SwirlTransformParams(BaseModel):
    """
    Parameters for swirl transformation.
    """
    center_range: Tuple[float, float] = Field(..., description="Range for swirl center position")
    strength_range: Tuple[float, float] = Field(..., description="Range for swirl strength")
    radius_range: Tuple[float, float] = Field(..., description="Range for swirl radius")

    @field_validator('center_range', 'strength_range', 'radius_range')
    def validate_ranges(cls, v):
        if len(v) != 2 or v[0] < 0 or v[0] > v[1]:
            raise ValueError("Range must be a tuple of two non-negative values with min <= max")
        return v


class PiecewiseAffineTransformParams(BaseModel):
    """
    Parameters for piecewise affine transformation.
    """
    grid_shape: Tuple[int, int] = Field(..., description="Shape of the grid")
    displacement_sigma: float = Field(..., description="Sigma for displacement")

    @field_validator('grid_shape')
    def validate_grid_shape(cls, v):
        if len(v) != 2 or v[0] < 2 or v[1] < 2:
            raise ValueError("grid_shape must be a tuple of two integers >= 2")
        return v

    @field_validator('displacement_sigma')
    def validate_displacement_sigma(cls, v):
        if v <= 0:
            raise ValueError("displacement_sigma must be positive")
        return v


class DisplacementFieldTransformParams(BaseModel):
    """
    Parameters for displacement field transformation.
    """
    max_displacement: float = Field(..., description="Maximum displacement")
    smooth_sigma: float = Field(..., description="Smoothing sigma")

    @field_validator('max_displacement', 'smooth_sigma')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class TransformParams(BaseModel):
    """
    Configuration for transformation parameters.
    """
    rotate: RotateTransformParams = Field(..., description="Parameters for rotation transformation")
    stretch: StretchTransformParams = Field(..., description="Parameters for stretch transformation")
    inclusion: InclusionTransformParams = Field(..., description="Parameters for inclusion transformation")
    perlin_noise: PerlinNoiseTransformParams = Field(..., description="Parameters for Perlin noise transformation")
    add_crossline: CrosslineTransformParams = Field(..., description="Parameters for crossline transformation")
    swirl: SwirlTransformParams = Field(..., description="Parameters for swirl transformation")
    piecewise_affine: PiecewiseAffineTransformParams = Field(..., description="Parameters for piecewise affine transformation")
    displacement_field: DisplacementFieldTransformParams = Field(..., description="Parameters for displacement field transformation")


class ModelGenerationConfig(BaseModel):
    """
    Class representing configuration for velocity model generation using Pydantic for validation.
    
    This class encapsulates all parameters needed for generating velocity models
    and provides validation to ensure all required parameters are present.
    """
    # Basic parameters
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    n_models: int = Field(..., description="Number of models to generate", gt=0)
    dimensionality: Literal["1D", "2D", "3D"] = Field("2D", description="Dimensionality of the models")
    
    # Model types and probabilities
    model_types: List[str] = Field(..., description="List of model types to generate")
    model_probabilities: Dict[str, float] = Field(..., description="Probabilities for each model type")
    
    # Model parameters
    model_params: ModelParams = Field(..., description="Parameters for model generation")
    model_blur_sigma: float = Field(1.0, description="Sigma for model blurring")
    
    # Transformation parameters
    transforms: List[str] = Field([], description="List of transformations to apply")
    transform_probabilities: Dict[str, float] = Field({}, description="Probabilities for each transformation")
    transform_params: TransformParams = Field(..., description="Parameters for transformations")
    
    # External transformation parameters
    skimage_transform_prob: float = Field(0.0, description="Probability of applying scikit-image transformations")
    skimage_transform_crop_ratio: float = Field(0.0, description="Crop ratio for scikit-image transformations")
    
    # Validation parameters
    validation: ValidationConfig = Field(
        ValidationConfig(), 
        description="Parameters for model validation"
    )
    
    # Configure pydantic model
    model_config = ConfigDict(
        extra="ignore",  # Allow extra fields for backward compatibility
    )
    
    @model_validator(mode='after')
    def validate_model_types_and_probabilities(self) -> 'ModelGenerationConfig':
        """Validate that all model types have probabilities."""
        for model_type in self.model_types:
            if model_type not in self.model_probabilities:
                raise ValueError(f"Missing probability for model type: {model_type}")
        return self
    
    @model_validator(mode='after')
    def validate_transform_types_and_probabilities(self) -> 'ModelGenerationConfig':
        """Validate that all transform types have probabilities."""
        if self.transforms:
            for transform_type in self.transforms:
                if transform_type not in self.transform_probabilities:
                    raise ValueError(f"Missing probability for transform type: {transform_type}")
        return self
    
    @model_validator(mode='after')
    def validate_dimensionality(self) -> 'ModelGenerationConfig':
        """Validate that ny and dy are provided for 3D models."""
        if self.dimensionality == "3D":
            if self.model_params.ny is None or self.model_params.dy is None:
                raise ValueError("3D models require 'ny' and 'dy' parameters")
        return self
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> ModelGenerationConfig:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Returns:
            ModelGenerationConfig: A new ModelGenerationConfig instance.
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
            ValueError: If the configuration is invalid.
        """
        config_path = Path(config_path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
        
        # Load yaml config
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        
        if not isinstance(data, dict):
            raise ValueError("Model generation configuration must be a mapping.")
        
        return cls.model_validate(data)
        
    def to_yaml(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config_path: Path where the YAML configuration file will be saved.
        """
        config_path = Path(config_path).expanduser()
        
        # Convert to dict and save as yaml
        data = self.model_dump()
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.dump(data, handle, default_flow_style=False, sort_keys=False)
    
    def get_dimensionality(self) -> Dimensionality:
        """
        Get the dimensionality as a Dimensionality enum.
        
        Returns:
            Dimensionality: The dimensionality enum value.
        """
        if self.dimensionality == "1D":
            return Dimensionality.DIM_1D
        elif self.dimensionality == "2D":
            return Dimensionality.DIM_2D
        else:  # "3D"
            return Dimensionality.DIM_3D
    
    def __str__(self) -> str:
        """
        Return a string representation of the configuration.
        
        Returns:
            str: A human-readable string describing the configuration.
        """
        nx = self.model_params.nx
        nz = self.model_params.nz
        
        result = f"ModelGenerationConfig({self.dimensionality}, {self.n_models} models, grid: {nx}x{nz}"
        
        if self.dimensionality == "3D":
            ny = self.model_params.ny
            result = f"ModelGenerationConfig({self.dimensionality}, {self.n_models} models, grid: {nx}x{ny}x{nz}"
        
        result += ")"
        return result