"""
Configuration classes for velocity model generation.

This module provides Pydantic models for configuring the generation of synthetic
velocity models and their transformations.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class ValidationConfig(BaseModel):
    """
    Configuration for model validation.
    
    Attributes:
        min_entropy: Minimum entropy score for a valid model
        max_entropy: Maximum entropy score for a valid model
        min_velocity: Minimum allowed velocity value
        max_velocity: Maximum allowed velocity value
    """
    min_entropy: Optional[float] = Field(None, description="Minimum entropy score for a valid model")
    max_entropy: Optional[float] = Field(None, description="Maximum entropy score for a valid model")
    min_velocity: float = Field(1500.0, description="Minimum allowed velocity value")
    max_velocity: float = Field(4500.0, description="Maximum allowed velocity value")
    
    model_config = ConfigDict(
        title="Validation Configuration",
        extra="forbid"
    )
# end class ValidationConfig


class GridConfig(BaseModel):
    """
    Configuration for the model grid.
    
    Attributes:
        nx: Number of grid points in x direction
        nz: Number of grid points in z direction
        dx: Grid spacing in x direction (meters)
        dz: Grid spacing in z direction (meters)
    """
    nx: int = Field(100, description="Number of grid points in x direction")
    nz: int = Field(100, description="Number of grid points in z direction")
    dx: float = Field(10.0, description="Grid spacing in x direction (meters)")
    dz: float = Field(10.0, description="Grid spacing in z direction (meters)")
    
    model_config = ConfigDict(
        title="Grid Configuration",
        extra="forbid"
    )
    
    @field_validator("nx", "nz")
    @classmethod
    def validate_grid_size(cls, v: int) -> int:
        """Validate that grid dimensions are positive."""
        if v <= 0:
            raise ValueError("Grid dimensions must be positive")
        # end if
        return v
    # end def validate_grid_size
    
    @field_validator("dx", "dz")
    @classmethod
    def validate_grid_spacing(cls, v: float) -> float:
        """Validate that grid spacing is positive."""
        if v <= 0:
            raise ValueError("Grid spacing must be positive")
        # end if
        return v
    # end def validate_grid_spacing
# end class GridConfig


class LayeredModelParams(BaseModel):
    """
    Parameters for generating layered velocity models.
    
    Attributes:
        num_layers: Number of layers in the model
        min_velocity: Minimum velocity value (m/s)
        max_velocity: Maximum velocity value (m/s)
        smoothing: Amount of smoothing to apply (0.0 = no smoothing)
    """
    num_layers: int = Field(5, description="Number of layers in the model")
    min_velocity: float = Field(1500.0, description="Minimum velocity value (m/s)")
    max_velocity: float = Field(4500.0, description="Maximum velocity value (m/s)")
    smoothing: float = Field(0.0, description="Amount of smoothing to apply (0.0 = no smoothing)")
    
    model_config = ConfigDict(
        title="Layered Model Parameters",
        extra="forbid"
    )
    
    @field_validator("num_layers")
    @classmethod
    def validate_num_layers(cls, v: int) -> int:
        """Validate that number of layers is positive."""
        if v <= 0:
            raise ValueError("Number of layers must be positive")
        # end if
        return v
    # end def validate_num_layers
    
    @field_validator("min_velocity", "max_velocity")
    @classmethod
    def validate_velocity(cls, v: float) -> float:
        """Validate that velocity values are positive."""
        if v <= 0:
            raise ValueError("Velocity values must be positive")
        # end if
        return v
    # end def validate_velocity
    
    @model_validator(mode="after")
    def validate_velocity_range(self) -> "LayeredModelParams":
        """Validate that min_velocity is less than max_velocity."""
        if self.min_velocity >= self.max_velocity:
            raise ValueError("min_velocity must be less than max_velocity")
        # end if
        return self
    # end def validate_velocity_range
# end class LayeredModelParams


class FaultModelParams(LayeredModelParams):
    """
    Parameters for generating fault velocity models.
    
    Attributes:
        num_faults: Number of faults to include
        max_fault_throw: Maximum displacement of layers across faults
    """
    num_faults: int = Field(2, description="Number of faults to include")
    max_fault_throw: Optional[int] = Field(None, description="Maximum displacement of layers across faults")
    
    model_config = ConfigDict(
        title="Fault Model Parameters",
        extra="forbid"
    )
    
    @field_validator("num_faults")
    @classmethod
    def validate_num_faults(cls, v: int) -> int:
        """Validate that number of faults is non-negative."""
        if v < 0:
            raise ValueError("Number of faults must be non-negative")
        # end if
        return v
    # end def validate_num_faults
    
    @field_validator("max_fault_throw")
    @classmethod
    def validate_max_fault_throw(cls, v: Optional[int]) -> Optional[int]:
        """Validate that max_fault_throw is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("max_fault_throw must be positive if provided")
        # end if
        return v
    # end def validate_max_fault_throw
# end class FaultModelParams


class DomeModelParams(LayeredModelParams):
    """
    Parameters for generating dome velocity models.
    
    Attributes:
        dome_height: Height of the dome in grid points
        dome_width: Width of the dome in grid points
        dome_center: X-coordinate of the dome center
    """
    dome_height: Optional[int] = Field(None, description="Height of the dome in grid points")
    dome_width: Optional[int] = Field(None, description="Width of the dome in grid points")
    dome_center: Optional[int] = Field(None, description="X-coordinate of the dome center")
    
    model_config = ConfigDict(
        title="Dome Model Parameters",
        extra="forbid"
    )
    
    @field_validator("dome_height", "dome_width")
    @classmethod
    def validate_dome_dimensions(cls, v: Optional[int]) -> Optional[int]:
        """Validate that dome dimensions are positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("Dome dimensions must be positive if provided")
        # end if
        return v
    # end def validate_dome_dimensions
    
    @field_validator("dome_center")
    @classmethod
    def validate_dome_center(cls, v: Optional[int]) -> Optional[int]:
        """Validate that dome_center is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError("dome_center must be non-negative if provided")
        # end if
        return v
    # end def validate_dome_center
# end class DomeModelParams


class PerlinModelParams(BaseModel):
    """
    Parameters for generating Perlin noise-based velocity models.
    
    Attributes:
        min_velocity: Minimum velocity value (m/s)
        max_velocity: Maximum velocity value (m/s)
        num_thresholds: Number of velocity regions
        scale: Scale of the Perlin noise
        octaves: Number of octaves for Perlin noise
        smoothing: Amount of smoothing to apply (0.0 = no smoothing)
    """
    min_velocity: float = Field(1500.0, description="Minimum velocity value (m/s)")
    max_velocity: float = Field(4500.0, description="Maximum velocity value (m/s)")
    num_thresholds: int = Field(4, description="Number of velocity regions")
    scale: float = Field(10.0, description="Scale of the Perlin noise")
    octaves: int = Field(6, description="Number of octaves for Perlin noise")
    smoothing: float = Field(0.0, description="Amount of smoothing to apply (0.0 = no smoothing)")
    
    model_config = ConfigDict(
        title="Perlin Model Parameters",
        extra="forbid"
    )
    
    @field_validator("min_velocity", "max_velocity")
    @classmethod
    def validate_velocity(cls, v: float) -> float:
        """Validate that velocity values are positive."""
        if v <= 0:
            raise ValueError("Velocity values must be positive")
        # end if
        return v
    # end def validate_velocity
    
    @field_validator("num_thresholds", "octaves")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate that integer parameters are positive."""
        if v <= 0:
            raise ValueError("Integer parameters must be positive")
        # end if
        return v
    # end def validate_positive_int
    
    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v: float) -> float:
        """Validate that scale is positive."""
        if v <= 0:
            raise ValueError("Scale must be positive")
        # end if
        return v
    # end def validate_scale
    
    @model_validator(mode="after")
    def validate_velocity_range(self) -> "PerlinModelParams":
        """Validate that min_velocity is less than max_velocity."""
        if self.min_velocity >= self.max_velocity:
            raise ValueError("min_velocity must be less than max_velocity")
        # end if
        return self
    # end def validate_velocity_range
# end class PerlinModelParams


class ModelParams(BaseModel):
    """
    Union of all model parameter types.
    
    Attributes:
        layered: Parameters for layered models
        fault: Parameters for fault models
        dome: Parameters for dome models
        perlin: Parameters for Perlin noise-based models
    """
    layered: Optional[LayeredModelParams] = Field(None, description="Parameters for layered models")
    fault: Optional[FaultModelParams] = Field(None, description="Parameters for fault models")
    dome: Optional[DomeModelParams] = Field(None, description="Parameters for dome models")
    perlin: Optional[PerlinModelParams] = Field(None, description="Parameters for Perlin noise-based models")
    
    model_config = ConfigDict(
        title="Model Parameters",
        extra="forbid"
    )
# end class ModelParams


class RotateTransformParams(BaseModel):
    """
    Parameters for random rotation transformation.
    
    Attributes:
        max_angle: Maximum rotation angle in degrees
        fill_value: Value to fill empty areas after rotation
    """
    max_angle: float = Field(10.0, description="Maximum rotation angle in degrees")
    fill_value: Optional[float] = Field(None, description="Value to fill empty areas after rotation")
    
    model_config = ConfigDict(
        title="Rotate Transform Parameters",
        extra="forbid"
    )
# end class RotateTransformParams


class StretchTransformParams(BaseModel):
    """
    Parameters for random stretching transformation.
    
    Attributes:
        max_factor_x: Maximum stretch factor in x direction
        max_factor_z: Maximum stretch factor in z direction
        fill_value: Value to fill empty areas after stretching
    """
    max_factor_x: float = Field(0.2, description="Maximum stretch factor in x direction")
    max_factor_z: float = Field(0.2, description="Maximum stretch factor in z direction")
    fill_value: Optional[float] = Field(None, description="Value to fill empty areas after stretching")
    
    model_config = ConfigDict(
        title="Stretch Transform Parameters",
        extra="forbid"
    )
    
    @field_validator("max_factor_x", "max_factor_z")
    @classmethod
    def validate_max_factor(cls, v: float) -> float:
        """Validate that stretch factors are positive."""
        if v <= 0:
            raise ValueError("Stretch factors must be positive")
        # end if
        return v
    # end def validate_max_factor
# end class StretchTransformParams


class InclusionTransformParams(BaseModel):
    """
    Parameters for adding random inclusions.
    
    Attributes:
        num_inclusions: Number of inclusions to add
        min_radius: Minimum radius of inclusions
        max_radius: Maximum radius of inclusions
        min_velocity: Minimum velocity value for inclusions
        max_velocity: Maximum velocity value for inclusions
    """
    num_inclusions: int = Field(3, description="Number of inclusions to add")
    min_radius: int = Field(5, description="Minimum radius of inclusions")
    max_radius: int = Field(15, description="Maximum radius of inclusions")
    min_velocity: Optional[float] = Field(None, description="Minimum velocity value for inclusions")
    max_velocity: Optional[float] = Field(None, description="Maximum velocity value for inclusions")
    
    model_config = ConfigDict(
        title="Inclusion Transform Parameters",
        extra="forbid"
    )
    
    @field_validator("num_inclusions")
    @classmethod
    def validate_num_inclusions(cls, v: int) -> int:
        """Validate that number of inclusions is non-negative."""
        if v < 0:
            raise ValueError("Number of inclusions must be non-negative")
        # end if
        return v
    # end def validate_num_inclusions
    
    @field_validator("min_radius", "max_radius")
    @classmethod
    def validate_radius(cls, v: int) -> int:
        """Validate that radius values are positive."""
        if v <= 0:
            raise ValueError("Radius values must be positive")
        # end if
        return v
    # end def validate_radius
    
    @model_validator(mode="after")
    def validate_radius_range(self) -> "InclusionTransformParams":
        """Validate that min_radius is less than or equal to max_radius."""
        if self.min_radius > self.max_radius:
            raise ValueError("min_radius must be less than or equal to max_radius")
        # end if
        return self
    # end def validate_radius_range
    
    @model_validator(mode="after")
    def validate_velocity_range(self) -> "InclusionTransformParams":
        """Validate that min_velocity is less than max_velocity if both are provided."""
        if (self.min_velocity is not None and self.max_velocity is not None and
                self.min_velocity >= self.max_velocity):
            raise ValueError("min_velocity must be less than max_velocity")
        # end if
        return self
    # end def validate_velocity_range
# end class InclusionTransformParams


class PerlinNoiseTransformParams(BaseModel):
    """
    Parameters for adding Perlin noise.
    
    Attributes:
        amplitude: Maximum amplitude of the noise relative to model range
        scale: Scale of the Perlin noise
        octaves: Number of octaves for Perlin noise
    """
    amplitude: float = Field(0.1, description="Maximum amplitude of the noise relative to model range")
    scale: float = Field(10.0, description="Scale of the Perlin noise")
    octaves: int = Field(4, description="Number of octaves for Perlin noise")
    
    model_config = ConfigDict(
        title="Perlin Noise Transform Parameters",
        extra="forbid"
    )
    
    @field_validator("amplitude")
    @classmethod
    def validate_amplitude(cls, v: float) -> float:
        """Validate that amplitude is positive."""
        if v <= 0:
            raise ValueError("Amplitude must be positive")
        # end if
        return v
    # end def validate_amplitude
    
    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v: float) -> float:
        """Validate that scale is positive."""
        if v <= 0:
            raise ValueError("Scale must be positive")
        # end if
        return v
    # end def validate_scale
    
    @field_validator("octaves")
    @classmethod
    def validate_octaves(cls, v: int) -> int:
        """Validate that octaves is positive."""
        if v <= 0:
            raise ValueError("Octaves must be positive")
        # end if
        return v
    # end def validate_octaves
# end class PerlinNoiseTransformParams


class CrosslineTransformParams(BaseModel):
    """
    Parameters for adding random cross-cutting lines.
    
    Attributes:
        num_lines: Number of lines to add
        min_width: Minimum width of lines
        max_width: Maximum width of lines
        min_velocity: Minimum velocity value for lines
        max_velocity: Maximum velocity value for lines
    """
    num_lines: int = Field(2, description="Number of lines to add")
    min_width: int = Field(1, description="Minimum width of lines")
    max_width: int = Field(5, description="Maximum width of lines")
    min_velocity: Optional[float] = Field(None, description="Minimum velocity value for lines")
    max_velocity: Optional[float] = Field(None, description="Maximum velocity value for lines")
    
    model_config = ConfigDict(
        title="Crossline Transform Parameters",
        extra="forbid"
    )
    
    @field_validator("num_lines")
    @classmethod
    def validate_num_lines(cls, v: int) -> int:
        """Validate that number of lines is non-negative."""
        if v < 0:
            raise ValueError("Number of lines must be non-negative")
        # end if
        return v
    # end def validate_num_lines
    
    @field_validator("min_width", "max_width")
    @classmethod
    def validate_width(cls, v: int) -> int:
        """Validate that width values are positive."""
        if v <= 0:
            raise ValueError("Width values must be positive")
        # end if
        return v
    # end def validate_width
    
    @model_validator(mode="after")
    def validate_width_range(self) -> "CrosslineTransformParams":
        """Validate that min_width is less than or equal to max_width."""
        if self.min_width > self.max_width:
            raise ValueError("min_width must be less than or equal to max_width")
        # end if
        return self
    # end def validate_width_range
    
    @model_validator(mode="after")
    def validate_velocity_range(self) -> "CrosslineTransformParams":
        """Validate that min_velocity is less than max_velocity if both are provided."""
        if (self.min_velocity is not None and self.max_velocity is not None and
                self.min_velocity >= self.max_velocity):
            raise ValueError("min_velocity must be less than max_velocity")
        # end if
        return self
    # end def validate_velocity_range
# end class CrosslineTransformParams


class SwirlTransformParams(BaseModel):
    """
    Parameters for random swirl effect.
    
    Attributes:
        max_strength: Maximum strength of the swirl effect
        radius: Radius of the swirl effect
    """
    max_strength: float = Field(10.0, description="Maximum strength of the swirl effect")
    radius: Optional[int] = Field(None, description="Radius of the swirl effect")
    
    model_config = ConfigDict(
        title="Swirl Transform Parameters",
        extra="forbid"
    )
    
    @field_validator("max_strength")
    @classmethod
    def validate_max_strength(cls, v: float) -> float:
        """Validate that max_strength is positive."""
        if v <= 0:
            raise ValueError("max_strength must be positive")
        # end if
        return v
    # end def validate_max_strength
    
    @field_validator("radius")
    @classmethod
    def validate_radius(cls, v: Optional[int]) -> Optional[int]:
        """Validate that radius is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("radius must be positive if provided")
        # end if
        return v
    # end def validate_radius
# end class SwirlTransformParams


class PiecewiseAffineTransformParams(BaseModel):
    """
    Parameters for random piecewise affine transformation.
    
    Attributes:
        num_points: Number of control points in each dimension
        max_displacement: Maximum displacement of control points
    """
    num_points: int = Field(4, description="Number of control points in each dimension")
    max_displacement: float = Field(0.1, description="Maximum displacement of control points")
    
    model_config = ConfigDict(
        title="Piecewise Affine Transform Parameters",
        extra="forbid"
    )
    
    @field_validator("num_points")
    @classmethod
    def validate_num_points(cls, v: int) -> int:
        """Validate that num_points is at least 3."""
        if v < 3:
            raise ValueError("num_points must be at least 3")
        # end if
        return v
    # end def validate_num_points
    
    @field_validator("max_displacement")
    @classmethod
    def validate_max_displacement(cls, v: float) -> float:
        """Validate that max_displacement is positive."""
        if v <= 0:
            raise ValueError("max_displacement must be positive")
        # end if
        return v
    # end def validate_max_displacement
# end class PiecewiseAffineTransformParams


class DisplacementFieldTransformParams(BaseModel):
    """
    Parameters for random displacement field.
    
    Attributes:
        alpha: Maximum displacement as fraction of image size
        sigma: Standard deviation of the Gaussian filter
    """
    alpha: float = Field(0.1, description="Maximum displacement as fraction of image size")
    sigma: float = Field(4.0, description="Standard deviation of the Gaussian filter")
    
    model_config = ConfigDict(
        title="Displacement Field Transform Parameters",
        extra="forbid"
    )
    
    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        """Validate that alpha is positive."""
        if v <= 0:
            raise ValueError("alpha must be positive")
        # end if
        return v
    # end def validate_alpha
    
    @field_validator("sigma")
    @classmethod
    def validate_sigma(cls, v: float) -> float:
        """Validate that sigma is positive."""
        if v <= 0:
            raise ValueError("sigma must be positive")
        # end if
        return v
    # end def validate_sigma
# end class DisplacementFieldTransformParams


class TransformParams(BaseModel):
    """
    Union of all transformation parameter types.
    
    Attributes:
        rotate: Parameters for random rotation
        stretch: Parameters for random stretching
        inclusion: Parameters for adding random inclusions
        perlin_noise: Parameters for adding Perlin noise
        crossline: Parameters for adding random cross-cutting lines
        swirl: Parameters for random swirl effect
        piecewise_affine: Parameters for random piecewise affine transformation
        displacement_field: Parameters for random displacement field
    """
    rotate: Optional[RotateTransformParams] = Field(None, description="Parameters for random rotation")
    stretch: Optional[StretchTransformParams] = Field(None, description="Parameters for random stretching")
    inclusion: Optional[InclusionTransformParams] = Field(None, description="Parameters for adding random inclusions")
    perlin_noise: Optional[PerlinNoiseTransformParams] = Field(None, description="Parameters for adding Perlin noise")
    crossline: Optional[CrosslineTransformParams] = Field(None, description="Parameters for adding random cross-cutting lines")
    swirl: Optional[SwirlTransformParams] = Field(None, description="Parameters for random swirl effect")
    piecewise_affine: Optional[PiecewiseAffineTransformParams] = Field(None, description="Parameters for random piecewise affine transformation")
    displacement_field: Optional[DisplacementFieldTransformParams] = Field(None, description="Parameters for random displacement field")
    
    model_config = ConfigDict(
        title="Transform Parameters",
        extra="forbid"
    )
# end class TransformParams


class ModelGenerationConfig(BaseModel):
    """
    Configuration for generating synthetic velocity models.
    
    Attributes:
        num_models: Number of models to generate
        output_dir: Directory to save the generated models
        model_type: Type of model to generate ('layered', 'fault', 'dome', 'perlin', or 'mixed')
        grid: Grid configuration
        validation: Validation configuration
        model_params: Model parameters
        transform_params: Transformation parameters
        fault_probability: Probability of generating a fault model when model_type is 'mixed'
        dome_probability: Probability of generating a dome model when model_type is 'mixed'
        perlin_probability: Probability of generating a perlin model when model_type is 'mixed'
        random_seed: Random seed for reproducibility
        plot_models: Whether to plot the generated models
    """
    num_models: int = Field(10, description="Number of models to generate")
    output_dir: str = Field(..., description="Directory to save the generated models")
    model_type: str = Field("layered", description="Type of model to generate ('layered', 'fault', 'dome', 'perlin', or 'mixed')")
    grid: GridConfig = Field(default_factory=GridConfig, description="Grid configuration")
    validation: ValidationConfig = Field(default_factory=ValidationConfig, description="Validation configuration")
    model_params: ModelParams = Field(default_factory=ModelParams, description="Model parameters")
    transform_params: TransformParams = Field(default_factory=TransformParams, description="Transformation parameters")
    fault_probability: float = Field(0.5, description="Probability of generating a fault model when model_type is 'mixed'")
    dome_probability: float = Field(0.3, description="Probability of generating a dome model when model_type is 'mixed'")
    perlin_probability: float = Field(0.2, description="Probability of generating a perlin model when model_type is 'mixed'")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    plot_models: bool = Field(True, description="Whether to plot the generated models")
    
    model_config = ConfigDict(
        title="Model Generation Configuration",
        extra="forbid"
    )
    
    @field_validator("num_models")
    @classmethod
    def validate_num_models(cls, v: int) -> int:
        """Validate that number of models is positive."""
        if v <= 0:
            raise ValueError("Number of models must be positive")
        # end if
        return v
    # end def validate_num_models
    
    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate that model_type is one of the allowed values."""
        allowed_types = ["layered", "fault", "dome", "perlin", "mixed"]
        if v not in allowed_types:
            raise ValueError(f"model_type must be one of {allowed_types}")
        # end if
        return v
    # end def validate_model_type
    
    @field_validator("fault_probability", "dome_probability", "perlin_probability")
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate that probability values are between 0 and 1."""
        if v < 0 or v > 1:
            raise ValueError("Probability values must be between 0 and 1")
        # end if
        return v
    # end def validate_probability
    
    @model_validator(mode="after")
    def validate_mixed_probabilities(self) -> "ModelGenerationConfig":
        """Validate that mixed model probabilities sum to at most 1."""
        if self.model_type == "mixed":
            total_prob = self.fault_probability + self.dome_probability + self.perlin_probability
            if total_prob > 1.0:
                raise ValueError("Mixed model probabilities must sum to at most 1.0")
            # end if
        # end if
        return self
    # end def validate_mixed_probabilities
# end class ModelGenerationConfig