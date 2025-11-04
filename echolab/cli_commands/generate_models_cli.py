"""
Module for generating synthetic velocity models via CLI.

This module provides functionality to generate synthetic velocity models
based on configuration parameters and save them to a file.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from echolab.modeling import generate_models as synthesize_velocity_models
from echolab.modeling.velocity import Dimensionality, save_velocity_models
from echolab.modeling.generation import ModelGenerationConfig

# Create console for rich output
console = Console()


def _load_generation_config(config_path: Path) -> ModelGenerationConfig:
    """
    Load model generation configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        ModelGenerationConfig object containing the loaded configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
        ValueError: If the configuration is invalid
    """
    try:
        return ModelGenerationConfig.from_yaml(config_path)
    except (FileNotFoundError, yaml.YAMLError) as exc:
        raise exc
# end def _load_generation_config


def _coerce_probability_map(
    item_names: Sequence[str], 
    values_map: Dict[str, Any], 
    label: str, 
    *, 
    normalise: bool
) -> Dict[str, float]:
    """
    Convert a mapping of string keys to normalized probability values.
    
    Args:
        item_names: Sequence of item names that must have probabilities
        values_map: Dictionary mapping item names to probability values
        label: Label for error messages (e.g., "model", "transform")
        normalise: Whether to normalize probabilities to sum to 1.0
        
    Returns:
        Dictionary mapping item names to normalized probability values
        
    Raises:
        ValueError: If probabilities are invalid or missing required items
    """
    result: Dict[str, float] = {}
    
    for name in item_names:
        try:
            probability = float(values_map.get(name, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label.capitalize()} probability for '{name}' must be numeric.") from exc
        
        if probability < 0.0:
            raise ValueError(f"{label.capitalize()} probability for '{name}' must be non-negative.")
        
        result[name] = probability
    # end for
    
    # Normalize probabilities if requested
    if normalise and result:
        total = sum(result.values())
        if total <= 0.0:
            raise ValueError(f"At least one {label} must have a positive probability.")
        
        for name in result:
            result[name] /= total
        # end for
    # end if
    
    return result
# end def _coerce_probability_map


def _extract_numeric(
    config: ModelGenerationConfig,
    global_model_settings: Dict[str, Any],
    name: str,
    converter,
    *,
    fallback: Any = None,
    required: bool = True,
) -> Any:
    """
    Extract a numeric parameter from configuration or global settings.
    
    Args:
        config: ModelGenerationConfig object
        global_model_settings: Dictionary of global model settings
        name: Name of the parameter to extract
        converter: Function to convert the parameter value (e.g., int, float)
        fallback: Default value if parameter is not found
        required: Whether the parameter is required
        
    Returns:
        Converted parameter value
        
    Raises:
        ValueError: If parameter is required but not found or has invalid value
    """
    # First try to get from ModelGenerationConfig
    value = config.get_parameter(name, None)
    
    # If not found, try global_model_settings
    if value is None and name in global_model_settings:
        value = global_model_settings[name]
    # If still not found, use fallback or raise error
    elif value is None:
        if fallback is not None:
            value = fallback
        elif not required:
            return None
        else:
            raise ValueError(f"Configuration must define '{name}'.")
    # end if
    
    try:
        return converter(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Configuration field '{name}' has an invalid value.") from exc
# end def _extract_numeric


def _extract_model_parameters(
    config: ModelGenerationConfig
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Extract model parameters from configuration.
    
    Args:
        config: ModelGenerationConfig object
        
    Returns:
        Tuple containing:
        - Dictionary of global model settings
        - Dictionary of per-model settings
    """
    # Convert model_params to dict for compatibility
    model_params_dict = config.model_params.model_dump()
    
    # Global settings (grid parameters)
    global_model_settings: Dict[str, Any] = {
        key: value
        for key, value in model_params_dict.items()
        if key in ["nx", "nz", "dx", "dz", "sigma"]
    }

    # Per model settings
    per_model_settings: Dict[str, Dict[str, Any]] = {
        key: value
        for key, value in model_params_dict.items()
        if key in ["layered", "fault", "dome", "perlin"]
    }
    
    return global_model_settings, per_model_settings
# end def _extract_model_parameters


def _extract_grid_parameters(
    config: ModelGenerationConfig,
    global_model_settings: Dict[str, Any]
) -> Tuple[int, int, float, float, Optional[int], Optional[float]]:
    """
    Extract grid parameters from configuration.
    
    Args:
        config: ModelGenerationConfig object
        global_model_settings: Dictionary of global model settings
        
    Returns:
        Tuple containing nx, nz, dx, dz, ny, dy parameters
    """
    # Extract parameters directly from model_params
    nx = config.model_params.nx
    nz = config.model_params.nz
    dx = config.model_params.dx
    dz = config.model_params.dz
    
    # For 3D models, get ny and dy parameters
    ny = config.model_params.ny if config.dimensionality == "3D" else None
    dy = config.model_params.dy if config.dimensionality == "3D" else None
    
    return nx, nz, dx, dz, ny, dy
# end def _extract_grid_parameters


def _extract_validation_parameters(
    config: ModelGenerationConfig,
    global_model_settings: Dict[str, Any]
) -> Tuple[float, float, float, float, float]:
    """
    Extract validation parameters from configuration.
    
    Args:
        config: ModelGenerationConfig object
        global_model_settings: Dictionary of global model settings
        
    Returns:
        Tuple containing min_velocity, max_velocity, unique_thresh, entropy_thresh, zero_thresh
    """
    # Get validation parameters directly from the validation attribute
    min_velocity = config.validation.min_v
    max_velocity = config.validation.max_v
    unique_thresh = config.validation.unique_thresh
    entropy_thresh = config.validation.entropy_thresh
    zero_thresh = config.validation.zero_thresh
    
    return min_velocity, max_velocity, unique_thresh, entropy_thresh, zero_thresh
# end def _extract_validation_parameters


def _extract_model_blur_sigma(
    config: ModelGenerationConfig,
    global_model_settings: Dict[str, Any],
    per_model_settings: Dict[str, Dict[str, Any]]
) -> float:
    """
    Extract model blur sigma parameter from configuration.
    
    Args:
        config: ModelGenerationConfig object
        global_model_settings: Dictionary of global model settings
        per_model_settings: Dictionary of per-model settings
        
    Returns:
        Model blur sigma value
    """
    # Get model_blur_sigma directly from config
    model_blur_sigma = config.model_blur_sigma
    
    # If it's 0, try to get from global_model_settings
    if model_blur_sigma == 0.0 and "sigma" in global_model_settings:
        model_blur_sigma = float(global_model_settings["sigma"])
    
    # If still 0, try to find in per_model_settings
    if model_blur_sigma == 0.0:
        for params in per_model_settings.values():
            if isinstance(params, dict) and "sigma" in params:
                try:
                    model_blur_sigma = float(params["sigma"])
                    break
                except (TypeError, ValueError):
                    continue
                # end try
            # end if
        # end for
    # end if
    
    return model_blur_sigma
# end def _extract_model_blur_sigma


def _extract_model_types_and_probabilities(
    config: ModelGenerationConfig
) -> Tuple[List[str], Dict[str, float]]:
    """
    Extract model types and their probabilities from configuration.
    
    Args:
        config: ModelGenerationConfig object
        
    Returns:
        Tuple containing:
        - List of model types
        - Dictionary mapping model types to probabilities
    """
    # Get model types and probabilities directly from config
    model_types = list(config.model_types)
    model_probs_map = config.model_probabilities
    
    # Convert a mapping of string keys to floats,
    # ensuring each requested key exists.
    model_probs = _coerce_probability_map(
        item_names=model_types,
        values_map=model_probs_map,
        label="model",
        normalise=True
    )
    
    return model_types, model_probs
# end def _extract_model_types_and_probabilities


def _sanitize_model_parameters(
    model_types: List[str],
    per_model_settings: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Sanitize model parameters for each model type.
    
    Args:
        model_types: List of model types
        per_model_settings: Dictionary of per-model settings
        
    Returns:
        Dictionary of sanitized model parameters
        
    Raises:
        ValueError: If parameters for a model are not a mapping
    """
    sanitised_model_params: Dict[str, Dict[str, Any]] = {}
    for name in model_types:
        params = per_model_settings.get(name, {})
        if params is None:
            params = {}
        # end if
        
        if not isinstance(params, dict):
            raise ValueError(f"Parameters for model '{name}' must be a mapping.")
        # end if
        
        params_copy = dict(params)
        params_copy.pop("sigma", None)
        sanitised_model_params[name] = params_copy
    # end for
    
    return sanitised_model_params
# end def _sanitize_model_parameters


def _extract_transform_types_and_probabilities(
    config: ModelGenerationConfig
) -> Tuple[List[str], Dict[str, float]]:
    """
    Extract transform types and their probabilities from configuration.
    
    Args:
        config: ModelGenerationConfig object
        
    Returns:
        Tuple containing:
        - List of transform types
        - Dictionary mapping transform types to probabilities
    """
    # Get transform types and probabilities directly from config
    transform_types = list(config.transforms)
    transform_probs_map = config.transform_probabilities
    
    transform_probs: Dict[str, float] = {}
    for name in transform_types:
        try:
            probability = float(transform_probs_map.get(name, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Transform probability for '{name}' must be numeric.") from exc
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"Transform probability for '{name}' must lie within [0, 1].")
        # end if
        transform_probs[name] = probability
    # end for
    
    return transform_types, transform_probs
# end def _extract_transform_types_and_probabilities


def _extract_transform_parameters(
    config: ModelGenerationConfig,
    transform_types: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Extract transform parameters from configuration.
    
    Args:
        config: ModelGenerationConfig object
        transform_types: List of transform types
        
    Returns:
        Dictionary mapping transform types to their parameters
        
    Raises:
        ValueError: If transform parameters are invalid
    """
    # Get transform parameters
    transform_params = config.get_parameter("transform_params", {})
    if transform_params is None:
        transform_params = {}
    # end if
    if not isinstance(transform_params, dict):
        raise ValueError("'transform_params' must be a mapping of transform names to parameter dictionaries.")
    # end if
    
    return {name: transform_params.get(name, {}) for name in transform_types}
# end def _extract_transform_parameters


def _display_results_table(
    config_path: Path,
    output_path: Path,
    effective_seed: Optional[int],
    velocity_models: List[Any],
    dim_str: str,
    nx: int,
    ny: Optional[int],
    nz: int,
    model_types: List[str],
    transform_types: List[str]
) -> None:
    """
    Display a table with information about the generated models.
    
    Args:
        config_path: Path to the configuration file
        output_path: Path to the output file
        effective_seed: Random seed used for generation
        velocity_models: List of generated velocity model objects
        dim_str: Dimensionality string (1D, 2D, 3D)
        nx: Number of grid points in x direction
        ny: Number of grid points in y direction (for 3D)
        nz: Number of grid points in z direction
        model_types: List of model types used
        transform_types: List of transform types used
    """
    info_table = Table(title="Synthetic Velocity Models")
    info_table.add_column("Setting", style="cyan", no_wrap=True)
    info_table.add_column("Value", style="magenta")
    info_table.add_row("Configuration", str(config_path))
    info_table.add_row("Output file", str(output_path))
    info_table.add_row("Seed", str(effective_seed) if effective_seed is not None else "None")
    info_table.add_row("Model count", str(len(velocity_models)))
    info_table.add_row("Dimensionality", dim_str)
    
    # Display grid size based on dimensionality
    if dim_str == "1D":
        info_table.add_row("Grid size", f"{nz}")
    elif dim_str == "2D":
        info_table.add_row("Grid size", f"{nz} x {nx}")
    elif dim_str == "3D":
        info_table.add_row("Grid size", f"{nz} x {ny} x {nx}")
    # end if
        
    info_table.add_row("Model types", ", ".join(model_types))
    info_table.add_row(
        "Transforms",
        ", ".join(transform_types) if transform_types else "None",
    )
    console.print(info_table)
# end def _display_results_table


def generate_models(
    config_path: Path,
    output_path: Path,
    seed: Optional[int] = None,
    overwrite: bool = False,
    verbose: bool = False,
    dim: Optional[str] = None
) -> None:
    """
    Generate synthetic velocity models and save them to a file.
    
    This function loads configuration parameters, generates velocity models
    according to the specified parameters, and saves them to the output file.
    
    Args:
        config_path: Path to the YAML configuration file
        output_path: Path to save the generated models
        seed: Optional random seed to override the one in configuration
        overwrite: Whether to overwrite existing output file
        verbose: Whether to display validation messages for discarded models
        dim: Optional dimensionality override (1D, 2D, 3D)
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        FileExistsError: If the output file exists and overwrite is False
        ValueError: If configuration parameters are invalid
        yaml.YAMLError: If the YAML file is malformed
    """
    try:
        # Load configuration using the ModelGenerationConfig class
        config = _load_generation_config(config_path)
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file '{output_path}' already exists. Use --overwrite to replace it."
            )
        # end if

        # Override dimensionality if specified
        if dim:
            config._config["dimensionality"] = dim
        # end if

        # Extract model parameters
        global_model_settings, per_model_settings = _extract_model_parameters(
            config=config
        )

        # Extract grid parameters
        nx, nz, dx, dz, ny, dy = _extract_grid_parameters(config, global_model_settings)
        
        # Extract number of models
        n_models = _extract_numeric(config, global_model_settings, "n_models", int)
        if n_models <= 0:
            raise ValueError("Configuration 'n_models' must be a positive integer.")
        # end if
        
        # Extract validation parameters
        min_velocity, max_velocity, unique_thresh, entropy_thresh, zero_thresh = _extract_validation_parameters(
            config, global_model_settings
        )
        
        # Extract model blur sigma
        model_blur_sigma = _extract_model_blur_sigma(config, global_model_settings, per_model_settings)
        
        # Extract model types and probabilities
        model_types, model_probs = _extract_model_types_and_probabilities(config)
        
        # Sanitize model parameters
        sanitised_model_params = _sanitize_model_parameters(model_types, per_model_settings)
        
        # Extract transform types and probabilities
        transform_types, transform_probs = _extract_transform_types_and_probabilities(config)
        
        # Extract transform parameters
        transform_params = _extract_transform_parameters(config, transform_types)
        
        # Get seed
        effective_seed = seed if seed is not None else config.get_parameter("seed")
        # Initialize random number generator
        rng = np.random.default_rng(effective_seed)
        
        # Get verbose flag
        verbose_flag = verbose or bool(config.get_parameter("verbose", False))
        
        # Get dimensionality from config
        dim_enum = config.get_dimensionality()
        
        # Generate velocity models
        velocity_models = synthesize_velocity_models(
            rng=rng,
            nx=nx,
            nz=nz,
            dx=dx,
            dz=dz,
            model_types=model_types,
            model_probs=model_probs,
            model_params=sanitised_model_params,
            model_blur_sigma=model_blur_sigma,
            transform_types=transform_types,
            transform_probs=transform_probs,
            transform_params=transform_params,
            n_models=n_models,
            min_v=min_velocity,
            max_v=max_velocity,
            unique_thresh=unique_thresh,
            entropy_thresh=entropy_thresh,
            zero_thresh=zero_thresh,
            dimensionality=dim_enum,
            ny=ny,
            dy=dy,
            verbose=verbose_flag,
        )

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save velocity models
        save_velocity_models(velocity_models, output_path)

        # Display results table
        dim_str = config.get_parameter("dimensionality")
        _display_results_table(
            config_path=config_path,
            output_path=output_path,
            effective_seed=effective_seed,
            velocity_models=velocity_models,
            dim_str=dim_str,
            nx=nx,
            ny=ny,
            nz=nz,
            model_types=model_types,
            transform_types=transform_types
        )
    except (FileNotFoundError, FileExistsError, ValueError, yaml.YAMLError) as exc:
        raise exc
    # end try
# end def generate_models