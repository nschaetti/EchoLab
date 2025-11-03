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
from echolab.modeling.velocity_models import Dimensionality, save_velocity_models
from echolab.modeling.model_generation_config import ModelGenerationConfig

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
    
    # Normalize probabilities if requested
    if normalise and result:
        total = sum(result.values())
        if total <= 0.0:
            raise ValueError(f"At least one {label} must have a positive probability.")
        
        for name in result:
            result[name] /= total
    
    return result


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
    
    try:
        return converter(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Configuration field '{name}' has an invalid value.") from exc


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
        
    Raises:
        ValueError: If model_params is not a mapping
    """
    model_params_section = config.get_parameter("model_params", {})
    if model_params_section is None:
        model_params_section = {}
    
    if not isinstance(model_params_section, dict):
        raise ValueError("'model_params' must be a mapping.")
    
    # Global settings
    global_model_settings: Dict[str, Any] = {
        key: value
        for key, value in model_params_section.items()
        if not isinstance(value, dict)
    }

    # Per model settings
    per_model_settings: Dict[str, Dict[str, Any]] = {
        key: value
        for key, value in model_params_section.items()
        if isinstance(value, dict)
    }
    
    return global_model_settings, per_model_settings


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
        
    Raises:
        ValueError: If grid dimensions or spacing are invalid
    """
    # Extract parameters
    nx = _extract_numeric(config, global_model_settings, "nx", int)
    nz = _extract_numeric(config, global_model_settings, "nz", int)
    dx = _extract_numeric(config, global_model_settings, "dx", float)
    dz = _extract_numeric(config, global_model_settings, "dz", float)
    
    # Valid nx, nz, dx, dz
    if nx <= 0 or nz <= 0 or dx <= 0 or dz <= 0:
        raise ValueError("Grid dimensions (nx, nz) and spacing (dx, dz) must be positive.")
    
    # For 3D models, get ny and dy parameters
    ny = config.get_parameter("ny", None)
    dy = config.get_parameter("dy", None)
    
    return nx, nz, dx, dz, ny, dy


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
        
    Raises:
        ValueError: If validation section is not a mapping
    """
    # Get validation parameters
    validation_section = config.get_parameter("validation", {})
    if validation_section is None:
        validation_section = {}
    
    if not isinstance(validation_section, dict):
        raise ValueError("'validation' must be a mapping when provided.")
    
    # Get validation parameters
    min_velocity = float(validation_section.get("min_v", config.get_parameter("min_v", config.get_parameter("min_velocity", 1000.0))))
    max_velocity = float(validation_section.get("max_v", config.get_parameter("max_v", config.get_parameter("max_velocity", 5000.0))))
    unique_thresh = float(validation_section.get("unique_thresh", config.get_parameter("unique_thresh", 0.99)))
    entropy_thresh = float(validation_section.get("entropy_thresh", config.get_parameter("entropy_thresh", 1.0)))
    zero_thresh = float(validation_section.get("zero_thresh", config.get_parameter("zero_thresh", 0.01)))
    
    return min_velocity, max_velocity, unique_thresh, entropy_thresh, zero_thresh


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
    # Get model_blur_sigma from config or fallback to global_model_settings
    model_blur_sigma = float(
        config.get_parameter(
            "model_blur_sigma",
            global_model_settings.get("sigma", 0.0),
        )
    )

    # If not found, try to find in per_model_settings
    if model_blur_sigma == 0.0:
        for params in per_model_settings.values():
            if isinstance(params, dict) and "sigma" in params:
                try:
                    model_blur_sigma = float(params["sigma"])
                    break
                except (TypeError, ValueError):
                    continue
    
    return model_blur_sigma


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
        
    Raises:
        ValueError: If model types or probabilities are invalid
    """
    # Get model types and probabilities
    model_types = list(config.get_parameter("model_types", []))
    if not model_types:
        raise ValueError("Configuration must provide at least one entry in 'model_types'.")
    
    model_probs_map = config.get_parameter("model_probs") or config.get_parameter("model_probabilities") or {}
    if not isinstance(model_probs_map, dict):
        raise ValueError("'model_probs' or 'model_probabilities' must map model names to probability weights.")
    
    # Convert a mapping of string keys to floats,
    # ensuring each requested key exists.
    model_probs = _coerce_probability_map(
        item_names=model_types,
        values_map=model_probs_map,
        label="model",
        normalise=True
    )
    
    return model_types, model_probs


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
        
        if not isinstance(params, dict):
            raise ValueError(f"Parameters for model '{name}' must be a mapping.")
        
        params_copy = dict(params)
        params_copy.pop("sigma", None)
        sanitised_model_params[name] = params_copy
    
    return sanitised_model_params


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
        
    Raises:
        ValueError: If transform types or probabilities are invalid
    """
    # Get transform types and probabilities
    transform_types = list(config.get_parameter("transform_types") or config.get_parameter("transforms") or [])
    transform_probs_map = config.get_parameter("transform_probs") or config.get_parameter("transform_probabilities") or {}
    if transform_types and not isinstance(transform_probs_map, dict):
        raise ValueError("'transform_probs' or 'transform_probabilities' must map transform names to probabilities.")
    
    transform_probs: Dict[str, float] = {}
    for name in transform_types:
        try:
            probability = float(transform_probs_map.get(name, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Transform probability for '{name}' must be numeric.") from exc
        if probability < 0.0 or probability > 1.0:
            raise ValueError(f"Transform probability for '{name}' must lie within [0, 1].")
        transform_probs[name] = probability
    
    return transform_types, transform_probs


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
    if not isinstance(transform_params, dict):
        raise ValueError("'transform_params' must be a mapping of transform names to parameter dictionaries.")
    
    return {name: transform_params.get(name, {}) for name in transform_types}


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
        
    info_table.add_row("Model types", ", ".join(model_types))
    info_table.add_row(
        "Transforms",
        ", ".join(transform_types) if transform_types else "None",
    )
    console.print(info_table)


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

        # Override dimensionality if specified
        if dim:
            config._config["dimensionality"] = dim

        # Extract model parameters
        global_model_settings, per_model_settings = _extract_model_parameters(config)
        
        # Extract grid parameters
        nx, nz, dx, dz, ny, dy = _extract_grid_parameters(config, global_model_settings)
        
        # Extract number of models
        n_models = _extract_numeric(config, global_model_settings, "n_models", int)
        if n_models <= 0:
            raise ValueError("Configuration 'n_models' must be a positive integer.")
        
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