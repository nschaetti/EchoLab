"""
Class for model generation configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from .velocity_map import Dimensionality


class ModelGenerationConfig:
    """
    Class representing configuration for velocity model generation.
    
    This class encapsulates all parameters needed for generating velocity models
    and provides validation to ensure all required parameters are present.
    
    Supports both the original flat configuration format and the improved
    hierarchical format with sections like 'general', 'model_types', etc.
    """
    
    # Required parameters that must be present in the configuration
    REQUIRED_PARAMS = [
        "n_models", 
        "model_types", "model_params",
        "validation"
    ]
    
    # Alternative key names for some parameters
    ALTERNATIVE_KEYS = {
        "model_probabilities": ["model_probs"],
        "transform_probabilities": ["transform_probs"],
        "transforms": ["transform_types"]
    }
    
    # Optional parameters with default values
    DEFAULT_PARAMS = {
        "model_blur_sigma": 1.0,
        "transforms": [],
        "transform_probabilities": {},
        "transform_params": {},
        "skimage_transform_prob": 0.0,
        "skimage_transform_crop_ratio": 0.0,
        "dimensionality": "2D",
        "ny": None,
        "dy": None,
        "seed": None,
        "verbose": False,
        "validation": {
            "min_v": 1000,
            "max_v": 5000,
            "unique_thresh": 0.85,
            "entropy_thresh": 0.2,
            "zero_thresh": 0.01
        }
    }
    
    def __init__(self, config_data: Dict[str, Any]):
        """
        Initialize a ModelGenerationConfig from a dictionary.
        
        Args:
            config_data: Dictionary containing configuration parameters.
            
        Raises:
            ValueError: If required parameters are missing or have invalid values.
        """
        # Check if this is the improved format with sections
        if self._is_improved_format(config_data):
            # Convert improved format to flat format
            self._config = self._convert_improved_to_flat(config_data)
        else:
            # Use the original format as is
            self._config = config_data.copy()
        
        self._validate_config(self._config)
        
        # Set default values for optional parameters if not present
        for param, default_value in self.DEFAULT_PARAMS.items():
            if param not in self._config:
                self._config[param] = default_value
    
    @staticmethod
    def _is_improved_format(config: Dict[str, Any]) -> bool:
        """
        Check if the configuration uses the improved format with sections.
        
        Args:
            config: Configuration dictionary.
            
        Returns:
            bool: True if the configuration uses the improved format, False otherwise.
        """
        # The improved format has 'general', 'model_types', 'model_parameters', etc. sections
        return all(section in config for section in ['general', 'model_types', 'model_parameters'])
    
    @staticmethod
    def _convert_improved_to_flat(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the improved hierarchical format to the flat format expected by the rest of the code.
        
        Args:
            config: Configuration dictionary in the improved format.
            
        Returns:
            Dict[str, Any]: Configuration dictionary in the flat format.
        """
        flat_config = {}
        
        # Extract general parameters
        if 'general' in config:
            general = config['general']
            if 'random_seed' in general:
                flat_config['seed'] = general['random_seed']
            if 'model_count' in general:
                flat_config['n_models'] = general['model_count']
            if 'smoothing_sigma' in general:
                flat_config['model_blur_sigma'] = general['smoothing_sigma']
            
            # Extract grid parameters
            if 'grid' in general:
                grid = general['grid']
                flat_config['model_params'] = flat_config.get('model_params', {})
                if 'spacing_x' in grid:
                    flat_config['model_params']['dx'] = grid['spacing_x']
                if 'spacing_z' in grid:
                    flat_config['model_params']['dz'] = grid['spacing_z']
                if 'points_x' in grid:
                    flat_config['model_params']['nx'] = grid['points_x']
                if 'points_z' in grid:
                    flat_config['model_params']['nz'] = grid['points_z']
        
        # Extract model types
        if 'model_types' in config:
            model_types_section = config['model_types']
            if 'available_types' in model_types_section:
                flat_config['model_types'] = model_types_section['available_types']
            
            # Extract model probabilities
            if 'selection_probabilities' in model_types_section:
                flat_config['model_probabilities'] = model_types_section['selection_probabilities']
        
        # Extract model parameters
        if 'model_parameters' in config:
            model_params = config['model_parameters']
            flat_config['model_params'] = flat_config.get('model_params', {})
            
            # Extract parameters for each model type
            for model_type in ['layered', 'fault', 'dome', 'perlin']:
                if model_type in model_params:
                    # Map parameter names from improved format to original format
                    type_params = model_params[model_type]
                    mapped_params = {}
                    
                    # Layered model parameters
                    if model_type == 'layered':
                        if 'layer_count_range' in type_params:
                            mapped_params['n_layers_range'] = type_params['layer_count_range']
                        if 'velocity_range' in type_params:
                            mapped_params['v_range'] = type_params['velocity_range']
                        if 'max_angle' in type_params:
                            mapped_params['angle'] = type_params['max_angle']
                    
                    # Fault model parameters
                    elif model_type == 'fault':
                        if 'base_velocity_range' in type_params:
                            mapped_params['v_range'] = type_params['base_velocity_range']
                        if 'velocity_difference_range' in type_params:
                            mapped_params['delta_v_range'] = type_params['velocity_difference_range']
                        if 'slope_range' in type_params:
                            mapped_params['slope_range'] = type_params['slope_range']
                        if 'offset_range' in type_params:
                            mapped_params['offset_range'] = type_params['offset_range']
                        if 'smoothing_sigma' in type_params:
                            mapped_params['sigma'] = type_params['smoothing_sigma']
                    
                    # Dome model parameters
                    elif model_type == 'dome':
                        if 'center_position_range' in type_params:
                            mapped_params['center_range'] = type_params['center_position_range']
                        if 'radius_range' in type_params:
                            mapped_params['radius_range'] = type_params['radius_range']
                        if 'high_velocity_range' in type_params:
                            mapped_params['v_high_range'] = type_params['high_velocity_range']
                        if 'low_velocity_range' in type_params:
                            mapped_params['v_low_range'] = type_params['low_velocity_range']
                        if 'smoothing_sigma' in type_params:
                            mapped_params['sigma'] = type_params['smoothing_sigma']
                    
                    # Perlin model parameters
                    elif model_type == 'perlin':
                        if 'noise_scale_range' in type_params:
                            mapped_params['scale_range'] = type_params['noise_scale_range']
                        if 'velocity_range' in type_params:
                            mapped_params['v_range'] = type_params['velocity_range']
                        if 'contrast_range' in type_params:
                            mapped_params['contrast_range'] = type_params['contrast_range']
                        if 'octave_count_range' in type_params:
                            mapped_params['n_level_range'] = type_params['octave_count_range']
                        if 'smoothing_sigma' in type_params:
                            mapped_params['sigma'] = type_params['smoothing_sigma']
                    
                    flat_config['model_params'][model_type] = mapped_params
        
        # Extract transformations
        if 'transformations' in config:
            transformations = config['transformations']
            if 'available_transforms' in transformations:
                flat_config['transforms'] = transformations['available_transforms']
            
            # Extract transformation probabilities
            if 'application_probabilities' in transformations:
                flat_config['transform_probabilities'] = transformations['application_probabilities']
            
            # Extract external transformation parameters
            if 'external' in transformations:
                external = transformations['external']
                if 'skimage_probability' in external:
                    flat_config['skimage_transform_prob'] = external['skimage_probability']
                if 'skimage_crop_ratio' in external:
                    flat_config['skimage_transform_crop_ratio'] = external['skimage_crop_ratio']
        
        # Extract transformation parameters
        if 'transformation_parameters' in config:
            transform_params = config['transformation_parameters']
            flat_config['transform_params'] = {}
            
            # Extract parameters for each transformation type
            for transform_type, params in transform_params.items():
                # Map parameter names from improved format to original format
                mapped_params = {}
                
                # Rotation parameters
                if transform_type == 'rotate':
                    if 'angle_range' in params:
                        mapped_params['angle_range'] = params['angle_range']
                
                # Stretch parameters
                elif transform_type == 'stretch':
                    if 'scale_range' in params:
                        mapped_params['scale_range'] = params['scale_range']
                
                # Inclusion parameters
                elif transform_type == 'inclusion':
                    if 'radius_range' in params:
                        mapped_params['radius_range'] = params['radius_range']
                    if 'velocity_range' in params:
                        mapped_params['velocity_range'] = params['velocity_range']
                    if 'inclusion_count_range' in params:
                        mapped_params['n_inclusions_range'] = params['inclusion_count_range']
                
                # Perlin noise parameters
                elif transform_type == 'perlin_noise':
                    if 'scale_range' in params:
                        mapped_params['scale_range'] = params['scale_range']
                    if 'amplitude' in params:
                        mapped_params['amplitude'] = params['amplitude']
                    if 'octave_levels' in params:
                        mapped_params['n_levels'] = params['octave_levels']
                
                # Crossline parameters
                elif transform_type == 'add_crossline':
                    if 'thickness_range' in params:
                        mapped_params['thickness_range'] = params['thickness_range']
                    if 'velocity_range' in params:
                        mapped_params['velocity_range'] = params['velocity_range']
                
                # Swirl parameters
                elif transform_type == 'swirl':
                    if 'center_range' in params:
                        mapped_params['center_range'] = params['center_range']
                    if 'strength_range' in params:
                        mapped_params['strength_range'] = params['strength_range']
                    if 'radius_range' in params:
                        mapped_params['radius_range'] = params['radius_range']
                
                # Piecewise affine parameters
                elif transform_type == 'piecewise_affine':
                    if 'grid_shape' in params:
                        mapped_params['grid_shape'] = params['grid_shape']
                    if 'displacement_sigma' in params:
                        mapped_params['displacement_sigma'] = params['displacement_sigma']
                
                # Displacement field parameters
                elif transform_type == 'displacement_field':
                    if 'max_displacement' in params:
                        mapped_params['max_displacement'] = params['max_displacement']
                    if 'smoothing_sigma' in params:
                        mapped_params['smooth_sigma'] = params['smoothing_sigma']
                
                flat_config['transform_params'][transform_type] = mapped_params
        
        # Extract validation parameters
        if 'validation' in config:
            validation = config['validation']
            flat_config['validation'] = {}
            
            # Extract velocity constraints
            if 'velocity' in validation:
                velocity = validation['velocity']
                if 'min_velocity' in velocity:
                    flat_config['validation']['min_v'] = velocity['min_velocity']
                if 'max_velocity' in velocity:
                    flat_config['validation']['max_v'] = velocity['max_velocity']
            
            # Extract quality thresholds
            if 'quality' in validation:
                quality = validation['quality']
                if 'zero_threshold' in quality:
                    flat_config['validation']['zero_thresh'] = quality['zero_threshold']
                if 'uniqueness_threshold' in quality:
                    flat_config['validation']['unique_thresh'] = quality['uniqueness_threshold']
                if 'entropy_threshold' in quality:
                    flat_config['validation']['entropy_thresh'] = quality['entropy_threshold']
        
        return flat_config
    
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
        
        return cls(data)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration dictionary.
        
        Args:
            config: Dictionary containing configuration parameters.
            
        Raises:
            ValueError: If required parameters are missing or have invalid values.
        """
        # Helper function to check if a parameter exists, considering alternative keys
        def param_exists(param_name):
            if param_name in config:
                return True
            if param_name in self.ALTERNATIVE_KEYS:
                return any(alt_name in config for alt_name in self.ALTERNATIVE_KEYS[param_name])
            return False
        
        # Helper function to get a parameter value, considering alternative keys
        def get_param(param_name, default=None):
            if param_name in config:
                return config[param_name]
            if param_name in self.ALTERNATIVE_KEYS:
                for alt_name in self.ALTERNATIVE_KEYS[param_name]:
                    if alt_name in config:
                        return config[alt_name]
            return default
        
        # Check for required parameters
        missing_params = [param for param in self.REQUIRED_PARAMS if not param_exists(param)]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        # Validate n_models
        n_models = get_param("n_models")
        if not isinstance(n_models, int) or n_models <= 0:
            raise ValueError("Parameter 'n_models' must be a positive integer.")
        
        # Validate model_params contains required grid parameters
        model_params = get_param("model_params")
        if not isinstance(model_params, dict):
            raise ValueError("'model_params' must be a dictionary.")
            
        # Check for required grid parameters in model_params
        for param in ["nx", "nz", "dx", "dz"]:
            if param not in model_params:
                raise ValueError(f"Missing required parameter '{param}' in model_params.")
            value = model_params[param]
            if param in ["nx", "nz"]:
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"Parameter 'model_params.{param}' must be a positive integer.")
            else:  # dx, dz
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(f"Parameter 'model_params.{param}' must be a positive number.")
        
        # Validate validation section
        validation = get_param("validation")
        if not isinstance(validation, dict):
            raise ValueError("'validation' must be a dictionary.")
            
        # Check for required validation parameters
        for param in ["min_v", "max_v"]:
            if param not in validation:
                raise ValueError(f"Missing required parameter '{param}' in validation.")
            value = validation[param]
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"Parameter 'validation.{param}' must be a positive number.")
        
        # Validate model types and probabilities
        model_types = get_param("model_types", [])
        if not isinstance(model_types, list) or not model_types:
            raise ValueError("'model_types' must be a non-empty list.")
        
        model_probabilities = get_param("model_probabilities", {})
        if not isinstance(model_probabilities, dict):
            raise ValueError("'model_probabilities' must be a dictionary.")
        
        # Check that all model types have probabilities
        missing_probs = [model for model in model_types if model not in model_probabilities]
        if missing_probs:
            raise ValueError(f"Missing probabilities for model types: {', '.join(missing_probs)}")
        
        # Validate model parameters
        model_params = get_param("model_params", {})
        if not isinstance(model_params, dict):
            raise ValueError("'model_params' must be a dictionary.")
        
        # Verify that model_params contains entries for each model type
        for model_type in model_types:
            if model_type not in model_params:
                raise ValueError(f"Missing model parameters for model type: {model_type}")
        
        # Check dimensionality if provided
        dim = get_param("dimensionality", "2D")  # Default to 2D if not specified
        if dim not in ["1D", "2D", "3D"]:
            raise ValueError(f"Invalid dimensionality: {dim}. Must be one of: 1D, 2D, 3D.")
        
        # For 3D models, ny and dy are required
        if dim == "3D":
            ny = get_param("ny")
            dy = get_param("dy")
            if ny is None or dy is None:
                raise ValueError("3D models require 'ny' and 'dy' parameters.")
            if not isinstance(ny, int) or ny <= 0:
                raise ValueError("Parameter 'ny' must be a positive integer.")
            if not isinstance(dy, (int, float)) or dy <= 0:
                raise ValueError("Parameter 'dy' must be a positive number.")
        
        # Validate transforms and transform_probabilities if provided
        transforms = get_param("transforms", [])
            
        if transforms:
            if not isinstance(transforms, list):
                raise ValueError("'transforms' must be a list.")
            
            # Check transform_probabilities if transforms is provided
            transform_probabilities = get_param("transform_probabilities")
                
            if transform_probabilities is None:
                raise ValueError("'transform_probabilities' is required when 'transforms' is provided.")
            
            if not isinstance(transform_probabilities, dict):
                raise ValueError("'transform_probabilities' must be a dictionary.")
            
            # Check that all transforms have probabilities
            missing_transform_probs = [t for t in transforms if t not in transform_probabilities]
            if missing_transform_probs:
                raise ValueError(f"Missing probabilities for transform types: {', '.join(missing_transform_probs)}")
            
            # Check transform_params if transforms is provided
            transform_params = get_param("transform_params")
            if transform_params is None:
                raise ValueError("'transform_params' is required when 'transforms' is provided.")
            
            if not isinstance(transform_params, dict):
                raise ValueError("'transform_params' must be a dictionary.")
            
            # Check that all transforms have parameters
            for transform_type in transforms:
                if transform_type not in transform_params:
                    raise ValueError(f"Missing parameters for transform type: {transform_type}")
    
    def get_dimensionality(self) -> Dimensionality:
        """
        Get the dimensionality as a Dimensionality enum.
        
        Returns:
            Dimensionality: The dimensionality enum value.
        """
        dim_str = self._config["dimensionality"]
        if dim_str == "1D":
            return Dimensionality.DIM_1D
        elif dim_str == "2D":
            return Dimensionality.DIM_2D
        else:  # "3D"
            return Dimensionality.DIM_3D
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get a parameter value from the configuration.
        
        Args:
            name: Name of the parameter. Can use dot notation for nested parameters (e.g., "validation.min_v").
            default: Default value to return if parameter is not found.
            
        Returns:
            The parameter value or default if not found.
        """
        # Check if this is a nested parameter (contains dots)
        if "." in name:
            parts = name.split(".")
            current = self._config
            
            # Navigate through the nested structure
            for i, part in enumerate(parts):
                # For the first level, check alternative keys
                if i == 0 and part in self.ALTERNATIVE_KEYS:
                    for alt_name in self.ALTERNATIVE_KEYS[part]:
                        if alt_name in current:
                            current = current[alt_name]
                            break
                    else:  # If no alternative key was found
                        if part in current:
                            current = current[part]
                        else:
                            return default
                else:
                    # For deeper levels or if no alternative keys
                    if part in current:
                        current = current[part]
                    else:
                        return default
            
            return current
        
        # Handle special cases for backward compatibility
        if name == "nx" or name == "nz" or name == "dx" or name == "dz":
            model_params = self.get_parameter("model_params", {})
            return model_params.get(name, default)
            
        if name == "min_v" or name == "max_v" or name == "zero_thresh" or name == "unique_thresh" or name == "entropy_thresh":
            validation = self.get_parameter("validation", {})
            return validation.get(name, default)
        
        # First try to get the parameter with the given name
        if name in self._config:
            return self._config[name]
        
        # If not found, check for alternative key names
        if name in self.ALTERNATIVE_KEYS:
            for alt_name in self.ALTERNATIVE_KEYS[name]:
                if alt_name in self._config:
                    return self._config[alt_name]
        
        # If still not found, return the default value
        return default
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """
        Get all parameters as a dictionary.
        
        Returns:
            Dict[str, Any]: All configuration parameters.
        """
        return self._config.copy()
    
    def __str__(self) -> str:
        """
        Return a string representation of the configuration.
        
        Returns:
            str: A human-readable string describing the configuration.
        """
        dim = self.get_parameter("dimensionality")
        n_models = self.get_parameter("n_models")
        model_params = self.get_parameter("model_params", {})
        nx = model_params.get("nx")
        nz = model_params.get("nz")
        
        result = f"ModelGenerationConfig({dim}, {n_models} models, grid: {nx}x{nz}"
        
        if dim == "3D":
            ny = self.get_parameter("ny")
            result = f"ModelGenerationConfig({dim}, {n_models} models, grid: {nx}x{ny}x{nz}"
        
        result += ")"
        return result