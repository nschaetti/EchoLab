"""
Data augmentation utilities for velocity models.

This module provides classes and functions for applying various transformations
to velocity models for data augmentation purposes.
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class RandomRotation:
    """
    Apply random rotation to a velocity model.
    
    Attributes:
        max_angle: Maximum rotation angle in degrees
        fill_value: Value to fill empty areas after rotation
    """
    
    def __init__(self, max_angle: float = 10.0, fill_value: Optional[float] = None):
        """
        Initialize the RandomRotation transformer.
        
        Args:
            max_angle: Maximum rotation angle in degrees
            fill_value: Value to fill empty areas after rotation (default: use edge values)
        """
        self.max_angle = max_angle
        self.fill_value = fill_value
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Apply random rotation to the model.
        
        Args:
            model: 2D numpy array representing the velocity model
            
        Returns:
            Transformed velocity model
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for RandomRotation. "
                             "Install it with 'pip install opencv-python'.")
        # end if
        
        # Generate random angle
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        
        # Get model dimensions
        height, width = model.shape
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Determine border mode and value
        if self.fill_value is None:
            border_mode = cv2.BORDER_REPLICATE
            border_value = 0
        else:
            border_mode = cv2.BORDER_CONSTANT
            border_value = self.fill_value
        # end if
        
        # Apply rotation
        rotated = cv2.warpAffine(
            model.astype(np.float32),
            rotation_matrix,
            (width, height),
            borderMode=border_mode,
            borderValue=border_value
        )
        
        return rotated
    # end def __call__
# end class RandomRotation


class RandomStretch:
    """
    Apply random stretching to a velocity model.
    
    Attributes:
        max_factor_x: Maximum stretch factor in x direction
        max_factor_z: Maximum stretch factor in z direction
        fill_value: Value to fill empty areas after stretching
    """
    
    def __init__(
        self,
        max_factor_x: float = 0.2,
        max_factor_z: float = 0.2,
        fill_value: Optional[float] = None
    ):
        """
        Initialize the RandomStretch transformer.
        
        Args:
            max_factor_x: Maximum stretch factor in x direction
            max_factor_z: Maximum stretch factor in z direction
            fill_value: Value to fill empty areas after stretching (default: use edge values)
        """
        self.max_factor_x = max_factor_x
        self.max_factor_z = max_factor_z
        self.fill_value = fill_value
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Apply random stretching to the model.
        
        Args:
            model: 2D numpy array representing the velocity model
            
        Returns:
            Transformed velocity model
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for RandomStretch. "
                             "Install it with 'pip install opencv-python'.")
        # end if
        
        # Generate random stretch factors
        factor_x = 1.0 + np.random.uniform(-self.max_factor_x, self.max_factor_x)
        factor_z = 1.0 + np.random.uniform(-self.max_factor_z, self.max_factor_z)
        
        # Get model dimensions
        height, width = model.shape
        
        # Determine new dimensions
        new_width = int(width * factor_x)
        new_height = int(height * factor_z)
        
        # Determine border mode and value
        if self.fill_value is None:
            border_mode = cv2.BORDER_REPLICATE
            border_value = 0
        else:
            border_mode = cv2.BORDER_CONSTANT
            border_value = self.fill_value
        # end if
        
        # Apply stretching
        stretched = cv2.resize(
            model.astype(np.float32),
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Crop or pad to original size
        if new_width > width:
            # Crop horizontally
            start_x = (new_width - width) // 2
            stretched = stretched[:, start_x:start_x + width]
        elif new_width < width:
            # Pad horizontally
            pad_x = (width - new_width) // 2
            stretched = cv2.copyMakeBorder(
                stretched,
                0, 0, pad_x, width - new_width - pad_x,
                borderType=border_mode,
                value=border_value
            )
        # end if
        
        if new_height > height:
            # Crop vertically
            start_y = (new_height - height) // 2
            stretched = stretched[start_y:start_y + height, :]
        elif new_height < height:
            # Pad vertically
            pad_y = (height - new_height) // 2
            stretched = cv2.copyMakeBorder(
                stretched,
                pad_y, height - new_height - pad_y, 0, 0,
                borderType=border_mode,
                value=border_value
            )
        # end if
        
        return stretched
    # end def __call__
# end class RandomStretch


class AddInclusion:
    """
    Add random inclusion shapes to a velocity model.
    
    Attributes:
        num_inclusions: Number of inclusions to add
        min_radius: Minimum radius of inclusions
        max_radius: Maximum radius of inclusions
        min_velocity: Minimum velocity value for inclusions
        max_velocity: Maximum velocity value for inclusions
    """
    
    def __init__(
        self,
        num_inclusions: int = 3,
        min_radius: int = 5,
        max_radius: int = 15,
        min_velocity: Optional[float] = None,
        max_velocity: Optional[float] = None
    ):
        """
        Initialize the AddInclusion transformer.
        
        Args:
            num_inclusions: Number of inclusions to add
            min_radius: Minimum radius of inclusions
            max_radius: Maximum radius of inclusions
            min_velocity: Minimum velocity value for inclusions (default: model min)
            max_velocity: Maximum velocity value for inclusions (default: model max)
        """
        self.num_inclusions = num_inclusions
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Add random inclusions to the model.
        
        Args:
            model: 2D numpy array representing the velocity model
            
        Returns:
            Transformed velocity model
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for AddInclusion. "
                             "Install it with 'pip install opencv-python'.")
        # end if
        
        # Get model dimensions
        height, width = model.shape
        
        # Determine velocity range for inclusions
        min_vel = self.min_velocity if self.min_velocity is not None else np.min(model)
        max_vel = self.max_velocity if self.max_velocity is not None else np.max(model)
        
        # Create a copy of the model
        result = model.copy()
        
        # Add inclusions
        for _ in range(self.num_inclusions):
            # Random center position
            center_x = np.random.randint(0, width)
            center_y = np.random.randint(0, height)
            
            # Random radius
            radius = np.random.randint(self.min_radius, self.max_radius + 1)
            
            # Random velocity
            velocity = np.random.uniform(min_vel, max_vel)
            
            # Draw circle
            cv2.circle(
                result,
                (center_x, center_y),
                radius,
                velocity,
                -1  # Fill the circle
            )
        # end for
        
        return result
    # end def __call__
# end class AddInclusion


class AddPerlinNoise:
    """
    Add Perlin noise to a velocity model.
    
    Attributes:
        amplitude: Maximum amplitude of the noise relative to model range
        scale: Scale of the Perlin noise
        octaves: Number of octaves for Perlin noise
    """
    
    def __init__(self, amplitude: float = 0.1, scale: float = 10.0, octaves: int = 4):
        """
        Initialize the AddPerlinNoise transformer.
        
        Args:
            amplitude: Maximum amplitude of the noise relative to model range
            scale: Scale of the Perlin noise
            octaves: Number of octaves for Perlin noise
        """
        self.amplitude = amplitude
        self.scale = scale
        self.octaves = octaves
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Add Perlin noise to the model.
        
        Args:
            model: 2D numpy array representing the velocity model
            
        Returns:
            Transformed velocity model
        """
        try:
            from noise import pnoise2
        except ImportError:
            raise ImportError("The 'noise' package is required for Perlin noise. "
                             "Install it with 'pip install noise'.")
        # end if
        
        # Get model dimensions
        height, width = model.shape
        
        # Calculate model range for scaling noise
        model_min = np.min(model)
        model_max = np.max(model)
        model_range = model_max - model_min
        
        # Generate Perlin noise
        noise = np.zeros_like(model)
        for i in range(height):
            for j in range(width):
                noise[i, j] = pnoise2(
                    i / self.scale,
                    j / self.scale,
                    octaves=self.octaves,
                    persistence=0.5,
                    lacunarity=2.0,
                    repeatx=width,
                    repeaty=height,
                    base=0
                )
            # end for
        # end for
        
        # Normalize noise to [-1, 1]
        noise = 2.0 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise)) - 1.0
        
        # Scale noise by amplitude and model range
        scaled_noise = noise * self.amplitude * model_range
        
        # Add noise to model
        result = model + scaled_noise
        
        # Ensure result is within original range
        result = np.clip(result, model_min, model_max)
        
        return result
    # end def __call__
# end class AddPerlinNoise


class AddCrossLine:
    """
    Add random cross-cutting lines to a velocity model.
    
    Attributes:
        num_lines: Number of lines to add
        min_width: Minimum width of lines
        max_width: Maximum width of lines
        min_velocity: Minimum velocity value for lines
        max_velocity: Maximum velocity value for lines
    """
    
    def __init__(
        self,
        num_lines: int = 2,
        min_width: int = 1,
        max_width: int = 5,
        min_velocity: Optional[float] = None,
        max_velocity: Optional[float] = None
    ):
        """
        Initialize the AddCrossLine transformer.
        
        Args:
            num_lines: Number of lines to add
            min_width: Minimum width of lines
            max_width: Maximum width of lines
            min_velocity: Minimum velocity value for lines (default: model min)
            max_velocity: Maximum velocity value for lines (default: model max)
        """
        self.num_lines = num_lines
        self.min_width = min_width
        self.max_width = max_width
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Add random cross-cutting lines to the model.
        
        Args:
            model: 2D numpy array representing the velocity model
            
        Returns:
            Transformed velocity model
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for AddCrossLine. "
                             "Install it with 'pip install opencv-python'.")
        # end if
        
        # Get model dimensions
        height, width = model.shape
        
        # Determine velocity range for lines
        min_vel = self.min_velocity if self.min_velocity is not None else np.min(model)
        max_vel = self.max_velocity if self.max_velocity is not None else np.max(model)
        
        # Create a copy of the model
        result = model.copy()
        
        # Add lines
        for _ in range(self.num_lines):
            # Random start and end points
            start_x = np.random.randint(0, width)
            start_y = np.random.randint(0, height)
            end_x = np.random.randint(0, width)
            end_y = np.random.randint(0, height)
            
            # Random width
            line_width = np.random.randint(self.min_width, self.max_width + 1)
            
            # Random velocity
            velocity = np.random.uniform(min_vel, max_vel)
            
            # Draw line
            cv2.line(
                result,
                (start_x, start_y),
                (end_x, end_y),
                velocity,
                line_width
            )
        # end for
        
        return result
    # end def __call__
# end class AddCrossLine


class RandomSwirl:
    """
    Apply random swirl effect to a velocity model.
    
    Attributes:
        max_strength: Maximum strength of the swirl effect
        radius: Radius of the swirl effect
    """
    
    def __init__(self, max_strength: float = 10.0, radius: Optional[int] = None):
        """
        Initialize the RandomSwirl transformer.
        
        Args:
            max_strength: Maximum strength of the swirl effect
            radius: Radius of the swirl effect (default: half of min dimension)
        """
        self.max_strength = max_strength
        self.radius = radius
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Apply random swirl effect to the model.
        
        Args:
            model: 2D numpy array representing the velocity model
            
        Returns:
            Transformed velocity model
        """
        try:
            from skimage.transform import swirl
        except ImportError:
            raise ImportError("scikit-image is required for RandomSwirl. "
                             "Install it with 'pip install scikit-image'.")
        # end if
        
        # Get model dimensions
        height, width = model.shape
        
        # Determine radius if not provided
        radius = self.radius if self.radius is not None else min(height, width) // 2
        
        # Random strength
        strength = np.random.uniform(-self.max_strength, self.max_strength)
        
        # Random center
        center = (
            np.random.randint(radius, width - radius) if width > 2 * radius else width // 2,
            np.random.randint(radius, height - radius) if height > 2 * radius else height // 2
        )
        
        # Apply swirl
        result = swirl(
            model,
            center=center,
            strength=strength,
            radius=radius,
            mode='reflect'
        )
        
        return result
    # end def __call__
# end class RandomSwirl


class RandomPiecewiseAffine:
    """
    Apply random piecewise affine transformation to a velocity model.
    
    Attributes:
        num_points: Number of control points in each dimension
        max_displacement: Maximum displacement of control points
    """
    
    def __init__(self, num_points: int = 4, max_displacement: float = 0.1):
        """
        Initialize the RandomPiecewiseAffine transformer.
        
        Args:
            num_points: Number of control points in each dimension
            max_displacement: Maximum displacement of control points as fraction of image size
        """
        self.num_points = num_points
        self.max_displacement = max_displacement
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Apply random piecewise affine transformation to the model.
        
        Args:
            model: 2D numpy array representing the velocity model
            
        Returns:
            Transformed velocity model
        """
        try:
            from skimage.transform import PiecewiseAffineTransform, warp
        except ImportError:
            raise ImportError("scikit-image is required for RandomPiecewiseAffine. "
                             "Install it with 'pip install scikit-image'.")
        # end if
        
        # Get model dimensions
        height, width = model.shape
        
        # Create grid of control points
        rows = np.linspace(0, height - 1, self.num_points)
        cols = np.linspace(0, width - 1, self.num_points)
        src_rows, src_cols = np.meshgrid(rows, cols, indexing='ij')
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        
        # Create destination control points with random displacements
        dst_rows = src_rows + np.random.uniform(
            -self.max_displacement * height,
            self.max_displacement * height,
            src_rows.shape
        )
        dst_cols = src_cols + np.random.uniform(
            -self.max_displacement * width,
            self.max_displacement * width,
            src_cols.shape
        )
        dst = np.dstack([dst_cols.flat, dst_rows.flat])[0]
        
        # Create transformation
        transform = PiecewiseAffineTransform()
        transform.estimate(src, dst)
        
        # Apply transformation
        result = warp(model, transform, mode='reflect')
        
        # Rescale to original range
        result = result * (np.max(model) - np.min(model)) + np.min(model)
        
        return result
    # end def __call__
# end class RandomPiecewiseAffine


class RandomDisplacementField:
    """
    Apply random displacement field to a velocity model.
    
    Attributes:
        alpha: Maximum displacement as fraction of image size
        sigma: Standard deviation of the Gaussian filter
    """
    
    def __init__(self, alpha: float = 0.1, sigma: float = 4.0):
        """
        Initialize the RandomDisplacementField transformer.
        
        Args:
            alpha: Maximum displacement as fraction of image size
            sigma: Standard deviation of the Gaussian filter
        """
        self.alpha = alpha
        self.sigma = sigma
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Apply random displacement field to the model.
        
        Args:
            model: 2D numpy array representing the velocity model
            
        Returns:
            Transformed velocity model
        """
        try:
            from scipy.ndimage import gaussian_filter
            from skimage.transform import warp
        except ImportError:
            raise ImportError("scipy and scikit-image are required for RandomDisplacementField. "
                             "Install them with 'pip install scipy scikit-image'.")
        # end if
        
        # Get model dimensions
        height, width = model.shape
        
        # Create random displacement fields
        dx = gaussian_filter(
            (np.random.rand(height, width) * 2 - 1),
            self.sigma, mode="constant", cval=0
        ) * self.alpha * width
        dy = gaussian_filter(
            (np.random.rand(height, width) * 2 - 1),
            self.sigma, mode="constant", cval=0
        ) * self.alpha * height
        
        # Create coordinate grid
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Displace coordinates
        coordinates = np.stack([y + dy, x + dx], axis=0)
        
        # Apply transformation
        result = warp(model, coordinates, mode='reflect')
        
        # Rescale to original range
        result = result * (np.max(model) - np.min(model)) + np.min(model)
        
        return result
    # end def __call__
# end class RandomDisplacementField


def apply_random_transformations(
    model: np.ndarray,
    transformations: List[Callable[[np.ndarray], np.ndarray]],
    num_transforms: int = 2
) -> np.ndarray:
    """
    Apply a random subset of transformations to a velocity model.
    
    Args:
        model: 2D numpy array representing the velocity model
        transformations: List of transformation callables
        num_transforms: Number of transformations to apply
        
    Returns:
        Transformed velocity model
    """
    # Create a copy of the model
    result = model.copy()
    
    # Randomly select transformations
    if num_transforms > len(transformations):
        num_transforms = len(transformations)
    # end if
    
    selected_transforms = np.random.choice(
        transformations,
        size=num_transforms,
        replace=False
    )
    
    # Apply selected transformations
    for transform in selected_transforms:
        result = transform(result)
    # end for
    
    return result
# end def apply_random_transformations


class RandomTransformer:
    """
    Apply random transformations to velocity models.
    
    This class provides a convenient way to apply multiple random transformations
    to velocity models with configurable probabilities.
    
    Attributes:
        transformations: Dictionary mapping transformation classes to their probabilities
        max_transforms: Maximum number of transformations to apply
    """
    
    def __init__(
        self,
        transformations: Dict[Callable[[np.ndarray], np.ndarray], float],
        max_transforms: int = 3
    ):
        """
        Initialize the RandomTransformer.
        
        Args:
            transformations: Dictionary mapping transformation classes to their probabilities
            max_transforms: Maximum number of transformations to apply
        """
        self.transformations = transformations
        self.max_transforms = max_transforms
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Apply random transformations to the model.
        
        Args:
            model: 2D numpy array representing the velocity model
            
        Returns:
            Transformed velocity model
        """
        # Create a copy of the model
        result = model.copy()
        
        # Determine which transformations to apply
        transforms_to_apply = []
        for transform, probability in self.transformations.items():
            if np.random.random() < probability:
                transforms_to_apply.append(transform)
            # end if
        # end for
        
        # Limit number of transformations
        if len(transforms_to_apply) > self.max_transforms:
            transforms_to_apply = np.random.choice(
                transforms_to_apply,
                size=self.max_transforms,
                replace=False
            )
        # end if
        
        # Apply transformations
        for transform in transforms_to_apply:
            result = transform(result)
        # end for
        
        return result
    # end def __call__
# end class RandomTransformer


def crop_and_resize(
    model: np.ndarray,
    crop_size: Tuple[int, int],
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Crop a random region from a velocity model and resize it.
    
    Args:
        model: 2D numpy array representing the velocity model
        crop_size: Size of the region to crop (height, width)
        output_size: Size to resize the cropped region to (default: original model size)
        
    Returns:
        Transformed velocity model
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for crop_and_resize. "
                         "Install it with 'pip install opencv-python'.")
    # end if
    
    # Get model dimensions
    height, width = model.shape
    crop_height, crop_width = crop_size
    
    # Ensure crop size is not larger than model
    crop_height = min(crop_height, height)
    crop_width = min(crop_width, width)
    
    # Random crop position
    top = np.random.randint(0, height - crop_height + 1)
    left = np.random.randint(0, width - crop_width + 1)
    
    # Crop the model
    cropped = model[top:top + crop_height, left:left + crop_width]
    
    # Resize if output_size is provided
    if output_size is not None:
        resized = cv2.resize(
            cropped.astype(np.float32),
            (output_size[1], output_size[0]),
            interpolation=cv2.INTER_LINEAR
        )
        return resized
    # end if
    
    return cropped
# end def crop_and_resize


# Additional transformations for convenience
class RandomStretchZoom:
    """Alias for RandomStretch with different default parameters."""
    def __init__(self, max_factor: float = 0.2, fill_value: Optional[float] = None):
        self.transformer = RandomStretch(max_factor, max_factor, fill_value)
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        return self.transformer(model)
    # end def __call__
# end class RandomStretchZoom


class RandomDisplacement:
    """Alias for RandomDisplacementField with different default parameters."""
    def __init__(self, strength: float = 0.05):
        self.transformer = RandomDisplacementField(alpha=strength)
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        return self.transformer(model)
    # end def __call__
# end class RandomDisplacement


class RandomNoise:
    """Add random Gaussian noise to a velocity model."""
    def __init__(self, std: float = 0.05):
        self.std = std
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.std * (np.max(model) - np.min(model)), model.shape)
        result = model + noise
        return np.clip(result, np.min(model), np.max(model))
    # end def __call__
# end class RandomNoise


class RandomGaussianBlur:
    """Apply random Gaussian blur to a velocity model."""
    def __init__(self, max_sigma: float = 2.0):
        self.max_sigma = max_sigma
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            raise ImportError("scipy is required for RandomGaussianBlur. "
                             "Install it with 'pip install scipy'.")
        # end if
        
        sigma = np.random.uniform(0, self.max_sigma)
        return gaussian_filter(model, sigma=sigma)
    # end def __call__
# end class RandomGaussianBlur


class RandomBrightnessContrast:
    """Apply random brightness and contrast adjustments to a velocity model."""
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1):
        self.brightness = brightness
        self.contrast = contrast
    # end def __init__
    
    def __call__(self, model: np.ndarray) -> np.ndarray:
        # Normalize to [0, 1]
        model_min = np.min(model)
        model_max = np.max(model)
        model_range = model_max - model_min
        normalized = (model - model_min) / model_range
        
        # Apply brightness adjustment
        brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        adjusted = normalized * brightness_factor
        
        # Apply contrast adjustment
        contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        mean = np.mean(adjusted)
        adjusted = (adjusted - mean) * contrast_factor + mean
        
        # Clip to [0, 1] and rescale to original range
        adjusted = np.clip(adjusted, 0, 1)
        result = adjusted * model_range + model_min
        
        return result
    # end def __call__
# end class RandomBrightnessContrast