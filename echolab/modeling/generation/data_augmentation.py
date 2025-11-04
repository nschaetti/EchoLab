"""
Data augmentation utilities for velocity models.

This module provides classes and functions for applying various transformations
to velocity models for data augmentation purposes.
"""

# Imports
from typing import Tuple, List, Dict, Union, Optional, Any, Callable
import numpy as np
from skimage.transform import swirl, resize, warp, PiecewiseAffineTransform
from .layer_generation import generate_perlin_noise_2d
from skimage.util import img_as_float32
from skimage.transform import rotate
from scipy.ndimage import rotate as sci_rotate
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter


# region METHODS


def crop_and_resize(
        img: np.ndarray,
        target_shape: Tuple[int, int] = (70, 70),
        crop_ratio: float = 0.9
) -> np.ndarray:
    """
    Center-crop an image and resize it to the target shape.

    Args:
        img: 2D array representing the image
        target_shape: Final (height, width) of the output image
        crop_ratio: Zoom ratio (e.g., 0.9 to crop 10% from edges)

    Returns:
        Cropped and resized image
    """
    h, w = img.shape
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    start_h = (h - ch) // 2
    start_w = (w - cw) // 2
    cropped = img[start_h:start_h + ch, start_w:start_w + cw]
    return resize(cropped, target_shape, mode='reflect', anti_aliasing=True)
# end crop_and_resize



def apply_random_transformations(
        vel: np.ndarray,
        rng: np.random.Generator,
        apply_prob: float = 0.5,
        crop_ratio: float = 0.9
) -> np.ndarray:
    """
    Apply a series of random transformations to a velocity model.

    Args:
        vel: Input velocity model as a 2D array
        rng: Random number generator
        apply_prob: Probability of applying each transformation
        crop_ratio: Ratio for final cropping

    Returns:
        Transformed velocity model
    """
    transformed = img_as_float32(vel)

    # Light swirl
    if rng.random() < apply_prob:
        strength = rng.uniform(1, 3)
        radius = rng.uniform(100, 300)
        transformed = swirl(transformed, strength=strength, radius=radius)
    # end if

    # Resize (then back to original size)
    if rng.random() < apply_prob:
        factor = rng.uniform(0.8, 1.2)
        small = resize(transformed, (int(vel.shape[0] * factor), int(vel.shape[1] * factor)),
                      mode='reflect', anti_aliasing=True)
        transformed = resize(small, vel.shape, mode='reflect', anti_aliasing=True)
    # end if

    # Piecewise Affine
    if rng.random() < apply_prob:
        rows, cols = vel.shape
        src_cols = np.linspace(0, cols, 5)
        src_rows = np.linspace(0, rows, 5)
        src = np.dstack(np.meshgrid(src_cols, src_rows)).reshape(-1, 2)
        dst = src + rng.normal(scale=5.0, size=src.shape)  # small displacements

        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        transformed = warp(transformed, tform, output_shape=vel.shape)
    # end if

    # Rescale to original range
    transformed = transformed - transformed.min()
    transformed = transformed / transformed.max()
    transformed = transformed * (vel.max() - vel.min()) + vel.min()

    # Zoom to avoid edges and resize to original size
    transformed = crop_and_resize(transformed, target_shape=vel.shape, crop_ratio=crop_ratio)

    return transformed
# end apply_random_transformations


# endregion METHODS


# region TRANSFORMS


class RandomSwirl:
    """
    Apply random swirl transformation to an image.

    This transformation creates a swirling effect by rotating pixels around a center point,
    with the rotation angle decreasing with distance from the center.
    """

    def __init__(
            self,
            center_range: Tuple[float, float],
            strength_range: Tuple[float, float],
            radius_range: Tuple[float, float],
            rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize the RandomSwirl transformation.

        Args:
            center_range: Range for center coordinates
            strength_range: Range for swirl strength
            radius_range: Range for swirl radius
            rng: Random number generator
        """
        self.strength_range = strength_range
        self.radius_range = radius_range
        self.rng = np.random.default_rng() if rng is None else rng
    # end __init__

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the swirl transformation to an image.

        Args:
            image: Input image as a 2D array

        Returns:
            Transformed image
        """
        center_x = self.rng.uniform(-self.radius_range[0], self.radius_range[0])
        center_z = self.rng.uniform(-self.radius_range[0], self.radius_range[0])
        strength = self.rng.uniform(*self.strength_range)
        radius = self.rng.uniform(*self.radius_range)
        return swirl(image, center=(center_x, center_z), strength=strength, radius=radius)
    # end __call__

# end RandomSwirl


class RandomPiecewiseAffine:
    """
    Apply random piecewise affine transformation to an image.

    This transformation divides the image into a grid and applies random displacements
    to the grid points, creating local distortions in the image.
    """

    def __init__(
            self,
            grid_shape: Tuple[int, int] = (5, 5),
            displacement_sigma: float = 5.0,
            rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize the RandomPiecewiseAffine transformation.

        Args:
            grid_shape: Shape of the grid (rows, cols)
            displacement_sigma: Standard deviation of the random displacements
            rng: Random number generator
        """
        self.grid_shape = grid_shape
        self.displacement_sigma = displacement_sigma
        self.rng = np.random.default_rng() if rng is None else rng
    # end __init__

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the piecewise affine transformation to an image.

        Args:
            image: Input image as a 2D array

        Returns:
            Transformed image
        """
        rows, cols = image.shape
        src_cols = np.linspace(0, cols, self.grid_shape[1])
        src_rows = np.linspace(0, rows, self.grid_shape[0])
        src = np.dstack(np.meshgrid(src_cols, src_rows)).reshape(-1, 2)
        dst = src + self.rng.normal(scale=self.displacement_sigma, size=src.shape)

        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        return warp(image, tform, output_shape=image.shape)
    # end __call__

# end RandomPiecewiseAffine


class RandomDisplacementField:
    """
    Apply random displacement field transformation to an image.

    This transformation creates a smooth random displacement field and warps
    the image according to this field, creating natural-looking deformations.
    """

    def __init__(
            self,
            max_displacement: float,
            smooth_sigma: float,
            rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize the RandomDisplacementField transformation.

        Args:
            max_displacement: Maximum amplitude of displacements (in pixels)
            smooth_sigma: Standard deviation of Gaussian blur for smoothing the field
            rng: Random number generator
        """
        self.max_displacement = max_displacement
        self.smooth_sigma = smooth_sigma
        self.rng = np.random.default_rng() if rng is None else rng
    # end __init__

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the displacement field transformation to an image.

        Args:
            image: Input image as a 2D array

        Returns:
            Transformed image
        """
        h, w = image.shape

        # Random noise for dx and dy
        dx = self.rng.normal(0, 1, (h, w))
        dy = self.rng.normal(0, 1, (h, w))

        # Smoothing to make the field spatially coherent
        dx = gaussian_filter(dx, sigma=self.smooth_sigma)
        dy = gaussian_filter(dy, sigma=self.smooth_sigma)

        # Normalization and scaling
        dx = (dx / np.max(np.abs(dx))) * self.max_displacement
        dy = (dy / np.max(np.abs(dy))) * self.max_displacement

        # Deformed coordinate grid
        coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords_deformed = np.array([coords[0] + dx, coords[1] + dy])

        # Apply the deformation
        warped = warp(image, coords_deformed, mode='reflect')

        return warped
    # end __call__

# end RandomDisplacementField




class AddCrossLine:
    """
    Add a random cross line to a velocity model.

    This transformation adds a straight line with random thickness, velocity,
    position, and orientation to the velocity model.
    """

    def __init__(self, thickness_range: Tuple[int, int] = (2, 10),
                 velocity_range: Tuple[float, float] = (1000, 4000),
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize the AddCrossLine transformation.

        Args:
            thickness_range: Range for line thickness (min, max)
            velocity_range: Range for line velocity (min, max)
            rng: Random number generator
        """
        self.thickness_range = thickness_range
        self.velocity_range = velocity_range
        self.rng = np.random.default_rng() if rng is None else rng
    # end __init__

    def __call__(self, vel: np.ndarray) -> np.ndarray:
        """
        Add a cross line to the velocity model.

        Args:
            vel: Input velocity model as a 2D array

        Returns:
            Velocity model with added cross line
        """
        nz, nx = vel.shape
        line_img = np.copy(vel)

        # Line thickness and velocity
        thickness = self.rng.integers(*self.thickness_range)
        v_line = self.rng.uniform(*self.velocity_range)

        # Create a binary image with a horizontal line in the middle
        line_mask = np.zeros_like(vel, dtype=bool)
        center = self.rng.integers(thickness, nz - thickness)
        line_mask[center - thickness // 2 : center + thickness // 2, :] = True

        # Apply random rotation around the center of the image
        angle = self.rng.uniform(-45, 45)
        rotated = rotate(line_mask.astype(float), angle=angle, resize=False, order=1, mode='reflect', cval=0)

        # Apply the line to the model
        line_img[rotated > 0.5] = v_line

        return line_img
    # end __call__

# end AddCrossLine


class RandomTransformer:
    """
    Apply a sequence of transformations to a velocity model.

    This class allows chaining multiple transformations together and applying
    them in sequence to a velocity model.
    """

    def __init__(self, transforms: List[Callable[[np.ndarray], np.ndarray]]):
        """
        Initialize the RandomTransformer.

        Args:
            transforms: List of transformation functions or callable objects
        """
        self.transforms = transforms
    # end __init__

    def __call__(self, vel: np.ndarray) -> np.ndarray:
        """
        Apply all transformations in sequence.

        Args:
            vel: Input velocity model as a 2D array

        Returns:
            Transformed velocity model
        """
        for t in self.transforms:
            vel = t(vel)
        # end for
        return vel
    # end __call__

# end RandomTransformer


class RandomRotation:
    """
    Apply random rotation to a velocity model.

    This transformation rotates the velocity model by a random angle within
    the specified range.
    """

    def __init__(self, angle_range: Tuple[float, float] = (-10, 10),
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize the RandomRotation transformation.

        Args:
            angle_range: Range of rotation angles in degrees (min, max)
            rng: Random number generator
        """
        self.angle_range = angle_range
        self.rng = rng if rng is not None else np.random.default_rng()
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Apply random rotation to the model.

        Args:
            model: Input velocity model as a 2D array

        Returns:
            Rotated velocity model
        """
        angle = self.rng.uniform(*self.angle_range)
        return sci_rotate(model, angle, reshape=False, mode="nearest")
    # end __call__

# end RandomRotation


class RandomStretch:
    """
    Apply random stretching to a velocity model.

    This transformation scales the velocity model by a random factor within
    the specified range, and then crops or pads it to maintain the original size.
    """

    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1),
                 rng: Optional[np.random.Generator] = None):
        """
        Initialize the RandomStretch transformation.

        Args:
            scale_range: Range of scaling factors (min, max)
            rng: Random number generator
        """
        self.scale_range = scale_range
        self.rng = rng if rng is not None else np.random.default_rng()
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Apply random stretching to the model.

        Args:
            model: Input velocity model as a 2D array

        Returns:
            Stretched velocity model
        """
        scale = self.rng.uniform(*self.scale_range)
        stretched = zoom(model, scale, order=1)

        # Crop or pad to return to original size
        nz, nx = model.shape
        sz, sx = stretched.shape

        if sz >= nz and sx >= nx:
            stretched = stretched[:nz, :nx]
        else:
            padded = np.ones((nz, nx)) * np.mean(model)
            padded[:min(sz, nz), :min(sx, nx)] = stretched[:min(sz, nz), :min(sx, nx)]
            stretched = padded
        # end if

        return stretched
    # end __call__

# end RandomStretch


class AddInclusion:
    """
    Add random circular inclusions to a velocity model.

    This transformation adds a random number of circular regions with different
    velocities to the model, simulating geological inclusions.
    """

    def __init__(
            self,
            radius_range: Tuple[int, int] = (3, 10),
            velocity_range: Tuple[float, float] = (1000, 2500),
            n_inclusions_range: Tuple[int, int] = (1, 5),
            rng: Optional[np.random.Generator] = None):
        """
        Initialize the AddInclusion transformation.

        Args:
            radius_range: Range of inclusion radii (min, max)
            velocity_range: Range of inclusion velocities (min, max)
            n_inclusions_range: Range for number of inclusions to add (min, max)
            rng: Random number generator
        """
        self.radius_range = radius_range
        self.velocity_range = velocity_range
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_inclusions_range = n_inclusions_range
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Add random inclusions to the model.

        Args:
            model: Input velocity model as a 2D array

        Returns:
            Velocity model with added inclusions
        """
        nz, nx = model.shape
        n_inclusions = int(self.rng.uniform(*self.n_inclusions_range))
        for _ in range(n_inclusions):
            r = self.rng.integers(*self.radius_range)
            cx = self.rng.integers(r, nx - r)
            cz = self.rng.integers(r, nz - r)
            vel = self.rng.uniform(*self.velocity_range)
            yy, xx = np.ogrid[:nz, :nx]
            mask = (xx - cx)**2 + (yy - cz)**2 <= r**2
            model[mask] = vel
        # end for
        return model
    # end __call__

# end AddInclusion


class AddPerlinNoise:
    """
    Add Perlin noise to a velocity model.

    This transformation adds coherent noise based on Perlin noise algorithm
    to the velocity model, creating natural-looking variations.
    """

    def __init__(
            self,
            scale_range: Tuple[float, float],
            amplitude: float,
            n_levels: List[float],
            rng: Optional[np.random.Generator] = None):
        """
        Initialize the AddPerlinNoise transformation.

        Args:
            scale_range: Range for noise scale (min, max)
            amplitude: Amplitude of the noise
            n_levels: Discretization levels for the noise
            rng: Random number generator
        """
        self.rng = rng or np.random.default_rng()
        self.scale_range = scale_range
        self.n_levels = np.array(n_levels)
        self.amplitude = amplitude
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        """
        Add Perlin noise to the model.

        Args:
            model: Input velocity model as a 2D array

        Returns:
            Velocity model with added Perlin noise
        """
        scale = int(self.rng.uniform(*self.scale_range))

        nz, nx = model.shape

        # Resolution derived from global scale
        res = (
            max(1, int(nz / scale)),
            max(1, int(nx / scale))
        )

        noise = generate_perlin_noise_2d(model.shape, res=res, rng=self.rng) * 10.0
        noise = np.vectorize(lambda n: self.n_levels[np.argmin(np.abs(self.n_levels - n))])(noise)

        return model + self.amplitude * noise
    # end __call__

# end AddPerlinNoise


# endregion TRANSFORMS

