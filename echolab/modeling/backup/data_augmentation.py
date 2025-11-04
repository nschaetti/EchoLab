"""
Data augmentation utilities for velocity models.

This module provides various dataset augmentation techniques for velocity
models used in seismic imaging. These include geometric transformations
(rotation, stretching, swirling), addition of features (inclusions,
cross-lines), and noise-based transformations (Perlin noise, displacement
fields).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage
from perlin_noise import PerlinNoise
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate as sci_rotate
from scipy.ndimage import zoom
from skimage import img_as_float32
from skimage.transform import PiecewiseAffineTransform, resize, rotate, swirl, warp
from skimage.util import random_noise

from .layer_generation import generate_perlin_noise_2d


def crop_and_resize(
    img: np.ndarray, target_shape: Tuple[int, int] = (70, 70), crop_ratio: float = 0.9
) -> np.ndarray:
    """
    Center-crop an image and resize it to the target shape.
    """
    h, w = img.shape
    ch, cw = int(h * crop_ratio), int(w * crop_ratio)
    start_h = (h - ch) // 2
    start_w = (w - cw) // 2
    cropped = img[start_h : start_h + ch, start_w : start_w + cw]
    return resize(cropped, target_shape, mode="reflect", anti_aliasing=True)
# end crop_and_resize


class RandomSwirl:
    """
    Apply random swirl transformation to an image.
    """

    def __init__(
        self,
        center_range: Tuple[float, float],
        strength_range: Tuple[float, float],
        radius_range: Tuple[float, float],
        rng: Optional[np.random.Generator] = None,
    ):
        self.strength_range = strength_range
        self.radius_range = radius_range
        self.rng = np.random.default_rng() if rng is None else rng
    # end __init__

    def __call__(self, image: np.ndarray) -> np.ndarray:
        center_x = self.rng.uniform(-self.radius_range[0], self.radius_range[0])
        center_z = self.rng.uniform(-self.radius_range[0], self.radius_range[0])
        strength = self.rng.uniform(*self.strength_range)
        radius = self.rng.uniform(*self.radius_range)
        return swirl(image, center=(center_x, center_z), strength=strength, radius=radius)
    # end __call__


class RandomPiecewiseAffine:
    """
    Apply random piecewise affine transformation to an image.
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (5, 5),
        displacement_sigma: float = 5.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.grid_shape = grid_shape
        self.displacement_sigma = displacement_sigma
        self.rng = np.random.default_rng() if rng is None else rng
    # end __init__

    def __call__(self, image: np.ndarray) -> np.ndarray:
        rows, cols = image.shape
        src_cols = np.linspace(0, cols, self.grid_shape[1])
        src_rows = np.linspace(0, rows, self.grid_shape[0])
        src = np.dstack(np.meshgrid(src_cols, src_rows)).reshape(-1, 2)
        dst = src + self.rng.normal(scale=self.displacement_sigma, size=src.shape)

        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        return warp(image, tform, output_shape=image.shape)
    # end __call__


class RandomDisplacementField:
    """
    Apply random displacement field transformation to an image.
    """

    def __init__(
        self,
        max_displacement: float,
        smooth_sigma: float,
        rng: Optional[np.random.Generator] = None,
    ):
        self.max_displacement = max_displacement
        self.smooth_sigma = smooth_sigma
        self.rng = np.random.default_rng() if rng is None else rng
    # end __init__

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        dx = self.rng.normal(0, 1, (h, w))
        dy = self.rng.normal(0, 1, (h, w))
        dx = gaussian_filter(dx, sigma=self.smooth_sigma)
        dy = gaussian_filter(dy, sigma=self.smooth_sigma)
        dx = (dx / np.max(np.abs(dx))) * self.max_displacement
        dy = (dy / np.max(np.abs(dy))) * self.max_displacement
        coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        coords_deformed = np.array([coords[0] + dx, coords[1] + dy])
        warped = warp(image, coords_deformed, mode="reflect")
        return warped
    # end __call__


def apply_random_transformations(
    vel: np.ndarray,
    rng: np.random.Generator,
    apply_prob: float = 0.5,
    crop_ratio: float = 0.9,
) -> np.ndarray:
    """
    Apply a series of random transformations to a velocity model.
    """
    transformed = img_as_float32(vel)

    if rng.random() < apply_prob:
        strength = rng.uniform(1, 3)
        radius = rng.uniform(100, 300)
        transformed = swirl(transformed, strength=strength, radius=radius)
    if rng.random() < apply_prob:
        factor = rng.uniform(0.8, 1.2)
        small = resize(
            transformed,
            (int(vel.shape[0] * factor), int(vel.shape[1] * factor)),
            mode="reflect",
            anti_aliasing=True,
        )
        transformed = resize(small, vel.shape, mode="reflect", anti_aliasing=True)
    if rng.random() < apply_prob:
        rows, cols = vel.shape
        src_cols = np.linspace(0, cols, 5)
        src_rows = np.linspace(0, rows, 5)
        src = np.dstack(np.meshgrid(src_cols, src_rows)).reshape(-1, 2)
        dst = src + rng.normal(scale=5.0, size=src.shape)
        tform = PiecewiseAffineTransform()
        tform.estimate(src, dst)
        transformed = warp(transformed, tform, output_shape=vel.shape)
    transformed = transformed - transformed.min()
    transformed = transformed / transformed.max()
    transformed = transformed * (vel.max() - vel.min()) + vel.min()
    transformed = crop_and_resize(transformed, target_shape=vel.shape, crop_ratio=crop_ratio)
    return transformed
# end apply_random_transformations


class AddCrossLine:
    """
    Add a random cross line to a velocity model.
    """

    def __init__(
        self,
        thickness_range: Tuple[int, int] = (2, 10),
        velocity_range: Tuple[float, float] = (1000, 4000),
        rng: Optional[np.random.Generator] = None,
    ):
        self.thickness_range = thickness_range
        self.velocity_range = velocity_range
        self.rng = np.random.default_rng() if rng is None else rng
    # end __init__

    def __call__(self, vel: np.ndarray) -> np.ndarray:
        nz, nx = vel.shape
        line_img = np.copy(vel)
        thickness = self.rng.integers(*self.thickness_range)
        v_line = self.rng.uniform(*self.velocity_range)
        line_mask = np.zeros_like(vel, dtype=bool)
        center = self.rng.integers(thickness, nz - thickness)
        line_mask[center - thickness // 2 : center + thickness // 2, :] = True
        angle = self.rng.uniform(-45, 45)
        rotated = rotate(line_mask.astype(float), angle=angle, resize=False, order=1, mode="reflect", cval=0)
        line_img[rotated > 0.5] = v_line
        return line_img
    # end __call__


class RandomTransformer:
    """
    Apply a sequence of transformations to a velocity model.
    """

    def __init__(self, transforms: List[Callable[[np.ndarray], np.ndarray]]):
        self.transforms = transforms
    # end __init__

    def __call__(self, vel: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            vel = transform(vel)
        return vel
    # end __call__


class RandomRotation:
    """
    Apply random rotation to a velocity model.
    """

    def __init__(
        self,
        angle_range: Tuple[float, float] = (-10, 10),
        rng: Optional[np.random.Generator] = None,
    ):
        self.angle_range = angle_range
        self.rng = rng if rng is not None else np.random.default_rng()
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        angle = self.rng.uniform(*self.angle_range)
        return sci_rotate(model, angle, reshape=False, mode="nearest")
    # end __call__


class RandomStretch:
    """
    Apply random stretching to a velocity model.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        rng: Optional[np.random.Generator] = None,
    ):
        self.scale_range = scale_range
        self.rng = rng if rng is not None else np.random.default_rng()
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        scale = self.rng.uniform(*self.scale_range)
        stretched = zoom(model, scale, order=1)
        nz, nx = model.shape
        sz, sx = stretched.shape
        if sz >= nz and sx >= nx:
            stretched = stretched[:nz, :nx]
        else:
            padded = np.ones((nz, nx)) * np.mean(model)
            padded[: min(sz, nz), : min(sx, nx)] = stretched[: min(sz, nz), : min(sx, nx)]
            stretched = padded
        return stretched
    # end __call__


class AddInclusion:
    """
    Add random circular inclusions to a velocity model.
    """

    def __init__(
        self,
        radius_range: Tuple[int, int] = (3, 10),
        velocity_range: Tuple[float, float] = (1000, 2500),
        n_inclusions_range: Tuple[int, int] = (1, 5),
        rng: Optional[np.random.Generator] = None,
    ):
        self.radius_range = radius_range
        self.velocity_range = velocity_range
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_inclusions_range = n_inclusions_range
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        nz, nx = model.shape
        n_inclusions = int(self.rng.uniform(*self.n_inclusions_range))
        for _ in range(n_inclusions):
            r = self.rng.integers(*self.radius_range)
            cx = self.rng.integers(r, nx - r)
            cz = self.rng.integers(r, nz - r)
            vel = self.rng.uniform(*self.velocity_range)
            yy, xx = np.ogrid[:nz, :nx]
            mask = (xx - cx) ** 2 + (yy - cz) ** 2 <= r**2
            model[mask] = vel
        return model
    # end __call__


class AddPerlinNoise:
    """
    Add Perlin noise to a velocity model.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float],
        amplitude: float,
        n_levels: List[float],
        rng: Optional[np.random.Generator] = None,
    ):
        self.rng = rng or np.random.default_rng()
        self.scale_range = scale_range
        self.n_levels = np.array(n_levels)
        self.amplitude = amplitude
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        scale = int(self.rng.uniform(*self.scale_range))
        nz, nx = model.shape
        res = (max(1, int(nz / scale)), max(1, int(nx / scale)))
        noise = generate_perlin_noise_2d(model.shape, res=res, rng=self.rng) * 10.0
        noise = np.vectorize(lambda n: self.n_levels[np.argmin(np.abs(self.n_levels - n))])(noise)
        return model + self.amplitude * noise
    # end __call__


class RandomStretchZoom:
    """
    Apply random zoom and stretch using scipy.ndimage.zoom.
    """

    def __init__(self, zoom_range: Tuple[float, float] = (0.9, 1.1), rng: Optional[np.random.Generator] = None):
        self.zoom_range = zoom_range
        self.rng = rng if rng is not None else np.random.default_rng()
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        zoom_factor = self.rng.uniform(*self.zoom_range)
        zoomed = scipy.ndimage.zoom(model, zoom_factor, order=1)
        nz, nx = model.shape
        zz, zx = zoomed.shape
        if zz >= nz and zx >= nx:
            zoomed = zoomed[:nz, :nx]
        else:
            padded = np.ones((nz, nx)) * np.mean(model)
            padded[: min(zz, nz), : min(zx, nx)] = zoomed[: min(zz, nz), : min(zx, nx)]
            zoomed = padded
        return zoomed
    # end __call__


class RandomDisplacement:
    """
    Apply random displacement fields using Perlin noise for smooth displacements.
    """

    def __init__(
        self,
        scale_x: float = 15,
        scale_z: float = 15,
        amplitude: float = 5,
        rng: Optional[np.random.Generator] = None,
    ):
        self.scale_x = scale_x
        self.scale_z = scale_z
        self.amplitude = amplitude
        self.rng = rng if rng is not None else np.random.default_rng()
        self.perlin_x = PerlinNoise(octaves=self.scale_x, seed=self.rng.integers(0, 10_000))
        self.perlin_z = PerlinNoise(octaves=self.scale_z, seed=self.rng.integers(0, 10_000))
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        nz, nx = model.shape
        coords_x = np.zeros_like(model, dtype=float)
        coords_z = np.zeros_like(model, dtype=float)
        for i in range(nz):
            for j in range(nx):
                coords_x[i, j] = j + self.amplitude * self.perlin_x([i / nz, j / nx])
                coords_z[i, j] = i + self.amplitude * self.perlin_z([i / nz, j / nx])
        warped = scipy.ndimage.map_coordinates(model, [coords_z, coords_x], order=1, mode="reflect")
        return warped
    # end __call__


class RandomNoise:
    """
    Add random noise to a velocity model.
    """

    def __init__(
        self,
        mode: str = "gaussian",
        mean: float = 0.0,
        var: float = 0.01,
        amount: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ):
        self.mode = mode
        self.mean = mean
        self.var = var
        self.amount = amount
        self.rng = rng if rng is not None else np.random.default_rng()
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        if self.mode == "salt_pepper":
            noisy = random_noise(model, mode="s&p", amount=self.amount)
        else:
            noisy = random_noise(model, mode=self.mode, mean=self.mean, var=self.var)
        noisy = noisy - noisy.min()
        noisy = noisy / noisy.max()
        noisy = noisy * (model.max() - model.min()) + model.min()
        return noisy.astype(model.dtype)
    # end __call__


class RandomGaussianBlur:
    """
    Apply random Gaussian blur to a velocity model.
    """

    def __init__(self, sigma_range: Tuple[float, float] = (0.5, 2.0), rng: Optional[np.random.Generator] = None):
        self.sigma_range = sigma_range
        self.rng = rng if rng is not None else np.random.default_rng()
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        sigma = self.rng.uniform(*self.sigma_range)
        return gaussian_filter(model, sigma=sigma)
    # end __call__


class RandomBrightnessContrast:
    """
    Adjust brightness and contrast randomly.
    """

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        rng: Optional[np.random.Generator] = None,
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rng = rng if rng is not None else np.random.default_rng()
    # end __init__

    def __call__(self, model: np.ndarray) -> np.ndarray:
        brightness = self.rng.uniform(*self.brightness_range)
        contrast = self.rng.uniform(*self.contrast_range)
        adjusted = (model - np.mean(model)) * contrast + np.mean(model)
        adjusted *= brightness
        return adjusted
    # end __call__

