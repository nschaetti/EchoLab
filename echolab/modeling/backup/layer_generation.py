"""
Velocity model generation utilities.

This module provides functions for generating various types of synthetic
velocity models for seismic imaging, including layered models, fault models,
dome models, and Perlin noise-based models.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate


def layered_model(
    nz: int,
    nx: int,
    dx: float,
    dz: float,
    layers: Optional[List[Tuple[int, float]]] = None,
    n_layers_range: Tuple[int, int] = (3, 6),
    v_range: Tuple[float, float] = (1500, 2500),
    angle: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a layered velocity model.

    Args:
        nz: Number of grid points in z-direction.
        nx: Number of grid points in x-direction.
        dx: Grid spacing in x-direction.
        dz: Grid spacing in z-direction.
        layers: List of (thickness, velocity) tuples for each layer.
        n_layers_range: Range for number of layers if layers is None.
        v_range: Range for velocities if layers is None.
        angle: Maximum rotation angle in degrees applied to the model.
        rng: Random number generator.

    Returns:
        2D array representing the layered velocity model.
    """
    if rng is None:
        rng = np.random.default_rng()
    # end if

    if layers is None:
        n_layers = rng.integers(n_layers_range[0], n_layers_range[1] + 1)
        raw = rng.uniform(1, 2, size=n_layers)
        thicknesses = (raw / raw.sum() * nz).astype(int)
        thicknesses[-1] += nz - thicknesses.sum()
        velocities = np.sort(rng.uniform(*v_range, size=n_layers))
        layers = list(zip(thicknesses, velocities))
    # end if

    vel = np.zeros((nz, nx))
    cd = 0
    for thickness, velocity in layers:
        layer_mask = np.zeros((nz, nx), dtype=bool)
        layer_mask[cd : cd + thickness, :] = True
        vel[layer_mask] = velocity
        cd += thickness
    # end for

    vel[vel == 0] = layers[-1][1]
    random_angle = (rng.random() - 0.5) * 2.0 * angle
    vel = rotate(vel, angle=random_angle, reshape=False, order=0, mode="nearest")
    return vel
# end layered_model


def random_fault_model(
    nz: int,
    nx: int,
    dx: float,
    dz: float,
    v_range: Tuple[float, float],
    delta_v_range: Tuple[float, float] = (200, 500),
    slope_range: Tuple[float, float] = (0.1, 0.5),
    offset_range: Tuple[int, int] = (-20, 20),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a velocity model with a random fault.
    """
    if rng is None:
        rng = np.random.default_rng()
    # end if

    base_v = rng.uniform(*v_range)
    vel = np.ones((nz, nx)) * base_v
    delta_v = rng.uniform(*delta_v_range)
    slope = rng.uniform(*slope_range)
    offset = rng.integers(*offset_range)

    for i in range(nz):
        shift = int(i * slope) + offset
        if shift < nx:
            vel[i, max(0, shift) :] += delta_v
        # end if
    # end for
    return vel
# end random_fault_model


def dome_model(
    nz: int,
    nx: int,
    dx: float,
    dz: float,
    v_high_range: Tuple[float, float],
    v_low_range: Tuple[float, float],
    center_range: Tuple[float, float],
    radius_range: Tuple[float, float],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a velocity model with a dome structure.
    """
    if rng is None:
        rng = np.random.default_rng()
    # end if

    center = (rng.uniform(*center_range), rng.uniform(*radius_range))
    v_high = rng.uniform(*v_high_range)
    v_low = rng.uniform(*v_low_range)
    radius = rng.uniform(*radius_range)

    vel = np.ones((nz, nx)) * v_low
    for i in range(nz):
        for j in range(nx):
            dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            if dist < radius:
                vel[i, j] = v_high
            # end if
        # end for
    # end for
    return vel
# end dome_model


def perlin_threshold_model(
    nz: int,
    nx: int,
    dx: float,
    dz: float,
    scale_range: Tuple[float, float],
    v_range: Tuple[float, float],
    contrast_range: Tuple[float, float],
    n_level_range: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a velocity model based on Perlin noise with thresholding.
    """
    if rng is None:
        rng = np.random.default_rng()
    # end if

    base_v = rng.uniform(*v_range)
    scale = rng.uniform(*scale_range)
    contrast = rng.uniform(*contrast_range)
    n_levels = rng.integers(*n_level_range)
    v_levels = rng.uniform(*v_range, n_levels)

    res = (max(1, int(nz / scale)), max(1, int(nx / scale)))
    noise = generate_perlin_noise_2d((nz, nx), res=res, rng=rng)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    vel = base_v + contrast * noise
    v_levels = np.array(v_levels)
    vel = np.vectorize(lambda v: v_levels[np.argmin(np.abs(v_levels - v))])(vel)
    return vel
# end perlin_threshold_model


def save_and_plot(vel: np.ndarray, name: str = "vel_model", output_dir: Union[str, Path] = "models") -> None:
    """
    Save a velocity model as a numpy array and plot it as an image.
    """
    Path(output_dir).mkdir(exist_ok=True)
    np.save(Path(output_dir) / f"{name}.npy", vel)
    plt.imshow(vel, cmap="viridis", aspect="auto")
    plt.colorbar(label="Velocity (m/s)")
    plt.title(name)
    plt.savefig(Path(output_dir) / f"{name}.png")
    plt.close()
# end save_and_plot


def generate_perlin_noise_2d(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate 2D Perlin noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    # end if

    def fade(t: np.ndarray) -> np.ndarray:
        return 6 * t**5 - 15 * t**4 + 10 * t**3
    # end fade

    res = (max(1, res[0]), max(1, res[1]))
    delta = (res[0] / shape[0], res[1] / shape[1])
    grid = rng.normal(size=(res[0] + 1, res[1] + 1, 2))
    grid /= np.linalg.norm(grid, axis=-1, keepdims=True)

    xs = np.linspace(0, res[1], shape[1], endpoint=False)
    ys = np.linspace(0, res[0], shape[0], endpoint=False)
    xi = xs.astype(int)
    yi = ys.astype(int)
    xf = xs - xi
    yf = ys - yi
    u = fade(xf)
    v = fade(yf)

    def gradient(ix: int, iy: int) -> np.ndarray:
        return grid[iy % res[0], ix % res[1]]
    # end gradient

    def dot(ix: int, iy: int, x: float, y: float) -> float:
        grad = gradient(ix, iy)
        return x * grad[..., 0] + y * grad[..., 1]
    # end dot

    noise = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            xi0 = xi[j]
            yi0 = yi[i]
            xf0 = xf[j]
            yf0 = yf[i]

            dots = [
                dot(xi0, yi0, xf0, yf0),
                dot(xi0 + 1, yi0, xf0 - 1, yf0),
                dot(xi0, yi0 + 1, xf0, yf0 - 1),
                dot(xi0 + 1, yi0 + 1, xf0 - 1, yf0 - 1),
            ]
            x1 = dots[0] + u[j] * (dots[1] - dots[0])
            x2 = dots[2] + u[j] * (dots[3] - dots[2])
            noise[i, j] = x1 + v[i] * (x2 - x1)
        # end for
    # end for
    return noise
# end generate_perlin_noise_2d
