"""
Functions for generating synthetic layered velocity models.

This module provides functions for generating various types of synthetic velocity models
including layered models, fault models, dome models, and perlin noise-based models.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.ndimage import rotate
from pathlib import Path
from skimage import filters
from typing import Tuple, List, Dict, Union, Optional, Any, Callable



def layered_model(
        nz: int,
        nx: int,
        layers: Optional[List[Tuple[int, float]]] = None,
        n_layers_range: Tuple[int, int] = (3, 6),
        v_range: Tuple[float, float] = (1500, 2500),
        angle: float = 0.0,
        rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate a layered velocity model.

    Args:
        nz: Number of grid points in z-direction
        nx: Number of grid points in x-direction
        layers: List of (thickness, velocity) tuples for each layer
        n_layers_range: Range for number of layers if layers is None
        v_range: Range for velocities if layers is None
        angle: Maximum rotation angle in degrees
        rng: Random number generator

    Returns:
        2D array representing the layered velocity model
    """
    if rng is None:
        rng = np.random.default_rng()
    # end if

    # Random layer generation if not provided
    if layers is None:
        n_layers = rng.integers(n_layers_range[0], n_layers_range[1] + 1)

        # Distribute thicknesses by normalizing
        raw = rng.uniform(1, 2, size=n_layers)
        thicknesses = (raw / raw.sum() * nz).astype(int)

        # Ensure the sum is exactly nz
        thicknesses[-1] += nz - thicknesses.sum()
        velocities = np.sort(rng.uniform(*v_range, size=n_layers))  # increasing velocity
        layers = list(zip(thicknesses, velocities))
    # end if

    # Initialize
    vel = np.zeros((nz, nx))
    cd = 0

    # Add each layer
    for thickness, v in layers:
        layer_mask = np.zeros((nz, nx), dtype=bool)
        layer_mask[cd:cd+thickness, :] = True
        vel[layer_mask] = v
        cd += thickness
    # end for

    vel[vel == 0] = layers[-1][1]

    # Rotate model
    random_angle = (rng.random() - 0.5) * 2.0 * angle
    vel = rotate(vel, angle=random_angle, reshape=False, order=0, mode='nearest')

    return vel
# end layered_model


def random_fault_model(
        nz: int,
        nx: int,
        v_range: Tuple[float, float],
        delta_v_range: Tuple[float, float] = (200, 500),
        slope_range: Tuple[float, float] = (0.1, 0.5),
        offset_range: Tuple[int, int] = (-20, 20),
        rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate a velocity model with a random fault.

    Args:
        nz: Number of grid points in z-direction
        nx: Number of grid points in x-direction
        v_range: Range for base velocity
        delta_v_range: Range for velocity contrast across the fault
        slope_range: Range for fault slope
        offset_range: Range for fault horizontal offset
        rng: Random number generator

    Returns:
        2D array representing the velocity model with a fault
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
            vel[i, max(0, shift):] += delta_v
        # end if
    # end for

    return vel
# end random_fault_model


def dome_model(
        nz: int,
        nx: int,
        v_high_range: Tuple[float, float],
        v_low_range: Tuple[float, float],
        center_range: Tuple[float, float],
        radius_range: Tuple[float, float],
        rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate a velocity model with a dome structure.

    Args:
        nz: Number of grid points in z-direction
        nx: Number of grid points in x-direction
        v_high_range: Range for high velocity (inside dome)
        v_low_range: Range for low velocity (outside dome)
        center_range: Range for dome center coordinates
        radius_range: Range for dome radius
        rng: Random number generator

    Returns:
        2D array representing the velocity model with a dome
    """
    if rng is None:
        rng = np.random.default_rng()
    # end if

    # Random parameters
    center = (rng.uniform(*center_range), rng.uniform(*radius_range))
    v_high = rng.uniform(*v_high_range)
    v_low = rng.uniform(*v_low_range)
    radius = rng.uniform(*radius_range)

    # Base model
    vel = np.ones((nz, nx)) * v_low

    for i in range(nz):
        for j in range(nx):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
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
        scale_range: Tuple[float, float],
        v_range: Tuple[float, float],
        contrast_range: Tuple[float, float],
        n_level_range: Tuple[int, int],
        rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate a velocity model based on Perlin noise with thresholding.

    Args:
        nz: Number of grid points in z-direction
        nx: Number of grid points in x-direction
        scale_range: Range for noise scale
        v_range: Range for velocities
        contrast_range: Range for noise contrast
        n_level_range: Range for number of velocity levels
        rng: Random number generator

    Returns:
        2D array representing the velocity model with Perlin noise patterns
    """
    if rng is None:
        rng = np.random.default_rng()
    # end if

    base_v = rng.uniform(*v_range)
    scale = rng.uniform(*scale_range)
    contrast = rng.uniform(*contrast_range)
    n_levels = rng.integers(*n_level_range)
    v_levels = rng.uniform(*v_range, n_levels)

    # Resolution derived from global scale
    res = (
        max(1, int(nz / scale)),
        max(1, int(nx / scale))
    )

    noise = generate_perlin_noise_2d((nz, nx), res=res, rng=rng)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    vel = base_v + contrast * noise
    v_levels = np.array(v_levels)
    vel = np.vectorize(lambda v: v_levels[np.argmin(np.abs(v_levels - v))])(vel)

    return vel
# end perlin_threshold_model


def save_and_plot(vel: np.ndarray, name: str = "vel_model", output_dir: str = "models") -> None:
    """
    Save a velocity model as a numpy array and plot it as an image.

    Args:
        vel: Velocity model as a 2D array
        name: Base name for the saved files
        output_dir: Directory where files will be saved
    """
    Path(output_dir).mkdir(exist_ok=True)
    np.save(f"{output_dir}/{name}.npy", vel)
    plt.imshow(vel, cmap="viridis", aspect='auto')
    plt.colorbar(label="Velocity (m/s)")
    plt.title(name)
    plt.savefig(f"{output_dir}/{name}.png")
    plt.close()
# end save_and_plot



def generate_perlin_noise_2d(
        shape: Tuple[int, int],
        res: Tuple[int, int],
        rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate 2D Perlin noise.

    Args:
        shape: Shape of the output noise array (height, width)
        res: Resolution of the noise (height_res, width_res)
        rng: Random number generator

    Returns:
        2D array of Perlin noise
    """
    if rng is None:
        rng = np.random.default_rng()
    # end if

    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    # end f

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = rng.normal(size=(res[0]+1, res[1]+1, 2))
    grid /= np.linalg.norm(grid, axis=-1, keepdims=True)

    # coordinates
    xs = np.linspace(0, res[1], shape[1], endpoint=False)
    ys = np.linspace(0, res[0], shape[0], endpoint=False)
    xi = xs.astype(int)
    yi = ys.astype(int)
    xf = xs - xi
    yf = ys - yi
    u = f(xf)
    v = f(yf)

    # gradients
    def g(ix, iy):
        return grid[iy % res[0], ix % res[1]]
    # end g

    def dot(ix, iy, x, y):
        return (x * g(ix, iy)[..., 0] + y * g(ix, iy)[..., 1])
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
                dot(xi0+1, yi0, xf0-1, yf0),
                dot(xi0, yi0+1, xf0, yf0-1),
                dot(xi0+1, yi0+1, xf0-1, yf0-1),
            ]
            x1 = dots[0] + u[j] * (dots[1] - dots[0])
            x2 = dots[2] + u[j] * (dots[3] - dots[2])
            noise[i, j] = x1 + v[i] * (x2 - x1)
        # end for
    # end for

    return noise
# end generate_perlin_noise_2d



