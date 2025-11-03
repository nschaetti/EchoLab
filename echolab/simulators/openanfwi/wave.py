"""
OpenANFWI Acoustic Wavefield Simulation.

This module provides functions to run and visualize acoustic wavefield simulations
using the OpenANFWI approach with various noise sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import numpy as np
import yaml

# Import from the new modeling subpackage
from echolab.modeling import (
    OpenANFWISimulator,
    AcousticPressureField,
    RickerWavelet,
    RandomRickerWavelet,
    BrownNoise,
    PinkNoise,
    CompositeNoiseSource,
    strip_absorbing_boundary,
)

# Type alias for configuration dictionaries
ConfigDict = Dict[str, Any]

# Default configuration values
DEFAULT_SOURCE = {
    "ix": 50,
    "iz": 50,
    "frequency": 25.0,
    "amplitude": 1.0,
    "delay": 0.0,
    "type": "ricker",  # Options: ricker, random_ricker, brown, pink, composite
    "min_frequency": 10.0,  # For random_ricker
    "max_frequency": 50.0,  # For random_ricker
    "weights": [0.3, 0.3, 0.4],  # For composite [brown, pink, random_ricker]
}

DEFAULT_SIMULATION = {
    "nt": 1000,
    "dt": 0.001,
    "output_interval": 20,
    "damping": 0.0,
    "save_snapshots": False,
    "snapshot_dir": "outputs",
    "seed": None,  # Random seed for reproducibility
}

DEFAULT_PLOT = {
    "title": "Acoustic Wave Simulation",
    "title_color": "black",
    "title_size": 12,
    "tick_color": "black",
    "tick_size": 10,
    "grid_visible": True,
    "colormap": "seismic",
    "vmin": None,
    "vmax": None,
    "show_colorbar": True,
    "colorbar_label_size": 10,
    "colorbar_tick_size": 8,
    "source_marker": "o",
    "source_color": "red",
    "source_size": 8,
    "source_label": "Source",
    "legend_loc": "upper right",
    "legend_fontsize": 10,
    "figsize": (10, 8),
    "dpi": 100,
}


def load_config(path: Path) -> ConfigDict:
    """
    Load simulation configuration from a YAML file.

    Args:
        path (Path): Path to the YAML configuration file.

    Returns:
        ConfigDict: Dictionary containing simulation parameters.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    # end if

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise IOError(f"Failed to load configuration from {path}: {e}")
    # end try

    # Set default values for missing sections
    if "source" not in config:
        config["source"] = DEFAULT_SOURCE
    else:
        for key, value in DEFAULT_SOURCE.items():
            if key not in config["source"]:
                config["source"][key] = value
    # end if

    if "simulation" not in config:
        config["simulation"] = DEFAULT_SIMULATION
    else:
        for key, value in DEFAULT_SIMULATION.items():
            if key not in config["simulation"]:
                config["simulation"][key] = value
    # end if

    if "plot" not in config:
        config["plot"] = DEFAULT_PLOT
    else:
        for key, value in DEFAULT_PLOT.items():
            if key not in config["plot"]:
                config["plot"][key] = value
    # end if

    # Check for required keys
    required_keys = {"nx", "nz", "dx", "dz"}
    missing = required_keys - config.keys()
    if missing:
        raise KeyError(f"Configuration file is missing keys: {', '.join(sorted(missing))}")
    # end if

    return config
# end def load_config


def _load_source_section(config: ConfigDict) -> ConfigDict:
    """
    Extract and validate the source configuration.

    Args:
        config (ConfigDict): Full configuration dictionary.

    Returns:
        ConfigDict: Source configuration section.
    """
    source_cfg = config.get("source", DEFAULT_SOURCE).copy()
    
    # Validate source type
    valid_types = ["ricker", "random_ricker", "brown", "pink", "composite"]
    if source_cfg["type"] not in valid_types:
        raise ValueError(f"Invalid source type: {source_cfg['type']}. Must be one of {valid_types}")
    
    # Validate source position
    nx = config["nx"]
    nz = config["nz"]
    ix = int(source_cfg["ix"])
    iz = int(source_cfg["iz"])
    
    if not (1 <= ix <= nx):
        raise ValueError(f"Source x-index {ix} out of bounds (1 <= ix <= {nx})")
    
    if not (1 <= iz <= nz):
        raise ValueError(f"Source z-index {iz} out of bounds (1 <= iz <= {nz})")
    
    # Convert 1-based indices to 0-based
    source_cfg["ix"] = ix
    source_cfg["iz"] = iz
    
    return source_cfg
# end def _load_source_section


def _load_simulation_section(config: ConfigDict) -> ConfigDict:
    """
    Extract and validate the simulation configuration.

    Args:
        config (ConfigDict): Full configuration dictionary.

    Returns:
        ConfigDict: Simulation configuration section.
    """
    sim_cfg = config.get("simulation", DEFAULT_SIMULATION).copy()
    
    # Validate time steps
    nt = int(sim_cfg["nt"])
    if nt <= 0:
        raise ValueError(f"Number of time steps must be positive, got {nt}")
    
    # Validate time step size
    dt = float(sim_cfg["dt"])
    if dt <= 0:
        raise ValueError(f"Time step size must be positive, got {dt}")
    
    # Validate output interval
    output_interval = int(sim_cfg["output_interval"])
    if output_interval < 0:
        raise ValueError(f"Output interval must be non-negative, got {output_interval}")
    
    # Update with validated values
    sim_cfg["nt"] = nt
    sim_cfg["dt"] = dt
    sim_cfg["output_interval"] = output_interval
    
    return sim_cfg
# end def _load_simulation_section


def _load_plot_section(config: ConfigDict) -> ConfigDict:
    """
    Extract and validate the plot configuration.

    Args:
        config (ConfigDict): Full configuration dictionary.

    Returns:
        ConfigDict: Plot configuration section.
    """
    plot_cfg = config.get("plot", DEFAULT_PLOT).copy()
    
    # Validate marker
    if "source_marker" in plot_cfg:
        try:
            _resolve_marker(plot_cfg["source_marker"])
        except ValueError:
            plot_cfg["source_marker"] = DEFAULT_PLOT["source_marker"]
    
    # Validate colormap
    if "colormap" in plot_cfg:
        try:
            plt.get_cmap(plot_cfg["colormap"])
        except ValueError:
            plot_cfg["colormap"] = DEFAULT_PLOT["colormap"]
    
    # Validate figure size
    if "figsize" in plot_cfg:
        figsize = plot_cfg["figsize"]
        if not (isinstance(figsize, (list, tuple)) and len(figsize) == 2):
            plot_cfg["figsize"] = DEFAULT_PLOT["figsize"]
    
    return plot_cfg
# end def _load_plot_section


def _resolve_marker(marker: str) -> Union[str, MarkerStyle]:
    """
    Resolve a marker string to a valid matplotlib marker.

    Args:
        marker (str): Marker string.

    Returns:
        Union[str, MarkerStyle]: Valid matplotlib marker.
    """
    if marker in MarkerStyle.markers:
        return marker
    
    raise ValueError(f"Invalid marker: {marker}")
# end def _resolve_marker


def _init_axes(plot_cfg: Dict[str, Any]) -> Tuple[plt.Figure, plt.Axes]:
    """
    Initialize a figure and axes with the specified configuration.

    Args:
        plot_cfg (Dict[str, Any]): Plot configuration.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=plot_cfg["figsize"], dpi=plot_cfg["dpi"])
    return fig, ax
# end def _init_axes


def _style_axes(ax: plt.Axes, plot_cfg: Dict[str, Any], title: str) -> None:
    """
    Apply styling to the axes.

    Args:
        ax (plt.Axes): Axes to style.
        plot_cfg (Dict[str, Any]): Plot configuration.
        title (str): Title for the axes.
    """
    ax.set_title(title, color=plot_cfg["title_color"], fontsize=plot_cfg["title_size"])
    ax.set_xlabel("Distance (m)", color=plot_cfg["tick_color"], fontsize=plot_cfg["tick_size"])
    ax.set_ylabel("Depth (m)", color=plot_cfg["tick_color"], fontsize=plot_cfg["tick_size"])
    
    ax.tick_params(
        axis="both",
        which="both",
        colors=plot_cfg["tick_color"],
        labelsize=plot_cfg["tick_size"],
    )
# end def _style_axes


def _style_colorbar(cbar: plt.Axes, plot_cfg: Dict[str, Any], label: str) -> None:
    """
    Apply styling to the colorbar.

    Args:
        cbar (plt.Axes): Colorbar to style.
        plot_cfg (Dict[str, Any]): Plot configuration.
        label (str): Label for the colorbar.
    """
    cbar.set_label(label, color=plot_cfg["tick_color"], size=plot_cfg["colorbar_label_size"])
    cbar.ax.tick_params(colors=plot_cfg["tick_color"], labelsize=plot_cfg["colorbar_tick_size"])
# end def _style_colorbar


def _annotate_source(
    ax: plt.Axes, plot_cfg: Dict[str, Any], source_cfg: Dict[str, Any]
) -> Tuple[List[Line2D], List[str]]:
    """
    Annotate the source position on the plot.

    Args:
        ax (plt.Axes): Axes to annotate.
        plot_cfg (Dict[str, Any]): Plot configuration.
        source_cfg (Dict[str, Any]): Source configuration.

    Returns:
        Tuple[List[Line2D], List[str]]: Legend handles and labels.
    """
    ix = source_cfg["ix"] - 1  # Convert to 0-based index
    iz = source_cfg["iz"] - 1  # Convert to 0-based index
    dx = ax.get_figure().get_axes()[0].get_images()[0].get_extent()[1] / (ax.get_figure().get_axes()[0].get_images()[0].get_array().shape[1] - 1)
    dz = ax.get_figure().get_axes()[0].get_images()[0].get_extent()[2] / (ax.get_figure().get_axes()[0].get_images()[0].get_array().shape[0] - 1)
    
    x = ix * dx
    z = iz * dz
    
    marker = _resolve_marker(plot_cfg["source_marker"])
    source_point = ax.plot(
        x,
        z,
        marker=marker,
        color=plot_cfg["source_color"],
        markersize=plot_cfg["source_size"],
        label=plot_cfg["source_label"],
    )[0]
    
    return [source_point], [plot_cfg["source_label"]]
# end def _annotate_source


def _finalize_legend(
    ax: plt.Axes, handles: list, labels: list, plot_cfg: Dict[str, Any]
) -> None:
    """
    Finalize the legend on the plot.

    Args:
        ax (plt.Axes): Axes to add the legend to.
        handles (list): Legend handles.
        labels (list): Legend labels.
        plot_cfg (Dict[str, Any]): Plot configuration.
    """
    if handles and labels:
        ax.legend(
            handles,
            labels,
            loc=plot_cfg["legend_loc"],
            fontsize=plot_cfg["legend_fontsize"],
        )
# end def _finalize_legend


def create_grid(config: ConfigDict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create coordinate grids for the simulation domain.

    Args:
        config (ConfigDict): Configuration dictionary.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X and Z coordinate grids.
    """
    nx = config["nx"]
    nz = config["nz"]
    dx = config["dx"]
    dz = config["dz"]
    
    x = np.arange(nx) * dx
    z = np.arange(nz) * dz
    
    return np.meshgrid(x, z)
# end def create_grid


def _load_velocity_map(path: Path) -> np.ndarray:
    """
    Load a velocity map from an image file.

    Args:
        path (Path): Path to the image file.

    Returns:
        np.ndarray: Velocity map.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Velocity map file not found: {path}")
    
    try:
        img = mpimg.imread(path)
        if img.ndim == 3:
            # Convert RGB to grayscale
            img = np.mean(img, axis=2)
        return img
    except Exception as e:
        raise IOError(f"Failed to load velocity map from {path}: {e}")
# end def _load_velocity_map


def _resample_velocity_map(
    velocity_map: np.ndarray, target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Resample a velocity map to the target shape.

    Args:
        velocity_map (np.ndarray): Velocity map to resample.
        target_shape (Tuple[int, int]): Target shape (nz, nx).

    Returns:
        np.ndarray: Resampled velocity map.
    """
    from scipy.ndimage import zoom
    
    current_shape = velocity_map.shape
    zoom_factors = (
        target_shape[0] / current_shape[0],
        target_shape[1] / current_shape[1],
    )
    
    return zoom(velocity_map, zoom_factors, order=1)
# end def _resample_velocity_map


def initialise_velocity(config: ConfigDict, grid_shape: Tuple[int, int]) -> np.ndarray:
    """
    Initialize the velocity model for the simulation.

    Args:
        config (ConfigDict): Configuration dictionary.
        grid_shape (Tuple[int, int]): Shape of the grid (nz, nx).

    Returns:
        np.ndarray: Velocity model.
    """
    if "velocity_map" in config:
        velocity_map_path = Path(config["velocity_map"])
        velocity_map = _load_velocity_map(velocity_map_path)
        
        if velocity_map.shape != grid_shape:
            velocity_map = _resample_velocity_map(velocity_map, grid_shape)
        
        # Scale to velocity range
        v_min = config.get("v_min", 1500.0)
        v_max = config.get("v_max", 3500.0)
        velocity_map = v_min + (v_max - v_min) * velocity_map
        
        return velocity_map
    else:
        # Create a constant velocity model
        v_const = config.get("v_const", 2000.0)
        return np.ones(grid_shape) * v_const
# end def initialise_velocity


def plot_velocity(c: np.ndarray, config: ConfigDict) -> None:
    """
    Plot the velocity model.

    Args:
        c (np.ndarray): Velocity model.
        config (ConfigDict): Configuration dictionary.
    """
    plot_cfg = _load_plot_section(config)
    source_cfg = _load_source_section(config)
    
    domain_x = (config["nx"] - 1) * config["dx"]
    domain_z = (config["nz"] - 1) * config["dz"]
    extent = [0.0, domain_x, domain_z, 0.0]
    
    fig, ax = _init_axes(plot_cfg)
    img = ax.imshow(
        c,
        extent=extent,
        origin="upper",
        cmap="viridis",
        aspect="auto",
    )
    
    if plot_cfg["show_colorbar"]:
        cbar = plt.colorbar(img, ax=ax)
        _style_colorbar(cbar, plot_cfg, "Velocity (m/s)")
    
    title = f"{plot_cfg['title']} - Velocity Model"
    _style_axes(ax, plot_cfg, title)
    
    if plot_cfg["grid_visible"]:
        ax.grid(color=plot_cfg["tick_color"], alpha=0.3)
    
    legend_handles, legend_labels = _annotate_source(ax, plot_cfg, source_cfg)
    _finalize_legend(ax, legend_handles, legend_labels, plot_cfg)
    
    plt.tight_layout()
    plt.show()
# end def plot_velocity


def plot_frame(
    p: np.ndarray,
    config: ConfigDict,
    step: int,
    display: bool = True,
    state: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """
    Render or save a pressure field snapshot with styling.

    Args:
        p (np.ndarray): Pressure field to display.
        config (ConfigDict): Simulation configuration dictionary.
        step (int): Current time step (1-based index when called from the simulation loop).
        display (bool, optional): When True, keep an interactive window updated.
            When False, only snapshot saving (if configured) is performed without opening a window.
        state (Dict[str, Any], optional): Optional persistent state returned by a previous call,
            enabling in-place updates of the same matplotlib figure across time steps.

    Returns:
        Dict[str, Any] | None: State dictionary for subsequent calls, or None.
    """
    plot_cfg = _load_plot_section(config)
    source_cfg = _load_source_section(config)
    
    domain_x = (config["nx"] - 1) * config["dx"]
    domain_z = (config["nz"] - 1) * config["dz"]
    extent = [0.0, domain_x, domain_z, 0.0]
    
    vmax_dynamic = float(np.max(np.abs(p))) or 1.0
    vmin = plot_cfg["vmin"] if plot_cfg["vmin"] is not None else -vmax_dynamic
    vmax = plot_cfg["vmax"] if plot_cfg["vmax"] is not None else vmax_dynamic
    sim_cfg = _load_simulation_section(config)
    
    if display:
        if state is None:
            fig, ax = _init_axes(plot_cfg)
            img = ax.imshow(
                p,
                extent=extent,
                origin="upper",
                cmap=plot_cfg["colormap"],
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            cbar = None
            if plot_cfg["show_colorbar"]:
                cbar = plt.colorbar(img, ax=ax)
                _style_colorbar(cbar, plot_cfg, "Pressure (Pa)")
            
            dt = sim_cfg["dt"]
            time = step * dt
            title = f"{plot_cfg['title']} – Step {step} (t = {time:.3f} s)"
            _style_axes(ax, plot_cfg, title)
            
            if plot_cfg["grid_visible"]:
                ax.grid(color=plot_cfg["tick_color"], alpha=0.3)
            
            legend_handles, legend_labels = _annotate_source(ax, plot_cfg, source_cfg)
            _finalize_legend(ax, legend_handles, legend_labels, plot_cfg)
            
            plt.tight_layout()
            state = {"fig": fig, "ax": ax, "im": img, "cbar": cbar}
        else:
            fig = state["fig"]
            ax = state["ax"]
            img = state["im"]
            cbar = state.get("cbar")
            img.set_data(p)
            img.set_clim(vmin, vmax)
            dt = sim_cfg["dt"]
            time = step * dt
            title = f"{plot_cfg['title']} – Step {step} (t = {time:.3f} s)"
            ax.set_title(title, color=plot_cfg["title_color"], fontsize=plot_cfg["title_size"])
            if cbar is not None:
                img.axes.figure.canvas.draw_idle()
        
        if sim_cfg.get("save_snapshots"):
            snapshot_dir = Path(sim_cfg.get("snapshot_dir", "outputs"))
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            filename = snapshot_dir / f"pressure_step_{step:05d}.png"
            state["fig"].savefig(filename, dpi=150)
        
        state["fig"].canvas.draw_idle()
        plt.pause(0.001)
        return state
    
    fig, ax = _init_axes(plot_cfg)
    img = ax.imshow(
        p,
        extent=extent,
        origin="upper",
        cmap=plot_cfg["colormap"],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    if plot_cfg["show_colorbar"]:
        cbar = plt.colorbar(img, ax=ax)
        _style_colorbar(cbar, plot_cfg, "Pressure (Pa)")
    
    dt = sim_cfg["dt"]
    time = step * dt
    title = f"{plot_cfg['title']} – Step {step} (t = {time:.3f} s)"
    _style_axes(ax, plot_cfg, title)
    
    if plot_cfg["grid_visible"]:
        ax.grid(color=plot_cfg["tick_color"], alpha=0.3)
    
    legend_handles, legend_labels = _annotate_source(ax, plot_cfg, source_cfg)
    _finalize_legend(ax, legend_handles, legend_labels, plot_cfg)
    
    plt.tight_layout()
    
    if sim_cfg.get("save_snapshots"):
        snapshot_dir = Path(sim_cfg.get("snapshot_dir", "outputs"))
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        filename = snapshot_dir / f"pressure_step_{step:05d}.png"
        fig.savefig(filename, dpi=150)
    
    plt.close(fig)
    return state
# end def plot_frame


def ricker_wavelet(time: float, frequency: float, amplitude: float) -> float:
    """
    Compute the Ricker wavelet value at a given time.

    Args:
        time (float): Time in seconds.
        frequency (float): Central frequency in Hz.
        amplitude (float): Amplitude scaling factor.

    Returns:
        float: Ricker wavelet value.
    """
    pi_f_t = np.pi * frequency * time
    return amplitude * (1.0 - 2.0 * pi_f_t**2) * np.exp(-(pi_f_t**2))
# end def ricker_wavelet


def run_simulation(
    config: ConfigDict, c: np.ndarray, visualize: bool = True
) -> np.ndarray:
    """
    Run an FDTD simulation for the acoustic wave equation using the new modeling classes.

    Args:
        config (ConfigDict): Configuration dictionary.
        c (np.ndarray): Velocity model.
        visualize (bool, optional): Whether to visualize the simulation. Defaults to True.

    Returns:
        np.ndarray: Final pressure field.
    """
    # Load configuration sections
    sim_cfg = _load_simulation_section(config)
    source_cfg = _load_source_section(config)
    
    # Extract simulation parameters
    nt = sim_cfg["nt"]
    dt = sim_cfg["dt"]
    output_interval = sim_cfg["output_interval"]
    seed = sim_cfg.get("seed")
    
    # Extract grid parameters
    dx = config["dx"]
    dz = config["dz"]
    
    # Extract source parameters
    ix = int(source_cfg["ix"]) - 1  # Convert to 0-based index
    iz = int(source_cfg["iz"]) - 1  # Convert to 0-based index
    source_x = ix * dx
    source_z = iz * dz
    
    # Create the pressure field
    pressure_field = AcousticPressureField()
    
    # Create the appropriate noise source based on the source type
    source_type = source_cfg["type"]
    if source_type == "ricker":
        noise_source = RickerWavelet(source_cfg["frequency"])
    elif source_type == "random_ricker":
        noise_source = RandomRickerWavelet(
            source_cfg["min_frequency"],
            source_cfg["max_frequency"]
        )
    elif source_type == "brown":
        noise_source = BrownNoise()
    elif source_type == "pink":
        noise_source = PinkNoise()
    elif source_type == "composite":
        # Create a composite noise source with brown noise, pink noise, and random Ricker wavelets
        brown_noise = BrownNoise()
        pink_noise = PinkNoise()
        random_ricker = RandomRickerWavelet(
            source_cfg["min_frequency"],
            source_cfg["max_frequency"]
        )
        
        weights = source_cfg.get("weights", [0.3, 0.3, 0.4])
        noise_source = CompositeNoiseSource(
            sources=[brown_noise, pink_noise, random_ricker],
            weights=weights
        )
    else:
        raise ValueError(f"Invalid source type: {source_type}")
    
    # Create the simulator
    simulator = OpenANFWISimulator(pressure_field, noise_source)
    
    # Set up the simulation parameters
    simulation_params = {
        "velocity_model": c,
        "grid_spacing": dx,
        "time_step": dt,
        "num_time_steps": nt,
        "source_x_m": source_x,
        "source_z_m": source_z,
        "receiver_x_m": np.array([]),  # No receivers for now
        "receiver_z_m": np.array([]),  # No receivers for now
        "absorbing_boundary_thickness": 20,  # Default value
        "apply_free_surface": False,
        "store_wavefields": True,
        "seed": seed
    }
    
    # Run the simulation
    results = simulator.simulate(**simulation_params)
    
    # Extract the wavefields
    wavefields = results.get("wavefields", [])
    
    # Visualize the simulation if requested
    if visualize and wavefields:
        plot_state = None
        for i, wavefield in enumerate(wavefields):
            if output_interval > 0 and (i % output_interval == 0 or i == len(wavefields) - 1):
                step = i + 1  # 1-based index for plotting
                plot_state = plot_frame(wavefield, config, step, display=True, state=plot_state)
    
    # Return the final pressure field
    if wavefields:
        return wavefields[-1]
    else:
        return np.zeros_like(c)
# end def run_simulation


def prepare_wave_simulation(
    config_path: Path
) -> Tuple[ConfigDict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load configuration, build grid, and compute the velocity field.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Tuple[ConfigDict, np.ndarray, np.ndarray, np.ndarray]: Configuration, velocity model,
            X grid, and Z grid.
    """
    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file '{config_path}' not found")
    
    config = load_config(config_path)
    x_grid, z_grid = create_grid(config)
    velocity = initialise_velocity(config, (config["nz"], config["nx"]))
    if velocity.shape != x_grid.shape:
        raise ValueError(
            f"Velocity field shape {velocity.shape} does not match grid shape {x_grid.shape}"
        )
    
    return config, velocity, x_grid, z_grid
# end def prepare_wave_simulation