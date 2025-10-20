"""Utility functions supporting 2D acoustic wave simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import numpy as np
import yaml

ConfigDict = Dict[str, Any]

DEFAULT_SOURCE = {
    "x": 0.0,
    "z": 0.0,
    "frequency": 10.0,
    "amplitude": 1.0,
}

DEFAULT_SIMULATION = {
    "nt": 1000,
    "dt": 1e-3,
    "output_interval": 50,
    "damping": 0.0,
    "save_snapshots": False,
    "snapshot_dir": "outputs",
}

DEFAULT_PLOT = {
    "background_color": "white",
    "border_color": "black",
    "border_thickness": 1.0,
    "colormap": "viridis",
    "title": "Velocity model",
    "xlabel": "x (m)",
    "ylabel": "z (m)",
    "tick_color": "black",
    "tick_labelsize": 10.0,
    "axis_label_color": "black",
    "axis_label_size": 12.0,
    "title_color": "black",
    "title_size": 14.0,
    "legend_color": "black",
    "legend_linewidth": 1.0,
    "source_marker": "*",
    "source_color": "red",
    "source_size": 120.0,
    "grid_visible": False,
    "show_colorbar": True,
    "vmin": None,
    "vmax": None,
}

MARKER_ALIASES = {
    "★": "*",
    "☆": "o",
    "◆": "D",
    "◇": "D",
    "■": "s",
    "□": "s",
}


def load_config(path: Path) -> ConfigDict:
    """Load and validate the simulation configuration from a YAML file."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    grid_section = config.get("grid") or {}
    velocity_section = config.get("velocity") or {}

    merged = {**config, **grid_section, **velocity_section}

    required = ("nx", "nz", "dx", "dz", "c0")
    missing = [key for key in required if key not in merged]
    if missing:
        raise KeyError(f"Missing required configuration keys: {', '.join(missing)}")

    try:
        config["nx"] = int(merged["nx"])
        config["nz"] = int(merged["nz"])
        config["dx"] = float(merged["dx"])
        config["dz"] = float(merged["dz"])
        config["c0"] = float(merged["c0"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid numeric value in configuration") from exc

    if config["nx"] <= 0 or config["nz"] <= 0:
        raise ValueError("Grid dimensions 'nx' and 'nz' must be positive integers")
    if config["dx"] <= 0 or config["dz"] <= 0:
        raise ValueError("Grid spacings 'dx' and 'dz' must be positive numbers")
    if config["c0"] <= 0:
        raise ValueError("Reference velocity 'c0' must be a positive number")

    config["cmap"] = merged.get("cmap")

    if config.get("cmap") is not None:
        cmap_path = Path(config["cmap"]).expanduser()
        if not cmap_path.is_absolute():
            cmap_path = (path.parent / cmap_path).resolve()
        if not cmap_path.exists():
            raise FileNotFoundError(f"Velocity map '{cmap_path}' not found")
        config["cmap"] = cmap_path

    _load_source_section(config)
    _load_simulation_section(config)
    _load_plot_section(config)

    return config


def _load_source_section(config: ConfigDict) -> None:
    """Merge user supplied source configuration with defaults and validate."""
    source_config = dict(DEFAULT_SOURCE)
    user_source = config.get("source") or {}
    source_config.update(user_source)

    try:
        source_config["x"] = float(source_config["x"])
        source_config["z"] = float(source_config["z"])
        source_config["frequency"] = float(source_config["frequency"])
        source_config["amplitude"] = float(source_config["amplitude"])
        source_config["delay"] = float(source_config.get("delay", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid numeric value in source configuration") from exc

    if source_config["frequency"] <= 0.0:
        raise ValueError("Source frequency must be a positive number")

    domain_x = (config["nx"] - 1) * config["dx"]
    domain_z = (config["nz"] - 1) * config["dz"]

    if not (0.0 <= source_config["x"] <= domain_x):
        raise ValueError(
            f"Source x-position {source_config['x']} out of bounds (0, {domain_x})"
        )
    if not (0.0 <= source_config["z"] <= domain_z):
        raise ValueError(
            f"Source z-position {source_config['z']} out of bounds (0, {domain_z})"
        )

    ix = int(round(source_config["x"] / config["dx"]))
    iz = int(round(source_config["z"] / config["dz"]))

    if ix <= 0 or ix >= config["nx"] - 1:
        raise ValueError("Source x-position must fall strictly inside the domain interior")
    if iz <= 0 or iz >= config["nz"] - 1:
        raise ValueError("Source z-position must fall strictly inside the domain interior")

    source_config["ix"] = ix
    source_config["iz"] = iz
    config["source"] = source_config


def _load_simulation_section(config: ConfigDict) -> None:
    """Merge user supplied simulation configuration with defaults and validate."""
    sim_config = dict(DEFAULT_SIMULATION)
    user_sim = config.get("simulation") or {}
    sim_config.update(user_sim)

    try:
        sim_config["nt"] = int(sim_config["nt"])
        sim_config["output_interval"] = int(sim_config["output_interval"])
        sim_config["dt"] = float(sim_config["dt"])
        sim_config["damping"] = float(sim_config["damping"])
        sim_config["save_snapshots"] = bool(sim_config["save_snapshots"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid value in simulation configuration") from exc

    if sim_config["nt"] <= 0:
        raise ValueError("Number of time steps 'nt' must be a positive integer")
    if sim_config["dt"] <= 0.0:
        raise ValueError("Time step 'dt' must be a positive number")
    if sim_config["output_interval"] < 0:
        raise ValueError("Output interval must be non-negative")
    if sim_config["damping"] < 0.0:
        raise ValueError("Damping coefficient must be non-negative")

    snapshot_dir = sim_config.get("snapshot_dir")
    if snapshot_dir is not None:
        sim_config["snapshot_dir"] = str(snapshot_dir)

    config["simulation"] = sim_config


def _load_plot_section(config: ConfigDict) -> None:
    """Merge user supplied plotting options with defaults and validate."""
    plot_config = dict(DEFAULT_PLOT)
    user_plot = config.get("plot") or {}
    plot_config.update(user_plot)

    numeric_fields = (
        ("border_thickness", "Plot border thickness must be a numeric value"),
        ("tick_labelsize", "Tick label size must be numeric"),
        ("axis_label_size", "Axis label size must be numeric"),
        ("title_size", "Title size must be numeric"),
        ("legend_linewidth", "Legend line width must be numeric"),
        ("source_size", "Source marker size must be numeric"),
    )
    for key, error_msg in numeric_fields:
        try:
            plot_config[key] = float(plot_config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(error_msg) from exc

    if plot_config["border_thickness"] < 0.0:
        raise ValueError("Plot border thickness must be non-negative")
    if plot_config["tick_labelsize"] <= 0.0:
        raise ValueError("Tick label size must be positive")
    if plot_config["axis_label_size"] <= 0.0:
        raise ValueError("Axis label size must be positive")
    if plot_config["title_size"] <= 0.0:
        raise ValueError("Title size must be positive")
    if plot_config["legend_linewidth"] < 0.0:
        raise ValueError("Legend line width must be non-negative")
    if plot_config["source_size"] <= 0.0:
        raise ValueError("Source marker size must be positive")

    bool_fields = ("grid_visible", "show_colorbar")
    for key in bool_fields:
        value = plot_config.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            plot_config[key] = value
        elif isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "on", "1"}:
                plot_config[key] = True
            elif lowered in {"false", "no", "off", "0"}:
                plot_config[key] = False
            else:
                raise ValueError(f"Plot option '{key}' must be a boolean-like value")
        else:
            plot_config[key] = bool(value)

    for key in ("vmin", "vmax"):
        value = plot_config.get(key, None)
        if value is None:
            plot_config[key] = None
            continue
        try:
            plot_config[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Plot option '{key}' must be numeric or null") from exc

    # Remaining entries are treated as strings; enforce type for clarity.
    string_fields = (
        "background_color",
        "border_color",
        "colormap",
        "title",
        "xlabel",
        "ylabel",
        "tick_color",
        "axis_label_color",
        "title_color",
        "legend_color",
        "source_marker",
        "source_color",
    )
    for key in string_fields:
        value = plot_config.get(key)
        if value is None:
            continue
        plot_config[key] = str(value)

    config["plot"] = plot_config


def _resolve_marker(marker: str) -> Tuple[str, bool]:
    """Return a Matplotlib-compatible marker or flag to use text rendering."""
    marker_candidate = MARKER_ALIASES.get(marker, marker)
    try:
        MarkerStyle(marker_candidate)
    except (ValueError, TypeError):
        return marker, True
    return marker_candidate, False


def _init_axes(plot_cfg: Dict[str, Any]) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure/axes pair with shared styling."""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=plot_cfg["background_color"])
    fig.patch.set_facecolor(plot_cfg["background_color"])
    ax.set_facecolor(plot_cfg["background_color"])
    return fig, ax


def _style_axes(ax: plt.Axes, plot_cfg: Dict[str, Any], title: str) -> None:
    """Apply consistent axis styling."""
    ax.set_xlabel(
        plot_cfg["xlabel"],
        color=plot_cfg["axis_label_color"],
        fontsize=plot_cfg["axis_label_size"],
    )
    ax.set_ylabel(
        plot_cfg["ylabel"],
        color=plot_cfg["axis_label_color"],
        fontsize=plot_cfg["axis_label_size"],
    )
    ax.set_title(
        title,
        color=plot_cfg["title_color"],
        fontsize=plot_cfg["title_size"],
    )
    ax.tick_params(colors=plot_cfg["tick_color"], labelsize=plot_cfg["tick_labelsize"])
    for spine in ax.spines.values():
        spine.set_color(plot_cfg["border_color"])
        spine.set_linewidth(plot_cfg["border_thickness"])


def _style_colorbar(cbar: plt.Axes, plot_cfg: Dict[str, Any], label: str) -> None:
    """Ensure the colorbar styling matches the configured theme."""
    cbar.set_label(label, color=plot_cfg["axis_label_color"], fontsize=plot_cfg["axis_label_size"])
    cbar.ax.tick_params(colors=plot_cfg["tick_color"], labelsize=plot_cfg["tick_labelsize"])


def _annotate_source(ax: plt.Axes, plot_cfg: Dict[str, Any], source_cfg: Dict[str, Any]) -> Tuple[list, list]:
    """Mark the source location and return legend handles/labels."""
    resolved_marker, use_text_marker = _resolve_marker(plot_cfg["source_marker"])
    legend_handles: list = []
    legend_labels: list = []

    if not use_text_marker:
        scatter = ax.scatter(
            source_cfg["x"],
            source_cfg["z"],
            marker=resolved_marker,
            s=plot_cfg["source_size"],
            edgecolors="white",
            linewidths=0.8,
            color=plot_cfg["source_color"],
            label="Source",
            zorder=3,
        )
        legend_handles.append(scatter)
        legend_labels.append("Source")
    else:
        ax.text(
            source_cfg["x"],
            source_cfg["z"],
            plot_cfg["source_marker"],
            color=plot_cfg["source_color"],
            fontsize=plot_cfg["source_size"],
            ha="center",
            va="center",
            zorder=3,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=plot_cfg["source_color"],
                markerfacecolor=plot_cfg["source_color"],
                linestyle="None",
                markersize=np.sqrt(plot_cfg["source_size"]),
            )
        )
        legend_labels.append("Source")

    return legend_handles, legend_labels


def _finalize_legend(ax: plt.Axes, handles: list, labels: list, plot_cfg: Dict[str, Any]) -> None:
    """Create a legend if handles exist and style it."""
    if not handles:
        return

    legend = ax.legend(handles, labels, loc="upper right")
    if legend:
        plt.setp(legend.get_texts(), color=plot_cfg["legend_color"])
        legend.get_frame().set_edgecolor(plot_cfg["legend_color"])
        legend.get_frame().set_linewidth(plot_cfg["legend_linewidth"])


def create_grid(config: ConfigDict) -> Tuple[np.ndarray, np.ndarray]:
    """Create regular Cartesian grids in metres for the x and z coordinates."""
    x_max = (config["nx"] - 1) * config["dx"]
    z_max = (config["nz"] - 1) * config["dz"]
    x_coords = np.linspace(0.0, x_max, config["nx"])
    z_coords = np.linspace(0.0, z_max, config["nz"])
    x_grid, z_grid = np.meshgrid(x_coords, z_coords, indexing="xy")
    return x_grid, z_grid


def _load_velocity_map(path: Path) -> np.ndarray:
    """Load a velocity map from a .npy file or image file."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    else:
        data = mpimg.imread(path)

    if data.ndim == 3:
        # Average colour channels to obtain a scalar field.
        data = data.mean(axis=-1)
    if data.ndim != 2:
        raise ValueError(f"Velocity map at {path} must be 2D, found shape {data.shape}")

    return np.array(data, dtype=float)


def _resample_velocity_map(velocity_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resample a 2D velocity map to match the simulation grid using bilinear interpolation."""
    nz_target, nx_target = target_shape
    nz_src, nx_src = velocity_map.shape

    if (nz_src, nx_src) == target_shape:
        return velocity_map

    x_old = np.linspace(0.0, 1.0, nx_src)
    x_new = np.linspace(0.0, 1.0, nx_target)
    z_old = np.linspace(0.0, 1.0, nz_src)
    z_new = np.linspace(0.0, 1.0, nz_target)

    # Interpolate along x for every row (vectorised via apply_along_axis).
    interpolated_x = np.apply_along_axis(lambda row: np.interp(x_new, x_old, row), 1, velocity_map)

    # Interpolate the intermediate result along z for every column.
    interpolated = np.apply_along_axis(lambda col: np.interp(z_new, z_old, col), 0, interpolated_x)

    return interpolated


def initialise_velocity(config: ConfigDict, grid_shape: Tuple[int, int]) -> np.ndarray:
    """Initialise the velocity field either from a map or as a uniform field."""
    cmap_path = config.get("cmap")
    if not cmap_path:
        return np.full(grid_shape, config["c0"], dtype=float)

    velocity_map = _load_velocity_map(cmap_path)
    resampled_map = _resample_velocity_map(velocity_map, grid_shape)
    return resampled_map


def plot_velocity(c: np.ndarray, config: ConfigDict) -> None:
    """Display the velocity field as a heatmap with annotation for the source."""
    domain_x = (config["nx"] - 1) * config["dx"]
    domain_z = (config["nz"] - 1) * config["dz"]
    extent = [0.0, domain_x, domain_z, 0.0]

    plot_cfg = config.get("plot", DEFAULT_PLOT)
    source_cfg = config.get("source", DEFAULT_SOURCE)

    fig, ax = _init_axes(plot_cfg)
    img_kwargs = {
        "extent": extent,
        "origin": "upper",
        "cmap": plot_cfg["colormap"],
        "aspect": "auto",
    }
    if plot_cfg["vmin"] is not None:
        img_kwargs["vmin"] = plot_cfg["vmin"]
    if plot_cfg["vmax"] is not None:
        img_kwargs["vmax"] = plot_cfg["vmax"]

    img = ax.imshow(c, **img_kwargs)
    if plot_cfg["show_colorbar"]:
        cbar = plt.colorbar(img, ax=ax)
        _style_colorbar(cbar, plot_cfg, "Velocity (m/s)")
    _style_axes(ax, plot_cfg, plot_cfg["title"])

    if plot_cfg["grid_visible"]:
        ax.grid(color=plot_cfg["tick_color"], alpha=0.3)

    legend_handles, legend_labels = _annotate_source(ax, plot_cfg, source_cfg)
    _finalize_legend(ax, legend_handles, legend_labels, plot_cfg)

    plt.tight_layout()
    plt.show()


def ricker_wavelet(time: float, frequency: float, amplitude: float) -> float:
    """Compute the Ricker wavelet value at a given time."""
    pi_f_t = np.pi * frequency * time
    return amplitude * (1.0 - 2.0 * pi_f_t**2) * np.exp(-(pi_f_t**2))


def plot_frame(
    p: np.ndarray,
    config: ConfigDict,
    step: int,
    display: bool = True,
    state: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """Render or save a pressure field snapshot with styling.

    Parameters
    ----------
    p:
        Pressure field to display.
    config:
        Simulation configuration dictionary.
    step:
        Current time step (1-based index when called from the simulation loop).
    display:
        When True, keep an interactive window updated. When False, only snapshot
        saving (if configured) is performed without opening a window.
    state:
        Optional persistent state returned by a previous call, enabling in-place
        updates of the same matplotlib figure across time steps.
    """
    plot_cfg = config.get("plot", DEFAULT_PLOT)
    source_cfg = config.get("source", DEFAULT_SOURCE)

    domain_x = (config["nx"] - 1) * config["dx"]
    domain_z = (config["nz"] - 1) * config["dz"]
    extent = [0.0, domain_x, domain_z, 0.0]

    vmax_dynamic = float(np.max(np.abs(p))) or 1.0
    vmin = plot_cfg["vmin"] if plot_cfg["vmin"] is not None else -vmax_dynamic
    vmax = plot_cfg["vmax"] if plot_cfg["vmax"] is not None else vmax_dynamic
    sim_cfg = config.get("simulation", DEFAULT_SIMULATION)

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

            dt = config["simulation"]["dt"]
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
            dt = config["simulation"]["dt"]
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

    dt = config["simulation"]["dt"]
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


def run_simulation(config: ConfigDict, c: np.ndarray, visualize: bool = True) -> np.ndarray:
    """Run an FDTD simulation for the acoustic wave equation.

    Note: stability requires satisfying the 2D CFL condition
    dt <= 1 / (c_max * sqrt(1/dx^2 + 1/dz^2)).
    """
    sim_cfg = config["simulation"]
    source_cfg = config["source"]

    nt = sim_cfg["nt"]
    dt = sim_cfg["dt"]
    output_interval = sim_cfg["output_interval"]
    damping = sim_cfg["damping"]
    save_snapshots = sim_cfg.get("save_snapshots", False)

    dx = config["dx"]
    dz = config["dz"]

    ix = int(source_cfg["ix"])
    iz = int(source_cfg["iz"])

    p_prev = np.zeros_like(c)
    p_curr = np.zeros_like(c)
    p_next = np.zeros_like(c)

    c_dt_sq = (c * dt) ** 2
    damping_factor = np.exp(-damping * dt) if damping > 0.0 else 1.0

    if visualize:
        plt.ion()

    plot_state: Dict[str, Any] | None = None

    for step in range(nt):
        p_next.fill(0.0)
        time = step * dt
        source_time = time - source_cfg.get("delay", 0.0)
        source_val = ricker_wavelet(source_time, source_cfg["frequency"], source_cfg["amplitude"])

        lap_x = (
            (p_curr[1:-1, 2:] - 2.0 * p_curr[1:-1, 1:-1] + p_curr[1:-1, :-2]) / dx**2
        )
        lap_z = (
            (p_curr[2:, 1:-1] - 2.0 * p_curr[1:-1, 1:-1] + p_curr[:-2, 1:-1]) / dz**2
        )
        laplacian = lap_x + lap_z
        laplacian[iz - 1, ix - 1] += source_val

        # Finite-difference time-domain update for the acoustic wave equation:
        # p_next = 2*p_curr - p_prev + (c*dt)^2 * (∇² p_curr + source_term)
        p_next[1:-1, 1:-1] = (
            2.0 * p_curr[1:-1, 1:-1]
            - p_prev[1:-1, 1:-1]
            + c_dt_sq[1:-1, 1:-1] * laplacian
        )

        if damping_factor != 1.0:
            p_next[1:-1, 1:-1] *= damping_factor

        p_prev, p_curr, p_next = p_curr, p_next, p_prev

        should_output = (visualize or save_snapshots) and output_interval > 0
        if should_output:
            if (step + 1) % output_interval == 0 or step == nt - 1:
                plot_state = plot_frame(p_curr, config, step + 1, display=visualize, state=plot_state)

    if visualize and plot_state is not None:
        plt.show()
        plt.ioff()

    return p_curr


def prepare_wave_simulation(config_path: Path) -> Tuple[ConfigDict, np.ndarray, np.ndarray, np.ndarray]:
    """Load configuration, build grid, and compute the velocity field."""
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


__all__ = [
    "ConfigDict",
    "create_grid",
    "initialise_velocity",
    "load_config",
    "plot_velocity",
    "plot_frame",
    "prepare_wave_simulation",
    "ricker_wavelet",
    "run_simulation",
]
