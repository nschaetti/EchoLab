"""High-level helpers to run OpenFWI velocity model simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rich.console import Console
from matplotlib.colors import Normalize

from .a2d_mod_abc28 import simulate_acoustic_wavefield
from .ricker import ricker

console = Console()


def _load_velocity_models(models_path: Path) -> np.ndarray:
    """Load the velocity models stored in a NumPy ``.npy`` file."""
    models_path = Path(models_path)
    if not models_path.exists():
        raise FileNotFoundError(f"Velocity models file '{models_path}' not found")

    with models_path.open("rb") as handle:
        models = np.load(handle)

    if models.ndim == 2:
        models = np.expand_dims(models, axis=0)
    elif models.ndim != 3:
        raise ValueError(
            f"Velocity models array must be 2D or 3D; got shape {models.shape}"
        )

    return models


def _load_simulation_config(config_path: Path) -> Dict[str, Any]:
    """Load the OpenFWI simulation configuration from YAML."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file '{config_path}' not found")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a mapping of parameters")

    required_keys = {
        "dx",
        "dz",
        "sx1",
        "sz1",
        "sx2",
        "sz2",
        "nt",
        "dt",
        "freq",
        "nbc",
        "gx_start",
        "gx_end",
        "gx_step",
        "gz_value",
    }
    missing = required_keys - config.keys()
    if missing:
        raise KeyError(f"Configuration file is missing keys: {', '.join(sorted(missing))}")

    return config


def run_openfwi_simulation(
    models_path: Path,
    model_index: int,
    config_path: Path,
) -> Dict[str, Any]:
    """Run the OpenFWI acoustic wavefield simulation for a single model.

    Parameters
    ----------
    models_path:
        Path to the ``.npy`` file containing a 3-D velocity model array of shape
        ``(n_models, nz, nx)``.
    model_index:
        Index of the velocity model to simulate.
    config_path:
        Path to the YAML configuration describing acquisition geometry. The
        configuration may optionally define ``snapshot_stride`` (int) to control
        how frequently wavefield frames are stored and ``capture_wavefield``
        (bool) to turn animation outputs on or off.

    Returns
    -------
    dict
        A dictionary containing the velocity model, simulated shot gathers, and
        auxiliary metadata required for plotting.
    """
    models = _load_velocity_models(models_path)
    n_models, nz, nx = models.shape

    if not 0 <= model_index < n_models:
        raise IndexError(f"Model index {model_index} out of bounds (0 <= idx < {n_models})")

    config = _load_simulation_config(config_path)

    dx = float(config["dx"])
    dz = float(config["dz"])
    sx1 = float(config["sx1"])
    sz1 = float(config["sz1"])
    sx2 = float(config["sx2"])
    sz2 = float(config["sz2"])
    nt = int(config["nt"])
    dt = float(config["dt"])
    freq = float(config["freq"])
    nbc = int(config["nbc"])

    gx_start = int(config["gx_start"])
    gx_end = int(config["gx_end"])
    gx_step = int(config["gx_step"])
    gz_value = float(config["gz_value"])

    min_x = 0.0
    max_x = (nx - 1) * dx
    min_z = 0.0
    max_z = (nz - 1) * dz

    velocity_model = models[model_index]

    console.print(f"[cyan]dx, dz[/cyan] {dx}, {dz}")
    console.print(f"[cyan]sx1, sz1[/cyan] {sx1}, {sz1}")
    console.print(f"[cyan]sx2, sz2[/cyan] {sx2}, {sz2}")
    console.print(f"[cyan]min_x, max_x[/cyan] {min_x}, {max_x}")
    console.print(f"[cyan]min_z, max_z[/cyan] {min_z}, {max_z}")
    console.print(f"[cyan]Grid size:[/cyan] {nz} x {nx}")
    console.print(f"[cyan]Frequency:[/cyan] {freq} Hz")
    console.print(f"[cyan]Time steps:[/cyan] {nt} with dt = {dt}")

    x_positions = np.arange(nx) * dx
    z_positions = np.arange(nz) * dz

    if not (min_x <= sx1 <= max_x):
        raise ValueError("Source position sx1 lies outside the domain")
    if not (min_z <= sz1 <= max_z):
        raise ValueError("Source position sz1 lies outside the domain")
    if not (min_x <= sx2 <= max_x):
        raise ValueError("Source position sx2 lies outside the domain")
    if not (min_z <= sz2 <= max_z):
        raise ValueError("Source position sz2 lies outside the domain")

    source_wavelet, _ = ricker(frequency=freq, time_step=dt, num_samples=nt)

    receiver_indices = np.arange(gx_start, gx_end + 1, gx_step)
    receiver_x_positions = receiver_indices * dx
    receiver_z_positions = np.ones_like(receiver_x_positions) * gz_value

    console.print(
        f"[cyan]gx, gz[/cyan] {receiver_x_positions}, {receiver_z_positions}"
    )

    snapshot_stride = max(1, int(config.get("snapshot_stride", 5)))
    capture_wavefield = bool(config.get("capture_wavefield", True))

    wavefield_snapshots_shot1: List[np.ndarray] = []
    wavefield_snapshots_shot2: List[np.ndarray] = []

    def make_wavefield_callback(
        store: List[np.ndarray],
    ) -> Optional[Callable[[np.ndarray, int, float, int], None]]:
        if not capture_wavefield:
            return None

        def _callback(
            pressure_field: np.ndarray,
            time_step_index: int,
            _time_step: float,
            boundary_cell_count: int,
        ) -> None:
            if time_step_index % snapshot_stride != 0:
                return

            store.append(
                _strip_absorbing_boundary(
                    pressure_field, boundary_cell_count
                )
            )

        return _callback

    receiver_traces_shot1 = simulate_acoustic_wavefield(
        velocity_model,
        nbc,
        dx,
        nt,
        dt,
        source_wavelet,
        sx1,
        sz1,
        receiver_x_positions,
        receiver_z_positions,
        apply_free_surface=False,
        callback=make_wavefield_callback(wavefield_snapshots_shot1),
    )

    receiver_traces_shot2 = simulate_acoustic_wavefield(
        velocity_model,
        nbc,
        dx,
        nt,
        dt,
        source_wavelet,
        sx2,
        sz2,
        receiver_x_positions,
        receiver_z_positions,
        apply_free_surface=False,
        callback=make_wavefield_callback(wavefield_snapshots_shot2),
    )

    time_axis = np.arange(nt) * dt

    return {
        "velocity_model": velocity_model,
        "x_positions": x_positions,
        "z_positions": z_positions,
        "receiver_traces_shot1": receiver_traces_shot1,
        "receiver_traces_shot2": receiver_traces_shot2,
        "receiver_indices": receiver_indices + 1,
        "receiver_x_positions": receiver_x_positions,
        "receiver_z_positions": receiver_z_positions,
        "time_axis": time_axis,
        "dx": dx,
        "dz": dz,
        "model_index": model_index,
        "wavefield_snapshots_shot1": wavefield_snapshots_shot1,
        "wavefield_snapshots_shot2": wavefield_snapshots_shot2,
        "snapshot_stride": snapshot_stride,
        "time_step": dt,
    }


def plot_openfwi_results(
    results: Dict[str, Any],
    output_dir: Path,
    show: bool = True,
) -> Path:
    """Visualise the OpenFWI simulation outputs and save the figure.

    Parameters
    ----------
    results:
        Dictionary produced by :func:`run_openfwi_simulation`.
    output_dir:
        Directory where the composite figure should be written.
    show:
        When True, display the figure interactively after saving.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    velocity_model = results["velocity_model"]
    x_positions = results["x_positions"]
    z_positions = results["z_positions"]
    receiver_traces_shot1 = results["receiver_traces_shot1"]
    receiver_traces_shot2 = results["receiver_traces_shot2"]
    time_axis = results["time_axis"]
    receiver_indices = results["receiver_indices"]
    model_index = int(results["model_index"])

    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(3, 1, 1)
    im1 = ax1.imshow(
        velocity_model,
        extent=[x_positions[0], x_positions[-1], z_positions[-1], z_positions[0]],
        aspect="auto",
    )
    fig.colorbar(im1, ax=ax1, label="Velocity (m/s)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_title("Velocity Model")

    extent = [
        receiver_indices[0],
        receiver_indices[-1],
        time_axis[-1],
        time_axis[0],
    ]

    ax2 = fig.add_subplot(3, 1, 2)
    im2 = ax2.imshow(
        receiver_traces_shot1,
        aspect="auto",
        cmap="gray",
        extent=extent,
        vmin=-0.5,
        vmax=0.5,
    )
    ax2.set_title("Seismogram (Shot 1)")
    ax2.set_ylabel("Time (s)")
    ax2.set_xlabel("Receiver #")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.imshow(
        receiver_traces_shot2,
        aspect="auto",
        cmap="gray",
        extent=extent,
        vmin=-0.5,
        vmax=0.5,
    )
    ax3.set_title("Seismogram (Shot 2)")
    ax3.set_ylabel("Time (s)")
    ax3.set_xlabel("Receiver #")

    plt.tight_layout()

    figure_path = output_dir / f"openfwi_simulation_model_{model_index:04d}.png"
    fig.savefig(figure_path, dpi=150)

    console.print(f"[cyan]Saved figure to:[/cyan] {figure_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return figure_path


def animate_openfwi_wavefields(
    results: Dict[str, Any],
    output_dir: Path,
    show: bool = True,
    fps: int = 20,
) -> List[Path]:
    """Create animated visualisations of the simulated wavefields."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_index = int(results["model_index"])
    x_positions = results["x_positions"]
    z_positions = results["z_positions"]
    time_step = float(results["time_step"])
    snapshot_stride = int(results["snapshot_stride"])

    animations: List[Path] = []
    shot_data = [
        ("shot1", results.get("wavefield_snapshots_shot1", [])),
        ("shot2", results.get("wavefield_snapshots_shot2", [])),
    ]

    for shot_label, snapshots in shot_data:
        if not snapshots:
            continue

        times = np.arange(len(snapshots)) * time_step * snapshot_stride
        amplitude = max(np.max(np.abs(snapshot)) for snapshot in snapshots)
        if amplitude == 0:
            amplitude = 1.0
        norm = Normalize(vmin=-amplitude, vmax=amplitude)

        fig, ax = plt.subplots(figsize=(6, 5))
        extent = [
            x_positions[0],
            x_positions[-1],
            z_positions[-1],
            z_positions[0],
        ]
        image = ax.imshow(
            snapshots[0],
            extent=extent,
            aspect="auto",
            cmap="seismic",
            norm=norm,
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        title = ax.set_title(f"Wavefield {shot_label.capitalize()} - t={times[0]:.3f}s")
        fig.colorbar(image, ax=ax, label="Pressure")

        def _update(frame_index: int) -> List[Any]:
            image.set_data(snapshots[frame_index])
            title.set_text(
                f"Wavefield {shot_label.capitalize()} - t={times[frame_index]:.3f}s"
            )
            return [image, title]

        anim = animation.FuncAnimation(
            fig,
            _update,
            frames=len(snapshots),
            interval=1000 / fps,
            blit=False,
        )

        gif_path = output_dir / f"openfwi_{shot_label}_model_{model_index:04d}.gif"
        anim.save(gif_path, writer=animation.PillowWriter(fps=fps))
        animations.append(gif_path)

        console.print(f"[cyan]Saved animation to:[/cyan] {gif_path}")

        if show:
            plt.show()
        plt.close(fig)

    return animations


def _strip_absorbing_boundary(
    pressure_field: np.ndarray, boundary_cell_count: int
) -> np.ndarray:
    """Remove the absorbing boundary padding from a wavefield snapshot."""
    if boundary_cell_count <= 0:
        return pressure_field.copy()

    interior = pressure_field[
        boundary_cell_count:-boundary_cell_count or None,
        boundary_cell_count:-boundary_cell_count or None,
    ]
    return interior.copy()

__all__ = [
    "run_openfwi_simulation",
    "plot_openfwi_results",
    "animate_openfwi_wavefields",
]
