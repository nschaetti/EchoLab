"""
High-level helpers to run OpenFWI velocity model simulations.
"""

# Imports
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rich.console import Console
from matplotlib.colors import Normalize
from echolab.modeling import ricker

from .a2d_mod_abc28 import simulate_acoustic_wavefield


console = Console()


def _load_velocity_models(
        models_path: Path
) -> np.ndarray:
    """
    Load the velocity models stored in a NumPy ``.npy`` file.
    """
    models_path = Path(models_path)
    if not models_path.exists():
        raise FileNotFoundError(f"Velocity models file '{models_path}' not found")
    # end if

    with models_path.open("rb") as handle:
        models = np.load(handle)
    # end with

    if models.ndim == 2:
        models = np.expand_dims(models, axis=0)
    elif models.ndim != 3:
        raise ValueError(
            f"Velocity models array must be 2D or 3D; got shape {models.shape}"
        )
    # end if

    return models
# end def _load_velocity_models


def _load_simulation_config(
        config_path: Path
) -> Dict[str, Any]:
    """
    Load the OpenFWI simulation configuration from YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file '{config_path}' not found")
    # end if

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # end with

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a mapping of parameters")
    # end if

    required_keys = {
        "dx",
        "dz",
        "sx",
        "sz",
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
    # end if

    return config
# end def _load_simulation_config


def run_openfwi_simulation(
    models_path: Path,
    model_index: int,
    config_path: Path,
) -> Dict[str, Any]:
    """
    Run the OpenFWI acoustic wavefield simulation for a single model.

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
    # end if

    # Load configuration
    config = _load_simulation_config(config_path)

    # Simulation parameters
    dx = float(config["dx"])
    dz = float(config["dz"])
    sx = float(config["sx"])
    sz = float(config["sz"])
    nt = int(config["nt"])
    dt = float(config["dt"])
    freq = float(config["freq"])
    nbc = int(config["nbc"])
    gx_start = int(config["gx_start"])
    gx_end = int(config["gx_end"])
    gx_step = int(config["gx_step"])
    gz_value = float(config["gz_value"])

    # Limits
    min_x = 0.0
    max_x = (nx - 1) * dx
    min_z = 0.0
    max_z = (nz - 1) * dz

    # Get current model
    velocity_model = models[model_index]

    # Print informations
    console.print(f"[yellow]nx, nz[/]: {nx}, {nz}")
    console.print(f"[yellow]dx, dz[/] {dx}, {dz}")
    console.print(f"[yellow]sx, sz[/] {sx}, {sz}")
    console.print(f"[yellow]min_x, max_x[/] {min_x}, {max_x}")
    console.print(f"[yellow]min_z, max_z[/] {min_z}, {max_z}")
    console.print(f"[yellow]Grid size:[/] {nz} x {nx}")
    console.print(f"[yellow]Frequency:[/] {freq} Hz")
    console.print(f"[yellow]Time steps:[/] {nt} with dt = {dt}")
    console.print(f"[yellow]Velocity mean:[/] {np.mean(velocity_model)}")
    console.print(f"[yellow]Velocity std:[/] {np.std(velocity_model)}")
    console.print(f"[yellow]Velocity min:[/] {np.min(velocity_model)}")
    console.print(f"[yellow]Velocity max:[/] {np.max(velocity_model)}")

    # X and Z positions on the grid
    x_positions = np.arange(nx) * dx
    z_positions = np.arange(nz) * dz

    if not (min_x <= sx <= max_x):
        raise ValueError("Source position sx lies outside the domain")
    # end if

    if not (min_z <= sz <= max_z):
        raise ValueError("Source position sz lies outside the domain")
    # end if

    # Create source perturbation
    # The perturbation wavelet is at the beginning
    # of the simulation.
    source_wavelet, time_vector = ricker(
        frequency=freq,
        time_step=dt,
        num_samples=nt
    )

    # Receiver information
    receiver_indices = np.arange(gx_start, gx_end + 1, gx_step)
    receiver_x_positions = receiver_indices * dx
    receiver_z_positions = np.ones_like(receiver_x_positions) * gz_value

    # Print receiver information
    console.print(f"[yellow]gx[/] {receiver_x_positions[:10]}..{receiver_x_positions[-10:]}")
    console.print(f"[yellow]gz[/] {receiver_z_positions[:10]}..{receiver_z_positions[-10:]}")

    # ...
    snapshot_stride = max(1, int(config.get("snapshot_stride", 5)))
    capture_wavefield = bool(config.get("capture_wavefield", True))

    # ...
    wavefield_snapshots_shot: List[np.ndarray] = []

    def make_wavefield_callback(
        store: List[np.ndarray],
    ) -> Optional[Callable[[np.ndarray, int, float, int], None]]:
        if not capture_wavefield:
            return None
        # end if

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
    # end def make_wavefield_callback

    receiver_traces_shot1 = simulate_acoustic_wavefield(
        velocity_model=velocity_model,
        absorbing_boundary_thickness=nbc,
        grid_spacing=dx,
        num_time_steps=nt,
        time_step=dt,
        source_wavelet=source_wavelet,
        source_x_m=sx,
        source_z_m=sz,
        receiver_x_m=receiver_x_positions,
        receiver_z_m=receiver_z_positions,
        apply_free_surface=False,
        callback=make_wavefield_callback(wavefield_snapshots_shot),
    )

    # print(f"velocity_model: {velocity_model.shape}")
    # print(f"x_positions: {x_positions}")
    # print(f"z_positions: {z_positions}")
    # print(f"receiver_traces_shot1: {receiver_traces_shot1.shape}")
    # print(f"receiver_indices:{receiver_indices.shape}")
    # print(f"receiver_x_positions: {receiver_x_positions.shape}")
    # print(f"receiver_z_positions: {receiver_z_positions.shape}")

    time_axis = np.arange(nt) * dt

    # print(f"time_axis: {time_axis.shape}")
    # print(f"dx: {dx}")
    # print(f"dz: {dz}")
    # print(f"model_index: {model_index}")
    # print(f"wavefield_snapshots_shot: {len(wavefield_snapshots_shot)}")
    # print(f"snapshot_stride: {snapshot_stride}")
    # print(f"time_step: {dt}")

    return {
        "velocity_model": velocity_model,
        "x_positions": x_positions,
        "z_positions": z_positions,
        "receiver_traces_shot1": receiver_traces_shot1,
        "receiver_indices": receiver_indices + 1,
        "receiver_x_positions": receiver_x_positions,
        "receiver_z_positions": receiver_z_positions,
        "time_axis": time_axis,
        "dx": dx,
        "dz": dz,
        "model_index": model_index,
        "wavefield_snapshots_shot": wavefield_snapshots_shot,
        "snapshot_stride": snapshot_stride,
        "time_step": dt,
    }
# end def run_openfwi_simulation


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

    # Get results
    velocity_model = results["velocity_model"]
    x_positions = results["x_positions"]
    z_positions = results["z_positions"]
    receiver_traces_shot1 = results["receiver_traces_shot1"]
    time_axis = results["time_axis"]
    receiver_indices = results["receiver_indices"]
    model_index = int(results["model_index"])

    # Subplots
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 5),
        dpi=150,
        gridspec_kw={
            # 'height_ratios': [1, 2],
            'hspace': 0.4
        },
        constrained_layout=False
    )

    fig.suptitle("OpenFWI Velocity Model and Simulation")

    im1 = ax1.imshow(
        velocity_model,
        extent=(x_positions[0], x_positions[-1], z_positions[-1], z_positions[0]),
        aspect='auto',
        cmap='viridis'
    )
    cbar = fig.colorbar(im1, ax=ax1, label="Velocity (m/s)")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Z (m)")
    ax1.set_title("Velocity Model")

    extent = (
        x_positions[0],
        x_positions[-1],
        time_axis[-1],
        time_axis[0],
    )

    im2 = ax2.imshow(
        receiver_traces_shot1,
        cmap='gray',
        aspect='auto',
        extent=extent,
        vmin=-0.5,
        vmax=0.5,
    )

    ax2.set_title("Seismogram")
    ax2.set_xlabel("Receiver position (m)")
    ax2.set_ylabel("Time (s)")

    figure_path = output_dir / f"openfwi_simulation_model_{model_index:04d}.png"
    fig.savefig(figure_path, dpi=150)

    console.print(f"[cyan]Saved figure to:[/cyan] {figure_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    # end if

    return figure_path
# end def plot_openfwi_results


def animate_openfwi_wavefields(
    results: Dict[str, Any],
    output_dir: Path,
    show: bool = True,
    fps: int = 20,
    velocity_alpha: float = 0.5,
    wavefield_alpha: Optional[float] = None,
) -> List[Path]:
    """
    Create animated visualisations of the simulated wavefields.

    Parameters
    ----------
    results:
        Dictionary produced by :func:`run_openfwi_simulation`.
    output_dir:
        Directory where the animations should be saved.
    show:
        When True, display each animation interactively after saving.
    fps:
        Frame rate (frames per second) for the generated GIF.
    velocity_alpha:
        Opacity applied to the velocity model that is plotted beneath each
        wavefield snapshot.
    wavefield_alpha:
        Opacity applied to the wavefield. When ``None``, an opacity is derived
        automatically so the velocity background remains visible.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_index = int(results["model_index"])
    x_positions = results["x_positions"]
    z_positions = results["z_positions"]
    time_axis = results["time_axis"]
    time_step = float(results["time_step"])
    snapshot_stride = int(results["snapshot_stride"])
    velocity_model = results["velocity_model"]
    receiver_x_positions = results.get("receiver_x_positions")

    animations: List[Path] = []
    shot_data = [
        ("shot1", results.get("wavefield_snapshots_shot1", [])),
        ("shot2", results.get("wavefield_snapshots_shot2", [])),
    ]

    if wavefield_alpha is None:
        wavefield_alpha = max(0.2, 1.0 - float(np.clip(velocity_alpha, 0.0, 1.0)))
    else:
        wavefield_alpha = float(np.clip(wavefield_alpha, 0.0, 1.0))
    # end if

    for shot_label, snapshots in shot_data:
        if not snapshots:
            continue
        # end if

        times = np.arange(len(snapshots)) * time_step * snapshot_stride
        amplitude = max(np.max(np.abs(snapshot)) for snapshot in snapshots)
        if amplitude == 0:
            amplitude = 1.0
        # end if
        norm = Normalize(vmin=-amplitude, vmax=amplitude)

        fig, (ax, ax_receivers) = plt.subplots(
            ncols=2,
            figsize=(12, 4),
            dpi=150,
            gridspec_kw={"width_ratios": [2, 2]},
        )
        extent = [
            x_positions[0],
            x_positions[-1],
            z_positions[-1],
            z_positions[0],
        ]

        if velocity_alpha > 0:
            ax.imshow(
                velocity_model,
                extent=extent,
                aspect="auto",
                cmap="viridis",
                alpha=velocity_alpha,
                zorder=0,
            )
        # end if

        image = ax.imshow(
            snapshots[0],
            extent=extent,
            aspect="auto",
            cmap="seismic",
            norm=norm,
            alpha=wavefield_alpha,
            zorder=1,
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        title = ax.set_title(f"Wavefield {shot_label.capitalize()} - t={times[0]:.3f}s")
        fig.colorbar(image, ax=ax, label="Pressure")

        receiver_traces_key = f"receiver_traces_{shot_label}"
        receiver_traces = results.get(receiver_traces_key)
        if receiver_traces is None and shot_label == "shot1":
            receiver_traces = results.get("receiver_traces_shot1")
        # end if

        seismo_image = None
        partial_traces = None
        receiver_extent = None
        if (
            receiver_traces is not None
            and receiver_x_positions is not None
            and len(receiver_traces) > 0
            and len(receiver_x_positions) > 0
        ):
            receiver_traces = np.asarray(receiver_traces)
            seismo_vlim = np.max(np.abs(receiver_traces))
            if seismo_vlim == 0:
                seismo_vlim = 1.0
            # end if
            partial_traces = np.ma.masked_all(receiver_traces.shape, dtype=float)
            receiver_extent = (
                receiver_x_positions[0],
                receiver_x_positions[-1],
                time_axis[-1],
                time_axis[0],
            )
            ax_receivers.set_facecolor("white")
            seismo_image = ax_receivers.imshow(
                partial_traces,
                extent=receiver_extent,
                aspect="auto",
                cmap="gray",
                vmin=-seismo_vlim,
                vmax=seismo_vlim,
                interpolation="nearest",
            )
            ax_receivers.set_title("Receiver Gather")
            ax_receivers.set_xlabel("Receiver position (m)")
            ax_receivers.set_ylabel("Time (s)")
            fig.colorbar(seismo_image, ax=ax_receivers, label="Amplitude")
            ax_receivers.set_ylim(time_axis[-1], time_axis[0])
        else:
            ax_receivers.axis("off")
            ax_receivers.text(
                0.5,
                0.5,
                "No receiver data available",
                ha="center",
                va="center",
                transform=ax_receivers.transAxes,
            )
        # end if

        def _update(frame_index: int) -> List[Any]:
            image.set_data(snapshots[frame_index])
            title.set_text(
                f"Wavefield {shot_label.capitalize()} - t={times[frame_index]:.3f}s"
            )
            artists: List[Any] = [image, title]
            if seismo_image is not None and partial_traces is not None:
                current_time = times[frame_index]
                if frame_index == 0 and np.isclose(current_time, 0.0):
                    sample_count = 0
                else:
                    sample_index = int(
                        round(current_time / time_step)
                    )
                    sample_count = min(
                        sample_index + 1,
                        receiver_traces.shape[0],
                    )
                # end if
                partial_traces.mask[...] = True
                if sample_count > 0:
                    partial_traces.data[-sample_count:, :] = receiver_traces[
                        :sample_count, :
                    ]
                    partial_traces.mask[-sample_count:, :] = False
                # end if
                seismo_image.set_data(partial_traces)
                artists.append(seismo_image)
            # end if
            return artists
        # end def _update

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
    # end for

    return animations
# end def animate_openfwi_wavefields


def _strip_absorbing_boundary(
    pressure_field: np.ndarray, boundary_cell_count: int
) -> np.ndarray:
    """Remove the absorbing boundary padding from a wavefield snapshot."""
    if boundary_cell_count <= 0:
        return pressure_field.copy()
    # end if

    interior = pressure_field[
        boundary_cell_count:-boundary_cell_count or None,
        boundary_cell_count:-boundary_cell_count or None,
    ]
    return interior.copy()
# end def _strip_absorbing_boundary

__all__ = [
    "run_openfwi_simulation",
    "plot_openfwi_results",
    "animate_openfwi_wavefields",
]
