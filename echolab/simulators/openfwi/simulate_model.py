"""
OpenFWI Acoustic Wavefield Simulation.

This module provides functions to run and visualize acoustic wavefield simulations
using the OpenFWI approach.
"""

from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from matplotlib.colors import Normalize

# Import from the new modeling subpackage
from echolab.modeling import (
    OpenFWISimulator, 
    RickerWavelet, 
    AcousticPressureField,
    strip_absorbing_boundary,
    ricker,
    VelocityModel2D
)
from echolab.modeling.velocity_models import load_velocity_models, load_velocity_model


console = Console()


def _load_velocity_models(
    models_path: Path,
    default_grid_spacing: Tuple[float, float] = (10.0, 10.0)
) -> List[VelocityModel2D]:
    """
    Load velocity models from a file.

    This function can load velocity models from:
    - A pickle file containing a list of VelocityModel objects
    - A pickle file containing a list of VelocityMap objects
    - A NumPy file containing velocity data arrays

    Args:
        models_path (Path): Path to the file containing velocity models.
        default_grid_spacing (Tuple[float, float]): Default grid spacing (dx, dz) to use
            when creating VelocityModel2D objects from NumPy arrays. Defaults to (10.0, 10.0).

    Returns:
        List[VelocityModel2D]: List of VelocityModel2D objects.
    """
    models_path = Path(models_path)
    if not models_path.exists():
        raise FileNotFoundError(f"Models file not found: {models_path}")
    # end if

    # Try to load using the new load_velocity_models function
    try:
        models = load_velocity_models(models_path)
        # Filter to only include VelocityModel2D objects
        velocity_models = [model for model in models if isinstance(model, VelocityModel2D)]
        if velocity_models:
            return velocity_models
    except Exception as e:
        # If loading with load_velocity_models fails, try the legacy approach
        console.print(f"[yellow]Warning: Failed to load models using load_velocity_models: {e}[/yellow]")
        console.print("[yellow]Falling back to legacy loading method...[/yellow]")
    
    # Legacy loading approach
    try:
        models_array = np.load(models_path)
    except Exception as e:
        raise IOError(f"Failed to load models from {models_path}: {e}")
    # end try

    if models_array.ndim != 3:
        raise ValueError(
            f"Expected 3D array of models with shape (n_models, nz, nx), "
            f"got shape {models_array.shape}"
        )
    # end if

    # Convert to list of VelocityModel2D objects
    velocity_models = []
    for i in range(models_array.shape[0]):
        velocity_models.append(VelocityModel2D(models_array[i], grid_spacing=default_grid_spacing))
    # end for

    return velocity_models
# end def _load_velocity_models


def _load_simulation_config(
    config_path: Path
) -> Dict[str, Any]:
    """
    Load simulation configuration from a YAML file.

    Args:
        config_path (Path): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Dictionary containing simulation parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    # end if

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise IOError(f"Failed to load configuration from {config_path}: {e}")
    # end try

    # Check for required keys
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


def run_openfwi_simulation_with_progress(
    simulator: OpenFWISimulator,
    simulation_params: Dict[str, Any],
    num_time_steps: int
) -> Dict[str, Any]:
    """
    Run the OpenFWI simulation with a progress bar.
    
    This is a wrapper around the OpenFWISimulator.simulate method that adds a progress bar
    to track the simulation progress.
    
    Parameters
    ----------
    simulator:
        The OpenFWISimulator instance to use for simulation.
    simulation_params:
        Dictionary of parameters to pass to the simulate method.
    num_time_steps:
        The total number of time steps in the simulation.
        
    Returns
    -------
    dict
        The simulation results.
    """
    # Get parameters
    store_wavefields = simulation_params.get("store_wavefields", False)
    wavefields = []
    
    # Create a copy of simulation parameters without callback
    sim_params = simulation_params.copy()
    if "callback" in sim_params:
        del sim_params["callback"]
    
    # Create progress bar
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Running simulation...", total=num_time_steps)
        
        # Run the simulation
        sim_results = simulator.simulate(**sim_params)
        
        # Update progress bar to completion
        progress.update(task, completed=num_time_steps)
    
    return sim_results

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
    # Load configuration first to get dx and dz
    config = _load_simulation_config(config_path)
    
    # Load velocity models with the grid spacing from the config
    velocity_models = _load_velocity_models(
        models_path, 
        default_grid_spacing=(float(config["dx"]), float(config["dz"]))
    )
    n_models = len(velocity_models)

    if not 0 <= model_index < n_models:
        raise IndexError(f"Model index {model_index} out of bounds (0 <= idx < {n_models})")
    # end if

    # Get current model
    velocity_model = velocity_models[model_index]
    print(velocity_model)
    # Get model dimensions and properties
    nz, nx = velocity_model.shape
    dx = velocity_model.dx
    dz = velocity_model.dz

    # Other simulation parameters
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
    min_x = velocity_model.x_origin if hasattr(velocity_model, 'x_origin') else 0.0
    max_x = min_x + (nx - 1) * dx
    min_z = velocity_model.z_origin if hasattr(velocity_model, 'z_origin') else 0.0
    max_z = min_z + (nz - 1) * dz

    # Print informations
    console.log(f"[yellow]nx, nz[/]: {nx}, {nz}")
    console.log(f"[yellow]dx, dz[/] {dx}, {dz}")
    console.log(f"[yellow]sx, sz[/] {sx}, {sz}")
    console.log(f"[yellow]min_x, max_x[/] {min_x}, {max_x}")
    console.log(f"[yellow]min_z, max_z[/] {min_z}, {max_z}")
    console.log(f"[yellow]Grid size:[/] {nz} x {nx}")
    console.log(f"[yellow]Frequency:[/] {freq} Hz")
    console.log(f"[yellow]Time steps:[/] {nt} with dt = {dt}")
    console.log(f"[yellow]Velocity mean:[/] {velocity_model.mean_velocity}")
    console.log(f"[yellow]Velocity std:[/] {velocity_model.std_velocity}")
    console.log(f"[yellow]Velocity min:[/] {velocity_model.min_velocity}")
    console.log(f"[yellow]Velocity max:[/] {velocity_model.max_velocity}")

    # X and Z positions on the grid
    x_positions = velocity_model.x if hasattr(velocity_model, 'x') else np.arange(nx) * dx + min_x
    z_positions = velocity_model.z if hasattr(velocity_model, 'z') else np.arange(nz) * dz + min_z

    if not (min_x <= sx <= max_x):
        raise ValueError("Source position sx lies outside the domain")
    # end if

    if not (min_z <= sz <= max_z):
        raise ValueError("Source position sz lies outside the domain")
    # end if

    # Receiver information
    receiver_indices = np.arange(gx_start, gx_end + 1, gx_step)
    receiver_x_positions = receiver_indices * dx
    receiver_z_positions = np.ones_like(receiver_x_positions) * gz_value

    # Print receiver information
    console.log(f"[yellow]gx[/] {receiver_x_positions[:10]}..{receiver_x_positions[-10:]}")
    console.log(f"[yellow]gz[/] {receiver_z_positions[:10]}..{receiver_z_positions[-10:]}")

    # Wavefield capture settings
    snapshot_stride = max(1, int(config.get("snapshot_stride", 5)))
    capture_wavefield = bool(config.get("capture_wavefield", True))

    # Create the simulator
    pressure_field = AcousticPressureField()
    noise_source = RickerWavelet(freq)
    simulator = OpenFWISimulator(pressure_field, noise_source)

    # Set up the simulation parameters
    simulation_params = {
        "velocity_model": velocity_model.as_numpy(),  # Convert to numpy array for simulation
        "grid_spacing": dx,
        "time_step": dt,
        "num_time_steps": nt,
        "source_x_m": sx,
        "source_z_m": sz,
        "receiver_x_m": receiver_x_positions,
        "receiver_z_m": receiver_z_positions,
        "absorbing_boundary_thickness": nbc,
        "apply_free_surface": False,
        "store_wavefields": capture_wavefield
    }
    
    # Run the simulation with progress bar
    sim_results = run_openfwi_simulation_with_progress(
        simulator=simulator,
        simulation_params=simulation_params,
        num_time_steps=nt
    )
    
    # Extract the receiver data
    receiver_traces_shot = sim_results["receiver_data"]
    
    # Extract the wavefields if captured
    wavefield_snapshots_shot = []
    if capture_wavefield and "wavefields" in sim_results:
        # Apply the snapshot stride
        wavefield_snapshots_shot = sim_results["wavefields"][::snapshot_stride]
    
    # Create the time axis
    time_axis = np.arange(nt) * dt
    
    # Return the results in the same format as the original implementation
    return {
        "velocity_model": velocity_model,  # Return the VelocityModel2D object
        "velocity_model_array": velocity_model.as_numpy(),  # Also provide the numpy array for backward compatibility
        "x_positions": x_positions,
        "z_positions": z_positions,
        "receiver_traces_shot": receiver_traces_shot,
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
        "grid_spacing": dx,  # Add grid_spacing for compatibility with simulator.visualize
    }
# end def run_openfwi_simulation


def plot_openfwi_results(
    results: Dict[str, Any],
    output_dir: Path,
    show: bool = False
) -> Path:
    """Visualise the OpenFWI simulation outputs and save the figure.

    Parameters
    ----------
    results:
        Dictionary produced by :func:`run_openfwi_simulation`.
    output_dir:
        Directory where the composite figure should be written.
    show:
        Whether to display the figure interactively.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get results
    velocity_model = results["velocity_model"]
    x_positions = results["x_positions"]
    z_positions = results["z_positions"]
    receiver_traces_shot1 = results["receiver_traces_shot"]
    time_axis = results["time_axis"]
    receiver_indices = results["receiver_indices"]
    model_index = int(results["model_index"])

    # Get velocity model array (handle both VelocityModel objects and numpy arrays)
    if hasattr(velocity_model, 'as_numpy'):
        velocity_array = velocity_model.as_numpy()
        # Use model's extent if available
        if hasattr(velocity_model, 'x_extent') and hasattr(velocity_model, 'z_extent'):
            extent = [
                velocity_model.x_extent[0],
                velocity_model.x_extent[1],
                velocity_model.z_extent[1],
                velocity_model.z_extent[0],
            ]
        else:
            extent = [
                x_positions[0],
                x_positions[-1],
                z_positions[-1],
                z_positions[0],
            ]
    else:
        # For backward compatibility with numpy arrays
        velocity_array = velocity_model
        extent = [
            x_positions[0],
            x_positions[-1],
            z_positions[-1],
            z_positions[0],
        ]

    # Subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1.5]}
    )

    # Plot velocity model
    im1 = ax1.imshow(
        velocity_array,
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title(f"Velocity Model {model_index}")
    fig.colorbar(im1, ax=ax1, label="Velocity (m/s)")

    # Plot shot gather
    im2 = ax2.imshow(
        receiver_traces_shot1,
        extent=[
            receiver_indices[0],
            receiver_indices[-1],
            time_axis[-1],
            time_axis[0],
        ],
        aspect="auto",
        cmap="seismic",
    )
    ax2.set_xlabel("Receiver Index")
    ax2.set_ylabel("Time (s)")
    ax2.set_title("Shot Gather")
    fig.colorbar(im2, ax=ax2, label="Amplitude")

    # Save figure
    figure_path = output_dir / f"openfwi_simulation_model_{model_index:04d}.png"
    fig.savefig(figure_path, dpi=150)

    console.log(f"[cyan]Saved figure to:[/cyan] {figure_path}")

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
    fps: int = 20,
    velocity_alpha: float = 0.5,
    wavefield_alpha: Optional[float] = None,
) -> Optional[animation.Animation]:
    """
    Create an animation of the wavefield propagation.

    Parameters
    ----------
    results:
        Dictionary produced by :func:`run_openfwi_simulation`.
    output_dir:
        Directory where the animation file should be written.
    fps:
        Frames per second for the animation.
    velocity_alpha:
        Transparency of the velocity model overlay (0.0 to 1.0).
    wavefield_alpha:
        Transparency of the wavefield (0.0 to 1.0). If None, no transparency
        is applied.

    Returns
    -------
    matplotlib.animation.Animation or None
        The animation object if wavefields are available, otherwise None.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if wavefields are available
    shot_label = "shot"
    wavefield_key = f"wavefield_snapshots_{shot_label}"
    wavefields = results.get(wavefield_key)
    if wavefields is None:
        console.log("[red]No wavefields available for animation.[/]")
        return None
    # end if

    # Get results
    velocity_model = results["velocity_model"]
    x_positions = results["x_positions"]
    z_positions = results["z_positions"]
    model_index = int(results["model_index"])
    snapshot_stride = int(results["snapshot_stride"])
    time_step = float(results["time_step"])

    # Get velocity model array (handle both VelocityModel objects and numpy arrays)
    if hasattr(velocity_model, 'as_numpy'):
        velocity_array = velocity_model.as_numpy()
        # Use model's extent if available
        if hasattr(velocity_model, 'x_extent') and hasattr(velocity_model, 'z_extent'):
            extent = [
                velocity_model.x_extent[0],
                velocity_model.x_extent[1],
                velocity_model.z_extent[1],
                velocity_model.z_extent[0],
            ]
        else:
            extent = [
                x_positions[0],
                x_positions[-1],
                z_positions[-1],
                z_positions[0],
            ]
    else:
        # For backward compatibility with numpy arrays
        velocity_array = velocity_model
        extent = [
            x_positions[0],
            x_positions[-1],
            z_positions[-1],
            z_positions[0],
        ]

    # Calculate time for each frame
    times = np.arange(len(wavefields)) * snapshot_stride * time_step

    # Create a temporary directory to store the frames
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        
        # Create progress bar for generating frames
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            frame_task = progress.add_task("[cyan]Generating frames...", total=len(wavefields))
            
            # Generate frames on-the-fly
            for frame_idx, wavefield in enumerate(wavefields):
                # Create figure and axes for this frame
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Plot velocity model
                vel_im = ax.imshow(
                    velocity_array,
                    extent=extent,
                    aspect="auto",
                    cmap="viridis",
                    alpha=velocity_alpha,
                )
                fig.colorbar(vel_im, ax=ax, label="Velocity (m/s)")
                
                # Plot wavefield
                vmin, vmax = np.min(wavefields), np.max(wavefields)
                abs_max = max(abs(vmin), abs(vmax))
                norm = Normalize(vmin=-abs_max, vmax=abs_max)
                
                image = ax.imshow(
                    wavefield,
                    extent=extent,
                    aspect="auto",
                    cmap="seismic",
                    norm=norm,
                    alpha=wavefield_alpha,
                )
                
                # Add labels and title
                ax.set_xlabel("Distance (m)")
                ax.set_ylabel("Depth (m)")
                ax.set_title(f"Wavefield - t={times[frame_idx]:.3f}s")
                fig.colorbar(image, ax=ax, label="Pressure")
                
                # Save the frame
                frame_path = tmp_dir_path / f"frame_{frame_idx:04d}.png"
                fig.savefig(frame_path)
                plt.close(fig)
                
                # Update progress
                progress.update(frame_task, advance=1)
        
        # Create the output path for the animation
        output_path = output_dir / f"openfwi_shot{shot_label}_model_{model_index:04d}.gif"
        
        # Create animation from the saved frames
        console.log("[cyan]Creating animation from frames...[/]")
        
        # Use imageio or PIL to create the GIF
        try:
            import imageio
            
            # Get all frame files sorted by name
            frame_files = sorted(list(tmp_dir_path.glob("frame_*.png")))
            
            # Read all frames
            frames = [imageio.imread(str(frame)) for frame in frame_files]
            
            # Save as GIF
            imageio.mimsave(str(output_path), frames, fps=fps)
            
        except ImportError:
            # Fallback to using matplotlib's animation
            from matplotlib.animation import FuncAnimation
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Function to update the figure for each frame
            def update(frame_idx):
                ax.clear()
                img = plt.imread(tmp_dir_path / f"frame_{frame_idx:04d}.png")
                ax.imshow(img)
                ax.axis('off')
                return [ax]
            
            # Create the animation
            anim = FuncAnimation(
                fig, update, frames=len(wavefields), interval=1000 / fps, blit=True
            )
            
            # Save the animation
            anim.save(str(output_path), writer='pillow', fps=fps)
            plt.close(fig)
    
    console.log(f"[cyan]Saved animation to:[/cyan] {output_path}")
    
    return None  # We don't return the animation object since we're creating it on-the-fly
# end def animate_openfwi_wavefields