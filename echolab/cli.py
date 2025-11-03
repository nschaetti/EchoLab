"""
Command-line interface for running echolab simulation utilities.

This module exposes a Click-based CLI that wraps the core simulation helpers
from the echolab package. The goal of the documentation added here is to make
the intent of each command and supporting object immediately clear to future
maintainers.
"""

# Imports
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import click
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from echolab.simulators.openanfwi.wave import (
    plot_velocity,
    plot_multiple_velocity_models,
    prepare_wave_simulation,
    run_simulation,
    DEFAULT_PLOT,
    DEFAULT_SOURCE,
)
from echolab.simulators.openfwi.simulate_model import (
    animate_openfwi_wavefields,
    plot_openfwi_results,
    run_openfwi_simulation,
)
from echolab.modeling import generate_models as synthesize_velocity_models
from echolab.modeling.velocity_map import Dimensionality, save_velocity_maps
from echolab.cli_commands.generate_models_cli import generate_models
from echolab.cli_wavelets import wavelets

# Shared rich console instance to keep styling consistent across commands.
console = Console()


class ClickBaseException(click.ClickException):
    """
    Convert arbitrary exceptions into Click-friendly messages.

    Click expects errors to inherit from ``click.ClickException`` to show user
    friendly output without tracebacks. Wrapping raised exceptions in this
    helper keeps the CLI surface predictable while preserving the original
    message.
    """

    def __init__(self, exc: Exception):
        super().__init__(str(exc))
    # end def __init__

# end class ClickBaseException


@click.group(help="Command-line interface for echolab simulators.")
def cli() -> None:
    """
    Top-level Click group used as the entry point for all subcommands.
    """
# end def cli

# Add the wavelets command group to the main CLI
cli.add_command(wavelets)


def generate_stratified_velocity_model(
        nx: int,
        nz: int,
        layers: Sequence[Tuple[int, float]]
) -> np.ndarray:
    """
    Construct a 2D stratified velocity model from layer specifications.

    Args:
        nx: Horizontal size (number of columns) of the velocity grid.
        nz: Vertical size (number of rows) of the velocity grid.
        layers: Sequence of layer ``(thickness, velocity)`` pairs describing
            the stratified structure. The final layer may use a thickness of
            ``0`` to indicate that it should fill the remaining depth.

    Returns:
        A NumPy array of shape ``(nz, nx)`` filled with the requested velocities
        using ``float32`` precision.

    Raises:
        ValueError: If the layers sequence is empty, contains a non-positive
            thickness (apart from a trailing zero), exceeds ``nz``, or the
            summed thickness differs from ``nz`` when no trailing zero is used.
    """
    if not layers:
        raise ValueError("At least one layer specification must be provided.")
    # end if

    model = np.empty((nz, nx), dtype=np.float32)

    depth_index = 0
    last_index = len(layers) - 1

    for idx, (layer_thickness, velocity) in enumerate(layers):
        is_last_layer = idx == last_index

        if layer_thickness < 0:
            raise ValueError("Layer thickness must be non-negative.")
        # end if

        if is_last_layer and layer_thickness == 0:
            remaining = nz - depth_index
            if remaining <= 0:
                raise ValueError(
                    "No remaining depth to fill: preceding layers already exceed nz."
                )
            # end if
            model[depth_index:, :] = float(velocity)
            depth_index = nz
            continue
        # end if

        if layer_thickness == 0:
            raise ValueError("Only the final layer may use zero thickness.")
        # end if

        next_depth = depth_index + layer_thickness
        if next_depth > nz:
            raise ValueError(
                f"Layer thicknesses exceed nz={nz}; adjust layer definitions."
            )
        # end if

        model[depth_index:next_depth, :] = float(velocity)
        depth_index = next_depth
    # end for

    if depth_index != nz:
        raise ValueError(
            f"Sum of layer thicknesses ({depth_index}) must match nz ({nz}). "
            "Set the final layer thickness to 0 to fill the remaining depth."
        )
    # end if

    return model
# end def generate_stratified_velocity_model


def _load_generation_config(
        config_path: Path
) -> 'ModelGenerationConfig':
    """
    Load and validate the YAML configuration used for model generation.

    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        ModelGenerationConfig: A validated configuration object.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        ValueError: If the configuration is invalid or missing required parameters.
    """
    from echolab.modeling.model_generation_config import ModelGenerationConfig
    try:
        return ModelGenerationConfig.from_yaml(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise exc
    # end try
# end def _load_generation_config


def _coerce_probability_map(
    item_names: Sequence[str],
    values_map: Dict[str, Any],
    label: str,
    *,
    normalise: bool,
) -> Dict[str, float]:
    """
    Convert a mapping of string keys to floats, ensuring each requested key exists.
    """
    if not item_names:
        return {}
    # end if

    try:
        values = [float(values_map[name]) for name in item_names]
    except KeyError as exc:
        raise ValueError(f"Missing {label} probability for '{exc.args[0]}'") from exc
    # end try

    if normalise:
        total = float(sum(values))
        if total <= 0:
            raise ValueError(f"{label.capitalize()} probabilities must sum to a positive value.")
        values = [value / total for value in values]
    # end if

    return dict(zip(item_names, values))
# end def _coerce_probability_map


@cli.command(help="Visualise a 2D acoustic velocity model from a YAML configuration.")
@click.option(
    "-c",
    "--config",
    "config_path",
    default="config.yaml",
    show_default=True,
    type=click.Path(path_type=Path, exists=False, dir_okay=False),
    help="Path to the YAML configuration file.",
)
@click.option(
    "--show-velocity",
    is_flag=True,
    help="Display the initial velocity model before running the simulation.",
)
@click.option(
    "--no-visualization",
    is_flag=True,
    help="Disable pressure field visualisations during the simulation loop.",
)
def wave(
        config_path: Path,
        show_velocity: bool,
        no_visualization: bool
) -> None:
    """
    Run the 2D acoustic wave simulator.

    Args:
        config_path: Path to the YAML configuration describing the simulation.
        show_velocity: Whether to render the initial velocity model before
            simulation.
        no_visualization: Whether to disable interactive pressure field plots.
    """
    try:
        config, velocity, _, _ = prepare_wave_simulation(config_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise ClickBaseException(exc) from exc
    # end try

    if show_velocity:
        console.print(f"[green]Plotting velocity model from[/green] {config_path}")
        plot_velocity(velocity, config)
    # end if show_velocity

    console.print("[green]Starting wave simulation...[/green]")
    run_simulation(config, velocity, visualize=not no_visualization)
    console.print("[green]Wave simulation completed.[/green]")
# end def wave


@cli.command(help="Run an OpenFWI acoustic simulation for a given velocity model.")
@click.option(
    "--models",
    "models_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the NumPy file containing velocity models.",
)
@click.option(
    "--model-i",
    "model_index",
    required=True,
    type=int,
    help="Index of the velocity model to simulate.",
)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the OpenFWI YAML configuration file.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    help="Directory where figures should be saved.",
)
@click.option(
    "--no-visualization",
    is_flag=True,
    help="Skip interactive visualisation (still saves the figure).",
)
@click.option(
    "--animate/--no-animate",
    default=False,
    help="Generate animation of the wavefield propagation.",
)
def openfwi(
    models_path: Path,
    model_index: int,
    config_path: Path,
    output_dir: Path,
    no_visualization: bool,
    animate: bool,
) -> None:
    """
    Run OpenFWI simulation and visualise/save the results.

    Args:
        models_path: Path to a NumPy array holding the velocity models.
        model_index: Index of the specific velocity model to simulate.
        config_path: Path to the YAML configuration for the OpenFWI simulator.
        output_dir: Directory where static figures and animations are saved.
        no_visualization: Whether to disable interactive rendering while still
            writing files to disk.
        animate: Whether to generate an animation of the wavefield propagation.
    """
    try:
        console.log("[green]Running OpenFWI simulation...[/green]")
        results = run_openfwi_simulation(
            models_path=models_path,
            model_index=model_index,
            config_path=config_path,
        )
    except (FileNotFoundError, KeyError, ValueError, IndexError) as exc:
        raise ClickBaseException(exc) from exc
    # end try

    # Static figure summarizing the simulation; optionally shown interactively.
    figure_path = None
    if not no_visualization:
        console.log("[green]Plotting simulation output[/green]")
        figure_path = plot_openfwi_results(
            results,
            output_dir=output_dir
        )
    # end if

    # Sequence of animations illustrating wavefield evolution for each survey.
    animation_path = None
    if animate and not no_visualization:
        console.log("[green]Animating OpenFWI simulation...[/green]")
        animation_path = animate_openfwi_wavefields(
            results=results,
            output_dir=output_dir
        )
    # end if

    # Present a human-friendly summary of what was just generated and where.
    info_table = Table(title="OpenFWI Simulation Summary")
    info_table.add_column("Setting", style="cyan", no_wrap=True)
    info_table.add_column("Value", style="magenta")
    info_table.add_row("Models file", str(models_path))
    info_table.add_row("Model index", str(model_index))
    info_table.add_row("Configuration", str(config_path))
    if figure_path:
        info_table.add_row("Figure path", str(figure_path))
    if animation_path:
        info_table.add_row("Animation", str(animation_path))
    console.print(info_table)
# end def openfwi


@cli.command(
    name="generate-models",
    help="Generate synthetic velocity models and write them to a NumPy ``.npy`` file.",
)
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the YAML configuration describing the generation parameters.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Destination ``.npy`` file where the generated models will be stored.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Optional random seed that overrides the value provided in the configuration.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Allow replacing an existing output file.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Display validation messages for discarded models.",
)
@click.option(
    "--dim",
    type=click.Choice(["1D", "2D", "3D"]),
    default="2D",
    show_default=True,
    help="Dimensionality of the velocity models to generate.",
)
def generate_models_cli(
    config_path: Path,
    output_path: Path,
    seed: Optional[int],
    overwrite: bool,
    verbose: bool,
    dim: str,
) -> None:
    """
    Generate a library of velocity models using the shared echolab synthesiser.
    
    This function delegates to the implementation in echolab.cli_commands.generate_models_cli.
    
    Args:
        config_path: Path to the YAML configuration file
        output_path: Path to save the generated models
        seed: Optional random seed to override the one in configuration
        overwrite: Whether to overwrite existing output file
        verbose: Whether to display validation messages for discarded models
        dim: Dimensionality of the velocity models to generate (1D, 2D, 3D)
    """
    try:
        # Call the implementation in the dedicated module
        generate_models(
            config_path=config_path,
            output_path=output_path,
            seed=seed,
            overwrite=overwrite,
            verbose=verbose,
            dim=dim
        )
    except (FileNotFoundError, FileExistsError, ValueError, yaml.YAMLError) as exc:
        raise ClickBaseException(exc) from exc
    # end try
# end def generate_models_cli


@cli.command(
    name="velmap-2d",
    help="Generate a stratified 2D velocity model and save it to a NumPy file.",
)
@click.option(
    "--nx",
    type=int,
    required=True,
    help="Number of horizontal samples (columns) in the velocity model.",
)
@click.option(
    "--nz",
    type=int,
    required=True,
    help="Number of vertical samples (rows) in the velocity model.",
)
@click.option(
    "--layer",
    "layer_specs",
    nargs=2,
    type=(int, float),
    multiple=True,
    required=True,
    help=(
        "Specify layers as THICKNESS VELOCITY pairs. Repeat for multiple layers. "
        "Use thickness 0 on the final layer to fill the remaining depth."
    ),
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path, dir_okay=False),
    help="Destination path for the generated NumPy ``.npy`` file.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Allow replacing an existing output file (default: do not overwrite).",
)
def velmap_2d(
        nx: int,
        nz: int,
        layer_specs: Sequence[Tuple[int, float]],
        output_path: Path,
        overwrite: bool
) -> None:
    """
    Create a stratified velocity model and serialize it in ``.npy`` format.

    Args:
        nx: Number of grid samples in the horizontal dimension.
        nz: Number of grid samples in the vertical dimension.
        layer_specs: Sequence of ``(thickness, velocity)`` pairs describing the
            stratified layers from top to bottom. The last layer may use a
            thickness of ``0`` to extend indefinitely (fills remaining depth).
        output_path: Path where the generated velocity grid should be stored.
        overwrite: Whether an existing file may be replaced.
    """
    try:
        if nx <= 0 or nz <= 0:
            raise ValueError("Both nx and nz must be positive integers.")
        # end if

        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file '{output_path}' already exists. Use --overwrite to replace it."
            )
        # end if

        layers: List[Tuple[int, float]] = [
            (int(thickness), float(velocity)) for thickness, velocity in layer_specs
        ]

        velocity_model = generate_stratified_velocity_model(
            nx=nx,
            nz=nz,
            layers=layers,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, velocity_model)  # Persist the generated model to disk.

        console.print(
            f"[green]Saved stratified velocity model to[/green] {output_path}"
        )
    except (ValueError, FileExistsError, OSError) as exc:
        raise ClickBaseException(exc) from exc
    # end try
# end def velmap_2d


@cli.command(
    name="visualize-model",
    help="Visualize a velocity model from a NumPy .npy file.",
)
@click.option(
    "--file",
    "model_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to the NumPy .npy file containing velocity models.",
)
@click.argument(
    "model_indices",
    nargs=-1,
    type=int,
    required=True,
)
@click.option(
    "--figwidth",
    type=float,
    default=None,
    help="Width of the figure in inches. If not specified, calculated automatically.",
)
@click.option(
    "--figheight",
    type=float,
    default=None,
    help="Height of the figure in inches. If not specified, calculated automatically.",
)
@click.option(
    "--title",
    type=str,
    default="Velocity Model Visualization",
    show_default=True,
    help="Title for the visualization.",
)
def visualize_model(
    model_path: Path,
    model_indices: List[int],
    figwidth: Optional[float],
    figheight: Optional[float],
    title: str,
) -> None:
    """
    Visualize one or more velocity models from a NumPy .npy file.
    
    Usage:
        echolab visualize-model --file <model_file.npy> [model_indices...] [options]
    
    Args:
        model_path: Path to the NumPy .npy file containing velocity models.
        model_indices: List of indices of the velocity models to visualize (0-based).
        figwidth: Width of the figure in inches. If None, calculated automatically.
        figheight: Height of the figure in inches. If None, calculated automatically.
        title: Title for the visualization.
    """
    try:
        # Load the velocity models
        console.print(f"[green]Loading velocity models from[/green] {model_path}")
        try:
            # First try to load as a list of VelocityModel objects
            # Use the pydantic-based implementation
            from echolab.modeling.velocity_models import load_velocity_models as load_velocity_model_objects
            from echolab.modeling.velocity_models import VelocityModel, VelocityModel2D
            try:
                velocity_model_objects = load_velocity_model_objects(model_path)
                is_velocity_model = True
                is_velocity_map = False
            except Exception as e:
                console.print(f"[yellow]Not a VelocityModel file, trying VelocityMap...[/yellow]")
                is_velocity_model = False
                
                # Try to load as a list of VelocityMap objects
                from echolab.modeling.velocity_map import load_velocity_maps, VelocityMap
                try:
                    velocity_maps = load_velocity_maps(model_path)
                    is_velocity_map = True
                except:
                    # If that fails, try loading as a NumPy array
                    velocity_models_array = np.load(model_path, allow_pickle=True)
                    is_velocity_map = False
        except Exception as exc:
            raise ValueError(f"Failed to load velocity models from {model_path}: {exc}")
        
        # Lists to store models, configs, and parameters
        selected_models = []
        configs = []
        model_params = []
        
        # Process each requested model index
        for model_index in model_indices:
            # Handle VelocityModel objects
            if is_velocity_model:
                try:
                    velocity_model_obj = velocity_model_objects[model_index]
                    velocity_model = velocity_model_obj.as_numpy()
                    
                    # Get model parameters
                    params = {
                        "dx": velocity_model_obj.grid_spacing[0],
                        "dz": velocity_model_obj.grid_spacing[-1],
                    }
                    
                    # Add metadata from the VelocityModel
                    if hasattr(velocity_model_obj, '_metadata') and velocity_model_obj._metadata:
                        params.update(velocity_model_obj._metadata)
                    
                    # Use grid_spacing from the VelocityModel for this specific model
                    model_dx = params["dx"]
                    model_dz = params["dz"]
                except IndexError:
                    raise ValueError(
                        f"Model index {model_index} out of bounds. "
                        f"File contains {len(velocity_model_objects)} models (0-{len(velocity_model_objects)-1})."
                    )
            # Handle VelocityMap objects
            elif is_velocity_map:
                try:
                    velocity_map = velocity_maps[model_index]
                    velocity_model = velocity_map.data

                    # Get model parameters
                    params = {
                        "dx": velocity_map.dx,
                        "dz": velocity_map.dz,
                    }
                    # Add any additional parameters from the VelocityMap
                    for attr in dir(velocity_map):
                        if not attr.startswith('_') and attr not in ['data', 'dx', 'dz']:
                            value = getattr(velocity_map, attr)
                            if not callable(value):
                                params[attr] = value
                    
                    # Use dx and dz from the VelocityMap for this specific model
                    model_dx = velocity_map.dx
                    model_dz = velocity_map.dz
                except IndexError:
                    raise ValueError(
                        f"Model index {model_index} out of bounds. "
                        f"File contains {len(velocity_maps)} models (0-{len(velocity_maps)-1})."
                    )
            else:
                # Handle NumPy arrays
                if not isinstance(velocity_models_array, np.ndarray):
                    raise ValueError(f"File {model_path} does not contain a NumPy array.")
                
                # Handle both single models and arrays of models
                if velocity_models_array.ndim == 2:
                    # Single model
                    if model_index != 0:
                        raise ValueError(f"File contains a single model, but index {model_index} was requested.")
                    velocity_model = velocity_models_array
                else:
                    # Array of models
                    try:
                        velocity_model = velocity_models_array[model_index]
                    except IndexError:
                        raise ValueError(
                            f"Model index {model_index} out of bounds. "
                            f"File contains {velocity_models_array.shape[0]} models (0-{velocity_models_array.shape[0]-1})."
                        )
                
                # For NumPy array models, we need to raise an error since dx and dz must be in the model
                raise ValueError(
                    f"The model in {model_path} does not contain dx and dz information. "
                    "Please use VelocityModel or VelocityMap format which includes this information."
                )
            
            # Create a configuration dictionary for this model
            nz, nx = velocity_model.shape
            config = {
                "nx": nx,
                "nz": nz,
                "dx": model_dx,
                "dz": model_dz,
                "plot": DEFAULT_PLOT.copy(),
                "source": DEFAULT_SOURCE.copy(),
            }
            
            # Update the plot title
            config["plot"]["title"] = f"{title} - Model {model_index}"
            
            # Add model, config, and parameters to lists
            selected_models.append(velocity_model)
            configs.append(config)
            model_params.append(params)
        
        # Display model parameters in the console
        console.print(f"[green]Visualizing {len(selected_models)} velocity models[/green]")
        
        # Print parameters for each model
        for i, params in enumerate(model_params):
            console.print(f"[bold blue]Model {i} Parameters:[/bold blue]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Parameter", style="dim")
            table.add_column("Value")
            
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    table.add_row(key, f"{value:.4f}")
                else:
                    table.add_row(key, str(value))
            
            console.print(table)
            console.print("")
        
        # Plot all selected models with their parameters
        plot_multiple_velocity_models(
            velocity_models=selected_models,
            configs=configs,
            model_params=model_params,
            show_source=False,
            figwidth=figwidth,
            figheight=figheight
        )
        
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise ClickBaseException(exc) from exc
# end def visualize_model


def main(
        argv: Optional[Sequence[str]] = None
) -> int:
    """Execute the CLI entry point as expected by ``console_scripts`` hooks.

    Args:
        argv: Optional sequence of command-line arguments. When ``None`` the
            process ``sys.argv`` is used instead.

    Returns:
        Zero on success, or one if a Click-handled exception was raised.
    """
    try:
        cli.main(args=list(argv) if argv is not None else None, prog_name="echolab")
    except click.ClickException as exc:
        exc.show()
        return 1
    # end try
    return 0
# end def main


if __name__ == "__main__":
    raise SystemExit(main())
# end if
