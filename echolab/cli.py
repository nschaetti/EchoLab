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
from typing import Any, Dict, List, Optional, Sequence, Tuple
import click
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from echolab.simulators.openanfwi.wave import (
    plot_velocity,
    prepare_wave_simulation,
    run_simulation,
)
from echolab.simulators.openfwi.simulate_model import (
    animate_openfwi_wavefields,
    plot_openfwi_results,
    run_openfwi_simulation,
)
from echolab.modeling import generate_models as synthesize_velocity_models
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


def _load_generation_config(config_path: Path) -> Dict[str, Any]:
    """
    Load and validate the YAML configuration used for model generation.
    """
    config_path = config_path.expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    # end if

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    # end with

    if not isinstance(data, dict):
        raise ValueError("Model generation configuration must be a mapping.")
    # end if
    return data
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
def generate_models_cli(
    config_path: Path,
    output_path: Path,
    seed: Optional[int],
    overwrite: bool,
    verbose: bool,
) -> None:
    """
    Generate a library of velocity models using the shared echolab synthesiser.
    """
    try:
        config = _load_generation_config(config_path)
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file '{output_path}' already exists. Use --overwrite to replace it."
            )
        # end if

        model_params_section = config.get("model_params", {})
        if model_params_section is None:
            model_params_section = {}
        if not isinstance(model_params_section, dict):
            raise ValueError("'model_params' must be a mapping.")
        global_model_settings: Dict[str, Any] = {
            key: value
            for key, value in model_params_section.items()
            if not isinstance(value, dict)
        }
        per_model_settings: Dict[str, Dict[str, Any]] = {
            key: value
            for key, value in model_params_section.items()
            if isinstance(value, dict)
        }

        def _extract_numeric(
            name: str,
            converter,
            *,
            fallback: Any = None,
            required: bool = True,
        ) -> Any:
            if name in config:
                value = config[name]
            elif name in global_model_settings:
                value = global_model_settings[name]
            elif fallback is not None:
                value = fallback
            elif not required:
                return None
            else:
                raise ValueError(f"Configuration must define '{name}'.")
            try:
                return converter(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Configuration field '{name}' has an invalid value.") from exc
            # end try
        # end _extract_numeric

        nx = _extract_numeric("nx", int)
        nz = _extract_numeric("nz", int)
        dx = _extract_numeric("dx", float)
        dz = _extract_numeric("dz", float)
        n_models = _extract_numeric("n_models", int)

        if nx <= 0 or nz <= 0 or dx <= 0 or dz <= 0:
            raise ValueError("Grid dimensions (nx, nz) and spacing (dx, dz) must be positive.")
        if n_models <= 0:
            raise ValueError("Configuration 'n_models' must be a positive integer.")

        validation_section = config.get("validation", {})
        if validation_section is None:
            validation_section = {}
        if not isinstance(validation_section, dict):
            raise ValueError("'validation' must be a mapping when provided.")

        model_blur_sigma = float(
            config.get(
                "model_blur_sigma",
                global_model_settings.get("sigma", 0.0),
            )
        )
        if model_blur_sigma == 0.0:
            for params in per_model_settings.values():
                if isinstance(params, dict) and "sigma" in params:
                    try:
                        model_blur_sigma = float(params["sigma"])
                        break
                    except (TypeError, ValueError):
                        continue
            # end for
        min_velocity = float(validation_section.get("min_v", config.get("min_velocity", 1000.0)))
        max_velocity = float(validation_section.get("max_v", config.get("max_velocity", 5000.0)))
        unique_thresh = float(validation_section.get("unique_thresh", config.get("unique_thresh", 0.99)))
        entropy_thresh = float(validation_section.get("entropy_thresh", config.get("entropy_thresh", 1.0)))
        zero_thresh = float(validation_section.get("zero_thresh", config.get("zero_thresh", 0.01)))

        model_types = list(config.get("model_types", []))
        if not model_types:
            raise ValueError("Configuration must provide at least one entry in 'model_types'.")
        model_probs_map = config.get("model_probs") or config.get("model_probabilities") or {}
        if not isinstance(model_probs_map, dict):
            raise ValueError("'model_probs' or 'model_probabilities' must map model names to probability weights.")
        model_probs = _coerce_probability_map(model_types, model_probs_map, "model", normalise=True)

        sanitised_model_params: Dict[str, Dict[str, Any]] = {}
        for name in model_types:
            params = per_model_settings.get(name, {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise ValueError(f"Parameters for model '{name}' must be a mapping.")
            params_copy = dict(params)
            params_copy.pop("sigma", None)
            sanitised_model_params[name] = params_copy

        transform_types = list(config.get("transform_types") or config.get("transforms") or [])
        transform_probs_map = config.get("transform_probs") or config.get("transform_probabilities") or {}
        if transform_types and not isinstance(transform_probs_map, dict):
            raise ValueError("'transform_probs' or 'transform_probabilities' must map transform names to probabilities.")
        transform_probs: Dict[str, float] = {}
        for name in transform_types:
            try:
                probability = float(transform_probs_map.get(name, 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Transform probability for '{name}' must be numeric.") from exc
            if probability < 0.0 or probability > 1.0:
                raise ValueError(f"Transform probability for '{name}' must lie within [0, 1].")
            transform_probs[name] = probability

        transform_params = config.get("transform_params", {})
        if transform_params is None:
            transform_params = {}
        if not isinstance(transform_params, dict):
            raise ValueError("'transform_params' must be a mapping of transform names to parameter dictionaries.")

        effective_seed = seed if seed is not None else config.get("seed")
        rng = np.random.default_rng(effective_seed)
        verbose_flag = verbose or bool(config.get("verbose", False))

        models = synthesize_velocity_models(
            rng=rng,
            nx=nx,
            nz=nz,
            dx=dx,
            dz=dz,
            model_types=model_types,
            model_probs=model_probs,
            model_params=sanitised_model_params,
            model_blur_sigma=model_blur_sigma,
            transform_types=transform_types,
            transform_probs=transform_probs,
            transform_params={name: transform_params.get(name, {}) for name in transform_types},
            n_models=n_models,
            min_v=min_velocity,
            max_v=max_velocity,
            unique_thresh=unique_thresh,
            entropy_thresh=entropy_thresh,
            zero_thresh=zero_thresh,
            verbose=verbose_flag,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, models.astype(np.float32))

        info_table = Table(title="Synthetic Velocity Models")
        info_table.add_column("Setting", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="magenta")
        info_table.add_row("Configuration", str(config_path))
        info_table.add_row("Output file", str(output_path))
        info_table.add_row("Seed", str(effective_seed) if effective_seed is not None else "None")
        info_table.add_row("Model count", str(models.shape[0]))
        info_table.add_row("Grid size", f"{nz} x {nx}")
        info_table.add_row("Model types", ", ".join(model_types))
        info_table.add_row(
            "Transforms",
            ", ".join(transform_types) if transform_types else "None",
        )
        console.print(info_table)
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
