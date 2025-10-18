"""Command-line interface for echolab simulators."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import click
from rich.console import Console
from rich.table import Table

from echolab.simulators.openanfwi.wave import (
    plot_velocity,
    prepare_wave_simulation,
    run_simulation,
)
from echolab.simulators.openfwi.simulate_model import (
    plot_openfwi_results,
    run_openfwi_simulation,
)

console = Console()


class ClickBaseException(click.ClickException):
    """Wrapper to convert arbitrary exceptions into Click-friendly messages."""

    def __init__(self, exc: Exception):
        super().__init__(str(exc))


@click.group(help="Command-line interface for echolab simulators.")
def cli() -> None:
    """Top-level Click group."""


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
def wave(config_path: Path, show_velocity: bool, no_visualization: bool) -> None:
    """Run the 2D acoustic wave simulator."""
    try:
        config, velocity, _, _ = prepare_wave_simulation(config_path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise ClickBaseException(exc) from exc

    if show_velocity:
        console.print(f"[green]Plotting velocity model from[/green] {config_path}")
        plot_velocity(velocity, config)

    console.print("[green]Starting wave simulation...[/green]")
    run_simulation(config, velocity, visualize=not no_visualization)
    console.print("[green]Wave simulation completed.[/green]")


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
def openfwi(
    models_path: Path,
    model_index: int,
    config_path: Path,
    output_dir: Path,
    no_visualization: bool,
) -> None:
    """Run OpenFWI simulation and visualise/save the results."""
    try:
        results = run_openfwi_simulation(
            models_path=models_path,
            model_index=model_index,
            config_path=config_path,
        )
    except (FileNotFoundError, KeyError, ValueError, IndexError) as exc:
        raise ClickBaseException(exc) from exc

    console.print("[green]Running OpenFWI simulation...[/green]")
    figure_path = plot_openfwi_results(
        results,
        output_dir=output_dir,
        show=not no_visualization,
    )

    info_table = Table(title="OpenFWI Simulation Summary")
    info_table.add_column("Setting", style="cyan", no_wrap=True)
    info_table.add_column("Value", style="magenta")
    info_table.add_row("Models file", str(models_path))
    info_table.add_row("Model index", str(model_index))
    info_table.add_row("Configuration", str(config_path))
    info_table.add_row("Figure path", str(figure_path))
    console.print(info_table)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point compatible with setuptools console_scripts."""
    try:
        cli.main(args=list(argv) if argv is not None else None, prog_name="echolab")
    except click.ClickException as exc:
        exc.show()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
