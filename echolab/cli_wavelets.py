"""
Command-line interface for generating and visualizing wavelets.

This module provides CLI commands to generate various wavelets used in seismic modeling
and display them in plots.
"""

# Imports
import click
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from echolab.modeling.wavelets import ricker


@click.group(help="Commands for generating and visualizing wavelets.")
def wavelets() -> None:
    """
    Click group for wavelet-related commands.
    """
    pass
# end wavelets


@wavelets.command(help="Generate and visualize a Ricker wavelet.")
@click.option(
    "--frequency",
    "-f",
    type=float,
    required=True,
    help="Central frequency of the wavelet in Hz.",
)
@click.option(
    "--time-step",
    "-dt",
    type=float,
    required=True,
    help="Sampling interval in seconds.",
)
@click.option(
    "--num-samples",
    "-n",
    type=int,
    default=None,
    help="Total number of samples to generate. If not specified, an appropriate length will be calculated.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to save the plot. If not provided, the plot will be displayed interactively.",
)
@click.option(
    "--save-data",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to save the wavelet data as a NumPy .npy file.",
)
def ricker_wavelet(
    frequency: float,
    time_step: float,
    num_samples: Optional[int],
    output: Optional[Path],
    save_data: Optional[Path],
) -> None:
    """
    Generate a Ricker wavelet and display or save it as a plot.

    Args:
        frequency: Central frequency of the wavelet in Hz.
        time_step: Sampling interval in seconds.
        num_samples: Total number of samples to generate. If not specified,
            an appropriate length will be calculated.
        output: Path to save the plot. If not provided, the plot will be
            displayed interactively.
        save_data: Path to save the wavelet data as a NumPy .npy file.
    """
    try:
        # Generate the Ricker wavelet
        wavelet, time_vector = ricker(
            frequency=frequency,
            time_step=time_step,
            num_samples=num_samples,
            log=True
        )

        # Save the wavelet data if requested
        if save_data is not None:
            save_data = Path(save_data)
            save_data.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_data, {"wavelet": wavelet, "time": time_vector})
            click.echo(f"Wavelet data saved to {save_data}")
        # end if

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_vector, wavelet)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Ricker Wavelet (f={frequency} Hz, dt={time_step} s)")
        plt.grid(True)

        # Save or display the plot
        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output, dpi=300, bbox_inches="tight")
            click.echo(f"Plot saved to {output}")
        else:
            plt.show()
        # end if
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    # end try
# end def ricker_wavelet

