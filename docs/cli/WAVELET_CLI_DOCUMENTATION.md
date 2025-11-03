# Wavelet CLI Commands Documentation

## Overview

This document provides documentation for the wavelet CLI commands implemented in the echolab project. These commands allow users to generate and visualize various wavelets used in seismic modeling.

## Implementation Details

The implementation consists of:

1. A new module `cli_wavelets.py` that defines a Click command group for wavelet-related commands
2. Integration with the main CLI interface in `cli.py`
3. A command for generating and visualizing the Ricker wavelet

## Available Commands

### Ricker Wavelet

The `ricker-wavelet` command generates a Ricker wavelet (Mexican hat wavelet) with specified parameters and displays or saves it as a plot.

#### Usage

```bash
echolab wavelets ricker-wavelet --frequency 25.0 --time-step 0.001 [OPTIONS]
```

#### Parameters

- `--frequency, -f` (required): Central frequency of the wavelet in Hz
- `--time-step, -dt` (required): Sampling interval in seconds
- `--num-samples, -n` (optional): Total number of samples to generate. If not specified, an appropriate length will be calculated
- `--output, -o` (optional): Path to save the plot. If not provided, the plot will be displayed interactively
- `--save-data` (optional): Path to save the wavelet data as a NumPy .npy file

#### Examples

Generate and display a 25 Hz Ricker wavelet with 1ms sampling:
```bash
echolab wavelets ricker-wavelet --frequency 25.0 --time-step 0.001
```

Generate a 30 Hz Ricker wavelet with 2ms sampling and save the plot:
```bash
echolab wavelets ricker-wavelet --frequency 30.0 --time-step 0.002 --output ricker_30hz.png
```

Generate a 20 Hz Ricker wavelet with 200 samples and save both the plot and data:
```bash
echolab wavelets ricker-wavelet --frequency 20.0 --time-step 0.001 --num-samples 200 --output ricker_20hz.png --save-data ricker_20hz.npy
```

## Testing

The implementation has been tested using:

1. Direct testing of the Ricker wavelet function (`test_ricker_file_direct.py`)
2. CLI command testing using Click's CliRunner (`test_cli_wavelets_direct.py`)

Both tests confirm that the implementation works correctly, generating Ricker wavelets with the specified parameters and saving them as plots and data files.

## Future Enhancements

Additional wavelet types can be added to the `wavelets.py` module and corresponding CLI commands can be implemented in the `cli_wavelets.py` module following the same pattern as the Ricker wavelet.

Potential wavelet types to add:
- Morlet wavelet
- Gaussian wavelet
- Klauder wavelet
- Ormsby wavelet