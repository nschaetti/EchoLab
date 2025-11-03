# Implementation Notes: Acoustic Field Simulation Refactoring

## Overview

This document describes the implementation of the acoustic field simulation code in the `echolab/modeling` subpackage. The implementation follows the requirements specified in the issue description:

1. Move the acoustic field simulation code to the subpackage `echolab/modeling`
2. Use classes with design patterns to build objects for pressure field, noise source, and simulator
3. For openfwi, keep the Ricker wave
4. For openanfwi, create brown, pink noises and random Ricker waves

## Design Patterns Used

### Strategy Pattern

The Strategy pattern is used to define a family of algorithms, encapsulate each one, and make them interchangeable. This pattern is used in three main areas:

1. **Pressure Field Simulation**:
   - `PressureField` abstract base class defines the interface
   - `AcousticPressureField` provides a concrete implementation

2. **Noise Sources**:
   - `NoiseSource` abstract base class defines the interface
   - Concrete implementations include `RickerWavelet`, `RandomRickerWavelet`, `BrownNoise`, `PinkNoise`

3. **Simulators**:
   - `Simulator` abstract base class defines the interface
   - Concrete implementations include `OpenFWISimulator` and `OpenANFWISimulator`

This pattern allows for easy extension with new algorithms in the future, such as new types of pressure fields, noise sources, or simulators.

### Composite Pattern

The Composite pattern is used to compose objects into tree structures to represent part-whole hierarchies. This pattern is used in:

1. **Noise Sources**:
   - `CompositeNoiseSource` allows combining multiple noise sources with different weights
   - This is particularly useful for openanfwi, which uses a combination of brown noise, pink noise, and random Ricker waves

### Factory Method Pattern

The Factory Method pattern is used to create objects without specifying the exact class to create. This pattern is used in:

1. **Simulator Initialization**:
   - `OpenFWISimulator.__init__` creates default objects if none are provided
   - `OpenANFWISimulator.__init__` creates a composite noise source with brown, pink, and random Ricker noise if none is provided

### Dependency Injection

Dependency Injection is used to pass dependencies to objects rather than having them create their own. This pattern is used in:

1. **Simulator Construction**:
   - `Simulator.__init__` takes a pressure field and a noise source as parameters
   - This allows for flexible configuration and easier testing

## Implementation Details

### Pressure Field (`acoustic_field.py`)

The `PressureField` abstract base class defines the interface for pressure field simulations with an abstract `simulate` method. The `AcousticPressureField` class provides a concrete implementation of this interface, implementing the finite-difference time-domain scheme for acoustic wave propagation.

Key methods include:
- `pad_velocity_model`: Pads the velocity model with boundary cells
- `extend_source_wavelet`: Ensures the source wavelet spans the entire simulation time
- `map_coordinates_to_grid_indices`: Converts physical coordinates to grid indices
- `compute_absorbing_boundary_mask`: Computes the absorbing boundary mask
- `simulate`: Runs the simulation and returns the results

### Noise Sources (`noise.py`)

The `NoiseSource` abstract base class defines the interface for noise sources with an abstract `generate` method. Concrete implementations include:

- `RickerWavelet`: Generates a Ricker wavelet with a specified frequency
- `RandomRickerWavelet`: Generates Ricker wavelets with random frequencies within a specified range
- `BrownNoise`: Generates brown noise (Brownian noise) by integrating white noise
- `PinkNoise`: Generates pink noise (1/f noise) using frequency domain filtering
- `CompositeNoiseSource`: Combines multiple noise sources with different weights

### Simulators (`simulator.py`)

The `Simulator` abstract base class defines the interface for simulators with abstract `simulate` and `visualize` methods. Concrete implementations include:

- `OpenFWISimulator`: Implements the openfwi simulator using the Ricker wavelet
- `OpenANFWISimulator`: Implements the openanfwi simulator using brown noise, pink noise, and random Ricker waves

Both simulators use composition to combine a pressure field and a noise source, and provide sensible defaults if none are provided.

## Benefits of the New Implementation

1. **Modularity**: The code is now more modular, with clear separation of concerns between pressure fields, noise sources, and simulators.

2. **Extensibility**: New pressure fields, noise sources, and simulators can be easily added by implementing the appropriate interfaces.

3. **Reusability**: Common functionality is now in a central location and can be reused across different simulators.

4. **Configurability**: The simulators can be configured with different pressure fields and noise sources, allowing for more flexible experimentation.

5. **Maintainability**: The code is now more maintainable, with clear interfaces and responsibilities.

## Future Improvements

1. **Additional Noise Sources**: More noise sources could be added, such as white noise, blue noise, or violet noise.

2. **Additional Pressure Fields**: Different pressure field simulation algorithms could be implemented, such as higher-order finite-difference schemes or spectral methods.

3. **Parallelization**: The simulation code could be parallelized to take advantage of multiple cores or GPUs.

4. **Visualization Improvements**: The visualization code could be enhanced with more options and interactivity.

5. **Configuration System**: A more comprehensive configuration system could be implemented to make it easier to configure and run simulations.