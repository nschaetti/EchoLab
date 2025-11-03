"""
2D Acoustic Wave Propagation Simulation with Absorbing Boundary Conditions.

This module implements a finite-difference time-domain scheme for acoustic
wave propagation in heterogeneous media. The numerical recipe follows a
physicist's mindset: we watch the pressure field evolve while damping the
edges to emulate an infinite half-space.
"""

import numpy as np


def pad_velocity_model(
    velocity_model: np.ndarray,
    boundary_cell_count: int
):
    """
    Pad the velocity model with a buffer of mirrored values.

    As seismologists we extend the computational grid so that the sponge layer
    fits around the experiment without altering the interior geology.

    Args:
        velocity_model (numpy.ndarray): Interior acoustic velocity model (m/s).
        boundary_cell_count (int): Number of absorbing cells on each edge.

    Returns:
        numpy.ndarray: Velocity model enlarged with boundary padding.
    """
    return np.pad(
        velocity_model,
        ((boundary_cell_count, boundary_cell_count), (boundary_cell_count, boundary_cell_count)),
        mode="edge",
    )
# end def pad_velocity_model


def extend_source_wavelet(
    source_signal: np.ndarray,
    num_time_steps: int
):
    """
    Ensure the source wavelet spans the entire simulation time.

    Args:
        source_signal (numpy.ndarray): Input source-time function.
        num_time_steps (int): Desired number of temporal samples.

    Returns:
        numpy.ndarray: Wavelet padded with zeros if needed.
    """
    source_signal = np.asarray(source_signal).flatten()
    if source_signal.size >= num_time_steps:
        return source_signal
    # end if

    extended_wavelet = np.zeros(num_time_steps, dtype=source_signal.dtype)
    extended_wavelet[:source_signal.size] = source_signal
    return extended_wavelet
# end def extend_source_wavelet


def map_coordinates_to_grid_indices(
    source_x_m: float,
    source_z_m: float,
    receiver_x_m: np.ndarray,
    receiver_z_m: np.ndarray,
    grid_spacing: float,
    boundary_cell_count: int
):
    """
    Convert physical source/receiver coordinates into grid indices.

    Args:
        source_x_m (float): Source horizontal position (m).
        source_z_m (float): Source depth (m).
        receiver_x_m (array_like): Receiver horizontal positions (m).
        receiver_z_m (array_like): Receiver depths (m).
        grid_spacing (float): Spatial sampling interval (m).
        boundary_cell_count (int): Number of padded cells at each edge.

    Returns:
        tuple: (source_x_idx, source_z_idx, receiver_x_idx, receiver_z_idx)
    """
    receiver_x_m = np.asarray(receiver_x_m)
    receiver_z_m = np.asarray(receiver_z_m)

    source_x_idx = int(round(source_x_m / grid_spacing)) + 1 + boundary_cell_count
    source_z_idx = int(round(source_z_m / grid_spacing)) + 1 + boundary_cell_count

    receiver_x_idx = (np.round(receiver_x_m / grid_spacing)).astype(int) + 1 + boundary_cell_count
    receiver_z_idx = (np.round(receiver_z_m / grid_spacing)).astype(int) + 1 + boundary_cell_count

    # Push grid indices one cell deeper when source/receivers sit at the surface.
    if abs(source_z_m) < 0.5:
        source_z_idx += 1
    # end if
    receiver_z_idx += (np.abs(receiver_z_m) < 0.5).astype(int)

    return source_x_idx, source_z_idx, receiver_x_idx, receiver_z_idx
# end def map_coordinates_to_grid_indices


def compute_absorbing_boundary_mask(
    padded_velocity_model: np.ndarray,
    boundary_cell_count: int,
    grid_spacing: float
):
    """
    Build a quadratic damping mask mimicking a viscous sponge boundary.

    Args:
        padded_velocity_model (numpy.ndarray): Velocity model including padding.
        boundary_cell_count (int): Number of absorbing cells at each boundary.
        grid_spacing (float): Spatial sampling interval (m).

    Returns:
        numpy.ndarray: Damping coefficients applied during time stepping.
    """
    # Initialize the absorbing mask with zeros (same shape as velocity model)
    # Programming: Creates an array of zeros with the same dimensions as the input model
    # Physics: Initially, no damping is applied anywhere in the domain
    absorbing_mask = np.zeros_like(padded_velocity_model)

    # Early return if no absorbing boundary is requested
    # Programming: Optimization to avoid unnecessary calculations
    # Physics: Without absorbing cells, waves will reflect from the domain boundaries
    if boundary_cell_count <= 0:
        return absorbing_mask
    # end if

    # Extract the dimensions of the padded velocity model
    # Programming: Unpacks the shape tuple into separate variables for clarity
    # Physics: Defines the total computational domain size including absorbing boundaries
    nz_tot, nx_tot = padded_velocity_model.shape
    
    # Find the minimum velocity in the model
    # Programming: Determines the smallest value in the velocity array
    # Physics: Used to calculate optimal damping parameters; slower velocities require 
    # gentler damping to prevent reflections
    velocity_min = np.min(padded_velocity_model)

    # Calculate the physical thickness of the absorbing layer in meters
    # Programming: Ensures the thickness is at least one grid cell
    # Physics: Defines the spatial extent of the absorbing boundary layer
    absorbing_thickness = max((boundary_cell_count - 1) * grid_spacing, grid_spacing)
    
    # Calculate the damping scale factor
    # Programming: Computes a coefficient that scales the strength of the damping
    # Physics: This formula is derived from the theory of perfectly matched layers (PML),
    # where 3.0 is a tuning parameter, log(1e7) represents a target reflection coefficient,
    # and the denominator relates to the wave transit time across the absorbing layer
    damping_scale = 3.0 * velocity_min * np.log(1e7) / (2.0 * absorbing_thickness)

    # Create the damping profile that increases quadratically from the interior toward the boundary
    # Programming: Generates an array of increasing values using vectorized operations
    # Physics: Quadratic increase provides a smooth transition that minimizes reflections;
    # stronger damping is applied at the outer edges of the computational domain
    edge_profile = damping_scale * ((np.arange(boundary_cell_count) * grid_spacing / absorbing_thickness) ** 2)

    # Apply the damping profile to the left and right boundaries for all depths
    # Programming: Iterates through each depth row and assigns damping values
    # Physics: Creates vertical absorbing strips on the left and right sides of the domain;
    # the reversed profile (::-1) ensures damping increases toward the outer boundary
    for depth_index in range(nz_tot):
        absorbing_mask[depth_index, 0:boundary_cell_count] = edge_profile[::-1]  # Left boundary (reversed profile)
        absorbing_mask[depth_index, nx_tot - boundary_cell_count:] = edge_profile  # Right boundary
    # end for

    # Define the interior region (excluding the left and right absorbing boundaries)
    # Programming: Calculates indices that define the non-absorbing lateral region
    # Physics: Identifies the portion of the domain where horizontal boundaries need to be applied
    # without overlapping with the already-defined vertical boundaries
    interior_start = boundary_cell_count
    interior_end = nx_tot - boundary_cell_count
    
    # Apply the damping profile to the top and bottom boundaries within the interior region
    # Programming: Iterates through each lateral position in the interior and assigns damping values
    # Physics: Creates horizontal absorbing strips on the top and bottom of the domain,
    # completing the absorbing frame around the computational area
    for lateral_index in range(interior_start, interior_end):
        absorbing_mask[0:boundary_cell_count, lateral_index] = edge_profile[::-1]  # Top boundary (reversed profile)
        absorbing_mask[nz_tot - boundary_cell_count:, lateral_index] = edge_profile  # Bottom boundary
    # end for

    return absorbing_mask
# end def compute_absorbing_boundary_mask


def simulate_acoustic_wavefield(
    velocity_model: np.ndarray,
    absorbing_boundary_thickness: int,
    grid_spacing: float,
    num_time_steps: int,
    time_step: float,
    source_wavelet: np.ndarray,
    source_x_m: float,
    source_z_m: float,
    receiver_x_m: np.ndarray,
    receiver_z_m: np.ndarray,
    apply_free_surface: bool,
    callback: callable = None
):
    """
    Propagate a 2D acoustic wavefield with an absorbing fringe.

    Args:
        velocity_model (numpy.ndarray): Acoustic velocity in m/s.
        absorbing_boundary_thickness (int): Width of the absorbing layer (cells).
        grid_spacing (float): Spatial sampling interval (m).
        num_time_steps (int): Number of temporal samples.
        time_step (float): Time sampling interval (s).
        source_wavelet (numpy.ndarray): Source-time signature.
        source_x_m (float): Source horizontal position (m).
        source_z_m (float): Source depth (m).
        receiver_x_m (array_like): Receiver horizontal positions (m).
        receiver_z_m (array_like): Receiver depths (m).
        apply_free_surface (bool): Toggle the free-surface boundary condition.
        callback (callable, optional): Inspection hook receiving wavefield snapshots.

    Returns:
        numpy.ndarray: Pressure traces recorded at the receivers (time, receiver).
    """
    #
    # STEP 1: Defining the problem
    # (x_i, z_i) = (i * delta x, j * delta z)
    # tn = n * delta t
    #

    # Convert receiver positions to numpy arrays for consistent handling
    # Programming: Ensures input arrays are numpy arrays for vectorized operations
    # Physics: Prepares receiver coordinates for later mapping to grid indices
    receiver_x_m = np.asarray(receiver_x_m)
    receiver_z_m = np.asarray(receiver_z_m)
    receiver_count = receiver_x_m.size

    # Initialize array to store pressure recordings at each receiver for all time steps
    # Programming: Pre-allocates memory for efficiency
    # Physics: Will store the pressure time series at each receiver location
    receiver_pressure_traces = np.zeros((num_time_steps, receiver_count))

    #
    # STEP 2: Defining the spatial stencil
    # Coefficient for 2D laplacian discrete approximation
    # 2D Laplacian => weighted sum of neighbour
    #

    # Define 8th-order accurate spatial finite-difference stencil coefficients
    # Programming: Constants used in the Laplacian approximation
    # Physics: Higher-order stencils reduce numerical dispersion, improving accuracy of wave propagation
    STENCIL_C1 = -205.0 / 72.0  # Central point coefficient
    STENCIL_C2 = 8.0 / 5.0      # 1-cell offset coefficient
    STENCIL_C3 = -1.0 / 5.0     # 2-cell offset coefficient
    STENCIL_C4 = 8.0 / 315.0    # 3-cell offset coefficient
    STENCIL_C5 = -1.0 / 560.0   # 4-cell offset coefficient

    #
    # STEP 3: border condition
    # Add margin to pressure field with absorbing condition.
    #

    # Extend the velocity model with absorbing boundary layers
    # Programming: Creates a padded array with additional cells around the original model
    # Physics: Provides space for implementing absorbing boundary conditions to prevent reflections
    padded_velocity = pad_velocity_model(
        velocity_model=velocity_model,
        boundary_cell_count=absorbing_boundary_thickness
    )

    # Compute damping coefficients for the absorbing boundary
    # Programming: Creates a mask with values between 0-1 that increase toward boundaries
    # Physics: Implements a sponge-like boundary that gradually attenuates waves to prevent reflections
    absorbing_boundary_mask = compute_absorbing_boundary_mask(
        padded_velocity_model=padded_velocity,
        boundary_cell_count=absorbing_boundary_thickness,
        grid_spacing=grid_spacing
    )

    #
    # STEP 4: Courant-Friedrichs-Lewy condition number
    # C = (v * delta t) / (delta x)
    # The wave cannot skip on cell
    #

    # Calculate the squared Courant number (stability parameter)
    # Programming: Vectorized computation for the entire grid
    # Physics: Relates to the CFL condition for numerical stability; should be ≤ 1 for stable simulation
    courant_number_squared = (padded_velocity * time_step / grid_spacing) ** 2
    
    # Scale the absorbing boundary mask by time step for damping application
    # Programming: Element-wise multiplication to prepare damping coefficients
    # Physics: Controls the rate of energy absorption at boundaries
    damping_profile = absorbing_boundary_mask * time_step

    #
    # STEP 5: temporal coefficients
    # Temporal discretization
    #

    # Compute coefficients for the finite-difference time-stepping scheme
    # Programming: Vectorized operations to prepare update equation coefficients
    # Physics: These coefficients implement the second-order accurate time discretization
    # of the acoustic wave equation with absorbing boundary conditions
    current_step_multiplier = 2 + 2 * STENCIL_C1 * courant_number_squared - damping_profile
    previous_step_multiplier = 1 - damping_profile
    
    # Calculate source amplitude scaling based on local velocity
    # Programming: Squared term used in source injection
    # Physics: Scales the source amplitude according to the local medium properties
    source_weighting = (padded_velocity * time_step) ** 2

    # Ensure source wavelet is long enough for the simulation
    # Programming: Extends or truncates the source signal to match simulation length
    # Physics: Provides the time-varying source amplitude for all simulation time steps
    source_time_series = extend_source_wavelet(
        source_signal=source_wavelet,
        num_time_steps=num_time_steps
    )

    # Convert physical coordinates (meters) to grid indices
    # Programming: Maps real-world positions to array indices
    # Physics: Locates source and receivers within the computational grid
    (
        source_x_index,
        source_z_index,
        receiver_x_indices,
        receiver_z_indices,
    ) = map_coordinates_to_grid_indices(
        source_x_m=source_x_m,
        source_z_m=source_z_m,
        receiver_x_m=receiver_x_m,
        receiver_z_m=receiver_z_m,
        grid_spacing=grid_spacing,
        boundary_cell_count=absorbing_boundary_thickness,
    )

    # Initialize pressure fields for time-stepping algorithm
    # Programming: Creates zero-filled arrays for the first two time steps
    # Physics: Implements initial conditions of zero pressure everywhere
    previous_pressure_field = np.zeros_like(padded_velocity)
    current_pressure_field = np.zeros_like(padded_velocity)

    # Main time-stepping loop
    # Programming: Iterates through all time steps to evolve the wavefield
    # Physics: Solves the acoustic wave equation using a second-order finite-difference scheme
    for time_step_index in range(num_time_steps):
        # Apply the finite-difference stencil to compute the Laplacian operator
        # Programming: Uses np.roll for efficient stencil application without explicit loops
        # Physics: Approximates ∇²p (the Laplacian of pressure) using an 8th-order accurate spatial discretization
        laplacian = (
            STENCIL_C2
            * (
                # 1-cell offsets in all four directions (east, west, north, south)
                np.roll(current_pressure_field, 1, axis=1)   # East shift
                + np.roll(current_pressure_field, -1, axis=1) # West shift
                + np.roll(current_pressure_field, 1, axis=0)  # South shift
                + np.roll(current_pressure_field, -1, axis=0) # North shift
            )
            + STENCIL_C3
            * (
                # 2-cell offsets in all four directions
                np.roll(current_pressure_field, 2, axis=1)
                + np.roll(current_pressure_field, -2, axis=1)
                + np.roll(current_pressure_field, 2, axis=0)
                + np.roll(current_pressure_field, -2, axis=0)
            )
            + STENCIL_C4
            * (
                # 3-cell offsets in all four directions
                np.roll(current_pressure_field, 3, axis=1)
                + np.roll(current_pressure_field, -3, axis=1)
                + np.roll(current_pressure_field, 3, axis=0)
                + np.roll(current_pressure_field, -3, axis=0)
            )
            + STENCIL_C5
            * (
                # 4-cell offsets in all four directions
                np.roll(current_pressure_field, 4, axis=1)
                + np.roll(current_pressure_field, -4, axis=1)
                + np.roll(current_pressure_field, 4, axis=0)
                + np.roll(current_pressure_field, -4, axis=0)
            )
        )

        # Update pressure field using the acoustic wave equation finite-difference scheme
        # Programming: Implements the core update equation for the next time step
        # Physics: This is the discretized form of the acoustic wave equation:
        # p(t+dt) = 2p(t) - p(t-dt) + c²dt²∇²p(t) with additional damping terms
        next_pressure_field = (
            current_step_multiplier * current_pressure_field
            - previous_step_multiplier * previous_pressure_field
            + courant_number_squared * laplacian
        )

        # Inject the source term at the specified location
        # Programming: Adds source amplitude to a single grid point
        # Physics: Implements a point source (monopole) that generates acoustic waves
        next_pressure_field[source_z_index, source_x_index] += (
            source_weighting[source_z_index, source_x_index] * source_time_series[time_step_index]
        )

        # Apply free-surface boundary condition if requested
        # Programming: Sets pressure to zero at surface and applies antisymmetry below
        # Physics: Simulates the air-earth interface where pressure must be zero (free surface)
        # and creates the effect of wave reflection with phase reversal
        if apply_free_surface:
            # Free-surface: null pressure at the air/earth interface and antisymmetry beneath.
            free_surface_depth = absorbing_boundary_thickness
            # Set pressure to zero at the free surface (Dirichlet boundary condition)
            next_pressure_field[free_surface_depth, :] = 0.0
            # Apply antisymmetry for cells below the free surface (mirror with sign change)
            # This creates the effect of reflection with phase reversal
            next_pressure_field[free_surface_depth - 1 : free_surface_depth - 5 : -1, :] = -next_pressure_field[
                free_surface_depth + 1 : free_surface_depth + 5, :
            ]
        # end if

        # Execute callback function if provided (for visualization or analysis)
        # Programming: Conditional execution of external function with current wavefield
        # Physics: Allows inspection of the wavefield at each time step (e.g., for visualization)
        if callback is not None:
            callback(
                next_pressure_field,
                time_step_index,
                time_step,
                absorbing_boundary_thickness
            )
        # end if

        # Record pressure values at receiver locations
        # Programming: Extracts values at specific indices and stores them in the output array
        # Physics: Simulates receivers recording pressure values at their physical locations
        for receiver_index in range(receiver_count):
            receiver_pressure_traces[time_step_index, receiver_index] = next_pressure_field[
                receiver_z_indices[receiver_index], receiver_x_indices[receiver_index]
            ]
        # end for

        # Cycle pressure fields for next time step (avoiding unnecessary array copies)
        # Programming: Efficient variable swapping to prepare for next iteration
        # Physics: Advances the simulation in time by one step
        previous_pressure_field, current_pressure_field = current_pressure_field, next_pressure_field
    # end for

    # Return the recorded pressure traces at all receiver locations
    # Programming: Returns the pre-allocated array now filled with simulation results
    # Physics: These are the synthetic seismograms that would be recorded by receivers
    return receiver_pressure_traces
# end def simulate_acoustic_wavefield
