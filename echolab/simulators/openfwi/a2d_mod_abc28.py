"""
2D Acoustic Wave Propagation Simulation with Absorbing Boundary Conditions.

This module implements a finite-difference time-domain scheme for acoustic
wave propagation in heterogeneous media. The numerical recipe follows a
physicist's mindset: we watch the pressure field evolve while damping the
edges to emulate an infinite half-space.
"""

import numpy as np


def pad_velocity_model(velocity_model, boundary_cell_count):
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


def extend_source_wavelet(source_signal, num_time_steps):
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

    extended_wavelet = np.zeros(num_time_steps, dtype=source_signal.dtype)
    extended_wavelet[:source_signal.size] = source_signal
    return extended_wavelet


def map_coordinates_to_grid_indices(
    source_x_m,
    source_z_m,
    receiver_x_m,
    receiver_z_m,
    grid_spacing,
    boundary_cell_count,
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
    receiver_z_idx += (np.abs(receiver_z_m) < 0.5).astype(int)

    return source_x_idx, source_z_idx, receiver_x_idx, receiver_z_idx


def compute_absorbing_boundary_mask(padded_velocity_model, boundary_cell_count, grid_spacing):
    """
    Build a quadratic damping mask mimicking a viscous sponge boundary.

    Args:
        padded_velocity_model (numpy.ndarray): Velocity model including padding.
        boundary_cell_count (int): Number of absorbing cells at each boundary.
        grid_spacing (float): Spatial sampling interval (m).

    Returns:
        numpy.ndarray: Damping coefficients applied during time stepping.
    """
    absorbing_mask = np.zeros_like(padded_velocity_model)

    if boundary_cell_count <= 0:
        return absorbing_mask

    nz_tot, nx_tot = padded_velocity_model.shape
    velocity_min = np.min(padded_velocity_model)

    absorbing_thickness = max((boundary_cell_count - 1) * grid_spacing, grid_spacing)
    damping_scale = 3.0 * velocity_min * np.log(1e7) / (2.0 * absorbing_thickness)

    edge_profile = damping_scale * ((np.arange(boundary_cell_count) * grid_spacing / absorbing_thickness) ** 2)

    for depth_index in range(nz_tot):
        absorbing_mask[depth_index, 0:boundary_cell_count] = edge_profile[::-1]
        absorbing_mask[depth_index, nx_tot - boundary_cell_count :] = edge_profile

    interior_start = boundary_cell_count
    interior_end = nx_tot - boundary_cell_count
    for lateral_index in range(interior_start, interior_end):
        absorbing_mask[0:boundary_cell_count, lateral_index] = edge_profile[::-1]
        absorbing_mask[nz_tot - boundary_cell_count :, lateral_index] = edge_profile

    return absorbing_mask


def simulate_acoustic_wavefield(
    velocity_model,
    absorbing_boundary_thickness,
    grid_spacing,
    num_time_steps,
    time_step,
    source_wavelet,
    source_x_m,
    source_z_m,
    receiver_x_m,
    receiver_z_m,
    apply_free_surface,
    callback=None,
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
    receiver_x_m = np.asarray(receiver_x_m)
    receiver_z_m = np.asarray(receiver_z_m)
    receiver_count = receiver_x_m.size

    receiver_pressure_traces = np.zeros((num_time_steps, receiver_count))

    # 8th-order accurate spatial stencil coefficients.
    STENCIL_C1 = -205.0 / 72.0
    STENCIL_C2 = 8.0 / 5.0
    STENCIL_C3 = -1.0 / 5.0
    STENCIL_C4 = 8.0 / 315.0
    STENCIL_C5 = -1.0 / 560.0

    padded_velocity = pad_velocity_model(velocity_model, absorbing_boundary_thickness)
    absorbing_boundary_mask = compute_absorbing_boundary_mask(
        padded_velocity, absorbing_boundary_thickness, grid_spacing
    )

    courant_number_squared = (padded_velocity * time_step / grid_spacing) ** 2
    damping_profile = absorbing_boundary_mask * time_step

    current_step_multiplier = 2 + 2 * STENCIL_C1 * courant_number_squared - damping_profile
    previous_step_multiplier = 1 - damping_profile
    source_weighting = (padded_velocity * time_step) ** 2

    source_time_series = extend_source_wavelet(source_wavelet, num_time_steps)
    (
        source_x_index,
        source_z_index,
        receiver_x_indices,
        receiver_z_indices,
    ) = map_coordinates_to_grid_indices(
        source_x_m,
        source_z_m,
        receiver_x_m,
        receiver_z_m,
        grid_spacing,
        absorbing_boundary_thickness,
    )

    previous_pressure_field = np.zeros_like(padded_velocity)
    current_pressure_field = np.zeros_like(padded_velocity)

    for time_step_index in range(num_time_steps):
        # Apply the finite-difference stencil capturing the discrete Laplacian.
        laplacian = (
            STENCIL_C2
            * (
                np.roll(current_pressure_field, 1, axis=1)
                + np.roll(current_pressure_field, -1, axis=1)
                + np.roll(current_pressure_field, 1, axis=0)
                + np.roll(current_pressure_field, -1, axis=0)
            )
            + STENCIL_C3
            * (
                np.roll(current_pressure_field, 2, axis=1)
                + np.roll(current_pressure_field, -2, axis=1)
                + np.roll(current_pressure_field, 2, axis=0)
                + np.roll(current_pressure_field, -2, axis=0)
            )
            + STENCIL_C4
            * (
                np.roll(current_pressure_field, 3, axis=1)
                + np.roll(current_pressure_field, -3, axis=1)
                + np.roll(current_pressure_field, 3, axis=0)
                + np.roll(current_pressure_field, -3, axis=0)
            )
            + STENCIL_C5
            * (
                np.roll(current_pressure_field, 4, axis=1)
                + np.roll(current_pressure_field, -4, axis=1)
                + np.roll(current_pressure_field, 4, axis=0)
                + np.roll(current_pressure_field, -4, axis=0)
            )
        )

        next_pressure_field = (
            current_step_multiplier * current_pressure_field
            - previous_step_multiplier * previous_pressure_field
            + courant_number_squared * laplacian
        )

        # Inject the volumetric source term at the physical shot location.
        next_pressure_field[source_z_index, source_x_index] += (
            source_weighting[source_z_index, source_x_index] * source_time_series[time_step_index]
        )

        if apply_free_surface:
            # Free-surface: null pressure at the air/earth interface and antisymmetry beneath.
            free_surface_depth = absorbing_boundary_thickness
            next_pressure_field[free_surface_depth, :] = 0.0
            next_pressure_field[free_surface_depth - 1 : free_surface_depth - 5 : -1, :] = -next_pressure_field[
                free_surface_depth + 1 : free_surface_depth + 5, :
            ]

        if callback is not None:
            callback(next_pressure_field, time_step_index, time_step, absorbing_boundary_thickness)

        for receiver_index in range(receiver_count):
            receiver_pressure_traces[time_step_index, receiver_index] = next_pressure_field[
                receiver_z_indices[receiver_index], receiver_x_indices[receiver_index]
            ]

        previous_pressure_field, current_pressure_field = current_pressure_field, next_pressure_field

    return receiver_pressure_traces
