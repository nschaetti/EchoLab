"""
2D Acoustic Wave Propagation Simulation with Absorbing Boundary Conditions.

This module implements a finite-difference time-domain scheme for acoustic
wave propagation in heterogeneous media. The numerical recipe follows a
physicist's mindset: we watch the pressure field evolve while damping the
edges to emulate an infinite half-space.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import numpy as np


class PressureField(ABC):
    """
    Abstract base class for pressure field simulations.
    
    This class defines the interface for pressure field simulations.
    """
    
    @abstractmethod
    def simulate(self, **kwargs):
        """
        Simulate the pressure field.
        
        Args:
            **kwargs: Additional arguments for the simulation.
            
        Returns:
            The simulation results.
        """
        pass


class AcousticPressureField(PressureField):
    """
    Implementation of acoustic pressure field simulation.
    
    This class implements the finite-difference time-domain scheme for acoustic
    wave propagation in heterogeneous media.
    """
    
    @staticmethod
    def pad_velocity_model(
        velocity_model: np.ndarray,
        boundary_cell_count: int
    ) -> np.ndarray:
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

    @staticmethod
    def extend_source_wavelet(
        source_signal: np.ndarray,
        num_time_steps: int
    ) -> np.ndarray:
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

    @staticmethod
    def map_coordinates_to_grid_indices(
        source_x_m: float,
        source_z_m: float,
        receiver_x_m: np.ndarray,
        receiver_z_m: np.ndarray,
        grid_spacing: float,
        boundary_cell_count: int
    ) -> Tuple[int, int, np.ndarray, np.ndarray]:
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

    @staticmethod
    def compute_absorbing_boundary_mask(
        padded_velocity_model: np.ndarray,
        boundary_cell_count: int,
        grid_spacing: float
    ) -> np.ndarray:
        """
        Build a quadratic damping mask mimicking a viscous sponge boundary.

        This function creates a damping profile that attenuates waves near the
        edges of the computational domain, preventing reflections from the
        artificial boundaries.

        Args:
            padded_velocity_model (numpy.ndarray): Velocity model with boundary padding.
            boundary_cell_count (int): Number of absorbing cells on each edge.
            grid_spacing (float): Spatial sampling interval (m).

        Returns:
            numpy.ndarray: Damping coefficients for each grid point.
        """
        # Unpack dimensions of the padded velocity model
        nz, nx = padded_velocity_model.shape

        # Initialize the damping mask with ones (no damping in the interior)
        damping_mask = np.ones((nz, nx), dtype=np.float64)

        # Compute the maximum velocity for scaling the damping profile
        max_velocity = np.max(padded_velocity_model)

        # Compute the damping factor based on the maximum velocity and grid spacing
        # This is a heuristic that works well in practice
        damping_factor = 0.015 * max_velocity / grid_spacing

        # Create coordinate arrays for the grid
        z_coords = np.arange(nz)
        x_coords = np.arange(nx)

        # Compute distance from each boundary
        dist_from_top = z_coords.reshape(-1, 1)
        dist_from_bottom = (nz - 1 - z_coords).reshape(-1, 1)
        dist_from_left = x_coords.reshape(1, -1)
        dist_from_right = (nx - 1 - x_coords).reshape(1, -1)

        # Apply quadratic damping profile to each boundary region
        # Top boundary
        top_region = dist_from_top < boundary_cell_count
        damping_mask[top_region] *= 1.0 - damping_factor * (
            (boundary_cell_count - dist_from_top[top_region]) / boundary_cell_count
        ) ** 2

        # Bottom boundary
        bottom_region = dist_from_bottom < boundary_cell_count
        damping_mask[bottom_region] *= 1.0 - damping_factor * (
            (boundary_cell_count - dist_from_bottom[bottom_region]) / boundary_cell_count
        ) ** 2

        # Left boundary
        left_region = dist_from_left < boundary_cell_count
        damping_mask[left_region] *= 1.0 - damping_factor * (
            (boundary_cell_count - dist_from_left[left_region]) / boundary_cell_count
        ) ** 2

        # Right boundary
        right_region = dist_from_right < boundary_cell_count
        damping_mask[right_region] *= 1.0 - damping_factor * (
            (boundary_cell_count - dist_from_right[right_region]) / boundary_cell_count
        ) ** 2

        return damping_mask

    def simulate(
        self,
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
        apply_free_surface: bool = False,
        callback: Optional[Callable] = None
    ) -> dict:
        """
        Simulate acoustic wave propagation using a finite-difference scheme.

        This function implements a 2nd-order finite-difference time-domain method
        for the acoustic wave equation. It includes absorbing boundary conditions
        to minimize reflections from the edges of the computational domain.

        Args:
            velocity_model (numpy.ndarray): 2D velocity model (m/s).
            absorbing_boundary_thickness (int): Number of absorbing cells on each edge.
            grid_spacing (float): Spatial sampling interval (m).
            num_time_steps (int): Number of time steps to simulate.
            time_step (float): Temporal sampling interval (s).
            source_wavelet (numpy.ndarray): Source time function.
            source_x_m (float): Source horizontal position (m).
            source_z_m (float): Source depth (m).
            receiver_x_m (array_like): Receiver horizontal positions (m).
            receiver_z_m (array_like): Receiver depths (m).
            apply_free_surface (bool, optional): Whether to apply free surface boundary
                condition at the top. Defaults to False.
            callback (callable, optional): Function to call after each time step.
                Should accept (pressure_field, time_step_index, time_step, boundary_cell_count).

        Returns:
            dict: Simulation results containing:
                - 'receiver_data': Recorded pressure at receiver locations
                - 'velocity_model': The input velocity model
                - 'padded_velocity_model': Velocity model with boundary padding
                - 'grid_spacing': Spatial sampling interval
                - 'time_step': Temporal sampling interval
                - 'source_position': (source_x_idx, source_z_idx)
                - 'receiver_positions': (receiver_x_idx, receiver_z_idx)
                - 'boundary_cell_count': Number of absorbing cells on each edge
        """
        # Pad the velocity model with boundary cells
        padded_velocity_model = self.pad_velocity_model(
            velocity_model, absorbing_boundary_thickness
        )

        # Ensure the source wavelet spans the entire simulation time
        source_wavelet = self.extend_source_wavelet(source_wavelet, num_time_steps)

        # Map physical coordinates to grid indices
        source_x_idx, source_z_idx, receiver_x_idx, receiver_z_idx = self.map_coordinates_to_grid_indices(
            source_x_m, source_z_m, receiver_x_m, receiver_z_m,
            grid_spacing, absorbing_boundary_thickness
        )

        # Compute the absorbing boundary mask
        damping_mask = self.compute_absorbing_boundary_mask(
            padded_velocity_model, absorbing_boundary_thickness, grid_spacing
        )

        # Extract dimensions of the padded velocity model
        nz, nx = padded_velocity_model.shape

        # Initialize pressure fields for current and previous time steps
        p_current = np.zeros((nz, nx), dtype=np.float64)
        p_previous = np.zeros((nz, nx), dtype=np.float64)

        # Precompute squared velocities multiplied by squared time step
        # This is a common optimization in FDTD simulations
        velocity_factor = (padded_velocity_model * time_step) ** 2

        # Initialize array to store receiver data
        receiver_data = np.zeros((num_time_steps, len(receiver_x_idx)), dtype=np.float64)

        # Main time-stepping loop
        for time_idx in range(num_time_steps):
            # Store current pressure at receiver locations
            for i, (rx, rz) in enumerate(zip(receiver_x_idx, receiver_z_idx)):
                receiver_data[time_idx, i] = p_current[rz, rx]

            # Compute next pressure field using finite-difference stencil
            p_next = np.zeros_like(p_current)

            # Interior points (excluding boundaries)
            p_next[1:-1, 1:-1] = (
                2 * p_current[1:-1, 1:-1]
                - p_previous[1:-1, 1:-1]
                + velocity_factor[1:-1, 1:-1] * (
                    (p_current[2:, 1:-1] - 2 * p_current[1:-1, 1:-1] + p_current[:-2, 1:-1]) / grid_spacing**2
                    + (p_current[1:-1, 2:] - 2 * p_current[1:-1, 1:-1] + p_current[1:-1, :-2]) / grid_spacing**2
                )
            )

            # Apply free surface boundary condition if requested
            if apply_free_surface:
                # Set vertical derivative to zero at the top boundary
                p_next[1, 1:-1] = (
                    2 * p_current[1, 1:-1]
                    - p_previous[1, 1:-1]
                    + velocity_factor[1, 1:-1] * (
                        (2 * p_current[2, 1:-1] - 2 * p_current[1, 1:-1]) / grid_spacing**2
                        + (p_current[1, 2:] - 2 * p_current[1, 1:-1] + p_current[1, :-2]) / grid_spacing**2
                    )
                )

            # Add source term
            p_next[source_z_idx, source_x_idx] += source_wavelet[time_idx]

            # Apply absorbing boundary conditions
            p_next *= damping_mask

            # Update pressure fields for next time step
            p_previous = p_current.copy()
            p_current = p_next.copy()

            # Call the callback function if provided
            if callback is not None:
                callback(p_current, time_idx, time_step, absorbing_boundary_thickness)

        # Return simulation results
        return {
            'receiver_data': receiver_data,
            'velocity_model': velocity_model,
            'padded_velocity_model': padded_velocity_model,
            'grid_spacing': grid_spacing,
            'time_step': time_step,
            'source_position': (source_x_idx, source_z_idx),
            'receiver_positions': (receiver_x_idx, receiver_z_idx),
            'boundary_cell_count': absorbing_boundary_thickness
        }


def strip_absorbing_boundary(
    pressure_field: np.ndarray,
    boundary_cell_count: int
) -> np.ndarray:
    """
    Remove the absorbing boundary cells from a pressure field.

    Args:
        pressure_field (numpy.ndarray): Pressure field with boundary cells.
        boundary_cell_count (int): Number of boundary cells on each edge.

    Returns:
        numpy.ndarray: Pressure field without boundary cells.
    """
    return pressure_field[
        boundary_cell_count:-boundary_cell_count,
        boundary_cell_count:-boundary_cell_count
    ]