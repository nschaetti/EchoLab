
import numpy as np


def compute_absorbing_boundary_coefficients(velocity_field, boundary_cell_count, grid_spacing):
    """
    Build a quadratic absorbing boundary mask for a finite-difference grid.

    Args:
        velocity_field (numpy.ndarray): Velocity model including the padding (m/s).
        boundary_cell_count (int): Number of absorbing cells on each edge.
        grid_spacing (float): Spatial sampling interval (m).

    Returns:
        numpy.ndarray: Damping coefficients ramping up toward the edges.
    """
    padded_depth, padded_width = velocity_field.shape
    velocity_minimum = np.min(velocity_field)

    interior_depth = padded_depth - 2 * boundary_cell_count
    interior_width = padded_width - 2 * boundary_cell_count

    absorbing_thickness = max((boundary_cell_count - 1) * grid_spacing, grid_spacing)
    damping_scale = 3.0 * velocity_minimum * np.log(1e7) / (2.0 * absorbing_thickness)

    damping_profile = damping_scale * ((np.arange(boundary_cell_count) * grid_spacing / absorbing_thickness) ** 2)
    absorbing_coefficients = np.zeros_like(velocity_field)

    for depth_index in range(padded_depth):
        absorbing_coefficients[depth_index, 0:boundary_cell_count] = damping_profile[::-1]
        absorbing_coefficients[depth_index, interior_width + boundary_cell_count :] = damping_profile

    for width_index in range(boundary_cell_count, boundary_cell_count + interior_width):
        absorbing_coefficients[0:boundary_cell_count, width_index] = damping_profile[::-1]
        absorbing_coefficients[interior_depth + boundary_cell_count :, width_index] = damping_profile

    return absorbing_coefficients
