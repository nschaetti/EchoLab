from __future__ import annotations

"""
Python implementation of the surfdisp96 subroutine from surfdisp96.f90.

This module provides a Python implementation of the surfdisp96 subroutine
from the surfdisp96 Fortran code, which is responsible for calculating
surface wave dispersion curves.
"""

from typing import Tuple, List, Union, Optional, Any
import numpy as np
import torch
import tqdm
import compearth.extensions.surfdisp2k25 as sd2k25


def dispsurf2k25_simulator(
        model_parameters: torch.Tensor,
        p_min: float,
        p_max: float,
        target_period_count: int,
        spherical_model_flag: int = 0,
        wave_type: int = 2,
        mode_number: int = 1,
        group_velocity_option: int = 1,
        dtype: torch.dtype = torch.float32,
        progress: bool = False,
) -> torch.Tensor:
    """
    Compute dispersion curves using the dispsurf2k25 simulator.
    If target_period_count > 60, interpolation is performed to match the requested number of periods.

    Parameters
    ----------
    model_parameters : torch.Tensor
        Model parameters [n_layers, vpvs, h_1...h_Nmax, vs_1...vs_Nmax], shape (B, D)
    p_min, p_max : float
        Minimum and maximum periods (s)
    target_period_count : int
        Desired number of output periods (interpolated if > 60)
    spherical_model_flag, wave_type, mode_number, group_velocity_option : int
        Physical echolab flags (see surfdisp96 documentation)
    dtype : torch.dtype
        Output tensor type (default: float32)
    progress : bool
        If True, display a tqdm progress bar.

    Returns
    -------
    torch.Tensor
        Simulated dispersion curves, shape (B, target_period_count)
    """

    batch_size = model_parameters.shape[0]
    model_parameters_np = model_parameters.detach().cpu().numpy()

    # The legacy dispersion kernel only supports 60 distinct periods,
    # so we evaluate on that native grid before interpolating to the user's request.
    internal_period_count = 60
    internal_periods = np.linspace(p_min, p_max, internal_period_count)
    dispersion_curves_internal = np.zeros((batch_size, internal_period_count))

    # Progress bar
    simulation_progress = tqdm.tqdm(
        range(batch_size),
        desc="Running dispsurf2k25",
        disable=not progress
    )

    for batch_index in simulation_progress:
        layer_thickness_buffer_km = np.zeros(100)
        p_velocity_buffer_km_s = np.zeros(100)
        s_velocity_buffer_km_s = np.zeros(100)
        density_buffer_g_cm3 = np.zeros(100)

        max_layer_count = (model_parameters_np.shape[1] - 2) // 2
        active_layer_count = int(model_parameters_np[batch_index, 0])
        vp_to_vs_ratio = model_parameters_np[batch_index, 1]

        raw_layer_thicknesses = model_parameters_np[batch_index, 2:max_layer_count + 2]
        raw_shear_velocities = model_parameters_np[batch_index, max_layer_count + 2:]

        # Convert Vs to Vp via the supplied Vp/Vs ratio and infer density from Gardner's relation.
        p_wave_velocities = raw_shear_velocities * vp_to_vs_ratio
        densities = 0.32 + 0.77 * p_wave_velocities

        # Populate the fixed-size buffers expected by the Fortran-inspired backend.
        layer_thickness_buffer_km[:active_layer_count] = raw_layer_thicknesses[:active_layer_count]
        p_velocity_buffer_km_s[:active_layer_count] = p_wave_velocities[:active_layer_count]
        s_velocity_buffer_km_s[:active_layer_count] = raw_shear_velocities[:active_layer_count]
        density_buffer_g_cm3[:active_layer_count] = densities[:active_layer_count]

        # Run the core dispersion solver for this 1-D stratified Earth model.
        dispersion_curve, error_code = dispsurf2k25(
            layer_thicknesses_km=layer_thickness_buffer_km,
            s_velocity_profile_km_s=s_velocity_buffer_km_s,
            p_velocity_profile_km_s=p_velocity_buffer_km_s,
            density_profile_g_cm3=density_buffer_g_cm3,
            layer_count=active_layer_count,
            spherical_model_flag=spherical_model_flag,
            wave_type=wave_type,
            mode_number=mode_number,
            group_velocity_option=group_velocity_option,
            period_count=internal_period_count,
            periods_s=internal_periods,
        )

        if error_code != 0:
            raise RuntimeError(
                f"Simulation {batch_index} failed with error: {error_code}"
            )
        # end if

        dispersion_curves_internal[batch_index, :] = dispersion_curve
    # end for

    # Interpolate to the requested sampling if the user asked for a different period count.
    if target_period_count != internal_period_count:
        target_periods = np.linspace(p_min, p_max, target_period_count)
        dispersion_curves = np.zeros((batch_size, target_period_count))
        for batch_index in range(batch_size):
            dispersion_curves[batch_index, :] = np.interp(
                target_periods,
                internal_periods,
                dispersion_curves_internal[batch_index, :]
            )
        # end for
    else:
        dispersion_curves = dispersion_curves_internal
    # end if

    return torch.from_numpy(dispersion_curves).to(dtype)
# end dispsurf2k25_simulator


def dispsurf2k25(
        layer_thicknesses_km: np.ndarray,
        p_velocity_profile_km_s: np.ndarray,
        s_velocity_profile_km_s: np.ndarray,
        density_profile_g_cm3: np.ndarray,
        layer_count: int,
        spherical_model_flag: int,
        wave_type: int,
        mode_number: int,
        group_velocity_option: int,
        period_count: int,
        periods_s: np.ndarray
) -> Tuple[np.ndarray, int]:
    """
    Calculate surface wave dispersion curves.
    
    This is a pure Python implementation of the surfdisp96 Fortran subroutine.
    
    Parameters
    ----------
    layer_thicknesses_km : array_like
        Layer thicknesses in km.
    p_velocity_profile_km_s : array_like
        P-wave velocities in km/s.
    s_velocity_profile_km_s : array_like
        S-wave velocities in km/s.
    density_profile_g_cm3 : array_like
        Densities in g/cm^3.
    spherical_model_flag : int
        Flag for spherical earth model: 0 for flat earth, 1 for spherical earth.
    wave_type : int
        Wave type: 1 for Love waves, 2 for Rayleigh waves.
    mode_number : int
        Mode number to calculate.
    group_velocity_option : int
        Flag for group velocity calculation: 0 for phase velocity only, 1 for phase and group velocity.
    period_count : int
        Number of periods to calculate.
    periods_s : array_like
        Periods in seconds.
    
    Returns
    -------
    tuple
        A tuple containing:
        - computed_velocities: array_like, calculated phase or group velocities in km/s.
        - error_code: int, error code (0 for success, 1 for error).
    """
    # Parameters
    LER = 0
    LIN = 5
    LOT = 6
    NL = 100
    NLAY = 100
    NL2 = NL + NL
    NP = 60

    # Check sizes
    assert layer_thicknesses_km.ndim == 1 and layer_thicknesses_km.shape[0] == NLAY, f"layer_thicknesses_km must be 1dim (but is {layer_thicknesses_km.ndim}), and shape {NLAY} (but is {layer_thicknesses_km.shape})."
    assert p_velocity_profile_km_s.ndim == 1 and p_velocity_profile_km_s.shape[0] == NLAY, f"p_velocity_profile_km_s must be 1dim (but is {p_velocity_profile_km_s.ndim}), and shape {NLAY} (but is {p_velocity_profile_km_s.shape})."
    assert s_velocity_profile_km_s.ndim == 1 and s_velocity_profile_km_s.shape[0] == NLAY, f"s_velocity_profile_km_s must be 1dim (but is {s_velocity_profile_km_s.ndim}), and shape {NLAY} (but is {s_velocity_profile_km_s.shape})."
    assert density_profile_g_cm3.ndim == 1 and density_profile_g_cm3.shape[0] == NLAY, f"density_profile_g_cm3 must be 1dim (but is {density_profile_g_cm3.ndim}), and shape {NLAY} (but is {density_profile_g_cm3.shape})."
    assert periods_s.ndim == 1 and periods_s.shape[0] == NP, f"periods_s must be 1dim (but is {periods_s.ndim}), and shape {NP} (but is {periods_s.shape})."

    # Arguments
    computed_velocities = np.zeros(NP, dtype=np.float64)

    # Local variables
    phase_velocity_profile = np.zeros(NP, dtype=np.float64)
    group_velocity_profile = np.zeros(NP, dtype=np.float64)
    mutable_thicknesses = np.zeros(NL, dtype=np.float64)
    mutable_p_velocity = np.zeros(NL, dtype=np.float64)
    mutable_s_velocity = np.zeros(NL, dtype=np.float64)
    mutable_density = np.zeros(NL, dtype=np.float64)
    radius_transform = np.zeros(NL, dtype=np.float64)
    thickness_transform = np.zeros(NL, dtype=np.float64)
    shear_transform = np.zeros(NL, dtype=np.float64)
    verbosity_flags = np.zeros(3, dtype=np.int32)

    active_layer_count_local = layer_count
    spherical_flag = spherical_model_flag
    error_code = 0

    # Save current values
    # Copy the user model into working arrays so we can apply spherical corrections in-place.
    mutable_s_velocity[:active_layer_count_local] = s_velocity_profile_km_s[:active_layer_count_local]
    mutable_p_velocity[:active_layer_count_local] = p_velocity_profile_km_s[:active_layer_count_local]
    mutable_thicknesses[:active_layer_count_local] = layer_thicknesses_km[:active_layer_count_local]
    mutable_density[:active_layer_count_local] = density_profile_g_cm3[:active_layer_count_local]
    
    # Check if wave_type is valid
    if wave_type not in [1, 2]:
        raise ValueError(
            "wave_type must be 1 (Love waves) or 2 (Rayleigh waves)"
        )
    # end if
    
    # Set up which wave family we are interested in; the other branch is skipped entirely.
    if wave_type == 1:
        love_wave_period_limit = period_count
        rayleigh_wave_period_limit = 0
    elif wave_type == 2:
        love_wave_period_limit = 0
        rayleigh_wave_period_limit = period_count
    else:
        raise ValueError("wave_type must be 1 or 2")
    # end if

    verbosity_flags[1] = 0
    verbosity_flags[2] = 0
    
    # Constants
    sone0 = 1.500
    ddc0 = 0.005
    h0 = 0.005

    # Check for a water layer: Love waves cannot propagate through fluids.
    water_layer_flag = 1
    if mutable_s_velocity[0] <= 0.0:
        water_layer_flag = 2
    # end if
    two_pi = 2.0 * np.pi
    fractional_step = 1.0e-2  # Small safety factor used when nudging phase-velocity brackets

    # Apply spherical Earth corrections if requested, effectively mapping the flat model to great-circle geometry.
    if spherical_flag == 1:
        mutable_thicknesses, mutable_p_velocity, mutable_s_velocity, mutable_density, radius_transform, thickness_transform, shear_transform = sd2k25.sphere(
            ifunc=0,
            iflag=0,
            d=mutable_thicknesses,
            a=mutable_p_velocity,
            b=mutable_s_velocity,
            rho=mutable_density,
            rtp=radius_transform,
            dtp=thickness_transform,
            btp=shear_transform,
            mmax=active_layer_count_local,
            llw=water_layer_flag,
            twopi=two_pi
        )
    # end if
    
    min_velocity_layer_index = 0
    maximum_s_velocity = -1.0e20
    minimum_reference_velocity = 1.0e20
    initial_solution_is_solid = 1

    # Find the extremal shear (or longitudinal) velocities to guide the root search bounds.
    for i in range(layer_count):
        if 0.01 < mutable_s_velocity[i] < minimum_reference_velocity:
            minimum_reference_velocity = mutable_s_velocity[i]
            min_velocity_layer_index = i
            initial_solution_is_solid = 1
        elif mutable_s_velocity[i] <= 0.01 and mutable_p_velocity[i] < minimum_reference_velocity:
            minimum_reference_velocity = mutable_p_velocity[i]
            min_velocity_layer_index = i
            initial_solution_is_solid = 0
        # end if

        if mutable_s_velocity[i] > maximum_s_velocity:
            maximum_s_velocity = mutable_s_velocity[i]
        # end if
    # end for

    # Evaluate both Love (1) and Rayleigh (2) branches; skip whichever the caller did not request.
    for wave_component_id in [1, 2]:
        if wave_component_id == 1 and love_wave_period_limit <= 0:
            continue
        # end if

        if wave_component_id == 2 and rayleigh_wave_period_limit <= 0:
            continue
        # end if
        
        # Apply spherical earth transformation for the current wave type
        if spherical_flag == 1:
            mutable_thicknesses, mutable_p_velocity, mutable_s_velocity, mutable_density, radius_transform, thickness_transform, shear_transform = sd2k25.sphere(
                ifunc=wave_component_id,
                iflag=1,
                d=mutable_thicknesses,
                a=mutable_p_velocity,
                b=mutable_s_velocity,
                rho=mutable_density,
                rtp=radius_transform,
                dtp=thickness_transform,
                btp=shear_transform,
                mmax=active_layer_count_local,
                llw=water_layer_flag,
                twopi=two_pi
            )
        # end if

        phase_step_size = ddc0
        phase_step_multiplier = sone0
        group_velocity_window = h0
        
        if phase_step_multiplier < 0.01:
            phase_step_multiplier = 2.0
        # end if
        
        scaled_phase_step = phase_step_multiplier
        
        # Get starting value for phase velocity
        if initial_solution_is_solid == 0:
            # Water layer
            phase_velocity_seed = minimum_reference_velocity
        else:
            # Solid layer solves halfspace period equation
            phase_velocity_seed = sd2k25.getsolh(
                a=float(mutable_p_velocity[min_velocity_layer_index]),
                b=float(mutable_s_velocity[min_velocity_layer_index])
            )
        # end if

        # Back off slightly to start the phase velocity search safely below the half-space value
        phase_velocity_seed = 0.95 * phase_velocity_seed
        phase_velocity_seed = 0.90 * phase_velocity_seed
        current_phase_velocity = phase_velocity_seed
        phase_velocity_increment = phase_step_size
        phase_velocity_increment = abs(phase_velocity_increment)
        phase_velocity_estimate = current_phase_velocity
        phase_velocity_floor = current_phase_velocity
        
        # Initialize arrays
        group_velocity_profile[:period_count] = 0.0
        phase_velocity_profile[:period_count] = 0.0
        
        termination_period_index = 999

        # Loop over modes
        # Warning: mode_index is from [1...mode_number]
        for mode_index in range(1, mode_number + 1):
            period_start_index = 0
            period_end_index = period_count
            
            period_index = period_start_index

            # Loop over periods
            for period_index in range(period_start_index, period_end_index):
                if period_index >= termination_period_index:
                    break
                # end if

                target_period = periods_s[period_index]

                if group_velocity_option > 0:
                    period_upper = target_period / (1.0 + group_velocity_window)
                    period_lower = target_period / (1.0 - group_velocity_window)
                    working_period = period_upper
                else:
                    period_upper = target_period
                    period_lower = target_period
                    working_period = target_period
                # end if group_velocity_option > 0

                if period_index == period_start_index and mode_index == 1:
                    phase_velocity_estimate = current_phase_velocity
                    lower_phase_velocity_bound = current_phase_velocity
                    is_first_iteration = 1
                elif period_index == period_start_index and mode_index > 1:
                    phase_velocity_estimate = phase_velocity_profile[period_start_index] + fractional_step * phase_velocity_increment
                    lower_phase_velocity_bound = phase_velocity_estimate
                    is_first_iteration = 1
                elif period_index > period_start_index and mode_index > 1:
                    is_first_iteration = 0
                    lower_phase_velocity_bound = phase_velocity_profile[period_index] + fractional_step * phase_velocity_increment
                    phase_velocity_estimate = phase_velocity_profile[period_index - 1]
                    if phase_velocity_estimate < lower_phase_velocity_bound:
                        phase_velocity_estimate = lower_phase_velocity_bound
                    # end if
                elif period_index > period_start_index and mode_index == 1:
                    is_first_iteration = 0
                    phase_velocity_estimate = phase_velocity_profile[period_index - 1] - scaled_phase_step * phase_velocity_increment
                    lower_phase_velocity_bound = phase_velocity_floor
                else:
                    raise ValueError("Impossible to get initial phase velocity")
                # end if

                # Solve the dispersion equation for the phase velocity at this period.
                phase_velocity_estimate, iteration_status = sd2k25.getsol(
                    t1=working_period,
                    c1=phase_velocity_estimate,
                    clow=lower_phase_velocity_bound,
                    dc=phase_velocity_increment,
                    cm=phase_velocity_floor,
                    betmx=maximum_s_velocity,
                    ifunc=wave_component_id,
                    ifirst=is_first_iteration,
                    d=mutable_thicknesses,
                    a=mutable_p_velocity,
                    b=mutable_s_velocity,
                    rho=mutable_density,
                    rtp=radius_transform,
                    dtp=thickness_transform,
                    btp=shear_transform,
                    mmax=active_layer_count_local,
                    llw=water_layer_flag
                )

                if iteration_status == -1:
                    break
                # end if

                phase_velocity_profile[period_index] = phase_velocity_estimate

                if group_velocity_option > 0:
                    working_period = period_lower
                    is_first_iteration = 0
                    lower_phase_velocity_bound = group_velocity_profile[period_index] + fractional_step * phase_velocity_increment
                    phase_velocity_estimate = phase_velocity_estimate - scaled_phase_step * phase_velocity_increment

                    # Solve it again around the slightly shifted period to approximate group velocity.
                    phase_velocity_estimate, iteration_status = sd2k25.getsol(
                        t1=working_period,
                        c1=phase_velocity_estimate,
                        clow=lower_phase_velocity_bound,
                        dc=phase_velocity_increment,
                        cm=phase_velocity_floor,
                        betmx=maximum_s_velocity,
                        ifunc=wave_component_id,
                        ifirst=is_first_iteration,
                        d=mutable_thicknesses,
                        a=mutable_p_velocity,
                        b=mutable_s_velocity,
                        rho=mutable_density,
                        rtp=radius_transform,
                        dtp=thickness_transform,
                        btp=shear_transform,
                        mmax=active_layer_count_local,
                        llw=water_layer_flag
                    )

                    if iteration_status == -1:
                        phase_velocity_estimate = phase_velocity_profile[period_index]
                    # end if

                    group_velocity_profile[period_index] = phase_velocity_estimate
                else:
                    phase_velocity_estimate = 0.0
                # end if group_velocity_option

                phase_velocity_solution = phase_velocity_profile[period_index]
                group_velocity_candidate = phase_velocity_estimate

                if group_velocity_option == 0:
                    # Output only phase velocity
                    computed_velocities[period_index] = phase_velocity_solution
                else:
                    # Calculate group velocity and output phase and group velocities
                    group_velocity_value = (1.0 / period_upper - 1.0 / period_lower) / (1.0 / (period_upper * phase_velocity_solution) - 1.0 / (period_lower * group_velocity_candidate))
                    computed_velocities[period_index] = group_velocity_value
                # end if group_velocity_option
            # end for period_index

            # If we broke out of the loop early
            if verbosity_flags[wave_component_id] == 0 and mode_index <= 1:
                # raise RuntimeError(f"Error, loop finished to early")
                pass
            # end if
            
            termination_period_index = period_index
            
            # Set the remaining values to 0
            # for i in range(period_index, period_end_index):
            #     period_upper = periods_s[i]
            #     computed_velocities[i] = 0.0
            # # end for k...ie

        # end for mode_index 1...mode_number
    # end for wave_component_id
    
    return computed_velocities, error_code
# end def dispsurf2k25
