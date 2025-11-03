"""
Acoustic Field Simulators.

This module provides classes for simulating acoustic fields using different
approaches and noise sources.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize

from .acoustic_field import AcousticPressureField, PressureField, strip_absorbing_boundary
from .noise import NoiseSource, RickerWavelet, RandomRickerWavelet, BrownNoise, PinkNoise, CompositeNoiseSource


class Simulator(ABC):
    """
    Abstract base class for acoustic field simulators.
    
    This class defines the interface for acoustic field simulators.
    """
    
    def __init__(
        self,
        pressure_field: PressureField,
        noise_source: NoiseSource
    ):
        """
        Initialize an acoustic field simulator.
        
        Args:
            pressure_field (PressureField): The pressure field implementation to use.
            noise_source (NoiseSource): The noise source to use.
        """
        self.pressure_field = pressure_field
        self.noise_source = noise_source
    
    @abstractmethod
    def simulate(self, **kwargs) -> Dict[str, Any]:
        """
        Run the simulation.
        
        Args:
            **kwargs: Additional arguments for the simulation.
            
        Returns:
            Dict[str, Any]: The simulation results.
        """
        pass
    
    @abstractmethod
    def visualize(self, results: Dict[str, Any], **kwargs):
        """
        Visualize the simulation results.
        
        Args:
            results (Dict[str, Any]): The simulation results.
            **kwargs: Additional arguments for visualization.
        """
        pass


class OpenFWISimulator(Simulator):
    """
    OpenFWI acoustic field simulator.
    
    This simulator uses the Ricker wavelet as the noise source and
    implements the full waveform inversion approach.
    """
    
    def __init__(
        self,
        pressure_field: Optional[PressureField] = None,
        noise_source: Optional[NoiseSource] = None,
        frequency: float = 25.0
    ):
        """
        Initialize an OpenFWI simulator.
        
        Args:
            pressure_field (PressureField, optional): The pressure field implementation to use.
                If None, an AcousticPressureField will be created.
            noise_source (NoiseSource, optional): The noise source to use.
                If None, a RickerWavelet with the specified frequency will be created.
            frequency (float, optional): The frequency for the Ricker wavelet if noise_source is None.
                Defaults to 25.0 Hz.
        """
        if pressure_field is None:
            pressure_field = AcousticPressureField()
        
        if noise_source is None:
            noise_source = RickerWavelet(frequency)
        
        super().__init__(pressure_field, noise_source)
    
    def simulate(
        self,
        velocity_model: np.ndarray,
        grid_spacing: float,
        time_step: float,
        num_time_steps: int,
        source_x_m: float,
        source_z_m: float,
        receiver_x_m: np.ndarray,
        receiver_z_m: np.ndarray,
        absorbing_boundary_thickness: int = 20,
        apply_free_surface: bool = False,
        store_wavefields: bool = False
    ) -> Dict[str, Any]:
        """
        Run the OpenFWI simulation.
        
        Args:
            velocity_model (np.ndarray): 2D velocity model (m/s).
            grid_spacing (float): Spatial sampling interval (m).
            time_step (float): Temporal sampling interval (s).
            num_time_steps (int): Number of time steps to simulate.
            source_x_m (float): Source horizontal position (m).
            source_z_m (float): Source depth (m).
            receiver_x_m (np.ndarray): Receiver horizontal positions (m).
            receiver_z_m (np.ndarray): Receiver depths (m).
            absorbing_boundary_thickness (int, optional): Number of absorbing cells on each edge.
                Defaults to 20.
            apply_free_surface (bool, optional): Whether to apply free surface boundary
                condition at the top. Defaults to False.
            store_wavefields (bool, optional): Whether to store the wavefields at each time step.
                Defaults to False.
                
        Returns:
            Dict[str, Any]: The simulation results.
        """
        # Generate the source wavelet
        source_wavelet, _ = self.noise_source.generate(time_step, num_time_steps)
        
        # Set up the callback for storing wavefields if requested
        wavefields = []
        
        if store_wavefields:
            def callback(pressure_field, time_step_index, _time_step, boundary_cell_count):
                # Strip the absorbing boundary from the pressure field
                interior_pressure = strip_absorbing_boundary(
                    pressure_field, boundary_cell_count
                )
                wavefields.append(interior_pressure.copy())
        else:
            callback = None
        
        # Run the simulation
        results = self.pressure_field.simulate(
            velocity_model=velocity_model,
            absorbing_boundary_thickness=absorbing_boundary_thickness,
            grid_spacing=grid_spacing,
            num_time_steps=num_time_steps,
            time_step=time_step,
            source_wavelet=source_wavelet,
            source_x_m=source_x_m,
            source_z_m=source_z_m,
            receiver_x_m=receiver_x_m,
            receiver_z_m=receiver_z_m,
            apply_free_surface=apply_free_surface,
            callback=callback
        )
        
        # Add the wavefields to the results if stored
        if store_wavefields:
            results['wavefields'] = np.array(wavefields)
        
        return results
    
    def visualize(
        self,
        results: Dict[str, Any],
        output_dir: Optional[Path] = None,
        fps: int = 20,
        velocity_alpha: float = 0.5,
        wavefield_alpha: Optional[float] = None
    ):
        """
        Visualize the OpenFWI simulation results.
        
        Args:
            results (Dict[str, Any]): The simulation results.
            output_dir (Path, optional): Directory to save the visualization.
                If None, the visualization will only be displayed.
            fps (int, optional): Frames per second for the animation. Defaults to 20.
            velocity_alpha (float, optional): Alpha value for the velocity model overlay.
                Defaults to 0.5.
            wavefield_alpha (float, optional): Alpha value for the wavefield.
                If None, no transparency is applied. Defaults to None.
                
        Returns:
            matplotlib.animation.Animation: The animation object if wavefields are available.
        """
        # Check if wavefields are available
        if 'wavefields' not in results:
            print("No wavefields available for visualization.")
            return None
        
        # Extract data from results
        wavefields = results['wavefields']
        velocity_model = results['velocity_model']
        grid_spacing = results['grid_spacing']
        
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up the axes
        extent = [0, velocity_model.shape[1] * grid_spacing,
                 velocity_model.shape[0] * grid_spacing, 0]
        
        # Plot the velocity model
        vel_im = ax.imshow(
            velocity_model,
            extent=extent,
            cmap='viridis',
            alpha=velocity_alpha
        )
        
        # Add a colorbar for the velocity model
        vel_cbar = plt.colorbar(vel_im, ax=ax)
        vel_cbar.set_label('Velocity (m/s)')
        
        # Initialize the wavefield plot
        vmin, vmax = np.min(wavefields), np.max(wavefields)
        abs_max = max(abs(vmin), abs(vmax))
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
        
        wave_im = ax.imshow(
            wavefields[0],
            extent=extent,
            cmap='seismic',
            norm=norm,
            alpha=wavefield_alpha
        )
        
        # Add a colorbar for the wavefield
        wave_cbar = plt.colorbar(wave_im, ax=ax)
        wave_cbar.set_label('Pressure')
        
        # Add labels and title
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Acoustic Wave Propagation')
        
        # Add a timestamp
        timestamp = ax.text(
            0.02, 0.02, f"Time: 0.000 s",
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        # Define the animation update function
        def update(frame_index):
            wave_im.set_array(wavefields[frame_index])
            timestamp.set_text(f"Time: {frame_index * results['time_step']:.3f} s")
            return wave_im, timestamp
        
        # Create the animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(wavefields),
            interval=1000/fps, blit=True
        )
        
        # Save the animation if output_dir is provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            anim.save(output_dir / 'wavefield_animation.mp4', writer='ffmpeg', fps=fps)
        
        plt.tight_layout()
        plt.show()
        
        return anim


class OpenANFWISimulator(Simulator):
    """
    OpenANFWI acoustic field simulator.
    
    This simulator can use various noise sources including brown noise,
    pink noise, and random Ricker wavelets.
    """
    
    def __init__(
        self,
        pressure_field: Optional[PressureField] = None,
        noise_source: Optional[NoiseSource] = None
    ):
        """
        Initialize an OpenANFWI simulator.
        
        Args:
            pressure_field (PressureField, optional): The pressure field implementation to use.
                If None, an AcousticPressureField will be created.
            noise_source (NoiseSource, optional): The noise source to use.
                If None, a composite noise source with brown noise, pink noise,
                and random Ricker wavelets will be created.
        """
        if pressure_field is None:
            pressure_field = AcousticPressureField()
        
        if noise_source is None:
            # Create a composite noise source with brown noise, pink noise,
            # and random Ricker wavelets
            brown_noise = BrownNoise()
            pink_noise = PinkNoise()
            random_ricker = RandomRickerWavelet(10.0, 50.0)
            
            noise_source = CompositeNoiseSource(
                sources=[brown_noise, pink_noise, random_ricker],
                weights=[0.3, 0.3, 0.4]
            )
        
        super().__init__(pressure_field, noise_source)
    
    def simulate(
        self,
        velocity_model: np.ndarray,
        grid_spacing: float,
        time_step: float,
        num_time_steps: int,
        source_x_m: float,
        source_z_m: float,
        receiver_x_m: np.ndarray,
        receiver_z_m: np.ndarray,
        absorbing_boundary_thickness: int = 20,
        apply_free_surface: bool = False,
        store_wavefields: bool = False,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the OpenANFWI simulation.
        
        Args:
            velocity_model (np.ndarray): 2D velocity model (m/s).
            grid_spacing (float): Spatial sampling interval (m).
            time_step (float): Temporal sampling interval (s).
            num_time_steps (int): Number of time steps to simulate.
            source_x_m (float): Source horizontal position (m).
            source_z_m (float): Source depth (m).
            receiver_x_m (np.ndarray): Receiver horizontal positions (m).
            receiver_z_m (np.ndarray): Receiver depths (m).
            absorbing_boundary_thickness (int, optional): Number of absorbing cells on each edge.
                Defaults to 20.
            apply_free_surface (bool, optional): Whether to apply free surface boundary
                condition at the top. Defaults to False.
            store_wavefields (bool, optional): Whether to store the wavefields at each time step.
                Defaults to False.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
                
        Returns:
            Dict[str, Any]: The simulation results.
        """
        # Generate the source wavelet
        source_wavelet, _ = self.noise_source.generate(time_step, num_time_steps, seed=seed)
        
        # Set up the callback for storing wavefields if requested
        wavefields = []
        
        if store_wavefields:
            def callback(pressure_field, time_step_index, _time_step, boundary_cell_count):
                # Strip the absorbing boundary from the pressure field
                interior_pressure = strip_absorbing_boundary(
                    pressure_field, boundary_cell_count
                )
                wavefields.append(interior_pressure.copy())
        else:
            callback = None
        
        # Run the simulation
        results = self.pressure_field.simulate(
            velocity_model=velocity_model,
            absorbing_boundary_thickness=absorbing_boundary_thickness,
            grid_spacing=grid_spacing,
            num_time_steps=num_time_steps,
            time_step=time_step,
            source_wavelet=source_wavelet,
            source_x_m=source_x_m,
            source_z_m=source_z_m,
            receiver_x_m=receiver_x_m,
            receiver_z_m=receiver_z_m,
            apply_free_surface=apply_free_surface,
            callback=callback
        )
        
        # Add the wavefields to the results if stored
        if store_wavefields:
            results['wavefields'] = np.array(wavefields)
        
        # Add the source wavelet to the results
        results['source_wavelet'] = source_wavelet
        
        return results
    
    def visualize(
        self,
        results: Dict[str, Any],
        output_dir: Optional[Path] = None,
        fps: int = 20,
        velocity_alpha: float = 0.5,
        wavefield_alpha: Optional[float] = None
    ):
        """
        Visualize the OpenANFWI simulation results.
        
        Args:
            results (Dict[str, Any]): The simulation results.
            output_dir (Path, optional): Directory to save the visualization.
                If None, the visualization will only be displayed.
            fps (int, optional): Frames per second for the animation. Defaults to 20.
            velocity_alpha (float, optional): Alpha value for the velocity model overlay.
                Defaults to 0.5.
            wavefield_alpha (float, optional): Alpha value for the wavefield.
                If None, no transparency is applied. Defaults to None.
                
        Returns:
            matplotlib.animation.Animation: The animation object if wavefields are available.
        """
        # Check if wavefields are available
        if 'wavefields' not in results:
            print("No wavefields available for visualization.")
            return None
        
        # Extract data from results
        wavefields = results['wavefields']
        velocity_model = results['velocity_model']
        grid_spacing = results['grid_spacing']
        source_wavelet = results.get('source_wavelet')
        
        # Create the figure and axes
        fig = plt.figure(figsize=(15, 10))
        
        if source_wavelet is not None:
            # Create a 2x1 grid for wavefield and source wavelet
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
            ax_wave = fig.add_subplot(gs[0])
            ax_source = fig.add_subplot(gs[1])
        else:
            # Just create a single axis for the wavefield
            ax_wave = fig.add_subplot(111)
        
        # Set up the axes for the wavefield
        extent = [0, velocity_model.shape[1] * grid_spacing,
                 velocity_model.shape[0] * grid_spacing, 0]
        
        # Plot the velocity model
        vel_im = ax_wave.imshow(
            velocity_model,
            extent=extent,
            cmap='viridis',
            alpha=velocity_alpha
        )
        
        # Add a colorbar for the velocity model
        vel_cbar = plt.colorbar(vel_im, ax=ax_wave)
        vel_cbar.set_label('Velocity (m/s)')
        
        # Initialize the wavefield plot
        vmin, vmax = np.min(wavefields), np.max(wavefields)
        abs_max = max(abs(vmin), abs(vmax))
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
        
        wave_im = ax_wave.imshow(
            wavefields[0],
            extent=extent,
            cmap='seismic',
            norm=norm,
            alpha=wavefield_alpha
        )
        
        # Add a colorbar for the wavefield
        wave_cbar = plt.colorbar(wave_im, ax=ax_wave)
        wave_cbar.set_label('Pressure')
        
        # Add labels and title for the wavefield
        ax_wave.set_xlabel('Distance (m)')
        ax_wave.set_ylabel('Depth (m)')
        ax_wave.set_title('Acoustic Wave Propagation')
        
        # Add a timestamp
        timestamp = ax_wave.text(
            0.02, 0.02, f"Time: 0.000 s",
            transform=ax_wave.transAxes,
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        # Set up the source wavelet plot if available
        if source_wavelet is not None:
            time_vector = np.arange(len(source_wavelet)) * results['time_step']
            source_line, = ax_source.plot(time_vector, source_wavelet)
            source_marker, = ax_source.plot([0], [source_wavelet[0]], 'ro')
            
            ax_source.set_xlabel('Time (s)')
            ax_source.set_ylabel('Amplitude')
            ax_source.set_title('Source Wavelet')
            ax_source.grid(True)
        
        # Define the animation update function
        def update(frame_index):
            wave_im.set_array(wavefields[frame_index])
            timestamp.set_text(f"Time: {frame_index * results['time_step']:.3f} s")
            
            if source_wavelet is not None:
                current_time = frame_index * results['time_step']
                source_marker.set_data([current_time], [source_wavelet[frame_index]])
                
                # Update the source wavelet plot limits to follow the marker
                ax_source.set_xlim(max(0, current_time - 0.1), current_time + 0.1)
                
                return wave_im, timestamp, source_marker
            else:
                return wave_im, timestamp
        
        # Create the animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(wavefields),
            interval=1000/fps, blit=True
        )
        
        # Save the animation if output_dir is provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            anim.save(output_dir / 'wavefield_animation.mp4', writer='ffmpeg', fps=fps)
        
        plt.tight_layout()
        plt.show()
        
        return anim