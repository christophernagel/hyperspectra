"""
HSI Visualizer - Matplotlib-based visualizations

Provides static and animated visualizations for understanding
hyperspectral imaging concepts.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation


class HSIVisualizer:
    """
    Visualization tools for hyperspectral imaging education.

    All methods return matplotlib figures that can be saved or displayed.
    """

    # Color scheme for consistent plots
    COLORS = {
        'visible': '#3498db',
        'nir': '#e74c3c',
        'swir': '#9b59b6',
        'atmosphere': '#95a5a6',
        'surface': '#27ae60',
        'sensor': '#f39c12',
        'noise': '#e74c3c',
        'signal': '#2ecc71'
    }

    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            pass  # Use default if style not available

    def plot_atmospheric_transmittance(self,
                                         wavelengths: np.ndarray,
                                         transmittances: Dict[str, np.ndarray],
                                         title: str = "Atmospheric Transmittance"
                                         ) -> Figure:
        """
        Plot atmospheric transmittance with labeled absorption features.

        Args:
            wavelengths: Wavelength array (nm)
            transmittances: Dict of {name: transmittance_array}
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each component
        colors = plt.cm.Set2(np.linspace(0, 1, len(transmittances)))
        for (name, trans), color in zip(transmittances.items(), colors):
            ax.plot(wavelengths, trans, label=name, color=color, linewidth=1.5)

        # Mark major absorption bands
        absorption_bands = [
            (760, "O₂"),
            (940, "H₂O"),
            (1130, "H₂O"),
            (1380, "H₂O"),
            (1880, "H₂O"),
            (2050, "CO₂"),
        ]

        for wl, gas in absorption_bands:
            if wavelengths[0] <= wl <= wavelengths[-1]:
                ax.axvline(wl, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
                ax.annotate(gas, (wl, 0.95), fontsize=8, ha='center',
                           color='gray')

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmittance")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_radiance_components(self,
                                   wavelengths: np.ndarray,
                                   components: Dict[str, np.ndarray],
                                   title: str = "At-Sensor Radiance Components"
                                   ) -> Figure:
        """
        Plot radiance components as stacked area chart.

        Args:
            wavelengths: Wavelength array (nm)
            components: Dict with 'path', 'direct', 'diffuse' radiances
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Top: Absolute values
        ax1.fill_between(wavelengths, 0, components.get('path', np.zeros_like(wavelengths)),
                        alpha=0.7, label='Path Radiance', color='#e74c3c')
        ax1.fill_between(wavelengths,
                        components.get('path', np.zeros_like(wavelengths)),
                        components.get('path', np.zeros_like(wavelengths)) +
                        components.get('surface', np.zeros_like(wavelengths)),
                        alpha=0.7, label='Surface Contribution', color='#27ae60')

        ax1.set_ylabel("Radiance (W/m²/sr/nm)")
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom: Relative fractions
        total = components.get('total', np.ones_like(wavelengths))
        path_frac = components.get('path', np.zeros_like(wavelengths)) / np.maximum(total, 1e-10)
        surface_frac = 1 - path_frac

        ax2.fill_between(wavelengths, 0, path_frac * 100,
                        alpha=0.7, label='Path Radiance', color='#e74c3c')
        ax2.fill_between(wavelengths, path_frac * 100, 100,
                        alpha=0.7, label='Surface Signal', color='#27ae60')

        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Fraction (%)")
        ax2.set_title("Signal Composition")
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_spectral_comparison(self,
                                   wavelengths: np.ndarray,
                                   spectra: Dict[str, np.ndarray],
                                   ylabel: str = "Reflectance",
                                   title: str = "Spectral Comparison"
                                   ) -> Figure:
        """
        Plot multiple spectra for comparison.

        Args:
            wavelengths: Wavelength array (nm)
            spectra: Dict of {name: spectrum_array}
            ylabel: Y-axis label
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(spectra)))

        for (name, spectrum), color in zip(spectra.items(), colors):
            ax.plot(wavelengths, spectrum, label=name, color=color, linewidth=1.5)

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add spectral region labels
        self._add_spectral_regions(ax, wavelengths)

        plt.tight_layout()
        return fig

    def plot_snr_spectrum(self,
                           wavelengths: np.ndarray,
                           snr: np.ndarray,
                           noise_components: Optional[Dict[str, np.ndarray]] = None,
                           title: str = "Signal-to-Noise Ratio"
                           ) -> Figure:
        """
        Plot SNR spectrum with noise breakdown.

        Args:
            wavelengths: Wavelength array (nm)
            snr: SNR array
            noise_components: Optional dict with 'shot', 'read', 'dark'
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if noise_components:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        else:
            fig, ax1 = plt.subplots(figsize=(12, 5))

        # SNR plot
        ax1.semilogy(wavelengths, snr, 'b-', linewidth=2, label='SNR')
        ax1.axhline(100, color='green', linestyle='--', alpha=0.7, label='Target (100)')
        ax1.axhline(10, color='red', linestyle='--', alpha=0.7, label='Minimum (10)')

        ax1.set_ylabel("SNR")
        ax1.set_title(title)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(1, max(snr) * 2)

        # Noise breakdown
        if noise_components:
            colors = {'shot': '#3498db', 'read': '#e74c3c', 'dark': '#9b59b6'}
            for name, noise in noise_components.items():
                ax2.plot(wavelengths, noise, label=f'{name.capitalize()} noise',
                        color=colors.get(name, 'gray'), linewidth=1.5)

            ax2.set_xlabel("Wavelength (nm)")
            ax2.set_ylabel("Noise (electrons)")
            ax2.set_title("Noise Components")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_sensitivity_analysis(self,
                                    wavelengths: np.ndarray,
                                    parameter_name: str,
                                    parameter_values: list,
                                    radiances: np.ndarray,
                                    title: str = "Sensitivity Analysis"
                                    ) -> Figure:
        """
        Plot how radiance changes with a parameter.

        Args:
            wavelengths: Wavelength array
            parameter_name: Name of varied parameter
            parameter_values: List of parameter values
            radiances: Array of radiance spectra (n_values, n_wavelengths)
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Absolute radiances
        colors = plt.cm.viridis(np.linspace(0, 1, len(parameter_values)))
        for val, rad, color in zip(parameter_values, radiances, colors):
            ax1.plot(wavelengths, rad, color=color, label=f'{parameter_name}={val}')

        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Radiance (W/m²/sr/nm)")
        ax1.set_title(f"Radiance vs {parameter_name}")
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Right: Relative change from baseline
        baseline = radiances[0]
        for val, rad, color in zip(parameter_values[1:], radiances[1:], colors[1:]):
            relative_change = (rad - baseline) / np.maximum(baseline, 1e-10) * 100
            ax2.plot(wavelengths, relative_change, color=color, label=f'{parameter_name}={val}')

        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Change from baseline (%)")
        ax2.set_title("Relative Sensitivity")
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='black', linewidth=0.5)

        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        return fig

    def plot_imaging_chain_diagram(self) -> Figure:
        """
        Create a schematic diagram of the imaging chain.

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')

        # Define boxes
        boxes = [
            (1, 10, 8, 1.5, 'SUN', '#f1c40f', 'Solar irradiance E₀'),
            (1, 7.5, 8, 1.5, 'ATMOSPHERE (downward)', '#3498db',
             'Absorption + Scattering\nT_down, E_diffuse'),
            (1, 5, 8, 1.5, 'SURFACE', '#27ae60',
             'Reflectance ρ(λ)\nL_surface = ρ × E_down / π'),
            (1, 2.5, 8, 1.5, 'ATMOSPHERE (upward)', '#3498db',
             'Path radiance L_path\nTransmittance T_up'),
            (1, 0, 8, 1.5, 'SENSOR', '#e74c3c',
             'Optics → Grating → Detector\nRadiance → Electrons → DN'),
        ]

        for x, y, w, h, label, color, desc in boxes:
            rect = mpatches.FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                facecolor=color, edgecolor='black', alpha=0.7
            )
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2 + 0.3, label, ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
            ax.text(x + w/2, y + h/2 - 0.3, desc, ha='center', va='center',
                   fontsize=9, color='white')

        # Arrows
        arrow_style = dict(arrowstyle='->', color='black', lw=2)
        for y_start, y_end in [(10, 9), (7.5, 6.5), (5, 4), (2.5, 1.5)]:
            ax.annotate('', xy=(5, y_end), xytext=(5, y_start),
                       arrowprops=arrow_style)

        # Title
        ax.text(5, 11.7, 'Hyperspectral Imaging Chain', ha='center',
               fontsize=16, fontweight='bold')

        return fig

    def plot_pushbroom_concept(self) -> Figure:
        """
        Visualize pushbroom scanning concept.

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 6))

        # Left panel: 3D view
        ax1 = fig.add_subplot(121, projection='3d')

        # Sensor platform
        platform = np.array([[0, 0, 5]])
        ax1.scatter(*platform.T, s=200, c='red', marker='^', label='Sensor')

        # Ground swath
        x_ground = np.linspace(-2, 2, 50)
        y_ground = np.linspace(-3, 3, 60)
        X, Y = np.meshgrid(x_ground, y_ground)
        Z = np.zeros_like(X)

        # Simulated reflectance pattern
        pattern = (np.sin(X * 2) ** 2 + np.cos(Y * 1.5) ** 2) / 2

        ax1.plot_surface(X, Y, Z, facecolors=plt.cm.terrain(pattern),
                        alpha=0.8, rstride=1, cstride=1)

        # FOV lines
        for x in [-2, 2]:
            ax1.plot([0, x], [0, 0], [5, 0], 'r--', alpha=0.5)

        # Current scan line
        ax1.plot(x_ground, np.zeros_like(x_ground), np.zeros_like(x_ground),
                'r-', linewidth=3, label='Current scan line')

        # Flight direction arrow
        ax1.quiver(0, -3, 5, 0, 2, 0, color='blue', arrow_length_ratio=0.1)
        ax1.text(0, -2, 5.3, 'Flight\ndirection', fontsize=9)

        ax1.set_xlabel('Cross-track')
        ax1.set_ylabel('Along-track')
        ax1.set_zlabel('Altitude')
        ax1.set_title('Pushbroom Scanner Geometry')

        # Right panel: Data cube concept
        ax2 = fig.add_subplot(122, projection='3d')

        # Data cube outline
        n_lines = 20
        n_samples = 30
        n_bands = 15

        # Draw cube edges
        for z in [0, n_bands]:
            ax2.plot([0, n_samples], [0, 0], [z, z], 'k-', alpha=0.5)
            ax2.plot([0, n_samples], [n_lines, n_lines], [z, z], 'k-', alpha=0.5)
            ax2.plot([0, 0], [0, n_lines], [z, z], 'k-', alpha=0.5)
            ax2.plot([n_samples, n_samples], [0, n_lines], [z, z], 'k-', alpha=0.5)

        for x in [0, n_samples]:
            for y in [0, n_lines]:
                ax2.plot([x, x], [y, y], [0, n_bands], 'k-', alpha=0.5)

        # Highlight a single frame (line)
        frame_idx = 10
        X_frame = np.arange(n_samples)
        Z_frame = np.arange(n_bands)
        Xf, Zf = np.meshgrid(X_frame, Z_frame)
        Yf = np.ones_like(Xf) * frame_idx

        ax2.plot_surface(Xf, Yf, Zf, alpha=0.7, color='red')
        ax2.text(n_samples/2, frame_idx, n_bands+1, 'Single frame\n(all wavelengths)',
                fontsize=8, ha='center')

        ax2.set_xlabel('Cross-track (samples)')
        ax2.set_ylabel('Along-track (lines)')
        ax2.set_zlabel('Spectral (bands)')
        ax2.set_title('Hyperspectral Data Cube')

        plt.tight_layout()
        return fig

    def plot_grating_dispersion(self,
                                  wavelengths: np.ndarray,
                                  angles: np.ndarray,
                                  efficiency: np.ndarray,
                                  blaze_wavelength: float
                                  ) -> Figure:
        """
        Visualize grating dispersion and efficiency.

        Args:
            wavelengths: Wavelength array
            angles: Diffraction angles
            efficiency: Grating efficiency
            blaze_wavelength: Blaze wavelength

        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Dispersion relation
        ax1.plot(wavelengths, angles, 'b-', linewidth=2)
        ax1.axvline(blaze_wavelength, color='red', linestyle='--',
                   label=f'Blaze λ = {blaze_wavelength} nm')

        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Diffraction angle (°)")
        ax1.set_title("Grating Dispersion: λ → θ")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: Efficiency
        ax2.plot(wavelengths, efficiency * 100, 'g-', linewidth=2)
        ax2.axvline(blaze_wavelength, color='red', linestyle='--',
                   label=f'Blaze λ = {blaze_wavelength} nm')

        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Efficiency (%)")
        ax2.set_title("Grating Efficiency (Blaze Envelope)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 110)

        plt.tight_layout()
        return fig

    def plot_detector_noise_model(self,
                                    signal_electrons: np.ndarray,
                                    noise_components: Dict[str, np.ndarray],
                                    snr: np.ndarray
                                    ) -> Figure:
        """
        Visualize detector noise model.

        Args:
            signal_electrons: Signal array (electrons)
            noise_components: Dict with 'shot', 'read', 'dark' noise
            snr: Signal-to-noise ratio

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Signal histogram
        ax1 = axes[0, 0]
        ax1.hist(signal_electrons.flatten(), bins=50, color='blue', alpha=0.7)
        ax1.set_xlabel("Signal (electrons)")
        ax1.set_ylabel("Count")
        ax1.set_title("Signal Distribution")

        # Noise comparison
        ax2 = axes[0, 1]
        noise_types = list(noise_components.keys())
        noise_means = [np.mean(np.abs(v)) for v in noise_components.values()]

        bars = ax2.bar(noise_types, noise_means, color=['#3498db', '#e74c3c', '#9b59b6'])
        ax2.set_ylabel("Mean noise (electrons)")
        ax2.set_title("Noise Sources")

        # SNR vs Signal
        ax3 = axes[1, 0]
        signal_sorted = np.sort(signal_electrons.flatten())
        snr_sorted = np.sort(snr.flatten())

        ax3.loglog(signal_sorted, snr_sorted, 'b.', alpha=0.5, markersize=2)
        ax3.set_xlabel("Signal (electrons)")
        ax3.set_ylabel("SNR")
        ax3.set_title("SNR vs Signal")
        ax3.grid(True, alpha=0.3)

        # Add theoretical curves
        signal_theory = np.logspace(1, 6, 100)
        read_noise = noise_means[noise_types.index('read')] if 'read' in noise_types else 20
        snr_shot_limited = np.sqrt(signal_theory)
        snr_read_limited = signal_theory / read_noise

        ax3.loglog(signal_theory, snr_shot_limited, 'g--', label='Shot-limited')
        ax3.loglog(signal_theory, snr_read_limited, 'r--', label='Read-limited')
        ax3.legend()

        # SNR histogram
        ax4 = axes[1, 1]
        ax4.hist(snr.flatten(), bins=50, color='green', alpha=0.7)
        ax4.axvline(100, color='red', linestyle='--', label='Target SNR=100')
        ax4.set_xlabel("SNR")
        ax4.set_ylabel("Count")
        ax4.set_title("SNR Distribution")
        ax4.legend()

        plt.tight_layout()
        return fig

    def create_spectral_animation(self,
                                    wavelengths: np.ndarray,
                                    radiance_frames: List[np.ndarray],
                                    reflectance_frames: List[np.ndarray],
                                    titles: List[str],
                                    interval: int = 500
                                    ) -> animation.FuncAnimation:
        """
        Create animated comparison of different atmospheric conditions.

        Args:
            wavelengths: Wavelength array
            radiance_frames: List of radiance arrays
            reflectance_frames: List of reflectance arrays
            titles: Titles for each frame
            interval: Frame interval in ms

        Returns:
            Matplotlib animation
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Initialize lines
        line1, = ax1.plot([], [], 'b-', linewidth=2)
        line2, = ax2.plot([], [], 'g-', linewidth=2)
        title = ax1.set_title('')

        ax1.set_xlim(wavelengths[0], wavelengths[-1])
        ax1.set_ylim(0, np.max([r.max() for r in radiance_frames]) * 1.1)
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Radiance (W/m²/sr/nm)")
        ax1.set_title("At-Sensor Radiance")
        ax1.grid(True, alpha=0.3)

        ax2.set_xlim(wavelengths[0], wavelengths[-1])
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Reflectance")
        ax2.set_title("Surface Reflectance")
        ax2.grid(True, alpha=0.3)

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1, line2

        def animate(i):
            line1.set_data(wavelengths, radiance_frames[i])
            line2.set_data(wavelengths, reflectance_frames[i])
            ax1.set_title(f"At-Sensor Radiance: {titles[i]}")
            return line1, line2

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(radiance_frames),
                                       interval=interval, blit=True)
        return anim

    def _add_spectral_regions(self, ax: Axes, wavelengths: np.ndarray):
        """Add spectral region shading and labels."""
        regions = [
            (380, 450, 'violet', '#8e44ad'),
            (450, 495, 'blue', '#3498db'),
            (495, 570, 'green', '#27ae60'),
            (570, 590, 'yellow', '#f1c40f'),
            (590, 620, 'orange', '#e67e22'),
            (620, 750, 'red', '#e74c3c'),
            (750, 1400, 'NIR', '#95a5a6'),
            (1400, 2500, 'SWIR', '#7f8c8d'),
        ]

        ymin, ymax = ax.get_ylim()

        for wl_min, wl_max, name, color in regions:
            if wl_max > wavelengths[0] and wl_min < wavelengths[-1]:
                wl_min = max(wl_min, wavelengths[0])
                wl_max = min(wl_max, wavelengths[-1])
                ax.axvspan(wl_min, wl_max, alpha=0.1, color=color)

    def save_figure(self, fig: Figure, filename: str, dpi: int = 150):
        """
        Save figure to file.

        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filename}")
