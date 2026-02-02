"""
Sensor Simulator - Complete Imaging Spectrometer Model

Integrates pushbroom geometry, grating dispersion, and detector
characteristics into a complete sensor simulation.

This module answers the question: Given a scene radiance, what
digital numbers would the sensor record?

The chain is:
    Scene Radiance → Optics → Grating → Detector → Digital Number

Each stage introduces its own effects:
    - Optics: Transmission, aberrations, stray light
    - Grating: Efficiency, dispersion, smile/keystone
    - Detector: QE, noise, saturation, digitization
"""

import numpy as np
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

from .pushbroom_geometry import PushbroomGeometry, ScanConfiguration
from .grating_dispersion import GratingDispersion, SpectralCalibration
from .detector_model import DetectorModel, NoiseModel


@dataclass
class SensorConfiguration:
    """Complete sensor configuration."""
    # Optical
    aperture_diameter_mm: float = 100
    focal_length_mm: float = 200
    optical_transmission: float = 0.7

    # Grating
    groove_density: float = 600
    blaze_wavelength: float = 1000
    grating_order: int = 1

    # Detector
    pixel_pitch_um: float = 27
    n_spectral_pixels: int = 400
    n_spatial_pixels: int = 1240

    # Spectral range
    wavelength_min_nm: float = 380
    wavelength_max_nm: float = 2500

    # Scan
    frame_rate_hz: float = 100


class SensorSimulator:
    """
    Complete pushbroom imaging spectrometer simulator.

    Simulates the full detection chain from scene radiance to
    digital numbers, including all sensor effects.
    """

    def __init__(self,
                 config: Optional[SensorConfiguration] = None,
                 scan_config: Optional[ScanConfiguration] = None,
                 noise_model: Optional[NoiseModel] = None):
        """
        Initialize sensor simulator.

        Args:
            config: Sensor configuration
            scan_config: Pushbroom scan configuration
            noise_model: Detector noise model
        """
        self.config = config if config is not None else SensorConfiguration()
        self.scan_config = scan_config if scan_config is not None else ScanConfiguration()

        # Initialize component models
        self.geometry = PushbroomGeometry(self.scan_config)
        self.grating = GratingDispersion(
            self.config.groove_density,
            self.config.blaze_wavelength,
            self.config.grating_order,
            self.config.focal_length_mm
        )
        self.detector = DetectorModel(noise_model)

        # Create spectral calibration
        self.spectral_cal = self.grating.create_spectral_calibration(
            (self.config.wavelength_min_nm, self.config.wavelength_max_nm),
            self.config.n_spectral_pixels
        )

        # Store wavelengths for convenience
        self.wavelengths = self.spectral_cal.wavelength_centers

        # Calculate derived parameters
        self._calculate_system_parameters()

    def _calculate_system_parameters(self):
        """Calculate system-level parameters."""
        c = self.config

        # F-number
        self.f_number = c.focal_length_mm / c.aperture_diameter_mm

        # Pixel solid angle (steradians)
        pixel_size_rad = c.pixel_pitch_um * 1e-6 / (c.focal_length_mm * 1e-3)
        self.pixel_solid_angle = pixel_size_rad ** 2

        # Pixel area (m²)
        self.pixel_area = (c.pixel_pitch_um * 1e-6) ** 2

        # Integration time
        self.integration_time = 1.0 / self.scan_config.frame_rate_hz

        # Collecting area (m²)
        self.collecting_area = np.pi * (c.aperture_diameter_mm * 1e-3 / 2) ** 2

        # System throughput
        self.system_throughput = (
            self.collecting_area *
            self.pixel_solid_angle *
            c.optical_transmission
        )

    def radiance_to_electrons(self, radiance: np.ndarray) -> np.ndarray:
        """
        Convert spectral radiance to photoelectrons.

        Args:
            radiance: Spectral radiance (W/m²/sr/nm) shape: (n_wavelengths,)
                      or (n_spatial, n_wavelengths)

        Returns:
            Photoelectrons array
        """
        radiance = np.asarray(radiance)

        # Photon energy at each wavelength
        h = 6.626e-34  # Planck's constant
        c = 2.998e8     # Speed of light
        wavelength_m = self.wavelengths * 1e-9
        photon_energy = h * c / wavelength_m

        # Spectral bandwidth per channel
        if len(self.spectral_cal.fwhm) > 0:
            bandwidth_nm = self.spectral_cal.fwhm
        else:
            bandwidth_nm = np.gradient(self.wavelengths)

        # Grating efficiency
        efficiency = self.grating.efficiency(self.wavelengths)

        # Power at detector per spectral channel
        # P = L × A_collect × Ω_pixel × Δλ × T_optics × η_grating
        power = (radiance * self.collecting_area * self.pixel_solid_angle *
                 bandwidth_nm * self.config.optical_transmission * efficiency)

        # Energy per integration
        energy = power * self.integration_time

        # Number of photons
        n_photons = energy / photon_energy

        # Convert to electrons (QE)
        n_electrons = self.detector.photons_to_electrons(n_photons)

        return n_electrons

    def simulate_measurement(self, radiance: np.ndarray,
                             add_noise: bool = True,
                             random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Simulate complete measurement from radiance to DN.

        Args:
            radiance: Input spectral radiance (W/m²/sr/nm)
            add_noise: Whether to add detector noise
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with measurement results
        """
        rng = np.random.RandomState(random_state)

        # Convert radiance to electrons
        signal_e = self.radiance_to_electrons(radiance)

        # Apply detector model
        detection_result = self.detector.simulate_detection(
            signal_e,
            self.integration_time,
            add_noise=add_noise,
            random_state=rng
        )

        # Add spectral calibration info
        detection_result['wavelengths'] = self.wavelengths
        detection_result['spectral_fwhm'] = self.spectral_cal.fwhm

        return detection_result

    def simulate_datacube(self, radiance_cube: np.ndarray,
                          add_noise: bool = True,
                          random_state: Optional[int] = None) -> np.ndarray:
        """
        Simulate full data cube measurement.

        Args:
            radiance_cube: Input radiance (n_lines, n_samples, n_wavelengths)
            add_noise: Whether to add noise
            random_state: Random seed

        Returns:
            Digital number cube (n_lines, n_samples, n_wavelengths)
        """
        n_lines, n_samples, n_wavelengths = radiance_cube.shape
        rng = np.random.RandomState(random_state)

        # Output cube
        dn_cube = np.zeros((n_lines, n_samples, n_wavelengths), dtype=np.uint16)

        # Process line by line (as a pushbroom scanner would)
        for line in range(n_lines):
            for sample in range(n_samples):
                radiance_spectrum = radiance_cube[line, sample, :]
                result = self.simulate_measurement(radiance_spectrum, add_noise, rng)
                dn_cube[line, sample, :] = result['digital_number']

        return dn_cube

    def add_smile_distortion(self, datacube: np.ndarray,
                              smile_amplitude_nm: float = 0.5) -> np.ndarray:
        """
        Add spectral smile distortion.

        Smile causes wavelength calibration to vary across the swath
        (wavelengths shift in a smile-shaped pattern).

        Args:
            datacube: Input data cube
            smile_amplitude_nm: Peak smile amplitude in nm

        Returns:
            Distorted data cube
        """
        n_lines, n_samples, n_bands = datacube.shape

        # Smile pattern (parabolic across track)
        x = np.linspace(-1, 1, n_samples)
        smile_shift = smile_amplitude_nm * (x ** 2)  # nm shift

        # Apply by interpolating spectra
        distorted = np.zeros_like(datacube)
        for sample in range(n_samples):
            shifted_wavelengths = self.wavelengths + smile_shift[sample]
            for line in range(n_lines):
                spectrum = datacube[line, sample, :]
                distorted[line, sample, :] = np.interp(
                    self.wavelengths, shifted_wavelengths, spectrum
                )

        return distorted

    def add_keystone_distortion(self, datacube: np.ndarray,
                                 keystone_pixels: float = 0.5) -> np.ndarray:
        """
        Add spatial keystone distortion.

        Keystone causes spatial registration to vary with wavelength
        (different wavelengths see slightly different ground locations).

        Args:
            datacube: Input data cube
            keystone_pixels: Peak keystone shift in pixels

        Returns:
            Distorted data cube (with inter-band misregistration)
        """
        n_lines, n_samples, n_bands = datacube.shape

        # Keystone pattern (linear with wavelength)
        wl_normalized = (self.wavelengths - self.wavelengths[0]) / (self.wavelengths[-1] - self.wavelengths[0])
        spatial_shift = keystone_pixels * (wl_normalized - 0.5)  # pixels

        # Apply by shifting each band
        distorted = np.zeros_like(datacube)
        for band in range(n_bands):
            shift = spatial_shift[band]
            # Subpixel shift using interpolation
            for line in range(n_lines):
                x_orig = np.arange(n_samples)
                x_shifted = x_orig - shift
                distorted[line, :, band] = np.interp(
                    x_orig, x_shifted, datacube[line, :, band],
                    left=datacube[line, 0, band],
                    right=datacube[line, -1, band]
                )

        return distorted

    def calculate_snr_spectrum(self, radiance: np.ndarray) -> np.ndarray:
        """
        Calculate SNR at each wavelength for given radiance.

        Args:
            radiance: Input spectral radiance

        Returns:
            SNR array
        """
        signal_e = self.radiance_to_electrons(radiance)
        return self.detector.calculate_snr(signal_e, self.integration_time)

    def get_system_summary(self) -> str:
        """Get summary of sensor system parameters."""
        c = self.config

        summary = f"""
Imaging Spectrometer System Summary
===================================

Optics:
  Aperture: {c.aperture_diameter_mm} mm
  Focal length: {c.focal_length_mm} mm
  F-number: f/{self.f_number:.1f}
  Optical transmission: {c.optical_transmission*100:.0f}%
  Collecting area: {self.collecting_area*1e4:.1f} cm²

Grating:
  Groove density: {c.groove_density} lines/mm
  Blaze wavelength: {c.blaze_wavelength} nm
  Order: {c.grating_order}

Spectral:
  Range: {c.wavelength_min_nm:.0f} - {c.wavelength_max_nm:.0f} nm
  Channels: {c.n_spectral_pixels}
  Sampling: {(c.wavelength_max_nm - c.wavelength_min_nm) / c.n_spectral_pixels:.1f} nm/channel

Spatial:
  Cross-track pixels: {c.n_spatial_pixels}
  Pixel pitch: {c.pixel_pitch_um} µm
  Total FOV: {self.scan_config.total_fov_deg}°

Timing:
  Frame rate: {self.scan_config.frame_rate_hz} Hz
  Integration time: {self.integration_time*1000:.2f} ms

Radiometry:
  Pixel solid angle: {self.pixel_solid_angle:.2e} sr
  System throughput: {self.system_throughput:.2e} m²·sr
"""
        return summary

    def build_datacube_animation_frames(self, n_frames: int = 20) -> Dict:
        """
        Generate data for animating data cube construction.

        Shows how pushbroom scanning builds the cube frame by frame.

        Args:
            n_frames: Number of frames to generate

        Returns:
            Dictionary with animation data
        """
        # Simulate scan geometry
        scan = self.geometry.simulate_frame_acquisition(n_frames)

        # Generate synthetic scene (gradient for visualization)
        n_samples = self.config.n_spatial_pixels
        n_bands = len(self.wavelengths)

        # Example: spatial gradient × spectral curve
        spatial = np.linspace(0, 1, n_samples)
        spectral = np.exp(-((self.wavelengths - 1000) / 500) ** 2)

        # Build frames
        frames = []
        cube_accumulator = np.zeros((n_frames, n_samples, n_bands))

        for frame_idx in range(n_frames):
            # This frame's "line" of data
            frame_data = np.outer(spatial, spectral) * (0.5 + 0.5 * np.random.random())
            cube_accumulator[frame_idx] = frame_data
            frames.append({
                'frame_idx': frame_idx,
                'time_s': scan['frame_times_s'][frame_idx],
                'along_track_m': scan['along_track_m'][frame_idx],
                'frame_data': frame_data.copy()
            })

        return {
            'frames': frames,
            'scan_geometry': scan,
            'final_cube': cube_accumulator,
            'wavelengths': self.wavelengths,
            'cross_track_m': scan['cross_track_m']
        }

    def explain_sensor_chain(self) -> str:
        """Educational explanation of the sensor detection chain."""
        explanation = """
The Imaging Spectrometer Detection Chain
========================================

Scene Radiance (L)
      │
      │  Light from ground travels to sensor
      ▼
┌──────────────────┐
│ TELESCOPE OPTICS │
│                  │
│ • Collects light │
│ • Focuses on slit│
│ • Transmission T │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   ENTRANCE SLIT  │
│                  │
│ • Defines FOV    │
│ • Sets spectral  │
│   resolution     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ COLLIMATING OPTIC│
│                  │
│ • Makes parallel │
│   beams          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│DIFFRACTION GRATING│
│                  │
│ • Disperses by λ │
│ • Efficiency η(λ)│
│ • d·sin(θ) = mλ  │
└────────┬─────────┘
         │
         ╲ Different λ at different angles
          ╲
           ▼
┌──────────────────┐
│  FOCUSING OPTIC  │
│                  │
│ • Images onto    │
│   detector       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│     DETECTOR     │
│                  │
│ • Quantum eff QE │
│ • Shot noise √N  │
│ • Read noise σR  │
│ • Dark current D │
│ • ADC → DN       │
└────────┬─────────┘
         │
         ▼
   Digital Number (DN)


The Radiometric Equation:

  DN = G × { QE × η(λ) × T × [L × A × Ω × Δλ × t / (hc/λ)] + D×t + noise }

where:
  G = ADC gain (DN/e-)
  QE = quantum efficiency
  η(λ) = grating efficiency
  T = optical transmission
  L = scene radiance (W/m²/sr/nm)
  A = collecting area (m²)
  Ω = pixel solid angle (sr)
  Δλ = spectral bandwidth (nm)
  t = integration time (s)
  hc/λ = photon energy (J)
  D = dark current (e-/s)
"""
        return explanation
