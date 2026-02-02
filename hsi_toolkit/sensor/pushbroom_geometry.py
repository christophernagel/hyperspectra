"""
Pushbroom Imaging Geometry Module

Models the geometric aspects of pushbroom (along-track) scanning.

Pushbroom Concept:
    Unlike whiskbroom scanners that sweep a single detector across the track,
    pushbroom sensors use a linear detector array oriented perpendicular to
    the flight direction. All cross-track pixels are imaged simultaneously,
    and aircraft motion provides the along-track dimension.

    ┌─────────────────────────────────────────┐
    │         Flight Direction  ↓             │
    │                                         │
    │    ←── Slit (cross-track) ──→          │
    │    |   |   |   |   |   |   |           │
    │    ▼   ▼   ▼   ▼   ▼   ▼   ▼           │
    │    Multiple ground pixels imaged       │
    │    simultaneously at each instant      │
    │                                         │
    │    Aircraft motion builds the image    │
    │    line by line in along-track         │
    └─────────────────────────────────────────┘

Key Geometric Parameters:
    - IFOV (Instantaneous Field of View): Angular extent of one pixel
    - Total FOV: Angular swath width
    - GSD (Ground Sampling Distance): Pixel size on ground
    - Frame rate: Lines per second

AVIRIS-3 Specifications:
    - Total FOV: 39.5 degrees
    - Spatial sampling: 0.56 mrad IFOV
    - Cross-track pixels: ~1240 at typical altitude
    - Spectral range: 380-2500 nm
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScanConfiguration:
    """
    Pushbroom scan configuration parameters.

    All angles in degrees, distances in meters.
    """
    # Platform parameters
    altitude_m: float = 8500           # Flight altitude (m)
    velocity_m_s: float = 80           # Platform velocity (m/s)

    # Optical parameters
    focal_length_mm: float = 100       # Telescope focal length (mm)
    slit_width_um: float = 20          # Entrance slit width (µm)

    # Detector parameters
    n_cross_track: int = 1240          # Number of cross-track pixels
    n_spectral: int = 284              # Number of spectral channels
    pixel_pitch_um: float = 27         # Detector pixel pitch (µm)

    # FOV parameters
    total_fov_deg: float = 39.5        # Total field of view (degrees)
    ifov_mrad: float = 0.56            # Instantaneous FOV (mrad)

    # Timing
    frame_rate_hz: float = 100         # Frame rate (Hz)


class PushbroomGeometry:
    """
    Pushbroom imaging geometry calculator.

    Computes ground footprint, view angles, and scan timing
    for a pushbroom imaging spectrometer.
    """

    def __init__(self, config: Optional[ScanConfiguration] = None):
        """
        Initialize pushbroom geometry model.

        Args:
            config: Scan configuration (uses defaults if None)
        """
        self.config = config if config is not None else ScanConfiguration()
        self._calculate_derived_parameters()

    def _calculate_derived_parameters(self):
        """Calculate derived geometric parameters."""
        c = self.config

        # IFOV in radians
        self.ifov_rad = c.ifov_mrad / 1000

        # Ground sampling distance at nadir
        self.gsd_nadir = c.altitude_m * self.ifov_rad

        # Swath width
        self.fov_rad = np.radians(c.total_fov_deg)
        self.swath_width = 2 * c.altitude_m * np.tan(self.fov_rad / 2)

        # Along-track sampling
        self.along_track_gsd = c.velocity_m_s / c.frame_rate_hz

        # Integration time per frame
        self.integration_time_ms = 1000 / c.frame_rate_hz

        # View angles for each cross-track pixel
        self._calculate_view_angles()

    def _calculate_view_angles(self):
        """Calculate view zenith and azimuth for each cross-track pixel."""
        c = self.config

        # Pixel indices (-0.5 to 0.5 of swath)
        pixel_fraction = np.linspace(-0.5, 0.5, c.n_cross_track)

        # View zenith angle (0 at nadir, increasing toward edges)
        self.view_zenith = np.abs(pixel_fraction * c.total_fov_deg)

        # View azimuth (perpendicular to flight direction)
        # Left side = 270°, right side = 90° (assuming north flight)
        self.view_azimuth = np.where(pixel_fraction < 0, 270, 90)

        # Ground position relative to nadir (cross-track distance)
        self.cross_track_position = pixel_fraction * self.swath_width

    def ground_footprint(self, pixel_index: int) -> Tuple[float, float]:
        """
        Calculate ground footprint size for a pixel.

        Off-nadir pixels have larger footprints due to:
        1. Increased slant range
        2. Oblique viewing angle

        Args:
            pixel_index: Cross-track pixel index (0 to n-1)

        Returns:
            Tuple of (cross_track_size, along_track_size) in meters
        """
        vza = np.radians(self.view_zenith[pixel_index])

        # Slant range increases with view angle
        slant_range = self.config.altitude_m / np.cos(vza)

        # Cross-track footprint (larger due to oblique view)
        cross_track = self.gsd_nadir / np.cos(vza)

        # Along-track footprint (unchanged by cross-track view angle)
        along_track = self.along_track_gsd

        return cross_track, along_track

    def pixel_solid_angle(self, pixel_index: int) -> float:
        """
        Calculate solid angle subtended by a pixel.

        Ω = IFOV_x × IFOV_y / cos(θ)

        This affects radiometric calculations - off-nadir pixels
        collect light from larger solid angles.

        Args:
            pixel_index: Cross-track pixel index

        Returns:
            Solid angle in steradians
        """
        vza = np.radians(self.view_zenith[pixel_index])

        # Base solid angle at nadir
        omega_nadir = self.ifov_rad ** 2

        # Increases with view angle
        return omega_nadir / np.cos(vza)

    def path_length(self, pixel_index: int) -> float:
        """
        Calculate atmospheric path length for a pixel.

        Path length affects:
        1. Atmospheric absorption/scattering
        2. Signal attenuation

        Args:
            pixel_index: Cross-track pixel index

        Returns:
            Path length in meters
        """
        vza = np.radians(self.view_zenith[pixel_index])
        return self.config.altitude_m / np.cos(vza)

    def airmass_factor(self, pixel_index: int) -> float:
        """
        Calculate airmass factor (relative path length).

        Airmass = 1.0 at nadir, increases toward edges.

        Args:
            pixel_index: Cross-track pixel index

        Returns:
            Airmass factor (≥ 1.0)
        """
        vza = np.radians(self.view_zenith[pixel_index])
        return 1.0 / np.cos(vza)

    def simulate_frame_acquisition(self, n_frames: int = 100) -> dict:
        """
        Simulate geometry for multiple frames of data acquisition.

        This shows how the data cube is built frame by frame.

        Args:
            n_frames: Number of frames to simulate

        Returns:
            Dictionary with frame timing and positions
        """
        c = self.config

        # Time array
        frame_times = np.arange(n_frames) / c.frame_rate_hz

        # Along-track positions
        along_track_positions = frame_times * c.velocity_m_s

        # Ground coverage
        return {
            'n_frames': n_frames,
            'frame_times_s': frame_times,
            'along_track_m': along_track_positions,
            'cross_track_m': self.cross_track_position.copy(),
            'swath_width_m': self.swath_width,
            'total_length_m': along_track_positions[-1] + self.along_track_gsd,
            'integration_time_ms': self.integration_time_ms,
            'data_cube_shape': (n_frames, c.n_cross_track, c.n_spectral)
        }

    def get_geometry_summary(self) -> str:
        """Get a summary of geometric parameters."""
        c = self.config

        summary = f"""
Pushbroom Scan Geometry Summary
===============================

Platform:
  Altitude: {c.altitude_m:.0f} m ({c.altitude_m/1000:.1f} km)
  Velocity: {c.velocity_m_s:.0f} m/s ({c.velocity_m_s*3.6:.0f} km/h)

Field of View:
  Total FOV: {c.total_fov_deg:.1f}°
  IFOV: {c.ifov_mrad:.3f} mrad ({c.ifov_mrad*1000:.1f} µrad)

Ground Footprint:
  Swath width: {self.swath_width:.0f} m ({self.swath_width/1000:.1f} km)
  GSD at nadir: {self.gsd_nadir:.2f} m
  Along-track sampling: {self.along_track_gsd:.2f} m

Detector:
  Cross-track pixels: {c.n_cross_track}
  Spectral channels: {c.n_spectral}
  Pixel pitch: {c.pixel_pitch_um} µm

Timing:
  Frame rate: {c.frame_rate_hz:.0f} Hz
  Integration time: {self.integration_time_ms:.2f} ms

View Angle Range:
  Edge pixels view zenith: ±{c.total_fov_deg/2:.1f}°
  Edge pixel GSD: {self.ground_footprint(0)[0]:.2f} m (vs {self.gsd_nadir:.2f} m at nadir)
"""
        return summary

    def explain_pushbroom(self) -> str:
        """Educational explanation of pushbroom scanning."""
        explanation = """
How Pushbroom Scanning Works
============================

Traditional vs Pushbroom:
┌─────────────────────┬─────────────────────┐
│    Whiskbroom       │     Pushbroom       │
├─────────────────────┼─────────────────────┤
│ Mirror sweeps       │ Linear array        │
│ across track        │ images full swath   │
│                     │                     │
│    ←─●─→            │   ════════════      │
│     ↓↑              │   ||||||||||||      │
│   ══════            │   ↓↓↓↓↓↓↓↓↓↓↓↓      │
│                     │                     │
│ One pixel at a time │ All pixels at once  │
└─────────────────────┴─────────────────────┘

Pushbroom Advantages:
  1. Longer dwell time: Each pixel gets the full integration
     period (vs brief moment when mirror sweeps past)
  2. Better SNR: More photons collected per pixel
  3. No moving parts: More reliable, no mirror jitter
  4. Consistent geometry: All pixels imaged simultaneously

Pushbroom Challenges:
  1. Detector uniformity: Need 1000+ pixels to behave identically
  2. Optical distortions: Smile and keystone effects
  3. Calibration: Must characterize each pixel individually

Building the Data Cube:
  Frame 1:  ████████████████████  (t=0 ms)
  Frame 2:  ████████████████████  (t=10 ms)
  Frame 3:  ████████████████████  (t=20 ms)
     ...
  Frame N:  ████████████████████  (t=N×10 ms)

  ↓ Stack frames ↓

  Result: 3D cube (along-track × cross-track × wavelength)

Each frame captures the full spectrum for every cross-track pixel.
The aircraft motion provides the third (along-track) dimension.
"""
        return explanation

    def visualize_scan_geometry(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate data for visualizing scan geometry.

        Returns:
            Tuple of (x_ground, y_ground, view_angles) for plotting
        """
        # Cross-track ground positions
        x = self.cross_track_position

        # View zenith angles
        vza = self.view_zenith

        # GSD variation
        gsd = np.array([self.ground_footprint(i)[0] for i in range(self.config.n_cross_track)])

        return x, vza, gsd
