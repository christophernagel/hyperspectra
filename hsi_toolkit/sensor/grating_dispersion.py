"""
Diffraction Grating Dispersion Module

Models how a diffraction grating separates light by wavelength
in an imaging spectrometer.

Physics Background:
    When light hits a diffraction grating (a surface with many parallel
    grooves), each groove acts as a secondary source. Constructive
    interference occurs at angles where path differences equal whole
    numbers of wavelengths:

    Grating Equation:
        d × sin(θ) = m × λ

    where:
        d = groove spacing (typically 1-10 µm)
        θ = diffraction angle
        m = diffraction order (integer: 0, ±1, ±2, ...)
        λ = wavelength

    Different wavelengths diffract at different angles, creating
    spectral dispersion.

Spectrometer Design:
    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │  Light from slit                                      │
    │       │                                               │
    │       ▼                                               │
    │  ┌─────────┐                                          │
    │  │Collimator│  Makes rays parallel                    │
    │  └────┬────┘                                          │
    │       │                                               │
    │       ▼                                               │
    │  ╔═════════╗                                          │
    │  ║ Grating ║  Disperses by wavelength                 │
    │  ╚════╤════╝                                          │
    │       │╲                                              │
    │       │  ╲  Different λ go to different angles        │
    │       ▼    ╲                                          │
    │  ┌─────────┐ ╲                                        │
    │  │Focuser  │  ╲  Focuses onto detector                │
    │  └────┬────┘   ╲                                      │
    │       │         ╲                                     │
    │  ═════════════════  Detector (spectral × spatial)     │
    │                                                       │
    └───────────────────────────────────────────────────────┘
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class SpectralCalibration:
    """
    Spectral calibration parameters.

    Maps detector row to wavelength and characterizes
    the spectral response function (SRF).
    """
    wavelength_centers: np.ndarray   # Center wavelength for each channel (nm)
    fwhm: np.ndarray                  # Full width at half maximum (nm)
    wavelength_uncertainty: np.ndarray = None  # Wavelength calibration uncertainty


class GratingDispersion:
    """
    Diffraction grating dispersion model.

    Models how a diffraction grating separates wavelengths and
    how they map to detector positions.
    """

    # Speed of light (m/s)
    C = 2.998e8

    def __init__(self,
                 groove_density: float = 600,   # lines/mm
                 blaze_wavelength: float = 1000,  # nm
                 order: int = 1,
                 focal_length_mm: float = 200):
        """
        Initialize grating model.

        Args:
            groove_density: Grooves per millimeter
            blaze_wavelength: Blaze wavelength in nm (peak efficiency)
            order: Diffraction order (typically 1 or -1)
            focal_length_mm: Spectrometer focal length
        """
        self.groove_density = groove_density
        self.blaze_wavelength = blaze_wavelength
        self.order = order
        self.focal_length_mm = focal_length_mm

        # Groove spacing in nm
        self.d_nm = 1e6 / groove_density  # nm

        # Calculate derived parameters
        self._calculate_parameters()

    def _calculate_parameters(self):
        """Calculate derived grating parameters."""
        # Blaze angle (angle of groove facets)
        # At blaze wavelength, specular reflection equals diffraction angle
        sin_blaze = self.order * self.blaze_wavelength / self.d_nm
        self.blaze_angle_deg = np.degrees(np.arcsin(np.clip(sin_blaze, -1, 1)))

        # Angular dispersion at blaze wavelength (rad/nm)
        self.angular_dispersion = self.order / (self.d_nm * np.cos(np.radians(self.blaze_angle_deg)))

        # Linear dispersion at detector (nm/mm)
        self.linear_dispersion = 1 / (self.angular_dispersion * self.focal_length_mm)

    def diffraction_angle(self, wavelength_nm: np.ndarray,
                          incident_angle_deg: float = 0) -> np.ndarray:
        """
        Calculate diffraction angle for given wavelength(s).

        Uses grating equation: d(sin(θ_i) + sin(θ_d)) = mλ

        Args:
            wavelength_nm: Wavelength(s) in nanometers
            incident_angle_deg: Incident angle (from normal)

        Returns:
            Diffraction angle(s) in degrees
        """
        wavelength_nm = np.asarray(wavelength_nm)

        # Grating equation
        sin_theta_i = np.sin(np.radians(incident_angle_deg))
        sin_theta_d = (self.order * wavelength_nm / self.d_nm) - sin_theta_i

        # Clip to valid range
        sin_theta_d = np.clip(sin_theta_d, -1, 1)

        return np.degrees(np.arcsin(sin_theta_d))

    def wavelength_from_angle(self, diffraction_angle_deg: float,
                               incident_angle_deg: float = 0) -> float:
        """
        Calculate wavelength for a given diffraction angle.

        Inverse of diffraction_angle().

        Args:
            diffraction_angle_deg: Diffraction angle from normal
            incident_angle_deg: Incident angle from normal

        Returns:
            Wavelength in nm
        """
        sin_theta_i = np.sin(np.radians(incident_angle_deg))
        sin_theta_d = np.sin(np.radians(diffraction_angle_deg))

        wavelength = self.d_nm * (sin_theta_i + sin_theta_d) / self.order
        return wavelength

    def detector_position(self, wavelength_nm: np.ndarray,
                          incident_angle_deg: float = 0) -> np.ndarray:
        """
        Calculate detector position for wavelength(s).

        Converts diffraction angle to position on detector plane.

        Args:
            wavelength_nm: Wavelength(s) in nm
            incident_angle_deg: Incident angle

        Returns:
            Position(s) on detector in mm (relative to optical axis)
        """
        theta_d = self.diffraction_angle(wavelength_nm, incident_angle_deg)

        # Position = focal_length × tan(θ_d - θ_blaze)
        # Relative to blaze wavelength position
        theta_blaze = self.diffraction_angle(self.blaze_wavelength, incident_angle_deg)

        position_mm = self.focal_length_mm * np.tan(np.radians(theta_d - theta_blaze))

        return position_mm

    def efficiency(self, wavelength_nm: np.ndarray) -> np.ndarray:
        """
        Calculate grating efficiency vs wavelength.

        Blazed gratings have peak efficiency at the blaze wavelength.
        Efficiency falls off at other wavelengths.

        Uses simplified sinc² model for blaze efficiency envelope.

        Args:
            wavelength_nm: Wavelength array

        Returns:
            Efficiency array (0-1)
        """
        wavelength_nm = np.asarray(wavelength_nm)

        # Blaze function: sinc²(π × (λ - λ_blaze) / λ_blaze)
        x = np.pi * (wavelength_nm - self.blaze_wavelength) / self.blaze_wavelength

        # Avoid division by zero at blaze wavelength
        efficiency = np.where(
            np.abs(x) < 0.01,
            1.0,
            (np.sin(x) / x) ** 2
        )

        # Clip to reasonable range
        return np.clip(efficiency, 0.1, 1.0)

    def spectral_resolution(self, wavelength_nm: float,
                            slit_width_um: float = 20) -> float:
        """
        Calculate spectral resolution (FWHM) at given wavelength.

        Resolution is limited by:
        1. Slit width (image of slit on detector)
        2. Diffraction limit of grating

        Args:
            wavelength_nm: Wavelength in nm
            slit_width_um: Entrance slit width in µm

        Returns:
            Spectral FWHM in nm
        """
        # Slit-limited resolution
        # FWHM ≈ slit_width × linear_dispersion
        slit_width_mm = slit_width_um / 1000
        fwhm_slit = slit_width_mm * self.linear_dispersion

        # Diffraction-limited resolution
        # R = λ/Δλ = m × N, where N = number of illuminated grooves
        # Assuming beam fills ~50mm of grating
        n_grooves = 50 * self.groove_density
        fwhm_diff = wavelength_nm / (self.order * n_grooves)

        # Total resolution (quadrature sum)
        fwhm_total = np.sqrt(fwhm_slit**2 + fwhm_diff**2)

        return fwhm_total

    def spectral_response_function(self, center_wavelength: float,
                                    wavelength_grid: np.ndarray,
                                    slit_width_um: float = 20) -> np.ndarray:
        """
        Calculate spectral response function (SRF) for a channel.

        The SRF describes how a monochromatic input is spread
        across the detector. It's approximately Gaussian.

        Args:
            center_wavelength: Channel center wavelength (nm)
            wavelength_grid: Fine wavelength grid for SRF (nm)
            slit_width_um: Slit width

        Returns:
            Normalized SRF array
        """
        fwhm = self.spectral_resolution(center_wavelength, slit_width_um)
        sigma = fwhm / 2.355  # FWHM to sigma for Gaussian

        srf = np.exp(-0.5 * ((wavelength_grid - center_wavelength) / sigma) ** 2)

        # Normalize to unit integral
        srf /= np.trapz(srf, wavelength_grid)

        return srf

    def create_spectral_calibration(self,
                                     wavelength_range: Tuple[float, float] = (380, 2500),
                                     n_channels: int = 284,
                                     slit_width_um: float = 20) -> SpectralCalibration:
        """
        Create spectral calibration for detector channels.

        Args:
            wavelength_range: (min, max) wavelength in nm
            n_channels: Number of spectral channels
            slit_width_um: Slit width for resolution calculation

        Returns:
            SpectralCalibration object
        """
        # Linear wavelength sampling (simplified)
        # Real systems may have slight nonlinearity
        wavelengths = np.linspace(
            wavelength_range[0],
            wavelength_range[1],
            n_channels
        )

        # FWHM for each channel
        fwhm = np.array([
            self.spectral_resolution(wl, slit_width_um)
            for wl in wavelengths
        ])

        # Wavelength uncertainty (typical ~0.1 nm)
        uncertainty = np.full(n_channels, 0.1)

        return SpectralCalibration(
            wavelength_centers=wavelengths,
            fwhm=fwhm,
            wavelength_uncertainty=uncertainty
        )

    def simulate_order_overlap(self, wavelength_nm: np.ndarray) -> dict:
        """
        Check for order overlap (different orders at same angle).

        Order overlap occurs when:
            m₁ × λ₁ = m₂ × λ₂

        For example, 2nd order of 500nm appears at same angle as
        1st order of 1000nm.

        Args:
            wavelength_nm: Primary wavelengths being measured

        Returns:
            Dictionary of overlap information
        """
        overlaps = []

        for wl in wavelength_nm:
            # Check if m=2 of some shorter wavelength overlaps
            wl_overlap_m2 = wl * self.order / 2
            if wl_overlap_m2 > 200:  # UV cutoff
                overlaps.append({
                    'primary_wl': wl,
                    'primary_order': self.order,
                    'overlap_wl': wl_overlap_m2,
                    'overlap_order': 2 * self.order
                })

        return {
            'n_overlaps': len(overlaps),
            'overlaps': overlaps[:5],  # First 5 examples
            'note': "Order-sorting filters typically used to prevent overlap"
        }

    def get_grating_summary(self) -> str:
        """Get summary of grating parameters."""
        summary = f"""
Diffraction Grating Parameters
==============================

Physical:
  Groove density: {self.groove_density} lines/mm
  Groove spacing: {self.d_nm:.2f} nm
  Blaze wavelength: {self.blaze_wavelength} nm
  Blaze angle: {self.blaze_angle_deg:.1f}°
  Diffraction order: {self.order}

Optical System:
  Focal length: {self.focal_length_mm} mm
  Angular dispersion: {self.angular_dispersion*1000:.3f} mrad/nm
  Linear dispersion: {self.linear_dispersion:.3f} nm/mm

At blaze wavelength ({self.blaze_wavelength} nm):
  Diffraction angle: {self.diffraction_angle(self.blaze_wavelength):.1f}°
  Efficiency: ~100% (peak)

Resolution (20 µm slit):
  @ 500 nm: {self.spectral_resolution(500):.2f} nm FWHM
  @ 1000 nm: {self.spectral_resolution(1000):.2f} nm FWHM
  @ 2000 nm: {self.spectral_resolution(2000):.2f} nm FWHM
"""
        return summary

    def explain_grating_equation(self) -> str:
        """Educational explanation of the grating equation."""
        explanation = f"""
The Grating Equation
====================

When light hits a grating with groove spacing d, waves from adjacent
grooves interfere. Constructive interference (bright output) occurs
when the path difference equals a whole number of wavelengths:

    d × sin(θ) = m × λ

where:
    d = groove spacing = {self.d_nm:.1f} nm
    θ = diffraction angle
    m = order (integer)
    λ = wavelength

Example with your grating (m = {self.order}):

    Wavelength     sin(θ)        Angle θ
    ─────────────────────────────────────
    400 nm         {self.order * 400 / self.d_nm:.3f}          {self.diffraction_angle(400):.1f}°
    550 nm         {self.order * 550 / self.d_nm:.3f}          {self.diffraction_angle(550):.1f}°
    700 nm         {self.order * 700 / self.d_nm:.3f}          {self.diffraction_angle(700):.1f}°
    1000 nm        {self.order * 1000 / self.d_nm:.3f}          {self.diffraction_angle(1000):.1f}°
    2000 nm        {self.order * 2000 / self.d_nm:.3f}          {self.diffraction_angle(2000):.1f}°

Different wavelengths diffract at different angles - this is how
the spectrometer separates colors!

Blazed Gratings:
  Your grating is "blazed" at {self.blaze_wavelength} nm, meaning the
  groove facets are angled to maximize efficiency at this wavelength.
  This is like tilting tiny mirrors to reflect into the desired order.
"""
        return explanation

    def visualize_dispersion(self, wavelengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate data for visualizing dispersion.

        Returns:
            Tuple of (wavelengths, angles, positions) for plotting
        """
        angles = self.diffraction_angle(wavelengths)
        positions = self.detector_position(wavelengths)

        return wavelengths, angles, positions
