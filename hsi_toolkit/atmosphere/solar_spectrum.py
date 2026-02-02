"""
Solar Spectrum Module

Provides solar irradiance data for radiative transfer calculations.

Physics:
    The Sun emits radiation approximately as a 5778K blackbody, but with
    absorption features from the solar atmosphere (Fraunhofer lines).

    At the top of Earth's atmosphere (TOA), the solar irradiance is
    ~1361 W/m² (solar constant). The spectral distribution peaks in
    the visible (~500nm) and extends from UV to infrared.

    For remote sensing calculations, we need the spectral irradiance
    E₀(λ) in W/m²/nm.

Data Sources:
    - ASTM G173-03: Standard reference spectrum
    - Thuillier (2003): High-resolution solar spectrum
    - TSIS-1: Modern satellite measurements

References:
    - Gueymard, 2004: The sun's total and spectral irradiance
    - Coddington et al., 2016: A solar irradiance climate data record
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class SolarSpectrum:
    """
    Solar spectral irradiance model.

    Provides extraterrestrial (top-of-atmosphere) solar irradiance
    and methods to adjust for Earth-Sun distance and solar geometry.
    """

    # Solar constant (total solar irradiance) in W/m²
    SOLAR_CONSTANT = 1361.0

    def __init__(self, wavelengths: np.ndarray):
        """
        Initialize solar spectrum model.

        Args:
            wavelengths: Wavelength array in nanometers
        """
        self.wavelengths = np.asarray(wavelengths)
        self._compute_solar_spectrum()

    def _compute_solar_spectrum(self):
        """
        Compute solar spectral irradiance using Planck function
        with empirical corrections.

        This provides a smooth approximation to the true solar spectrum.
        For high-accuracy work, use tabulated data (ASTM G173).
        """
        # Planck function for 5778K blackbody
        T_sun = 5778  # K
        h = 6.626e-34  # Planck constant
        c = 2.998e8    # Speed of light
        k = 1.381e-23  # Boltzmann constant

        # Convert wavelength to meters
        wl_m = self.wavelengths * 1e-9

        # Planck function B(λ,T) in W/m³/sr
        exp_term = h * c / (wl_m * k * T_sun)
        # Clip to avoid overflow
        exp_term = np.clip(exp_term, 0, 700)
        B_lambda = (2 * h * c**2 / wl_m**5) / (np.exp(exp_term) - 1)

        # Scale to match TOA solar irradiance
        # Account for solid angle of sun and Earth-Sun distance
        # Empirical scaling factor to match ASTM G173
        R_sun = 6.957e8   # Sun radius (m)
        d_AU = 1.496e11   # 1 AU in meters
        omega_sun = np.pi * (R_sun / d_AU)**2  # Solid angle of sun

        # E₀(λ) = π * B(λ,T) * (R_sun/d_AU)²
        self._e0_base = np.pi * B_lambda * (R_sun / d_AU)**2

        # Convert from W/m³ to W/m²/nm
        self._e0_base *= 1e-9

        # Empirical correction for Fraunhofer lines (simplified)
        # Real spectrum has absorption features we're approximating
        self._apply_fraunhofer_correction()

    def _apply_fraunhofer_correction(self):
        """Apply simplified Fraunhofer absorption corrections."""
        # Major Fraunhofer lines (simplified as Gaussian dips)
        fraunhofer = [
            (393.4, 5, 0.15),   # Ca II K
            (396.8, 5, 0.12),   # Ca II H
            (430.8, 3, 0.08),   # CH G band
            (486.1, 3, 0.10),   # H-beta
            (516.7, 2, 0.05),   # Mg I b
            (518.4, 2, 0.05),   # Mg I b
            (527.0, 2, 0.04),   # Fe I
            (588.9, 3, 0.08),   # Na D2
            (589.6, 3, 0.08),   # Na D1
            (656.3, 4, 0.12),   # H-alpha
            (686.7, 2, 0.06),   # O2 B-band (atmospheric)
            (759.4, 4, 0.10),   # O2 A-band (atmospheric)
        ]

        correction = np.ones_like(self.wavelengths)
        for center, width, depth in fraunhofer:
            sigma = width / 2.355
            correction -= depth * np.exp(-0.5 * ((self.wavelengths - center) / sigma)**2)

        self._e0_base *= np.clip(correction, 0.7, 1.0)

    def irradiance(self, earth_sun_distance_au: float = 1.0) -> np.ndarray:
        """
        Get solar spectral irradiance at TOA.

        Args:
            earth_sun_distance_au: Earth-Sun distance in AU
                (varies from 0.983 to 1.017 over the year)

        Returns:
            Solar irradiance in W/m²/nm
        """
        # Irradiance varies as 1/r²
        distance_factor = 1.0 / (earth_sun_distance_au ** 2)
        return self._e0_base * distance_factor

    def irradiance_at_surface(self, solar_zenith_deg: float,
                               earth_sun_distance_au: float = 1.0) -> np.ndarray:
        """
        Get solar irradiance at a horizontal surface (no atmosphere).

        Args:
            solar_zenith_deg: Solar zenith angle in degrees
            earth_sun_distance_au: Earth-Sun distance in AU

        Returns:
            Surface irradiance in W/m²/nm
        """
        cos_sza = np.cos(np.radians(solar_zenith_deg))
        if cos_sza <= 0:
            return np.zeros_like(self.wavelengths)

        return self.irradiance(earth_sun_distance_au) * cos_sza

    @staticmethod
    def earth_sun_distance(day_of_year: int) -> float:
        """
        Calculate Earth-Sun distance for a given day.

        Earth's orbit is elliptical with eccentricity ~0.017.
        Perihelion (closest) is around January 3.
        Aphelion (farthest) is around July 4.

        Args:
            day_of_year: Day of year (1-365)

        Returns:
            Distance in AU
        """
        # Simple sinusoidal approximation
        # Perihelion at day ~3, so phase shift
        angle = 2 * np.pi * (day_of_year - 3) / 365
        eccentricity = 0.0167
        distance = 1.0 - eccentricity * np.cos(angle)
        return distance

    @staticmethod
    def solar_zenith_angle(latitude_deg: float, longitude_deg: float,
                           day_of_year: int, hour_utc: float) -> float:
        """
        Calculate solar zenith angle.

        Args:
            latitude_deg: Latitude in degrees (-90 to 90)
            longitude_deg: Longitude in degrees (-180 to 180)
            day_of_year: Day of year (1-365)
            hour_utc: Hour in UTC (0-24)

        Returns:
            Solar zenith angle in degrees
        """
        # Declination angle
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))

        # Hour angle
        solar_noon_offset = longitude_deg / 15  # hours from UTC
        hour_angle = 15 * (hour_utc + solar_noon_offset - 12)

        # Solar zenith angle
        lat_rad = np.radians(latitude_deg)
        dec_rad = np.radians(declination)
        ha_rad = np.radians(hour_angle)

        cos_sza = (np.sin(lat_rad) * np.sin(dec_rad) +
                   np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))

        sza = np.degrees(np.arccos(np.clip(cos_sza, -1, 1)))
        return sza

    def get_integrated_irradiance(self) -> float:
        """
        Calculate total integrated solar irradiance.

        Returns:
            Total irradiance in W/m²

        Note:
            Should be close to solar constant (~1361 W/m²)
            if wavelength range covers most of solar emission.
        """
        # Trapezoidal integration
        return np.trapz(self._e0_base, self.wavelengths)

    def explain_solar_spectrum(self) -> str:
        """Educational explanation of the solar spectrum."""
        peak_wl = self.wavelengths[np.argmax(self._e0_base)]
        total_irradiance = self.get_integrated_irradiance()

        explanation = f"""
The Solar Spectrum
==================

The Sun is approximately a 5778K blackbody, with peak emission
at ~500 nm (green light). However, the solar atmosphere absorbs
at specific wavelengths, creating dark lines (Fraunhofer lines).

Your spectrum coverage: {self.wavelengths[0]:.0f} - {self.wavelengths[-1]:.0f} nm
Peak wavelength: {peak_wl:.0f} nm
Integrated irradiance: {total_irradiance:.1f} W/m²
  (Solar constant: ~1361 W/m²)

Key spectral regions:
  - UV (< 400 nm): High energy, absorbed by ozone
  - Visible (400-700 nm): Peak solar output, what we see
  - NIR (700-1400 nm): Significant energy, water vapor absorption
  - SWIR (1400-2500 nm): Less energy, more atmospheric absorption

Why this matters for remote sensing:
  1. More solar photons = better signal-to-noise in visible
  2. NIR/SWIR useful for vegetation, minerals, water
  3. Must account for Earth-Sun distance (±3% over year)
  4. Solar zenith angle affects illumination intensity
"""
        return explanation


class ASTMG173Spectrum(SolarSpectrum):
    """
    Solar spectrum based on ASTM G173-03 standard.

    This is the reference spectrum used for most remote sensing
    and photovoltaic applications. Values are tabulated rather
    than computed from blackbody model.

    Note: For simplicity, this class interpolates tabulated
    values. For production use, load the actual ASTM data file.
    """

    # Key reference values from ASTM G173-03 (W/m²/nm)
    # Extraterrestrial spectrum at 1 AU
    REFERENCE_DATA = {
        # wavelength (nm): irradiance (W/m²/nm)
        300: 0.5357,
        350: 1.0540,
        400: 1.5180,
        450: 2.0660,
        500: 1.9580,
        550: 1.8940,
        600: 1.8170,
        650: 1.6520,
        700: 1.5240,
        750: 1.3610,
        800: 1.1900,
        850: 1.0560,
        900: 0.9450,
        950: 0.8440,
        1000: 0.7534,
        1100: 0.6310,
        1200: 0.5070,
        1300: 0.4166,
        1400: 0.3346,
        1500: 0.2893,
        1600: 0.2430,
        1700: 0.2059,
        1800: 0.1737,
        1900: 0.1479,
        2000: 0.1273,
        2100: 0.1074,
        2200: 0.0926,
        2300: 0.0793,
        2400: 0.0683,
        2500: 0.0584,
    }

    def _compute_solar_spectrum(self):
        """Interpolate from ASTM reference data."""
        ref_wl = np.array(list(self.REFERENCE_DATA.keys()))
        ref_irr = np.array(list(self.REFERENCE_DATA.values()))

        # Interpolate to our wavelength grid
        self._e0_base = np.interp(self.wavelengths, ref_wl, ref_irr)

        # Handle extrapolation (set to small value)
        self._e0_base[self.wavelengths < ref_wl.min()] = 0.01
        self._e0_base[self.wavelengths > ref_wl.max()] = 0.01
