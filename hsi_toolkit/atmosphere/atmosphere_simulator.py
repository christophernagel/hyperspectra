"""
Atmosphere Simulator - Main Integration Module

Combines gas absorption, Rayleigh scattering, and aerosol scattering
into a complete atmospheric radiative transfer model.

The key equation we're solving (simplified):

    L_sensor = L_path + T ⋅ (ρ ⋅ E_down / π) / (1 - ρ ⋅ S)

Where:
    L_sensor = Radiance measured by sensor
    L_path = Path radiance (atmospheric scattering)
    T = Total atmospheric transmission
    ρ = Surface reflectance
    E_down = Downwelling irradiance at surface
    S = Spherical albedo (backscatter factor)

This module computes T, L_path, E_down, and S from physical parameters.
"""

import numpy as np
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

from .gas_absorption import CombinedGasAbsorption
from .rayleigh_scattering import RayleighScattering
from .aerosol_scattering import AerosolScattering, AerosolType
from .solar_spectrum import SolarSpectrum


@dataclass
class AtmosphericState:
    """
    Atmospheric state parameters.

    Encapsulates all parameters needed to define atmospheric conditions.
    """
    # Water vapor
    pwv_cm: float = 1.5              # Precipitable water vapor (cm)

    # Aerosols
    aod_550: float = 0.15            # Aerosol optical depth at 550nm
    aerosol_type: str = 'continental'  # maritime, continental, urban, desert

    # Other gases
    ozone_du: float = 300.0          # Column ozone (Dobson Units)
    co2_ppm: float = 420.0           # CO2 concentration (ppm)

    # Geometry
    solar_zenith_deg: float = 30.0   # Solar zenith angle
    view_zenith_deg: float = 0.0     # Sensor view zenith angle
    relative_azimuth_deg: float = 0.0  # Relative azimuth

    # Surface
    surface_elevation_m: float = 0.0  # Surface elevation (m)
    surface_albedo: float = 0.15      # Mean surface albedo

    # Location/time (for solar calculations)
    day_of_year: int = 172            # Day of year (default: summer solstice)


class AtmosphereSimulator:
    """
    Complete atmospheric radiative transfer simulator.

    Integrates all atmospheric effects:
    1. Molecular absorption (H2O, O2, CO2, O3)
    2. Rayleigh (molecular) scattering
    3. Aerosol (Mie) scattering

    Provides methods to:
    - Calculate atmospheric transmission
    - Calculate path radiance
    - Calculate downwelling irradiance
    - Simulate the complete sun-surface-sensor path
    """

    def __init__(self, wavelengths: np.ndarray,
                 state: Optional[AtmosphericState] = None):
        """
        Initialize atmosphere simulator.

        Args:
            wavelengths: Wavelength array in nanometers (e.g., 400-2500)
            state: Initial atmospheric state (uses defaults if None)
        """
        self.wavelengths = np.asarray(wavelengths)
        self.n_wavelengths = len(wavelengths)

        # Initialize atmospheric state
        self.state = state if state is not None else AtmosphericState()

        # Initialize component models
        self._init_components()

        # Pre-calculate atmospheric terms
        self._calculate_atmospheric_terms()

    def _init_components(self):
        """Initialize component models."""
        self.gas_absorption = CombinedGasAbsorption(self.wavelengths)
        self.rayleigh = RayleighScattering(self.wavelengths)
        self.aerosol = AerosolScattering(
            self.wavelengths,
            AerosolType(self.state.aerosol_type)
        )
        self.solar = SolarSpectrum(self.wavelengths)

    def update_state(self, **kwargs):
        """
        Update atmospheric state parameters.

        Args:
            **kwargs: Any AtmosphericState parameters to update

        Example:
            atm.update_state(pwv_cm=2.5, aod_550=0.3)
        """
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

                # Update aerosol model if type changed
                if key == 'aerosol_type':
                    self.aerosol.set_aerosol_type(AerosolType(value))

        # Recalculate atmospheric terms
        self._calculate_atmospheric_terms()

    def _calculate_atmospheric_terms(self):
        """Pre-calculate all atmospheric terms for current state."""
        s = self.state

        # Airmass factors
        self.airmass_sun = self._calculate_airmass(s.solar_zenith_deg)
        self.airmass_view = self._calculate_airmass(s.view_zenith_deg)

        # Surface pressure from elevation
        scale_height = 8500  # m
        self.surface_pressure_mb = 1013.25 * np.exp(-s.surface_elevation_m / scale_height)

        # Gas transmission (two-way: sun-surface-sensor)
        self.T_gas_down = self.gas_absorption.transmission(
            pwv_cm=s.pwv_cm,
            ozone_du=s.ozone_du,
            co2_ppm=s.co2_ppm,
            surface_pressure_mb=self.surface_pressure_mb,
            airmass=self.airmass_sun
        )
        self.T_gas_up = self.gas_absorption.transmission(
            pwv_cm=s.pwv_cm,
            ozone_du=s.ozone_du,
            co2_ppm=s.co2_ppm,
            surface_pressure_mb=self.surface_pressure_mb,
            airmass=self.airmass_view
        )
        self.T_gas_total = self.T_gas_down * self.T_gas_up

        # Rayleigh transmission
        self.T_rayleigh_down = self.rayleigh.transmission(
            self.surface_pressure_mb, self.airmass_sun
        )
        self.T_rayleigh_up = self.rayleigh.transmission(
            self.surface_pressure_mb, self.airmass_view
        )
        self.T_rayleigh_total = self.T_rayleigh_down * self.T_rayleigh_up

        # Aerosol transmission
        self.T_aerosol_down = self.aerosol.transmission(
            s.aod_550, self.airmass_sun
        )
        self.T_aerosol_up = self.aerosol.transmission(
            s.aod_550, self.airmass_view
        )
        self.T_aerosol_total = self.T_aerosol_down * self.T_aerosol_up

        # Total transmission
        self.T_total_down = self.T_gas_down * self.T_rayleigh_down * self.T_aerosol_down
        self.T_total_up = self.T_gas_up * self.T_rayleigh_up * self.T_aerosol_up
        self.T_total = self.T_total_down * self.T_total_up

        # Solar irradiance at TOA
        earth_sun_dist = self.solar.earth_sun_distance(s.day_of_year)
        self.E_sun_toa = self.solar.irradiance(earth_sun_dist)

        # Downwelling irradiance at surface (direct + diffuse)
        mu_0 = np.cos(np.radians(s.solar_zenith_deg))
        self.E_direct = self.E_sun_toa * self.T_total_down * mu_0

        # Diffuse component (simplified estimate)
        diffuse_fraction = 0.1 + 0.3 * (1 - self.T_rayleigh_down) + 0.2 * (1 - self.T_aerosol_down)
        self.E_diffuse = self.E_sun_toa * diffuse_fraction * mu_0

        self.E_down = self.E_direct + self.E_diffuse

        # Path radiance
        self._calculate_path_radiance()

        # Spherical albedo (backscatter factor)
        self._calculate_spherical_albedo()

    def _calculate_airmass(self, zenith_deg: float) -> float:
        """Calculate airmass for given zenith angle."""
        if zenith_deg >= 90:
            return 40.0  # Maximum practical airmass

        # Kasten & Young (1989) formula
        z_rad = np.radians(zenith_deg)
        airmass = 1.0 / (np.cos(z_rad) + 0.50572 * (96.07995 - zenith_deg) ** (-1.6364))
        return min(airmass, 40.0)

    def _calculate_path_radiance(self):
        """Calculate total atmospheric path radiance."""
        s = self.state

        # Rayleigh path radiance
        L_rayleigh = self.rayleigh.path_radiance(
            self.E_sun_toa,
            s.solar_zenith_deg,
            s.view_zenith_deg,
            self.surface_pressure_mb,
            s.surface_albedo
        )

        # Aerosol path radiance
        L_aerosol = self.aerosol.path_radiance(
            self.E_sun_toa,
            s.solar_zenith_deg,
            s.view_zenith_deg,
            s.relative_azimuth_deg,
            s.aod_550,
            s.surface_albedo
        )

        # Total path radiance (attenuated by gas absorption on upward path)
        self.L_path = (L_rayleigh + L_aerosol) * self.T_gas_up

    def _calculate_spherical_albedo(self):
        """
        Calculate spherical albedo (atmosphere backscatter factor).

        This accounts for multiple reflections between surface and atmosphere.
        For bright surfaces, some reflected light bounces back down from
        the atmosphere, reflects again, and so on.
        """
        # Simplified estimate based on Rayleigh + aerosol optical depths
        tau_R = self.rayleigh.optical_depth(self.surface_pressure_mb, 1.0)
        tau_A = self.aerosol.optical_depth(self.state.aod_550, 1.0)

        # Spherical albedo ≈ 0.1 * τ_R + 0.05 * τ_A for typical conditions
        self.S = 0.1 * tau_R + 0.05 * tau_A * self.aerosol.model.ssa

    def get_transmission(self) -> Dict[str, np.ndarray]:
        """
        Get all transmission components.

        Returns:
            Dictionary with transmission arrays for each component
        """
        return {
            'gas': self.T_gas_total,
            'rayleigh': self.T_rayleigh_total,
            'aerosol': self.T_aerosol_total,
            'total': self.T_total,
            'downward': self.T_total_down,
            'upward': self.T_total_up
        }

    def get_gas_absorption_components(self) -> Dict[str, np.ndarray]:
        """Get individual gas transmission components."""
        s = self.state
        return self.gas_absorption.get_absorption_components(
            s.pwv_cm, s.ozone_du, s.co2_ppm,
            self.surface_pressure_mb, self.airmass_sun
        )

    def radiance_at_sensor(self, surface_reflectance: np.ndarray) -> np.ndarray:
        """
        Calculate radiance at sensor for given surface reflectance.

        This is the forward model:
            L = L_path + T_up * (ρ * E_down / π) / (1 - ρ * S)

        Args:
            surface_reflectance: Surface reflectance spectrum (0-1)

        Returns:
            Sensor radiance in same units as E_down (W/m²/sr/nm)
        """
        rho = np.asarray(surface_reflectance)

        # Surface-leaving radiance (Lambertian assumption)
        L_surface = rho * self.E_down / np.pi

        # Coupling term (multiple reflections)
        coupling = 1.0 / (1.0 - rho * self.S)

        # Radiance at sensor
        L_sensor = self.L_path + self.T_total_up * L_surface * coupling

        return L_sensor

    def reflectance_from_radiance(self, radiance: np.ndarray) -> np.ndarray:
        """
        Retrieve surface reflectance from measured radiance.

        This is the inverse model (atmospheric correction):
            ρ = π * (L - L_path) / (T_up * E_down + S * π * (L - L_path))

        Args:
            radiance: Measured radiance at sensor

        Returns:
            Retrieved surface reflectance
        """
        L = np.asarray(radiance)

        # Remove path radiance
        L_ground = L - self.L_path

        # Inverse of forward model
        numerator = np.pi * L_ground
        denominator = self.T_total_up * self.E_down + self.S * np.pi * L_ground

        # Avoid division by zero
        rho = np.where(denominator > 0, numerator / denominator, 0)

        # Clip to physical range (allow small negatives for noise)
        return np.clip(rho, -0.05, 1.5)

    def get_atmospheric_summary(self) -> Dict[str, Any]:
        """Get summary of current atmospheric state for display."""
        s = self.state

        # Mean values over wavelength range
        T_gas_mean = np.mean(self.T_gas_total)
        T_rayleigh_mean = np.mean(self.T_rayleigh_total)
        T_aerosol_mean = np.mean(self.T_aerosol_total)
        T_total_mean = np.mean(self.T_total)

        return {
            'state': {
                'pwv_cm': s.pwv_cm,
                'aod_550': s.aod_550,
                'aerosol_type': s.aerosol_type,
                'ozone_du': s.ozone_du,
                'solar_zenith': s.solar_zenith_deg,
                'view_zenith': s.view_zenith_deg,
                'elevation_m': s.surface_elevation_m,
            },
            'derived': {
                'surface_pressure_mb': self.surface_pressure_mb,
                'airmass_sun': self.airmass_sun,
                'airmass_view': self.airmass_view,
            },
            'transmission_mean': {
                'gas': T_gas_mean,
                'rayleigh': T_rayleigh_mean,
                'aerosol': T_aerosol_mean,
                'total': T_total_mean,
            },
            'path_radiance_mean': np.mean(self.L_path),
            'downwelling_mean': np.mean(self.E_down),
        }

    def sensitivity_analysis(self, parameter: str,
                             values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform sensitivity analysis for a parameter.

        Args:
            parameter: Parameter name (e.g., 'pwv_cm', 'aod_550')
            values: Array of values to test

        Returns:
            Dictionary with transmission arrays for each value
        """
        results = {
            'values': values,
            'transmission': [],
            'path_radiance': []
        }

        original_value = getattr(self.state, parameter)

        for val in values:
            self.update_state(**{parameter: val})
            results['transmission'].append(self.T_total.copy())
            results['path_radiance'].append(self.L_path.copy())

        # Restore original value
        self.update_state(**{parameter: original_value})

        results['transmission'] = np.array(results['transmission'])
        results['path_radiance'] = np.array(results['path_radiance'])

        return results

    def explain_atmospheric_correction(self, surface_reflectance: float = 0.3) -> str:
        """
        Educational explanation of atmospheric correction for current state.

        Args:
            surface_reflectance: Example reflectance value

        Returns:
            Explanation string
        """
        s = self.state

        # Calculate example values at 550nm
        idx_550 = np.argmin(np.abs(self.wavelengths - 550))

        rho = surface_reflectance
        L_sensor = self.radiance_at_sensor(np.full(self.n_wavelengths, rho))[idx_550]
        L_path = self.L_path[idx_550]
        T_up = self.T_total_up[idx_550]
        E_down = self.E_down[idx_550]

        explanation = f"""
Atmospheric Correction Example (at 550 nm)
==========================================

Current Atmospheric State:
  - Water vapor: {s.pwv_cm:.2f} cm
  - Aerosol optical depth: {s.aod_550:.3f}
  - Solar zenith: {s.solar_zenith_deg:.1f}°
  - View zenith: {s.view_zenith_deg:.1f}°

Step 1: Surface reflects sunlight
  - True surface reflectance: ρ = {rho:.2f}
  - Downwelling irradiance: E_down = {E_down:.2f} W/m²/nm
  - Surface-leaving radiance: L_surface = ρ × E_down / π = {rho * E_down / np.pi:.4f} W/m²/sr/nm

Step 2: Light travels up through atmosphere
  - Upward transmission: T_up = {T_up:.4f}
  - Attenuated surface signal: {T_up * rho * E_down / np.pi:.4f} W/m²/sr/nm

Step 3: Atmosphere adds path radiance
  - Path radiance: L_path = {L_path:.4f} W/m²/sr/nm
  - (Light scattered into view without touching surface)

Step 4: Sensor measures total
  - Measured radiance: L_sensor = {L_sensor:.4f} W/m²/sr/nm

To retrieve ρ from L_sensor, we must:
  1. Subtract path radiance: L_ground = L_sensor - L_path = {L_sensor - L_path:.4f}
  2. Divide by atmospheric terms: ρ = π × L_ground / (T_up × E_down)

This is atmospheric correction in a nutshell!
"""
        return explanation
