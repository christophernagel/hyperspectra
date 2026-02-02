"""
Rayleigh Scattering Module

Models molecular (Rayleigh) scattering by air molecules.

Physics:
    When light encounters particles much smaller than its wavelength
    (like N2 and O2 molecules), it scatters in all directions. The
    scattering probability varies as λ^(-4), meaning:

    - Blue light (400nm) scatters ~16x more than red light (700nm)
    - This is why the sky is blue and sunsets are red
    - This adds "path radiance" to sensor measurements

    The Rayleigh scattering coefficient is:

        β_R(λ) = (8π³/3) * ((n²-1)²/(N*λ⁴)) * ((6+3ρ)/(6-7ρ))

    where:
        n = refractive index of air
        N = number density of molecules
        λ = wavelength
        ρ = depolarization factor (~0.035 for air)

    For practical calculations, we use empirical fits to this formula.

References:
    - Bucholtz, 1995: Rayleigh-scattering calculations for the terrestrial atmosphere
    - Hansen & Travis, 1974: Light scattering in planetary atmospheres
"""

import numpy as np
from typing import Optional, Tuple


class RayleighScattering:
    """
    Rayleigh molecular scattering model.

    Models the λ^(-4) wavelength dependence of scattering by
    atmospheric molecules (primarily N2 and O2).
    """

    # Standard atmosphere parameters
    LOSCHMIDT = 2.6867774e19  # molecules/cm³ at STP
    DEPOLARIZATION = 0.0279   # King factor for air

    def __init__(self, wavelengths: np.ndarray):
        """
        Initialize Rayleigh scattering model.

        Args:
            wavelengths: Wavelength array in nanometers
        """
        self.wavelengths = np.asarray(wavelengths)
        self._precompute_coefficients()

    def _precompute_coefficients(self):
        """Precompute wavelength-dependent scattering coefficients."""
        # Convert wavelengths to micrometers for standard formula
        wl_um = self.wavelengths / 1000.0

        # Rayleigh optical depth at sea level for one air mass
        # Using Bucholtz (1995) parametrization
        # tau_R = A * λ^(-B) where A and B depend on wavelength range

        # For visible/NIR (λ > 0.55 µm)
        self._tau_coefficients = np.zeros_like(self.wavelengths)

        # UV/blue (λ < 0.5 µm)
        mask_uv = wl_um < 0.5
        if np.any(mask_uv):
            self._tau_coefficients[mask_uv] = (
                0.00864 * wl_um[mask_uv] ** (-(3.916 + 0.074 * wl_um[mask_uv] +
                                               0.050 / wl_um[mask_uv]))
            )

        # Visible/NIR (λ >= 0.5 µm)
        mask_vis = wl_um >= 0.5
        if np.any(mask_vis):
            self._tau_coefficients[mask_vis] = (
                0.00864 * wl_um[mask_vis] ** (-(3.916 + 0.074 * wl_um[mask_vis] +
                                                0.050 / wl_um[mask_vis]))
            )

    def optical_depth(self, surface_pressure_mb: float = 1013.25,
                      airmass: float = 1.0) -> np.ndarray:
        """
        Calculate Rayleigh optical depth.

        Args:
            surface_pressure_mb: Surface pressure in millibars
            airmass: Atmospheric path length factor

        Returns:
            Optical depth array

        Note:
            Optical depth scales linearly with pressure (more molecules = more scattering)
        """
        pressure_factor = surface_pressure_mb / 1013.25
        return self._tau_coefficients * pressure_factor * airmass

    def transmission(self, surface_pressure_mb: float = 1013.25,
                     airmass: float = 1.0) -> np.ndarray:
        """
        Calculate direct beam transmission through Rayleigh scattering.

        T = exp(-tau_R)

        This is the fraction of direct light that reaches the surface
        without being scattered.
        """
        tau = self.optical_depth(surface_pressure_mb, airmass)
        return np.exp(-tau)

    def scattering_coefficient(self, altitude_km: float = 0.0) -> np.ndarray:
        """
        Calculate volume scattering coefficient β(λ) at given altitude.

        Args:
            altitude_km: Altitude in kilometers

        Returns:
            Scattering coefficient (1/km)
        """
        # Approximate exponential atmosphere with scale height ~8 km
        scale_height = 8.0  # km
        density_factor = np.exp(-altitude_km / scale_height)

        # Base scattering coefficient at sea level (1/km)
        wl_um = self.wavelengths / 1000.0
        beta_0 = 0.0116 * wl_um ** (-4.08)  # Empirical fit

        return beta_0 * density_factor

    def phase_function(self, scattering_angle_deg: float) -> float:
        """
        Rayleigh phase function P(θ).

        Describes angular distribution of scattered light.

        P(θ) = (3/4) * (1 + cos²θ)

        Rayleigh scattering is nearly symmetric between forward
        and backward directions, with minima at 90°.

        Args:
            scattering_angle_deg: Angle between incident and scattered directions

        Returns:
            Phase function value (normalized so integral = 4π)
        """
        theta_rad = np.radians(scattering_angle_deg)
        return 0.75 * (1 + np.cos(theta_rad) ** 2)

    def path_radiance(self, solar_irradiance: np.ndarray,
                      solar_zenith_deg: float,
                      view_zenith_deg: float,
                      surface_pressure_mb: float = 1013.25,
                      surface_albedo: float = 0.0) -> np.ndarray:
        """
        Estimate Rayleigh path radiance (single scattering approximation).

        This is the light scattered into the sensor's view without
        touching the surface - it must be subtracted in atmospheric correction.

        Args:
            solar_irradiance: Solar irradiance spectrum (W/m²/nm)
            solar_zenith_deg: Solar zenith angle (degrees)
            view_zenith_deg: Sensor view zenith angle (degrees)
            surface_pressure_mb: Surface pressure (mb)
            surface_albedo: Mean surface albedo for multiple scattering

        Returns:
            Path radiance (same units as irradiance, per steradian)

        Note:
            This is a simplified single-scattering approximation.
            Full treatment requires multiple scattering calculations.
        """
        # Airmass factors
        mu_0 = np.cos(np.radians(solar_zenith_deg))
        mu = np.cos(np.radians(view_zenith_deg))

        if mu_0 <= 0:
            return np.zeros_like(self.wavelengths)

        airmass_sun = 1.0 / mu_0
        airmass_view = 1.0 / mu if mu > 0.1 else 10.0

        # Optical depth
        tau = self.optical_depth(surface_pressure_mb, 1.0)

        # Scattering angle (simplified: assume principal plane)
        scatter_angle = abs(solar_zenith_deg - view_zenith_deg)
        phase = self.phase_function(scatter_angle)

        # Single scattering path radiance (Chandrasekhar formula simplified)
        # L_path = (E0/4π) * P(θ) * ω₀ * [1 - exp(-τ(1/μ₀ + 1/μ))] / (1/μ₀ + 1/μ)
        # For Rayleigh, single scattering albedo ω₀ ≈ 1

        tau_total = tau * (airmass_sun + airmass_view)
        scatter_factor = (1 - np.exp(-tau_total)) / (airmass_sun + airmass_view)

        L_path = (solar_irradiance / (4 * np.pi)) * phase * scatter_factor

        # Multiple scattering correction (approximate)
        # Accounts for light bouncing between surface and atmosphere
        ms_factor = 1 + 0.5 * surface_albedo * tau

        return L_path * ms_factor

    def sky_radiance_distribution(self, solar_zenith_deg: float,
                                   n_angles: int = 90) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate relative sky radiance as function of view angle.

        Useful for visualizing why the sky is brightest near the horizon
        and near the sun.

        Args:
            solar_zenith_deg: Solar zenith angle
            n_angles: Number of view angles to compute

        Returns:
            Tuple of (view_zenith_angles, relative_radiance_at_450nm)
        """
        view_angles = np.linspace(0, 90, n_angles)

        # For demonstration, compute at 450 nm (blue)
        wl_450_idx = np.argmin(np.abs(self.wavelengths - 450))
        tau_450 = self._tau_coefficients[wl_450_idx]

        radiance = np.zeros(n_angles)
        for i, vza in enumerate(view_angles):
            mu = np.cos(np.radians(vza)) if vza < 90 else 0.01
            airmass = 1.0 / mu

            # Path radiance increases with airmass
            scatter_angle = abs(solar_zenith_deg - vza)
            phase = self.phase_function(scatter_angle)

            radiance[i] = phase * (1 - np.exp(-tau_450 * airmass))

        # Normalize
        radiance /= np.max(radiance) if np.max(radiance) > 0 else 1

        return view_angles, radiance

    def explain_blue_sky(self) -> str:
        """
        Return educational explanation of why the sky is blue.
        """
        tau_400 = self._tau_coefficients[np.argmin(np.abs(self.wavelengths - 400))]
        tau_700 = self._tau_coefficients[np.argmin(np.abs(self.wavelengths - 700))]

        ratio = tau_400 / tau_700 if tau_700 > 0 else 10

        explanation = f"""
Why is the sky blue?

Rayleigh scattering probability varies as λ^(-4), meaning shorter
wavelengths scatter much more strongly than longer wavelengths.

At your current wavelength grid:
  - Optical depth at 400nm (blue): {tau_400:.4f}
  - Optical depth at 700nm (red):  {tau_700:.4f}
  - Ratio (blue/red): {ratio:.1f}x

Blue light scatters {ratio:.1f}x more than red light!

When sunlight enters the atmosphere:
  1. Blue light scatters in all directions (including toward your eyes)
  2. Red light mostly continues straight (direct beam)
  3. The scattered blue light fills the sky dome
  4. Hence: blue sky during day, red/orange sunsets (we see the
     unscattered direct beam through long atmospheric paths)
"""
        return explanation
