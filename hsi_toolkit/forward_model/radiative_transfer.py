"""
Radiative Transfer Core Module

Implements the fundamental equations that describe how electromagnetic
radiation interacts with the atmosphere and surface.

The Radiative Transfer Equation:
    The change in radiance L along a path ds is:

    dL/ds = -κ(s)·L + j(s)

    where:
        κ(s) = extinction coefficient (absorption + scattering)
        j(s) = source function (emission + in-scattering)

For solar remote sensing in the visible/NIR/SWIR:
    - Thermal emission is negligible (λ < 3μm)
    - Main sources: direct solar, scattered solar (path radiance)
    - Main sinks: absorption and out-scattering

The At-Sensor Radiance Equation:
    L_sensor = L_path + T_up · (ρ · E_down / π) / (1 - ρ · S)

    where:
        L_path  = atmospheric path radiance
        T_up    = upward atmospheric transmittance
        ρ       = surface reflectance
        E_down  = downwelling irradiance at surface
        S       = spherical albedo (adjacency effect)

References:
    - Vermote et al., 1997: Second Simulation of a Satellite Signal (6S)
    - Richter & Schläpfer, 2002: Geo-atmospheric processing of airborne data
    - Gao et al., 2009: Atmospheric correction algorithms for hyperspectral
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class RTParameters:
    """
    Radiative transfer parameters for a single atmospheric state.

    All arrays are dimensioned by wavelength.
    """
    wavelength_nm: np.ndarray        # Wavelength grid (nm)

    # Transmittances (0-1)
    T_total: np.ndarray              # Total two-way transmittance
    T_down_direct: np.ndarray        # Direct downward transmittance
    T_down_diffuse: np.ndarray       # Diffuse downward transmittance
    T_up: np.ndarray                 # Upward transmittance

    # Radiances and irradiances (W/m²/sr/nm or W/m²/nm)
    L_path: np.ndarray               # Path radiance
    E_sun_toa: np.ndarray            # Solar irradiance at TOA
    E_down_surface: np.ndarray       # Downwelling irradiance at surface

    # Auxiliary
    spherical_albedo: np.ndarray     # Atmospheric backscatter
    solar_zenith_deg: float          # Solar zenith angle
    view_zenith_deg: float           # Viewing zenith angle


class RadiativeTransfer:
    """
    Core radiative transfer calculations.

    Implements the fundamental RT equations connecting surface
    reflectance to at-sensor radiance.
    """

    def __init__(self, wavelengths: np.ndarray):
        """
        Initialize RT calculator.

        Args:
            wavelengths: Wavelength grid in nm
        """
        self.wavelengths = np.asarray(wavelengths)
        self.n_wavelengths = len(wavelengths)

    def surface_to_toa_radiance(self,
                                  reflectance: np.ndarray,
                                  rt_params: RTParameters) -> np.ndarray:
        """
        Calculate top-of-atmosphere radiance from surface reflectance.

        This is the FORWARD model:
            Given ρ (surface), compute L (sensor)

        The equation (including adjacency effect):
            L_sensor = L_path + T_up × (ρ × E_down/π) / (1 - ρ×S)

        Args:
            reflectance: Surface reflectance (0-1)
            rt_params: Atmospheric parameters

        Returns:
            At-sensor radiance (W/m²/sr/nm)
        """
        reflectance = np.asarray(reflectance)

        # Reflected radiance at surface (Lambertian assumption)
        # L_surface = ρ × E_down / π
        L_surface = reflectance * rt_params.E_down_surface / np.pi

        # Multiple scattering correction (adjacency effect)
        # Accounts for light bouncing between surface and atmosphere
        denominator = 1 - reflectance * rt_params.spherical_albedo
        L_surface_corrected = L_surface / denominator

        # Transmit through atmosphere and add path radiance
        L_toa = rt_params.L_path + rt_params.T_up * L_surface_corrected

        return L_toa

    def toa_radiance_to_reflectance(self,
                                     radiance: np.ndarray,
                                     rt_params: RTParameters) -> np.ndarray:
        """
        Calculate surface reflectance from at-sensor radiance.

        This is the INVERSE model (atmospheric correction):
            Given L (sensor), compute ρ (surface)

        Inverting the forward equation:
            ρ = π(L - L_path) / [T_up × E_down + S × π(L - L_path)]

        Args:
            radiance: At-sensor radiance (W/m²/sr/nm)
            rt_params: Atmospheric parameters

        Returns:
            Surface reflectance (0-1)
        """
        radiance = np.asarray(radiance)

        # Remove path radiance
        L_ground_contribution = radiance - rt_params.L_path

        # Numerator
        numerator = np.pi * L_ground_contribution

        # Denominator (includes adjacency correction)
        denominator = (rt_params.T_up * rt_params.E_down_surface +
                      rt_params.spherical_albedo * numerator)

        # Avoid division by zero
        reflectance = np.where(
            denominator > 0,
            numerator / denominator,
            0
        )

        return reflectance

    def apparent_reflectance(self, radiance: np.ndarray,
                              E_sun_toa: np.ndarray,
                              solar_zenith_deg: float) -> np.ndarray:
        """
        Calculate apparent (at-sensor) reflectance.

        This is a quick approximation that ignores atmospheric effects:
            ρ* = π × L / (E₀ × cos(θ_s))

        Used for quick visualization and as starting point for
        iterative corrections.

        Args:
            radiance: At-sensor radiance
            E_sun_toa: Solar irradiance at TOA
            solar_zenith_deg: Solar zenith angle

        Returns:
            Apparent reflectance
        """
        cos_sza = np.cos(np.radians(solar_zenith_deg))

        return np.pi * radiance / (E_sun_toa * cos_sza)

    def calculate_components(self,
                              reflectance: np.ndarray,
                              rt_params: RTParameters) -> Dict[str, np.ndarray]:
        """
        Break down the at-sensor radiance into components.

        Useful for understanding which effects dominate.

        Args:
            reflectance: Surface reflectance
            rt_params: Atmospheric parameters

        Returns:
            Dictionary with radiance components
        """
        reflectance = np.asarray(reflectance)

        # Total radiance
        L_total = self.surface_to_toa_radiance(reflectance, rt_params)

        # Path radiance (atmosphere only)
        L_path = rt_params.L_path

        # Surface contribution
        L_surface = L_total - L_path

        # Direct beam contribution (approximate)
        L_direct = (reflectance * rt_params.E_sun_toa *
                   np.cos(np.radians(rt_params.solar_zenith_deg)) / np.pi *
                   rt_params.T_down_direct * rt_params.T_up)

        # Diffuse contribution (approximate)
        L_diffuse = L_surface - L_direct

        return {
            'total': L_total,
            'path': L_path,
            'surface': L_surface,
            'direct': np.maximum(L_direct, 0),
            'diffuse': np.maximum(L_diffuse, 0),
            'path_fraction': L_path / np.maximum(L_total, 1e-10),
            'direct_fraction': np.maximum(L_direct, 0) / np.maximum(L_surface, 1e-10)
        }

    def sensitivity_to_reflectance(self, reflectance: np.ndarray,
                                    rt_params: RTParameters) -> np.ndarray:
        """
        Calculate dL/dρ (sensitivity of radiance to reflectance).

        Higher values mean the sensor is more sensitive to surface
        changes - good for discrimination. Lower values indicate
        atmospheric dominance.

        Args:
            reflectance: Surface reflectance
            rt_params: Atmospheric parameters

        Returns:
            Sensitivity (W/m²/sr/nm per unit reflectance)
        """
        reflectance = np.asarray(reflectance)

        # From L = L_path + T_up × (ρ × E_down/π) / (1 - ρ×S)
        # dL/dρ = T_up × E_down / (π × (1 - ρ×S)²)

        denominator = (1 - reflectance * rt_params.spherical_albedo) ** 2

        sensitivity = (rt_params.T_up * rt_params.E_down_surface /
                      (np.pi * denominator))

        return sensitivity

    def equivalent_reflectance_uncertainty(self,
                                            radiance_noise: np.ndarray,
                                            reflectance: np.ndarray,
                                            rt_params: RTParameters) -> np.ndarray:
        """
        Calculate reflectance uncertainty from radiance noise.

        σ_ρ = σ_L × |dρ/dL|

        Args:
            radiance_noise: Radiance noise (W/m²/sr/nm)
            reflectance: Surface reflectance
            rt_params: Atmospheric parameters

        Returns:
            Reflectance uncertainty
        """
        # dρ/dL = 1 / (dL/dρ)
        sensitivity = self.sensitivity_to_reflectance(reflectance, rt_params)

        return radiance_noise / sensitivity

    def two_stream_approximation(self,
                                  tau_total: np.ndarray,
                                  omega: np.ndarray,
                                  g: np.ndarray,
                                  solar_zenith_deg: float,
                                  surface_reflectance: float = 0.1
                                  ) -> Dict[str, np.ndarray]:
        """
        Two-stream approximation for plane-parallel atmosphere.

        A simplified RT solution that divides radiation into upward
        and downward streams. Fast but approximate.

        Args:
            tau_total: Total optical depth
            omega: Single scattering albedo
            g: Asymmetry parameter
            solar_zenith_deg: Solar zenith angle
            surface_reflectance: Surface reflectance

        Returns:
            Dictionary with fluxes and albedos
        """
        mu0 = np.cos(np.radians(solar_zenith_deg))

        # Delta-Eddington scaling
        tau_star = tau_total * (1 - omega * g**2)
        omega_star = omega * (1 - g**2) / (1 - omega * g**2)
        g_star = g / (1 + g)

        # Two-stream coefficients
        gamma1 = (7 - omega_star * (4 + 3*g_star)) / 4
        gamma2 = -(1 - omega_star * (4 - 3*g_star)) / 4
        gamma3 = (2 - 3*g_star*mu0) / 4
        gamma4 = 1 - gamma3

        # Transmission and reflection
        lambda_val = np.sqrt(gamma1**2 - gamma2**2)

        # Avoid numerical issues
        lambda_val = np.maximum(lambda_val, 1e-10)

        # Direct beam transmission
        T_direct = np.exp(-tau_star / mu0)

        # Diffuse transmission (approximate)
        T_diffuse = np.exp(-tau_star * 1.66)  # Diffusivity factor

        # Spherical albedo (approximate)
        A_sphere = omega_star * (1 - np.exp(-2*tau_star)) / (1 + 2*lambda_val*mu0)

        return {
            'T_direct': T_direct,
            'T_diffuse': T_diffuse,
            'spherical_albedo': np.clip(A_sphere, 0, 1),
            'tau_effective': tau_star
        }

    def explain_rt_equation(self) -> str:
        """Educational explanation of radiative transfer."""
        explanation = """
The Radiative Transfer Equation for Remote Sensing
==================================================

What We Measure:
    A sensor looking down at Earth measures radiance L (W/m²/sr/nm)
    that has traveled from the surface through the atmosphere.

The Journey of Light:

    SUN ────────────────────────────────┐
     │                                   │
     │ 1. Solar irradiance E₀           │
     │    travels to atmosphere         │
     ▼                                   │
    ┌─────────────────────────────────┐ │
    │      ATMOSPHERE                  │ │
    │                                  │ │
    │  2. Some scattered ────────► PATH RADIANCE
    │     toward sensor      L_path   │
    │                                  │
    │  3. Rest continues               │
    │     downward                     │
    │     ↓                            │
    │     T_down (transmittance)       │
    └─────────────────────────────────┘
                 │
                 ▼
    ═══════════════════════════════════
          SURFACE (reflectance ρ)
    ═══════════════════════════════════
                 │
                 │ 4. Surface reflects: L = ρ × E_down / π
                 ▼
    ┌─────────────────────────────────┐
    │      ATMOSPHERE                  │
    │                                  │
    │  5. Some absorbed/scattered     │
    │     ↓                           │
    │     T_up (transmittance)        │
    └─────────────────────────────────┘
                 │
                 ▼
         🛰️ SENSOR measures L_total

The At-Sensor Radiance Equation:

    L_sensor = L_path + T_up × (ρ × E_down / π)
               ──────   ─────────────────────────
               Atmos.     Surface contribution
               only       (what we want!)

The Inverse Problem (Atmospheric Correction):

    ρ = π × (L_sensor - L_path) / (T_up × E_down)

    We need to know L_path, T_up, and E_down to recover ρ.
    This is what atmospheric correction codes like 6S compute!

Why It's Hard:
    - L_path varies with view angle and atmosphere
    - T_up and T_down vary with wavelength (absorption bands)
    - E_down includes direct and diffuse components
    - Surface may not be Lambertian
    - Adjacency effect: neighboring pixels influence each other
"""
        return explanation

    def explain_signal_components(self, reflectance: float = 0.2) -> str:
        """
        Explain typical signal breakdown for a given reflectance.
        """
        explanation = f"""
At-Sensor Signal Breakdown (ρ = {reflectance})
==============================================

For a typical atmosphere (clear sky, mid-latitudes):

Visible (550 nm):
    Path radiance:     ~40% of signal  ← Atmosphere dominates!
    Surface signal:    ~60% of signal

    The atmosphere contributes significant "haze"
    that must be removed.

Near-IR (850 nm):
    Path radiance:     ~10% of signal
    Surface signal:    ~90% of signal  ← Surface dominates

    Much cleaner view of surface.

SWIR (2200 nm):
    Path radiance:     ~2% of signal
    Surface signal:    ~98% of signal

    Almost pure surface signal, BUT beware of
    absorption bands (H₂O, CO₂)!

Key Insight:
    Shorter wavelengths → More scattering → More path radiance
    Longer wavelengths → Less scattering → Cleaner surface signal

    This is why SWIR bands are great for mineral mapping,
    but visible bands need careful atmospheric correction.
"""
        return explanation
