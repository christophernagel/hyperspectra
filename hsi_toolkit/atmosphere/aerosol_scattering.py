"""
Aerosol (Mie) Scattering Module

Models scattering and absorption by atmospheric aerosols (particles).

Physics:
    Aerosols are suspended particles with sizes comparable to or larger
    than visible wavelengths. Their scattering is described by Mie theory,
    which is more complex than Rayleigh scattering:

    - Wavelength dependence: typically λ^(-α) where α ≈ 0.5-2.5
      (Ångström exponent depends on particle size distribution)
    - Strong forward scattering (asymmetric phase function)
    - Can both scatter AND absorb light (single scattering albedo ω₀ < 1)

    Aerosol Optical Depth (AOD or τ_a) is the key parameter:
        - Clean air: AOD ≈ 0.05
        - Typical: AOD ≈ 0.1-0.3
        - Hazy/polluted: AOD ≈ 0.5-1.0
        - Heavy pollution/dust: AOD > 1.0

Aerosol Types:
    - Maritime: Sea salt, larger particles, α ≈ 0.5-1.0
    - Continental: Dust, sulfates, smaller particles, α ≈ 1.0-1.5
    - Urban: Combustion particles, very small, α ≈ 1.5-2.5
    - Desert: Mineral dust, large particles, α ≈ 0-0.5

References:
    - Shettle & Fenn, 1979: Models for the aerosols of the lower atmosphere
    - d'Almeida et al., 1991: Atmospheric aerosols: global climatology
    - OPAC database (Optical Properties of Aerosols and Clouds)
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class AerosolType(Enum):
    """Standard aerosol model types."""
    MARITIME = "maritime"
    CONTINENTAL = "continental"
    URBAN = "urban"
    DESERT = "desert"
    RURAL = "rural"
    BIOMASS = "biomass_burning"


@dataclass
class AerosolModel:
    """
    Defines aerosol optical properties.

    Attributes:
        name: Model name
        angstrom_alpha: Ångström wavelength exponent
        angstrom_beta: Ångström turbidity coefficient
        ssa: Single scattering albedo (0-1, fraction scattered vs absorbed)
        asymmetry: Asymmetry parameter g (0=isotropic, 1=pure forward)
    """
    name: str
    angstrom_alpha: float    # Wavelength exponent (typically 0-2.5)
    angstrom_beta: float     # Turbidity at 1000nm
    ssa: float               # Single scattering albedo (typically 0.85-0.99)
    asymmetry: float         # Asymmetry parameter g (typically 0.5-0.8)


# Standard aerosol models (based on Shettle & Fenn, OPAC)
AEROSOL_MODELS = {
    AerosolType.MARITIME: AerosolModel(
        name="Maritime",
        angstrom_alpha=0.8,    # Larger particles -> weaker wavelength dependence
        angstrom_beta=0.05,
        ssa=0.99,              # Sea salt is highly scattering
        asymmetry=0.72
    ),
    AerosolType.CONTINENTAL: AerosolModel(
        name="Continental",
        angstrom_alpha=1.3,
        angstrom_beta=0.06,
        ssa=0.92,
        asymmetry=0.65
    ),
    AerosolType.URBAN: AerosolModel(
        name="Urban",
        angstrom_alpha=1.8,    # Small soot particles -> strong λ dependence
        angstrom_beta=0.08,
        ssa=0.85,              # Soot absorbs light
        asymmetry=0.60
    ),
    AerosolType.DESERT: AerosolModel(
        name="Desert Dust",
        angstrom_alpha=0.3,    # Very large particles
        angstrom_beta=0.10,
        ssa=0.90,
        asymmetry=0.75
    ),
    AerosolType.RURAL: AerosolModel(
        name="Rural/Background",
        angstrom_alpha=1.5,
        angstrom_beta=0.04,
        ssa=0.95,
        asymmetry=0.65
    ),
    AerosolType.BIOMASS: AerosolModel(
        name="Biomass Burning",
        angstrom_alpha=1.6,
        angstrom_beta=0.15,
        ssa=0.88,              # Smoke absorbs
        asymmetry=0.58
    ),
}


class AerosolScattering:
    """
    Aerosol scattering and extinction model.

    Uses the Ångström law to model wavelength-dependent extinction:
        τ(λ) = β * (λ/λ₀)^(-α)

    where:
        τ = aerosol optical depth
        β = turbidity coefficient (AOD at reference wavelength)
        α = Ångström exponent
        λ₀ = reference wavelength (typically 550nm or 1000nm)
    """

    REFERENCE_WAVELENGTH = 550.0  # nm

    def __init__(self, wavelengths: np.ndarray,
                 aerosol_type: AerosolType = AerosolType.CONTINENTAL):
        """
        Initialize aerosol model.

        Args:
            wavelengths: Wavelength array in nm
            aerosol_type: Type of aerosol model to use
        """
        self.wavelengths = np.asarray(wavelengths)
        self.set_aerosol_type(aerosol_type)

    def set_aerosol_type(self, aerosol_type: AerosolType):
        """Change the aerosol model type."""
        self.aerosol_type = aerosol_type
        self.model = AEROSOL_MODELS[aerosol_type]
        self._precompute_wavelength_factors()

    def _precompute_wavelength_factors(self):
        """Precompute wavelength-dependent factors."""
        # Ångström wavelength factor: (λ/λ₀)^(-α)
        self._wl_factor = (
            self.wavelengths / self.REFERENCE_WAVELENGTH
        ) ** (-self.model.angstrom_alpha)

    def optical_depth(self, aod_550: float = 0.1,
                      airmass: float = 1.0) -> np.ndarray:
        """
        Calculate aerosol optical depth at all wavelengths.

        Args:
            aod_550: Aerosol optical depth at 550nm
            airmass: Atmospheric path length factor

        Returns:
            AOD array at all wavelengths

        Note:
            AOD values reported (AERONET, MODIS) are usually at 550nm.
        """
        tau_550 = aod_550 * airmass
        return tau_550 * self._wl_factor

    def transmission(self, aod_550: float = 0.1,
                     airmass: float = 1.0) -> np.ndarray:
        """
        Calculate direct beam transmission.

        T = exp(-τ_a)
        """
        tau = self.optical_depth(aod_550, airmass)
        return np.exp(-tau)

    def scattering_optical_depth(self, aod_550: float = 0.1,
                                  airmass: float = 1.0) -> np.ndarray:
        """
        Calculate scattering component of optical depth.

        τ_scat = τ_total * ω₀

        where ω₀ is the single scattering albedo.
        """
        tau = self.optical_depth(aod_550, airmass)
        return tau * self.model.ssa

    def absorption_optical_depth(self, aod_550: float = 0.1,
                                  airmass: float = 1.0) -> np.ndarray:
        """
        Calculate absorption component of optical depth.

        τ_abs = τ_total * (1 - ω₀)
        """
        tau = self.optical_depth(aod_550, airmass)
        return tau * (1 - self.model.ssa)

    def phase_function(self, scattering_angle_deg: float) -> float:
        """
        Henyey-Greenstein phase function.

        P(θ) = (1 - g²) / (1 + g² - 2g*cos(θ))^(3/2)

        This is a common approximation for aerosol phase functions.
        The asymmetry parameter g controls the shape:
            g = 0: isotropic scattering
            g > 0: forward scattering
            g < 0: backward scattering

        Typical aerosols have g ≈ 0.6-0.8 (strong forward scattering).

        Args:
            scattering_angle_deg: Angle between incident and scattered directions

        Returns:
            Phase function value
        """
        g = self.model.asymmetry
        theta_rad = np.radians(scattering_angle_deg)
        cos_theta = np.cos(theta_rad)

        numerator = 1 - g ** 2
        denominator = (1 + g ** 2 - 2 * g * cos_theta) ** 1.5

        return numerator / denominator

    def path_radiance(self, solar_irradiance: np.ndarray,
                      solar_zenith_deg: float,
                      view_zenith_deg: float,
                      relative_azimuth_deg: float = 0.0,
                      aod_550: float = 0.1,
                      surface_albedo: float = 0.0) -> np.ndarray:
        """
        Estimate aerosol path radiance.

        This is light scattered by aerosols into the sensor view
        without touching the surface.

        Args:
            solar_irradiance: Solar irradiance spectrum
            solar_zenith_deg: Solar zenith angle
            view_zenith_deg: View zenith angle
            relative_azimuth_deg: Relative azimuth between sun and view
            aod_550: Aerosol optical depth at 550nm
            surface_albedo: Mean surface albedo

        Returns:
            Path radiance spectrum
        """
        mu_0 = np.cos(np.radians(solar_zenith_deg))
        mu = np.cos(np.radians(view_zenith_deg))

        if mu_0 <= 0:
            return np.zeros_like(self.wavelengths)

        # Scattering angle from geometry
        cos_scatter = (
            -mu_0 * mu +
            np.sin(np.radians(solar_zenith_deg)) *
            np.sin(np.radians(view_zenith_deg)) *
            np.cos(np.radians(relative_azimuth_deg))
        )
        scatter_angle = np.degrees(np.arccos(np.clip(cos_scatter, -1, 1)))

        # Phase function
        phase = self.phase_function(scatter_angle)

        # Optical depth
        tau = self.optical_depth(aod_550, 1.0)

        # Single scattering approximation
        # L_path ≈ (E0 * ω₀ * P(θ) / 4π) * [1 - exp(-τ/μ₀ - τ/μ)] / (1/μ₀ + 1/μ)

        airmass_total = 1/mu_0 + 1/max(mu, 0.1)
        tau_path = tau * airmass_total

        scatter_term = (1 - np.exp(-tau_path)) / airmass_total
        L_path = (solar_irradiance * self.model.ssa * phase / (4 * np.pi)) * scatter_term

        # Multiple scattering correction
        ms_factor = 1 + 0.3 * surface_albedo + 0.1 * tau

        return L_path * ms_factor

    @staticmethod
    def estimate_aod_from_dos(dark_radiance: float,
                               expected_radiance: float,
                               solar_irradiance: float,
                               solar_zenith_deg: float) -> float:
        """
        Estimate AOD using Dark Object Subtraction.

        The idea: dark objects (deep water, shadows) should have
        near-zero reflectance. Any radiance we measure is path radiance,
        which scales with AOD.

        Args:
            dark_radiance: Measured radiance from dark target
            expected_radiance: Expected radiance if AOD=0
            solar_irradiance: Solar irradiance at this wavelength
            solar_zenith_deg: Solar zenith angle

        Returns:
            Estimated AOD at this wavelength
        """
        # Path radiance excess
        L_excess = dark_radiance - expected_radiance

        if L_excess <= 0:
            return 0.0

        # Rough inversion (simplified)
        mu_0 = np.cos(np.radians(solar_zenith_deg))
        aod = L_excess * 4 * np.pi / (solar_irradiance * mu_0)

        return np.clip(aod, 0, 2.0)

    def get_angstrom_parameters(self) -> Dict[str, float]:
        """Return current Ångström parameters for display."""
        return {
            'alpha': self.model.angstrom_alpha,
            'beta': self.model.angstrom_beta,
            'ssa': self.model.ssa,
            'asymmetry': self.model.asymmetry,
            'name': self.model.name
        }

    def explain_aerosol_type(self) -> str:
        """Educational explanation of current aerosol model."""
        m = self.model

        explanation = f"""
Aerosol Model: {m.name}
{'=' * 40}

Ångström Exponent (α): {m.angstrom_alpha:.2f}
  - Controls wavelength dependence: τ(λ) ∝ λ^(-α)
  - Small α (< 1): Large particles (dust, sea salt)
  - Large α (> 1.5): Small particles (smoke, pollution)
  - Your model: {'Larger particles' if m.angstrom_alpha < 1.0 else 'Smaller particles'}

Single Scattering Albedo (ω₀): {m.ssa:.2f}
  - Fraction of light scattered (vs absorbed)
  - 1.0 = pure scattering (white clouds)
  - 0.8 = significant absorption (soot)
  - Your model: {m.ssa*100:.0f}% scattered, {(1-m.ssa)*100:.0f}% absorbed

Asymmetry Parameter (g): {m.asymmetry:.2f}
  - 0 = equal forward/backward scattering
  - 1 = pure forward scattering
  - Your model: {'Strong forward scattering' if m.asymmetry > 0.6 else 'Moderate scattering'}

Typical Sources:
"""
        sources = {
            AerosolType.MARITIME: "  - Sea spray and breaking waves\n  - Largest over open ocean",
            AerosolType.CONTINENTAL: "  - Soil dust, vegetation emissions\n  - Background over land",
            AerosolType.URBAN: "  - Vehicle exhaust, industrial emissions\n  - Strong absorber (black carbon)",
            AerosolType.DESERT: "  - Windblown mineral dust\n  - Can transport thousands of km",
            AerosolType.BIOMASS: "  - Forest fires, agricultural burning\n  - Seasonal patterns",
            AerosolType.RURAL: "  - Mixed natural and anthropogenic\n  - Typical background",
        }
        explanation += sources.get(self.aerosol_type, "")

        return explanation


def compare_aerosol_models(wavelengths: np.ndarray,
                           aod_550: float = 0.2) -> Dict[str, np.ndarray]:
    """
    Compare optical depth spectra for all aerosol models.

    Useful for visualization of how different aerosol types
    affect different wavelengths.

    Args:
        wavelengths: Wavelength array in nm
        aod_550: AOD at 550nm (same for all models)

    Returns:
        Dictionary of model_name: optical_depth_array
    """
    results = {}

    for atype in AerosolType:
        model = AerosolScattering(wavelengths, atype)
        results[atype.value] = model.optical_depth(aod_550)

    return results
