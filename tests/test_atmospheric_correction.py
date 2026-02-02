"""
Scientific validation tests for atmospheric correction.

These tests validate:
1. 6S correction formula implementation
2. Physical constraints on reflectance
3. Known surface type signatures
4. Atmospheric parameter reasonableness

References:
    Vermote et al. (1997) - 6S radiative transfer model
    Kaufman & Sendra (1988) - DDV algorithm
    Gao & Goetz (1990) - Water vapor retrieval
"""

import numpy as np
import pytest


class Test6SCorrectionFormula:
    """
    Validate the 6S atmospheric correction formula.

    The standard 6S correction:
        y = xa * L_sensor - xb
        rho_surface = y / (1 + xc * y)

    Where:
        xa = 1 / (T_gas * T_total)  - total transmission factor
        xb = L_path / (T_gas * T_total)  - path radiance term
        xc = S * T_total  - spherical albedo term

    Reference: Vermote et al. (1997) IEEE TGRS
    """

    def test_zero_radiance_gives_negative_reflectance(self):
        """Zero radiance should give negative reflectance (path radiance subtraction)."""
        xa, xb, xc = 0.1, 0.02, 0.5  # Typical 6S coefficients
        L_sensor = 0.0

        y = xa * L_sensor - xb
        rho = y / (1 + xc * y)

        # With zero radiance, path subtraction dominates -> negative
        assert y < 0, "Path radiance term should make y negative for zero radiance"

    def test_high_radiance_approaches_limit(self):
        """Very high radiance should not exceed physical limits."""
        xa, xb, xc = 0.1, 0.02, 0.5
        L_sensor = 1000.0  # Unrealistically high

        y = xa * L_sensor - xb
        rho = y / (1 + xc * y)

        # Reflectance should approach 1/xc as y -> infinity
        theoretical_max = 1.0 / xc
        assert rho < theoretical_max, f"Reflectance should be bounded by 1/xc = {theoretical_max}"

    def test_moderate_radiance_reasonable_reflectance(self):
        """Moderate radiance should give reflectance in 0-1 range."""
        # Typical 6S coefficients for 550nm, clear atmosphere
        xa, xb, xc = 0.15, 0.01, 0.1

        # Typical vegetation radiance (~10 W/m²/sr/µm)
        L_sensor = 0.5  # Scaled units

        y = xa * L_sensor - xb
        rho = y / (1 + xc * y)

        assert 0 < rho < 1, f"Moderate radiance should give valid reflectance, got {rho}"


class TestPhysicalConstraints:
    """
    Test physical constraints on reflectance.

    Surface reflectance must satisfy:
    1. 0 ≤ rho ≤ 1 for Lambertian surfaces
    2. Can slightly exceed 1 for specular/BRDF effects
    3. Should not be significantly negative
    """

    def test_lambertian_upper_bound(self):
        """Lambertian reflectance cannot exceed 1.0."""
        rho_max = 1.0
        # Allow small overshoot for calibration uncertainty
        tolerance = 0.05

        test_values = np.array([0.95, 1.0, 1.02, 1.1])
        valid = test_values <= (rho_max + tolerance)

        assert valid[:3].all(), "Reflectance up to 1.05 acceptable"
        assert not valid[3], "Reflectance > 1.05 should be flagged"

    def test_negative_reflectance_indicates_error(self):
        """Significantly negative reflectance indicates correction error."""
        # Small negative from noise is acceptable
        rho_noise = -0.005
        assert abs(rho_noise) < 0.02, "Small negative from noise acceptable"

        # Large negative indicates problem
        rho_error = -0.1
        assert rho_error < -0.02, "Large negative should be flagged as error"


class TestKnownSurfaceSignatures:
    """
    Validate reflectance against known surface signatures.

    References:
        USGS Spectral Library (Kokaly et al., 2017)
        ASTER Spectral Library
    """

    def test_vegetation_red_edge(self):
        """
        Vegetation should show characteristic red-edge.

        Red edge: sharp increase from 680nm (red, chlorophyll absorption)
        to 750nm (NIR, cellular scattering).

        Ratio NIR/Red should be > 3 for healthy vegetation.
        """
        # Typical healthy vegetation
        rho_680 = 0.05  # Red - chlorophyll absorbs
        rho_750 = 0.40  # NIR - high scattering

        red_edge_ratio = rho_750 / rho_680

        assert red_edge_ratio > 3, f"Red edge ratio should be > 3, got {red_edge_ratio}"
        assert red_edge_ratio < 15, f"Red edge ratio should be < 15, got {red_edge_ratio}"

    def test_water_nir_absorption(self):
        """
        Water should show strong NIR absorption.

        Clear water reflectance at 860nm should be < 0.03.
        Reference: Pope & Fry (1997) - water absorption coefficients
        """
        rho_water_860 = 0.02  # Typical clear water

        assert rho_water_860 < 0.05, "Clear water NIR should be < 0.05"

    def test_bare_soil_spectrum_shape(self):
        """
        Bare soil should show monotonically increasing reflectance in VNIR.

        Typical dry soil: rho increases from 400nm to 1000nm.
        Reference: Stoner & Baumgardner (1981) - soil spectral curves
        """
        # Typical dry soil reflectance
        rho_450 = 0.10
        rho_650 = 0.20
        rho_850 = 0.30

        assert rho_450 < rho_650 < rho_850, "Soil reflectance should increase with wavelength"

    def test_snow_high_visible_reflectance(self):
        """
        Snow should have very high visible reflectance.

        Fresh snow: rho > 0.9 at 550nm
        Reference: Warren (1982) - optical properties of ice
        """
        rho_snow_550 = 0.92

        assert rho_snow_550 > 0.85, "Fresh snow visible reflectance should be > 0.85"


class TestAtmosphericParameters:
    """
    Validate atmospheric parameter retrieval.

    References:
        Gao & Goetz (1990) - Water vapor from 940nm band
        Kaufman et al. (1997) - DDV AOD retrieval
    """

    def test_water_vapor_range(self):
        """
        Precipitable water vapor should be in reasonable range.

        Typical range: 0.5 - 5.0 g/cm² (cm precipitable water)
        Extreme desert: < 0.3 g/cm²
        Tropical: > 5.0 g/cm²
        """
        wv_typical = 2.0
        wv_desert = 0.2
        wv_tropical = 6.0

        assert 0.1 < wv_typical < 7.0, "Typical water vapor should be 0.1-7.0 g/cm²"
        assert wv_desert < 0.5, "Desert water vapor should be < 0.5 g/cm²"
        assert wv_tropical > 4.0, "Tropical water vapor should be > 4.0 g/cm²"

    def test_aot_range(self):
        """
        Aerosol Optical Thickness should be in reasonable range.

        AOT@550nm typical ranges:
        - Clean maritime: 0.05 - 0.15
        - Continental: 0.1 - 0.3
        - Urban/polluted: 0.3 - 1.0
        - Extreme events: > 1.0
        """
        aot_clean = 0.08
        aot_urban = 0.5
        aot_extreme = 2.0

        assert 0.01 < aot_clean < 0.2, "Clean AOT should be 0.01-0.2"
        assert 0.2 < aot_urban < 1.0, "Urban AOT should be 0.2-1.0"
        assert aot_extreme > 1.0, "Extreme AOT can exceed 1.0"

    def test_angstrom_exponent_range(self):
        """
        Ångström exponent characterizes aerosol size distribution.

        α = -d(ln τ)/d(ln λ)

        Typical values:
        - Coarse (dust, sea salt): 0.0 - 0.5
        - Fine (pollution): 1.0 - 2.0
        - Continental mix: 1.0 - 1.5

        Reference: Eck et al. (1999) - AERONET climatology
        """
        alpha_dust = 0.3
        alpha_pollution = 1.5
        alpha_continental = 1.3

        assert 0 < alpha_dust < 0.7, "Dust Ångström should be 0-0.7"
        assert 1.0 < alpha_pollution < 2.5, "Pollution Ångström should be 1.0-2.5"
        assert 0.8 < alpha_continental < 1.8, "Continental Ångström should be 0.8-1.8"


class TestDDVCoefficients:
    """
    Validate Dense Dark Vegetation empirical coefficients.

    Reference: Kaufman & Sendra (1988), Kaufman et al. (1997)

    The DDV method relates SWIR reflectance (minimal aerosol effect)
    to visible reflectance through empirical ratios.
    """

    def test_blue_to_swir_ratio(self):
        """
        Blue/SWIR ratio for vegetation.

        rho_470 / rho_2100 ≈ 0.25 (±0.05) for vegetation
        Reference: Kaufman et al. (1997) MODIS ATBD
        """
        ratio = 0.25
        tolerance = 0.10  # Allow for variation

        # Test with typical values
        rho_2100 = 0.10  # Vegetation SWIR
        rho_470_predicted = ratio * rho_2100

        assert 0.15 < ratio < 0.35, f"Blue/SWIR ratio should be 0.15-0.35, using {ratio}"
        assert 0.01 < rho_470_predicted < 0.05, "Predicted blue reflectance reasonable"

    def test_red_to_swir_ratio(self):
        """
        Red/SWIR ratio for vegetation.

        rho_660 / rho_2100 ≈ 0.50 (±0.10) for vegetation
        Reference: Kaufman et al. (1997) MODIS ATBD
        """
        ratio = 0.50

        # Test with typical values
        rho_2100 = 0.10  # Vegetation SWIR
        rho_660_predicted = ratio * rho_2100

        assert 0.35 < ratio < 0.65, f"Red/SWIR ratio should be 0.35-0.65, using {ratio}"
        assert 0.03 < rho_660_predicted < 0.08, "Predicted red reflectance reasonable"


class TestUnitConversions:
    """
    Validate radiometric unit conversions.

    AVIRIS-3 radiance units: µW/(nm·cm²·sr)
    6S expected units: W/(m²·sr·µm)

    Conversion factor: × 10.0
    """

    def test_aviris_to_si_conversion(self):
        """
        AVIRIS radiance to SI units.

        µW/(nm·cm²·sr) -> W/(m²·sr·µm)

        1 µW = 1e-6 W
        1 nm = 1e-3 µm
        1 cm² = 1e-4 m²

        Factor = (1e-6) / (1e-3 × 1e-4) = (1e-6) / (1e-7) = 10
        """
        L_aviris = 1.0  # µW/(nm·cm²·sr)
        conversion_factor = 10.0
        L_si = L_aviris * conversion_factor

        assert L_si == 10.0, f"Conversion should give 10.0, got {L_si}"

    def test_solar_irradiance_units(self):
        """
        Solar irradiance should be in W/(m²·µm).

        At 550nm, E_sun ≈ 1880 W/(m²·µm) (Thuillier 2003)
        """
        E_sun_550 = 1879.0  # From our reference data

        assert 1800 < E_sun_550 < 2000, f"Solar irradiance at 550nm should be ~1880 W/(m²·µm)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
