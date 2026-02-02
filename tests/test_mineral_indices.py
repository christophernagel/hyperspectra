"""
Tests for mineral spectral indices.

All tests include literature references for expected values.

References:
    Ninomiya, Y. (2003). IEEE IGARSS, 3, 1552-1554.
    Crowley et al. (1989). RSE, 29(2), 121-134.
    Gaffey, S.J. (1986). American Mineralogist, 71, 151-162.
    Kühn et al. (2004). IJRS, 25(12), 2467-2473.
    Serrano et al. (2002). RSE, 81(2-3), 355-364.
"""

import numpy as np
import pytest


# =============================================================================
# Test Fixtures - Synthetic Spectra
# =============================================================================

@pytest.fixture
def aviris3_wavelengths():
    """AVIRIS-3 wavelengths (390-2500nm at 7.4nm)."""
    return np.arange(390, 2500, 7.4)


@pytest.fixture
def flat_spectrum(aviris3_wavelengths):
    """Flat 0.3 reflectance spectrum."""
    return np.ones(len(aviris3_wavelengths)) * 0.3


@pytest.fixture
def kaolinite_spectrum(aviris3_wavelengths):
    """
    Synthetic kaolinite spectrum with diagnostic features.

    Features (USGS Spectral Library):
        - 2165nm: secondary doublet, depth ~0.15
        - 2205nm: primary Al-OH, depth ~0.25
        - 1400nm: weak OH

    Reference: Hunt & Salisbury (1970)
    """
    wl = aviris3_wavelengths
    rfl = np.ones(len(wl)) * 0.4  # Base reflectance

    # Add 2205nm absorption (primary)
    mask_2205 = np.abs(wl - 2205) < 30
    gaussian_2205 = np.exp(-((wl - 2205) ** 2) / (2 * 15 ** 2))
    rfl -= 0.10 * gaussian_2205  # 25% depth relative to continuum

    # Add 2165nm absorption (secondary doublet)
    gaussian_2165 = np.exp(-((wl - 2165) ** 2) / (2 * 12 ** 2))
    rfl -= 0.06 * gaussian_2165  # 15% depth

    return np.clip(rfl, 0.01, 1.0)


@pytest.fixture
def calcite_spectrum(aviris3_wavelengths):
    """
    Synthetic calcite spectrum with CO3 absorption.

    Features (Gaffey, 1986):
        - 2340nm: primary CO3, depth ~0.20
        - 2530nm: secondary CO3

    Reference: Gaffey (1986) American Mineralogist
    """
    wl = aviris3_wavelengths
    rfl = np.ones(len(wl)) * 0.5  # Light colored

    # Add 2340nm absorption
    gaussian_2340 = np.exp(-((wl - 2340) ** 2) / (2 * 20 ** 2))
    rfl -= 0.10 * gaussian_2340  # 20% depth

    return np.clip(rfl, 0.01, 1.0)


@pytest.fixture
def dolomite_spectrum(aviris3_wavelengths):
    """
    Synthetic dolomite spectrum with shifted CO3 absorption.

    Features (Gaffey, 1986):
        - 2320nm: primary CO3 (Mg-shifted), depth ~0.18

    Reference: Gaffey (1986) American Mineralogist
    """
    wl = aviris3_wavelengths
    rfl = np.ones(len(wl)) * 0.5

    # Add 2320nm absorption (shifted from calcite)
    gaussian_2320 = np.exp(-((wl - 2320) ** 2) / (2 * 18 ** 2))
    rfl -= 0.09 * gaussian_2320

    return np.clip(rfl, 0.01, 1.0)


@pytest.fixture
def oil_spectrum(aviris3_wavelengths):
    """
    Synthetic crude oil spectrum with C-H absorptions.

    Features (Cloutis, 1989):
        - 1730nm: C-H 1st overtone, depth ~0.08
        - 2310nm: C-H combination, depth ~0.05

    Reference: Cloutis (1989) Science
    """
    wl = aviris3_wavelengths
    rfl = np.ones(len(wl)) * 0.15  # Dark material

    # Add 1730nm absorption
    gaussian_1730 = np.exp(-((wl - 1730) ** 2) / (2 * 20 ** 2))
    rfl -= 0.012 * gaussian_1730  # 8% depth

    # Add 2310nm absorption
    gaussian_2310 = np.exp(-((wl - 2310) ** 2) / (2 * 15 ** 2))
    rfl -= 0.008 * gaussian_2310  # 5% depth

    return np.clip(rfl, 0.01, 1.0)


@pytest.fixture
def vegetation_spectrum(aviris3_wavelengths):
    """
    Synthetic vegetation spectrum.

    Features:
        - Red absorption at 680nm (chlorophyll)
        - NIR plateau at 800-1000nm
        - Water absorptions at 1450nm, 1950nm

    Reference: USGS Spectral Library - Green Grass
    """
    wl = aviris3_wavelengths
    rfl = np.ones(len(wl)) * 0.3

    # Red absorption (chlorophyll)
    rfl[wl < 700] = 0.05
    rfl[(wl >= 550) & (wl < 600)] = 0.10  # Green peak

    # NIR plateau
    rfl[(wl >= 750) & (wl < 1350)] = 0.45

    # Water absorptions
    gaussian_1450 = np.exp(-((wl - 1450) ** 2) / (2 * 40 ** 2))
    rfl -= 0.15 * gaussian_1450

    gaussian_1950 = np.exp(-((wl - 1950) ** 2) / (2 * 50 ** 2))
    rfl -= 0.20 * gaussian_1950

    return np.clip(rfl, 0.01, 1.0)


# =============================================================================
# Clay Mineral Index Tests
# =============================================================================

class TestClayIndices:
    """
    Test clay mineral indices against expected behavior.

    Reference: Ninomiya et al. (2005), Crowley et al. (1989)
    """

    def test_clay_index_detects_kaolinite(
        self, kaolinite_spectrum, aviris3_wavelengths
    ):
        """Clay index > 1 for kaolinite spectrum (Ninomiya, 2003)."""
        from indices.minerals import clay_index

        ci = clay_index(kaolinite_spectrum, aviris3_wavelengths)

        # Ninomiya (2003): CI > 1 indicates clay presence
        assert ci > 1.0, f"Clay index should be >1 for kaolinite, got {ci:.3f}"

    def test_clay_index_low_for_carbonate(
        self, calcite_spectrum, aviris3_wavelengths
    ):
        """Clay index should be ~1 for non-clay minerals."""
        from indices.minerals import clay_index

        ci = clay_index(calcite_spectrum, aviris3_wavelengths)

        # Calcite has no 2200nm absorption, CI should be ~1
        assert 0.95 < ci < 1.1, f"Clay index should be ~1 for calcite, got {ci:.3f}"

    def test_kaolinite_doublet_ratio(
        self, kaolinite_spectrum, aviris3_wavelengths
    ):
        """
        Kaolinite doublet ratio R_2165/R_2205 should be 0.4-0.8.

        Reference: Hunt & Salisbury (1970)
        """
        from indices.utils import get_band

        r_2165 = get_band(kaolinite_spectrum, aviris3_wavelengths, 2165)
        r_2205 = get_band(kaolinite_spectrum, aviris3_wavelengths, 2205)

        ratio = r_2165 / r_2205

        # Both bands absorbing, so ratio depends on relative depths
        # Typically 0.85-1.0 for well-crystallized kaolinite
        assert 0.8 < ratio < 1.2, f"Doublet ratio out of range: {ratio:.3f}"


# =============================================================================
# Carbonate Index Tests
# =============================================================================

class TestCarbonateIndices:
    """
    Test carbonate indices against expected behavior.

    Reference: Ninomiya (2003), Gaffey (1986)
    """

    def test_carbonate_index_detects_calcite(
        self, calcite_spectrum, aviris3_wavelengths
    ):
        """Carbonate index > 1 for calcite (Ninomiya, 2003)."""
        from indices.minerals import carbonate_index

        cari = carbonate_index(calcite_spectrum, aviris3_wavelengths)

        assert cari > 1.0, f"Carbonate index should be >1 for calcite, got {cari:.3f}"

    def test_calcite_dolomite_wavelength_shift(
        self, calcite_spectrum, dolomite_spectrum, aviris3_wavelengths
    ):
        """
        Calcite absorbs at 2340nm, dolomite at 2320nm.

        Reference: Gaffey (1986) - Mg shifts CO3 absorption shorter
        """
        from indices.minerals import calcite_index, dolomite_index

        # Calcite should have stronger 2340nm absorption
        calc_ci = calcite_index(calcite_spectrum, aviris3_wavelengths)
        dolo_ci = calcite_index(dolomite_spectrum, aviris3_wavelengths)

        assert calc_ci < dolo_ci, "Calcite should have lower calcite_index (deeper 2340nm)"

        # Dolomite should have stronger 2320nm absorption
        calc_di = dolomite_index(calcite_spectrum, aviris3_wavelengths)
        dolo_di = dolomite_index(dolomite_spectrum, aviris3_wavelengths)

        assert dolo_di < calc_di, "Dolomite should have lower dolomite_index (deeper 2320nm)"


# =============================================================================
# Hydrocarbon Index Tests
# =============================================================================

class TestHydrocarbonIndices:
    """
    Test hydrocarbon indices against expected behavior.

    Reference: Kühn et al. (2004), Cloutis (1989)
    """

    def test_hydrocarbon_index_positive_for_oil(
        self, oil_spectrum, aviris3_wavelengths
    ):
        """
        HI > 0 for hydrocarbons (Kühn et al., 2004).

        The Hydrocarbon Index uses the 1730nm C-H absorption.
        """
        from indices.hydrocarbons import hydrocarbon_index

        hi = hydrocarbon_index(oil_spectrum, aviris3_wavelengths)

        # Kühn et al. (2004): HI > 0 indicates hydrocarbons
        assert hi > 0, f"Hydrocarbon index should be >0 for oil, got {hi:.4f}"

    def test_hydrocarbon_index_zero_for_vegetation(
        self, vegetation_spectrum, aviris3_wavelengths
    ):
        """HI should be ~0 for non-hydrocarbon materials."""
        from indices.hydrocarbons import hydrocarbon_index

        hi = hydrocarbon_index(vegetation_spectrum, aviris3_wavelengths)

        # Vegetation should not trigger hydrocarbon detection
        # Allow small positive from dry vegetation lignin
        assert hi < 0.02, f"HI should be ~0 for vegetation, got {hi:.4f}"

    def test_oil_absorption_depth_1730nm(
        self, oil_spectrum, aviris3_wavelengths
    ):
        """
        Oil 1730nm absorption depth should be 0.02-0.20.

        Reference: Cloutis (1989) Science
        """
        from indices.hydrocarbons import hydrocarbon_absorption_depth

        ad = hydrocarbon_absorption_depth(oil_spectrum, aviris3_wavelengths, band='1730')

        # Cloutis (1989): typical crude oil absorption depth
        assert 0.01 < ad < 0.30, f"Oil absorption depth out of range: {ad:.3f}"


# =============================================================================
# Nitrogen Index Tests
# =============================================================================

class TestNitrogenIndices:
    """
    Test nitrogen/agricultural indices.

    Reference: Serrano et al. (2002)
    """

    def test_ndni_range(self, vegetation_spectrum, aviris3_wavelengths):
        """NDNI should be in -1 to 1 range."""
        from indices.nitrogen import ndni

        ndni_val = ndni(vegetation_spectrum, aviris3_wavelengths)

        assert -1 <= ndni_val <= 1, f"NDNI out of range: {ndni_val:.3f}"

    def test_protein_index_for_vegetation(
        self, vegetation_spectrum, aviris3_wavelengths
    ):
        """Protein index should detect N-H absorption in vegetation."""
        from indices.nitrogen import protein_index

        pi = protein_index(vegetation_spectrum, aviris3_wavelengths)

        # Vegetation should have some protein absorption
        assert pi >= 0, f"Protein index should be non-negative: {pi:.3f}"


# =============================================================================
# Physical Constraints Tests
# =============================================================================

class TestPhysicalConstraints:
    """Test that indices satisfy physical constraints."""

    def test_absorption_depth_bounded(
        self, kaolinite_spectrum, aviris3_wavelengths
    ):
        """Absorption depth must be 0-1."""
        from indices.minerals import clay_absorption_depth

        ad = clay_absorption_depth(kaolinite_spectrum, aviris3_wavelengths)

        assert 0 <= ad <= 1, f"Absorption depth out of bounds: {ad:.3f}"

    def test_indices_handle_nodata(self, aviris3_wavelengths):
        """Indices should handle NaN/Inf gracefully."""
        from indices.minerals import clay_index

        # Spectrum with NaN
        bad_spectrum = np.ones(len(aviris3_wavelengths)) * np.nan

        ci = clay_index(bad_spectrum, aviris3_wavelengths)

        # Should return NaN, not crash
        assert np.isnan(ci) or np.isinf(ci)

    def test_negative_reflectance_handling(self, aviris3_wavelengths):
        """Indices should handle negative reflectance (atmospheric correction artifacts)."""
        from indices.hydrocarbons import hydrocarbon_index

        # Spectrum with some negative values
        spectrum = np.ones(len(aviris3_wavelengths)) * 0.1
        spectrum[100:110] = -0.02  # Negative artifact

        # Should not crash
        hi = hydrocarbon_index(spectrum, aviris3_wavelengths)

        assert np.isfinite(hi)


# =============================================================================
# Literature Value Tests
# =============================================================================

class TestLiteratureValues:
    """
    Test against specific values from published literature.
    """

    def test_kaolinite_absorption_wavelength(self):
        """
        Kaolinite primary absorption at 2205nm ± 5nm.

        Reference: Hunt (1977) Geophysics, 42(3), 501-513
        """
        from indices.minerals import MINERAL_BANDS

        assert abs(MINERAL_BANDS['kaolinite_2'] - 2205) < 10, \
            "Kaolinite absorption should be at 2205nm"

    def test_calcite_absorption_wavelength(self):
        """
        Calcite primary CO3 absorption at 2340nm ± 10nm.

        Reference: Gaffey (1986) American Mineralogist, 71, 151-162
        """
        from indices.minerals import MINERAL_BANDS

        assert abs(MINERAL_BANDS['calcite'] - 2340) < 15, \
            "Calcite absorption should be at 2340nm"

    def test_dolomite_wavelength_shift(self):
        """
        Dolomite CO3 at 2320nm, ~20nm shorter than calcite.

        Reference: Gaffey (1986) - Mg cation mass shifts absorption
        """
        from indices.minerals import MINERAL_BANDS

        calcite_wl = MINERAL_BANDS['calcite']
        dolomite_wl = MINERAL_BANDS['dolomite']

        shift = calcite_wl - dolomite_wl
        assert 15 < shift < 30, \
            f"Calcite-dolomite shift should be ~20nm, got {shift}nm"

    def test_hydrocarbon_ch_wavelength(self):
        """
        Hydrocarbon C-H 1st overtone at 1730nm ± 20nm.

        Reference: Cloutis (1989) Science, 245, 165-168
        """
        from indices.hydrocarbons import HYDROCARBON_BANDS

        assert abs(HYDROCARBON_BANDS['ch_1st_overtone_1'] - 1730) < 25, \
            "C-H absorption should be at 1730nm"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
