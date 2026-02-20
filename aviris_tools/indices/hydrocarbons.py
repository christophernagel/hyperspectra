"""
Hydrocarbon and petroleum spectral indices.

Detects oil, natural gas seeps, and hydrocarbon-contaminated soils
using C-H stretching absorptions in the SWIR.

Key absorption features:
    - 1200 nm: C-H stretch 2nd overtone
    - 1730 nm: C-H stretch 1st overtone (primary diagnostic)
    - 2310 nm: C-H stretch combination band
    - 2350 nm: C-H stretch combination band

References:
    Kühn, F., Oppermann, K., & Hörig, B. (2004). Hydrocarbon Index - An
        algorithm for hyperspectral detection of hydrocarbons. International
        Journal of Remote Sensing, 25(12), 2467-2473.

    Cloutis, E.A. (1989). Spectral reflectance properties of hydrocarbons:
        Remote-sensing implications. Science, 245(4914), 165-168.

    Lammoglia, T., & de Souza Filho, C.R. (2011). Spectroscopic characterization
        of oils yielded from Brazilian offshore basins: Potential applications
        of remote sensing. Remote Sensing of Environment, 115(10), 2525-2535.

    Thorpe, A.K., et al. (2014). High resolution mapping of methane emissions
        from marine and terrestrial sources using a Cluster-Tuned Matched Filter
        technique and imaging spectrometry. RSE, 134, 305-318.
"""

import numpy as np
from .utils import get_band, absorption_depth, interpolate_band


# =============================================================================
# Key Absorption Wavelengths (nm)
# =============================================================================

HYDROCARBON_BANDS = {
    # C-H stretch features
    'ch_2nd_overtone': 1200,      # Weak
    'ch_1st_overtone_1': 1720,    # Strong, CH2
    'ch_1st_overtone_2': 1760,    # Strong, CH3
    'ch_combination_1': 2310,      # Strong
    'ch_combination_2': 2350,      # Strong

    # Methane-specific
    'methane_1': 1670,             # CH4 absorption
    'methane_2': 2300,             # CH4 absorption

    # Reference/continuum points
    'swir_continuum_1': 1660,
    'swir_continuum_2': 1780,
    'swir_continuum_3': 2260,
    'swir_continuum_4': 2380,
}


# =============================================================================
# Hydrocarbon Indices
# =============================================================================

def hydrocarbon_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Hydrocarbon Index (HI) using 1730nm C-H absorption.

    The primary index for detecting hydrocarbon-bearing materials including
    oil slicks, contaminated soils, and natural seeps.

    Formula (Kühn et al., 2004):
        HI = (R_1720 + R_1760) / (2 × R_1740) - 1

    If HI > 0: hydrocarbon present
    If HI ≈ 0: no hydrocarbon

    Reference:
        Kühn, F., Oppermann, K., & Hörig, B. (2004). Hydrocarbon Index - An
        algorithm for hyperspectral detection of hydrocarbons. IJRS, 25(12),
        2467-2473.

    Parameters:
        rfl: Reflectance array (y, x, bands)
        wavelengths: Wavelength array in nm

    Returns:
        Hydrocarbon index (positive values indicate hydrocarbons)
    """
    r_1720 = get_band(rfl, wavelengths, 1720)
    r_1740 = get_band(rfl, wavelengths, 1740)
    r_1760 = get_band(rfl, wavelengths, 1760)

    eps = 1e-10
    hi = ((r_1720 + r_1760) / (2 * r_1740 + eps)) - 1.0

    return hi


def oil_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Oil Absorption Index combining 1730nm and 2310nm features.

    More robust than single-feature indices for oil detection.
    Uses both C-H 1st overtone and combination bands.

    Formula:
        OI = HI_1730 × AD_2310

    Where:
        HI_1730 = hydrocarbon index at 1730nm
        AD_2310 = absorption depth at 2310nm

    Reference:
        Lammoglia, T., & de Souza Filho, C.R. (2011). Spectroscopic
        characterization of oils. RSE, 115(10), 2525-2535.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Oil index (higher = stronger oil signature)
    """
    hi = hydrocarbon_index(rfl, wavelengths)
    ad_2310 = hydrocarbon_absorption_depth(rfl, wavelengths, band='2310')

    # Combine indices (both should be positive for oil)
    oi = np.maximum(hi, 0) * ad_2310

    return oi


def hydrocarbon_absorption_depth(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    band: str = '1730',
) -> np.ndarray:
    """
    Hydrocarbon absorption depth using continuum removal.

    More quantitative than ratio indices.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm
        band: Which C-H band to use ('1730', '2310', or '2350')

    Returns:
        Absorption depth (0-1, higher = more absorption)

    Reference:
        Cloutis, E.A. (1989). Spectral reflectance properties of hydrocarbons.
        Science, 245(4914), 165-168.
    """
    if band == '1730':
        return absorption_depth(rfl, wavelengths, target=1730, left=1660, right=1780)
    elif band == '2310':
        return absorption_depth(rfl, wavelengths, target=2310, left=2260, right=2350)
    elif band == '2350':
        return absorption_depth(rfl, wavelengths, target=2350, left=2310, right=2380)
    else:
        raise ValueError(f"Unknown band: {band}. Use '1730', '2310', or '2350'")


def methane_ratio(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Methane detection ratio using 2300nm absorption.

    Methane (CH4) has absorption at ~2300nm that can be detected
    in natural gas seeps and industrial emissions.

    Formula (simplified from Thorpe et al., 2014):
        MR = R_2260 / R_2300

    Higher values indicate methane absorption.

    Reference:
        Thorpe, A.K., et al. (2014). High resolution mapping of methane
        emissions from marine and terrestrial sources using a Cluster-Tuned
        Matched Filter technique. RSE, 134, 305-318.

    Note: For quantitative CH4 retrieval, use matched filter or
    IMAP-DOAS methods (not implemented here).

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Methane ratio (higher = more CH4 absorption)
    """
    r_2260 = get_band(rfl, wavelengths, 2260)
    r_2300 = get_band(rfl, wavelengths, 2300)

    eps = 1e-10
    mr = r_2260 / (r_2300 + eps)

    return mr


def asphaltic_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Asphaltic/tar-like material index.

    Heavy hydrocarbons (asphalt, tar, bitumen) show absorption at
    both 1730nm and 2310nm, plus overall darker appearance.

    Formula:
        AI = HI × (1 - mean_SWIR)

    Combines hydrocarbon signature with overall darkness.

    Reference:
        van der Meer, F., et al. (2002). Imaging spectrometry and petroleum
        geology. In Imaging Spectrometry (pp. 219-241). Springer.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Asphaltic index
    """
    hi = hydrocarbon_index(rfl, wavelengths)

    # Average SWIR reflectance as darkness proxy
    r_1600 = get_band(rfl, wavelengths, 1600)
    r_2200 = get_band(rfl, wavelengths, 2200)
    mean_swir = (r_1600 + r_2200) / 2

    # Dark + hydrocarbon signature
    ai = np.maximum(hi, 0) * (1 - mean_swir)

    return ai


# =============================================================================
# Oil Slick Specific (Marine Applications)
# =============================================================================

def oil_slick_thickness_proxy(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Oil slick thickness proxy for marine applications.

    Thicker oil slicks show stronger 1730nm absorption.
    Based on empirical relationship from lab studies.

    Detection threshold:
        - Thin sheen: HI < 0.01
        - Moderate: 0.01 < HI < 0.05
        - Thick: HI > 0.05

    Reference:
        Clark, R.N., et al. (2010). A method for quantitative mapping of
        thick oil spills using imaging spectroscopy. USGS Open-File Report
        2010-1167.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Thickness proxy (relative, not absolute thickness)
    """
    # Use absorption depth for better quantification
    ad = hydrocarbon_absorption_depth(rfl, wavelengths, band='1730')

    # Threshold for detection
    ad = np.where(ad > 0.005, ad, 0)  # Below noise floor

    return ad


def oil_water_ratio(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Oil-to-water emulsion ratio for marine slicks.

    Emulsions show different spectral characteristics than pure oil.
    Water absorption at 1450nm combined with oil at 1730nm.

    Formula:
        OWR = AD_1730 / (1 - R_1450)

    Reference:
        Wettle, M., et al. (2009). Assessing the effect of hydrocarbon oil
        type and thickness on a remote sensing signal: A sensitivity study
        based on the optical properties of two different oil types and
        the HYMAP and Quickbird sensors. RSE, 113(9), 2000-2010.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Oil-water ratio
    """
    ad_1730 = hydrocarbon_absorption_depth(rfl, wavelengths, band='1730')
    r_1450 = get_band(rfl, wavelengths, 1450)

    eps = 1e-10
    # Low R_1450 = water absorption = high water content
    owr = ad_1730 / (1 - r_1450 + eps)

    return owr
