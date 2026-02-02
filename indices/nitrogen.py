"""
Nitrogen and agricultural spectral indices.

Indices for detecting crop nitrogen status, protein content, and
fertilizer application effects using hyperspectral data.

Key absorption features:
    - 1510 nm: N-H stretch (protein)
    - 1680 nm: C-H stretch (lignin/cellulose)
    - 1940 nm: O-H stretch (water)
    - 2050 nm: N-H combination (protein)
    - 2170 nm: N-H/C-H combination (protein)
    - 2290 nm: C-H stretch (protein/starch)

References:
    Serrano, L., Penuelas, J., & Ustin, S.L. (2002). Remote sensing of
        nitrogen and lignin in Mediterranean vegetation from AVIRIS data:
        Decomposing biochemical from structural signals. RSE, 81(2-3), 355-364.

    Fourty, T., et al. (1996). Leaf optical properties with explicit
        description of its biochemical composition: Direct and inverse
        problems. RSE, 56(2), 104-117.

    Kokaly, R.F., & Clark, R.N. (1999). Spectroscopic determination of leaf
        biochemistry using band-depth analysis of absorption features and
        stepwise multiple linear regression. RSE, 67(3), 267-287.

    Berger, K., et al. (2020). Crop nitrogen monitoring: Recent progress
        and principal developments in the context of imaging spectroscopy
        missions. RSE, 242, 111758.
"""

import numpy as np
from .utils import get_band, absorption_depth, normalized_difference


# =============================================================================
# Key Absorption Wavelengths (nm)
# =============================================================================

NITROGEN_BANDS = {
    # Protein/Nitrogen features
    'protein_1': 1510,        # N-H stretch
    'protein_2': 2050,        # N-H combination
    'protein_3': 2170,        # N-H/C-H combination
    'protein_4': 2290,        # C-H stretch (amino acids)

    # Lignin/Cellulose features
    'lignin_1': 1680,         # C-H stretch
    'cellulose_1': 2100,      # C-O stretch

    # Chlorophyll (indirect N indicator)
    'chlorophyll_a': 680,     # Red absorption
    'chlorophyll_b': 640,     # Red absorption

    # Reference/continuum
    'swir_continuum_1': 1450,
    'swir_continuum_2': 1800,
    'swir_continuum_3': 2230,
}


# =============================================================================
# Nitrogen Indices
# =============================================================================

def ndni(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Normalized Difference Nitrogen Index.

    The primary hyperspectral index for leaf nitrogen estimation.
    Uses log-transformed reflectance at protein absorption bands.

    Formula:
        NDNI = [log(1/R_1510) - log(1/R_1680)] / [log(1/R_1510) + log(1/R_1680)]

    Reference:
        Serrano, L., Penuelas, J., & Ustin, S.L. (2002). Remote sensing of
        nitrogen and lignin in Mediterranean vegetation from AVIRIS data.
        RSE, 81(2-3), 355-364.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        NDNI values
    """
    r_1510 = get_band(rfl, wavelengths, 1510)
    r_1680 = get_band(rfl, wavelengths, 1680)

    eps = 1e-10
    log_1510 = np.log(1 / (r_1510 + eps))
    log_1680 = np.log(1 / (r_1680 + eps))

    ndni_val = (log_1510 - log_1680) / (log_1510 + log_1680 + eps)

    return ndni_val


def nri(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Nitrogen Reflectance Index.

    Simple ratio using green and NIR bands, related to chlorophyll
    content which correlates with leaf nitrogen.

    Formula:
        NRI = (R_570 - R_670) / (R_570 + R_670)

    Reference:
        Filella, I., & Penuelas, J. (1994). The red edge position and shape
        as indicators of plant chlorophyll content, biomass and hydric status.
        International Journal of Remote Sensing, 15(7), 1459-1470.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        NRI values
    """
    return normalized_difference(rfl, wavelengths, 570, 670)


def canopy_chlorophyll_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Canopy Chlorophyll Content Index.

    CCCI integrates NDVI with red-edge information for better
    nitrogen estimation across variable canopy cover.

    Formula:
        NDRE = (R_790 - R_720) / (R_790 + R_720)
        CCCI = (NDRE - NDRE_min) / (NDRE_max - NDRE_min)

    Simplified as:
        CCCI = NDRE / NDVI

    Reference:
        Barnes, E.M., et al. (2000). Coincident detection of crop water
        stress, nitrogen status and canopy density using ground based
        multispectral data. Proc. 5th Int. Conf. Precision Agriculture.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        CCCI values
    """
    # NDRE - Normalized Difference Red Edge
    ndre = normalized_difference(rfl, wavelengths, 790, 720)

    # NDVI
    ndvi_val = normalized_difference(rfl, wavelengths, 860, 650)

    eps = 1e-10
    ccci = ndre / (ndvi_val + eps)

    return ccci


def protein_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Protein Absorption Index using 2170nm N-H feature.

    Direct detection of protein via N-H bond absorption.
    More accurate than chlorophyll proxies for nitrogen mapping.

    Formula:
        PI = AD_2170 (absorption depth at 2170nm)

    Reference:
        Kokaly, R.F., & Clark, R.N. (1999). Spectroscopic determination
        of leaf biochemistry using band-depth analysis. RSE, 67(3), 267-287.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Protein index (0-1, higher = more protein absorption)
    """
    return absorption_depth(rfl, wavelengths, target=2170, left=2100, right=2230)


def cellulose_absorption_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Cellulose Absorption Index at 2100nm.

    High cellulose with low protein indicates senescent or
    nitrogen-stressed vegetation.

    Formula:
        CAI = AD_2100 (absorption depth at 2100nm)

    Reference:
        Nagler, P.L., Inoue, Y., Glenn, E.P., Russ, A.L., & Daughtry, C.S.T.
        (2003). Cellulose absorption index (CAI) to quantify mixed soil-plant
        litter scenes. RSE, 87(2-3), 310-325.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Cellulose absorption index
    """
    return absorption_depth(rfl, wavelengths, target=2100, left=2030, right=2170)


def nitrogen_stress_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Nitrogen Stress Index combining multiple indicators.

    Integrates protein absorption with cellulose ratio
    to detect N-deficient vegetation.

    Formula:
        NSI = CAI / (PI + eps)

    Higher values indicate nitrogen stress (high cellulose, low protein).

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Nitrogen stress index (higher = more stress)
    """
    pi = protein_index(rfl, wavelengths)
    cai = cellulose_absorption_index(rfl, wavelengths)

    eps = 1e-10
    nsi = cai / (pi + eps)

    return nsi


# =============================================================================
# Fertilizer Application Detection
# =============================================================================

def nitrogen_sufficiency_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    reference_ndvi: float = 0.8,
) -> np.ndarray:
    """
    Nitrogen Sufficiency Index for precision agriculture.

    Compares vegetation vigor to well-fertilized reference.
    Used to guide variable-rate nitrogen application.

    Formula:
        NSI = NDVI / NDVI_reference

    Reference:
        Bausch, W.C., & Duke, H.R. (1996). Remote sensing of plant nitrogen
        status in corn. Transactions of ASAE, 39(5), 1869-1875.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm
        reference_ndvi: NDVI of well-fertilized reference (default 0.8)

    Returns:
        NSI values (1.0 = sufficient, <1.0 = deficient)
    """
    from .vegetation import ndvi as calc_ndvi

    ndvi_val = calc_ndvi(rfl, wavelengths)
    nsi = ndvi_val / reference_ndvi

    return nsi


def leaf_nitrogen_concentration(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Estimated Leaf Nitrogen Concentration (% dry weight).

    Empirical model based on AVIRIS studies of agricultural crops.
    Calibrated for maize, wheat, and soybean.

    Formula (from Berger et al., 2020 review):
        LNC = 0.31 + 3.97 × NDNI

    Units: % dry weight
    Valid range: approximately 1-5%

    Reference:
        Berger, K., et al. (2020). Crop nitrogen monitoring: Recent progress
        and principal developments. RSE, 242, 111758.

    Note: This is an empirical approximation. Accuracy varies by crop type
    and growth stage. For quantitative applications, calibrate with local
    field measurements.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Estimated LNC (% dry weight)
    """
    ndni_val = ndni(rfl, wavelengths)

    # Empirical coefficients (approximate, crop-dependent)
    lnc = 0.31 + 3.97 * ndni_val

    # Clamp to realistic range
    lnc = np.clip(lnc, 0.5, 6.0)

    return lnc
