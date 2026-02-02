"""
Vegetation spectral indices.

Standard vegetation indices for greenness, water content, and health assessment.

References:
    Rouse, J.W., et al. (1974). Monitoring vegetation systems in the Great Plains
        with ERTS. NASA SP-351, 309-317.

    Gao, B.C. (1996). NDWI - A normalized difference water index for remote
        sensing of vegetation liquid water from space. RSE, 58(3), 257-266.

    Huete, A.R. (1988). A soil-adjusted vegetation index (SAVI). RSE, 25(3), 295-309.

    Huete, A., et al. (2002). Overview of the radiometric and biophysical performance
        of the MODIS vegetation indices. RSE, 83(1-2), 195-213.

    Gitelson, A.A., & Merzlyak, M.N. (1997). Remote estimation of chlorophyll content
        in higher plant leaves. International Journal of Remote Sensing, 18(12),
        2691-2697.
"""

import numpy as np
from .utils import get_band, normalized_difference


# =============================================================================
# Key Wavelengths (nm)
# =============================================================================

VEGETATION_BANDS = {
    'blue': 480,
    'green': 560,
    'red': 650,
    'red_edge_1': 705,
    'red_edge_2': 750,
    'nir': 860,
    'nir_shoulder': 970,
    'swir_1': 1240,
    'water_1': 1450,
    'swir_2': 1650,
    'water_2': 1950,
    'swir_3': 2200,
}


# =============================================================================
# Basic Vegetation Indices
# =============================================================================

def ndvi(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    red_nm: float = 650,
    nir_nm: float = 860,
) -> np.ndarray:
    """
    Normalized Difference Vegetation Index.

    The most widely used vegetation index, sensitive to green biomass.

    Formula:
        NDVI = (NIR - Red) / (NIR + Red)

    Range: -1 to 1
        - < 0: water, snow, clouds
        - 0-0.2: bare soil, rock
        - 0.2-0.4: sparse vegetation
        - 0.4-0.6: moderate vegetation
        - > 0.6: dense vegetation

    Reference:
        Rouse, J.W., et al. (1974). Monitoring vegetation systems in the
        Great Plains with ERTS. NASA SP-351, 309-317.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm
        red_nm: Red band center wavelength
        nir_nm: NIR band center wavelength

    Returns:
        NDVI values (-1 to 1)
    """
    return normalized_difference(rfl, wavelengths, nir_nm, red_nm)


def ndwi(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Normalized Difference Water Index.

    Sensitive to vegetation water content using NIR and SWIR bands.

    Formula:
        NDWI = (NIR - SWIR) / (NIR + SWIR)

    Reference:
        Gao, B.C. (1996). NDWI - A normalized difference water index for
        remote sensing of vegetation liquid water from space. RSE, 58(3),
        257-266.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        NDWI values (-1 to 1)
    """
    return normalized_difference(rfl, wavelengths, 860, 1240)


def ndmi(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Normalized Difference Moisture Index.

    More sensitive to canopy moisture than NDWI.

    Formula:
        NDMI = (NIR - SWIR_1650) / (NIR + SWIR_1650)

    Reference:
        Wilson, E.H., & Sader, S.A. (2002). Detection of forest harvest type
        using multiple dates of Landsat TM imagery. RSE, 80(3), 385-396.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        NDMI values (-1 to 1)
    """
    return normalized_difference(rfl, wavelengths, 860, 1650)


def evi(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    G: float = 2.5,
    C1: float = 6.0,
    C2: float = 7.5,
    L: float = 1.0,
) -> np.ndarray:
    """
    Enhanced Vegetation Index.

    Optimized to reduce atmospheric and soil background effects.
    Used by MODIS and other satellite systems.

    Formula:
        EVI = G × (NIR - Red) / (NIR + C1×Red - C2×Blue + L)

    Reference:
        Huete, A., et al. (2002). Overview of the radiometric and biophysical
        performance of the MODIS vegetation indices. RSE, 83(1-2), 195-213.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm
        G: Gain factor (default 2.5)
        C1: Atmospheric correction coefficient for red (default 6.0)
        C2: Atmospheric correction coefficient for blue (default 7.5)
        L: Canopy background adjustment (default 1.0)

    Returns:
        EVI values
    """
    red = get_band(rfl, wavelengths, 650)
    nir = get_band(rfl, wavelengths, 860)
    blue = get_band(rfl, wavelengths, 480)

    eps = 1e-10
    evi_val = G * (nir - red) / (nir + C1 * red - C2 * blue + L + eps)

    return evi_val


def savi(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    L: float = 0.5,
) -> np.ndarray:
    """
    Soil Adjusted Vegetation Index.

    Minimizes soil brightness influences on vegetation indices.

    Formula:
        SAVI = (NIR - Red) × (1 + L) / (NIR + Red + L)

    Reference:
        Huete, A.R. (1988). A soil-adjusted vegetation index (SAVI).
        Remote Sensing of Environment, 25(3), 295-309.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm
        L: Soil adjustment factor (default 0.5, range 0-1)
            - L=0: same as NDVI
            - L=1: maximum soil adjustment

    Returns:
        SAVI values
    """
    red = get_band(rfl, wavelengths, 650)
    nir = get_band(rfl, wavelengths, 860)

    eps = 1e-10
    savi_val = (nir - red) * (1 + L) / (nir + red + L + eps)

    return savi_val


# =============================================================================
# Red-Edge and Chlorophyll Indices
# =============================================================================

def pssr(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Pigment Specific Simple Ratio.

    Sensitive to chlorophyll a concentration.

    Formula:
        PSSR = R_800 / R_680

    Reference:
        Blackburn, G.A. (1998). Spectral indices for estimating photosynthetic
        pigment concentrations: A test using senescent tree leaves.
        International Journal of Remote Sensing, 19(4), 657-675.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        PSSR values
    """
    r_680 = get_band(rfl, wavelengths, 680)
    r_800 = get_band(rfl, wavelengths, 800)

    eps = 1e-10
    return r_800 / (r_680 + eps)


def mcari(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Modified Chlorophyll Absorption in Reflectance Index.

    Sensitive to chlorophyll a while minimizing soil background.

    Formula:
        MCARI = [(R_700 - R_670) - 0.2 × (R_700 - R_550)] × (R_700 / R_670)

    Reference:
        Daughtry, C.S.T., et al. (2000). Estimating corn leaf chlorophyll
        concentration from leaf and canopy reflectance. RSE, 74(2), 229-239.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        MCARI values
    """
    r_550 = get_band(rfl, wavelengths, 550)
    r_670 = get_band(rfl, wavelengths, 670)
    r_700 = get_band(rfl, wavelengths, 700)

    eps = 1e-10
    mcari_val = ((r_700 - r_670) - 0.2 * (r_700 - r_550)) * (r_700 / (r_670 + eps))

    return mcari_val


def tcari(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Transformed Chlorophyll Absorption in Reflectance Index.

    Corrected version of MCARI for variable leaf structure.

    Formula:
        TCARI = 3 × [(R_700 - R_670) - 0.2 × (R_700 - R_550) × (R_700 / R_670)]

    Reference:
        Haboudane, D., et al. (2002). Integrated narrow-band vegetation indices
        for prediction of crop chlorophyll content. RSE, 81(2-3), 416-426.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        TCARI values
    """
    r_550 = get_band(rfl, wavelengths, 550)
    r_670 = get_band(rfl, wavelengths, 670)
    r_700 = get_band(rfl, wavelengths, 700)

    eps = 1e-10
    tcari_val = 3 * ((r_700 - r_670) - 0.2 * (r_700 - r_550) * (r_700 / (r_670 + eps)))

    return tcari_val


def osavi(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Optimized Soil Adjusted Vegetation Index.

    TCARI/OSAVI is highly correlated with chlorophyll content.

    Formula:
        OSAVI = (NIR - Red) / (NIR + Red + 0.16)

    Reference:
        Rondeaux, G., Steven, M., & Baret, F. (1996). Optimization of soil-
        adjusted vegetation indices. RSE, 55(2), 95-107.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        OSAVI values
    """
    red = get_band(rfl, wavelengths, 670)
    nir = get_band(rfl, wavelengths, 800)

    eps = 1e-10
    osavi_val = (nir - red) / (nir + red + 0.16 + eps)

    return osavi_val


def red_edge_position(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Red Edge Position (wavelength of maximum slope).

    The red edge position shifts with chlorophyll content:
        - Higher chlorophyll → longer wavelength (~740nm)
        - Lower chlorophyll → shorter wavelength (~700nm)

    Uses linear interpolation to find inflection point.

    Reference:
        Guyot, G., & Baret, F. (1988). Utilisation de la haute resolution
        spectrale pour suivre l'etat des couverts vegetaux. ESA SP-287, 279-286.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Red edge wavelength (nm)
    """
    # Get red edge region bands
    r_670 = get_band(rfl, wavelengths, 670)
    r_700 = get_band(rfl, wavelengths, 700)
    r_740 = get_band(rfl, wavelengths, 740)
    r_780 = get_band(rfl, wavelengths, 780)

    # Linear interpolation for inflection point
    # REP = 700 + 40 × [(R670 + R780)/2 - R700] / (R740 - R700)
    eps = 1e-10
    rep = 700 + 40 * (((r_670 + r_780) / 2 - r_700) / (r_740 - r_700 + eps))

    return rep
