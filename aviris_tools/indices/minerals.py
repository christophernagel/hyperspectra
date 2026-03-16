"""
Mineral spectral indices for clay, carbonate, and iron oxide mapping.

All indices include peer-reviewed citations. Wavelengths are in nanometers.

References:
    Ninomiya, Y. (2003). A stabilized vegetation index and several mineralogic
        indices defined for ASTER VNIR and SWIR data. IEEE IGARSS, 3, 1552-1554.

    Ninomiya, Y., Fu, B., & Cudahy, T. (2005). Detecting lithology with Advanced
        Spaceborne Thermal Emission and Reflection Radiometer (ASTER) multispectral
        thermal infrared "radiance-at-sensor" data. Remote Sensing of Environment,
        99(1-2), 127-139.

    Crowley, J.K., Brickey, D.W., & Rowan, L.C. (1989). Airborne imaging spectrometer
        data of the Ruby Mountains, Montana: Mineral discrimination using relative
        absorption band-depth images. Remote Sensing of Environment, 29(2), 121-134.

    Rowan, L.C., & Mars, J.C. (2003). Lithologic mapping in the Mountain Pass,
        California area using ASTER data. Remote Sensing of Environment, 84(3), 350-366.

    Hunt, G.R. (1977). Spectral signatures of particulate minerals in the visible
        and near infrared. Geophysics, 42(3), 501-513.

    Clark, R.N., Swayze, G.A., et al. (2003). Imaging spectroscopy: Earth and
        planetary remote sensing with the USGS Tetracorder and expert systems.
        Journal of Geophysical Research, 108(E12), 5131.
"""

import numpy as np
from .utils import get_band, absorption_depth


# =============================================================================
# Key Absorption Wavelengths (nm) - USGS Spectral Library v7
# =============================================================================

MINERAL_BANDS = {
    # Clay minerals - Al-OH absorption
    'kaolinite_1': 2165,      # Secondary doublet
    'kaolinite_2': 2205,      # Primary Al-OH
    'smectite': 2200,         # Broad Al-OH
    'illite': 2195,           # Al-OH shifted
    'alunite': 2170,          # Al-OH

    # Carbonates - CO3 vibration
    'calcite': 2340,          # Primary CO3
    'dolomite': 2320,         # Primary CO3 (Mg shifts blue)
    'magnesite': 2300,        # Primary CO3
    'carbonate_secondary': 2160,  # Weak feature

    # Iron oxides - electronic transitions
    'hematite': 860,          # Fe3+ crystal field
    'goethite': 900,          # Fe3+ crystal field
    'ferric_1': 480,          # Fe3+ charge transfer
    'ferric_2': 650,          # Fe3+
    'ferrous': 1000,          # Fe2+ crystal field

    # Reference shoulders/continuum
    'swir_shoulder_1': 2120,
    'swir_shoulder_2': 2250,
    'swir_shoulder_3': 2380,
    'vnir_shoulder': 750,
}


# =============================================================================
# Clay Mineral Indices
# =============================================================================

def clay_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    General clay mineral index for Al-OH bearing minerals.

    Detects kaolinite, smectite, illite, and muscovite by the 2200nm absorption.

    Formula (Ninomiya et al., 2005):
        CI = (R_2120 × R_2250) / R_2200²

    Where CI > 1 indicates clay presence.

    Reference:
        Ninomiya, Y., Fu, B., & Cudahy, T. (2005). Detecting lithology with
        ASTER multispectral thermal infrared data. RSE, 99(1-2), 127-139.

    Parameters:
        rfl: Reflectance array (y, x, bands) or (bands,)
        wavelengths: Wavelength array in nm

    Returns:
        Clay index values (>1 indicates clay)
    """
    r_2120 = get_band(rfl, wavelengths, 2120)
    r_2200 = get_band(rfl, wavelengths, 2200)
    r_2250 = get_band(rfl, wavelengths, 2250)

    denom = r_2200 ** 2
    # Guard: dark materials (reflectance < 0.05) produce false positives
    # from small-number division; mask them out. Preserve NaN propagation.
    bright_enough = r_2200 > 0.05
    out = np.full_like(r_2120, np.nan, dtype=np.float64)
    valid = np.isfinite(denom) & (denom != 0) & bright_enough
    ci = np.divide(r_2120 * r_2250, denom, out=out, where=valid)

    return ci


def kaolinite_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Kaolinite-specific index using the diagnostic doublet at 2165/2205nm.

    Kaolinite has a unique doublet absorption that distinguishes it from
    other Al-OH minerals. The 2165nm feature is diagnostic.

    Formula (modified from Crowley et al., 1989):
        KI = (R_2165 / R_2205) × (R_2165 / R_2120)

    Lower values indicate stronger kaolinite signature.

    Reference:
        Crowley, J.K., Brickey, D.W., & Rowan, L.C. (1989). Mineral discrimination
        using relative absorption band-depth images. RSE, 29(2), 121-134.

        Hunt, G.R., & Salisbury, J.W. (1970). Visible and near-infrared spectra
        of minerals and rocks: I. Silicate minerals. Modern Geology, 1, 283-300.

    Parameters:
        rfl: Reflectance array (y, x, bands)
        wavelengths: Wavelength array in nm

    Returns:
        Kaolinite index (lower = more kaolinite)
    """
    r_2120 = get_band(rfl, wavelengths, 2120)
    r_2165 = get_band(rfl, wavelengths, 2165)
    r_2205 = get_band(rfl, wavelengths, 2205)

    _zeros = np.zeros_like(r_2165, dtype=np.float64)
    ki = (
        np.divide(r_2165, r_2205, out=_zeros.copy(), where=r_2205 != 0)
        * np.divide(r_2165, r_2120, out=_zeros.copy(), where=r_2120 != 0)
    )

    return ki


def smectite_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Smectite (montmorillonite) index using broad 2200nm absorption.

    Smectites have a broad, symmetric absorption centered at 2200nm,
    unlike kaolinite's doublet.

    Formula:
        SI = R_2200 / ((R_2120 + R_2290) / 2)

    Lower values indicate smectite presence.

    Reference:
        Hunt, G.R. (1977). Spectral signatures of particulate minerals.
        Geophysics, 42(3), 501-513.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Smectite index (lower = more smectite)
    """
    r_2120 = get_band(rfl, wavelengths, 2120)
    r_2200 = get_band(rfl, wavelengths, 2200)
    r_2290 = get_band(rfl, wavelengths, 2290)

    continuum = (r_2120 + r_2290) / 2
    si = np.divide(
        r_2200, continuum,
        out=np.zeros_like(r_2200, dtype=np.float64), where=continuum != 0
    )

    return si


def illite_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Illite/muscovite index using 2195nm absorption.

    Illite and muscovite have Al-OH absorption shifted slightly
    shorter than kaolinite (2195nm vs 2205nm).

    Formula:
        II = (R_2120 + R_2250) / (2 × R_2195)

    Higher values indicate illite/muscovite presence.

    Reference:
        Rowan, L.C., & Mars, J.C. (2003). Lithologic mapping using ASTER.
        Remote Sensing of Environment, 84(3), 350-366.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Illite index (higher = more illite)
    """
    r_2120 = get_band(rfl, wavelengths, 2120)
    r_2195 = get_band(rfl, wavelengths, 2195)
    r_2250 = get_band(rfl, wavelengths, 2250)

    denom = 2 * r_2195
    ii = np.divide(
        r_2120 + r_2250, denom,
        out=np.zeros_like(r_2120, dtype=np.float64), where=denom != 0
    )

    return ii


def alunite_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Alunite index using 2170nm Al-OH absorption.

    Alunite (KAl3(SO4)2(OH)6) has distinctive absorption at 2170nm,
    shorter than kaolinite. Important for epithermal Au exploration.

    Formula (from Swayze et al., 2014 Cuprite study):
        AI = R_2170 / ((R_2120 + R_2220) / 2)

    Lower values indicate alunite presence.

    Reference:
        Swayze, G.A., et al. (2014). Mapping advanced argillic alteration
        at Cuprite, Nevada using imaging spectroscopy. Economic Geology,
        109(5), 1179-1221.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Alunite index (lower = more alunite)
    """
    r_2120 = get_band(rfl, wavelengths, 2120)
    r_2170 = get_band(rfl, wavelengths, 2170)
    r_2220 = get_band(rfl, wavelengths, 2220)

    continuum = (r_2120 + r_2220) / 2
    ai = np.divide(
        r_2170, continuum,
        out=np.zeros_like(r_2170, dtype=np.float64), where=continuum != 0
    )

    return ai


# =============================================================================
# Carbonate Indices
# =============================================================================

def carbonate_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    General carbonate index for calcite/dolomite detection.

    Uses the CO3 vibrational absorption at 2300-2350nm.

    Formula (Ninomiya, 2003):
        CarI = (R_2250 × R_2380) / R_2330²

    Values > 1 indicate carbonate presence.

    Reference:
        Ninomiya, Y. (2003). A stabilized vegetation index and mineralogic
        indices for ASTER. IEEE IGARSS, 3, 1552-1554.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Carbonate index (>1 indicates carbonate)
    """
    r_2250 = get_band(rfl, wavelengths, 2250)
    r_2330 = get_band(rfl, wavelengths, 2330)
    r_2380 = get_band(rfl, wavelengths, 2380)

    denom = r_2330 ** 2
    cari = np.divide(
        r_2250 * r_2380, denom,
        out=np.zeros_like(r_2250, dtype=np.float64), where=denom != 0
    )

    return cari


def calcite_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Calcite-specific index using 2340nm absorption.

    Calcite has CO3 absorption at 2340nm, longer wavelength than dolomite.

    Formula:
        CaI = R_2340 / ((R_2290 + R_2390) / 2)

    Lower values indicate calcite presence.

    Reference:
        Gaffey, S.J. (1986). Spectral reflectance of carbonate minerals.
        American Mineralogist, 71, 151-162.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Calcite index (lower = more calcite)
    """
    r_2290 = get_band(rfl, wavelengths, 2290)
    r_2340 = get_band(rfl, wavelengths, 2340)
    r_2390 = get_band(rfl, wavelengths, 2390)

    continuum = (r_2290 + r_2390) / 2
    cai = np.divide(
        r_2340, continuum,
        out=np.zeros_like(r_2340, dtype=np.float64), where=continuum != 0
    )

    return cai


def dolomite_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Dolomite-specific index using 2320nm absorption.

    Dolomite (CaMg(CO3)2) has CO3 absorption shifted to 2320nm due to Mg.
    This distinguishes it from calcite (2340nm).

    Formula:
        DoI = R_2320 / ((R_2260 + R_2380) / 2)

    Lower values indicate dolomite presence.

    Reference:
        Gaffey, S.J. (1986). Spectral reflectance of carbonate minerals.
        American Mineralogist, 71, 151-162.

        van der Meer, F. (1995). Spectral reflectance of carbonate mineral
        mixtures. Remote Sensing Reviews, 13(1-2), 67-94.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Dolomite index (lower = more dolomite)
    """
    r_2260 = get_band(rfl, wavelengths, 2260)
    r_2320 = get_band(rfl, wavelengths, 2320)
    r_2380 = get_band(rfl, wavelengths, 2380)

    continuum = (r_2260 + r_2380) / 2
    doi = np.divide(
        r_2320, continuum,
        out=np.zeros_like(r_2320, dtype=np.float64), where=continuum != 0
    )

    return doi


# =============================================================================
# Iron Oxide Indices
# =============================================================================

def ferric_iron_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Ferric iron (Fe3+) index for hematite/goethite detection.

    Fe3+ minerals show absorption at 480nm and 860-900nm due to
    crystal field transitions.

    Formula (Rowan & Mars, 2003):
        Fe3I = R_750 / R_860

    Higher values indicate ferric iron presence.

    Reference:
        Rowan, L.C., & Mars, J.C. (2003). Lithologic mapping using ASTER.
        Remote Sensing of Environment, 84(3), 350-366.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Ferric iron index (higher = more Fe3+)
    """
    r_750 = get_band(rfl, wavelengths, 750)
    r_860 = get_band(rfl, wavelengths, 860)

    fe3i = np.divide(r_750, r_860, out=np.zeros_like(r_750, dtype=np.float64), where=r_860 != 0)

    return fe3i


def ferrous_iron_index(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Ferrous iron (Fe2+) index for pyroxene/olivine detection.

    Fe2+ in silicates shows broad absorption centered at 1000nm.

    Formula:
        Fe2I = R_860 / R_1000

    Higher values indicate ferrous iron presence.

    Reference:
        Hunt, G.R. (1977). Spectral signatures of particulate minerals.
        Geophysics, 42(3), 501-513.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Ferrous iron index (higher = more Fe2+)
    """
    r_860 = get_band(rfl, wavelengths, 860)
    r_1000 = get_band(rfl, wavelengths, 1000)

    fe2i = np.divide(r_860, r_1000, out=np.zeros_like(r_860, dtype=np.float64), where=r_1000 != 0)

    return fe2i


# =============================================================================
# Absorption Depth Methods (More Accurate)
# =============================================================================

def clay_absorption_depth(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Clay absorption depth at 2200nm using continuum removal.

    More accurate than ratio indices for quantitative mapping.

    Reference:
        Clark, R.N., & Roush, T.L. (1984). Reflectance spectroscopy:
        Quantitative analysis techniques. JGR, 89(B7), 6329-6340.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Absorption depth at 2200nm (0-1, higher = more absorption)
    """
    return absorption_depth(rfl, wavelengths, target=2200, left=2120, right=2250)


def carbonate_absorption_depth(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Carbonate absorption depth at 2330nm using continuum removal.

    Reference:
        Clark, R.N., & Roush, T.L. (1984). Reflectance spectroscopy:
        Quantitative analysis techniques. JGR, 89(B7), 6329-6340.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm

    Returns:
        Absorption depth at 2330nm (0-1, higher = more absorption)
    """
    return absorption_depth(rfl, wavelengths, target=2330, left=2250, right=2380)
