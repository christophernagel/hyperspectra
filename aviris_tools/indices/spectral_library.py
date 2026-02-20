"""
Reference spectral library for mineral and material identification.

Contains key absorption features and reference spectra derived from
USGS Spectral Library Version 7 (Kokaly et al., 2017).

These are simplified diagnostic wavelengths and expected absorption depths
for automated spectral matching. For full spectra, download the complete
USGS library from: https://doi.org/10.5066/F7RR1WDJ

References:
    Kokaly, R.F., et al. (2017). USGS Spectral Library Version 7.
        USGS Data Series 1035. https://doi.org/10.3334/ORNLDAAC/1035

    Clark, R.N., et al. (2007). USGS Digital Spectral Library splib06a.
        USGS Digital Data Series 231.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MineralSignature:
    """Reference spectral signature for a mineral."""
    name: str
    formula: str
    primary_absorption: float  # nm
    secondary_absorptions: List[float]  # nm
    absorption_depths: Dict[float, Tuple[float, float]]  # wavelength: (min, max depth)
    diagnostic_ratios: Dict[str, Tuple[float, float]]  # ratio_name: (min, max)
    notes: str = ""


# =============================================================================
# Reference Mineral Signatures (USGS Spectral Library v7)
# =============================================================================

MINERAL_LIBRARY = {
    # -------------------------------------------------------------------------
    # Clay Minerals
    # -------------------------------------------------------------------------
    'kaolinite': MineralSignature(
        name='Kaolinite',
        formula='Al2Si2O5(OH)4',
        primary_absorption=2205,
        secondary_absorptions=[2165, 1400, 1900],
        absorption_depths={
            2165: (0.05, 0.25),  # Diagnostic doublet
            2205: (0.10, 0.40),  # Primary Al-OH
        },
        diagnostic_ratios={
            'doublet_ratio': (0.4, 0.8),  # R_2165/R_2205
        },
        notes="Diagnostic doublet at 2165/2205nm distinguishes from other clays"
    ),

    'montmorillonite': MineralSignature(
        name='Montmorillonite (Smectite)',
        formula='(Na,Ca)0.33(Al,Mg)2Si4O10(OH)2·nH2O',
        primary_absorption=2200,
        secondary_absorptions=[1400, 1900, 2300],
        absorption_depths={
            2200: (0.08, 0.35),  # Broad Al-OH
        },
        diagnostic_ratios={
            'clay_index': (1.02, 1.15),  # Ninomiya CI
        },
        notes="Broader absorption than kaolinite, no distinct doublet"
    ),

    'illite': MineralSignature(
        name='Illite/Muscovite',
        formula='(K,H3O)(Al,Mg,Fe)2(Si,Al)4O10[(OH)2,(H2O)]',
        primary_absorption=2195,
        secondary_absorptions=[2350, 2440],
        absorption_depths={
            2195: (0.05, 0.30),  # Al-OH (shifted)
            2350: (0.03, 0.15),  # Al-OH
        },
        diagnostic_ratios={
            'illite_index': (1.05, 1.20),
        },
        notes="Al-OH shifted ~10nm shorter than kaolinite"
    ),

    'alunite': MineralSignature(
        name='Alunite',
        formula='KAl3(SO4)2(OH)6',
        primary_absorption=2170,
        secondary_absorptions=[1480, 2210],
        absorption_depths={
            2170: (0.08, 0.35),  # Primary Al-OH
        },
        diagnostic_ratios={
            'alunite_index': (0.70, 0.95),
        },
        notes="Important for epithermal Au exploration (Cuprite)"
    ),

    # -------------------------------------------------------------------------
    # Carbonates
    # -------------------------------------------------------------------------
    'calcite': MineralSignature(
        name='Calcite',
        formula='CaCO3',
        primary_absorption=2340,
        secondary_absorptions=[2160, 2530, 1900],
        absorption_depths={
            2340: (0.10, 0.50),  # Primary CO3
            2530: (0.05, 0.25),  # Secondary CO3
        },
        diagnostic_ratios={
            'carbonate_index': (1.02, 1.20),
        },
        notes="Primary CO3 at 2340nm, longer than dolomite"
    ),

    'dolomite': MineralSignature(
        name='Dolomite',
        formula='CaMg(CO3)2',
        primary_absorption=2320,
        secondary_absorptions=[2160, 2510],
        absorption_depths={
            2320: (0.08, 0.45),  # Primary CO3 (Mg-shifted)
            2510: (0.04, 0.20),  # Secondary CO3
        },
        diagnostic_ratios={
            'calcite_dolomite_ratio': (0.85, 0.98),  # R_2340/R_2320
        },
        notes="Mg shifts CO3 absorption ~20nm shorter than calcite"
    ),

    # -------------------------------------------------------------------------
    # Iron Oxides
    # -------------------------------------------------------------------------
    'hematite': MineralSignature(
        name='Hematite',
        formula='Fe2O3',
        primary_absorption=860,
        secondary_absorptions=[480, 650, 2200],
        absorption_depths={
            860: (0.10, 0.40),  # Fe3+ crystal field
        },
        diagnostic_ratios={
            'ferric_index': (1.1, 2.0),  # R_750/R_860
        },
        notes="Red color from Fe3+ charge transfer at 480nm"
    ),

    'goethite': MineralSignature(
        name='Goethite',
        formula='FeO(OH)',
        primary_absorption=900,
        secondary_absorptions=[480, 670, 2200],
        absorption_depths={
            900: (0.08, 0.35),  # Fe3+ crystal field
        },
        diagnostic_ratios={
            'ferric_index': (1.05, 1.8),
        },
        notes="Yellow-brown, absorption ~900nm vs 860nm for hematite"
    ),

    # -------------------------------------------------------------------------
    # Hydrocarbons
    # -------------------------------------------------------------------------
    'crude_oil': MineralSignature(
        name='Crude Oil',
        formula='CxHy',
        primary_absorption=1730,
        secondary_absorptions=[1200, 2310, 2350],
        absorption_depths={
            1730: (0.02, 0.20),  # C-H 1st overtone
            2310: (0.02, 0.15),  # C-H combination
        },
        diagnostic_ratios={
            'hydrocarbon_index': (0.01, 0.15),  # Kühn HI
        },
        notes="Diagnostic C-H absorption at 1730nm and 2310nm"
    ),

    'asphalt': MineralSignature(
        name='Asphalt/Bitumen',
        formula='CxHy (heavy)',
        primary_absorption=1730,
        secondary_absorptions=[2310],
        absorption_depths={
            1730: (0.05, 0.30),  # Strong C-H
        },
        diagnostic_ratios={
            'hydrocarbon_index': (0.05, 0.25),
        },
        notes="Darker overall than crude oil, stronger absorptions"
    ),
}


# =============================================================================
# Spectral Matching Functions
# =============================================================================

def identify_mineral(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    candidates: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Identify minerals by matching absorption features.

    Parameters:
        rfl: Reflectance spectrum (1D array)
        wavelengths: Wavelength array in nm
        candidates: Optional list of mineral names to check
                   (default: all minerals in library)

    Returns:
        Dictionary of {mineral_name: confidence_score}
        Scores 0-1, higher = better match
    """
    from .utils import absorption_depth, get_band

    if candidates is None:
        candidates = list(MINERAL_LIBRARY.keys())

    scores = {}

    for mineral_name in candidates:
        if mineral_name not in MINERAL_LIBRARY:
            continue

        sig = MINERAL_LIBRARY[mineral_name]
        score = 0.0
        n_features = 0

        # Check primary absorption
        if sig.primary_absorption > 0:
            # Simple absorption depth check
            left = sig.primary_absorption - 50
            right = sig.primary_absorption + 50

            try:
                ad = absorption_depth(rfl, wavelengths,
                                     target=sig.primary_absorption,
                                     left=left, right=right)

                if sig.primary_absorption in sig.absorption_depths:
                    min_depth, max_depth = sig.absorption_depths[sig.primary_absorption]
                    if min_depth <= float(ad) <= max_depth:
                        score += 1.0
                    elif float(ad) > 0.01:  # Some absorption present
                        score += 0.5
                n_features += 1
            except:
                pass

        # Check secondary absorptions
        for wl in sig.secondary_absorptions[:2]:  # Check first 2
            try:
                ad = absorption_depth(rfl, wavelengths,
                                     target=wl, left=wl-50, right=wl+50)
                if float(ad) > 0.02:
                    score += 0.3
                n_features += 0.5
            except:
                pass

        # Normalize score
        if n_features > 0:
            scores[mineral_name] = score / n_features
        else:
            scores[mineral_name] = 0.0

    return scores


def get_expected_absorption(
    mineral: str,
    wavelength: float,
) -> Tuple[float, float]:
    """
    Get expected absorption depth range for a mineral at given wavelength.

    Parameters:
        mineral: Mineral name
        wavelength: Wavelength in nm

    Returns:
        Tuple of (min_depth, max_depth) or (0, 0) if not defined
    """
    if mineral not in MINERAL_LIBRARY:
        return (0, 0)

    sig = MINERAL_LIBRARY[mineral]

    # Check if wavelength is close to a defined feature
    for wl, depths in sig.absorption_depths.items():
        if abs(wl - wavelength) < 20:
            return depths

    return (0, 0)


def list_minerals_by_feature(
    wavelength: float,
    tolerance: float = 30.0,
) -> List[str]:
    """
    List minerals with absorption features near a wavelength.

    Parameters:
        wavelength: Target wavelength in nm
        tolerance: Search tolerance in nm

    Returns:
        List of mineral names with features in range
    """
    matches = []

    for name, sig in MINERAL_LIBRARY.items():
        if abs(sig.primary_absorption - wavelength) < tolerance:
            matches.append(name)
        for sec in sig.secondary_absorptions:
            if abs(sec - wavelength) < tolerance:
                if name not in matches:
                    matches.append(name)

    return matches


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_mineral_spectrum(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    mineral: str,
    strict: bool = False,
) -> Tuple[bool, str]:
    """
    Validate that a spectrum matches expected mineral signature.

    Parameters:
        rfl: Reflectance spectrum
        wavelengths: Wavelength array in nm
        mineral: Expected mineral name
        strict: If True, require all features present

    Returns:
        Tuple of (is_valid, message)
    """
    if mineral not in MINERAL_LIBRARY:
        return False, f"Unknown mineral: {mineral}"

    sig = MINERAL_LIBRARY[mineral]
    issues = []
    passes = []

    from .utils import absorption_depth

    # Check primary absorption
    try:
        ad = absorption_depth(rfl, wavelengths,
                             target=sig.primary_absorption,
                             left=sig.primary_absorption - 50,
                             right=sig.primary_absorption + 50)

        if sig.primary_absorption in sig.absorption_depths:
            min_d, max_d = sig.absorption_depths[sig.primary_absorption]
            if float(ad) < min_d:
                issues.append(f"Primary {sig.primary_absorption}nm too weak: {float(ad):.3f} < {min_d}")
            elif float(ad) > max_d:
                issues.append(f"Primary {sig.primary_absorption}nm too strong: {float(ad):.3f} > {max_d}")
            else:
                passes.append(f"Primary {sig.primary_absorption}nm: {float(ad):.3f}")
    except Exception as e:
        issues.append(f"Could not check primary absorption: {e}")

    if strict and issues:
        return False, "; ".join(issues)

    if len(passes) > 0:
        return True, "; ".join(passes)

    return False, "; ".join(issues) if issues else "No features detected"
