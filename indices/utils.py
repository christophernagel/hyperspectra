"""
Utility functions for spectral index calculations.

These functions handle band selection, continuum removal, and absorption
depth calculations used across all index modules.

References:
    Clark, R.N., & Roush, T.L. (1984). Reflectance spectroscopy: Quantitative
        analysis techniques for remote sensing applications. Journal of
        Geophysical Research, 89(B7), 6329-6340.
"""

import numpy as np
from typing import Union, Optional, Tuple


def get_band(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    target_nm: float,
    tolerance: float = 20.0,
) -> np.ndarray:
    """
    Extract band closest to target wavelength.

    Parameters:
        rfl: Reflectance array (y, x, bands) or (bands,)
        wavelengths: Wavelength array in nm
        target_nm: Target wavelength in nm
        tolerance: Maximum acceptable difference (nm), warns if exceeded

    Returns:
        Reflectance values at nearest band
    """
    wavelengths = np.asarray(wavelengths)
    idx = np.argmin(np.abs(wavelengths - target_nm))
    actual = wavelengths[idx]

    if abs(actual - target_nm) > tolerance:
        import warnings
        warnings.warn(
            f"Requested {target_nm}nm, using {actual:.1f}nm "
            f"(diff={abs(actual - target_nm):.1f}nm)"
        )

    if rfl.ndim == 1:
        return rfl[idx]
    elif rfl.ndim == 3:
        return rfl[:, :, idx]
    else:
        raise ValueError(f"Expected 1D or 3D array, got {rfl.ndim}D")


def get_band_index(
    wavelengths: np.ndarray,
    target_nm: float,
) -> int:
    """
    Get index of band closest to target wavelength.

    Parameters:
        wavelengths: Wavelength array in nm
        target_nm: Target wavelength in nm

    Returns:
        Band index
    """
    wavelengths = np.asarray(wavelengths)
    return int(np.argmin(np.abs(wavelengths - target_nm)))


def interpolate_band(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    target_nm: float,
) -> np.ndarray:
    """
    Interpolate reflectance at exact target wavelength.

    Uses linear interpolation between adjacent bands.

    Parameters:
        rfl: Reflectance array (y, x, bands) or (bands,)
        wavelengths: Wavelength array in nm
        target_nm: Target wavelength in nm

    Returns:
        Interpolated reflectance values
    """
    wavelengths = np.asarray(wavelengths)

    # Find bracketing bands
    idx_upper = np.searchsorted(wavelengths, target_nm)
    idx_lower = max(0, idx_upper - 1)
    idx_upper = min(len(wavelengths) - 1, idx_upper)

    if idx_lower == idx_upper:
        return get_band(rfl, wavelengths, target_nm)

    # Linear interpolation weight
    wl_lower = wavelengths[idx_lower]
    wl_upper = wavelengths[idx_upper]
    weight = (target_nm - wl_lower) / (wl_upper - wl_lower)

    if rfl.ndim == 1:
        return rfl[idx_lower] * (1 - weight) + rfl[idx_upper] * weight
    elif rfl.ndim == 3:
        return rfl[:, :, idx_lower] * (1 - weight) + rfl[:, :, idx_upper] * weight
    else:
        raise ValueError(f"Expected 1D or 3D array, got {rfl.ndim}D")


def continuum_removal(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    wl_start: float,
    wl_end: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply continuum removal to spectral region.

    Fits a convex hull (simplified as linear here) and divides
    spectrum by continuum.

    Reference:
        Clark, R.N., & Roush, T.L. (1984). Reflectance spectroscopy:
        Quantitative analysis techniques. JGR, 89(B7), 6329-6340.

    Parameters:
        rfl: Reflectance array (y, x, bands) or (bands,)
        wavelengths: Wavelength array in nm
        wl_start: Start wavelength for continuum (nm)
        wl_end: End wavelength for continuum (nm)

    Returns:
        Tuple of (continuum_removed_rfl, continuum) in spectral subset
    """
    wavelengths = np.asarray(wavelengths)

    # Find spectral region
    mask = (wavelengths >= wl_start) & (wavelengths <= wl_end)
    wl_subset = wavelengths[mask]

    # Get endpoint reflectances
    r_start = interpolate_band(rfl, wavelengths, wl_start)
    r_end = interpolate_band(rfl, wavelengths, wl_end)

    # Linear continuum
    slope = (r_end - r_start) / (wl_end - wl_start)

    if rfl.ndim == 1:
        rfl_subset = rfl[mask]
        continuum = r_start + slope * (wl_subset - wl_start)
    else:
        rfl_subset = rfl[:, :, mask]
        # Broadcast for 3D array
        continuum = r_start[:, :, np.newaxis] + \
                    slope[:, :, np.newaxis] * (wl_subset - wl_start)

    eps = 1e-10
    cr_rfl = rfl_subset / (continuum + eps)

    return cr_rfl, continuum


def absorption_depth(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    target: float,
    left: float,
    right: float,
) -> np.ndarray:
    """
    Calculate absorption depth at target wavelength.

    Uses linear continuum between left and right shoulders.

    Depth = 1 - (R_target / R_continuum)

    Reference:
        Clark, R.N., & Roush, T.L. (1984). Reflectance spectroscopy:
        Quantitative analysis techniques. JGR, 89(B7), 6329-6340.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm
        target: Absorption center wavelength (nm)
        left: Left shoulder wavelength (nm)
        right: Right shoulder wavelength (nm)

    Returns:
        Absorption depth (0-1, higher = deeper absorption)
    """
    r_left = interpolate_band(rfl, wavelengths, left)
    r_right = interpolate_band(rfl, wavelengths, right)
    r_target = interpolate_band(rfl, wavelengths, target)

    # Linear continuum at target wavelength
    weight = (target - left) / (right - left)
    r_continuum = r_left * (1 - weight) + r_right * weight

    eps = 1e-10
    depth = 1.0 - (r_target / (r_continuum + eps))

    # Clamp to valid range
    depth = np.clip(depth, 0, 1)

    return depth


def band_ratio(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    numerator_nm: float,
    denominator_nm: float,
) -> np.ndarray:
    """
    Simple band ratio calculation.

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm
        numerator_nm: Numerator band wavelength (nm)
        denominator_nm: Denominator band wavelength (nm)

    Returns:
        Band ratio values
    """
    r_num = get_band(rfl, wavelengths, numerator_nm)
    r_den = get_band(rfl, wavelengths, denominator_nm)

    eps = 1e-10
    return r_num / (r_den + eps)


def normalized_difference(
    rfl: np.ndarray,
    wavelengths: np.ndarray,
    band1_nm: float,
    band2_nm: float,
) -> np.ndarray:
    """
    Normalized difference index: (B1 - B2) / (B1 + B2)

    Parameters:
        rfl: Reflectance array
        wavelengths: Wavelength array in nm
        band1_nm: First band wavelength (nm)
        band2_nm: Second band wavelength (nm)

    Returns:
        Normalized difference values (-1 to 1)
    """
    r1 = get_band(rfl, wavelengths, band1_nm)
    r2 = get_band(rfl, wavelengths, band2_nm)

    eps = 1e-10
    return (r1 - r2) / (r1 + r2 + eps)
