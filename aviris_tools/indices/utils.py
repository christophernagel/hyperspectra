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
from typing import Tuple


def validate_spectral_inputs(rfl: np.ndarray, wavelengths: np.ndarray) -> None:
    """
    Validate reflectance and wavelength arrays for spectral index computation.

    Checks:
        - rfl is 1D (single pixel) or 3D (image cube)
        - wavelengths is 1D
        - spectral axis of rfl matches len(wavelengths)

    Raises:
        ValueError: If validation fails
    """
    if rfl.ndim not in (1, 3):
        raise ValueError(
            f"rfl must be 1D (pixel) or 3D (y, x, bands), got {rfl.ndim}D "
            f"with shape {rfl.shape}"
        )
    if wavelengths.ndim != 1:
        raise ValueError(
            f"wavelengths must be 1D, got {wavelengths.ndim}D "
            f"with shape {wavelengths.shape}"
        )
    n_bands = rfl.shape[-1] if rfl.ndim == 3 else rfl.shape[0]
    if n_bands != len(wavelengths):
        raise ValueError(
            f"Spectral dimension mismatch: rfl has {n_bands} bands "
            f"but wavelengths has {len(wavelengths)} entries"
        )


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
    validate_spectral_inputs(rfl, wavelengths)
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
    validate_spectral_inputs(rfl, wavelengths)

    # Warn if target is outside the wavelength range (extrapolation)
    if target_nm < wavelengths[0] or target_nm > wavelengths[-1]:
        import warnings
        warnings.warn(
            f"interpolate_band: target {target_nm}nm is outside data range "
            f"[{wavelengths[0]:.1f}, {wavelengths[-1]:.1f}]nm; "
            f"returning nearest edge band (no extrapolation)"
        )

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
        continuum = (
            r_start[:, :, np.newaxis]
            + slope[:, :, np.newaxis] * (wl_subset - wl_start)
        )

    cr_rfl = np.divide(rfl_subset, continuum,
                       out=np.zeros_like(rfl_subset, dtype=np.float64),
                       where=continuum != 0)

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

    ratio = np.divide(r_target, r_continuum,
                      out=np.ones_like(r_target, dtype=np.float64),
                      where=r_continuum != 0)
    depth = np.clip(1.0 - ratio, 0, 1)

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

    return np.divide(r_num, r_den, out=np.zeros_like(r_num, dtype=np.float64), where=r_den != 0)


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

    denom = r1 + r2
    return np.divide(r1 - r2, denom, out=np.zeros_like(r1, dtype=np.float64), where=denom != 0)
