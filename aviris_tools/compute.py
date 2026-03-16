"""
Shared computation logic for hyperspectral data analysis.

Functions extracted from viewer/app.py and web/engine.py to eliminate
duplication. Both the desktop (napari) and web (trame) viewers import
from this module.
"""

import numpy as np

from aviris_tools.viewer.constants import (
    INDEX_DEFINITIONS,
    ROBUST_PERCENTILE_LOW,
    ROBUST_PERCENTILE_HIGH,
    LARGE_ARRAY_THRESHOLD,
    SAMPLE_SIZE,
)


def normalize_rgb(r, g, b, percentile_lo=2, percentile_hi=98):
    """
    Percentile-normalize three bands and stack into float32 [0, 1] RGB.

    Parameters
    ----------
    r, g, b : np.ndarray
        2D arrays for each channel.
    percentile_lo, percentile_hi : float
        Percentile bounds for contrast stretching.

    Returns
    -------
    np.ndarray
        Float32 array of shape (rows, cols, 3) in [0, 1].
    """
    def _norm(band):
        with np.errstate(all="ignore"):
            plo, phi = np.nanpercentile(band, [percentile_lo, percentile_hi])
        if phi <= plo or np.isnan(plo):
            return np.zeros_like(band)
        return np.clip((band - plo) / (phi - plo), 0, 1)

    rgb = np.stack([_norm(r), _norm(g), _norm(b)], axis=-1)
    return np.nan_to_num(rgb, nan=0.0).astype(np.float32)


def calculate_nd_index(loader, b1_wl, b2_wl):
    """
    Normalized difference index: (b1 - b2) / (b1 + b2).

    Returns float32 2D array.
    """
    b1 = np.asarray(loader.get_interpolated_band(b1_wl), dtype=np.float64)
    b2 = np.asarray(loader.get_interpolated_band(b2_wl), dtype=np.float64)
    denom = b1 + b2
    result = np.divide(b1 - b2, denom, out=np.zeros_like(b1), where=denom != 0)
    return result.astype(np.float32)


def calculate_ratio_index(loader, b1_wl, b2_wl):
    """
    Band ratio index: b1 / b2, clipped to [0.01, 10.0].

    Returns float32 2D array.
    """
    b1 = np.asarray(loader.get_interpolated_band(b1_wl), dtype=np.float64)
    b2 = np.asarray(loader.get_interpolated_band(b2_wl), dtype=np.float64)
    result = np.divide(b1, b2, out=np.zeros_like(b1), where=b2 != 0)
    return np.clip(result, 0.01, 10.0).astype(np.float32)


def calculate_continuum_index(loader, feature_wl, left_wl, right_wl):
    """
    Continuum-removed absorption depth.

    Returns float32 2D array with NaN for invalid (dark) pixels.
    """
    r_left = np.asarray(loader.get_interpolated_band(left_wl), dtype=np.float64)
    r_right = np.asarray(loader.get_interpolated_band(right_wl), dtype=np.float64)
    r_feature = np.asarray(loader.get_interpolated_band(feature_wl), dtype=np.float64)

    weight = (feature_wl - left_wl) / (right_wl - left_wl)
    r_continuum = r_left * (1 - weight) + r_right * weight

    valid = r_continuum > 0.005
    ratio = np.divide(r_feature, r_continuum, out=np.ones_like(r_feature), where=valid)
    depth = 1.0 - ratio
    depth[~valid] = np.nan
    return depth.astype(np.float32)


def calculate_index(loader, index_name, index_defs=None):
    """
    Calculate a predefined spectral index.

    Parameters
    ----------
    loader : LazyHyperspectralData
        Data loader with get_interpolated_band method.
    index_name : str
        Index name (must be in index_defs).
    index_defs : dict or None
        Index definitions. Defaults to INDEX_DEFINITIONS.

    Returns
    -------
    tuple
        (data, idx_def) or (None, None) if unknown.
    """
    if index_defs is None:
        index_defs = INDEX_DEFINITIONS

    if index_name not in index_defs:
        return None, None

    idx_def = index_defs[index_name]
    idx_type = idx_def["type"]

    if idx_type == "nd":
        data = calculate_nd_index(loader, idx_def["b1"], idx_def["b2"])
    elif idx_type == "ratio":
        data = calculate_ratio_index(loader, idx_def["b1"], idx_def["b2"])
    elif idx_type == "continuum":
        data = calculate_continuum_index(
            loader, idx_def["feature"], idx_def["left"], idx_def["right"]
        )
    else:
        return None, None

    return data, idx_def


def robust_percentile(data, percentiles=None,
                      sample_threshold=LARGE_ARRAY_THRESHOLD,
                      sample_size=SAMPLE_SIZE):
    """
    Calculate robust percentiles, sampling if data is large.

    Returns list of percentile values (default: [2nd, 98th]).
    """
    if percentiles is None:
        percentiles = [ROBUST_PERCENTILE_LOW, ROBUST_PERCENTILE_HIGH]

    flat = data.ravel()
    valid = flat[~np.isnan(flat)]

    if len(valid) == 0:
        return [0.0, 1.0]

    if len(valid) > sample_threshold:
        indices = np.random.choice(len(valid), sample_size, replace=False)
        valid = valid[indices]

    return [float(np.percentile(valid, p)) for p in percentiles]
