"""
Illumination Normalization for Indoor Hyperspectral Scenes

Indoor scenes have spatially varying illumination from multiple light sources
(windows, lamps, overhead lighting). This causes identical materials in different
parts of the scene to produce different spectral profiles.

The observed signal at each pixel is:
    L(λ) = R(λ) × I(λ) × cos(θ)

Where:
    L(λ) = observed radiance/reflectance
    R(λ) = true surface reflectance
    I(λ) = illumination spectrum (varies spatially)
    θ = angle between surface normal and illumination

Without correction, spectral matching becomes "material + lighting" matching.

This module provides:
1. Flat-field normalization using a white/neutral reference
2. Automatic white reference detection
3. Per-pixel illumination estimation (experimental)

Reference:
    Nascimento et al. (2005) "Statistics of spatial cone-excitation ratios
    in natural scenes" JOSA A 22(10): 2197-2208
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class IlluminationNormalizer:
    """
    Normalize illumination variations in indoor hyperspectral scenes.

    The simplest approach is flat-field correction using a known white reference.
    More advanced methods attempt to estimate and separate illumination from
    reflectance.
    """

    def __init__(self, datacube: np.ndarray, wavelengths: np.ndarray):
        """
        Initialize normalizer with a datacube.

        Parameters
        ----------
        datacube : np.ndarray
            Shape (bands, rows, cols) - the hyperspectral image
        wavelengths : np.ndarray
            Wavelength values in nm for each band
        """
        self.datacube = datacube
        self.wavelengths = wavelengths
        self.n_bands, self.n_rows, self.n_cols = datacube.shape

        self._white_reference = None
        self._normalized_cube = None

    # -------------------------------------------------------------------------
    # Flat-field normalization with white reference
    # -------------------------------------------------------------------------

    def set_white_reference(self, reference: np.ndarray) -> None:
        """
        Set the white reference spectrum for normalization.

        Parameters
        ----------
        reference : np.ndarray
            Shape (bands,) - spectrum of a known white/neutral surface
            in the scene under the scene illumination
        """
        if reference.shape[0] != self.n_bands:
            raise ValueError(f"Reference must have {self.n_bands} bands")

        # Avoid division by zero
        reference = np.maximum(reference, 1e-6)
        self._white_reference = reference

    def set_white_reference_from_roi(self, row_slice: slice, col_slice: slice) -> np.ndarray:
        """
        Set white reference from a region of interest in the scene.

        The ROI should contain a known white/neutral surface (white paper,
        white wall, gray card, etc.).

        Parameters
        ----------
        row_slice, col_slice : slice
            Region containing the white reference

        Returns
        -------
        np.ndarray
            The computed white reference spectrum
        """
        roi = self.datacube[:, row_slice, col_slice]
        reference = np.nanmean(roi, axis=(1, 2))

        logger.info(f"White reference from ROI: mean={reference.mean():.4f}, "
                   f"shape=({row_slice.stop - row_slice.start}, {col_slice.stop - col_slice.start})")

        self.set_white_reference(reference)
        return reference

    def normalize_flatfield(self) -> np.ndarray:
        """
        Apply flat-field normalization using the white reference.

        Each pixel spectrum is divided by the white reference, giving
        relative reflectance that's independent of the illumination spectrum.

        Returns
        -------
        np.ndarray
            Normalized datacube, same shape as input
        """
        if self._white_reference is None:
            raise ValueError("Set white reference first with set_white_reference()")

        # Divide each pixel by white reference
        # Shape: (bands, rows, cols) / (bands, 1, 1)
        ref = self._white_reference[:, np.newaxis, np.newaxis]
        normalized = self.datacube / ref

        # Clip to reasonable range (artifacts can cause values > 1)
        normalized = np.clip(normalized, 0, 2)

        self._normalized_cube = normalized
        logger.info("Applied flat-field normalization")
        return normalized

    # -------------------------------------------------------------------------
    # Automatic white reference detection
    # -------------------------------------------------------------------------

    def detect_white_reference(self, percentile: float = 98) -> Tuple[np.ndarray, np.ndarray]:
        """
        Automatically detect likely white/bright reference pixels.

        Uses the assumption that the brightest pixels in the scene are
        likely white or near-white surfaces.

        Parameters
        ----------
        percentile : float
            Percentile threshold for "bright" pixels (default 98)

        Returns
        -------
        Tuple of:
            - reference spectrum (bands,)
            - binary mask of detected white pixels (rows, cols)
        """
        # Compute broadband brightness (average across all bands)
        brightness = np.nanmean(self.datacube, axis=0)

        # Find threshold
        threshold = np.nanpercentile(brightness, percentile)

        # Create mask of bright pixels
        white_mask = brightness >= threshold

        # Compute mean spectrum of bright pixels
        white_pixels = self.datacube[:, white_mask]
        reference = np.nanmean(white_pixels, axis=1)

        n_white = np.sum(white_mask)
        logger.info(f"Detected {n_white} white reference pixels "
                   f"({100 * n_white / (self.n_rows * self.n_cols):.1f}% of image)")

        return reference, white_mask

    def auto_normalize(self, percentile: float = 98) -> np.ndarray:
        """
        Automatically detect white reference and normalize.

        Convenience method that combines detection and normalization.

        Parameters
        ----------
        percentile : float
            Percentile for white pixel detection

        Returns
        -------
        np.ndarray
            Normalized datacube
        """
        reference, mask = self.detect_white_reference(percentile)
        self.set_white_reference(reference)
        return self.normalize_flatfield()

    # -------------------------------------------------------------------------
    # Experimental: per-pixel illumination estimation
    # -------------------------------------------------------------------------

    def estimate_illumination_map(self, reference_white: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimate relative illumination intensity at each pixel.

        This is experimental. Uses the assumption that true white surfaces
        should have flat spectra, so deviations indicate illumination color.

        For scenes without known white references, uses the brightest pixels
        as an approximation.

        Parameters
        ----------
        reference_white : np.ndarray, optional
            Expected spectrum of a perfect white surface under neutral light.
            If None, uses a flat spectrum.

        Returns
        -------
        np.ndarray
            Shape (rows, cols) - relative illumination intensity
        """
        if reference_white is None:
            # Assume ideal white = flat spectrum at the 95th percentile level
            reference_white = np.ones(self.n_bands)
            reference_white *= np.nanpercentile(self.datacube, 95)

        # Estimate illumination as ratio of observed to expected
        # Average across wavelengths for stability
        expected = reference_white[:, np.newaxis, np.newaxis]
        ratio = self.datacube / (expected + 1e-6)

        # Illumination map = median ratio across wavelengths
        illum_map = np.nanmedian(ratio, axis=0)

        return illum_map

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def get_normalized_spectrum(self, row: int, col: int) -> np.ndarray:
        """Get normalized spectrum at a single pixel."""
        if self._normalized_cube is None:
            raise ValueError("Run normalization first")
        return self._normalized_cube[:, row, col]

    def get_original_spectrum(self, row: int, col: int) -> np.ndarray:
        """Get original (unnormalized) spectrum at a single pixel."""
        return self.datacube[:, row, col]

    def compare_spectra(self, row: int, col: int) -> dict:
        """
        Compare original and normalized spectra at a pixel.

        Returns dict with both spectra and statistics.
        """
        original = self.get_original_spectrum(row, col)

        result = {
            'wavelengths': self.wavelengths,
            'original': original,
            'normalized': None,
            'illumination_factor': None,
        }

        if self._normalized_cube is not None:
            normalized = self.get_normalized_spectrum(row, col)
            result['normalized'] = normalized

            # Illumination factor = original / normalized (mean across bands)
            with np.errstate(divide='ignore', invalid='ignore'):
                factor = np.nanmean(original / (normalized + 1e-6))
            result['illumination_factor'] = float(factor)

        return result


def quick_normalize(datacube: np.ndarray, wavelengths: np.ndarray,
                    white_roi: Optional[Tuple[slice, slice]] = None) -> np.ndarray:
    """
    Quick illumination normalization for a datacube.

    Parameters
    ----------
    datacube : np.ndarray
        Shape (bands, rows, cols)
    wavelengths : np.ndarray
        Wavelength values in nm
    white_roi : tuple of slices, optional
        (row_slice, col_slice) for white reference region.
        If None, auto-detects from brightest pixels.

    Returns
    -------
    np.ndarray
        Normalized datacube
    """
    normalizer = IlluminationNormalizer(datacube, wavelengths)

    if white_roi is not None:
        normalizer.set_white_reference_from_roi(white_roi[0], white_roi[1])
        return normalizer.normalize_flatfield()
    else:
        return normalizer.auto_normalize()
