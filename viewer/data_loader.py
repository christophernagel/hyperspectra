"""
Memory-efficient hyperspectral data loader with caching and interpolation.
"""

import numpy as np
import netCDF4 as nc
from pathlib import Path
import logging

from .constants import ATMOSPHERIC_BANDS, DEFAULT_CACHE_SIZE

logger = logging.getLogger(__name__)


class LazyHyperspectralData:
    """
    Memory-efficient hyperspectral data loader with caching and interpolation.

    Features:
    - LRU-style band caching to minimize memory usage
    - Sub-band wavelength precision via interpolation
    - Atmospheric band detection and warnings
    - Data quality (SNR) estimation
    """

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.ds = nc.Dataset(filepath)

        # Find reflectance or radiance data
        if 'reflectance' in self.ds.groups:
            self.data_group = self.ds.groups['reflectance']
            self.data_var = self.data_group.variables['reflectance']
            self.data_type = 'reflectance'
        elif 'radiance' in self.ds.groups:
            self.data_group = self.ds.groups['radiance']
            self.data_var = self.data_group.variables['radiance']
            self.data_type = 'radiance'
        elif 'reflectance' in self.ds.variables:
            self.data_group = self.ds
            self.data_var = self.ds.variables['reflectance']
            self.data_type = 'reflectance'
        elif 'radiance' in self.ds.variables:
            self.data_group = self.ds
            self.data_var = self.ds.variables['radiance']
            self.data_type = 'radiance'
        else:
            raise ValueError("Could not find radiance or reflectance data")

        self.wavelengths = self.data_group.variables['wavelength'][:]
        self.shape = self.data_var.shape
        self.n_bands, self.n_rows, self.n_cols = self.shape
        self._band_cache = {}
        self._cache_max_size = DEFAULT_CACHE_SIZE

        logger.info(f"Loaded: {self.n_bands} bands, {self.n_rows}x{self.n_cols} pixels, type={self.data_type}")

    @property
    def wl_min(self):
        return float(self.wavelengths.min())

    @property
    def wl_max(self):
        return float(self.wavelengths.max())

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def clear_cache(self):
        """Explicitly clear the band cache to free memory."""
        self._band_cache.clear()
        logger.info("Band cache cleared")

    def set_cache_size(self, max_size):
        """Set maximum cache size. Evicts oldest entries if reducing size."""
        self._cache_max_size = max_size
        while len(self._band_cache) > self._cache_max_size:
            del self._band_cache[next(iter(self._band_cache))]
        logger.info(f"Cache size set to {max_size}")

    def get_cache_info(self):
        """Return cache statistics."""
        return {
            'current_size': len(self._band_cache),
            'max_size': self._cache_max_size,
            'cached_bands': list(self._band_cache.keys()),
            'memory_mb': sum(b.nbytes for b in self._band_cache.values()) / 1e6
        }

    # -------------------------------------------------------------------------
    # Band Access
    # -------------------------------------------------------------------------

    def get_band(self, band_idx):
        """Get a single band by index (with caching)."""
        if band_idx in self._band_cache:
            return self._band_cache[band_idx]

        band_data = self.data_var[band_idx, :, :].astype(np.float32)

        if len(self._band_cache) >= self._cache_max_size:
            del self._band_cache[next(iter(self._band_cache))]

        self._band_cache[band_idx] = band_data
        return band_data

    def get_cube_region(self, row_slice, col_slice):
        """Get a spatial subset of the full cube."""
        return self.data_var[:, row_slice, col_slice].astype(np.float32)

    def find_band(self, target_wavelength):
        """Find the nearest band index to a target wavelength."""
        return int(np.argmin(np.abs(self.wavelengths - target_wavelength)))

    def get_wavelength_offset(self, target_wavelength):
        """Get the offset between target and actual nearest band wavelength."""
        band_idx = self.find_band(target_wavelength)
        actual_wl = self.wavelengths[band_idx]
        return float(actual_wl - target_wavelength)

    # -------------------------------------------------------------------------
    # Band Interpolation (for exact wavelength retrieval)
    # -------------------------------------------------------------------------

    def get_interpolated_band(self, target_wavelength, method='linear'):
        """
        Get reflectance/radiance at exact wavelength via interpolation.

        This addresses the band selection precision issue where nearest-band
        selection can introduce +/-15nm error. Linear interpolation between
        adjacent bands provides sub-band wavelength precision.

        Args:
            target_wavelength: Exact wavelength in nm
            method: 'linear' or 'nearest'

        Returns:
            2D array of interpolated values at the target wavelength
        """
        if method == 'nearest':
            return self.get_band(self.find_band(target_wavelength))

        # Find bracketing bands
        idx = np.searchsorted(self.wavelengths, target_wavelength)

        # Handle edge cases
        if idx == 0:
            return self.get_band(0)
        if idx >= len(self.wavelengths):
            return self.get_band(len(self.wavelengths) - 1)

        # Get adjacent bands
        wl_before = self.wavelengths[idx - 1]
        wl_after = self.wavelengths[idx]
        band_before = self.get_band(idx - 1).astype(np.float64)
        band_after = self.get_band(idx).astype(np.float64)

        # Linear interpolation
        fraction = (target_wavelength - wl_before) / (wl_after - wl_before)
        interpolated = band_before * (1 - fraction) + band_after * fraction

        return interpolated.astype(np.float32)

    def get_multiple_interpolated_bands(self, wavelengths):
        """
        Get multiple interpolated bands efficiently in a single pass.

        This minimizes band loading by:
        1. Sorting wavelengths to access bands sequentially
        2. Reusing bands that are already in cache
        3. Loading each unique band only once

        Args:
            wavelengths: List/array of wavelengths to interpolate

        Returns:
            Dict mapping wavelength -> interpolated 2D array
        """
        wavelengths = np.asarray(wavelengths)
        results = {}

        # Find all unique band indices needed
        needed_bands = set()
        for wl in wavelengths:
            idx = np.searchsorted(self.wavelengths, wl)
            if idx == 0:
                needed_bands.add(0)
            elif idx >= len(self.wavelengths):
                needed_bands.add(len(self.wavelengths) - 1)
            else:
                needed_bands.add(idx - 1)
                needed_bands.add(idx)

        # Pre-load all needed bands into cache
        for band_idx in sorted(needed_bands):
            if band_idx not in self._band_cache:
                self.get_band(band_idx)

        # Now interpolate using cached bands
        for wl in wavelengths:
            idx = np.searchsorted(self.wavelengths, wl)

            if idx == 0:
                results[float(wl)] = self._band_cache[0].astype(np.float32)
            elif idx >= len(self.wavelengths):
                results[float(wl)] = self._band_cache[len(self.wavelengths) - 1].astype(np.float32)
            else:
                wl_before = self.wavelengths[idx - 1]
                wl_after = self.wavelengths[idx]
                fraction = (wl - wl_before) / (wl_after - wl_before)

                band_before = self._band_cache[idx - 1].astype(np.float64)
                band_after = self._band_cache[idx].astype(np.float64)

                interpolated = band_before * (1 - fraction) + band_after * fraction
                results[float(wl)] = interpolated.astype(np.float32)

        return results

    # -------------------------------------------------------------------------
    # Atmospheric Band Checking
    # -------------------------------------------------------------------------

    def is_in_atmospheric_band(self, wavelength):
        """
        Check if wavelength falls within known atmospheric absorption bands.

        Returns:
            Tuple of (is_in_band: bool, band_name: str or None, severity: str)
        """
        for band_name, (low, high) in ATMOSPHERIC_BANDS.items():
            if low <= wavelength <= high:
                if 'water_vapor' in band_name:
                    severity = 'severe'
                elif 'oxygen' in band_name:
                    severity = 'moderate'
                else:
                    severity = 'moderate'
                return True, band_name, severity
        return False, None, 'none'

    def check_wavelengths_for_atmospheric(self, wavelengths):
        """
        Check multiple wavelengths for atmospheric absorption issues.

        Returns:
            List of warnings for each affected wavelength
        """
        warnings = []
        for wl in wavelengths:
            in_band, band_name, severity = self.is_in_atmospheric_band(wl)
            if in_band:
                warnings.append({
                    'wavelength': wl,
                    'band': band_name,
                    'severity': severity,
                    'message': f"{wl:.0f}nm in {band_name.replace('_', ' ')} absorption"
                })
        return warnings

    # -------------------------------------------------------------------------
    # Data Quality Estimation
    # -------------------------------------------------------------------------

    def estimate_snr(self, wavelength):
        """
        Estimate signal-to-noise ratio at a given wavelength.

        Uses a simplified approach: SNR ~ mean / std in homogeneous regions.
        For AVIRIS-3, typical SNR is 300-500:1 in VNIR, 100-300:1 in SWIR.
        """
        band_idx = self.find_band(wavelength)
        band = self.get_band(band_idx)

        # Use central region (likely more homogeneous)
        h, w = band.shape
        center = band[h // 4:3 * h // 4, w // 4:3 * w // 4]

        # Replace NaN with local median to avoid propagation issues
        center_clean = center.astype(np.float64)
        nan_mask = np.isnan(center_clean)
        if nan_mask.any():
            center_clean[nan_mask] = np.nanmedian(center_clean)

        # Local SNR estimation using small windows
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(center_clean, size=5)
        variance = uniform_filter((center_clean - local_mean) ** 2, size=5)
        # Ensure non-negative before sqrt (numerical precision issues)
        local_std = np.sqrt(np.maximum(variance, 0))

        # Avoid division by zero
        valid = local_std > 1e-10
        if not np.any(valid):
            return 100.0  # Default assumption

        snr = np.median(local_mean[valid] / local_std[valid])
        return float(np.clip(snr, 10, 1000))

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_wavelength(self, wavelength):
        """Check if wavelength is within sensor range."""
        return self.wl_min <= wavelength <= self.wl_max

    def validate_index_calculation(self, b1_wl, b2_wl, index_type='ratio',
                                    block_atmospheric=True):
        """
        Comprehensive validation before index calculation.

        Args:
            b1_wl, b2_wl: Wavelengths for the index
            index_type: 'ratio' or 'nd' (normalized difference)
            block_atmospheric: If True, return error for severe atmospheric bands

        Returns:
            dict with 'valid': bool, 'warnings': list, 'errors': list
        """
        result = {'valid': True, 'warnings': [], 'errors': []}

        # Check wavelength range
        for wl, name in [(b1_wl, 'b1'), (b2_wl, 'b2')]:
            if not self.validate_wavelength(wl):
                result['errors'].append(
                    f"{name}={wl}nm outside sensor range ({self.wl_min:.0f}-{self.wl_max:.0f}nm)"
                )
                result['valid'] = False

        if not result['valid']:
            return result

        # Check atmospheric bands
        atm_warnings = self.check_wavelengths_for_atmospheric([b1_wl, b2_wl])
        for w in atm_warnings:
            if w['severity'] == 'severe' and block_atmospheric:
                result['errors'].append(f"BLOCKED: {w['message']} - data unreliable")
                result['valid'] = False
            else:
                result['warnings'].append(w['message'])

        # Check band offset (precision)
        for wl, name in [(b1_wl, 'b1'), (b2_wl, 'b2')]:
            offset = abs(self.get_wavelength_offset(wl))
            if offset > 10:  # >10nm is significant
                result['warnings'].append(
                    f"{name}: requested {wl}nm, nearest band is {offset:.1f}nm away"
                )

        # Check SNR for SWIR bands
        for wl, name in [(b1_wl, 'b1'), (b2_wl, 'b2')]:
            if wl > 2000:  # SWIR region has lower SNR
                snr = self.estimate_snr(wl)
                if snr < 50:
                    result['warnings'].append(f"{name}: low SNR ({snr:.0f}:1) at {wl}nm")

        return result

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def get_full_cube(self):
        """Load entire data cube (warning: may be large)."""
        return self.data_var[:].astype(np.float32)

    def close(self):
        """Close the NetCDF file and clear cache."""
        self.clear_cache()
        self.ds.close()
