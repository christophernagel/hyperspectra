"""
HDF5 data loader for HyperspecI indoor hyperspectral dataset.

Implements the same interface as LazyHyperspectralData for seamless
integration with the existing viewer infrastructure.

HyperspecI Dataset Structure (expected):
    Database 1 (D1): 400-1000nm, 10nm intervals, 61 channels, 960x1230px (~288MB)
    Database 2 (D2): 400-1700nm, 10nm intervals, 131 channels, 640x660px (~222MB)

IMPORTANT - Reconstruction Artifacts:
    HyperspecI data is computationally reconstructed by SRNet from 16 broadband
    filter measurements. Known limitations:
    - Spectral smoothing: sharp absorption dips may be flattened
    - Metamerism: materials with different spectra but similar filter responses
      will be reconstructed as nearly identical
    - Boundary artifacts: edge pixels between materials may have hallucinated features

    Treat spectral profiles as APPROXIMATE. Material matching should show
    similarity scores, not definitive identifications.

Memory Strategy:
    Single scenes fit comfortably in RAM (~300MB max). For interactive use,
    load the full cube into numpy rather than using h5py lazy slicing, which
    is ~1000x slower for per-pixel access.

Reference:
    "A broadband hyperspectral image sensor with high spatio-temporal resolution"
    Nature, November 2024
    https://github.com/bianlab/Hyperspectral-imaging-dataset
"""

import numpy as np
import h5py
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LazyH5Data:
    """
    Memory-efficient HDF5 hyperspectral data loader.

    Interface-compatible with LazyHyperspectralData for drop-in use
    with the existing viewer.

    Key differences from AVIRIS NetCDF loader:
    - Data is already reflectance-calibrated (no atmospheric correction needed)
    - Regular 10nm wavelength intervals (no irregular band centers)
    - Different array dimension ordering may need transpose
    - Includes synthesized RGB reference images
    """

    # Expected HyperspecI array layouts (to be confirmed when data arrives)
    # Typical conventions: (channels, height, width) or (height, width, channels)
    EXPECTED_LAYOUTS = ['CHW', 'HWC']

    def __init__(self, filepath, dataset_key=None):
        """
        Initialize loader for an HDF5 hyperspectral file.

        Parameters
        ----------
        filepath : str or Path
            Path to .h5 file
        dataset_key : str, optional
            HDF5 dataset key containing the datacube.
            If None, attempts auto-detection.
        """
        self.filepath = Path(filepath)
        self.h5 = h5py.File(filepath, 'r')

        # Auto-detect datacube location
        self.dataset_key = dataset_key or self._find_datacube()
        self.data_var = self.h5[self.dataset_key]

        # Determine array layout and dimensions
        self._detect_layout()

        # Build wavelength array (regular 10nm intervals)
        self.wavelengths = self._build_wavelength_array()

        # HyperspecI data is reflectance-calibrated
        self.data_type = 'reflectance'

        # Band cache (same pattern as NetCDF loader)
        self._band_cache = {}
        self._cache_max_size = 50
        self._preloaded_cube = None  # Full cube in memory for fast access

        # Look for RGB reference image
        self.rgb_reference = self._find_rgb_reference()

        logger.info(f"Loaded H5: {self.n_bands} bands ({self.wl_min:.0f}-{self.wl_max:.0f}nm), "
                    f"{self.n_rows}x{self.n_cols} pixels")
        if self.rgb_reference is not None:
            logger.info("  RGB reference image found")

    def _find_datacube(self):
        """
        Auto-detect the datacube dataset within the H5 file.

        Returns the key of the largest 3D array, assuming it's the datacube.
        """
        candidates = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
                candidates.append((name, obj.shape, np.prod(obj.shape)))

        self.h5.visititems(visitor)

        if not candidates:
            raise ValueError("No 3D datasets found in H5 file")

        # Sort by size, largest is likely the datacube
        candidates.sort(key=lambda x: x[2], reverse=True)

        logger.debug(f"Datacube candidates: {[(c[0], c[1]) for c in candidates]}")
        return candidates[0][0]

    def _detect_layout(self):
        """
        Detect array dimension ordering and set n_bands, n_rows, n_cols.

        HyperspecI D1: 61 bands, 960x1230 pixels
        HyperspecI D2: 131 bands, 640x660 pixels
        """
        shape = self.data_var.shape

        # Known HyperspecI dimensions for validation
        known_band_counts = {61, 131}
        known_spatial = {(960, 1230), (1230, 960), (640, 660), (660, 640)}

        # Try to match known dimensions
        if shape[0] in known_band_counts:
            # CHW layout: (channels, height, width)
            self._layout = 'CHW'
            self.n_bands, self.n_rows, self.n_cols = shape
        elif shape[2] in known_band_counts:
            # HWC layout: (height, width, channels)
            self._layout = 'HWC'
            self.n_rows, self.n_cols, self.n_bands = shape
        else:
            # Fall back to heuristic: smallest dimension is likely channels
            min_dim = np.argmin(shape)
            if min_dim == 0:
                self._layout = 'CHW'
                self.n_bands, self.n_rows, self.n_cols = shape
            else:
                self._layout = 'HWC'
                self.n_rows, self.n_cols, self.n_bands = shape

            logger.warning(f"Unknown array layout {shape}, guessing {self._layout}")

        logger.debug(f"Layout: {self._layout}, shape: {shape} -> "
                     f"bands={self.n_bands}, rows={self.n_rows}, cols={self.n_cols}")

    def _build_wavelength_array(self):
        """
        Build wavelength array from metadata or known HyperspecI specs.

        HyperspecI uses regular 10nm intervals:
            D1: 400-1000nm (61 bands)
            D2: 400-1700nm (131 bands)
        """
        # Try to read from file metadata
        wl_keys = ['wavelength', 'wavelengths', 'wl', 'lambda', 'bands']
        for key in wl_keys:
            if key in self.h5:
                return np.array(self.h5[key])
            if key in self.h5.attrs:
                return np.array(self.h5.attrs[key])

        # Fall back to HyperspecI specs
        if self.n_bands == 61:
            # D1: 400-1000nm
            return np.linspace(400, 1000, 61)
        elif self.n_bands == 131:
            # D2: 400-1700nm
            return np.linspace(400, 1700, 131)
        else:
            # Generic fallback: assume VNIR range
            logger.warning(f"Unknown band count {self.n_bands}, assuming 400-1000nm")
            return np.linspace(400, 1000, self.n_bands)

    def _find_rgb_reference(self):
        """Look for synthesized RGB reference image in the H5 file."""
        rgb_keys = ['rgb', 'RGB', 'reference', 'reference_rgb', 'image']

        for key in rgb_keys:
            if key in self.h5:
                arr = np.array(self.h5[key])
                # RGB should be 3-channel 2D image
                if len(arr.shape) == 3 and (arr.shape[-1] == 3 or arr.shape[0] == 3):
                    return arr

        return None

    @property
    def wl_min(self):
        return float(self.wavelengths.min())

    @property
    def wl_max(self):
        return float(self.wavelengths.max())

    @property
    def shape(self):
        return (self.n_bands, self.n_rows, self.n_cols)

    # -------------------------------------------------------------------------
    # Interface methods (matching LazyHyperspectralData)
    # -------------------------------------------------------------------------

    def preload(self):
        """
        Load entire datacube into memory for fast interactive access.

        HyperspecI files are small enough (~300MB max) to fit in RAM.
        This trades memory for speed: sub-millisecond pixel access vs ~100ms.
        """
        if self._preloaded_cube is not None:
            logger.info("Cube already preloaded")
            return

        logger.info(f"Preloading full cube ({self.n_bands}x{self.n_rows}x{self.n_cols})...")

        # Load and ensure CHW layout
        if self._layout == 'CHW':
            self._preloaded_cube = np.array(self.data_var[:, :, :], dtype=np.float32)
        else:  # HWC -> CHW
            self._preloaded_cube = np.array(self.data_var[:, :, :], dtype=np.float32).transpose(2, 0, 1)

        mem_mb = self._preloaded_cube.nbytes / 1e6
        logger.info(f"Preloaded {mem_mb:.1f} MB into memory")

        # Clear band cache since we have full cube now
        self._band_cache.clear()

    def is_preloaded(self):
        """Check if cube is preloaded into memory."""
        return self._preloaded_cube is not None

    def get_band(self, band_idx):
        """Get a single band by index (with caching)."""
        # Fast path: preloaded cube
        if self._preloaded_cube is not None:
            return self._preloaded_cube[band_idx]

        if band_idx in self._band_cache:
            return self._band_cache[band_idx]

        # Load based on layout
        if self._layout == 'CHW':
            band_data = self.data_var[band_idx, :, :]
        else:  # HWC
            band_data = self.data_var[:, :, band_idx]

        band_data = np.array(band_data, dtype=np.float32)

        # Cache management
        if len(self._band_cache) >= self._cache_max_size:
            del self._band_cache[next(iter(self._band_cache))]

        self._band_cache[band_idx] = band_data
        return band_data

    def find_band(self, target_wavelength):
        """Find the nearest band index to a target wavelength."""
        return int(np.argmin(np.abs(self.wavelengths - target_wavelength)))

    def get_interpolated_band(self, target_wavelength, method='linear'):
        """
        Get reflectance at exact wavelength via interpolation.

        Same interface as NetCDF loader for compatibility.
        """
        if method == 'nearest':
            return self.get_band(self.find_band(target_wavelength))

        # Linear interpolation between adjacent bands
        idx = np.searchsorted(self.wavelengths, target_wavelength)

        if idx == 0:
            return self.get_band(0)
        if idx >= len(self.wavelengths):
            return self.get_band(len(self.wavelengths) - 1)

        wl_before = self.wavelengths[idx - 1]
        wl_after = self.wavelengths[idx]
        band_before = self.get_band(idx - 1).astype(np.float64)
        band_after = self.get_band(idx).astype(np.float64)

        fraction = (target_wavelength - wl_before) / (wl_after - wl_before)
        interpolated = band_before * (1 - fraction) + band_after * fraction

        return interpolated.astype(np.float32)

    def get_wavelength_offset(self, target_wavelength):
        """Get offset between target and nearest band wavelength."""
        band_idx = self.find_band(target_wavelength)
        actual_wl = self.wavelengths[band_idx]
        return float(actual_wl - target_wavelength)

    def get_cube_region(self, row_slice, col_slice):
        """Get a spatial subset of the full cube."""
        # Fast path: preloaded cube (always CHW)
        if self._preloaded_cube is not None:
            return self._preloaded_cube[:, row_slice, col_slice]

        if self._layout == 'CHW':
            return np.array(self.data_var[:, row_slice, col_slice], dtype=np.float32)
        else:  # HWC
            region = np.array(self.data_var[row_slice, col_slice, :], dtype=np.float32)
            # Transpose to (bands, rows, cols) for consistency
            return np.transpose(region, (2, 0, 1))

    def get_full_cube(self):
        """
        Load entire datacube into memory for fast repeated access.

        Call this once before doing many pixel lookups (e.g., material matching).
        Subsequent get_cube_region() calls will use the in-memory array.

        Returns
        -------
        np.ndarray
            Full datacube as float32, shape (bands, rows, cols)
        """
        if self._preloaded_cube is None:
            logger.info(f"Preloading full cube into memory...")
            if self._layout == 'CHW':
                self._preloaded_cube = np.array(self.data_var[:], dtype=np.float32)
            else:
                cube = np.array(self.data_var[:], dtype=np.float32)
                self._preloaded_cube = np.transpose(cube, (2, 0, 1))
            size_mb = self._preloaded_cube.nbytes / 1e6
            logger.info(f"Loaded {size_mb:.0f} MB into RAM")
        return self._preloaded_cube

    def release_cube(self):
        """Release preloaded cube from memory."""
        if self._preloaded_cube is not None:
            self._preloaded_cube = None
            logger.info("Released preloaded cube from memory")

    def estimate_snr(self, wavelength):
        """
        Estimate signal-to-noise ratio at a given wavelength.

        For calibrated reflectance data, uses local variance method.
        """
        band_idx = self.find_band(wavelength)
        band = self.get_band(band_idx)

        # Use central region
        h, w = band.shape
        center = band[h // 4:3 * h // 4, w // 4:3 * w // 4]

        center_clean = center.astype(np.float64)
        nan_mask = np.isnan(center_clean)
        if nan_mask.any():
            center_clean[nan_mask] = np.nanmedian(center_clean)

        # Local SNR estimation
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(center_clean, size=5)
        variance = uniform_filter((center_clean - local_mean) ** 2, size=5)
        local_std = np.sqrt(np.maximum(variance, 0))

        valid = local_std > 1e-10
        if not np.any(valid):
            return 100.0

        snr = np.median(local_mean[valid] / local_std[valid])
        return float(np.clip(snr, 10, 1000))

    def clear_cache(self):
        """Clear the band cache."""
        self._band_cache.clear()
        logger.info("Band cache cleared")

    def set_cache_size(self, max_size):
        """Set maximum cache size."""
        self._cache_max_size = max_size
        while len(self._band_cache) > self._cache_max_size:
            del self._band_cache[next(iter(self._band_cache))]

    def get_cache_info(self):
        """Return cache statistics."""
        return {
            'current_size': len(self._band_cache),
            'max_size': self._cache_max_size,
            'cached_bands': list(self._band_cache.keys()),
            'memory_mb': sum(b.nbytes for b in self._band_cache.values()) / 1e6,
            'preloaded': self._preloaded_cube is not None,
        }

    def close(self):
        """Close the HDF5 file and clear cache."""
        self.clear_cache()
        self.h5.close()

    # -------------------------------------------------------------------------
    # Wavelength range validation (critical for index calculations)
    # -------------------------------------------------------------------------

    def validate_wavelength(self, wavelength):
        """Check if wavelength is within the loaded datacube's range."""
        return self.wl_min <= wavelength <= self.wl_max

    def validate_index_wavelengths(self, wavelengths):
        """
        Validate that all wavelengths required for an index are within range.

        Parameters
        ----------
        wavelengths : list or tuple
            Wavelengths required by the index (e.g., [1510, 1450, 1550])

        Returns
        -------
        dict with:
            'valid': bool - True if all wavelengths are within range
            'out_of_range': list - wavelengths outside the datacube range
            'message': str - human-readable status
        """
        out_of_range = [wl for wl in wavelengths if not self.validate_wavelength(wl)]

        if not out_of_range:
            return {
                'valid': True,
                'out_of_range': [],
                'message': f"All wavelengths within range ({self.wl_min:.0f}-{self.wl_max:.0f}nm)"
            }
        else:
            return {
                'valid': False,
                'out_of_range': out_of_range,
                'message': (f"Wavelength(s) {out_of_range} outside datacube range "
                           f"({self.wl_min:.0f}-{self.wl_max:.0f}nm)")
            }

    def get_database_type(self):
        """
        Identify HyperspecI database type from spectral range.

        Returns
        -------
        str: 'D1' (400-1000nm), 'D2' (400-1700nm), or 'unknown'
        """
        if self.n_bands == 61 and abs(self.wl_max - 1000) < 50:
            return 'D1'
        elif self.n_bands == 131 and abs(self.wl_max - 1700) < 50:
            return 'D2'
        else:
            return 'unknown'

    def get_available_features(self):
        """
        List which spectral features are available in this datacube.

        Based on the wavelength range of the loaded data.
        """
        features = {
            'chlorophyll': self.validate_wavelength(680) and self.validate_wavelength(750),
            'moisture_vnir': self.validate_wavelength(970),
            'moisture_swir': self.validate_wavelength(1430),
            'protein': self.validate_wavelength(1500),
            'lipid_2nd_ot': self.validate_wavelength(1210),
            'lipid_1st_ot': self.validate_wavelength(1730),  # Usually outside D2!
            'cellulose': self.validate_wavelength(1490),
            'aromatic': self.validate_wavelength(1670),
        }
        return features


# -------------------------------------------------------------------------
# H5 file exploration utility
# -------------------------------------------------------------------------

def explore_h5(filepath):
    """
    Print the complete structure of an HDF5 file.

    Use this to understand the internal layout before trusting auto-detection.

    Parameters
    ----------
    filepath : str or Path
        Path to .h5 file

    Returns
    -------
    dict with file structure information
    """
    filepath = Path(filepath)
    structure = {
        'filepath': str(filepath),
        'datasets': [],
        'groups': [],
        'attributes': {}
    }

    with h5py.File(filepath, 'r') as f:
        # Top-level attributes
        structure['attributes'] = dict(f.attrs)

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                info = {
                    'name': name,
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'size_mb': np.prod(obj.shape) * obj.dtype.itemsize / 1e6,
                    'compression': obj.compression,
                    'chunks': obj.chunks,
                }
                structure['datasets'].append(info)
                print(f"  Dataset: {name}")
                print(f"    shape={obj.shape}, dtype={obj.dtype}, size={info['size_mb']:.1f}MB")
                if obj.compression:
                    print(f"    compression={obj.compression}, chunks={obj.chunks}")
            elif isinstance(obj, h5py.Group):
                structure['groups'].append(name)
                print(f"  Group: {name}/")

        print(f"\nHDF5 Structure: {filepath.name}")
        print("=" * 60)
        f.visititems(visitor)

        # Identify likely datacube
        datacubes = [d for d in structure['datasets'] if len(d['shape']) == 3]
        if datacubes:
            largest = max(datacubes, key=lambda x: np.prod(x['shape']))
            print(f"\nLikely datacube: {largest['name']} {largest['shape']}")

            # Guess database type
            shape = largest['shape']
            if 61 in shape:
                print("  -> HyperspecI D1 (400-1000nm, 61 bands)")
            elif 131 in shape:
                print("  -> HyperspecI D2 (400-1700nm, 131 bands)")

    return structure


# -------------------------------------------------------------------------
# Factory function for unified loading
# -------------------------------------------------------------------------

def load_hyperspectral(filepath):
    """
    Load hyperspectral data from any supported format.

    Auto-detects file type and returns appropriate loader.

    Parameters
    ----------
    filepath : str or Path
        Path to data file (.nc, .h5, .hdf5)

    Returns
    -------
    loader : LazyHyperspectralData or LazyH5Data
        Data loader with unified interface
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix == '.nc':
        from .data_loader import LazyHyperspectralData
        return LazyHyperspectralData(filepath)
    elif suffix in ('.h5', '.hdf5', '.hdf'):
        return LazyH5Data(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
