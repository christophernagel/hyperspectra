"""
Computation engine for the web viewer.

Wraps the existing data loader, index calculations, and atmospheric
correction code. All methods are pure Python/numpy — no UI dependencies.
"""

import numpy as np
import logging
from pathlib import Path

from aviris_tools.viewer.data_loader import load_hyperspectral
from aviris_tools.compute import (
    normalize_rgb,
    calculate_index as _compute_index,
    robust_percentile as _robust_percentile,
)

logger = logging.getLogger(__name__)


class WebEngine:
    """Pure computation engine for the web viewer."""

    def __init__(self):
        self.data_loader = None
        self.filepath = None
        self.lat_axis = None  # 1D lat values (one per row)
        self.lon_axis = None  # 1D lon values (one per col)
        self._has_coords = False
        self._matcher = None  # cached MaterialMatcher

    def load_file(self, filepath):
        """Load a hyperspectral file. Returns metadata dict."""
        if self.data_loader is not None:
            self.data_loader.close()

        self.data_loader = load_hyperspectral(filepath)
        self.filepath = str(filepath)
        self._matcher = None  # invalidate cached matcher on new file
        self._load_coordinates()

        logger.info(
            f"Loaded: {self.data_loader.n_bands} bands, "
            f"{self.data_loader.n_rows}x{self.data_loader.n_cols}, "
            f"type={self.data_loader.data_type}"
        )

        return self.get_file_info()

    def _load_coordinates(self):
        """Extract lat/lon axes from the NetCDF file for axis labels."""
        self.lat_axis = None
        self.lon_axis = None
        self._has_coords = False

        ds = self.data_loader.ds
        n_rows = self.data_loader.n_rows
        n_cols = self.data_loader.n_cols

        # Find lat/lon variables (root or in groups)
        lat_2d = lon_2d = None
        lat_names = ["lat", "latitude", "Latitude", "LAT"]
        lon_names = ["lon", "longitude", "Longitude", "LON"]

        sources = [ds.variables]
        for grp in ds.groups.values():
            sources.append(grp.variables)

        for var_dict in sources:
            if lat_2d is not None:
                break
            for name in lat_names:
                if name in var_dict:
                    data = np.ma.filled(var_dict[name][:], np.nan)
                    if data.shape == (n_rows, n_cols):
                        lat_2d = data
                    break
            for name in lon_names:
                if name in var_dict:
                    data = np.ma.filled(var_dict[name][:], np.nan)
                    if data.shape == (n_rows, n_cols):
                        lon_2d = data
                    break

        if lat_2d is None or lon_2d is None:
            logger.info("No lat/lon coordinates found, using pixel indices")
            return

        # Use center column for lat axis, center row for lon axis
        mid_col = n_cols // 2
        mid_row = n_rows // 2
        self.lat_axis = lat_2d[:, mid_col].astype(np.float64)
        self.lon_axis = lon_2d[mid_row, :].astype(np.float64)
        self._has_coords = True
        logger.info(
            f"Coordinates loaded: lat [{self.lat_axis[0]:.4f}, {self.lat_axis[-1]:.4f}], "
            f"lon [{self.lon_axis[0]:.4f}, {self.lon_axis[-1]:.4f}]"
        )

    def get_file_info(self):
        """Return metadata about the loaded file."""
        if self.data_loader is None:
            return {}
        return {
            "filename": Path(self.filepath).name,
            "n_bands": int(self.data_loader.n_bands),
            "n_rows": int(self.data_loader.n_rows),
            "n_cols": int(self.data_loader.n_cols),
            "data_type": self.data_loader.data_type,
            "wl_min": float(self.data_loader.wl_min),
            "wl_max": float(self.data_loader.wl_max),
            "has_coords": self._has_coords,
        }

    def coords_to_pixel(self, lat_val, lon_val):
        """Convert lat/lon click values back to pixel row, col."""
        if not self._has_coords:
            return int(lat_val), int(lon_val)
        row = int(np.argmin(np.abs(self.lat_axis - lat_val)))
        col = int(np.argmin(np.abs(self.lon_axis - lon_val)))
        return row, col

    # -----------------------------------------------------------------
    # Single band
    # -----------------------------------------------------------------

    def get_single_band(self, wavelength):
        """Get a single band as 2D float32 array."""
        return np.asarray(self.data_loader.get_interpolated_band(float(wavelength)))

    # -----------------------------------------------------------------
    # RGB composites
    # -----------------------------------------------------------------

    def get_rgb_composite(self, r_wl, g_wl, b_wl):
        """Create an RGB composite as uint8 (rows, cols, 3)."""
        r = np.asarray(self.data_loader.get_interpolated_band(float(r_wl)))
        g = np.asarray(self.data_loader.get_interpolated_band(float(g_wl)))
        b = np.asarray(self.data_loader.get_interpolated_band(float(b_wl)))
        rgb = normalize_rgb(r, g, b)
        return (rgb * 255).astype(np.uint8)

    # -----------------------------------------------------------------
    # Spectral indices
    # -----------------------------------------------------------------

    def calculate_index(self, index_name):
        """
        Calculate a predefined spectral index.

        Returns (2D array, index_definition dict) or (None, None).
        """
        data, idx_def = _compute_index(self.data_loader, index_name)
        if data is None:
            logger.error(f"Unknown index: {index_name}")
        return data, idx_def

    def robust_percentile(self, data, percentiles=None):
        """Calculate robust percentiles, sampling if data is large."""
        return _robust_percentile(data, percentiles=percentiles)

    # -----------------------------------------------------------------
    # Pixel spectrum extraction
    # -----------------------------------------------------------------

    def get_pixel_spectrum(self, row, col):
        """Extract full spectrum at a pixel. Returns dict or None."""
        if self.data_loader is None:
            return None

        row, col = int(row), int(col)
        if not (0 <= row < self.data_loader.n_rows and 0 <= col < self.data_loader.n_cols):
            return None

        region = self.data_loader.get_cube_region(slice(row, row + 1), slice(col, col + 1))
        spectrum = region[:, 0, 0]

        return {
            "wavelengths": self.data_loader.wavelengths.tolist(),
            "values": spectrum.tolist(),
            "row": row,
            "col": col,
            "data_type": self.data_loader.data_type,
        }

    # -----------------------------------------------------------------
    # ROI spectrum extraction
    # -----------------------------------------------------------------

    def get_roi_spectrum(self, row_min, row_max, col_min, col_max, mask=None):
        """
        Extract mean/std spectrum over a bounding-box region.

        Parameters
        ----------
        row_min, row_max, col_min, col_max : int
            Pixel bounding box (exclusive end).
        mask : np.ndarray or None
            Boolean array shaped (row_max-row_min, col_max-col_min).
            True = include pixel. None = include all pixels in bbox.

        Returns dict with wavelengths, mean_values, std_values, pixel_count.
        """
        if self.data_loader is None:
            return None

        row_min = max(0, int(row_min))
        row_max = min(self.data_loader.n_rows, int(row_max))
        col_min = max(0, int(col_min))
        col_max = min(self.data_loader.n_cols, int(col_max))

        if row_max <= row_min or col_max <= col_min:
            return None

        # cube shape: (n_bands, n_rows_region, n_cols_region)
        cube = self.data_loader.get_cube_region(
            slice(row_min, row_max), slice(col_min, col_max)
        )

        # Reshape to (n_bands, n_pixels)
        n_bands = cube.shape[0]
        pixels = cube.reshape(n_bands, -1)  # (bands, rows*cols)

        if mask is not None:
            flat_mask = mask.ravel().astype(bool)
            if flat_mask.shape[0] == pixels.shape[1]:
                pixels = pixels[:, flat_mask]

        pixel_count = pixels.shape[1]
        if pixel_count == 0:
            return None

        with np.errstate(all="ignore"):
            mean_vals = np.nanmean(pixels, axis=1)
            std_vals = np.nanstd(pixels, axis=1)

        return {
            "wavelengths": self.data_loader.wavelengths.tolist(),
            "mean_values": mean_vals.tolist(),
            "std_values": std_vals.tolist(),
            "pixel_count": int(pixel_count),
            "data_type": self.data_loader.data_type,
        }

    # -----------------------------------------------------------------
    # Material matching
    # -----------------------------------------------------------------

    def get_material_matcher(self):
        """Lazy-init MaterialMatcher, cached until file changes."""
        if self._matcher is not None:
            return self._matcher
        if self.data_loader is None:
            return None
        try:
            from aviris_tools.viewer.material_matching import MaterialMatcher
            # Use minerals library instead of default indoor/artificial
            module_dir = Path(__file__).resolve().parent.parent
            minerals_lib = module_dir / "reference_spectra" / "usgs_minerals_d2.json"
            lib_path = str(minerals_lib) if minerals_lib.exists() else None
            self._matcher = MaterialMatcher(self.data_loader, library_path=lib_path)
            lib_label = "minerals" if lib_path else "default"
            logger.info(f"MaterialMatcher initialized with {lib_label} library")
        except Exception as e:
            logger.error(f"Failed to init MaterialMatcher: {e}")
            return None
        return self._matcher

    def match_roi_spectrum(self, mean_spectrum, top_n=5):
        """
        Match a mean spectrum against the spectral library.

        Parameters
        ----------
        mean_spectrum : list or np.ndarray
            1D spectrum (same length as wavelengths).
        top_n : int
            Number of top matches to return.

        Returns list of {name, score, category}.
        """
        matcher = self.get_material_matcher()
        if matcher is None:
            return []
        try:
            spectrum = np.asarray(mean_spectrum, dtype=np.float64)
            matches = matcher.match_spectrum(spectrum, top_n=top_n, method="sam")
            return matches
        except Exception as e:
            logger.error(f"Material matching failed: {e}")
            return []

    # -----------------------------------------------------------------
    # 3D cube extraction
    # -----------------------------------------------------------------

    def get_downsampled_cube(self, spatial_step=2, spectral_step=4):
        """
        Get a downsampled cube for 3D visualization.

        Returns (cube, wavelengths) where cube is (n_bands, n_rows, n_cols) float32.
        """
        if self.data_loader is None:
            return None, None

        dl = self.data_loader
        cube = dl.data_var[::spectral_step, ::spatial_step, ::spatial_step]
        cube = np.asarray(cube, dtype=np.float32)
        wavelengths = dl.wavelengths[::spectral_step].astype(np.float32)

        logger.info(
            f"Downsampled cube: {cube.shape} "
            f"(spatial {spatial_step}x, spectral {spectral_step}x), "
            f"{cube.nbytes / 1e6:.1f} MB"
        )
        return cube, wavelengths

    def get_slice_2d(self, axis, index, downsample=1):
        """
        Extract a 2D cross-section from the cube.

        Parameters
        ----------
        axis : str
            'band' (XY at band), 'row' (bands x cols at row), 'col' (bands x rows at col)
        index : int
            Index along the specified axis.
        downsample : int
            Spatial/spectral decimation factor.

        Returns (2D float32 array, metadata dict).
        """
        if self.data_loader is None:
            return None, None

        dl = self.data_loader
        index = int(index)

        if axis == "band":
            index = min(index, dl.n_bands - 1)
            data = np.asarray(dl.data_var[index, ::downsample, ::downsample], dtype=np.float32)
            return data, {
                "axis": "band",
                "index": index,
                "wavelength": float(dl.wavelengths[index]),
            }
        elif axis == "row":
            index = min(index, dl.n_rows - 1)
            data = np.asarray(dl.data_var[::downsample, index, ::downsample], dtype=np.float32)
            return data, {
                "axis": "row",
                "index": index,
            }
        elif axis == "col":
            index = min(index, dl.n_cols - 1)
            data = np.asarray(dl.data_var[::downsample, ::downsample, index], dtype=np.float32)
            return data, {
                "axis": "col",
                "index": index,
            }
        else:
            return None, None

    # -----------------------------------------------------------------
    # File scanning
    # -----------------------------------------------------------------

    @staticmethod
    def scan_directory(data_dir):
        """Scan a directory for supported files. Returns list of dicts."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return []

        files = []
        for ext in ("*.nc", "*.h5", "*.hdf5"):
            for f in sorted(data_dir.glob(ext)):
                size_mb = f.stat().st_size / 1e6
                files.append({
                    "title": f.name,
                    "value": str(f),
                    "subtitle": f"{size_mb:.0f} MB",
                })
        return files
