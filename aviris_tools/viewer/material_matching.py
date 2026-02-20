"""
Material Matching for Hyperspectral Data

Matches pixel spectra against reference spectral libraries (USGS, custom).
Works with any hyperspectral data (AVIRIS, EMIT, HyperspecI).

Usage:
    from aviris_tools.viewer.material_matching import MaterialMatcher

    # Initialize with data loader
    matcher = MaterialMatcher(data_loader)

    # Match a single pixel
    matches = matcher.match_pixel(row, col, top_n=5)

    # Match with custom library
    matcher.load_library('path/to/library.json')
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class MaterialMatcher:
    """
    Match hyperspectral pixel spectra against reference libraries.

    Handles wavelength resampling automatically.
    """

    def __init__(self, data_loader, library_path: Optional[str] = None):
        """
        Initialize matcher with a data loader.

        Parameters
        ----------
        data_loader : LazyHyperspectralData or LazyH5Data
            Loaded hyperspectral data with .wavelengths attribute
        library_path : str, optional
            Path to JSON spectral library. If None, uses default USGS indoor library.
        """
        self.data = data_loader
        self.library = None
        self.lib_wavelengths = None

        # Find default library
        if library_path is None:
            # Look for USGS library relative to this module
            module_dir = Path(__file__).parent.parent
            default_d2 = module_dir / 'reference_spectra' / 'usgs_indoor_d2.json'
            default_d1 = module_dir / 'reference_spectra' / 'usgs_indoor_d1.json'

            # Choose based on data wavelength range
            if hasattr(data_loader, 'wl_max') and data_loader.wl_max > 1000:
                library_path = default_d2 if default_d2.exists() else None
            else:
                library_path = default_d1 if default_d1.exists() else None

        if library_path:
            self.load_library(library_path)

    def load_library(self, json_path: str) -> None:
        """Load a spectral library from JSON."""
        import json

        with open(json_path, 'r') as f:
            data = json.load(f)

        wavelengths = np.array(data['metadata']['wavelengths_nm'])

        spectra = {}
        for name, spec_data in data['spectra'].items():
            refl = np.array(spec_data['reflectance'])
            refl = np.where(refl < 0, np.nan, refl)  # Mask bad values
            spectra[name] = {
                'reflectance': refl,
                'category': spec_data.get('category', 'unknown'),
            }

        self.library = {
            'wavelengths': wavelengths,
            'spectra': spectra,
            'metadata': data['metadata']
        }
        self.lib_wavelengths = wavelengths
        logger.info(f"Loaded library with {len(spectra)} spectra")

    def get_pixel_spectrum(self, row: int, col: int) -> np.ndarray:
        """Extract spectrum at a pixel location."""
        if hasattr(self.data, 'get_cube_region'):
            # Use get_cube_region for efficient single-pixel access
            # Get a 1x1 region and squeeze
            region = self.data.get_cube_region(
                slice(row, row+1),
                slice(col, col+1)
            )
            return region[:, 0, 0]
        elif hasattr(self.data, 'datacube'):
            # Direct datacube access
            return self.data.datacube[:, row, col]
        elif hasattr(self.data, 'get_band'):
            # Fallback: band-by-band (slow)
            return np.array([self.data.get_band(b)[row, col]
                           for b in range(self.data.n_bands)])
        else:
            raise ValueError("Unknown data loader interface")

    def resample_to_library(self, spectrum: np.ndarray) -> np.ndarray:
        """Resample a spectrum to match library wavelength grid."""
        if self.lib_wavelengths is None:
            raise ValueError("No library loaded")

        return np.interp(
            self.lib_wavelengths,
            self.data.wavelengths,
            spectrum,
            left=np.nan,
            right=np.nan
        )

    def match_spectrum(self, spectrum: np.ndarray, top_n: int = 5,
                       method: str = 'sam') -> List[Dict]:
        """
        Match a spectrum against the library.

        Parameters
        ----------
        spectrum : np.ndarray
            Spectrum to match (will be resampled to library grid)
        top_n : int
            Number of top matches to return
        method : str
            'sam' (spectral angle), 'correlation', or 'euclidean'

        Returns
        -------
        List of dicts with 'name', 'score', 'category' keys
        """
        if self.library is None:
            raise ValueError("No library loaded")

        # Resample to library grid
        resampled = self.resample_to_library(spectrum)

        # Compute matches
        matches = []
        unknown_valid = np.nan_to_num(resampled, nan=0)
        unknown_norm = unknown_valid / (np.linalg.norm(unknown_valid) + 1e-10)

        for name, spec_data in self.library['spectra'].items():
            ref = spec_data['reflectance']
            ref_valid = np.nan_to_num(ref, nan=0)
            ref_norm = ref_valid / (np.linalg.norm(ref_valid) + 1e-10)

            if method == 'sam':
                # Spectral Angle Mapper (lower = more similar)
                dot_product = np.clip(np.dot(unknown_norm, ref_norm), -1, 1)
                score = np.arccos(dot_product)
            elif method == 'correlation':
                # Pearson correlation (higher = more similar)
                score = np.corrcoef(unknown_valid, ref_valid)[0, 1]
                if np.isnan(score):
                    score = 0
            else:  # euclidean
                score = np.linalg.norm(unknown_valid - ref_valid)

            matches.append((name, score))

        # Sort
        if method == 'correlation':
            matches.sort(key=lambda x: -x[1])
        else:
            matches.sort(key=lambda x: x[1])

        # Return top_n with category info
        results = []
        for name, score in matches[:top_n]:
            results.append({
                'name': name,
                'score': score,
                'category': self.library['spectra'][name].get('category', 'unknown'),
            })

        return results

    def match_pixel(self, row: int, col: int, top_n: int = 5,
                    method: str = 'sam') -> List[Dict]:
        """
        Match spectrum at a pixel location.

        Parameters
        ----------
        row, col : int
            Pixel coordinates
        top_n : int
            Number of matches to return
        method : str
            Matching method ('sam', 'correlation', 'euclidean')

        Returns
        -------
        List of match results with 'name', 'score', 'category'
        """
        spectrum = self.get_pixel_spectrum(row, col)
        return self.match_spectrum(spectrum, top_n=top_n, method=method)

    def match_roi(self, row_slice: slice, col_slice: slice,
                  top_n: int = 5, method: str = 'sam') -> List[Dict]:
        """
        Match mean spectrum of a region of interest.

        Parameters
        ----------
        row_slice, col_slice : slice
            Region to average
        top_n : int
            Number of matches to return
        method : str
            Matching method

        Returns
        -------
        List of match results
        """
        # Get mean spectrum over ROI
        if hasattr(self.data, 'get_cube_region'):
            region = self.data.get_cube_region(row_slice, col_slice)
            spectrum = np.nanmean(region, axis=(1, 2))
        else:
            # Fallback: get pixel by pixel
            spectra = []
            for r in range(row_slice.start, row_slice.stop):
                for c in range(col_slice.start, col_slice.stop):
                    spectra.append(self.get_pixel_spectrum(r, c))
            spectrum = np.nanmean(spectra, axis=0)

        return self.match_spectrum(spectrum, top_n=top_n, method=method)

    def get_library_categories(self) -> Dict[str, int]:
        """Return count of spectra by category."""
        if self.library is None:
            return {}

        cats = {}
        for name, spec in self.library['spectra'].items():
            cat = spec.get('category', 'unknown')
            cats[cat] = cats.get(cat, 0) + 1
        return cats

    def get_reference_spectrum(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a reference spectrum by name.

        Returns (wavelengths, reflectance) tuple.
        """
        if self.library is None:
            raise ValueError("No library loaded")

        if name not in self.library['spectra']:
            raise KeyError(f"Unknown material: {name}")

        return (
            self.lib_wavelengths.copy(),
            self.library['spectra'][name]['reflectance'].copy()
        )


def quick_match(data_loader, row: int, col: int, top_n: int = 5) -> List[Dict]:
    """
    Quick one-liner for matching a pixel.

    Example:
        from aviris_tools.viewer.material_matching import quick_match
        from aviris_tools.viewer import load_hyperspectral

        data = load_hyperspectral('scene.nc')
        matches = quick_match(data, 100, 200)
        for m in matches:
            print(f"{m['name']}: {m['score']:.4f}")
    """
    matcher = MaterialMatcher(data_loader)
    return matcher.match_pixel(row, col, top_n=top_n)
