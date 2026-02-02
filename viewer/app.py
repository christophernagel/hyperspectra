"""
Core hyperspectral viewer application.

This module contains the main HyperspectralViewer class that:
- Manages the napari viewer
- Handles data loading
- Calculates spectral indices using the indices module
- Creates RGB composites
"""

import numpy as np
import napari
from pathlib import Path
import logging

from .data_loader import LazyHyperspectralData
from .constants import (
    INDEX_DEFINITIONS, ROBUST_PERCENTILE_LOW, ROBUST_PERCENTILE_HIGH,
    LARGE_ARRAY_THRESHOLD, SAMPLE_SIZE
)

# Import index calculation functions from the indices module
try:
    from aviris_tools.indices import utils as idx_utils
    HAS_INDICES_MODULE = True
except ImportError:
    HAS_INDICES_MODULE = False

logger = logging.getLogger(__name__)


class HyperspectralViewer:
    """
    Main hyperspectral viewer application.

    Provides:
    - RGB composite creation
    - Spectral index calculation
    - ROI spectral extraction
    - Layer management
    """

    def __init__(self, filepath):
        """
        Initialize the viewer with a data file.

        Args:
            filepath: Path to NetCDF file with reflectance/radiance data
        """
        self.filepath = Path(filepath)
        self.data_loader = LazyHyperspectralData(filepath)
        self.setup_viewer()

    def setup_viewer(self):
        """Initialize the napari viewer."""
        data_label = "Reflectance" if self.data_loader.data_type == 'reflectance' else "Radiance"
        self.viewer = napari.Viewer(title=f"Hyperspectral Explorer - {data_label}")

        # Calculate memory requirements
        n_pixels = self.data_loader.n_rows * self.data_loader.n_cols
        n_elements = n_pixels * self.data_loader.n_bands
        memory_gb = (n_elements * 4) / (1024 ** 3)

        logger.info(f"Dataset: {self.data_loader.n_bands} bands x "
                    f"{self.data_loader.n_rows} x {self.data_loader.n_cols} pixels")
        logger.info(f"Estimated memory for full cube: {memory_gb:.2f} GB")

        # Load initial view (first 50 bands if large)
        MEMORY_THRESHOLD_GB = 4.0
        if memory_gb > MEMORY_THRESHOLD_GB:
            logger.warning(f"Large dataset ({memory_gb:.1f} GB) - loading spectral subset")
            full_cube = self.data_loader.data_var[:50, :, :].astype(np.float32)
        else:
            full_cube = self.data_loader.get_full_cube()

        self.cube_layer = self.viewer.add_image(
            full_cube,
            name=f"{data_label} Cube",
            visible=False,
            colormap='viridis'
        )

        # Add ROI shapes layer
        self.roi_layer = self.viewer.add_shapes(
            name="ROIs",
            edge_color='cyan',
            face_color='transparent',
            edge_width=2
        )

        # Add initial RGB composite
        self.add_composite("True Color", 640, 550, 470)

    # =========================================================================
    # RGB Composites
    # =========================================================================

    def add_composite(self, name, r_wl, g_wl, b_wl):
        """
        Create and display an RGB composite.

        Args:
            name: Display name for the composite
            r_wl, g_wl, b_wl: Wavelengths for R, G, B channels

        Returns:
            Dict with composite info, or None on error
        """
        logger.info(f"Creating composite: {name} (R={r_wl}, G={g_wl}, B={b_wl})")

        # Get interpolated bands
        r_band = self.data_loader.get_interpolated_band(r_wl)
        g_band = self.data_loader.get_interpolated_band(g_wl)
        b_band = self.data_loader.get_interpolated_band(b_wl)

        # Normalize each band
        def normalize(band):
            p2, p98 = np.nanpercentile(band, [2, 98])
            if p98 <= p2:
                return np.zeros_like(band)
            return np.clip((band - p2) / (p98 - p2), 0, 1)

        rgb = np.stack([normalize(r_band), normalize(g_band), normalize(b_band)], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0).astype(np.float32)

        # Remove existing layer with same name
        self._remove_layer(name)

        self.viewer.add_image(rgb, name=name, rgb=True)
        logger.info(f"  Added RGB layer: {name}")

        return {
            'name': name,
            'r_wl': r_wl,
            'g_wl': g_wl,
            'b_wl': b_wl
        }

    # =========================================================================
    # Spectral Index Calculation
    # =========================================================================

    def calculate_index(self, index_name, estimate_uncertainty=False):
        """
        Calculate a pre-defined spectral index.

        Uses the indices module when available, falls back to inline calculation.

        Args:
            index_name: Name of the index (must be in INDEX_DEFINITIONS)
            estimate_uncertainty: If True, compute uncertainty estimate

        Returns:
            Dict with index info, or None on error
        """
        if index_name not in INDEX_DEFINITIONS:
            logger.error(f"Unknown index: {index_name}")
            return None

        idx_def = INDEX_DEFINITIONS[index_name]
        logger.info(f"Calculating {index_name}...")

        idx_type = idx_def['type']

        # Calculate based on type
        if idx_type == 'continuum':
            index_data = self._calculate_continuum_index(
                idx_def['feature'], idx_def['left'], idx_def['right']
            )
        elif idx_type == 'nd':
            index_data = self._calculate_nd_index(idx_def['b1'], idx_def['b2'])
        elif idx_type == 'ratio':
            index_data = self._calculate_ratio_index(idx_def['b1'], idx_def['b2'])
        else:
            logger.error(f"Unknown index type: {idx_type}")
            return None

        if index_data is None:
            return None

        # Calculate uncertainty if requested
        uncertainty = None
        if estimate_uncertainty and idx_type in ['nd', 'ratio']:
            uncertainty = self._estimate_uncertainty(idx_def['b1'], idx_def['b2'], idx_type)

        # Display the index
        layer_name = f"{index_name} Index"
        clim = self._display_index(index_data, layer_name, idx_def['cmap'])

        return {
            'name': index_name,
            'cmap': idx_def['cmap'],
            'clim': clim,
            'type': idx_type,
            'uncertainty': uncertainty
        }

    def calculate_custom_index(self, name, b1_wl, b2_wl, index_type='ratio', cmap='viridis'):
        """
        Calculate a custom spectral index from wavelengths.

        Args:
            name: Display name for the index
            b1_wl, b2_wl: Wavelengths for bands 1 and 2
            index_type: 'ratio' or 'nd' (normalized difference)
            cmap: Colormap name

        Returns:
            Dict with index info, or None on error
        """
        logger.info(f"Calculating custom index: {name} ({index_type}: {b1_wl}/{b2_wl})")

        if index_type == 'nd':
            index_data = self._calculate_nd_index(b1_wl, b2_wl)
        else:
            index_data = self._calculate_ratio_index(b1_wl, b2_wl)

        if index_data is None:
            return None

        # Display the index
        layer_name = f"{name}"
        clim = self._display_index(index_data, layer_name, cmap)

        return {
            'name': name,
            'cmap': cmap,
            'clim': clim,
            'type': index_type,
            'uncertainty': None
        }

    def _calculate_nd_index(self, b1_wl, b2_wl):
        """Calculate normalized difference index: (b1-b2)/(b1+b2)"""
        b1 = self.data_loader.get_interpolated_band(b1_wl).astype(np.float64)
        b2 = self.data_loader.get_interpolated_band(b2_wl).astype(np.float64)

        eps = 1e-10
        result = (b1 - b2) / (b1 + b2 + eps)
        return result.astype(np.float32)

    def _calculate_ratio_index(self, b1_wl, b2_wl):
        """Calculate band ratio index: b1/b2"""
        b1 = self.data_loader.get_interpolated_band(b1_wl).astype(np.float64)
        b2 = self.data_loader.get_interpolated_band(b2_wl).astype(np.float64)

        eps = 1e-10
        result = b1 / (b2 + eps)
        return np.clip(result, 0.01, 10.0).astype(np.float32)

    def _calculate_continuum_index(self, feature_wl, left_wl, right_wl):
        """
        Calculate continuum-removed absorption depth.

        Absorption depth = 1 - (R_feature / R_continuum)
        """
        r_left = self.data_loader.get_interpolated_band(left_wl).astype(np.float64)
        r_right = self.data_loader.get_interpolated_band(right_wl).astype(np.float64)
        r_feature = self.data_loader.get_interpolated_band(feature_wl).astype(np.float64)

        # Linear continuum at feature wavelength
        weight = (feature_wl - left_wl) / (right_wl - left_wl)
        r_continuum = r_left * (1 - weight) + r_right * weight

        eps = 1e-10
        depth = 1.0 - (r_feature / (r_continuum + eps))
        return np.clip(depth, 0, 1).astype(np.float32)

    def _estimate_uncertainty(self, b1_wl, b2_wl, idx_type):
        """Estimate relative uncertainty for the index."""
        snr1 = self.data_loader.estimate_snr(b1_wl)
        snr2 = self.data_loader.estimate_snr(b2_wl)

        # Propagate uncertainty
        if idx_type == 'nd':
            # For ND: relative uncertainty ~ sqrt(2) / SNR
            rel_uncert = np.sqrt(2) / min(snr1, snr2)
        else:
            # For ratio: relative uncertainty ~ sqrt(1/SNR1^2 + 1/SNR2^2)
            rel_uncert = np.sqrt(1 / snr1 ** 2 + 1 / snr2 ** 2)

        return {
            'median_relative_uncertainty': float(rel_uncert),
            'p95_relative_uncertainty': float(rel_uncert * 2),
            'snr_b1': snr1,
            'snr_b2': snr2
        }

    def _display_index(self, index_data, layer_name, cmap):
        """Display calculated index as a napari layer."""
        self._remove_layer(layer_name)

        # Calculate contrast limits
        clim = self._robust_percentile(index_data)

        if clim[1] <= clim[0]:
            logger.warning(f"  Constant data detected (all values ~ {clim[0]:.3f})")
            clim = [clim[0], clim[0] + 1e-6]

        display_data = np.nan_to_num(index_data, nan=0.0).astype(np.float32)

        self.viewer.add_image(
            display_data,
            name=layer_name,
            colormap=cmap,
            contrast_limits=clim
        )

        logger.info(f"  Range: {np.nanmin(index_data):.3f} to {np.nanmax(index_data):.3f}")
        return clim

    def _robust_percentile(self, data, percentiles=None):
        """Calculate robust percentiles, sampling if data is large."""
        if percentiles is None:
            percentiles = [ROBUST_PERCENTILE_LOW, ROBUST_PERCENTILE_HIGH]

        flat = data.ravel()
        valid = flat[~np.isnan(flat)]

        if len(valid) == 0:
            return [0, 1]

        if len(valid) > LARGE_ARRAY_THRESHOLD:
            indices = np.random.choice(len(valid), SAMPLE_SIZE, replace=False)
            valid = valid[indices]

        return [float(np.percentile(valid, p)) for p in percentiles]

    # =========================================================================
    # ROI Spectral Extraction
    # =========================================================================

    def set_roi_mode(self, mode):
        """Set the ROI drawing mode."""
        self.viewer.layers.selection.active = self.roi_layer
        if mode == 'rectangle':
            self.roi_layer.mode = 'add_rectangle'
        elif mode == 'polygon':
            self.roi_layer.mode = 'add_polygon'
        elif mode == 'ellipse':
            self.roi_layer.mode = 'add_ellipse'
        logger.info(f"ROI mode: {mode}")

    def clear_rois(self):
        """Clear all ROI shapes."""
        self.roi_layer.data = []
        logger.info("ROIs cleared")

    def extract_roi_spectra(self):
        """
        Extract mean spectra from all ROIs in the shapes layer.

        Returns:
            List of dicts, each with wavelengths, mean, std, label, n_pixels
        """
        if len(self.roi_layer.data) == 0:
            logger.warning("No ROIs defined")
            return []

        spectra_list = []
        loader = self.data_loader

        for i, shape in enumerate(self.roi_layer.data):
            # Get bounding box for this shape
            min_row = max(0, int(np.floor(np.min(shape[:, 0]))))
            max_row = min(loader.n_rows, int(np.ceil(np.max(shape[:, 0]))))
            min_col = max(0, int(np.floor(np.min(shape[:, 1]))))
            max_col = min(loader.n_cols, int(np.ceil(np.max(shape[:, 1]))))

            # Load region efficiently (all bands at once)
            region = loader.data_var[
                :, min_row:max_row, min_col:max_col
            ].astype(np.float32)

            # Compute stats
            mean_spectrum = np.nanmean(region, axis=(1, 2))
            std_spectrum = np.nanstd(region, axis=(1, 2))
            n_pixels = (max_row - min_row) * (max_col - min_col)

            roi_name = f"ROI {i+1}"
            spectra_list.append({
                'wavelengths': loader.wavelengths,
                'mean': mean_spectrum,
                'std': std_spectrum,
                'label': f'{roi_name} ({n_pixels} px)',
                'n_pixels': n_pixels
            })
            logger.info(f"{roi_name}: {max_row - min_row} x {max_col - min_col} pixels")

        return spectra_list

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _remove_layer(self, name):
        """Remove layer by name if it exists."""
        for layer in list(self.viewer.layers):
            if layer.name == name:
                self.viewer.layers.remove(layer)
                break

    def close(self):
        """Clean up resources."""
        self.data_loader.close()


def run_viewer(filepath=None):
    """
    Launch the hyperspectral viewer.

    Args:
        filepath: Optional path to NetCDF file
    """
    import sys

    if filepath:
        viewer = HyperspectralViewer(filepath)

        # Add control panel
        from .panels import HyperspectralControlPanel
        control_panel = HyperspectralControlPanel(viewer)
        control_panel.set_data_type(viewer.data_loader.data_type)
        control_panel.set_wavelength_range(
            viewer.data_loader.wl_min,
            viewer.data_loader.wl_max
        )

        viewer.viewer.window.add_dock_widget(
            control_panel,
            name="Hyperspectral Tools",
            area="right"
        )

        napari.run()
    else:
        # Launch empty viewer
        viewer = napari.Viewer(title="Hyperspectral Explorer")
        napari.run()
