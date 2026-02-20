"""
Hyperspectral viewer for AVIRIS-3 data.

This module provides:
- launch_viewer(): Full napari-based viewer with control panel
- launch_viewer_simple(): Lightweight matplotlib viewer
- SpectralIndex: Unified API for index calculations (uses indices module)

Features:
- Interactive spectral visualization with napari
- Spectral index calculations (NDVI, NDWI, minerals, etc.)
- ROI-based spectral extraction
- 3D data cube visualization
- Export to CSV
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def configure_qt():
    """Configure Qt settings from config."""
    try:
        from aviris_tools.utils.config import get_config
        config = get_config()
        scale = config.get('viewer', 'qt_scale_factor', default=1.0)
        if scale != 1.0:
            os.environ['QT_SCALE_FACTOR'] = str(scale)
    except ImportError:
        pass
    os.environ.setdefault('QT_AUTO_SCREEN_SCALE_FACTOR', '1')


def launch_viewer(filepath: Optional[str] = None):
    """
    Launch the hyperspectral viewer.

    Args:
        filepath: Optional path to NetCDF file to load on startup
    """
    configure_qt()

    try:
        if filepath:
            from .app import HyperspectralViewer
            from .panels import HyperspectralControlPanel
            import napari

            viewer = HyperspectralViewer(filepath)

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
            import napari
            viewer = napari.Viewer(title="AVIRIS-3 Hyperspectral Viewer")
            napari.run()

    except ImportError as e:
        logger.error(f"Could not import viewer: {e}")
        logger.error("Ensure napari is installed: conda install -c conda-forge napari")
        raise


def launch_viewer_simple(filepath: str):
    """
    Launch a simplified viewer for quick data inspection.

    Uses matplotlib instead of napari for lighter dependencies.

    Args:
        filepath: Path to NetCDF file
    """
    import netCDF4 as nc
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Loading {filepath.name}...")

    with nc.Dataset(filepath) as ds:
        if 'reflectance' in ds.groups:
            rfl_grp = ds.groups['reflectance']
            data = rfl_grp.variables['reflectance'][:]
            wavelengths = rfl_grp.variables['wavelength'][:]
        elif 'reflectance' in ds.variables:
            data = ds.variables['reflectance'][:]
            wavelengths = ds.variables['wavelength'][:]
        else:
            raise ValueError("Cannot find reflectance data in file")

        if data.shape[0] == len(wavelengths):
            data = np.transpose(data, (1, 2, 0))

    n_bands = len(wavelengths)
    logger.info(f"Loaded: {data.shape[0]}x{data.shape[1]} pixels, {n_bands} bands")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.2)

    init_band = n_bands // 2
    img = ax1.imshow(data[:, :, init_band], cmap='viridis')
    ax1.set_title(f'Band {init_band}: {wavelengths[init_band]:.1f} nm')
    plt.colorbar(img, ax=ax1, label='Reflectance')

    center_y, center_x = data.shape[0] // 2, data.shape[1] // 2
    spectrum = data[center_y, center_x, :]
    line, = ax2.plot(wavelengths, spectrum)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Reflectance')
    ax2.set_title(f'Spectrum at ({center_x}, {center_y})')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(wavelengths.min(), wavelengths.max())

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Band', 0, n_bands - 1, valinit=init_band, valstep=1)

    def update_band(val):
        band = int(slider.val)
        img.set_data(data[:, :, band])
        img.set_clim(vmin=np.nanpercentile(data[:, :, band], 2),
                     vmax=np.nanpercentile(data[:, :, band], 98))
        ax1.set_title(f'Band {band}: {wavelengths[band]:.1f} nm')
        fig.canvas.draw_idle()

    slider.on_changed(update_band)

    def onclick(event):
        if event.inaxes == ax1:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
                spectrum = data[y, x, :]
                line.set_ydata(spectrum)
                ax2.set_ylim(np.nanmin(spectrum) * 0.9, np.nanmax(spectrum) * 1.1)
                ax2.set_title(f'Spectrum at ({x}, {y})')
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


class SpectralIndex:
    """
    Unified API for spectral index calculations.

    This class wraps the aviris_tools.indices module to provide a simple
    interface for calculating spectral indices.

    Usage:
        result = SpectralIndex.calculate(data, wavelengths, 'ndvi')
        indices = SpectralIndex.list_indices()
    """

    @staticmethod
    def list_indices():
        """Return list of available indices grouped by category."""
        return {
            'Vegetation': [
                ('ndvi', 'Normalized Difference Vegetation Index'),
                ('ndwi', 'Normalized Difference Water Index'),
                ('ndmi', 'Normalized Difference Moisture Index'),
                ('evi', 'Enhanced Vegetation Index'),
                ('savi', 'Soil Adjusted Vegetation Index'),
            ],
            'Minerals': [
                ('clay', 'Clay Index (2200nm Al-OH)'),
                ('kaolinite', 'Kaolinite Index'),
                ('smectite', 'Smectite Index'),
                ('alunite', 'Alunite Index'),
                ('illite', 'Illite Index'),
                ('carbonate', 'Carbonate Index'),
                ('calcite', 'Calcite Index'),
                ('dolomite', 'Dolomite Index'),
                ('ferric', 'Ferric Iron Index'),
                ('ferrous', 'Ferrous Iron Index'),
            ],
            'Hydrocarbons': [
                ('hydrocarbon', 'Hydrocarbon Index (1730nm)'),
                ('oil', 'Oil Absorption Index'),
                ('methane', 'Methane Ratio'),
            ],
            'Agriculture': [
                ('ndni', 'Normalized Difference Nitrogen Index'),
                ('protein', 'Protein Absorption Index'),
                ('chlorophyll', 'Canopy Chlorophyll Index'),
            ],
        }

    @staticmethod
    def calculate(data, wavelengths, index_name: str):
        """
        Calculate spectral index using the indices module.

        Args:
            data: 3D array (y, x, bands)
            wavelengths: 1D array of wavelengths
            index_name: Name of index (e.g., 'ndvi', 'clay')

        Returns:
            2D array of index values
        """
        index_name = index_name.lower()

        # Import from indices module
        if index_name in ['ndvi', 'ndwi', 'ndmi', 'evi', 'savi']:
            from aviris_tools.indices import vegetation
            funcs = {
                'ndvi': vegetation.ndvi,
                'ndwi': vegetation.ndwi,
                'ndmi': vegetation.ndmi,
                'evi': vegetation.evi,
                'savi': vegetation.savi,
            }
            return funcs[index_name](data, wavelengths)

        elif index_name in ['clay', 'kaolinite', 'smectite', 'alunite',
                            'illite', 'carbonate', 'calcite', 'dolomite',
                            'ferric', 'ferrous']:
            from aviris_tools.indices import minerals
            funcs = {
                'clay': minerals.clay_index,
                'kaolinite': minerals.kaolinite_index,
                'smectite': minerals.smectite_index,
                'alunite': minerals.alunite_index,
                'illite': minerals.illite_index,
                'carbonate': minerals.carbonate_index,
                'calcite': minerals.calcite_index,
                'dolomite': minerals.dolomite_index,
                'ferric': minerals.ferric_iron_index,
                'ferrous': minerals.ferrous_iron_index,
            }
            return funcs[index_name](data, wavelengths)

        elif index_name in ['hydrocarbon', 'oil', 'methane']:
            from aviris_tools.indices import hydrocarbons
            funcs = {
                'hydrocarbon': hydrocarbons.hydrocarbon_index,
                'oil': hydrocarbons.oil_index,
                'methane': hydrocarbons.methane_ratio,
            }
            return funcs[index_name](data, wavelengths)

        elif index_name in ['ndni', 'protein', 'chlorophyll']:
            from aviris_tools.indices import nitrogen
            funcs = {
                'ndni': nitrogen.ndni,
                'protein': nitrogen.protein_index,
                'chlorophyll': nitrogen.canopy_chlorophyll_index,
            }
            return funcs[index_name](data, wavelengths)

        else:
            available = []
            for category in SpectralIndex.list_indices().values():
                available.extend([name for name, _ in category])
            raise ValueError(f"Unknown index: {index_name}. Available: {available}")


def main():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='AVIRIS-3 Hyperspectral Viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('filepath', nargs='?', default=None,
                        help='NetCDF file to open (optional)')
    parser.add_argument('--simple', action='store_true',
                        help='Use simple matplotlib viewer (no napari)')
    parser.add_argument('--scale', type=float, default=None,
                        help='Qt scale factor for UI')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    if args.scale:
        os.environ['QT_SCALE_FACTOR'] = str(args.scale)

    if args.simple:
        if args.filepath:
            launch_viewer_simple(args.filepath)
        else:
            print("Error: --simple requires a filepath")
            sys.exit(1)
    else:
        launch_viewer(args.filepath)


if __name__ == '__main__':
    main()
