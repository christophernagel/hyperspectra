"""
Hyperspectral viewer for AVIRIS-3 and indoor scene data.

This package provides an interactive napari-based viewer for hyperspectral imagery
with spectral index calculation, ROI extraction, and RGB composite creation.

Supported formats:
- NetCDF (.nc) - AVIRIS-3 L1B/L2 radiance and reflectance
- HDF5 (.h5) - HyperspecI indoor hyperspectral dataset

Usage:
    from aviris_tools.viewer import run_viewer, load_hyperspectral

    # Launch viewer with any supported format
    run_viewer('/path/to/data.nc')
    run_viewer('/path/to/scene.h5')

    # Or load data programmatically
    loader = load_hyperspectral('/path/to/data.nc')
    print(f"Bands: {loader.n_bands}, Range: {loader.wl_min}-{loader.wl_max}nm")
"""

# Lazy imports to avoid loading heavy dependencies until needed
def launch_viewer(filepath=None):
    """
    Launch the hyperspectral viewer (legacy API).

    Args:
        filepath: Optional path to NetCDF file to load on startup
    """
    from .app import run_viewer
    run_viewer(filepath)


def run_viewer(filepath=None):
    """
    Launch the hyperspectral viewer.

    Args:
        filepath: Optional path to NetCDF file to load on startup
    """
    from .app import run_viewer as _run
    _run(filepath)


__all__ = [
    'launch_viewer',
    'run_viewer',
    'load_hyperspectral',
    'HyperspectralViewer',
    'LazyHyperspectralData',
    'LazyH5Data',
    'MaterialMatcher',
    'quick_match',
]


def __getattr__(name):
    """Lazy import for heavy classes."""
    if name == 'HyperspectralViewer':
        from .app import HyperspectralViewer
        return HyperspectralViewer
    elif name == 'LazyHyperspectralData':
        from .data_loader import LazyHyperspectralData
        return LazyHyperspectralData
    elif name == 'LazyH5Data':
        from .h5_loader import LazyH5Data
        return LazyH5Data
    elif name == 'load_hyperspectral':
        from .data_loader import load_hyperspectral
        return load_hyperspectral
    elif name == 'INDEX_DEFINITIONS':
        from .constants import INDEX_DEFINITIONS
        return INDEX_DEFINITIONS
    elif name == 'INDEX_METADATA':
        from .constants import INDEX_METADATA
        return INDEX_METADATA
    elif name == 'INDOOR_INDEX_DEFINITIONS':
        from .constants import INDOOR_INDEX_DEFINITIONS
        return INDOOR_INDEX_DEFINITIONS
    elif name == 'INDOOR_MATERIAL_SIGNATURES':
        from .constants import INDOOR_MATERIAL_SIGNATURES
        return INDOOR_MATERIAL_SIGNATURES
    elif name == 'MaterialMatcher':
        from .material_matching import MaterialMatcher
        return MaterialMatcher
    elif name == 'quick_match':
        from .material_matching import quick_match
        return quick_match
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
