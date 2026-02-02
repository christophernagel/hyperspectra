"""
Hyperspectral viewer for AVIRIS-3 data.

This package provides an interactive napari-based viewer for hyperspectral imagery
with spectral index calculation, ROI extraction, and RGB composite creation.

Usage:
    from aviris_tools.viewer import HyperspectralViewer, run_viewer

    # Launch with a file
    run_viewer('/path/to/data.nc')

    # Or use programmatically
    viewer = HyperspectralViewer('/path/to/data.nc')
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
    'HyperspectralViewer',
    'LazyHyperspectralData',
]


def __getattr__(name):
    """Lazy import for heavy classes."""
    if name == 'HyperspectralViewer':
        from .app import HyperspectralViewer
        return HyperspectralViewer
    elif name == 'LazyHyperspectralData':
        from .data_loader import LazyHyperspectralData
        return LazyHyperspectralData
    elif name == 'INDEX_DEFINITIONS':
        from .constants import INDEX_DEFINITIONS
        return INDEX_DEFINITIONS
    elif name == 'INDEX_METADATA':
        from .constants import INDEX_METADATA
        return INDEX_METADATA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
