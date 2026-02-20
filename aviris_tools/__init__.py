"""
AVIRIS-3 Processing Tools
=========================

A modular toolkit for atmospheric correction and visualization of
AVIRIS-3 hyperspectral imagery.

Modules:
    atm_correction: Atmospheric correction (Py6S, ISOFIT)
    viewer: Interactive hyperspectral visualization
    utils: Common utilities (I/O, memory, config)
    indices: Spectral indices (minerals, hydrocarbons, vegetation, nitrogen)

Usage:
    from aviris_tools import process, view

    # Process radiance to reflectance
    process('radiance.nc', 'obs.nc', 'output.nc', method='py6s')

    # Launch viewer
    view('output.nc')

    # Calculate spectral indices
    from aviris_tools.indices import clay_index, hydrocarbon_index, ndvi
"""

__version__ = '2.2.0'
__author__ = 'Christopher / Claude'

from aviris_tools.utils.config import Config
from aviris_tools.utils.memory import get_available_memory, MemoryManager

# Convenience functions
def process(radiance_path, obs_path, output_path, method='py6s', **kwargs):
    """Process AVIRIS-3 radiance to surface reflectance."""
    if method == 'py6s':
        from aviris_tools.atm_correction.py6s_processor import Py6SProcessor
        processor = Py6SProcessor(radiance_path, obs_path, output_path, **kwargs)
    elif method == 'isofit':
        from aviris_tools.atm_correction.isofit_processor import ISOFITProcessor
        processor = ISOFITProcessor(radiance_path, obs_path, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'py6s' or 'isofit'")

    return processor.run()

def view(filepath=None):
    """Launch the hyperspectral viewer."""
    from aviris_tools.viewer.main import launch_viewer
    launch_viewer(filepath)
