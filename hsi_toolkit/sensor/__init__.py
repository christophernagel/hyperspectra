"""
Sensor Physics Module

Simulates pushbroom imaging spectrometer characteristics including
optical geometry, spectral dispersion, detector response, and noise.

Components:
    - PushbroomGeometry: Scan geometry and field of view
    - GratingDispersion: Diffraction grating physics
    - DetectorModel: Noise sources and response characteristics
    - SensorSimulator: Complete sensor simulation
"""

from .pushbroom_geometry import PushbroomGeometry, ScanConfiguration
from .grating_dispersion import GratingDispersion, SpectralCalibration
from .detector_model import DetectorModel, NoiseModel
from .sensor_simulator import SensorSimulator, SensorConfiguration

__all__ = [
    'SensorSimulator',
    'SensorConfiguration',
    'PushbroomGeometry',
    'ScanConfiguration',
    'GratingDispersion',
    'SpectralCalibration',
    'DetectorModel',
    'NoiseModel'
]
