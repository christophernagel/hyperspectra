"""
HSI Toolkit: Hyperspectral Imaging Learning Suite

An educational toolkit for understanding the complete imaging chain from
surface reflectance through atmospheric effects to sensor capture.

Modules:
    atmosphere: Atmospheric radiative transfer and interference modeling
    sensor: Pushbroom imaging spectrometer simulation
    forward_model: Complete radiative transfer chain integration
    visualization: Interactive visualization and exploration tools

Example:
    >>> from hsi_toolkit import AtmosphereSimulator, SensorSimulator, ForwardModel
    >>> import numpy as np
    >>> wavelengths = np.linspace(380, 2500, 224)
    >>> model = ForwardModel(wavelengths)
    >>> targets = model.generate_test_targets()
    >>> # See the explain methods for educational content
    >>> print(model.explain_forward_model())
"""

__version__ = "0.1.0"
__author__ = "HSI Learning Suite"

# Core atmosphere modules
from .atmosphere import (
    AtmosphereSimulator,
    AtmosphericState,
    GasAbsorption,
    AerosolScattering,
    RayleighScattering,
    SolarSpectrum
)

# Core sensor modules
from .sensor import (
    SensorSimulator,
    SensorConfiguration,
    PushbroomGeometry,
    ScanConfiguration,
    GratingDispersion,
    DetectorModel,
    NoiseModel
)

# Forward model
from .forward_model import ForwardModel, SceneParameters, RadiativeTransfer

# Visualization (optional, may not have all dependencies)
try:
    from .visualization import HSIVisualizer, launch_dashboard
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False
    HSIVisualizer = None
    launch_dashboard = None

__all__ = [
    # Atmosphere
    'AtmosphereSimulator',
    'AtmosphericState',
    'GasAbsorption',
    'AerosolScattering',
    'RayleighScattering',
    'SolarSpectrum',
    # Sensor
    'SensorSimulator',
    'SensorConfiguration',
    'PushbroomGeometry',
    'ScanConfiguration',
    'GratingDispersion',
    'DetectorModel',
    'NoiseModel',
    # Forward Model
    'ForwardModel',
    'SceneParameters',
    'RadiativeTransfer',
    # Visualization
    'HSIVisualizer',
    'launch_dashboard'
]


def print_summary():
    """Print a summary of available modules and capabilities."""
    summary = """
+==================================================================+
|           HSI Toolkit: Hyperspectral Imaging Learning            |
+==================================================================+
|                                                                  |
|  ATMOSPHERE MODULE                                               |
|  ----------------                                                |
|  * GasAbsorption      - H2O, O2, CO2, O3 absorption bands        |
|  * RayleighScattering - Molecular scattering (lambda^-4)         |
|  * AerosolScattering  - Particle scattering (Mie theory)         |
|  * SolarSpectrum      - Solar irradiance at TOA                  |
|  * AtmosphereSimulator - Complete atmospheric model              |
|                                                                  |
|  SENSOR MODULE                                                   |
|  -------------                                                   |
|  * PushbroomGeometry  - Scan geometry, GSD, FOV                  |
|  * GratingDispersion  - Spectral dispersion physics              |
|  * DetectorModel      - Noise sources, SNR, detection            |
|  * SensorSimulator    - Complete sensor chain                    |
|                                                                  |
|  FORWARD MODEL                                                   |
|  -------------                                                   |
|  * ForwardModel       - Surface -> Sensor complete chain         |
|  * RadiativeTransfer  - Core RT equations                        |
|                                                                  |
|  VISUALIZATION                                                   |
|  -------------                                                   |
|  * HSIVisualizer      - Matplotlib plotting utilities            |
|  * launch_dashboard() - Interactive web dashboard (needs Dash)   |
|                                                                  |
+==================================================================+

Quick Start:
    from hsi_toolkit import ForwardModel, SceneParameters
    import numpy as np

    # Create model
    model = ForwardModel()

    # Get test targets
    targets = model.generate_test_targets()

    # Simulate measurement
    scene = SceneParameters(
        surface_reflectance=targets['vegetation'],
        solar_zenith_deg=30,
        pwv_cm=1.5,
        aod_550=0.1
    )
    result = model.simulate(scene)

    # Educational content
    print(model.explain_forward_model())
"""
    print(summary)


def quick_demo():
    """Run a quick demonstration of the toolkit."""
    import numpy as np

    print("\n" + "="*60)
    print("HSI Toolkit Quick Demo")
    print("="*60)

    # Create wavelength grid
    wavelengths = np.linspace(380, 2500, 224)
    print(f"\nWavelength range: {wavelengths[0]:.0f} - {wavelengths[-1]:.0f} nm")
    print(f"Number of bands: {len(wavelengths)}")

    # Initialize forward model
    model = ForwardModel(wavelengths)
    print("\n[OK] Forward model initialized")

    # Get test targets
    targets = model.generate_test_targets()
    print(f"[OK] Generated {len(targets)} test targets: {list(targets.keys())}")

    # Create scene
    scene = SceneParameters(
        surface_reflectance=targets['vegetation'],
        solar_zenith_deg=30,
        pwv_cm=1.5,
        aod_550=0.1,
        aerosol_type='continental'
    )

    # Run simulation
    result = model.simulate(scene, add_noise=True, random_state=42)
    print("\n[OK] Simulation complete!")

    # Summary
    print(f"\nResults:")
    print(f"  At-sensor radiance: {result['at_sensor_radiance'].mean():.4f} W/m²/sr/nm (mean)")
    print(f"  Apparent reflectance: {result['apparent_reflectance'].mean():.3f} (mean)")
    print(f"  Signal electrons: {result['signal_electrons'].mean():.0f} (mean)")

    if 'snr' in result:
        print(f"  SNR range: {result['snr'].min():.0f} - {result['snr'].max():.0f}")

    print("\n" + "="*60)
    print("Use model.explain_forward_model() for educational content")
    print("Use launch_dashboard() for interactive exploration (needs Dash)")
    print("="*60 + "\n")

    return result
