"""
Atmospheric Radiative Transfer Module

Simulates how electromagnetic radiation interacts with Earth's atmosphere,
including molecular absorption, aerosol scattering, and Rayleigh scattering.

Physics Background:
    When light travels through the atmosphere, it undergoes:
    1. Absorption by gas molecules (H2O, CO2, O3, O2, CH4)
    2. Scattering by molecules (Rayleigh - wavelength dependent)
    3. Scattering by aerosols (Mie - particle size dependent)

    The result is that some wavelengths are strongly absorbed (creating
    "atmospheric windows" where we can see through, and "absorption bands"
    where we cannot), and the overall signal is attenuated and mixed with
    "path radiance" - light scattered into the sensor's view that never
    touched the surface.
"""

from .gas_absorption import GasAbsorption, WaterVaporAbsorption, CO2Absorption, O3Absorption, O2Absorption
from .aerosol_scattering import AerosolScattering, AerosolModel
from .rayleigh_scattering import RayleighScattering
from .atmosphere_simulator import AtmosphereSimulator, AtmosphericState
from .solar_spectrum import SolarSpectrum

__all__ = [
    'AtmosphereSimulator',
    'AtmosphericState',
    'GasAbsorption',
    'WaterVaporAbsorption',
    'CO2Absorption',
    'O3Absorption',
    'O2Absorption',
    'AerosolScattering',
    'AerosolModel',
    'RayleighScattering',
    'SolarSpectrum'
]
