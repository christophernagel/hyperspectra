"""
Forward Model - Complete Imaging Chain Simulator

Integrates atmospheric and sensor physics into a complete simulation
of the hyperspectral imaging process.

The Forward Model Chain:
    Surface Reflectance (ρ)
           │
           ▼
    ┌─────────────────────────────────┐
    │    ATMOSPHERIC MODEL            │
    │  • Gas absorption (H₂O, O₂...)  │
    │  • Rayleigh scattering          │
    │  • Aerosol scattering           │
    │  • Path radiance                │
    └─────────────────────────────────┘
           │
           ▼
    At-Sensor Radiance (L)
           │
           ▼
    ┌─────────────────────────────────┐
    │    SENSOR MODEL                 │
    │  • Optical throughput           │
    │  • Spectral dispersion          │
    │  • Detector response            │
    │  • Noise sources                │
    └─────────────────────────────────┘
           │
           ▼
    Digital Numbers (DN)

This module allows you to:
    1. Simulate realistic sensor measurements from known surfaces
    2. Understand how each component affects the final measurement
    3. Study the inverse problem (atmospheric correction) difficulty
    4. Design and optimize sensor parameters
"""

import numpy as np
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

from ..atmosphere.atmosphere_simulator import AtmosphereSimulator, AtmosphericState
from ..atmosphere.solar_spectrum import SolarSpectrum
from ..sensor.sensor_simulator import SensorSimulator, SensorConfiguration
from ..sensor.detector_model import NoiseModel
from .radiative_transfer import RadiativeTransfer, RTParameters


@dataclass
class SceneParameters:
    """Parameters defining the imaging scenario."""
    # Surface
    surface_reflectance: np.ndarray  # Spectral reflectance (0-1)

    # Geometry
    solar_zenith_deg: float = 30.0   # Sun position
    solar_azimuth_deg: float = 180.0
    view_zenith_deg: float = 0.0     # Sensor viewing angle
    view_azimuth_deg: float = 0.0

    # Atmosphere
    pwv_cm: float = 1.5              # Precipitable water vapor
    ozone_du: float = 300.0          # Ozone column (Dobson Units)
    aod_550: float = 0.1             # Aerosol optical depth at 550nm
    aerosol_type: str = 'continental'

    # Altitude
    ground_elevation_m: float = 0.0  # Target altitude
    sensor_altitude_m: float = 8500.0  # Sensor altitude


class ForwardModel:
    """
    Complete imaging spectrometer forward model.

    Simulates the entire chain from surface reflectance to
    sensor digital numbers, including all physical effects.
    """

    def __init__(self,
                 wavelengths: Optional[np.ndarray] = None,
                 sensor_config: Optional[SensorConfiguration] = None,
                 noise_model: Optional[NoiseModel] = None):
        """
        Initialize forward model.

        Args:
            wavelengths: Wavelength grid (nm). Default: AVIRIS-like 380-2500nm
            sensor_config: Sensor configuration parameters
            noise_model: Detector noise parameters
        """
        # Default wavelength grid (AVIRIS-3 like)
        if wavelengths is None:
            wavelengths = np.linspace(380, 2500, 224)
        self.wavelengths = wavelengths
        self.n_bands = len(wavelengths)

        # Initialize component models
        self.atmosphere = AtmosphereSimulator(wavelengths)
        self.solar = SolarSpectrum(wavelengths)
        self.rt = RadiativeTransfer(wavelengths)

        # Sensor model
        if sensor_config is None:
            sensor_config = SensorConfiguration(
                n_spectral_pixels=len(wavelengths),
                wavelength_min_nm=wavelengths[0],
                wavelength_max_nm=wavelengths[-1]
            )
        self.sensor = SensorSimulator(sensor_config, noise_model=noise_model)

    def simulate(self,
                 scene: SceneParameters,
                 add_noise: bool = True,
                 return_intermediates: bool = False,
                 random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete forward model simulation.

        Args:
            scene: Scene parameters (surface, atmosphere, geometry)
            add_noise: Whether to add sensor noise
            return_intermediates: Return intermediate results
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with simulation results
        """
        results = {
            'wavelengths': self.wavelengths,
            'scene_parameters': scene
        }

        # Step 1: Get solar irradiance at TOA
        E_sun = self.solar.irradiance()
        results['solar_irradiance_toa'] = E_sun

        # Step 2: Calculate atmospheric effects
        # Update atmosphere state with scene parameters
        self.atmosphere.update_state(
            solar_zenith_deg=scene.solar_zenith_deg,
            view_zenith_deg=scene.view_zenith_deg,
            relative_azimuth_deg=abs(scene.solar_azimuth_deg - scene.view_azimuth_deg),
            pwv_cm=scene.pwv_cm,
            ozone_du=scene.ozone_du,
            aod_550=scene.aod_550,
            aerosol_type=scene.aerosol_type,
            surface_elevation_m=scene.ground_elevation_m
        )

        at_sensor_radiance = self.atmosphere.radiance_at_sensor(scene.surface_reflectance)

        results['atmospheric_state'] = self.atmosphere.state
        results['at_sensor_radiance'] = at_sensor_radiance

        if return_intermediates:
            results['path_radiance'] = self.atmosphere.L_path
            results['transmittance'] = self.atmosphere.T_total
            results['downwelling_irradiance'] = self.atmosphere.E_down

        # Step 3: Apply sensor model
        sensor_result = self.sensor.simulate_measurement(
            at_sensor_radiance,
            add_noise=add_noise,
            random_state=random_state
        )

        results['digital_number'] = sensor_result['digital_number']
        results['signal_electrons'] = sensor_result.get('input_signal_e')

        if add_noise and 'snr' in sensor_result:
            results['snr'] = sensor_result['snr']

        # Step 4: Calculate apparent reflectance (quick uncorrected version)
        cos_sza = np.cos(np.radians(scene.solar_zenith_deg))
        results['apparent_reflectance'] = (
            np.pi * at_sensor_radiance / (E_sun * cos_sza)
        )

        return results

    def simulate_datacube(self,
                          reflectance_cube: np.ndarray,
                          scene: SceneParameters,
                          add_noise: bool = True,
                          random_state: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Simulate a full datacube from a reflectance cube.

        Args:
            reflectance_cube: Input reflectance (n_lines, n_samples, n_bands)
            scene: Scene parameters (atmosphere, geometry)
            add_noise: Whether to add noise
            random_state: Random seed

        Returns:
            Dictionary with radiance and DN cubes
        """
        n_lines, n_samples, n_bands = reflectance_cube.shape
        rng = np.random.RandomState(random_state)

        # Output arrays
        radiance_cube = np.zeros_like(reflectance_cube)
        dn_cube = np.zeros((n_lines, n_samples, n_bands), dtype=np.uint16)

        # Set atmospheric state (same for all pixels in simple model)
        self.atmosphere.update_state(
            solar_zenith_deg=scene.solar_zenith_deg,
            view_zenith_deg=scene.view_zenith_deg,
            relative_azimuth_deg=abs(scene.solar_azimuth_deg - scene.view_azimuth_deg),
            pwv_cm=scene.pwv_cm,
            ozone_du=scene.ozone_du,
            aod_550=scene.aod_550,
            aerosol_type=scene.aerosol_type
        )

        # Process each pixel
        for line in range(n_lines):
            for sample in range(n_samples):
                refl = reflectance_cube[line, sample, :]

                # Atmospheric forward model
                radiance = self.atmosphere.radiance_at_sensor(refl)
                radiance_cube[line, sample, :] = radiance

                # Sensor model
                sensor_result = self.sensor.simulate_measurement(
                    atm_result['total_radiance'],
                    add_noise=add_noise,
                    random_state=rng
                )
                dn_cube[line, sample, :] = sensor_result['digital_number']

        return {
            'reflectance': reflectance_cube,
            'radiance': radiance_cube,
            'digital_number': dn_cube
        }

    def compare_atmospheres(self,
                            reflectance: np.ndarray,
                            base_scene: SceneParameters,
                            parameter: str,
                            values: list) -> Dict[str, np.ndarray]:
        """
        Compare sensor signal across different atmospheric conditions.

        Args:
            reflectance: Surface reflectance spectrum
            base_scene: Base scene parameters
            parameter: Parameter to vary ('pwv_cm', 'aod_550', etc.)
            values: List of values for the parameter

        Returns:
            Dictionary with radiance arrays for each condition
        """
        results = {
            'wavelengths': self.wavelengths,
            'parameter': parameter,
            'values': values,
            'radiances': []
        }

        for val in values:
            # Create modified scene
            scene = SceneParameters(
                surface_reflectance=reflectance,
                solar_zenith_deg=base_scene.solar_zenith_deg,
                view_zenith_deg=base_scene.view_zenith_deg,
                pwv_cm=base_scene.pwv_cm,
                ozone_du=base_scene.ozone_du,
                aod_550=base_scene.aod_550,
                aerosol_type=base_scene.aerosol_type
            )

            # Update varied parameter
            setattr(scene, parameter, val)

            # Run forward model
            result = self.simulate(scene, add_noise=False)
            results['radiances'].append(result['at_sensor_radiance'])

        results['radiances'] = np.array(results['radiances'])
        return results

    def sensitivity_analysis(self,
                              reflectance: np.ndarray,
                              scene: SceneParameters,
                              delta: float = 0.01) -> Dict[str, np.ndarray]:
        """
        Analyze how radiance changes with reflectance.

        Args:
            reflectance: Base surface reflectance
            scene: Scene parameters
            delta: Reflectance perturbation

        Returns:
            Dictionary with sensitivity metrics
        """
        # Base radiance
        scene.surface_reflectance = reflectance
        base_result = self.simulate(scene, add_noise=False)
        L_base = base_result['at_sensor_radiance']

        # Perturbed radiance
        scene.surface_reflectance = reflectance + delta
        perturbed_result = self.simulate(scene, add_noise=False)
        L_perturbed = perturbed_result['at_sensor_radiance']

        # Sensitivity dL/dρ
        sensitivity = (L_perturbed - L_base) / delta

        return {
            'wavelengths': self.wavelengths,
            'base_radiance': L_base,
            'sensitivity': sensitivity,
            'relative_sensitivity': sensitivity / np.maximum(L_base, 1e-10)
        }

    def inverse_model_error(self,
                             scene: SceneParameters,
                             noise_electrons: float = 100) -> Dict[str, np.ndarray]:
        """
        Estimate reflectance retrieval error from sensor noise.

        Args:
            scene: Scene parameters
            noise_electrons: Detector noise (electrons)

        Returns:
            Dictionary with error estimates
        """
        # Forward model
        result = self.simulate(scene, add_noise=False)

        # Sensitivity
        sens = self.sensitivity_analysis(scene.surface_reflectance, scene)

        # Radiance noise (convert electrons to radiance units)
        # This is simplified; real conversion depends on calibration
        gain = self.sensor.config.optical_transmission
        radiance_noise = noise_electrons * gain / self.sensor.system_throughput

        # Reflectance error = radiance_noise / sensitivity
        refl_error = radiance_noise / np.maximum(sens['sensitivity'], 1e-10)

        return {
            'wavelengths': self.wavelengths,
            'reflectance_error': refl_error,
            'snr_equivalent': scene.surface_reflectance / np.maximum(refl_error, 1e-10)
        }

    def generate_test_targets(self) -> Dict[str, np.ndarray]:
        """
        Generate reflectance spectra for standard test targets.

        Returns:
            Dictionary with named reflectance spectra
        """
        wl = self.wavelengths

        targets = {}

        # Vegetation (simplified PROSPECT-like)
        targets['vegetation'] = self._vegetation_spectrum(wl)

        # Soil (linear approximation)
        targets['soil'] = self._soil_spectrum(wl)

        # Water
        targets['water'] = self._water_spectrum(wl)

        # Concrete/urban
        targets['concrete'] = self._concrete_spectrum(wl)

        # Reference panels
        targets['white_panel'] = np.ones_like(wl) * 0.95
        targets['gray_panel'] = np.ones_like(wl) * 0.25
        targets['black_panel'] = np.ones_like(wl) * 0.05

        return targets

    def _vegetation_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Generate vegetation-like reflectance spectrum."""
        refl = np.zeros_like(wl)

        # Chlorophyll absorption (visible)
        refl += 0.05 * np.exp(-((wl - 550) / 50) ** 2)  # Green peak
        refl += 0.02 * (wl < 700)  # Low visible

        # Red edge
        red_edge = 1 / (1 + np.exp(-(wl - 710) / 15))
        refl += 0.4 * red_edge

        # NIR plateau
        refl = np.where((wl >= 750) & (wl <= 1300), 0.45, refl)

        # SWIR decline with water absorption
        swir_decline = 0.45 - 0.15 * (wl - 1300) / 1200
        refl = np.where(wl > 1300, np.maximum(swir_decline, 0.1), refl)

        # Water absorption features
        h2o_absorption = (
            0.2 * np.exp(-((wl - 970) / 30) ** 2) +
            0.25 * np.exp(-((wl - 1200) / 40) ** 2) +
            0.3 * np.exp(-((wl - 1450) / 60) ** 2) +
            0.35 * np.exp(-((wl - 1940) / 80) ** 2)
        )
        refl = refl * (1 - h2o_absorption)

        return np.clip(refl, 0, 1)

    def _soil_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Generate soil-like reflectance spectrum."""
        # Increasing reflectance with wavelength
        refl = 0.1 + 0.2 * (wl - 400) / 2100

        # Iron absorption feature
        refl -= 0.05 * np.exp(-((wl - 900) / 100) ** 2)

        # Clay absorption
        refl -= 0.03 * np.exp(-((wl - 2200) / 50) ** 2)

        return np.clip(refl, 0.05, 0.5)

    def _water_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Generate water reflectance spectrum."""
        # Very low, decreasing with wavelength
        refl = 0.05 * np.exp(-(wl - 400) / 200)

        # Near zero in NIR/SWIR
        refl = np.where(wl > 700, 0.005, refl)

        return np.clip(refl, 0.001, 0.1)

    def _concrete_spectrum(self, wl: np.ndarray) -> np.ndarray:
        """Generate concrete-like reflectance spectrum."""
        # Relatively flat gray
        refl = 0.25 + 0.05 * np.sin(2 * np.pi * (wl - 400) / 500)

        # Slight increase in SWIR
        refl = np.where(wl > 1500, 0.3, refl)

        return np.clip(refl, 0.15, 0.4)

    def get_chain_summary(self) -> str:
        """Get summary of the complete imaging chain."""
        summary = f"""
Complete Hyperspectral Imaging Chain
====================================

FORWARD MODEL: ρ(surface) → DN(sensor)

1. Surface Reflectance
   Spectral range: {self.wavelengths[0]:.0f} - {self.wavelengths[-1]:.0f} nm
   Channels: {self.n_bands}

2. Atmospheric Effects
   • Gas absorption (H₂O, O₂, CO₂, O₃)
   • Rayleigh scattering (molecular)
   • Aerosol scattering (particles)
   • Path radiance addition

3. At-Sensor Radiance
   L = L_path + T × (ρ × E_down/π)

4. Sensor Response
{self.sensor.get_system_summary()}

INVERSE MODEL: DN(sensor) → ρ(surface)
   Requires atmospheric correction (remove steps 2-4)
"""
        return summary

    def explain_forward_model(self) -> str:
        """Educational explanation of the forward model."""
        explanation = """
The Forward Model: From Surface to Sensor
=========================================

Why Do We Need a Forward Model?

    To understand what a sensor measures, we need to trace
    light from the sun, through the atmosphere, to the surface,
    back through the atmosphere, and into the sensor.

    Each step modifies the signal:

    SUN (source of light)
     │
     │  E₀ = solar irradiance
     ▼
    ══════════════════════════════════════
    ATMOSPHERE (going down)
    ──────────────────────────────────────
    │ Absorption: H₂O, O₃, O₂, CO₂
    │ Scattering: Rayleigh (air molecules)
    │             Mie (aerosols/particles)
    │
    │ E_down = E₀ × T_down (direct)
    │        + E_diffuse (scattered sky light)
    ▼
    ══════════════════════════════════════
    SURFACE
    ══════════════════════════════════════
    │
    │ Reflection: L_surf = ρ × E_down / π
    │ (assuming Lambertian surface)
    │
    ▼
    ══════════════════════════════════════
    ATMOSPHERE (going up)
    ──────────────────────────────────────
    │ Same gases, different path
    │ Additional path radiance L_path
    │
    │ L_toa = L_path + T_up × L_surf
    ▼
    ══════════════════════════════════════
    SENSOR
    ──────────────────────────────────────
    │ Optical system: throughput, aberrations
    │ Grating: spectral dispersion
    │ Detector: QE, noise, digitization
    │
    │ DN = f(L_toa, sensor_params, noise)
    ▼
    DIGITAL OUTPUT


Key Equations:

    At-Sensor Radiance:
        L_sensor = L_path + T_up × (ρ × E_down / π) / (1 - ρ × S)
                   ─────   ─────────────────────────────────────
                   Atmos.   Surface contribution
                   only     (attenuated by atmosphere)

    Digital Number:
        DN = G × [QE × L_sensor × A × Ω × Δλ × t / (hc/λ) + noise]
             ─   ──────────────────────────────────────────────────
           gain  electrons (signal + noise)


Why This Matters:

    1. Understanding: Know what affects your measurements
    2. Simulation: Test algorithms on synthetic data
    3. Correction: Invert the model to recover true reflectance
    4. Optimization: Design better sensors and campaigns
"""
        return explanation
