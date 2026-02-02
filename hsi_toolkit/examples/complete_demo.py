"""
Complete HSI Toolkit Demonstration

This script demonstrates all the major features of the HSI Learning Toolkit,
providing a comprehensive tour of hyperspectral imaging physics.

Run this script to:
1. Understand atmospheric effects on remote sensing
2. Explore sensor physics and noise
3. See the complete imaging chain in action
4. Learn about the inverse problem (atmospheric correction)

Usage:
    python complete_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hsi_toolkit import (
    AtmosphereSimulator, AtmosphericState,
    SensorSimulator, SensorConfiguration, NoiseModel,
    ForwardModel, SceneParameters,
    RayleighScattering, GasAbsorption, AerosolScattering,
    GratingDispersion, DetectorModel
)


def demo_atmospheric_effects():
    """Demonstrate atmospheric effects on hyperspectral signals."""
    print("\n" + "="*70)
    print("PART 1: ATMOSPHERIC EFFECTS")
    print("="*70)

    # Create wavelength grid
    wavelengths = np.linspace(380, 2500, 500)

    # Initialize atmospheric components
    rayleigh = RayleighScattering()
    gas = GasAbsorption()
    aerosol = AerosolScattering()

    # 1. Rayleigh scattering explanation
    print("\n1. RAYLEIGH SCATTERING (Why the sky is blue)")
    print("-" * 50)
    print(rayleigh.explain_blue_sky())

    # Calculate Rayleigh optical depth
    tau_rayleigh = rayleigh.optical_depth(wavelengths)

    # 2. Gas absorption
    print("\n2. GAS ABSORPTION BANDS")
    print("-" * 50)
    T_gas = gas.combined_transmission(wavelengths, pwv_cm=2.0, ozone_atm_cm=0.34)

    print("Major absorption bands:")
    print("  H₂O: 720, 820, 940, 1140, 1380, 1880 nm")
    print("  O₂:  688, 762 nm")
    print("  CO₂: 2010, 2060 nm")
    print("  O₃:  Chappuis band (visible)")

    # 3. Aerosol effects
    print("\n3. AEROSOL SCATTERING")
    print("-" * 50)
    print("Different aerosol types:")
    for atype in ['continental', 'maritime', 'urban', 'desert']:
        aerosol_temp = AerosolScattering(aerosol_type=atype, aod_550=0.2)
        aod_at_1000 = aerosol_temp.optical_depth(np.array([1000]))[0]
        print(f"  {atype:12s}: AOD(1000nm) = {aod_at_1000:.3f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Rayleigh optical depth
    ax = axes[0, 0]
    ax.semilogy(wavelengths, tau_rayleigh, 'b-', linewidth=2)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Optical Depth')
    ax.set_title('Rayleigh Scattering Optical Depth\n(τ ∝ λ⁻⁴)')
    ax.grid(True, alpha=0.3)

    # Gas transmittance
    ax = axes[0, 1]
    ax.plot(wavelengths, T_gas, 'g-', linewidth=1)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmittance')
    ax.set_title('Gas Absorption Transmittance\n(PWV=2cm, O₃=0.34 atm-cm)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Annotate major bands
    bands = [(760, 'O₂'), (940, 'H₂O'), (1380, 'H₂O'), (1880, 'H₂O')]
    for wl, gas_name in bands:
        ax.annotate(gas_name, (wl, 0.15), fontsize=8, ha='center')

    # Aerosol comparison
    ax = axes[1, 0]
    for atype, color in [('continental', 'brown'), ('maritime', 'blue'),
                         ('urban', 'gray'), ('desert', 'orange')]:
        aerosol_temp = AerosolScattering(aerosol_type=atype, aod_550=0.2)
        T_aerosol = aerosol_temp.transmission(wavelengths)
        ax.plot(wavelengths, T_aerosol, label=atype, color=color, linewidth=1.5)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmittance')
    ax.set_title('Aerosol Transmittance by Type\n(AOD₅₅₀=0.2)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Combined transmittance
    ax = axes[1, 1]
    T_rayleigh = rayleigh.transmission(wavelengths, solar_zenith_deg=30, view_zenith_deg=0)
    T_aerosol = AerosolScattering(aod_550=0.15).transmission(wavelengths)
    T_total = T_gas * T_rayleigh * T_aerosol

    ax.fill_between(wavelengths, 0, T_total, alpha=0.3, label='Total')
    ax.plot(wavelengths, T_gas, 'g--', alpha=0.7, label='Gas', linewidth=1)
    ax.plot(wavelengths, T_rayleigh, 'b--', alpha=0.7, label='Rayleigh', linewidth=1)
    ax.plot(wavelengths, T_aerosol, 'r--', alpha=0.7, label='Aerosol', linewidth=1)
    ax.plot(wavelengths, T_total, 'k-', linewidth=2, label='Combined')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmittance')
    ax.set_title('Combined Atmospheric Transmittance')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_atmospheric_effects.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_atmospheric_effects.png")

    return fig


def demo_sensor_physics():
    """Demonstrate sensor physics and noise characteristics."""
    print("\n" + "="*70)
    print("PART 2: SENSOR PHYSICS")
    print("="*70)

    # Grating physics
    print("\n1. GRATING DISPERSION")
    print("-" * 50)
    grating = GratingDispersion(groove_density=600, blaze_wavelength=1000)
    print(grating.get_grating_summary())
    print(grating.explain_grating_equation())

    # Detector noise
    print("\n2. DETECTOR NOISE MODEL")
    print("-" * 50)
    detector = DetectorModel(NoiseModel(read_noise_e=20, dark_current_e_s=100))
    print(detector.get_noise_summary(integration_time_s=0.01))
    print(detector.explain_noise_sources())

    # Visualization
    wavelengths = np.linspace(380, 2500, 224)
    angles = grating.diffraction_angle(wavelengths)
    efficiency = grating.efficiency(wavelengths)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Grating dispersion
    ax = axes[0, 0]
    ax.plot(wavelengths, angles, 'b-', linewidth=2)
    ax.axvline(grating.blaze_wavelength, color='red', linestyle='--',
              label=f'Blaze = {grating.blaze_wavelength}nm')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Diffraction Angle (°)')
    ax.set_title('Grating Dispersion: d×sin(θ) = mλ')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Grating efficiency
    ax = axes[0, 1]
    ax.plot(wavelengths, efficiency * 100, 'g-', linewidth=2)
    ax.axvline(grating.blaze_wavelength, color='red', linestyle='--')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Blaze Efficiency (sinc² envelope)')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    # SNR vs signal
    ax = axes[1, 0]
    signals = np.logspace(2, 5, 100)  # 100 to 100,000 electrons
    snr = detector.calculate_snr(signals, integration_time_s=0.01)

    ax.loglog(signals, snr, 'b-', linewidth=2, label='Actual SNR')
    ax.loglog(signals, np.sqrt(signals), 'g--', label='Shot-limited (√N)')
    ax.loglog(signals, signals / 20, 'r--', label='Read-limited (N/σᵣ)')

    ax.set_xlabel('Signal (electrons)')
    ax.set_ylabel('SNR')
    ax.set_title('SNR vs Signal Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Noise breakdown vs signal
    ax = axes[1, 1]
    shot_noise = np.sqrt(signals)
    dark_noise = np.sqrt(100 * 0.01)  # 100 e-/s × 10ms
    read_noise = np.ones_like(signals) * 20

    ax.loglog(signals, shot_noise, 'b-', label='Shot noise √N')
    ax.axhline(read_noise[0], color='r', linestyle='-', label='Read noise')
    ax.axhline(dark_noise, color='purple', linestyle='-', label='Dark noise')
    ax.axhline(np.sqrt(read_noise[0]**2 + dark_noise**2), color='orange',
              linestyle='--', label='Floor (read + dark)')

    ax.set_xlabel('Signal (electrons)')
    ax.set_ylabel('Noise (electrons)')
    ax.set_title('Noise Components vs Signal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1, 1000)

    plt.tight_layout()
    plt.savefig('demo_sensor_physics.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_sensor_physics.png")

    return fig


def demo_forward_model():
    """Demonstrate the complete imaging chain."""
    print("\n" + "="*70)
    print("PART 3: FORWARD MODEL (Complete Imaging Chain)")
    print("="*70)

    wavelengths = np.linspace(380, 2500, 224)
    model = ForwardModel(wavelengths)

    # Print explanation
    print(model.explain_forward_model())

    # Get test targets
    targets = model.generate_test_targets()

    # Create scene with vegetation
    scene = SceneParameters(
        surface_reflectance=targets['vegetation'],
        solar_zenith_deg=30,
        pwv_cm=1.5,
        aod_550=0.1,
        aerosol_type='continental'
    )

    # Run simulation
    result = model.simulate(scene, add_noise=True, return_intermediates=True,
                           random_state=42)

    print("\nSimulation Results:")
    print("-" * 50)
    print(f"Surface type: Vegetation")
    print(f"Solar zenith: {scene.solar_zenith_deg}°")
    print(f"PWV: {scene.pwv_cm} cm")
    print(f"AOD: {scene.aod_550}")
    print(f"\nMean at-sensor radiance: {result['at_sensor_radiance'].mean():.4f} W/m²/sr/nm")
    print(f"Mean apparent reflectance: {result['apparent_reflectance'].mean():.3f}")

    # Compare surfaces
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Surface reflectances
    ax = axes[0, 0]
    for name, refl in targets.items():
        ax.plot(wavelengths, refl, label=name, linewidth=1.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')
    ax.set_title('Test Target Reflectance Spectra')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # At-sensor radiance components
    ax = axes[0, 1]
    total = result['at_sensor_radiance']
    path = result.get('path_radiance', np.zeros_like(wavelengths))
    surface_contrib = total - path

    ax.fill_between(wavelengths, 0, path, alpha=0.5, label='Path radiance', color='red')
    ax.fill_between(wavelengths, path, total, alpha=0.5, label='Surface signal', color='green')
    ax.plot(wavelengths, total, 'k-', linewidth=1, label='Total')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Radiance (W/m²/sr/nm)')
    ax.set_title('At-Sensor Radiance Components')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # True vs apparent reflectance
    ax = axes[1, 0]
    ax.plot(wavelengths, targets['vegetation'], 'g-', linewidth=2, label='True')
    ax.plot(wavelengths, result['apparent_reflectance'], 'r--', linewidth=2, label='Apparent')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance')
    ax.set_title('True vs Apparent Reflectance\n(Effect of atmosphere)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.8)

    # SNR spectrum
    ax = axes[1, 1]
    if 'snr' in result:
        ax.semilogy(wavelengths, result['snr'], 'b-', linewidth=1.5)
        ax.axhline(100, color='green', linestyle='--', label='Target (100)')
        ax.axhline(10, color='red', linestyle='--', label='Minimum (10)')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('SNR')
        ax.set_title('Signal-to-Noise Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1, result['snr'].max() * 2)

    plt.tight_layout()
    plt.savefig('demo_forward_model.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_forward_model.png")

    return fig


def demo_sensitivity_analysis():
    """Demonstrate parameter sensitivity analysis."""
    print("\n" + "="*70)
    print("PART 4: SENSITIVITY ANALYSIS")
    print("="*70)

    wavelengths = np.linspace(380, 2500, 224)
    model = ForwardModel(wavelengths)
    targets = model.generate_test_targets()

    # Base scene
    base_scene = SceneParameters(
        surface_reflectance=targets['vegetation'],
        solar_zenith_deg=30,
        pwv_cm=1.5,
        aod_550=0.1
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PWV sensitivity
    print("\n1. Water Vapor Sensitivity")
    print("-" * 50)
    ax = axes[0, 0]
    pwv_values = [0.5, 1.0, 2.0, 3.0, 4.0]
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(pwv_values)))

    for pwv, color in zip(pwv_values, colors):
        scene = SceneParameters(
            surface_reflectance=targets['vegetation'],
            solar_zenith_deg=30,
            pwv_cm=pwv,
            aod_550=0.1
        )
        result = model.simulate(scene, add_noise=False)
        ax.plot(wavelengths, result['at_sensor_radiance'],
               label=f'PWV={pwv}cm', color=color, linewidth=1.5)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Radiance (W/m²/sr/nm)')
    ax.set_title('Sensitivity to Water Vapor')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # AOD sensitivity
    print("\n2. Aerosol Optical Depth Sensitivity")
    print("-" * 50)
    ax = axes[0, 1]
    aod_values = [0.05, 0.1, 0.2, 0.3, 0.5]
    colors = plt.cm.Oranges(np.linspace(0.3, 1, len(aod_values)))

    for aod, color in zip(aod_values, colors):
        scene = SceneParameters(
            surface_reflectance=targets['vegetation'],
            solar_zenith_deg=30,
            pwv_cm=1.5,
            aod_550=aod
        )
        result = model.simulate(scene, add_noise=False)
        ax.plot(wavelengths, result['at_sensor_radiance'],
               label=f'AOD={aod}', color=color, linewidth=1.5)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Radiance (W/m²/sr/nm)')
    ax.set_title('Sensitivity to Aerosol Optical Depth')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Solar zenith sensitivity
    print("\n3. Solar Zenith Angle Sensitivity")
    print("-" * 50)
    ax = axes[1, 0]
    sza_values = [15, 30, 45, 60, 75]
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(sza_values)))

    for sza, color in zip(sza_values, colors):
        scene = SceneParameters(
            surface_reflectance=targets['vegetation'],
            solar_zenith_deg=sza,
            pwv_cm=1.5,
            aod_550=0.1
        )
        result = model.simulate(scene, add_noise=False)
        ax.plot(wavelengths, result['at_sensor_radiance'],
               label=f'SZA={sza}°', color=color, linewidth=1.5)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Radiance (W/m²/sr/nm)')
    ax.set_title('Sensitivity to Solar Zenith Angle')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Surface type comparison
    print("\n4. Surface Type Comparison")
    print("-" * 50)
    ax = axes[1, 1]
    surface_colors = {
        'vegetation': 'green',
        'soil': 'brown',
        'water': 'blue',
        'concrete': 'gray'
    }

    for surface_type, color in surface_colors.items():
        scene = SceneParameters(
            surface_reflectance=targets[surface_type],
            solar_zenith_deg=30,
            pwv_cm=1.5,
            aod_550=0.1
        )
        result = model.simulate(scene, add_noise=False)
        ax.plot(wavelengths, result['at_sensor_radiance'],
               label=surface_type.capitalize(), color=color, linewidth=1.5)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Radiance (W/m²/sr/nm)')
    ax.set_title('Radiance by Surface Type')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('demo_sensitivity.png', dpi=150, bbox_inches='tight')
    print("\nSaved: demo_sensitivity.png")

    return fig


def main():
    """Run the complete demonstration."""
    print("\n" + "="*70)
    print("HSI TOOLKIT: COMPLETE DEMONSTRATION")
    print("="*70)
    print("\nThis demo will:")
    print("  1. Explain atmospheric effects on hyperspectral data")
    print("  2. Demonstrate sensor physics and noise")
    print("  3. Show the complete forward model in action")
    print("  4. Perform sensitivity analysis")
    print("\nOutput files will be saved to the current directory.")

    # Run all demos
    demo_atmospheric_effects()
    demo_sensor_physics()
    demo_forward_model()
    demo_sensitivity_analysis()

    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nGenerated figures:")
    print("  - demo_atmospheric_effects.png")
    print("  - demo_sensor_physics.png")
    print("  - demo_forward_model.png")
    print("  - demo_sensitivity.png")
    print("\nFor interactive exploration, run:")
    print("  from hsi_toolkit import launch_dashboard")
    print("  launch_dashboard()")
    print("="*70 + "\n")

    plt.show()


if __name__ == '__main__':
    main()
