"""
Detector Model and Noise Sources

Models CCD/CMOS detector response and noise characteristics.

Noise Sources in Imaging Detectors:

1. PHOTON (SHOT) NOISE
   - Fundamental: quantum nature of light
   - Follows Poisson statistics: σ = √N
   - Dominant in well-lit conditions

2. READ NOISE
   - Readout electronics noise
   - Constant per read (independent of signal)
   - Gaussian distributed
   - Typical: 2-50 electrons

3. DARK CURRENT
   - Thermal electron generation
   - Temperature dependent (doubles every ~6-7°C)
   - Accumulates with integration time
   - Can be reduced by cooling

4. QUANTIZATION NOISE
   - Digitization error
   - Depends on ADC bit depth
   - Usually negligible for 14-16 bit ADCs

5. FIXED PATTERN NOISE (FPN)
   - Pixel-to-pixel response variation
   - Corrected by flat-field calibration
   - Photo Response Non-Uniformity (PRNU)

Signal-to-Noise Ratio:
    SNR = S / √(S + D + R²)

    where:
        S = signal (electrons)
        D = dark current (electrons)
        R = read noise (electrons RMS)

References:
    - Janesick, 2001: Scientific Charge-Coupled Devices
    - Holst & Lomheim, 2007: CMOS/CCD Sensors and Camera Systems
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class NoiseModel:
    """
    Detector noise model parameters.

    All noise values in electrons (e-) unless otherwise specified.
    """
    # Quantum efficiency
    quantum_efficiency: float = 0.8     # Fraction of photons detected (0-1)

    # Read noise
    read_noise_e: float = 20.0          # Read noise (electrons RMS)

    # Dark current
    dark_current_e_s: float = 100.0     # Dark current (electrons/pixel/second)
    temperature_c: float = -20.0         # Operating temperature (°C)
    reference_temp_c: float = 25.0       # Reference temperature for dark current

    # ADC
    full_well_e: float = 100000         # Full well capacity (electrons)
    adc_bits: int = 14                   # ADC bit depth
    gain_e_per_dn: float = 6.0          # System gain (electrons per DN)

    # Fixed pattern noise
    prnu_percent: float = 1.0           # Photo Response Non-Uniformity (%)
    dark_fpn_percent: float = 5.0       # Dark current FPN (%)


class DetectorModel:
    """
    Imaging detector simulation model.

    Converts photon flux to digital numbers including all noise sources.
    """

    # Physical constants
    PLANCK = 6.626e-34       # Planck constant (J⋅s)
    C = 2.998e8              # Speed of light (m/s)

    def __init__(self, noise_model: Optional[NoiseModel] = None):
        """
        Initialize detector model.

        Args:
            noise_model: Noise parameters (uses defaults if None)
        """
        self.noise = noise_model if noise_model is not None else NoiseModel()
        self._calculate_derived()

    def _calculate_derived(self):
        """Calculate derived parameters."""
        # ADC parameters
        self.max_dn = 2 ** self.noise.adc_bits - 1
        self.saturation_e = self.noise.full_well_e

        # Temperature-adjusted dark current
        # Dark current doubles every ~6.5°C
        temp_diff = self.noise.temperature_c - self.noise.reference_temp_c
        self.dark_current_actual = self.noise.dark_current_e_s * (2 ** (temp_diff / 6.5))

    def photons_to_electrons(self, n_photons: np.ndarray) -> np.ndarray:
        """
        Convert photon count to photoelectrons.

        Accounts for quantum efficiency.

        Args:
            n_photons: Number of incident photons

        Returns:
            Number of photoelectrons
        """
        return n_photons * self.noise.quantum_efficiency

    def radiance_to_photons(self, radiance: np.ndarray,
                            wavelength_nm: float,
                            pixel_area_m2: float,
                            solid_angle_sr: float,
                            integration_time_s: float,
                            bandwidth_nm: float) -> np.ndarray:
        """
        Convert spectral radiance to photon count.

        Args:
            radiance: Spectral radiance (W/m²/sr/nm)
            wavelength_nm: Wavelength in nm
            pixel_area_m2: Detector pixel area
            solid_angle_sr: Solid angle of pixel FOV
            integration_time_s: Integration time
            bandwidth_nm: Spectral bandwidth of channel

        Returns:
            Number of photons
        """
        # Photon energy E = hc/λ
        wavelength_m = wavelength_nm * 1e-9
        photon_energy = self.PLANCK * self.C / wavelength_m

        # Power at detector = L × A × Ω × Δλ
        power_w = radiance * pixel_area_m2 * solid_angle_sr * bandwidth_nm

        # Energy = Power × time
        energy_j = power_w * integration_time_s

        # Number of photons = Energy / photon_energy
        n_photons = energy_j / photon_energy

        return n_photons

    def add_shot_noise(self, signal_e: np.ndarray,
                       random_state: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        Add photon (shot) noise.

        Shot noise follows Poisson statistics: variance = signal.

        Args:
            signal_e: Signal in electrons
            random_state: Random state for reproducibility

        Returns:
            Signal with shot noise added
        """
        rng = random_state if random_state is not None else np.random

        # Poisson-distributed shot noise
        # For large signals, use Gaussian approximation for efficiency
        large_signal_mask = signal_e > 100

        noisy_signal = np.zeros_like(signal_e)

        # Small signals: true Poisson
        small_signals = signal_e[~large_signal_mask]
        if len(small_signals) > 0:
            noisy_signal[~large_signal_mask] = rng.poisson(np.maximum(small_signals, 0))

        # Large signals: Gaussian approximation (faster)
        large_signals = signal_e[large_signal_mask]
        if len(large_signals) > 0:
            sigma = np.sqrt(large_signals)
            noisy_signal[large_signal_mask] = large_signals + rng.normal(0, 1, len(large_signals)) * sigma

        return noisy_signal

    def add_dark_current(self, shape: Tuple[int, ...],
                         integration_time_s: float,
                         random_state: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        Add dark current signal and noise.

        Dark current is signal-independent thermal electron generation.

        Args:
            shape: Output array shape
            integration_time_s: Integration time
            random_state: Random state

        Returns:
            Dark current contribution (electrons)
        """
        rng = random_state if random_state is not None else np.random

        # Mean dark current
        dark_mean = self.dark_current_actual * integration_time_s

        # Dark current shot noise (Poisson)
        if dark_mean > 100:
            dark_signal = dark_mean + rng.normal(0, np.sqrt(dark_mean), shape)
        else:
            dark_signal = rng.poisson(max(dark_mean, 0), shape).astype(float)

        # Fixed pattern noise (pixel-to-pixel variation)
        dark_fpn = self.noise.dark_fpn_percent / 100 * dark_mean
        dark_signal += rng.normal(0, dark_fpn, shape)

        return np.maximum(dark_signal, 0)

    def add_read_noise(self, shape: Tuple[int, ...],
                       random_state: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        Add read noise.

        Read noise is Gaussian-distributed, independent of signal.

        Args:
            shape: Output array shape
            random_state: Random state

        Returns:
            Read noise contribution (electrons)
        """
        rng = random_state if random_state is not None else np.random
        return rng.normal(0, self.noise.read_noise_e, shape)

    def add_prnu(self, signal_e: np.ndarray,
                 prnu_pattern: Optional[np.ndarray] = None,
                 random_state: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        Apply Photo Response Non-Uniformity.

        PRNU is multiplicative: each pixel has slightly different sensitivity.

        Args:
            signal_e: Input signal
            prnu_pattern: Pre-computed PRNU pattern (generates if None)
            random_state: Random state

        Returns:
            Signal with PRNU applied
        """
        rng = random_state if random_state is not None else np.random

        if prnu_pattern is None:
            # Generate random PRNU pattern
            prnu_sigma = self.noise.prnu_percent / 100
            prnu_pattern = 1.0 + rng.normal(0, prnu_sigma, signal_e.shape)

        return signal_e * prnu_pattern

    def electrons_to_dn(self, electrons: np.ndarray) -> np.ndarray:
        """
        Convert electrons to digital numbers (DN).

        Applies gain and quantization.

        Args:
            electrons: Signal in electrons

        Returns:
            Digital numbers (integers)
        """
        # Apply gain
        dn = electrons / self.noise.gain_e_per_dn

        # Clip to ADC range
        dn = np.clip(dn, 0, self.max_dn)

        # Quantize (round to integer)
        return np.round(dn).astype(int)

    def simulate_detection(self, signal_e: np.ndarray,
                           integration_time_s: float,
                           add_noise: bool = True,
                           random_state: Optional[np.random.RandomState] = None) -> Dict:
        """
        Complete detection chain simulation.

        Args:
            signal_e: Clean signal in electrons
            integration_time_s: Integration time
            add_noise: Whether to add noise
            random_state: Random state for reproducibility

        Returns:
            Dictionary with signal components and final DN
        """
        rng = np.random.RandomState() if random_state is None else random_state

        result = {
            'input_signal_e': signal_e.copy(),
            'integration_time_s': integration_time_s
        }

        if add_noise:
            # Apply PRNU (multiplicative)
            signal_prnu = self.add_prnu(signal_e, random_state=rng)
            result['prnu_signal_e'] = signal_prnu.copy()

            # Add shot noise
            signal_shot = self.add_shot_noise(signal_prnu, random_state=rng)
            result['shot_noise_e'] = signal_shot - signal_prnu

            # Add dark current
            dark = self.add_dark_current(signal_e.shape, integration_time_s, random_state=rng)
            result['dark_current_e'] = dark

            # Add read noise
            read = self.add_read_noise(signal_e.shape, random_state=rng)
            result['read_noise_e'] = read

            # Total signal
            total_e = signal_shot + dark + read

            # Clip to full well
            saturated = total_e > self.saturation_e
            result['saturated_fraction'] = np.mean(saturated)
            total_e = np.minimum(total_e, self.saturation_e)

        else:
            total_e = signal_e

        result['total_electrons'] = total_e

        # Convert to DN
        result['digital_number'] = self.electrons_to_dn(total_e)

        # Calculate SNR
        if add_noise:
            result['snr'] = self.calculate_snr(signal_e, integration_time_s)

        return result

    def calculate_snr(self, signal_e: np.ndarray,
                      integration_time_s: float) -> np.ndarray:
        """
        Calculate signal-to-noise ratio.

        SNR = S / √(S + D + R²)

        Args:
            signal_e: Signal in electrons
            integration_time_s: Integration time

        Returns:
            SNR array
        """
        # Noise components
        shot_variance = signal_e  # Poisson variance = mean
        dark_variance = self.dark_current_actual * integration_time_s
        read_variance = self.noise.read_noise_e ** 2

        # Total noise
        total_noise = np.sqrt(shot_variance + dark_variance + read_variance)

        # SNR
        snr = np.where(total_noise > 0, signal_e / total_noise, 0)

        return snr

    def noise_equivalent_radiance(self, wavelength_nm: float,
                                   integration_time_s: float,
                                   pixel_area_m2: float = 2e-9,
                                   solid_angle_sr: float = 1e-6,
                                   bandwidth_nm: float = 10) -> float:
        """
        Calculate Noise Equivalent Spectral Radiance (NESR).

        This is the radiance that produces SNR = 1.

        Args:
            wavelength_nm: Wavelength
            integration_time_s: Integration time
            pixel_area_m2: Pixel area
            solid_angle_sr: Pixel solid angle
            bandwidth_nm: Spectral bandwidth

        Returns:
            NESR in W/m²/sr/nm
        """
        # Total noise (electrons)
        dark_e = self.dark_current_actual * integration_time_s
        total_noise_e = np.sqrt(dark_e + self.noise.read_noise_e ** 2)

        # Convert to photons (inverse of QE)
        noise_photons = total_noise_e / self.noise.quantum_efficiency

        # Convert photons to radiance
        wavelength_m = wavelength_nm * 1e-9
        photon_energy = self.PLANCK * self.C / wavelength_m

        # Radiance = (N × E_photon) / (A × Ω × Δλ × t)
        energy_j = noise_photons * photon_energy
        nesr = energy_j / (pixel_area_m2 * solid_angle_sr * bandwidth_nm * integration_time_s)

        return nesr

    def get_noise_summary(self, integration_time_s: float = 0.01) -> str:
        """Get summary of noise characteristics."""
        n = self.noise

        # Calculate noise components for typical signal
        signal_e = 10000  # Example signal
        dark_e = self.dark_current_actual * integration_time_s

        summary = f"""
Detector Noise Model Summary
============================

Quantum Efficiency: {n.quantum_efficiency * 100:.0f}%
Full Well Capacity: {n.full_well_e:,.0f} e-
ADC: {n.adc_bits} bit ({self.max_dn + 1:,} levels)
Gain: {n.gain_e_per_dn:.1f} e-/DN

Noise Sources (for {integration_time_s*1000:.1f} ms integration):
  Read Noise: {n.read_noise_e:.1f} e- RMS
  Dark Current: {self.dark_current_actual:.1f} e-/s → {dark_e:.1f} e-
  Dark Shot Noise: {np.sqrt(dark_e):.1f} e- RMS

Operating Temperature: {n.temperature_c:.0f}°C
  (Dark current at 25°C: {n.dark_current_e_s:.1f} e-/s)

Fixed Pattern Noise:
  PRNU: {n.prnu_percent:.1f}%
  Dark FPN: {n.dark_fpn_percent:.1f}%

For signal = {signal_e:,} e-:
  Shot Noise: {np.sqrt(signal_e):.1f} e- RMS
  Total Noise: {np.sqrt(signal_e + dark_e + n.read_noise_e**2):.1f} e- RMS
  SNR: {signal_e / np.sqrt(signal_e + dark_e + n.read_noise_e**2):.1f}
"""
        return summary

    def explain_noise_sources(self) -> str:
        """Educational explanation of detector noise."""
        explanation = """
Understanding Detector Noise
============================

1. PHOTON (SHOT) NOISE - Fundamental & Unavoidable
   ─────────────────────────────────────────────────
   Light consists of discrete photons arriving randomly.
   Even with constant illumination, the count varies.

   Statistics: Poisson distribution
   Variance = Mean signal
   σ_shot = √N (N = number of photoelectrons)

   Example: 10,000 e- signal → ~100 e- noise (SNR = 100)

2. READ NOISE - Electronics Noise
   ─────────────────────────────────────────────────
   Uncertainty in counting electrons during readout.
   Added every time the detector is read.

   Statistics: Gaussian
   σ_read ≈ 5-50 e- for scientific CCDs

   Dominates at low light levels!

3. DARK CURRENT - Thermal Noise
   ─────────────────────────────────────────────────
   Heat causes random electron generation.
   Accumulates with exposure time.

   Temperature dependent: Doubles every ~7°C
   Solution: Cool the detector (-20 to -100°C)

4. FIXED PATTERN NOISE (FPN)
   ─────────────────────────────────────────────────
   Pixel-to-pixel sensitivity variation.
   Appears as a fixed pattern, not random.

   Corrected by flat-field calibration:
     Corrected = Raw / Flat_field

The SNR Equation:
                        Signal
   SNR = ─────────────────────────────────────────
         √(Shot² + Dark² + Read²)

              S
       = ─────────────
         √(S + D + R²)

At HIGH signal: Shot noise dominates → SNR ≈ √S
At LOW signal: Read noise dominates → SNR ≈ S/R
"""
        return explanation
