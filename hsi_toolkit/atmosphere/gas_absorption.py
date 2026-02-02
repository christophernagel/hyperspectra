"""
Gas Absorption Module

Models molecular absorption by atmospheric gases using simplified but
physically-grounded absorption band models.

Physics:
    Molecules absorb light at specific wavelengths corresponding to their
    vibrational and rotational energy transitions. The absorption strength
    depends on:
    - Number of molecules in the path (column density)
    - Absorption cross-section at each wavelength
    - Temperature and pressure (affecting line broadening)

Key Absorbers in Solar-Reflected Region (400-2500nm):
    - H2O: 720, 820, 940, 1140, 1380, 1880 nm bands
    - O2:  687, 760 nm (A and B bands)
    - CO2: 1430, 1570, 2010, 2060 nm bands
    - O3:  Chappuis band (450-750 nm, weak)
    - CH4: 1660, 2200 nm bands

References:
    - HITRAN database (https://hitran.org)
    - Gao & Goetz, 1990: Water vapor retrieval
    - Green et al., 1993: AVIRIS atmospheric correction
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class AbsorptionBand:
    """Defines a molecular absorption band."""
    center_nm: float       # Band center wavelength (nm)
    width_nm: float        # Band width (nm)
    strength: float        # Relative absorption strength (0-1)
    shape: str = 'gaussian'  # 'gaussian', 'lorentzian', or 'voigt'


class GasAbsorption:
    """
    Base class for gas absorption modeling.

    The Beer-Lambert Law governs absorption:
        T = exp(-tau)

    where tau (optical depth) = k * c * L
        k = absorption coefficient (wavelength dependent)
        c = concentration (molecules/cm³)
        L = path length (cm)

    For atmospheric calculations, we use column density (molecules/cm²)
    which integrates concentration over the path.
    """

    def __init__(self, wavelengths: np.ndarray):
        """
        Initialize with wavelength grid.

        Args:
            wavelengths: Wavelength array in nanometers
        """
        self.wavelengths = np.asarray(wavelengths)
        self.n_wavelengths = len(wavelengths)

    def absorption_coefficient(self, wavelength: np.ndarray) -> np.ndarray:
        """
        Calculate absorption coefficient k(lambda).
        Override in subclasses.

        Args:
            wavelength: Wavelengths in nm

        Returns:
            Absorption coefficient array
        """
        raise NotImplementedError

    def optical_depth(self, column_density: float) -> np.ndarray:
        """
        Calculate optical depth tau(lambda).

        Args:
            column_density: Column density in appropriate units

        Returns:
            Optical depth array
        """
        k = self.absorption_coefficient(self.wavelengths)
        return k * column_density

    def transmission(self, column_density: float) -> np.ndarray:
        """
        Calculate transmission T(lambda) = exp(-tau).

        Args:
            column_density: Column density

        Returns:
            Transmission array (0-1)
        """
        tau = self.optical_depth(column_density)
        return np.exp(-tau)

    @staticmethod
    def _gaussian_band(wavelengths: np.ndarray, center: float,
                       width: float, strength: float) -> np.ndarray:
        """Generate Gaussian absorption band shape."""
        sigma = width / 2.355  # FWHM to sigma
        return strength * np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

    @staticmethod
    def _lorentzian_band(wavelengths: np.ndarray, center: float,
                         width: float, strength: float) -> np.ndarray:
        """Generate Lorentzian absorption band shape."""
        gamma = width / 2
        return strength * (gamma ** 2) / ((wavelengths - center) ** 2 + gamma ** 2)


class WaterVaporAbsorption(GasAbsorption):
    """
    Water vapor (H2O) absorption model.

    Water vapor is the most important atmospheric absorber in the
    solar-reflected region. It varies significantly in space and time
    (0.5 - 5 g/cm² column water vapor).

    Key absorption bands:
        - 720 nm: Weak, good for detection
        - 820 nm: Moderate
        - 940 nm: Strong, commonly used for retrieval
        - 1140 nm: Strong
        - 1380 nm: Very strong (often saturated)
        - 1880 nm: Very strong (often saturated)
    """

    # Water vapor absorption bands (center_nm, width_nm, relative_strength)
    BANDS = [
        AbsorptionBand(720, 30, 0.15),
        AbsorptionBand(820, 25, 0.25),
        AbsorptionBand(940, 60, 0.85),
        AbsorptionBand(1140, 80, 0.90),
        AbsorptionBand(1380, 100, 0.98),   # Nearly opaque
        AbsorptionBand(1880, 150, 0.99),   # Nearly opaque
    ]

    def __init__(self, wavelengths: np.ndarray):
        super().__init__(wavelengths)
        self._precompute_band_shapes()

    def _precompute_band_shapes(self):
        """Precompute normalized band shapes for efficiency."""
        self._band_shapes = np.zeros((len(self.BANDS), self.n_wavelengths))
        for i, band in enumerate(self.BANDS):
            self._band_shapes[i] = self._gaussian_band(
                self.wavelengths, band.center_nm, band.width_nm, band.strength
            )

    def absorption_coefficient(self, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate H2O absorption coefficient.

        Uses sum of Gaussian bands. For educational purposes -
        real retrievals use HITRAN line-by-line calculations.
        """
        if wavelengths is None:
            wavelengths = self.wavelengths

        k = np.zeros_like(wavelengths, dtype=float)
        for band in self.BANDS:
            k += self._gaussian_band(wavelengths, band.center_nm,
                                    band.width_nm, band.strength)
        return k

    def transmission(self, pwv_cm: float, airmass: float = 1.0) -> np.ndarray:
        """
        Calculate water vapor transmission.

        Args:
            pwv_cm: Precipitable water vapor in cm (typical: 0.5-5 cm)
            airmass: Atmospheric path length factor (1.0 = vertical)

        Returns:
            Transmission spectrum (0-1)

        Note:
            PWV of 1 cm = 1 g/cm² = 10 kg/m² = 10 mm liquid equivalent
        """
        k = self.absorption_coefficient(self.wavelengths)
        # Scale absorption by PWV and airmass
        # Factor calibrated to match typical atmospheric transmission
        tau = k * pwv_cm * airmass * 3.0
        return np.exp(-tau)

    def estimate_pwv_from_ratio(self, radiance: np.ndarray,
                                band_940: Tuple[int, int],
                                shoulder_bands: Tuple[Tuple[int, int], ...]) -> float:
        """
        Estimate PWV from 940nm absorption band depth.

        This is the standard method used in atmospheric correction:
        compare the radiance at 940nm (absorbed) to the continuum
        (interpolated from shoulder wavelengths).

        Args:
            radiance: Measured radiance spectrum
            band_940: Index range for 940nm absorption
            shoulder_bands: Index ranges for continuum shoulders

        Returns:
            Estimated PWV in cm
        """
        # Get absorption band radiance
        L_abs = np.mean(radiance[band_940[0]:band_940[1]])

        # Get continuum from shoulders
        L_cont = 0
        for shoulder in shoulder_bands:
            L_cont += np.mean(radiance[shoulder[0]:shoulder[1]])
        L_cont /= len(shoulder_bands)

        # Band depth ratio
        ratio = L_abs / L_cont

        # Empirical relationship (calibrated from radiative transfer)
        # T = exp(-k * PWV) => PWV = -ln(T) / k
        # For 940nm band, k ≈ 0.85
        if ratio > 0 and ratio < 1:
            pwv = -np.log(ratio) / 0.85
        else:
            pwv = 1.0  # Default

        return np.clip(pwv, 0.1, 5.0)


class O2Absorption(GasAbsorption):
    """
    Oxygen (O2) absorption model.

    O2 is well-mixed in the atmosphere (~21%) so its absorption
    is very predictable. Used for atmospheric path length estimation.

    Key bands:
        - 687 nm: B-band (weak)
        - 760 nm: A-band (strong, narrow) - "oxygen absorption band"
    """

    BANDS = [
        AbsorptionBand(687, 5, 0.3),   # B-band
        AbsorptionBand(760, 15, 0.95),  # A-band
    ]

    def absorption_coefficient(self, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        if wavelengths is None:
            wavelengths = self.wavelengths

        k = np.zeros_like(wavelengths, dtype=float)
        for band in self.BANDS:
            # O2 bands are narrower, use Lorentzian for sharper shape
            k += self._lorentzian_band(wavelengths, band.center_nm,
                                       band.width_nm, band.strength)
        return k

    def transmission(self, airmass: float = 1.0,
                     surface_pressure_mb: float = 1013.25) -> np.ndarray:
        """
        Calculate O2 transmission.

        Args:
            airmass: Atmospheric path length factor
            surface_pressure_mb: Surface pressure in millibars

        Returns:
            Transmission spectrum
        """
        k = self.absorption_coefficient(self.wavelengths)
        # O2 column scales with pressure
        pressure_factor = surface_pressure_mb / 1013.25
        tau = k * pressure_factor * airmass * 2.0
        return np.exp(-tau)


class CO2Absorption(GasAbsorption):
    """
    Carbon dioxide (CO2) absorption model.

    CO2 is well-mixed (~420 ppm) and has several absorption bands
    in the SWIR region.

    Key bands:
        - 1430 nm: Weak
        - 1570 nm: Moderate
        - 2010 nm: Strong
        - 2060 nm: Strong
    """

    BANDS = [
        AbsorptionBand(1430, 30, 0.20),
        AbsorptionBand(1570, 40, 0.35),
        AbsorptionBand(2010, 30, 0.70),
        AbsorptionBand(2060, 40, 0.75),
    ]

    def absorption_coefficient(self, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        if wavelengths is None:
            wavelengths = self.wavelengths

        k = np.zeros_like(wavelengths, dtype=float)
        for band in self.BANDS:
            k += self._gaussian_band(wavelengths, band.center_nm,
                                    band.width_nm, band.strength)
        return k

    def transmission(self, airmass: float = 1.0,
                     co2_ppm: float = 420.0) -> np.ndarray:
        """
        Calculate CO2 transmission.

        Args:
            airmass: Atmospheric path length factor
            co2_ppm: CO2 concentration in ppm

        Returns:
            Transmission spectrum
        """
        k = self.absorption_coefficient(self.wavelengths)
        concentration_factor = co2_ppm / 420.0
        tau = k * concentration_factor * airmass * 1.5
        return np.exp(-tau)


class O3Absorption(GasAbsorption):
    """
    Ozone (O3) absorption model.

    O3 has weak but broad absorption in the visible (Chappuis band)
    and strong UV absorption (Hartley-Huggins bands, not modeled here).

    The Chappuis band gives the sky a subtle blue tint and affects
    atmospheric correction in the 450-750 nm range.
    """

    def absorption_coefficient(self, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        if wavelengths is None:
            wavelengths = self.wavelengths

        # Chappuis band: broad absorption centered ~600nm
        k = self._gaussian_band(wavelengths, 600, 200, 0.08)
        return k

    def transmission(self, ozone_du: float = 300.0,
                     airmass: float = 1.0) -> np.ndarray:
        """
        Calculate O3 transmission.

        Args:
            ozone_du: Total column ozone in Dobson Units (typical: 250-450 DU)
            airmass: Atmospheric path length factor

        Returns:
            Transmission spectrum
        """
        k = self.absorption_coefficient(self.wavelengths)
        # Scale by ozone amount
        ozone_factor = ozone_du / 300.0
        tau = k * ozone_factor * airmass
        return np.exp(-tau)


class CombinedGasAbsorption:
    """
    Combined absorption from all major atmospheric gases.

    Calculates total transmission as product of individual transmissions:
        T_total = T_H2O * T_O2 * T_CO2 * T_O3

    This is the standard approach assuming no overlap effects
    (reasonable approximation for well-separated bands).
    """

    def __init__(self, wavelengths: np.ndarray):
        """
        Initialize combined gas absorption model.

        Args:
            wavelengths: Wavelength array in nm
        """
        self.wavelengths = wavelengths
        self.h2o = WaterVaporAbsorption(wavelengths)
        self.o2 = O2Absorption(wavelengths)
        self.co2 = CO2Absorption(wavelengths)
        self.o3 = O3Absorption(wavelengths)

    def transmission(self,
                     pwv_cm: float = 1.0,
                     ozone_du: float = 300.0,
                     co2_ppm: float = 420.0,
                     surface_pressure_mb: float = 1013.25,
                     airmass: float = 1.0) -> np.ndarray:
        """
        Calculate combined gas transmission.

        Args:
            pwv_cm: Precipitable water vapor (cm)
            ozone_du: Column ozone (Dobson Units)
            co2_ppm: CO2 concentration (ppm)
            surface_pressure_mb: Surface pressure (mb)
            airmass: Path length factor

        Returns:
            Total gas transmission spectrum
        """
        T_h2o = self.h2o.transmission(pwv_cm, airmass)
        T_o2 = self.o2.transmission(airmass, surface_pressure_mb)
        T_co2 = self.co2.transmission(airmass, co2_ppm)
        T_o3 = self.o3.transmission(ozone_du, airmass)

        return T_h2o * T_o2 * T_co2 * T_o3

    def get_absorption_components(self,
                                   pwv_cm: float = 1.0,
                                   ozone_du: float = 300.0,
                                   co2_ppm: float = 420.0,
                                   surface_pressure_mb: float = 1013.25,
                                   airmass: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Get individual transmission components for visualization.

        Returns:
            Dictionary with transmission arrays for each gas
        """
        return {
            'H2O': self.h2o.transmission(pwv_cm, airmass),
            'O2': self.o2.transmission(airmass, surface_pressure_mb),
            'CO2': self.co2.transmission(airmass, co2_ppm),
            'O3': self.o3.transmission(ozone_du, airmass),
            'total': self.transmission(pwv_cm, ozone_du, co2_ppm,
                                       surface_pressure_mb, airmass)
        }
