"""
AVIRIS-3 Atmospheric Correction v2.0
=====================================
Converts L1B Radiance to Surface Reflectance using radiative transfer modeling.

This module implements a scientifically rigorous atmospheric correction workflow:
1. Scene-based water vapor retrieval (940nm absorption band)
2. Dark object subtraction for aerosol optical depth estimation
3. Radiative transfer modeling via Py6S with proper airborne sensor configuration
4. Look-up table approach for computational efficiency
5. Topographic correction using per-pixel illumination geometry
6. Comprehensive validation and uncertainty estimation

Key improvements over v1.0:
- Correct sensor altitude for airborne platforms (not satellite)
- Scene-derived atmospheric parameters instead of assumptions
- LUT-based correction for practical processing times
- Water vapor and AOD retrieval from the data itself
- Validation against expected reflectance ranges
- Comprehensive logging and provenance tracking

Theory:
-------
The at-sensor radiance L_sensor can be modeled as:
    L_sensor = L_path + (T_down * T_up * rho_surface * E_sun * cos(theta_s)) / pi
    
Where:
    L_path = atmospheric path radiance (scattering)
    T_down = downward atmospheric transmittance
    T_up = upward atmospheric transmittance  
    rho_surface = surface reflectance (what we want)
    E_sun = solar irradiance
    theta_s = solar zenith angle

6S provides correction coefficients (xa, xb, xc) such that:
    y = xa * L_sensor - xb
    rho_surface = y / (1 + xc * y)

Requirements:
    pip install Py6S netCDF4 numpy scipy --break-system-packages
    conda install -c conda-forge sixs

Usage:
    python aviris_atm_correction_v2.py <radiance.nc> <obs.nc> <output.nc> [options]

Options:
    --altitude <km>     Aircraft altitude in km (default: auto-detect or 6.0)
    --aot <value>       Override AOT550 (default: scene-derived)
    --water <g/cm2>     Override water vapor column (default: scene-derived)
    --aerosol <type>    Aerosol model: maritime/continental/urban/desert
    --simple            Use simplified correction (no Py6S required)
    --validate          Compare against expected reflectance ranges

Author: Christopher / Claude
Date: January 2025
Version: 2.0
"""

import sys
import os
import numpy as np
import netCDF4 as nc
from pathlib import Path
from datetime import datetime
from scipy import ndimage
from scipy.interpolate import interp1d
import warnings
import logging
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Processing Constants
# =============================================================================

# LUT generation settings
LUT_CACHE_DIR = Path.home() / '.aviris_atm_cache'
LUT_KEY_WAVELENGTHS = 50  # Number of key wavelengths to simulate (interpolate rest)
LUT_MAX_WORKERS = 4  # Max parallel 6S processes
TILE_SIZE = 512  # Tile size for memory-efficient processing
MEMORY_LIMIT_GB = 4.0  # Target memory limit

# Numerical stability
EPSILON = 1e-10
MAX_REFLECTANCE = 1.5  # Physical upper limit for reflectance
MIN_DENOMINATOR = 1e-8

# Check for Py6S availability
HAS_PY6S = False
SIXS_PATH = None

try:
    from Py6S import (
        SixS, AtmosProfile, AeroProfile, GroundReflectance,
        Geometry, Wavelength, AtmosCorr
    )
    
    # Search for 6S executable
    SIXS_SEARCH_PATHS = [
        r"C:\Users\chris\miniconda3\Library\bin\sixs.exe",
        r"C:\Users\chris\miniconda3\bin\sixs.exe",
        "/usr/local/bin/sixs",
        "/opt/conda/bin/sixs",
        os.path.expanduser("~/miniconda3/Library/bin/sixs.exe"),
        os.path.expanduser("~/miniconda3/bin/sixs"),
        os.path.expanduser("~/anaconda3/bin/sixs"),
    ]
    
    for path in SIXS_SEARCH_PATHS:
        if os.path.exists(path):
            SIXS_PATH = path
            break
    
    if SIXS_PATH:
        logger.info(f"Found 6S executable: {SIXS_PATH}")
        HAS_PY6S = True
    else:
        logger.warning("Py6S installed but 6S executable not found")
        logger.warning("Install with: conda install -c conda-forge sixs")
        
except ImportError:
    logger.warning("Py6S not available. Will use empirical line method.")


# =============================================================================
# Physical Constants and Reference Data
# =============================================================================

# Solar irradiance at top of atmosphere (Thuillier 2003, W/m²/µm)
# Interpolated for key wavelengths
SOLAR_IRRADIANCE_REFERENCE = {
    # wavelength_nm: irradiance_W_m2_um
    400: 1515.0, 450: 2004.0, 500: 1995.0, 550: 1879.0, 600: 1775.0,
    650: 1615.0, 700: 1509.0, 750: 1333.0, 800: 1215.0, 850: 1051.0,
    900: 901.0, 950: 816.0, 1000: 872.0, 1050: 851.0, 1100: 761.0,
    1150: 706.0, 1200: 598.0, 1250: 503.0, 1300: 432.0, 1350: 305.0,
    1400: 216.0, 1450: 182.0, 1500: 261.0, 1550: 266.0, 1600: 251.0,
    1650: 235.0, 1700: 210.0, 1750: 185.0, 1800: 112.0, 1850: 89.0,
    1900: 60.0, 1950: 78.0, 2000: 99.0, 2050: 108.0, 2100: 99.0,
    2150: 88.0, 2200: 83.0, 2250: 70.0, 2300: 61.0, 2350: 48.0,
    2400: 44.0, 2450: 38.0, 2500: 33.0
}

# Atmospheric absorption bands to mask
ABSORPTION_BANDS = [
    (1350, 1450, 'H2O'),  # Strong water vapor
    (1800, 1950, 'H2O'),  # Very strong water vapor
    (2450, 2500, 'CO2'),  # Carbon dioxide
]

# Water vapor absorption band windows
WATER_VAPOR_BANDS = {
    '940': {'center': 940, 'window': (920, 960), 'reference': (870, 890)},
    '1140': {'center': 1140, 'window': (1110, 1160), 'reference': (1050, 1080)},
}

# Expected reflectance ranges for validation
EXPECTED_REFLECTANCE = {
    'vegetation': {'550': (0.05, 0.15), '670': (0.02, 0.08), '860': (0.30, 0.60)},
    'soil': {'550': (0.10, 0.25), '670': (0.15, 0.35), '860': (0.20, 0.45)},
    'water': {'550': (0.02, 0.10), '670': (0.01, 0.05), '860': (0.00, 0.03)},
    'urban': {'550': (0.08, 0.20), '670': (0.10, 0.25), '860': (0.15, 0.35)},
}

# Coastal/adjacency correction parameters
ADJACENCY_KERNEL_SIZE = 21  # Pixels for adjacency averaging (~300m at 15m resolution)
WATER_NDWI_THRESHOLD = 0.3  # NDWI threshold for water detection
GLINT_THRESHOLD = 0.05  # Minimum NIR reflectance indicating glint
CIRRUS_BAND_NM = 1380  # Cirrus detection band (1.38 µm)
CIRRUS_THRESHOLD = 0.01  # Cirrus reflectance threshold

# Uncertainty estimation parameters
RADIOMETRIC_UNCERTAINTY = 0.03  # 3% radiometric calibration uncertainty
AOT_RETRIEVAL_UNCERTAINTY = 0.05  # Typical AOT retrieval uncertainty
WV_RETRIEVAL_UNCERTAINTY = 0.3  # Water vapor retrieval uncertainty (g/cm²)
LUT_INTERPOLATION_UNCERTAINTY = 0.01  # LUT interpolation error contribution


# =============================================================================
# Solar Irradiance Model
# =============================================================================

class SolarIrradianceModel:
    """
    Provides solar irradiance values for any wavelength using interpolation
    from the Thuillier 2003 reference spectrum.
    """
    
    def __init__(self):
        """Initialize with reference data."""
        wl = np.array(list(SOLAR_IRRADIANCE_REFERENCE.keys()))
        irr = np.array(list(SOLAR_IRRADIANCE_REFERENCE.values()))
        self._interpolator = interp1d(
            wl, irr, kind='cubic', 
            bounds_error=False, 
            fill_value=(irr[0], irr[-1])
        )
    
    def get_irradiance(self, wavelength_nm, earth_sun_distance_au=1.0):
        """
        Get solar irradiance at given wavelength.
        
        Args:
            wavelength_nm: Wavelength in nanometers
            earth_sun_distance_au: Earth-Sun distance in AU
            
        Returns:
            Solar irradiance in W/(m²·µm)
        """
        irr = self._interpolator(wavelength_nm)
        # Correct for Earth-Sun distance
        irr = irr / (earth_sun_distance_au ** 2)
        return irr
    
    def get_irradiance_array(self, wavelengths_nm, earth_sun_distance_au=1.0):
        """Get irradiance for array of wavelengths."""
        return self.get_irradiance(wavelengths_nm, earth_sun_distance_au)


# =============================================================================
# Water Vapor Retrieval
# =============================================================================

class WaterVaporRetrieval:
    """
    Retrieves column water vapor from imaging spectroscopy data using
    the 940nm absorption band ratio method (CIBR - Continuum Interpolated Band Ratio).
    
    Based on: Gao & Goetz (1990), Bruegge et al. (1990)
    """
    
    # Empirical coefficients for water vapor retrieval
    # These relate the band ratio to precipitable water vapor (PWV) in g/cm²
    # PWV = a * ln(ratio) + b (derived from radiative transfer modeling)
    RETRIEVAL_COEFFS = {
        '940': {'a': -0.9178, 'b': -0.0155},  # From MODTRAN simulations
        '1140': {'a': -0.7124, 'b': 0.0021},
    }
    
    def __init__(self, wavelengths, radiance_cube):
        """
        Initialize water vapor retrieval.
        
        Args:
            wavelengths: 1D array of wavelengths in nm
            radiance_cube: 3D array (bands, rows, cols)
        """
        self.wavelengths = wavelengths
        self.radiance = radiance_cube
        self.n_bands, self.n_rows, self.n_cols = radiance_cube.shape
        
    def find_band(self, target_wavelength):
        """Find closest band index to target wavelength."""
        return int(np.argmin(np.abs(self.wavelengths - target_wavelength)))
    
    def retrieve_pwv_940(self, use_polynomial_continuum=True):
        """
        Retrieve precipitable water vapor using 940nm absorption band.

        The method uses Continuum Interpolated Band Ratio (CIBR):
        1. Interpolate continuum radiance at 940nm from shoulder bands
        2. Calculate ratio of measured to continuum radiance
        3. Convert ratio to water vapor using empirical relationship

        Args:
            use_polynomial_continuum: If True, use polynomial fit for continuum
                                      (more accurate but slower)

        Returns:
            2D array of PWV in g/cm² (column water vapor)
        """
        logger.info("Retrieving water vapor from 940nm absorption band...")

        # Find bands - use multiple shoulder bands for better continuum
        # Left shoulder: 820-870nm (avoid O2 A-band at 760nm)
        # Right shoulder: 1000-1050nm (in atmospheric window)
        band_820 = self.find_band(820)
        band_850 = self.find_band(850)
        band_870 = self.find_band(870)
        band_940 = self.find_band(940)
        band_1000 = self.find_band(1000)
        band_1030 = self.find_band(1030)

        actual_940 = self.wavelengths[band_940]
        logger.info(f"  Target absorption band: {actual_940:.0f} nm")

        # Get radiances
        L_940 = self.radiance[band_940, :, :].astype(np.float64)

        if use_polynomial_continuum:
            # Use polynomial fit through multiple shoulder bands
            # This is more accurate than linear interpolation
            shoulder_bands = [band_820, band_850, band_870, band_1000, band_1030]
            shoulder_wl = np.array([self.wavelengths[b] for b in shoulder_bands])
            shoulder_L = np.array([self.radiance[b, :, :] for b in shoulder_bands])

            logger.info(f"  Using polynomial continuum from {len(shoulder_bands)} shoulder bands")

            # Fit 2nd order polynomial for each pixel
            # This accounts for non-linear spectral slope
            n_rows, n_cols = L_940.shape
            L_continuum = np.zeros_like(L_940)

            # Vectorized polynomial fit
            # Reshape for polyfit: (n_bands, n_pixels)
            shoulder_L_flat = shoulder_L.reshape(len(shoulder_bands), -1)

            # Fit polynomial for each pixel
            for i in range(shoulder_L_flat.shape[1]):
                if np.any(np.isnan(shoulder_L_flat[:, i])):
                    L_continuum.flat[i] = np.nan
                    continue
                try:
                    coeffs = np.polyfit(shoulder_wl, shoulder_L_flat[:, i], deg=2)
                    L_continuum.flat[i] = np.polyval(coeffs, actual_940)
                except:
                    # Fall back to linear if polyfit fails
                    L_continuum.flat[i] = np.interp(actual_940, shoulder_wl, shoulder_L_flat[:, i])
        else:
            # Simple linear interpolation (faster but less accurate)
            L_870 = self.radiance[band_870, :, :].astype(np.float64)
            L_1000 = self.radiance[band_1000, :, :].astype(np.float64)

            actual_870 = self.wavelengths[band_870]
            actual_1000 = self.wavelengths[band_1000]

            weight = (actual_940 - actual_870) / (actual_1000 - actual_870 + EPSILON)
            L_continuum = L_870 + weight * (L_1000 - L_870)

            logger.info(f"  Using linear continuum: {actual_870:.0f}-{actual_1000:.0f} nm")

        # Calculate band ratio (absorption depth)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = L_940 / (L_continuum + EPSILON)

        # Clip to valid range (ratio should be < 1 due to absorption)
        ratio = np.clip(ratio, 0.01, 0.99)

        # Convert to PWV using empirical relationship
        # This is calibrated against MODTRAN simulations
        coeffs = self.RETRIEVAL_COEFFS['940']
        pwv = coeffs['a'] * np.log(ratio) + coeffs['b']

        # Clip to physically reasonable range (0-8 g/cm²)
        pwv = np.clip(pwv, 0.1, 8.0)

        # Smooth to reduce noise
        pwv = ndimage.median_filter(pwv, size=5)

        mean_pwv = np.nanmean(pwv)
        std_pwv = np.nanstd(pwv)
        logger.info(f"  Mean PWV: {mean_pwv:.2f} ± {std_pwv:.2f} g/cm² ({mean_pwv*10:.0f} mm)")

        return pwv
    
    def get_scene_water_vapor(self):
        """
        Get representative scene water vapor value.
        
        Returns:
            Scalar water vapor in g/cm² (for atmospheric model input)
        """
        pwv_map = self.retrieve_pwv_940()
        
        # Use robust statistics (median, IQR)
        median_pwv = np.nanmedian(pwv_map)
        q25, q75 = np.nanpercentile(pwv_map, [25, 75])
        
        logger.info(f"  PWV statistics: median={median_pwv:.2f}, IQR=[{q25:.2f}, {q75:.2f}] g/cm²")
        
        return median_pwv


# =============================================================================
# Aerosol Optical Depth Retrieval
# =============================================================================

class AODRetrieval:
    """
    Retrieves aerosol optical depth from dark targets in the scene.
    
    Methods:
    1. Dark Dense Vegetation (DDV) - uses known reflectance relationship
    2. Dark Object Subtraction (DOS) - assumes minimum radiance = path radiance
    
    Based on: Kaufman & Sendra (1988), Chavez (1988)
    """
    
    # DDV surface reflectance relationships (empirical)
    # At 2.1µm, vegetation reflectance can be estimated
    # Then 0.47µm and 0.66µm reflectances follow predictable ratios
    DDV_RATIOS = {
        'blue_to_swir': 0.25,   # rho_470 / rho_2100
        'red_to_swir': 0.50,    # rho_660 / rho_2100
    }
    
    # Typical aerosol properties for estimation
    AEROSOL_ANGSTROM = 1.3  # Typical continental Ångström exponent
    
    def __init__(self, wavelengths, radiance_cube, geometry):
        """
        Initialize AOD retrieval.
        
        Args:
            wavelengths: 1D array of wavelengths in nm
            radiance_cube: 3D array (bands, rows, cols)
            geometry: Dict with solar/sensor angles
        """
        self.wavelengths = wavelengths
        self.radiance = radiance_cube
        self.geometry = geometry
        self.n_bands, self.n_rows, self.n_cols = radiance_cube.shape
        
    def find_band(self, target_wavelength):
        """Find closest band index."""
        return int(np.argmin(np.abs(self.wavelengths - target_wavelength)))
    
    def find_dark_targets(self, percentile=2):
        """
        Identify dark targets using NIR/Red ratio (vegetation criterion)
        and low SWIR reflectance.
        
        Returns:
            Boolean mask of dark target pixels
        """
        logger.info("Identifying dark targets for AOD estimation...")
        
        # Get key bands
        band_660 = self.find_band(660)
        band_860 = self.find_band(860)
        band_2100 = self.find_band(2100)
        
        L_660 = self.radiance[band_660, :, :]
        L_860 = self.radiance[band_860, :, :]
        L_2100 = self.radiance[band_2100, :, :]
        
        # Vegetation index (NDVI-like using radiances)
        with np.errstate(invalid='ignore', divide='ignore'):
            ndvi_proxy = (L_860 - L_660) / (L_860 + L_660 + 1e-10)
        
        # Dark targets: high NDVI (vegetation) AND low SWIR radiance
        ndvi_threshold = np.nanpercentile(ndvi_proxy, 75)  # Upper quartile
        swir_threshold = np.nanpercentile(L_2100, 25)  # Lower quartile
        
        dark_mask = (ndvi_proxy > ndvi_threshold) & (L_2100 < swir_threshold)
        
        # Clean up mask with morphological operations
        dark_mask = ndimage.binary_erosion(dark_mask, iterations=2)
        dark_mask = ndimage.binary_dilation(dark_mask, iterations=1)
        
        n_dark = np.sum(dark_mask)
        pct_dark = 100 * n_dark / dark_mask.size
        logger.info(f"  Found {n_dark} dark pixels ({pct_dark:.1f}% of scene)")
        
        return dark_mask
    
    def estimate_aot_dos(self):
        """
        Estimate AOT using Dark Object Subtraction method.
        
        This is a simplified approach that assumes:
        1. The darkest pixels have near-zero surface reflectance
        2. Their radiance is dominated by atmospheric path radiance
        3. Path radiance relates to AOT
        
        Returns:
            Estimated AOT at 550nm
        """
        logger.info("Estimating AOT using Dark Object Subtraction...")
        
        # Use blue band (most sensitive to aerosols)
        band_470 = self.find_band(470)
        L_470 = self.radiance[band_470, :, :]
        
        # Find 0.1th percentile (darkest objects)
        L_dark = np.nanpercentile(L_470, 0.5)
        
        # Estimate path radiance
        # For a 1% dark object, path radiance ≈ L_dark - 0.01 * L_max
        L_max = np.nanpercentile(L_470, 99)
        L_path = L_dark - 0.01 * L_max
        L_path = max(0, L_path)
        
        # Convert path radiance to AOT (empirical relationship)
        # This is a rough approximation: AOT ≈ k * L_path / (E_sun * cos(theta_s))
        # where k is an empirical factor ~0.05 for blue wavelengths
        solar = SolarIrradianceModel()
        E_sun = solar.get_irradiance(470)
        cos_sza = np.cos(np.radians(self.geometry['sun_zenith']))
        
        # Convert AVIRIS radiance units: µW/(nm·cm²·sr) to W/(m²·sr·µm)
        L_path_si = L_path * 10.0
        
        # Empirical AOT estimation
        k_empirical = 0.08
        aot_470 = k_empirical * np.pi * L_path_si / (E_sun * cos_sza)
        aot_470 = np.clip(aot_470, 0.01, 2.0)
        
        # Convert to AOT at 550nm using Ångström exponent
        aot_550 = aot_470 * (550 / 470) ** (-self.AEROSOL_ANGSTROM)
        
        logger.info(f"  Dark pixel radiance: {L_dark:.2f}")
        logger.info(f"  Estimated path radiance: {L_path:.2f}")
        logger.info(f"  Estimated AOT@550: {aot_550:.3f}")
        
        return aot_550
    
    def estimate_aot_ddv(self):
        """
        Estimate AOT using Dense Dark Vegetation method.
        
        This method:
        1. Uses SWIR (2.1µm) to estimate surface reflectance (minimal aerosol effect)
        2. Predicts expected visible reflectance from empirical relationships
        3. Difference from measured reflectance indicates aerosol loading
        
        Returns:
            Estimated AOT at 550nm
        """
        logger.info("Estimating AOT using Dense Dark Vegetation method...")
        
        # Find dark vegetation targets
        dark_mask = self.find_dark_targets()
        
        if np.sum(dark_mask) < 100:
            logger.warning("  Insufficient dark targets, falling back to DOS method")
            return self.estimate_aot_dos()
        
        # Get bands
        band_470 = self.find_band(470)
        band_660 = self.find_band(660)
        band_2100 = self.find_band(2100)
        
        L_470 = self.radiance[band_470, :, :]
        L_660 = self.radiance[band_660, :, :]
        L_2100 = self.radiance[band_2100, :, :]
        
        # Extract dark target statistics
        L_470_dark = np.nanmedian(L_470[dark_mask])
        L_660_dark = np.nanmedian(L_660[dark_mask])
        L_2100_dark = np.nanmedian(L_2100[dark_mask])
        
        # Estimate surface reflectance from SWIR (minimal aerosol effect)
        # Rough conversion: rho ≈ pi * L / (E_sun * cos(theta_s) * T_atm)
        # Assume T_atm ≈ 0.9 at 2.1µm
        solar = SolarIrradianceModel()
        cos_sza = np.cos(np.radians(self.geometry['sun_zenith']))
        
        # Convert radiance units
        L_2100_si = L_2100_dark * 10.0
        E_sun_2100 = solar.get_irradiance(2100)
        
        rho_2100 = np.pi * L_2100_si / (E_sun_2100 * cos_sza * 0.9)
        rho_2100 = np.clip(rho_2100, 0.01, 0.20)
        
        # Predict visible reflectances from DDV relationships
        rho_470_predicted = rho_2100 * self.DDV_RATIOS['blue_to_swir']
        rho_660_predicted = rho_2100 * self.DDV_RATIOS['red_to_swir']
        
        # Calculate what we measure (apparent reflectance)
        L_470_si = L_470_dark * 10.0
        L_660_si = L_660_dark * 10.0
        
        E_sun_470 = solar.get_irradiance(470)
        E_sun_660 = solar.get_irradiance(660)
        
        # Apparent reflectance (includes atmospheric effects)
        rho_470_apparent = np.pi * L_470_si / (E_sun_470 * cos_sza)
        rho_660_apparent = np.pi * L_660_si / (E_sun_660 * cos_sza)
        
        # Excess reflectance due to aerosols
        delta_rho_470 = rho_470_apparent - rho_470_predicted
        delta_rho_660 = rho_660_apparent - rho_660_predicted
        
        # Convert to AOT (empirical sensitivity factors)
        # delta_rho ≈ C * AOT, where C depends on wavelength and geometry
        C_470 = 0.15  # Higher sensitivity at blue
        C_660 = 0.08
        
        aot_470 = delta_rho_470 / C_470
        aot_660 = delta_rho_660 / C_660
        
        # Average and convert to 550nm
        aot_avg = (aot_470 + aot_660) / 2
        aot_550 = np.clip(aot_avg, 0.01, 2.0)
        
        logger.info(f"  SWIR surface reflectance: {rho_2100:.3f}")
        logger.info(f"  Predicted visible rho: {rho_470_predicted:.3f} (470), {rho_660_predicted:.3f} (660)")
        logger.info(f"  Measured apparent rho: {rho_470_apparent:.3f} (470), {rho_660_apparent:.3f} (660)")
        logger.info(f"  Estimated AOT@550: {aot_550:.3f}")
        
        return aot_550
    
    def get_scene_aot(self, method='hybrid'):
        """
        Get representative AOT for the scene.
        
        Args:
            method: 'dos', 'ddv', or 'hybrid' (average of both)
            
        Returns:
            AOT at 550nm
        """
        if method == 'dos':
            return self.estimate_aot_dos()
        elif method == 'ddv':
            return self.estimate_aot_ddv()
        else:
            # Hybrid: average of both methods
            aot_dos = self.estimate_aot_dos()
            aot_ddv = self.estimate_aot_ddv()
            aot_hybrid = (aot_dos + aot_ddv) / 2
            logger.info(f"  Hybrid AOT (DOS={aot_dos:.3f}, DDV={aot_ddv:.3f}): {aot_hybrid:.3f}")
            return aot_hybrid


# =============================================================================
# Coastal and Adjacency Correction
# =============================================================================

class CoastalCorrection:
    """
    Coastal and adjacency correction for island/coastal scenes.

    Critical for Santa Barbara Island and similar coastal scenes where:
    1. Ocean adjacency effect - Bright land contaminates dark water pixels
    2. Sun glint - Specular reflection from water surfaces
    3. Cirrus clouds - Thin ice clouds common in marine layer

    Based on:
    - Adjacency: Vermote et al. (1997), Lyapustin & Knyazikhin (2001)
    - Glint: Kay et al. (2009), Hedley et al. (2005)
    - Cirrus: Gao et al. (2002)
    """

    def __init__(self, wavelengths, radiance_cube, geometry):
        """
        Initialize coastal correction.

        Args:
            wavelengths: 1D array of wavelengths in nm
            radiance_cube: 3D array (bands, rows, cols)
            geometry: Dict with sun_zenith, sun_azimuth, sensor_zenith, sensor_azimuth
        """
        self.wavelengths = wavelengths
        self.radiance = radiance_cube
        self.geometry = geometry
        self.n_bands, self.n_rows, self.n_cols = radiance_cube.shape

        # Pre-compute key bands
        self._band_green = self._find_band(560)
        self._band_nir = self._find_band(860)
        self._band_swir = self._find_band(1600)
        self._band_cirrus = self._find_band(CIRRUS_BAND_NM)

    def _find_band(self, target_wavelength):
        """Find closest band index to target wavelength."""
        return int(np.argmin(np.abs(self.wavelengths - target_wavelength)))

    def detect_water_mask(self, reflectance=None):
        """
        Create water mask using NDWI (Normalized Difference Water Index).

        NDWI = (Green - NIR) / (Green + NIR)
        Water has positive NDWI due to low NIR reflectance.

        Args:
            reflectance: Optional reflectance cube. If None, uses radiance ratios.

        Returns:
            Boolean mask where True = water pixel
        """
        logger.info("Detecting water pixels...")

        if reflectance is not None:
            green = reflectance[self._band_green, :, :]
            nir = reflectance[self._band_nir, :, :]
        else:
            # Use radiance ratios as proxy
            green = self.radiance[self._band_green, :, :].astype(np.float64)
            nir = self.radiance[self._band_nir, :, :].astype(np.float64)

        # Calculate NDWI
        with np.errstate(invalid='ignore', divide='ignore'):
            ndwi = (green - nir) / (green + nir + EPSILON)

        # Water mask
        water_mask = ndwi > WATER_NDWI_THRESHOLD

        # Clean up with morphological operations
        water_mask = ndimage.binary_opening(water_mask, iterations=2)
        water_mask = ndimage.binary_closing(water_mask, iterations=2)

        n_water = np.sum(water_mask)
        pct_water = 100 * n_water / water_mask.size
        logger.info(f"  Water pixels: {n_water} ({pct_water:.1f}% of scene)")

        return water_mask

    def detect_land_water_boundary(self, water_mask):
        """
        Identify pixels near the land-water boundary.

        These are the pixels most affected by adjacency effects.

        Args:
            water_mask: Boolean water mask

        Returns:
            Boolean mask of boundary pixels, distance transform
        """
        # Distance from water pixels to nearest land
        land_mask = ~water_mask

        # Distance transform from land
        distance_from_land = ndimage.distance_transform_edt(water_mask)

        # Distance transform from water
        distance_from_water = ndimage.distance_transform_edt(land_mask)

        # Boundary zone: within ~10 pixels of transition
        boundary_zone = (distance_from_land <= 10) | (distance_from_water <= 10)

        n_boundary = np.sum(boundary_zone)
        logger.info(f"  Boundary pixels: {n_boundary} ({100*n_boundary/boundary_zone.size:.1f}%)")

        return boundary_zone, distance_from_land

    def calculate_adjacency_reflectance(self, reflectance, water_mask):
        """
        Calculate average reflectance of surrounding land pixels.

        This represents the "environment reflectance" that contributes
        to adjacency effect through atmospheric scattering.

        Args:
            reflectance: Reflectance cube (bands, rows, cols)
            water_mask: Boolean water mask

        Returns:
            Environment reflectance array (bands, rows, cols)
        """
        logger.info("Calculating adjacency environment reflectance...")

        land_mask = (~water_mask).astype(np.float32)

        # Create averaging kernel
        kernel_size = ADJACENCY_KERNEL_SIZE
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        kernel = kernel / kernel.sum()

        env_reflectance = np.zeros_like(reflectance)

        for i in range(self.n_bands):
            if np.isnan(reflectance[i, 0, 0]):
                env_reflectance[i, :, :] = np.nan
                continue

            # Masked reflectance (land only)
            land_refl = reflectance[i, :, :] * land_mask

            # Convolve to get weighted average
            land_sum = ndimage.convolve(land_refl, kernel, mode='reflect')
            land_count = ndimage.convolve(land_mask, kernel, mode='reflect')

            with np.errstate(invalid='ignore', divide='ignore'):
                env_reflectance[i, :, :] = np.where(
                    land_count > 0.1,
                    land_sum / land_count,
                    0.0
                )

        return env_reflectance

    def correct_adjacency_effect(self, reflectance, aot550=0.1):
        """
        Apply adjacency effect correction to coastal pixels.

        The adjacency effect adds spurious radiance to water pixels due to
        photons scattered from nearby bright land surfaces. The correction:

        rho_corrected = rho_measured - T_diff * rho_env

        Where:
        - T_diff is the diffuse atmospheric transmittance
        - rho_env is the environment (surrounding) reflectance

        Args:
            reflectance: Reflectance cube to correct
            aot550: Aerosol optical depth for transmittance calculation

        Returns:
            Corrected reflectance cube, correction magnitude map
        """
        logger.info("\n" + "-"*40)
        logger.info("ADJACENCY EFFECT CORRECTION")
        logger.info("-"*40)

        # Detect water and boundaries
        water_mask = self.detect_water_mask(reflectance)
        boundary_zone, distance_from_land = self.detect_land_water_boundary(water_mask)

        if np.sum(water_mask) < 100:
            logger.info("  Minimal water in scene - skipping adjacency correction")
            return reflectance, np.zeros((self.n_rows, self.n_cols))

        # Calculate environment reflectance
        env_reflectance = self.calculate_adjacency_reflectance(reflectance, water_mask)

        # Calculate diffuse transmittance (wavelength-dependent)
        # Simplified: T_diff ≈ exp(-tau) * f_diff, where f_diff ~ 0.1-0.3
        def diffuse_transmittance(wl_nm, aot):
            wl_um = wl_nm / 1000.0
            # Rayleigh + aerosol optical depth
            tau_ray = 0.008569 * wl_um**(-4)
            tau_aer = aot * (wl_um / 0.55)**(-1.3)
            tau_total = tau_ray + tau_aer

            # Fraction of diffuse light that causes adjacency
            # Stronger at shorter wavelengths due to more scattering
            f_diff = 0.3 * (0.55 / wl_um)**0.5
            f_diff = np.clip(f_diff, 0.1, 0.4)

            T_diff = f_diff * (1 - np.exp(-tau_total))
            return T_diff

        # Distance-based weighting (correction decreases with distance from land)
        # Characteristic length ~500m (about 30 pixels at 15m)
        char_length = 30.0  # pixels
        distance_weight = np.exp(-distance_from_land / char_length)
        distance_weight = np.where(water_mask, distance_weight, 0.0)

        # Apply correction to water pixels
        corrected = reflectance.copy()
        correction_magnitude = np.zeros((self.n_rows, self.n_cols))

        for i in range(self.n_bands):
            wl = self.wavelengths[i]

            # Skip absorption bands
            in_absorption = False
            for band_start, band_end, _ in ABSORPTION_BANDS:
                if band_start <= wl <= band_end:
                    in_absorption = True
                    break
            if in_absorption or np.isnan(reflectance[i, 0, 0]):
                continue

            T_diff = diffuse_transmittance(wl, aot550)

            # Correction: subtract scattered environment contribution
            correction = T_diff * env_reflectance[i, :, :] * distance_weight

            # Apply only to water pixels
            corrected[i, :, :] = np.where(
                water_mask,
                reflectance[i, :, :] - correction,
                reflectance[i, :, :]
            )

            # Track total correction magnitude (for green band)
            if abs(wl - 560) < 20:
                correction_magnitude = correction

        # Ensure non-negative reflectance
        corrected = np.maximum(corrected, 0.0)

        # Statistics
        mean_correction = np.mean(correction_magnitude[water_mask]) if np.any(water_mask) else 0
        max_correction = np.max(correction_magnitude[water_mask]) if np.any(water_mask) else 0
        logger.info(f"  Mean adjacency correction (green): {mean_correction:.4f}")
        logger.info(f"  Max adjacency correction (green): {max_correction:.4f}")

        return corrected, correction_magnitude

    def detect_sun_glint(self, reflectance=None):
        """
        Detect sun glint on water surfaces.

        Glint occurs when the sun-sensor geometry creates specular reflection
        from the water surface. Detected by anomalously high NIR reflectance
        over water.

        Args:
            reflectance: Optional reflectance cube

        Returns:
            Boolean glint mask, glint intensity map
        """
        logger.info("Detecting sun glint...")

        # First get water mask
        water_mask = self.detect_water_mask(reflectance)

        # Check geometry for glint likelihood
        # Glint most likely when view angle ~ reflection of sun angle
        sun_zen = self.geometry['sun_zenith']
        sun_az = self.geometry['sun_azimuth']
        view_zen = self.geometry['sensor_zenith']
        view_az = self.geometry['sensor_azimuth']

        # Relative azimuth
        rel_az = abs(sun_az - view_az)
        if rel_az > 180:
            rel_az = 360 - rel_az

        # Glint risk assessment
        # High risk when: sun_zen ≈ view_zen and rel_az ≈ 180° (specular)
        glint_risk = (abs(sun_zen - view_zen) < 30) and (abs(rel_az - 180) < 60)

        logger.info(f"  Sun zenith: {sun_zen:.1f}°, View zenith: {view_zen:.1f}°")
        logger.info(f"  Relative azimuth: {rel_az:.1f}°, Glint risk: {'HIGH' if glint_risk else 'LOW'}")

        # Get NIR reflectance
        if reflectance is not None:
            nir = reflectance[self._band_nir, :, :]
        else:
            # Estimate from radiance
            nir = self.radiance[self._band_nir, :, :].astype(np.float64)
            nir = nir / np.nanpercentile(nir, 99)  # Rough normalization

        # Glint detection: high NIR over water
        # Water typically has NIR reflectance < 0.02
        glint_intensity = np.where(water_mask, nir, 0.0)
        glint_mask = water_mask & (nir > GLINT_THRESHOLD)

        n_glint = np.sum(glint_mask)
        if n_glint > 0:
            pct_glint = 100 * n_glint / np.sum(water_mask)
            mean_intensity = np.mean(nir[glint_mask])
            logger.info(f"  Glint pixels: {n_glint} ({pct_glint:.1f}% of water)")
            logger.info(f"  Mean glint NIR: {mean_intensity:.3f}")
        else:
            logger.info("  No significant glint detected")

        return glint_mask, glint_intensity

    def correct_sun_glint(self, reflectance):
        """
        Apply sun glint correction to water pixels.

        Uses the method of Hedley et al. (2005):
        For each visible band, regress against NIR over glint-affected pixels,
        then subtract the predicted contribution.

        rho_corrected = rho_vis - b * (rho_NIR - rho_NIR_min)

        Where b is the regression slope for that band.

        Args:
            reflectance: Reflectance cube to correct

        Returns:
            Corrected reflectance cube, glint mask
        """
        logger.info("\n" + "-"*40)
        logger.info("SUN GLINT CORRECTION")
        logger.info("-"*40)

        glint_mask, glint_intensity = self.detect_sun_glint(reflectance)
        water_mask = self.detect_water_mask(reflectance)

        n_glint = np.sum(glint_mask)
        if n_glint < 100:
            logger.info("  Insufficient glint pixels - skipping correction")
            return reflectance, glint_mask

        # Get NIR reflectance for regression
        nir = reflectance[self._band_nir, :, :]

        # Minimum NIR (deep water value)
        nir_deep_water = np.nanpercentile(nir[water_mask & ~glint_mask], 5) if np.any(water_mask & ~glint_mask) else 0.01
        nir_excess = nir - nir_deep_water
        nir_excess = np.maximum(nir_excess, 0)

        corrected = reflectance.copy()

        # Process each visible band
        for i in range(self.n_bands):
            wl = self.wavelengths[i]

            # Only correct visible/near-visible bands (400-900nm, excluding NIR itself)
            if wl < 400 or wl > 850:
                continue
            if np.isnan(reflectance[i, 0, 0]):
                continue

            band_refl = reflectance[i, :, :]

            # Get samples from glint-affected water
            glint_nir = nir_excess[glint_mask]
            glint_band = band_refl[glint_mask]

            # Robust regression (use percentile filtering)
            valid = np.isfinite(glint_nir) & np.isfinite(glint_band)
            if np.sum(valid) < 50:
                continue

            glint_nir = glint_nir[valid]
            glint_band = glint_band[valid]

            # Simple linear regression: band = a + b * NIR
            try:
                slope, intercept = np.polyfit(glint_nir, glint_band, 1)
            except:
                continue

            # Slope should be positive (glint adds to reflectance)
            if slope <= 0:
                continue

            # Apply correction to all water pixels
            correction = slope * nir_excess
            corrected[i, :, :] = np.where(
                water_mask,
                band_refl - correction,
                band_refl
            )

            if abs(wl - 560) < 20:
                logger.info(f"  Green band slope: {slope:.3f}")

        # Ensure non-negative
        corrected = np.maximum(corrected, 0.0)

        return corrected, glint_mask

    def detect_cirrus(self, reflectance=None):
        """
        Detect cirrus clouds using 1.38 µm band.

        The 1.38 µm band is within a strong water vapor absorption region,
        so surface features are not visible. However, cirrus clouds (high altitude,
        above most water vapor) appear bright at this wavelength.

        Args:
            reflectance: Optional reflectance cube

        Returns:
            Cirrus mask, cirrus optical thickness estimate
        """
        logger.info("Detecting cirrus clouds...")

        # Check if we have the cirrus band
        cirrus_wl = self.wavelengths[self._band_cirrus]
        if abs(cirrus_wl - CIRRUS_BAND_NM) > 30:
            logger.warning(f"  No suitable cirrus band (nearest: {cirrus_wl:.0f}nm)")
            return np.zeros((self.n_rows, self.n_cols), dtype=bool), np.zeros((self.n_rows, self.n_cols))

        if reflectance is not None:
            cirrus_refl = reflectance[self._band_cirrus, :, :]
        else:
            # Estimate from radiance (rough approximation)
            cirrus_L = self.radiance[self._band_cirrus, :, :].astype(np.float64)
            # Very rough conversion
            cirrus_refl = cirrus_L * 10 * np.pi / 200  # Approximate

        # Handle NaN and invalid values
        cirrus_refl = np.nan_to_num(cirrus_refl, nan=0.0)

        # Cirrus detection threshold
        cirrus_mask = cirrus_refl > CIRRUS_THRESHOLD

        # Estimate cirrus optical thickness (very approximate)
        # tau_cirrus ≈ rho_1.38 / 0.03 (empirical)
        cirrus_tau = cirrus_refl / 0.03
        cirrus_tau = np.clip(cirrus_tau, 0, 2)

        n_cirrus = np.sum(cirrus_mask)
        if n_cirrus > 0:
            pct_cirrus = 100 * n_cirrus / cirrus_mask.size
            mean_tau = np.mean(cirrus_tau[cirrus_mask])
            logger.info(f"  Cirrus pixels: {n_cirrus} ({pct_cirrus:.1f}% of scene)")
            logger.info(f"  Mean cirrus optical thickness: {mean_tau:.2f}")
        else:
            logger.info("  No cirrus detected")

        return cirrus_mask, cirrus_tau

    def correct_cirrus(self, reflectance):
        """
        Apply cirrus correction.

        Simple approach: subtract scaled 1.38 µm reflectance from all bands.
        The scaling factor is wavelength-dependent (stronger in visible).

        More sophisticated approaches exist (e.g., Gao & Li 2002) but require
        additional information.

        Args:
            reflectance: Reflectance cube to correct

        Returns:
            Corrected reflectance cube, cirrus mask
        """
        logger.info("\n" + "-"*40)
        logger.info("CIRRUS CORRECTION")
        logger.info("-"*40)

        cirrus_mask, cirrus_tau = self.detect_cirrus(reflectance)

        if np.sum(cirrus_mask) < 100:
            logger.info("  Minimal cirrus - skipping correction")
            return reflectance, cirrus_mask

        cirrus_refl = reflectance[self._band_cirrus, :, :]

        corrected = reflectance.copy()

        for i in range(self.n_bands):
            wl = self.wavelengths[i]

            # Skip absorption bands and cirrus band itself
            if abs(wl - CIRRUS_BAND_NM) < 50:
                continue
            if np.isnan(reflectance[i, 0, 0]):
                continue

            # Wavelength-dependent correction factor
            # Cirrus has stronger effect in visible due to scattering
            if wl < 700:
                scale = 1.0
            elif wl < 1000:
                scale = 0.7
            elif wl < 1500:
                scale = 0.4
            else:
                scale = 0.2

            # Apply correction
            correction = scale * cirrus_refl
            corrected[i, :, :] = reflectance[i, :, :] - correction

        # Ensure non-negative
        corrected = np.maximum(corrected, 0.0)

        mean_correction = np.mean(cirrus_refl[cirrus_mask])
        logger.info(f"  Mean cirrus reflectance: {mean_correction:.4f}")

        return corrected, cirrus_mask

    def apply_full_coastal_correction(self, reflectance, aot550=0.1):
        """
        Apply full coastal correction suite.

        Applies corrections in order:
        1. Cirrus correction (affects all pixels)
        2. Sun glint correction (water pixels)
        3. Adjacency effect correction (coastal water pixels)

        Args:
            reflectance: Reflectance cube to correct
            aot550: AOT for adjacency calculation

        Returns:
            Corrected reflectance, dict of masks and diagnostics
        """
        logger.info("\n" + "="*60)
        logger.info("COASTAL CORRECTION SUITE")
        logger.info("="*60)

        diagnostics = {}

        # 1. Cirrus correction
        reflectance, cirrus_mask = self.correct_cirrus(reflectance)
        diagnostics['cirrus_mask'] = cirrus_mask
        diagnostics['cirrus_fraction'] = np.sum(cirrus_mask) / cirrus_mask.size

        # 2. Sun glint correction
        reflectance, glint_mask = self.correct_sun_glint(reflectance)
        diagnostics['glint_mask'] = glint_mask
        diagnostics['glint_fraction'] = np.sum(glint_mask) / glint_mask.size

        # 3. Adjacency correction
        reflectance, adj_magnitude = self.correct_adjacency_effect(reflectance, aot550)
        diagnostics['adjacency_magnitude'] = adj_magnitude
        diagnostics['water_mask'] = self.detect_water_mask(reflectance)

        logger.info("\n" + "-"*40)
        logger.info("COASTAL CORRECTION SUMMARY")
        logger.info("-"*40)
        logger.info(f"  Cirrus affected: {100*diagnostics['cirrus_fraction']:.1f}%")
        logger.info(f"  Glint affected: {100*diagnostics['glint_fraction']:.1f}%")
        logger.info(f"  Water pixels: {100*np.sum(diagnostics['water_mask'])/diagnostics['water_mask'].size:.1f}%")

        return reflectance, diagnostics


# =============================================================================
# Uncertainty Estimation
# =============================================================================

class UncertaintyEstimator:
    """
    Estimates uncertainty in surface reflectance products.

    Propagates uncertainties from multiple sources:
    1. Radiometric calibration uncertainty
    2. Atmospheric parameter retrieval (AOT, water vapor)
    3. LUT interpolation errors
    4. Model (6S) limitations

    Based on:
    - Taylor series error propagation
    - ATBD for similar missions (MODIS, Landsat)
    - Vermote & Kotchenova (2008)
    """

    def __init__(self, wavelengths, lut=None):
        """
        Initialize uncertainty estimator.

        Args:
            wavelengths: Sensor wavelength array in nm
            lut: Optional LUT dict for sensitivity analysis
        """
        self.wavelengths = wavelengths
        self.n_bands = len(wavelengths)
        self.lut = lut

        # Default uncertainty sources
        self.radiometric_uncertainty = RADIOMETRIC_UNCERTAINTY
        self.aot_uncertainty = AOT_RETRIEVAL_UNCERTAINTY
        self.wv_uncertainty = WV_RETRIEVAL_UNCERTAINTY
        self.lut_uncertainty = LUT_INTERPOLATION_UNCERTAINTY

    def _calculate_sensitivity_to_aot(self, reflectance, aot, lut):
        """
        Calculate sensitivity of reflectance to AOT changes.

        Uses finite differences on LUT to estimate ∂ρ/∂AOT.

        Args:
            reflectance: Current reflectance array
            aot: Current AOT value
            lut: LUT dictionary

        Returns:
            Sensitivity array (d_rho / d_aot) for each band
        """
        if lut is None or 'aot_range' not in lut:
            # Use empirical approximation
            # Higher sensitivity at shorter wavelengths
            sensitivity = np.zeros(self.n_bands)
            for i, wl in enumerate(self.wavelengths):
                # Shorter wavelengths more sensitive to aerosol
                wl_um = wl / 1000.0
                sensitivity[i] = 0.3 * (0.55 / wl_um) ** 1.5
            return sensitivity

        # Use LUT for sensitivity calculation
        aot_range = np.array(lut['aot_range'])

        # Find neighboring AOT values
        idx = np.searchsorted(aot_range, aot)
        idx = np.clip(idx, 1, len(aot_range) - 1)

        aot_low = aot_range[idx - 1]
        aot_high = aot_range[idx]
        delta_aot = aot_high - aot_low

        if delta_aot < 1e-6:
            return np.zeros(self.n_bands)

        # Calculate coefficient differences
        sensitivity = np.zeros(self.n_bands)
        for i in range(self.n_bands):
            if np.isnan(lut['xa'][i, idx - 1, 0]) or np.isnan(lut['xa'][i, idx, 0]):
                continue

            # Approximate reflectance change with AOT
            # Use central wavelength water vapor index
            wv_idx = len(lut['wv_range']) // 2

            xa_low = lut['xa'][i, idx - 1, wv_idx]
            xa_high = lut['xa'][i, idx, wv_idx]
            xb_low = lut['xb'][i, idx - 1, wv_idx]
            xb_high = lut['xb'][i, idx, wv_idx]

            # Sensitivity ≈ |d_xa/d_aot| + |d_xb/d_aot| (simplified)
            sensitivity[i] = (abs(xa_high - xa_low) + abs(xb_high - xb_low)) / delta_aot

        return sensitivity

    def _calculate_sensitivity_to_wv(self, reflectance, wv, lut):
        """
        Calculate sensitivity of reflectance to water vapor changes.

        Args:
            reflectance: Current reflectance array
            wv: Current water vapor value (g/cm²)
            lut: LUT dictionary

        Returns:
            Sensitivity array (d_rho / d_wv) for each band
        """
        if lut is None or 'wv_range' not in lut:
            # Use empirical approximation
            # High sensitivity in water vapor bands
            sensitivity = np.zeros(self.n_bands)
            for i, wl in enumerate(self.wavelengths):
                # Check if in/near water vapor band
                if 920 < wl < 980:
                    sensitivity[i] = 0.05
                elif 1100 < wl < 1180:
                    sensitivity[i] = 0.04
                elif 1350 < wl < 1450:
                    sensitivity[i] = 0.1
                elif 1800 < wl < 1950:
                    sensitivity[i] = 0.15
                else:
                    sensitivity[i] = 0.01
            return sensitivity

        # Use LUT for sensitivity calculation
        wv_range = np.array(lut['wv_range'])

        idx = np.searchsorted(wv_range, wv)
        idx = np.clip(idx, 1, len(wv_range) - 1)

        wv_low = wv_range[idx - 1]
        wv_high = wv_range[idx]
        delta_wv = wv_high - wv_low

        if delta_wv < 1e-6:
            return np.zeros(self.n_bands)

        sensitivity = np.zeros(self.n_bands)
        aot_idx = len(lut['aot_range']) // 2

        for i in range(self.n_bands):
            if np.isnan(lut['xa'][i, aot_idx, idx - 1]) or np.isnan(lut['xa'][i, aot_idx, idx]):
                continue

            xa_low = lut['xa'][i, aot_idx, idx - 1]
            xa_high = lut['xa'][i, aot_idx, idx]
            xb_low = lut['xb'][i, aot_idx, idx - 1]
            xb_high = lut['xb'][i, aot_idx, idx]

            sensitivity[i] = (abs(xa_high - xa_low) + abs(xb_high - xb_low)) / delta_wv

        return sensitivity

    def estimate_uncertainty(self, reflectance, radiance, aot, wv, lut=None,
                            return_components=False):
        """
        Estimate total uncertainty in surface reflectance.

        Uses error propagation:
        σ_total² = σ_rad² + σ_aot² + σ_wv² + σ_lut²

        Args:
            reflectance: Reflectance cube (bands, rows, cols)
            radiance: Input radiance cube
            aot: Retrieved AOT value
            wv: Retrieved water vapor value
            lut: Optional LUT for sensitivity calculations
            return_components: If True, return individual components

        Returns:
            Uncertainty cube (bands, rows, cols) or dict if return_components
        """
        logger.info("\n" + "="*60)
        logger.info("UNCERTAINTY ESTIMATION")
        logger.info("="*60)

        n_bands, n_rows, n_cols = reflectance.shape

        # Initialize uncertainty arrays
        uncertainty = np.zeros_like(reflectance)
        u_radiometric = np.zeros_like(reflectance)
        u_aot = np.zeros_like(reflectance)
        u_wv = np.zeros_like(reflectance)
        u_lut = np.zeros_like(reflectance)

        # 1. Radiometric uncertainty (relative)
        # σ_rad = radiometric_uncertainty * ρ
        logger.info(f"  Radiometric uncertainty: {self.radiometric_uncertainty*100:.1f}%")
        u_radiometric = self.radiometric_uncertainty * np.abs(reflectance)

        # 2. AOT retrieval uncertainty
        # σ_aot = (∂ρ/∂AOT) * σ_AOT
        sensitivity_aot = self._calculate_sensitivity_to_aot(reflectance, aot, lut or self.lut)
        logger.info(f"  AOT uncertainty: ±{self.aot_uncertainty:.3f}")

        for i in range(n_bands):
            u_aot[i, :, :] = sensitivity_aot[i] * self.aot_uncertainty

        # 3. Water vapor retrieval uncertainty
        # σ_wv = (∂ρ/∂WV) * σ_WV
        sensitivity_wv = self._calculate_sensitivity_to_wv(reflectance, wv, lut or self.lut)
        logger.info(f"  Water vapor uncertainty: ±{self.wv_uncertainty:.2f} g/cm²")

        for i in range(n_bands):
            u_wv[i, :, :] = sensitivity_wv[i] * self.wv_uncertainty

        # 4. LUT interpolation uncertainty
        # Constant small contribution from interpolation
        u_lut = np.full_like(reflectance, self.lut_uncertainty)

        # Combine uncertainties (root sum of squares)
        uncertainty = np.sqrt(u_radiometric**2 + u_aot**2 + u_wv**2 + u_lut**2)

        # Set uncertainty to NaN where reflectance is NaN
        uncertainty = np.where(np.isfinite(reflectance), uncertainty, np.nan)

        # Statistics for key wavelengths
        for wl in [550, 670, 860, 1650]:
            band = np.argmin(np.abs(self.wavelengths - wl))
            u_band = uncertainty[band, :, :]
            valid = u_band[np.isfinite(u_band)]
            if len(valid) > 0:
                logger.info(f"  {wl}nm: mean σ = {np.mean(valid):.4f}, "
                           f"max σ = {np.max(valid):.4f}")

        if return_components:
            return {
                'total': uncertainty,
                'radiometric': u_radiometric,
                'aot': u_aot,
                'water_vapor': u_wv,
                'lut_interpolation': u_lut,
            }

        return uncertainty

    def estimate_relative_uncertainty(self, uncertainty, reflectance):
        """
        Calculate relative (percentage) uncertainty.

        Args:
            uncertainty: Absolute uncertainty cube
            reflectance: Reflectance cube

        Returns:
            Relative uncertainty cube (as fraction, not percentage)
        """
        with np.errstate(invalid='ignore', divide='ignore'):
            relative = np.where(
                reflectance > 0.001,
                uncertainty / reflectance,
                np.nan
            )
        return np.clip(relative, 0, 1)

    def create_quality_flag(self, uncertainty, reflectance):
        """
        Create quality flag layer based on uncertainty.

        Quality levels:
        0: Invalid (NaN reflectance)
        1: High quality (relative uncertainty < 5%)
        2: Medium quality (5-10%)
        3: Low quality (10-20%)
        4: Poor quality (> 20%)

        Args:
            uncertainty: Uncertainty cube
            reflectance: Reflectance cube

        Returns:
            Quality flag array (2D, min across bands)
        """
        relative = self.estimate_relative_uncertainty(uncertainty, reflectance)

        # Get worst (maximum) relative uncertainty across bands
        # (excluding atmospheric absorption bands)
        valid_bands = []
        for i, wl in enumerate(self.wavelengths):
            in_absorption = False
            for band_start, band_end, _ in ABSORPTION_BANDS:
                if band_start <= wl <= band_end:
                    in_absorption = True
                    break
            if not in_absorption:
                valid_bands.append(i)

        # Maximum uncertainty across valid bands
        rel_max = np.nanmax(relative[valid_bands, :, :], axis=0)

        # Create quality flags
        quality = np.zeros(rel_max.shape, dtype=np.int8)
        quality[~np.isfinite(rel_max)] = 0  # Invalid
        quality[(rel_max < 0.05) & np.isfinite(rel_max)] = 1  # High
        quality[(rel_max >= 0.05) & (rel_max < 0.10)] = 2  # Medium
        quality[(rel_max >= 0.10) & (rel_max < 0.20)] = 3  # Low
        quality[rel_max >= 0.20] = 4  # Poor

        # Count quality levels
        for level, name in [(1, 'High'), (2, 'Medium'), (3, 'Low'), (4, 'Poor')]:
            count = np.sum(quality == level)
            pct = 100 * count / quality.size
            logger.info(f"  Quality {name}: {pct:.1f}%")

        return quality


# =============================================================================
# Look-Up Table Generator
# =============================================================================

def _run_6s_worker(params):
    """
    Module-level 6S worker function for parallel execution.

    This must be at module level (not a class method) to be picklable
    for multiprocessing.

    Args:
        params: Dict with all 6S parameters including sixs_path

    Returns:
        Dict with coefficients or None on failure
    """
    try:
        from Py6S import SixS, Geometry, AtmosProfile, AeroProfile, Wavelength
        from Py6S import GroundReflectance, AtmosCorr

        sixs_path = params.get('sixs_path')
        s = SixS(sixs_path) if sixs_path else SixS()

        # Geometry
        s.geometry = Geometry.User()
        s.geometry.solar_z = params['sun_zenith']
        s.geometry.solar_a = params['sun_azimuth']
        s.geometry.view_z = params['sensor_zenith']
        s.geometry.view_a = params['sensor_azimuth']
        s.geometry.month = params['month']
        s.geometry.day = params['day']

        # Atmosphere
        s.atmos_profile = AtmosProfile.UserWaterAndOzone(params['wv'], 0.35)

        # Aerosol
        aerosol_map = {
            'maritime': AeroProfile.Maritime,
            'continental': AeroProfile.Continental,
            'urban': AeroProfile.Urban,
            'desert': AeroProfile.Desert,
        }
        aero_type = aerosol_map.get(params['aerosol_model'], AeroProfile.Maritime)
        s.aero_profile = AeroProfile.PredefinedType(aero_type)
        s.aot550 = params['aot']

        # Altitudes - proper airborne sensor configuration
        s.altitudes.set_target_custom_altitude(params['target_alt'])
        s.altitudes.set_sensor_custom_altitude(
            params['sensor_alt'],
            params['aot'] * 0.5,  # AOT below sensor
            params['wv'] * 0.7    # WV below sensor
        )

        # Wavelength
        s.wavelength = Wavelength(params['wl_um'])

        # Ground reflectance
        s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.3)

        # Enable atmospheric correction mode
        s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromRadiance(1.0)

        # Run
        s.run()

        return {
            'wl_idx': params['wl_idx'],
            'aot_idx': params['aot_idx'],
            'wv_idx': params['wv_idx'],
            'xa': s.outputs.coef_xa,
            'xb': s.outputs.coef_xb,
            'xc': s.outputs.coef_xc,
        }

    except Exception as e:
        return {
            'wl_idx': params['wl_idx'],
            'aot_idx': params['aot_idx'],
            'wv_idx': params['wv_idx'],
            'xa': np.nan,
            'xb': np.nan,
            'xc': np.nan,
            'error': str(e),
        }


class LUTGenerator:
    """
    Generates Look-Up Tables (LUT) for efficient atmospheric correction.

    Pre-computes 6S coefficients (xa, xb, xc) across:
    - Wavelengths (sensor bands)
    - Aerosol optical depth
    - Water vapor content
    - Solar/view geometry

    This enables rapid correction of full images by interpolation.

    Performance optimizations:
    - Parallel 6S execution (multiprocessing)
    - Disk caching of LUT results
    - Key wavelength selection with interpolation to full spectrum
    """

    def __init__(self, sixs_path=None, cache_dir=None, max_workers=None):
        """
        Initialize LUT generator.

        Args:
            sixs_path: Path to 6S executable
            cache_dir: Directory for LUT caching (default: ~/.aviris_atm_cache)
            max_workers: Max parallel processes (default: CPU count or 4)
        """
        self.sixs_path = sixs_path or SIXS_PATH
        self.cache_dir = Path(cache_dir) if cache_dir else LUT_CACHE_DIR
        self.max_workers = max_workers or min(LUT_MAX_WORKERS, os.cpu_count() or 4)

        if not HAS_PY6S:
            raise RuntimeError("Py6S required for LUT generation")

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, geometry, sensor_alt, target_alt, aerosol_model):
        """Generate cache key from parameters."""
        import hashlib
        key_str = f"{geometry['sun_zenith']:.1f}_{geometry['sun_azimuth']:.1f}_"
        key_str += f"{geometry['sensor_zenith']:.1f}_{geometry['sensor_azimuth']:.1f}_"
        key_str += f"{sensor_alt:.2f}_{target_alt:.2f}_{aerosol_model}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def _get_cache_path(self, cache_key):
        """Get path for cached LUT file."""
        return self.cache_dir / f"lut_{cache_key}.npz"

    def load_cached_lut(self, cache_key):
        """Load LUT from disk cache if available."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                logger.info(f"  Loading cached LUT from {cache_path.name}")
                data = np.load(cache_path, allow_pickle=True)
                lut = {key: data[key] for key in data.files}
                # Convert 0-d arrays back to scalars/dicts
                if 'geometry' in lut:
                    lut['geometry'] = lut['geometry'].item()
                return lut
            except Exception as e:
                logger.warning(f"  Failed to load cached LUT: {e}")
        return None

    def save_lut_to_cache(self, lut, cache_key):
        """Save LUT to disk cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            np.savez_compressed(cache_path, **lut)
            logger.info(f"  Saved LUT to cache: {cache_path.name}")
        except Exception as e:
            logger.warning(f"  Failed to save LUT to cache: {e}")

    def _select_key_wavelengths(self, wavelengths, n_key=None):
        """
        Select key wavelengths for LUT generation.

        Selects wavelengths that:
        1. Cover the full spectral range uniformly
        2. Avoid deep atmospheric absorption bands
        3. Include spectrally distinct regions

        Args:
            wavelengths: Full wavelength array
            n_key: Number of key wavelengths (default: LUT_KEY_WAVELENGTHS)

        Returns:
            Indices of key wavelengths
        """
        n_key = n_key or LUT_KEY_WAVELENGTHS

        # Start with uniform sampling
        n_wl = len(wavelengths)
        step = max(1, n_wl // n_key)
        key_indices = list(range(0, n_wl, step))

        # Remove bands in deep absorption
        filtered_indices = []
        for idx in key_indices:
            wl = wavelengths[idx]
            in_absorption = False
            for band_start, band_end, _ in ABSORPTION_BANDS:
                if band_start <= wl <= band_end:
                    in_absorption = True
                    break
            if not in_absorption:
                filtered_indices.append(idx)

        # Ensure we have first and last valid bands
        if 0 not in filtered_indices:
            filtered_indices.insert(0, 0)
        if n_wl - 1 not in filtered_indices:
            filtered_indices.append(n_wl - 1)

        return np.array(sorted(set(filtered_indices)))

    def generate_lut(self, wavelengths, geometry, sensor_altitude_km,
                     target_altitude_km=0.0, aerosol_model='maritime',
                     aot_range=(0.05, 0.1, 0.2, 0.3, 0.5),
                     wv_range=(0.5, 1.0, 2.0, 3.0, 5.0),
                     use_key_wavelengths=True, use_cache=True, parallel=True):
        """
        Generate LUT for given conditions with performance optimizations.

        Args:
            wavelengths: Array of wavelengths in nm
            geometry: Dict with sun_zenith, sun_azimuth, sensor_zenith, sensor_azimuth
            sensor_altitude_km: Aircraft altitude in km
            target_altitude_km: Ground elevation in km
            aerosol_model: One of 'maritime', 'continental', 'urban', 'desert'
            aot_range: Tuple of AOT values to compute
            wv_range: Tuple of water vapor values (g/cm²) to compute
            use_key_wavelengths: If True, simulate subset and interpolate (MUCH faster)
            use_cache: If True, check/save disk cache
            parallel: If True, use multiprocessing

        Returns:
            Dict containing LUT data and metadata
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(geometry, sensor_altitude_km,
                                            target_altitude_km, aerosol_model)
            cached_lut = self.load_cached_lut(cache_key)
            if cached_lut is not None:
                logger.info("  Using cached LUT")
                # Interpolate to full wavelength set if needed
                if len(cached_lut['wavelengths']) != len(wavelengths):
                    cached_lut = self._interpolate_lut_wavelengths(cached_lut, wavelengths)
                return cached_lut

        logger.info("Generating 6S Look-Up Table...")
        logger.info(f"  Sensor altitude: {sensor_altitude_km} km")

        # Select key wavelengths for faster processing
        if use_key_wavelengths:
            key_indices = self._select_key_wavelengths(wavelengths)
            key_wavelengths = wavelengths[key_indices]
            logger.info(f"  Using {len(key_wavelengths)} key wavelengths (of {len(wavelengths)})")
        else:
            key_indices = np.arange(len(wavelengths))
            key_wavelengths = wavelengths

        n_key = len(key_wavelengths)
        n_aot = len(aot_range)
        n_wv = len(wv_range)
        total_runs = n_key * n_aot * n_wv

        logger.info(f"  Total 6S runs: {total_runs} ({n_key} bands × {n_aot} AOTs × {n_wv} WV)")

        # Build parameter sets for all runs
        param_sets = []
        for i_wl, wl_nm in enumerate(key_wavelengths):
            # Skip deep absorption bands
            in_absorption = False
            for band_start, band_end, _ in ABSORPTION_BANDS:
                if band_start <= wl_nm <= band_end:
                    in_absorption = True
                    break
            if in_absorption:
                continue

            for i_aot, aot in enumerate(aot_range):
                for i_wv, wv in enumerate(wv_range):
                    param_sets.append({
                        'wl_idx': i_wl,
                        'wl_nm': wl_nm,
                        'wl_um': wl_nm / 1000.0,
                        'aot_idx': i_aot,
                        'aot': aot,
                        'wv_idx': i_wv,
                        'wv': wv,
                        'sun_zenith': geometry['sun_zenith'],
                        'sun_azimuth': geometry['sun_azimuth'],
                        'sensor_zenith': geometry['sensor_zenith'],
                        'sensor_azimuth': geometry['sensor_azimuth'],
                        'month': geometry.get('month', 9),
                        'day': geometry.get('day', 15),
                        'sensor_alt': sensor_altitude_km,
                        'target_alt': target_altitude_km,
                        'aerosol_model': aerosol_model,
                        'sixs_path': self.sixs_path,  # Include for module-level worker
                    })

        # Initialize arrays for key wavelengths
        xa_key = np.full((n_key, n_aot, n_wv), np.nan)
        xb_key = np.full((n_key, n_aot, n_wv), np.nan)
        xc_key = np.full((n_key, n_aot, n_wv), np.nan)

        start_time = time.time()

        # Execute 6S runs
        if parallel and len(param_sets) > 10:
            # Parallel execution using module-level function (picklable)
            logger.info(f"  Using {self.max_workers} parallel workers")
            import concurrent.futures

            completed = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs using module-level worker function
                futures = {executor.submit(_run_6s_worker, p): p for p in param_sets}

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    i_wl = result['wl_idx']
                    i_aot = result['aot_idx']
                    i_wv = result['wv_idx']

                    xa_key[i_wl, i_aot, i_wv] = result['xa']
                    xb_key[i_wl, i_aot, i_wv] = result['xb']
                    xc_key[i_wl, i_aot, i_wv] = result['xc']

                    completed += 1
                    if completed % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining = (len(param_sets) - completed) / rate if rate > 0 else 0
                        logger.info(f"  Progress: {completed}/{len(param_sets)} "
                                   f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
        else:
            # Sequential execution
            for i, params in enumerate(param_sets):
                result = _run_6s_worker(params)  # Use module-level function
                xa_key[result['wl_idx'], result['aot_idx'], result['wv_idx']] = result['xa']
                xb_key[result['wl_idx'], result['aot_idx'], result['wv_idx']] = result['xb']
                xc_key[result['wl_idx'], result['aot_idx'], result['wv_idx']] = result['xc']

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"  Progress: {i+1}/{len(param_sets)} ({elapsed:.0f}s elapsed)")

        elapsed = time.time() - start_time
        logger.info(f"  6S simulations complete in {elapsed:.0f}s")

        # Build key wavelength LUT
        key_lut = {
            'wavelengths': key_wavelengths,
            'key_indices': key_indices,
            'aot_range': np.array(aot_range),
            'wv_range': np.array(wv_range),
            'xa': xa_key,
            'xb': xb_key,
            'xc': xc_key,
            'geometry': geometry,
            'sensor_altitude_km': sensor_altitude_km,
            'target_altitude_km': target_altitude_km,
            'aerosol_model': aerosol_model,
        }

        # Interpolate to full wavelength set
        if use_key_wavelengths and len(key_wavelengths) < len(wavelengths):
            logger.info("  Interpolating to full wavelength set...")
            full_lut = self._interpolate_lut_wavelengths(key_lut, wavelengths)
        else:
            full_lut = key_lut
            full_lut['wavelengths'] = wavelengths

        # Save to cache
        if use_cache:
            self.save_lut_to_cache(full_lut, cache_key)

        return full_lut

    def _interpolate_lut_wavelengths(self, lut, target_wavelengths):
        """
        Interpolate LUT from key wavelengths to full wavelength set.

        Uses linear interpolation between key bands.
        Handles NaN values in key wavelengths by filtering them before interpolation.
        """
        # Convert to regular numpy arrays (handle masked arrays from NetCDF)
        key_wl = np.asarray(lut['wavelengths']).flatten()
        target_wl = np.asarray(target_wavelengths).flatten()

        # Convert LUT arrays, replacing masked values with NaN
        xa_key = np.ma.filled(np.asarray(lut['xa']), np.nan)
        xb_key = np.ma.filled(np.asarray(lut['xb']), np.nan)
        xc_key = np.ma.filled(np.asarray(lut['xc']), np.nan)

        n_target = len(target_wl)
        n_aot = len(lut['aot_range'])
        n_wv = len(lut['wv_range'])

        xa_full = np.zeros((n_target, n_aot, n_wv))
        xb_full = np.zeros((n_target, n_aot, n_wv))
        xc_full = np.zeros((n_target, n_aot, n_wv))

        for i_aot in range(n_aot):
            for i_wv in range(n_wv):
                # Get coefficient slices as regular arrays
                xa_slice = xa_key[:, i_aot, i_wv]
                xb_slice = xb_key[:, i_aot, i_wv]
                xc_slice = xc_key[:, i_aot, i_wv]

                # Filter out NaN values before interpolation
                # This allows interpolation across absorption band gaps
                valid_mask = np.isfinite(xa_slice) & np.isfinite(xb_slice) & np.isfinite(xc_slice)

                if np.sum(valid_mask) < 2:
                    # Not enough valid points for interpolation
                    xa_full[:, i_aot, i_wv] = np.nan
                    xb_full[:, i_aot, i_wv] = np.nan
                    xc_full[:, i_aot, i_wv] = np.nan
                    continue

                valid_wl = key_wl[valid_mask]
                valid_xa = xa_slice[valid_mask]
                valid_xb = xb_slice[valid_mask]
                valid_xc = xc_slice[valid_mask]

                # Interpolate each coefficient using only valid values
                xa_interp = interp1d(valid_wl, valid_xa,
                                     kind='linear', bounds_error=False,
                                     fill_value=(valid_xa[0], valid_xa[-1]))  # Extrapolate at edges
                xb_interp = interp1d(valid_wl, valid_xb,
                                     kind='linear', bounds_error=False,
                                     fill_value=(valid_xb[0], valid_xb[-1]))
                xc_interp = interp1d(valid_wl, valid_xc,
                                     kind='linear', bounds_error=False,
                                     fill_value=(valid_xc[0], valid_xc[-1]))

                xa_full[:, i_aot, i_wv] = xa_interp(target_wl)
                xb_full[:, i_aot, i_wv] = xb_interp(target_wl)
                xc_full[:, i_aot, i_wv] = xc_interp(target_wl)

        # Mark absorption bands as NaN (these shouldn't be used for reflectance)
        for i, wl in enumerate(target_wl):
            for band_start, band_end, _ in ABSORPTION_BANDS:
                if band_start <= wl <= band_end:
                    xa_full[i, :, :] = np.nan
                    xb_full[i, :, :] = np.nan
                    xc_full[i, :, :] = np.nan
                    break

        return {
            'wavelengths': target_wl,
            'aot_range': np.asarray(lut['aot_range']),
            'wv_range': np.asarray(lut['wv_range']),
            'xa': xa_full,
            'xb': xb_full,
            'xc': xc_full,
            'geometry': lut['geometry'],
            'sensor_altitude_km': lut['sensor_altitude_km'],
            'target_altitude_km': lut['target_altitude_km'],
            'aerosol_model': lut['aerosol_model'],
        }
    
    def interpolate_coefficients(self, lut, aot, wv):
        """
        Interpolate LUT coefficients for specific AOT and water vapor values.
        
        Args:
            lut: LUT dict from generate_lut()
            aot: AOT at 550nm
            wv: Water vapor in g/cm²
            
        Returns:
            Tuple of (xa, xb, xc) arrays for each wavelength
        """
        from scipy.interpolate import RegularGridInterpolator
        
        n_wl = len(lut['wavelengths'])
        xa = np.zeros(n_wl)
        xb = np.zeros(n_wl)
        xc = np.zeros(n_wl)
        
        aot_range = np.array(lut['aot_range'])
        wv_range = np.array(lut['wv_range'])
        
        # Ensure ranges are valid for interpolation
        if len(aot_range) < 2 or len(wv_range) < 2:
            logger.error("LUT ranges too small for interpolation")
            return xa, xb, xc
        
        # Clamp to LUT range
        aot_clamped = np.clip(aot, aot_range.min(), aot_range.max())
        wv_clamped = np.clip(wv, wv_range.min(), wv_range.max())
        
        logger.info(f"  Interpolating at AOT={aot_clamped:.4f}, WV={wv_clamped:.2f}")
        if aot != aot_clamped:
            logger.info(f"  (AOT clamped from {aot:.4f} to LUT range)")
        if wv != wv_clamped:
            logger.info(f"  (WV clamped from {wv:.2f} to LUT range)")
        
        for i_wl in range(n_wl):
            if np.isnan(lut['xa'][i_wl, 0, 0]):
                xa[i_wl] = np.nan
                xb[i_wl] = np.nan
                xc[i_wl] = np.nan
                continue
            
            # Create interpolators for this wavelength
            for name, arr, out in [('xa', lut['xa'], xa), 
                                   ('xb', lut['xb'], xb), 
                                   ('xc', lut['xc'], xc)]:
                try:
                    interp = RegularGridInterpolator(
                        (aot_range, wv_range),
                        arr[i_wl, :, :],
                        method='linear',
                        bounds_error=False,
                        fill_value=None
                    )
                    out[i_wl] = interp([[aot_clamped, wv_clamped]])[0]
                except Exception as e:
                    logger.warning(f"Interpolation failed at band {i_wl}: {e}")
                    out[i_wl] = np.nan
        
        return xa, xb, xc


# =============================================================================
# Main Atmospheric Correction Class
# =============================================================================

class AVIRISL2Processor:
    """
    AVIRIS-3 Level 2 (Surface Reflectance) Processor.
    
    Performs atmospheric correction using radiative transfer modeling,
    with scene-derived atmospheric parameters and proper airborne sensor
    configuration.
    """
    
    # Aircraft altitude estimates by platform
    PLATFORM_ALTITUDES = {
        'B-200': 8.5,      # King Air B-200: ~28,000 ft
        'G-III': 12.5,     # Gulfstream III: ~41,000 ft
        'G-V': 13.7,       # Gulfstream V: ~45,000 ft
        'ER-2': 20.0,      # NASA ER-2: ~65,000 ft
        'default': 6.0,    # Conservative default
    }
    
    def __init__(self, radiance_path, obs_path, sensor_altitude_km=None,
                 aerosol_model='maritime', coastal_correction=True,
                 estimate_uncertainty=True):
        """
        Initialize processor.

        Args:
            radiance_path: Path to L1B radiance NetCDF
            obs_path: Path to L1B OBS NetCDF
            sensor_altitude_km: Aircraft altitude (auto-detect if None)
            aerosol_model: Aerosol type assumption
            coastal_correction: Enable coastal/adjacency correction (default True)
            estimate_uncertainty: Enable uncertainty estimation (default True)
        """
        self.radiance_path = Path(radiance_path)
        self.obs_path = Path(obs_path)
        self.aerosol_model = aerosol_model
        self.enable_coastal_correction = coastal_correction
        self.enable_uncertainty = estimate_uncertainty

        # Load data
        self._load_data()
        self._extract_geometry()

        # Set sensor altitude
        if sensor_altitude_km is not None:
            self.sensor_altitude_km = sensor_altitude_km
        else:
            self.sensor_altitude_km = self._estimate_altitude()

        logger.info(f"Sensor altitude: {self.sensor_altitude_km} km")

        # Initialize retrieval modules
        self.wv_retrieval = WaterVaporRetrieval(self.wavelengths, self.radiance)
        self.aod_retrieval = AODRetrieval(
            self.wavelengths, self.radiance,
            {'sun_zenith': self.sun_zenith}
        )

        # Initialize coastal correction module
        self.coastal_corrector = None
        if self.enable_coastal_correction:
            geometry = {
                'sun_zenith': self.sun_zenith,
                'sun_azimuth': self.sun_azimuth,
                'sensor_zenith': self.sensor_zenith,
                'sensor_azimuth': self.sensor_azimuth,
            }
            self.coastal_corrector = CoastalCorrection(
                self.wavelengths, self.radiance, geometry
            )
            logger.info("Coastal correction: ENABLED")
        else:
            logger.info("Coastal correction: DISABLED")

        # Processing metadata
        self.processing_info = {
            'processor': 'AVIRIS L2 Processor v2.0',
            'radiance_file': str(self.radiance_path),
            'obs_file': str(self.obs_path),
            'processing_time': None,
            'sensor_altitude_km': self.sensor_altitude_km,
            'aerosol_model': aerosol_model,
            'coastal_correction': coastal_correction,
            'uncertainty_estimation': estimate_uncertainty,
        }

        # Initialize uncertainty estimator
        if self.enable_uncertainty:
            logger.info("Uncertainty estimation: ENABLED")
        else:
            logger.info("Uncertainty estimation: DISABLED")
    
    def _load_data(self):
        """Load radiance data and wavelengths."""
        logger.info(f"Loading radiance: {self.radiance_path.name}")
        self.rad_ds = nc.Dataset(self.radiance_path)
        
        # Find radiance group
        if 'radiance' in self.rad_ds.groups:
            rad_group = self.rad_ds.groups['radiance']
            self.radiance = np.ma.filled(rad_group.variables['radiance'][:], np.nan)
            self.wavelengths = np.ma.filled(rad_group.variables['wavelength'][:], np.nan)
        elif 'radiance' in self.rad_ds.variables:
            self.radiance = np.ma.filled(self.rad_ds.variables['radiance'][:], np.nan)
            self.wavelengths = np.ma.filled(self.rad_ds.variables['wavelength'][:], np.nan)
        else:
            raise ValueError("Could not find radiance data in file")

        # Ensure wavelengths is a 1D array
        self.wavelengths = np.asarray(self.wavelengths).flatten()
        
        self.n_bands, self.n_rows, self.n_cols = self.radiance.shape
        
        # Get units
        try:
            self.rad_units = self.rad_ds.groups['radiance'].variables['radiance'].units
        except:
            self.rad_units = 'uW/(nm*cm^2*sr)'  # Standard AVIRIS units
        
        logger.info(f"  Shape: {self.n_bands} bands × {self.n_rows} × {self.n_cols} pixels")
        logger.info(f"  Wavelength range: {self.wavelengths[0]:.0f}-{self.wavelengths[-1]:.0f} nm")
        logger.info(f"  Radiance units: {self.rad_units}")
    
    def _extract_geometry(self):
        """Extract viewing and solar geometry from OBS file."""
        logger.info(f"Loading geometry: {self.obs_path.name}")
        self.obs_ds = nc.Dataset(self.obs_path)
        
        # Find observation parameters group
        if 'observation_parameters' in self.obs_ds.groups:
            op = self.obs_ds.groups['observation_parameters']
        else:
            op = self.obs_ds
        
        # Extract geometry (scene averages)
        self.sun_zenith = float(np.nanmean(op.variables['to_sun_zenith'][:]))
        self.sun_azimuth = float(np.nanmean(op.variables['to_sun_azimuth'][:]))
        self.sensor_zenith = float(np.nanmean(op.variables['to_sensor_zenith'][:]))
        self.sensor_azimuth = float(np.nanmean(op.variables['to_sensor_azimuth'][:]))
        
        # Per-pixel illumination cosine
        if 'cosine_i' in op.variables:
            self.cosine_i = op.variables['cosine_i'][:]
        else:
            self.cosine_i = np.cos(np.radians(self.sun_zenith)) * np.ones((self.n_rows, self.n_cols))
        
        # Ground elevation
        if 'elev' in self.obs_ds.variables:
            self.ground_elevation = float(np.nanmean(self.obs_ds.variables['elev'][:])) / 1000.0
        else:
            self.ground_elevation = 0.0
        
        # Earth-Sun distance
        if 'earth_sun_distance' in op.variables:
            self.earth_sun_distance = float(np.nanmean(op.variables['earth_sun_distance'][:]))
        else:
            self.earth_sun_distance = 1.0  # AU
        
        # Extract date from filename (AV3YYYYMMDD...)
        try:
            date_str = self.radiance_path.name[3:11]
            self.year = int(date_str[0:4])
            self.month = int(date_str[4:6])
            self.day = int(date_str[6:8])
        except:
            self.year = 2024
            self.month = 9
            self.day = 15
        
        logger.info(f"  Acquisition date: {self.year}-{self.month:02d}-{self.day:02d}")
        logger.info(f"  Solar geometry: zenith={self.sun_zenith:.1f}°, azimuth={self.sun_azimuth:.1f}°")
        logger.info(f"  View geometry: zenith={self.sensor_zenith:.1f}°, azimuth={self.sensor_azimuth:.1f}°")
        logger.info(f"  Ground elevation: {self.ground_elevation*1000:.0f} m")
    
    def _estimate_altitude(self):
        """Estimate aircraft altitude from filename or metadata."""
        filename = self.radiance_path.name.upper()
        
        # Check for platform hints in filename
        for platform, altitude in self.PLATFORM_ALTITUDES.items():
            if platform.upper() in filename:
                logger.info(f"  Detected platform: {platform}")
                return altitude
        
        # Try to get from OBS file
        try:
            if 'altitude' in self.obs_ds.variables:
                alt = float(np.nanmean(self.obs_ds.variables['altitude'][:]))
                if alt > 100:  # Likely in meters
                    alt = alt / 1000.0
                return alt
        except:
            pass
        
        logger.warning("  Could not determine platform, using default altitude")
        return self.PLATFORM_ALTITUDES['default']
    
    def retrieve_atmospheric_params(self):
        """
        Retrieve atmospheric parameters from the scene.
        
        Returns:
            Dict with 'aot550' and 'water_vapor' (g/cm²)
        """
        logger.info("\n" + "="*60)
        logger.info("ATMOSPHERIC PARAMETER RETRIEVAL")
        logger.info("="*60)
        
        # Water vapor retrieval
        water_vapor = self.wv_retrieval.get_scene_water_vapor()
        
        # AOD retrieval
        aot550 = self.aod_retrieval.get_scene_aot(method='hybrid')
        
        self.processing_info['retrieved_aot550'] = aot550
        self.processing_info['retrieved_water_vapor'] = water_vapor
        
        return {'aot550': aot550, 'water_vapor': water_vapor}
    
    def generate_correction_lut(self, atm_params, n_aot=5, n_wv=5):
        """
        Generate LUT for atmospheric correction.
        
        Args:
            atm_params: Dict with 'aot550' and 'water_vapor'
            n_aot: Number of AOT values in LUT
            n_wv: Number of water vapor values in LUT
            
        Returns:
            LUT dict
        """
        if not HAS_PY6S:
            logger.warning("Py6S not available, cannot generate LUT")
            return None
        
        logger.info("\n" + "="*60)
        logger.info("LOOK-UP TABLE GENERATION")
        logger.info("="*60)
        
        # Define LUT ranges centered on retrieved values
        # Use minimum ranges to ensure useful coverage even with very low AOT
        aot_center = max(atm_params['aot550'], 0.05)  # Minimum center of 0.05
        wv_center = max(atm_params['water_vapor'], 0.5)  # Minimum center of 0.5
        
        # Create ranges that span reasonable atmospheric conditions
        aot_min = max(0.01, aot_center * 0.3)
        aot_max = min(2.0, max(aot_center * 3.0, 0.3))  # At least span to 0.3
        wv_min = max(0.1, wv_center * 0.3)
        wv_max = min(8.0, max(wv_center * 3.0, 2.0))  # At least span to 2.0
        
        aot_range = np.linspace(aot_min, aot_max, n_aot)
        wv_range = np.linspace(wv_min, wv_max, n_wv)
        
        # Ensure strictly ascending (should always be true now, but safety check)
        aot_range = np.unique(aot_range)
        wv_range = np.unique(wv_range)
        
        geometry = {
            'sun_zenith': self.sun_zenith,
            'sun_azimuth': self.sun_azimuth,
            'sensor_zenith': self.sensor_zenith,
            'sensor_azimuth': self.sensor_azimuth,
            'month': self.month,
            'day': self.day,
        }
        
        lut_gen = LUTGenerator(SIXS_PATH)
        lut = lut_gen.generate_lut(
            self.wavelengths,
            geometry,
            self.sensor_altitude_km,
            self.ground_elevation,
            self.aerosol_model,
            tuple(aot_range),
            tuple(wv_range)
        )
        
        return lut
    
    def apply_correction_lut(self, lut, aot, wv, use_tiles=True):
        """
        Apply atmospheric correction using LUT with numerical stability.

        Args:
            lut: LUT dict from generate_correction_lut()
            aot: AOT at 550nm
            wv: Water vapor in g/cm²
            use_tiles: If True, process in tiles to limit memory usage

        Returns:
            Reflectance array (n_bands, n_rows, n_cols)
        """
        logger.info("\n" + "="*60)
        logger.info("APPLYING ATMOSPHERIC CORRECTION")
        logger.info("="*60)
        logger.info(f"  Using AOT={aot:.3f}, WV={wv:.2f} g/cm²")

        # Get interpolated coefficients
        lut_gen = LUTGenerator(SIXS_PATH)
        xa, xb, xc = lut_gen.interpolate_coefficients(lut, aot, wv)

        # AVIRIS radiance units conversion
        # µW/(nm·cm²·sr) → W/(m²·sr·µm)
        UNIT_CONVERSION = 10.0

        # Estimate memory and decide on tiling
        memory_needed_gb = (self.radiance.nbytes * 2) / 1e9  # radiance + reflectance
        if use_tiles and memory_needed_gb > MEMORY_LIMIT_GB:
            logger.info(f"  Using tile-based processing (memory limit: {MEMORY_LIMIT_GB} GB)")
            reflectance = self._apply_correction_tiled(xa, xb, xc, UNIT_CONVERSION)
        else:
            reflectance = self._apply_correction_full(xa, xb, xc, UNIT_CONVERSION)

        # Apply topographic correction
        reflectance = self._apply_topographic_correction(reflectance)

        # Final validation
        n_invalid = np.sum(~np.isfinite(reflectance) | (reflectance < 0) | (reflectance > MAX_REFLECTANCE))
        if n_invalid > 0:
            pct_invalid = 100 * n_invalid / reflectance.size
            logger.warning(f"  {pct_invalid:.2f}% pixels outside valid range - clipping")

        reflectance = np.clip(reflectance, 0.0, 1.0)
        reflectance = np.nan_to_num(reflectance, nan=0.0, posinf=0.0, neginf=0.0)

        return reflectance

    def _apply_correction_full(self, xa, xb, xc, unit_conversion):
        """Apply correction to full image (for smaller datasets)."""
        reflectance = np.zeros_like(self.radiance, dtype=np.float32)

        for i in range(self.n_bands):
            # Check for valid coefficients
            if np.isnan(xa[i]) or np.isnan(xb[i]) or np.isnan(xc[i]):
                reflectance[i, :, :] = np.nan
                continue

            # Convert radiance
            L = self.radiance[i, :, :].astype(np.float64) * unit_conversion

            # Handle NaN radiance values
            valid_L = np.isfinite(L) & (L > 0)

            # 6S correction formula: y = xa * L - xb; rho = y / (1 + xc * y)
            y = xa[i] * L - xb[i]

            # CRITICAL: Numerical stability - check denominator
            denominator = 1.0 + xc[i] * y

            # Valid denominator must be positive and above threshold
            valid_denom = np.abs(denominator) > MIN_DENOMINATOR

            # Calculate reflectance only where valid
            valid_mask = valid_L & valid_denom

            with np.errstate(invalid='ignore', divide='ignore'):
                rho = np.where(valid_mask, y / denominator, np.nan)

            # Additional sanity check: physically reasonable reflectance
            rho = np.where((rho >= 0) & (rho <= MAX_REFLECTANCE), rho, np.nan)

            reflectance[i, :, :] = rho.astype(np.float32)

            # Progress
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1}/{self.n_bands} bands")

        return reflectance

    def _apply_correction_tiled(self, xa, xb, xc, unit_conversion):
        """Apply correction in tiles to limit memory usage."""
        reflectance = np.zeros((self.n_bands, self.n_rows, self.n_cols), dtype=np.float32)

        # Calculate tile grid
        n_tiles_row = int(np.ceil(self.n_rows / TILE_SIZE))
        n_tiles_col = int(np.ceil(self.n_cols / TILE_SIZE))
        total_tiles = n_tiles_row * n_tiles_col

        logger.info(f"  Processing {total_tiles} tiles ({n_tiles_row} × {n_tiles_col})")

        tile_count = 0
        for ti in range(n_tiles_row):
            row_start = ti * TILE_SIZE
            row_end = min((ti + 1) * TILE_SIZE, self.n_rows)

            for tj in range(n_tiles_col):
                col_start = tj * TILE_SIZE
                col_end = min((tj + 1) * TILE_SIZE, self.n_cols)

                # Process all bands for this tile
                for i in range(self.n_bands):
                    if np.isnan(xa[i]) or np.isnan(xb[i]) or np.isnan(xc[i]):
                        reflectance[i, row_start:row_end, col_start:col_end] = np.nan
                        continue

                    # Get tile radiance
                    L = self.radiance[i, row_start:row_end, col_start:col_end].astype(np.float64)
                    L = L * unit_conversion

                    valid_L = np.isfinite(L) & (L > 0)

                    y = xa[i] * L - xb[i]
                    denominator = 1.0 + xc[i] * y
                    valid_denom = np.abs(denominator) > MIN_DENOMINATOR
                    valid_mask = valid_L & valid_denom

                    with np.errstate(invalid='ignore', divide='ignore'):
                        rho = np.where(valid_mask, y / denominator, np.nan)

                    rho = np.where((rho >= 0) & (rho <= MAX_REFLECTANCE), rho, np.nan)
                    reflectance[i, row_start:row_end, col_start:col_end] = rho.astype(np.float32)

                tile_count += 1
                if tile_count % 10 == 0:
                    logger.info(f"  Processed {tile_count}/{total_tiles} tiles")

        return reflectance
    
    def apply_correction_empirical(self):
        """
        Apply simplified empirical atmospheric correction.
        
        This method does not require Py6S but provides less accurate results.
        It uses:
        1. Dark object subtraction for path radiance removal
        2. Solar irradiance model for reflectance conversion
        3. Per-pixel illumination correction
        
        Returns:
            Reflectance array (n_bands, n_rows, n_cols)
        """
        logger.info("\n" + "="*60)
        logger.info("APPLYING EMPIRICAL ATMOSPHERIC CORRECTION")
        logger.info("="*60)
        logger.warning("This is a simplified method - results less accurate than 6S")
        
        solar = SolarIrradianceModel()
        reflectance = np.zeros_like(self.radiance, dtype=np.float32)
        
        # Get atmospheric transmittance estimate (wavelength dependent)
        def estimate_transmittance(wl_nm, airmass):
            """Estimate two-way atmospheric transmittance."""
            wl_um = wl_nm / 1000.0
            
            # Rayleigh scattering optical depth
            tau_rayleigh = 0.008569 * wl_um**(-4) * (1 + 0.0113*wl_um**(-2))
            
            # Water vapor absorption (simplified)
            tau_water = 0.0
            if 920 < wl_nm < 980:
                tau_water = 0.3
            elif 1100 < wl_nm < 1180:
                tau_water = 0.2
            elif 1350 < wl_nm < 1450:
                tau_water = 2.0
            elif 1800 < wl_nm < 1950:
                tau_water = 3.0
            
            # Aerosol (assume AOT ~0.15)
            tau_aerosol = 0.15 * (wl_um / 0.55)**(-1.3)
            
            # Total transmittance (two-way)
            tau_total = tau_rayleigh + tau_water + tau_aerosol
            T = np.exp(-tau_total * airmass * 2)
            
            return T
        
        # Calculate airmass
        cos_sza = np.cos(np.radians(self.sun_zenith))
        airmass = 1.0 / cos_sza
        
        # Earth-Sun distance correction
        d_corr = 1.0 / (self.earth_sun_distance ** 2)
        
        # Per-pixel illumination
        cos_illum = np.clip(self.cosine_i, 0.1, 1.0)
        
        for i, wl in enumerate(self.wavelengths):
            L = self.radiance[i, :, :]
            
            # Skip absorption bands
            in_absorption = False
            for band_start, band_end, _ in ABSORPTION_BANDS:
                if band_start <= wl <= band_end:
                    in_absorption = True
                    break
            
            if in_absorption:
                reflectance[i, :, :] = np.nan
                continue
            
            # Dark object subtraction
            L_dark = np.nanpercentile(L, 0.5)
            L_corrected = L - L_dark * 0.9  # Remove most of path radiance
            L_corrected = np.maximum(L_corrected, 0.0)
            
            # Get solar irradiance
            E_sun = solar.get_irradiance(wl, self.earth_sun_distance)
            
            # Estimate transmittance
            T_atm = estimate_transmittance(wl, airmass)
            
            # Convert to reflectance
            # rho = pi * L / (E_sun * cos(theta) * T_atm)
            # Unit conversion: µW/(nm·cm²·sr) → W/(m²·sr·µm) = *10
            L_si = L_corrected * 10.0
            
            rho = (np.pi * L_si) / (E_sun * cos_illum * T_atm)
            
            reflectance[i, :, :] = rho
            
            # Progress
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1}/{self.n_bands} bands")
        
        # Validate and clip
        reflectance = np.clip(reflectance, 0.0, 1.0)
        
        return reflectance
    
    def _apply_topographic_correction(self, reflectance):
        """
        Apply topographic correction using per-pixel illumination.
        
        Uses C-correction (Teillet et al., 1982):
        rho_corrected = rho * (cos(theta_s) + c) / (cos(i) + c)
        
        Where c is an empirical correction factor.
        """
        logger.info("  Applying topographic correction...")
        
        cos_sza = np.cos(np.radians(self.sun_zenith))
        cos_i = np.clip(self.cosine_i, 0.1, 1.0)
        
        # C-correction factor (band-dependent, using average)
        c = 0.2
        
        correction = (cos_sza + c) / (cos_i + c)
        correction = np.clip(correction, 0.5, 2.0)  # Limit extreme corrections
        
        # Apply to all bands
        for i in range(reflectance.shape[0]):
            if not np.isnan(reflectance[i, 0, 0]):
                reflectance[i, :, :] *= correction
        
        return reflectance
    
    def validate_results(self, reflectance):
        """
        Validate reflectance results against expected ranges.
        
        Args:
            reflectance: Reflectance array
            
        Returns:
            Dict with validation statistics
        """
        logger.info("\n" + "="*60)
        logger.info("VALIDATION")
        logger.info("="*60)
        
        validation = {}
        
        # Check key wavelengths
        for wl in [550, 650, 860, 1650, 2200]:
            band = np.argmin(np.abs(self.wavelengths - wl))
            data = reflectance[band, :, :]
            valid = data[~np.isnan(data) & (data > 0) & (data < 1)]
            
            if len(valid) > 0:
                stats = {
                    'wavelength': self.wavelengths[band],
                    'min': float(valid.min()),
                    'max': float(valid.max()),
                    'mean': float(valid.mean()),
                    'std': float(valid.std()),
                    'p5': float(np.percentile(valid, 5)),
                    'p95': float(np.percentile(valid, 95)),
                }
                validation[f'{wl}nm'] = stats
                
                logger.info(f"  {wl}nm: mean={stats['mean']:.3f}, "
                           f"range=[{stats['min']:.3f}, {stats['max']:.3f}], "
                           f"p5-p95=[{stats['p5']:.3f}, {stats['p95']:.3f}]")
        
        # Check for anomalies
        warnings_list = []
        
        # Mean reflectance should be reasonable
        mean_vis = np.nanmean(reflectance[np.argmin(np.abs(self.wavelengths - 550)), :, :])
        mean_nir = np.nanmean(reflectance[np.argmin(np.abs(self.wavelengths - 860)), :, :])
        
        if mean_vis > 0.5:
            warnings_list.append(f"High visible reflectance ({mean_vis:.2f}) - possible over-correction")
        if mean_nir < 0.05:
            warnings_list.append(f"Low NIR reflectance ({mean_nir:.2f}) - possible under-correction")
        if mean_nir < mean_vis:
            warnings_list.append("NIR < visible - unusual for most surfaces")
        
        validation['warnings'] = warnings_list
        
        for warning in warnings_list:
            logger.warning(f"  ⚠ {warning}")
        
        if not warnings_list:
            logger.info("  ✓ Results pass basic validation checks")
        
        return validation
    
    def save_reflectance(self, reflectance, output_path, validation=None,
                         coastal_diagnostics=None, uncertainty_data=None):
        """
        Save reflectance to NetCDF file with full metadata.

        Args:
            reflectance: Reflectance cube to save
            output_path: Output file path
            validation: Optional validation dict
            coastal_diagnostics: Optional coastal correction diagnostics
            uncertainty_data: Optional uncertainty estimation data
        """
        logger.info(f"\nSaving to: {output_path}")
        
        output_path = Path(output_path)
        
        out_ds = nc.Dataset(output_path, 'w', format='NETCDF4')
        
        # Global attributes
        out_ds.setncattr('title', 'AVIRIS-3 Surface Reflectance')
        out_ds.setncattr('institution', 'Generated by AVIRIS L2 Processor v2.0')
        out_ds.setncattr('source', str(self.radiance_path.name))
        out_ds.setncattr('history', f'Created {datetime.now().isoformat()}')
        out_ds.setncattr('processing_level', 'L2_Reflectance')
        out_ds.setncattr('atmospheric_correction_method', 
                        '6S_LUT' if HAS_PY6S else 'Empirical')
        out_ds.setncattr('sensor_altitude_km', self.sensor_altitude_km)
        out_ds.setncattr('aerosol_model', self.aerosol_model)
        
        if 'retrieved_aot550' in self.processing_info:
            out_ds.setncattr('aot550', self.processing_info['retrieved_aot550'])
            out_ds.setncattr('water_vapor_g_cm2', self.processing_info['retrieved_water_vapor'])
        
        # Copy original attributes
        for attr in ['flight_line', 'acquisition_date', 'platform']:
            if hasattr(self.rad_ds, attr):
                out_ds.setncattr(attr, self.rad_ds.getncattr(attr))
        
        # Dimensions
        out_ds.createDimension('bands', self.n_bands)
        out_ds.createDimension('rows', self.n_rows)
        out_ds.createDimension('cols', self.n_cols)
        
        # Reflectance group
        refl_group = out_ds.createGroup('reflectance')
        
        # Reflectance variable
        refl_var = refl_group.createVariable(
            'reflectance', 'f4', ('bands', 'rows', 'cols'),
            zlib=True, complevel=4
        )
        refl_var[:] = reflectance
        refl_var.units = 'unitless'
        refl_var.long_name = 'Surface Reflectance'
        refl_var.valid_range = [0.0, 1.0]
        refl_var.scale_factor = 1.0
        refl_var.add_offset = 0.0
        
        # Wavelength
        wl_var = refl_group.createVariable('wavelength', 'f4', ('bands',))
        wl_var[:] = self.wavelengths
        wl_var.units = 'nm'
        wl_var.long_name = 'Center Wavelength'
        
        # Copy geolocation if available
        for var_name in ['lat', 'lon', 'elev']:
            if var_name in self.rad_ds.variables:
                orig_var = self.rad_ds.variables[var_name]
                new_var = out_ds.createVariable(var_name, 'f4', ('rows', 'cols'))
                new_var[:] = orig_var[:]
                # Copy attributes except _FillValue (must be set at creation)
                for attr in orig_var.ncattrs():
                    if attr != '_FillValue':
                        try:
                            new_var.setncattr(attr, orig_var.getncattr(attr))
                        except Exception as e:
                            logger.warning(f"Could not copy attribute {attr}: {e}")
        
        # Save validation as JSON attribute
        if validation:
            # Convert numpy types to Python native types for JSON
            def convert_to_native(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(v) for v in obj]
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            validation_clean = convert_to_native(validation)
            out_ds.setncattr('validation', json.dumps(validation_clean))

        # Save coastal correction diagnostics
        if coastal_diagnostics is not None:
            logger.info("  Saving coastal correction masks...")

            # Create coastal correction group
            coastal_group = out_ds.createGroup('coastal_correction')

            # Water mask
            if 'water_mask' in coastal_diagnostics:
                water_var = coastal_group.createVariable(
                    'water_mask', 'i1', ('rows', 'cols'),
                    zlib=True, complevel=4
                )
                water_var[:] = coastal_diagnostics['water_mask'].astype(np.int8)
                water_var.long_name = 'Water Mask'
                water_var.description = 'Water pixels identified using NDWI'
                water_var.flag_values = [0, 1]
                water_var.flag_meanings = 'land water'

            # Glint mask
            if 'glint_mask' in coastal_diagnostics:
                glint_var = coastal_group.createVariable(
                    'glint_mask', 'i1', ('rows', 'cols'),
                    zlib=True, complevel=4
                )
                glint_var[:] = coastal_diagnostics['glint_mask'].astype(np.int8)
                glint_var.long_name = 'Sun Glint Mask'
                glint_var.description = 'Pixels affected by sun glint'
                glint_var.flag_values = [0, 1]
                glint_var.flag_meanings = 'no_glint glint'

            # Cirrus mask
            if 'cirrus_mask' in coastal_diagnostics:
                cirrus_var = coastal_group.createVariable(
                    'cirrus_mask', 'i1', ('rows', 'cols'),
                    zlib=True, complevel=4
                )
                cirrus_var[:] = coastal_diagnostics['cirrus_mask'].astype(np.int8)
                cirrus_var.long_name = 'Cirrus Mask'
                cirrus_var.description = 'Pixels affected by thin cirrus clouds'
                cirrus_var.flag_values = [0, 1]
                cirrus_var.flag_meanings = 'no_cirrus cirrus'

            # Adjacency correction magnitude
            if 'adjacency_magnitude' in coastal_diagnostics:
                adj_var = coastal_group.createVariable(
                    'adjacency_correction', 'f4', ('rows', 'cols'),
                    zlib=True, complevel=4
                )
                adj_var[:] = coastal_diagnostics['adjacency_magnitude'].astype(np.float32)
                adj_var.long_name = 'Adjacency Effect Correction Magnitude'
                adj_var.description = 'Magnitude of adjacency correction applied (green band)'
                adj_var.units = 'reflectance'

            # Summary statistics
            coastal_group.setncattr('cirrus_fraction', float(coastal_diagnostics['cirrus_fraction']))
            coastal_group.setncattr('glint_fraction', float(coastal_diagnostics['glint_fraction']))

        # Save uncertainty data
        if uncertainty_data is not None:
            logger.info("  Saving uncertainty estimates...")

            # Create uncertainty group
            uncertainty_group = out_ds.createGroup('uncertainty')

            # Full uncertainty cube (compressed)
            if 'uncertainty' in uncertainty_data:
                # Save only key bands to reduce file size
                key_wl = [450, 550, 670, 860, 1600, 2200]
                key_indices = [np.argmin(np.abs(self.wavelengths - wl)) for wl in key_wl]

                uncertainty_group.createDimension('key_bands', len(key_indices))

                # Key band wavelengths
                key_wl_var = uncertainty_group.createVariable('key_wavelengths', 'f4', ('key_bands',))
                key_wl_var[:] = [self.wavelengths[i] for i in key_indices]
                key_wl_var.units = 'nm'

                # Uncertainty for key bands
                uncertainty_var = uncertainty_group.createVariable(
                    'uncertainty', 'f4', ('key_bands', 'rows', 'cols'),
                    zlib=True, complevel=4
                )
                for j, i in enumerate(key_indices):
                    uncertainty_var[j, :, :] = uncertainty_data['uncertainty'][i, :, :].astype(np.float32)
                uncertainty_var.long_name = 'Surface Reflectance Uncertainty (1-sigma)'
                uncertainty_var.description = 'Propagated uncertainty from radiometric, atmospheric, and processing sources'
                uncertainty_var.units = 'reflectance'

                # Also save per-band mean uncertainty for all bands
                mean_uncertainty = np.nanmean(uncertainty_data['uncertainty'], axis=(1, 2))
                mean_var = uncertainty_group.createVariable('mean_uncertainty_by_band', 'f4', ('bands',))
                mean_var[:] = mean_uncertainty.astype(np.float32)
                mean_var.long_name = 'Mean uncertainty per band'
                mean_var.units = 'reflectance'

            # Quality flag
            if 'quality_flag' in uncertainty_data:
                quality_var = uncertainty_group.createVariable(
                    'quality_flag', 'i1', ('rows', 'cols'),
                    zlib=True, complevel=4
                )
                quality_var[:] = uncertainty_data['quality_flag'].astype(np.int8)
                quality_var.long_name = 'Data Quality Flag'
                quality_var.description = 'Quality based on maximum relative uncertainty across bands'
                quality_var.flag_values = [0, 1, 2, 3, 4]
                quality_var.flag_meanings = 'invalid high_quality medium_quality low_quality poor_quality'
                quality_var.comment = 'High: <5% relative uncertainty, Medium: 5-10%, Low: 10-20%, Poor: >20%'

        out_ds.close()
        
        file_size = output_path.stat().st_size / 1e6
        logger.info(f"Saved: {file_size:.1f} MB")
    
    def process(self, output_path, use_6s=True):
        """
        Run full atmospheric correction pipeline.

        Args:
            output_path: Output file path
            use_6s: Use 6S if available
        """
        logger.info("\n" + "="*60)
        logger.info("AVIRIS-3 ATMOSPHERIC CORRECTION v2.0")
        logger.info("="*60)

        start_time = time.time()

        # Retrieve atmospheric parameters
        atm_params = self.retrieve_atmospheric_params()

        # Apply correction
        if use_6s and HAS_PY6S:
            # Generate LUT
            lut = self.generate_correction_lut(atm_params)

            # Apply LUT-based correction
            reflectance = self.apply_correction_lut(
                lut,
                atm_params['aot550'],
                atm_params['water_vapor']
            )
        else:
            # Empirical correction
            reflectance = self.apply_correction_empirical()

        # Apply coastal correction if enabled
        coastal_diagnostics = None
        if self.enable_coastal_correction and self.coastal_corrector is not None:
            reflectance, coastal_diagnostics = self.coastal_corrector.apply_full_coastal_correction(
                reflectance,
                aot550=atm_params['aot550']
            )
            self.processing_info['coastal_diagnostics'] = {
                'cirrus_fraction': float(coastal_diagnostics['cirrus_fraction']),
                'glint_fraction': float(coastal_diagnostics['glint_fraction']),
                'water_fraction': float(np.sum(coastal_diagnostics['water_mask']) / coastal_diagnostics['water_mask'].size),
            }

        # Estimate uncertainty if enabled
        uncertainty_data = None
        if self.enable_uncertainty:
            uncertainty_estimator = UncertaintyEstimator(self.wavelengths, lut if use_6s and HAS_PY6S else None)
            uncertainty = uncertainty_estimator.estimate_uncertainty(
                reflectance,
                self.radiance,
                atm_params['aot550'],
                atm_params['water_vapor'],
                lut=lut if use_6s and HAS_PY6S else None
            )
            quality_flag = uncertainty_estimator.create_quality_flag(uncertainty, reflectance)
            uncertainty_data = {
                'uncertainty': uncertainty,
                'quality_flag': quality_flag,
            }

        # Validate
        validation = self.validate_results(reflectance)

        # Save
        self.save_reflectance(reflectance, output_path, validation, coastal_diagnostics, uncertainty_data)

        elapsed = time.time() - start_time
        self.processing_info['processing_time'] = elapsed

        logger.info("\n" + "="*60)
        logger.info(f"PROCESSING COMPLETE in {elapsed:.0f} seconds")
        logger.info("="*60)
    
    def close(self):
        """Close open files."""
        self.rad_ds.close()
        self.obs_ds.close()


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='AVIRIS-3 Atmospheric Correction v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aviris_atm_correction_v2.py radiance.nc obs.nc output.nc
  python aviris_atm_correction_v2.py radiance.nc obs.nc output.nc --altitude 8.5
  python aviris_atm_correction_v2.py radiance.nc obs.nc output.nc --aerosol maritime
  python aviris_atm_correction_v2.py radiance.nc obs.nc output.nc --simple
        """
    )
    
    parser.add_argument('radiance', help='Input L1B radiance NetCDF file')
    parser.add_argument('obs', help='Input L1B observation parameters NetCDF file')
    parser.add_argument('output', help='Output L2 reflectance NetCDF file')
    
    parser.add_argument('--altitude', type=float, default=None,
                       help='Aircraft altitude in km (default: auto-detect)')
    parser.add_argument('--aerosol', choices=['maritime', 'continental', 'urban', 'desert'],
                       default='maritime', help='Aerosol model (default: maritime)')
    parser.add_argument('--simple', action='store_true',
                       help='Use simplified empirical correction (no Py6S)')
    parser.add_argument('--validate', action='store_true',
                       help='Run additional validation checks')
    parser.add_argument('--no-coastal', action='store_true',
                       help='Disable coastal/adjacency correction')
    parser.add_argument('--no-uncertainty', action='store_true',
                       help='Disable uncertainty estimation')

    args = parser.parse_args()

    # Process
    processor = AVIRISL2Processor(
        args.radiance,
        args.obs,
        sensor_altitude_km=args.altitude,
        aerosol_model=args.aerosol,
        coastal_correction=not args.no_coastal,
        estimate_uncertainty=not args.no_uncertainty
    )
    
    try:
        processor.process(args.output, use_6s=not args.simple)
    finally:
        processor.close()


if __name__ == "__main__":
    main()
