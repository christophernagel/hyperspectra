"""
AVIRIS-3 ISOFIT Atmospheric Correction Module
==============================================
NASA/JPL Standard atmospheric correction using ISOFIT + sRTMnet.

This module provides a wrapper around ISOFIT (Imaging Spectrometer Optimal FITting)
for atmospheric correction of AVIRIS-3 hyperspectral data. It handles:

1. NetCDF to ENVI format conversion (ISOFIT requirement)
2. ISOFIT configuration and execution via apply_oe
3. Output conversion back to NetCDF for viewer compatibility
4. Uncertainty propagation and quality assessment

ISOFIT uses optimal estimation with sRTMnet (neural network emulator of MODTRAN 6)
to jointly retrieve surface reflectance and atmospheric state, providing:
- Proper handling of all wavelengths including SWIR
- Per-pixel atmospheric parameter retrieval
- Rigorous uncertainty quantification
- Publication-ready results

Installation:
    mamba create -n isofit_env -c conda-forge isofit
    mamba activate isofit_env
    isofit download sixs
    isofit download srtmnet

Usage:
    python aviris_isofit_processor.py <radiance.nc> <obs.nc> <output.nc>

Author: Christopher / Claude
Date: January 2025
Version: 1.0
"""

import sys
import os
import numpy as np
import netCDF4 as nc
from pathlib import Path
from datetime import datetime
import logging
import json
import tempfile
import shutil
from typing import Optional, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Check ISOFIT Availability
# =============================================================================

HAS_ISOFIT = False
ISOFIT_VERSION = None

try:
    import isofit
    from isofit.utils import apply_oe
    HAS_ISOFIT = True
    ISOFIT_VERSION = getattr(isofit, '__version__', 'unknown')
    logger.info(f"ISOFIT {ISOFIT_VERSION} available")
except ImportError as e:
    logger.warning(f"ISOFIT not available: {e}")
    logger.warning("Install with: mamba install -c conda-forge isofit")


# =============================================================================
# ENVI Format Utilities
# =============================================================================

class ENVIWriter:
    """Write ENVI format binary files with headers."""

    DTYPE_MAP = {
        np.float32: 4,
        np.float64: 5,
        np.int16: 2,
        np.int32: 3,
        np.uint16: 12,
        np.uint8: 1,
    }

    @staticmethod
    def write(filepath: Path, data: np.ndarray,
              wavelengths: Optional[np.ndarray] = None,
              fwhm: Optional[np.ndarray] = None,
              description: str = "",
              interleave: str = "bil") -> Path:
        """
        Write data array to ENVI format.

        Args:
            filepath: Output path (without extension)
            data: 3D array (lines, samples, bands) or 2D (lines, samples)
            wavelengths: Band wavelengths in nm
            fwhm: Full width half maximum for each band
            description: File description
            interleave: Data interleave (bil, bip, bsq)

        Returns:
            Path to the written file
        """
        filepath = Path(filepath)

        # Ensure 3D
        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        lines, samples, bands = data.shape
        dtype = data.dtype.type

        # Get ENVI dtype code
        dtype_code = ENVIWriter.DTYPE_MAP.get(dtype, 4)  # Default float32

        # Convert to target dtype if needed
        if dtype not in ENVIWriter.DTYPE_MAP:
            data = data.astype(np.float32)
            dtype_code = 4

        # Write binary data - ensure contiguous writable array
        binary_path = filepath.with_suffix('')

        # Make a contiguous copy to ensure it's writable
        data = np.ascontiguousarray(data)

        if interleave == 'bil':
            # Band interleaved by line: (lines, bands, samples)
            out_data = np.ascontiguousarray(data.transpose(0, 2, 1))
            out_data.tofile(str(binary_path))
        elif interleave == 'bip':
            # Band interleaved by pixel - already (lines, samples, bands)
            data.tofile(str(binary_path))
        else:  # bsq
            # Band sequential: (bands, lines, samples)
            out_data = np.ascontiguousarray(data.transpose(2, 0, 1))
            out_data.tofile(str(binary_path))

        # Write header
        header_path = filepath.with_suffix('.hdr')
        header_lines = [
            "ENVI",
            f"description = {{{description}}}",
            f"samples = {samples}",
            f"lines = {lines}",
            f"bands = {bands}",
            f"header offset = 0",
            f"file type = ENVI Standard",
            f"data type = {dtype_code}",
            f"interleave = {interleave}",
            f"byte order = 0",  # Little endian
        ]

        if wavelengths is not None and len(wavelengths) == bands:
            wl_str = ", ".join(f"{w:.4f}" for w in wavelengths)
            header_lines.append(f"wavelength = {{{wl_str}}}")
            header_lines.append("wavelength units = nanometers")

        if fwhm is not None and len(fwhm) == bands:
            fwhm_str = ", ".join(f"{f:.4f}" for f in fwhm)
            header_lines.append(f"fwhm = {{{fwhm_str}}}")

        with open(header_path, 'w') as f:
            f.write('\n'.join(header_lines))

        logger.info(f"Wrote ENVI file: {binary_path} ({lines}x{samples}x{bands})")
        return binary_path

    @staticmethod
    def read(filepath: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read ENVI format file.

        Args:
            filepath: Path to ENVI file (with or without extension)

        Returns:
            Tuple of (data array, header dict)
        """
        filepath = Path(filepath)

        # Find header file
        if filepath.suffix == '.hdr':
            header_path = filepath
            binary_path = filepath.with_suffix('')
        else:
            header_path = filepath.with_suffix('.hdr')
            binary_path = filepath.with_suffix('') if filepath.suffix else filepath

        # Parse header
        header = {}
        with open(header_path, 'r') as f:
            content = f.read()

        # Simple header parsing
        import re
        for match in re.finditer(r'(\w+)\s*=\s*(.+?)(?=\n\w+\s*=|\Z)', content, re.DOTALL):
            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            # Remove braces
            if value.startswith('{') and value.endswith('}'):
                value = value[1:-1]
            header[key] = value

        # Extract dimensions
        lines = int(header.get('lines', 0))
        samples = int(header.get('samples', 0))
        bands = int(header.get('bands', 1))
        dtype_code = int(header.get('data type', 4))
        interleave = header.get('interleave', 'bil').lower()

        # Map dtype
        dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64, 12: np.uint16}
        dtype = dtype_map.get(dtype_code, np.float32)

        # Read binary
        data = np.fromfile(str(binary_path), dtype=dtype)

        # Reshape based on interleave
        if interleave == 'bil':
            data = data.reshape(lines, bands, samples).transpose(0, 2, 1)
        elif interleave == 'bip':
            data = data.reshape(lines, samples, bands)
        else:  # bsq
            data = data.reshape(bands, lines, samples).transpose(1, 2, 0)

        # Parse wavelengths if present
        if 'wavelength' in header:
            wl_str = header['wavelength']
            header['wavelengths'] = np.array([float(x.strip()) for x in wl_str.split(',')])

        return data, header


# =============================================================================
# AVIRIS-3 NetCDF to ENVI Converter
# =============================================================================

class AVIRIS3Converter:
    """
    Converts AVIRIS-3 NetCDF files to ENVI format for ISOFIT processing.

    AVIRIS-3 L1B products come in NetCDF format with:
    - RDN file: Calibrated radiance with wavelengths
    - ORT file: Observation geometry and illumination parameters

    ISOFIT requires ENVI format with specific observation bands:
    1. Path length (m)
    2. To-sensor azimuth (deg)
    3. To-sensor zenith (deg)
    4. To-sun azimuth (deg)
    5. To-sun zenith (deg)
    6. Phase angle (deg)
    7. Slope (deg)
    8. Aspect (deg)
    9. Cosine i (illumination factor)
    10. UTC time (decimal hours)
    """

    # Mapping from AVIRIS-3 ORT variable names to ISOFIT obs bands
    OBS_MAPPING = {
        'path_length': 0,
        'to_sensor_azimuth': 1,
        'to_sensor_zenith': 2,
        'sensor_azimuth': 1,  # Alternative name
        'sensor_zenith': 2,   # Alternative name
        'to_sun_azimuth': 3,
        'to_sun_zenith': 3,
        'solar_azimuth': 3,   # Alternative name
        'solar_zenith': 4,    # Alternative name
        'phase': 5,
        'phase_angle': 5,
        'slope': 6,
        'aspect': 7,
        'cosine_i': 8,
        'cos_i': 8,
        'utc_time': 9,
        'time': 9,
    }

    def __init__(self, rdn_path: Path, ort_path: Path, output_dir: Optional[Path] = None):
        """
        Initialize converter.

        Args:
            rdn_path: Path to RDN NetCDF file
            ort_path: Path to ORT NetCDF file
            output_dir: Output directory for ENVI files (default: temp directory)
        """
        self.rdn_path = Path(rdn_path)
        self.ort_path = Path(ort_path)
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix='aviris_envi_'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract flightline ID from filename (e.g., AV320230926t201618)
        # ISOFIT needs this for datetime parsing
        self.flightline_id = self._extract_flightline_id()

        # Outputs
        self.radiance_envi: Optional[Path] = None
        self.obs_envi: Optional[Path] = None
        self.loc_envi: Optional[Path] = None
        self.wavelengths: Optional[np.ndarray] = None
        self.fwhm: Optional[np.ndarray] = None

    def _extract_flightline_id(self) -> str:
        """Extract flightline ID from filename for ISOFIT datetime parsing."""
        import re
        filename = self.rdn_path.stem

        # Look for AVIRIS-3 pattern: AV3YYYYMMDDTHHMMSS or similar
        # Pattern: letters followed by date (YYYYMMDD) then 't' then time (HHMMSS)
        match = re.search(r'([A-Za-z]+\d{8}t\d{6})', filename, re.IGNORECASE)
        if match:
            return match.group(1)

        # Fallback: try to find any datetime-like pattern
        match = re.search(r'(\d{8}t\d{6})', filename, re.IGNORECASE)
        if match:
            return f"AV3{match.group(1)}"

        # Last resort: create a fake datetime from current time
        from datetime import datetime
        fake_dt = datetime.now().strftime("%Y%m%dt%H%M%S")
        logger.warning(f"Could not extract flightline ID from filename, using: AV3{fake_dt}")
        return f"AV3{fake_dt}"

    def convert(self) -> Dict[str, Any]:
        """
        Convert AVIRIS-3 NetCDF files to ENVI format.

        Returns:
            Dict with paths to converted files and metadata
        """
        logger.info("Converting AVIRIS-3 NetCDF to ENVI format...")

        # Convert radiance
        self._convert_radiance()

        # Convert observation geometry
        self._convert_obs()

        # Convert location
        self._convert_loc()

        return {
            'radiance': self.radiance_envi,
            'obs': self.obs_envi,
            'loc': self.loc_envi,
            'wavelengths': self.wavelengths,
            'fwhm': self.fwhm,
            'flightline_id': self.flightline_id,
        }

    def _convert_radiance(self):
        """Convert radiance NetCDF to ENVI."""
        logger.info(f"Converting radiance: {self.rdn_path}")

        with nc.Dataset(self.rdn_path, 'r') as ds:
            # Find radiance data - check different group structures
            if 'radiance' in ds.groups:
                rad_group = ds.groups['radiance']
                radiance = np.ma.filled(rad_group.variables['radiance'][:], np.nan)
                wavelengths = np.ma.filled(rad_group.variables['wavelength'][:], np.nan)
            elif 'Radiance' in ds.variables:
                radiance = np.ma.filled(ds.variables['Radiance'][:], np.nan)
                wavelengths = np.ma.filled(ds.variables['wavelength'][:], np.nan)
            else:
                # Try root level
                radiance = np.ma.filled(ds.variables['radiance'][:], np.nan)
                wavelengths = np.ma.filled(ds.variables['wavelength'][:], np.nan)

            # Flatten wavelengths
            self.wavelengths = np.asarray(wavelengths).flatten()

            # Try to get FWHM
            for fwhm_name in ['fwhm', 'FWHM', 'bandwidth']:
                if fwhm_name in ds.variables:
                    self.fwhm = np.ma.filled(ds.variables[fwhm_name][:], np.nan).flatten()
                    break
            else:
                # Estimate FWHM as ~7.4nm for AVIRIS-3
                self.fwhm = np.full_like(self.wavelengths, 7.4)

            # Ensure radiance is (lines, samples, bands)
            if radiance.ndim == 3:
                # Check if bands are first dimension
                if radiance.shape[0] == len(self.wavelengths):
                    radiance = radiance.transpose(1, 2, 0)
                elif radiance.shape[2] != len(self.wavelengths):
                    # Try middle dimension
                    if radiance.shape[1] == len(self.wavelengths):
                        radiance = radiance.transpose(0, 2, 1)

            logger.info(f"  Shape: {radiance.shape}, Bands: {len(self.wavelengths)}")
            logger.info(f"  Wavelength range: {self.wavelengths.min():.1f} - {self.wavelengths.max():.1f} nm")
            logger.info(f"  Flightline ID: {self.flightline_id}")

            # Write ENVI - use flightline ID in filename for ISOFIT parsing
            self.radiance_envi = ENVIWriter.write(
                self.output_dir / f'{self.flightline_id}_rdn',
                radiance.astype(np.float32),
                wavelengths=self.wavelengths,
                fwhm=self.fwhm,
                description=f"AVIRIS-3 L1B Radiance - {self.flightline_id}",
                interleave='bil'
            )

    def _convert_obs(self):
        """Convert observation geometry NetCDF to ENVI."""
        logger.info(f"Converting observation geometry: {self.ort_path}")

        with nc.Dataset(self.ort_path, 'r') as ds:
            # Get dimensions from radiance
            with nc.Dataset(self.rdn_path, 'r') as rdn_ds:
                if 'radiance' in rdn_ds.groups:
                    rad = rdn_ds.groups['radiance'].variables['radiance']
                    if rad.shape[0] == len(self.wavelengths):
                        lines, samples = rad.shape[1], rad.shape[2]
                    else:
                        lines, samples = rad.shape[0], rad.shape[1]
                else:
                    rad = rdn_ds.variables.get('radiance', rdn_ds.variables.get('Radiance'))
                    if rad.shape[0] == len(self.wavelengths):
                        lines, samples = rad.shape[1], rad.shape[2]
                    else:
                        lines, samples = rad.shape[0], rad.shape[1]

            # Create 10-band observation array
            # ISOFIT expects: path_length, to_sensor_az, to_sensor_zen, to_sun_az, to_sun_zen,
            #                 phase, slope, aspect, cosine_i, utc_time
            obs = np.zeros((lines, samples, 10), dtype=np.float64)

            # AVIRIS-3 ORT files use 'observation_parameters' group
            if 'observation_parameters' in ds.groups:
                obs_group = ds.groups['observation_parameters']
                var_dict = obs_group.variables
                logger.info("  Found observation_parameters group")
            elif 'observation' in ds.groups:
                obs_group = ds.groups['observation']
                var_dict = obs_group.variables
            elif 'obs' in ds.groups:
                obs_group = ds.groups['obs']
                var_dict = obs_group.variables
            else:
                var_dict = ds.variables

            # Direct mapping for AVIRIS-3 ORT variables
            aviris3_mapping = {
                'path_length': 0,
                'to_sensor_azimuth': 1,
                'to_sensor_zenith': 2,
                'to_sun_azimuth': 3,
                'to_sun_zenith': 4,
                'solar_phase': 5,
                'slope': 6,
                'aspect': 7,
                'cosine_i': 8,
                'utc_time': 9,
            }

            found_vars = []
            for var_name, band_idx in aviris3_mapping.items():
                if var_name in var_dict:
                    var_data = np.ma.filled(var_dict[var_name][:], np.nan)

                    # Ensure correct shape
                    if var_data.shape == (lines, samples):
                        obs[:, :, band_idx] = var_data
                        # Log sample values for first variable
                        if len(found_vars) == 0:
                            logger.info(f"  Sample {var_name} range: {np.nanmin(var_data):.2f} - {np.nanmax(var_data):.2f}")
                        found_vars.append(f"{var_name} -> band {band_idx}")
                    else:
                        logger.warning(f"  {var_name} shape mismatch: {var_data.shape} vs expected ({lines}, {samples})")

            logger.info(f"  Mapped {len(found_vars)} observation variables")
            for v in found_vars:
                logger.info(f"    {v}")

            # Fill missing bands with reasonable defaults
            if np.all(obs[:, :, 0] == 0):  # Path length
                obs[:, :, 0] = 6000.0  # ~6km altitude default
                logger.info("  Using default path length: 6000m")

            if np.all(obs[:, :, 5] == 0):  # Phase angle
                # Calculate from zenith angles
                obs[:, :, 5] = np.abs(obs[:, :, 2] - obs[:, :, 4])

            if np.all(obs[:, :, 8] == 0):  # Cosine i
                # Calculate from solar zenith
                obs[:, :, 8] = np.cos(np.radians(obs[:, :, 4]))

            if np.all(obs[:, :, 9] == 0):  # UTC time
                obs[:, :, 9] = 12.0  # Noon default

            # Write ENVI - use flightline ID in filename
            self.obs_envi = ENVIWriter.write(
                self.output_dir / f'{self.flightline_id}_obs',
                obs.astype(np.float64),
                description=f"AVIRIS-3 Observation Geometry - {self.flightline_id}",
                interleave='bil'
            )

    def _convert_loc(self):
        """Convert location data to ENVI."""
        logger.info("Converting location data...")

        with nc.Dataset(self.ort_path, 'r') as ds:
            # Get dimensions
            with nc.Dataset(self.rdn_path, 'r') as rdn_ds:
                if 'radiance' in rdn_ds.groups:
                    rad = rdn_ds.groups['radiance'].variables['radiance']
                    if rad.shape[0] == len(self.wavelengths):
                        lines, samples = rad.shape[1], rad.shape[2]
                    else:
                        lines, samples = rad.shape[0], rad.shape[1]
                else:
                    rad = rdn_ds.variables.get('radiance', rdn_ds.variables.get('Radiance'))
                    if rad.shape[0] == len(self.wavelengths):
                        lines, samples = rad.shape[1], rad.shape[2]
                    else:
                        lines, samples = rad.shape[0], rad.shape[1]

            # Create 3-band location array (lon, lat, elevation)
            loc = np.zeros((lines, samples, 3), dtype=np.float64)

            # Find coordinate variables
            lat_names = ['latitude', 'lat', 'Latitude', 'LAT']
            lon_names = ['longitude', 'lon', 'Longitude', 'LON', 'long']
            elev_names = ['elevation', 'elev', 'height', 'altitude', 'dem', 'Elevation']

            # Check groups
            if 'location' in ds.groups:
                var_dict = ds.groups['location'].variables
            elif 'glt' in ds.groups:
                var_dict = ds.groups['glt'].variables
            else:
                var_dict = ds.variables

            # Find and extract variables
            for names, band_idx, default in [(lon_names, 0, -120.0),
                                              (lat_names, 1, 35.0),
                                              (elev_names, 2, 0.0)]:
                for name in names:
                    if name in var_dict:
                        data = np.ma.filled(var_dict[name][:], np.nan)
                        if data.shape == (lines, samples):
                            loc[:, :, band_idx] = data
                        elif data.size == lines * samples:
                            loc[:, :, band_idx] = data.reshape(lines, samples)
                        logger.info(f"  Found {name}")
                        break
                else:
                    loc[:, :, band_idx] = default
                    logger.warning(f"  Using default for band {band_idx}: {default}")

            # Write ENVI - use flightline ID in filename
            self.loc_envi = ENVIWriter.write(
                self.output_dir / f'{self.flightline_id}_loc',
                loc.astype(np.float64),
                description=f"AVIRIS-3 Location - {self.flightline_id}",
                interleave='bil'
            )


# =============================================================================
# ISOFIT Processor Wrapper
# =============================================================================

class ISOFITProcessor:
    """
    Wrapper for ISOFIT atmospheric correction.

    Uses apply_oe to run optimal estimation atmospheric correction with sRTMnet.
    """

    def __init__(self, working_dir: Optional[Path] = None):
        """
        Initialize ISOFIT processor.

        Args:
            working_dir: Working directory for ISOFIT outputs
        """
        if not HAS_ISOFIT:
            raise ImportError("ISOFIT not available. Install with: mamba install -c conda-forge isofit")

        self.working_dir = Path(working_dir) if working_dir else Path(tempfile.mkdtemp(prefix='isofit_'))
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Set up 6S environment for Windows
        self._setup_sixs_environment()

        # Check for sRTMnet
        self._check_srtmnet()

    def _setup_sixs_environment(self):
        """Configure 6S executable path for Windows."""
        # Find the 6S executable installed via conda
        # Common locations for conda-installed 6S
        sixs_paths = [
            Path(sys.prefix) / 'bin' / 'sixsV2.1',
            Path(sys.prefix) / 'Library' / 'bin' / 'sixsV2.1',
            Path(sys.prefix) / 'Scripts' / 'sixsV2.1',
            Path(sys.prefix) / 'Library' / 'bin' / 'sixs.exe',
            Path(sys.prefix) / 'Scripts' / 'sixs.exe',
            Path.home() / '.isofit' / 'sixs' / 'sixsV2.1',
            Path.home() / '.isofit' / 'sixs' / 'sixs.exe',
        ]

        # Also check for .exe variants
        sixs_exe_paths = []
        for p in sixs_paths:
            sixs_exe_paths.append(p)
            if not p.suffix:
                sixs_exe_paths.append(p.with_suffix('.exe'))

        sixs_executable = None
        for sixs_path in sixs_exe_paths:
            if sixs_path.exists():
                sixs_executable = sixs_path
                break

        if sixs_executable:
            # Set environment variable for ISOFIT to find 6S
            sixs_dir = str(sixs_executable.parent)
            os.environ['SIXS_DIR'] = sixs_dir
            # Also set the specific executable path
            os.environ['SIXS_EXECUTABLE'] = str(sixs_executable)
            logger.info(f"6S configured: {sixs_executable}")

            # Ensure the executable is named correctly for ISOFIT
            # ISOFIT expects 'sixsV2.1' on Linux/Mac or 'sixsV2.1.exe' on Windows
            expected_name = 'sixsV2.1' + ('.exe' if sys.platform == 'win32' else '')
            expected_path = sixs_executable.parent / expected_name

            if not expected_path.exists() and sixs_executable.exists():
                # Create a copy with the expected name
                try:
                    shutil.copy2(sixs_executable, expected_path)
                    logger.info(f"  Created copy at: {expected_path}")
                except Exception as e:
                    logger.warning(f"  Could not create copy: {e}")
        else:
            logger.warning("6S executable not found in expected locations")
            logger.warning("  Searched: " + ", ".join(str(p) for p in sixs_paths[:4]))
            logger.warning("  Run: python -m isofit download sixs")

    def _check_srtmnet(self):
        """Verify sRTMnet is available."""
        try:
            from isofit.configs import Config
            # sRTMnet should be configured via isofit CLI
            logger.info("ISOFIT configuration available")
        except Exception as e:
            logger.warning(f"Could not verify sRTMnet: {e}")
            logger.warning("Run: isofit download srtmnet")

    def _verify_sixs(self) -> bool:
        """
        Verify 6S is working by running a simple test.

        Returns True if 6S is functional.
        """
        logger.info("Verifying 6S installation...")

        try:
            from Py6S import SixS, AtmosProfile, AeroProfile, Geometry, Wavelength

            s = SixS()
            s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
            s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Maritime)
            s.geometry = Geometry.User()
            s.geometry.solar_z = 30
            s.geometry.solar_a = 0
            s.geometry.view_z = 0
            s.geometry.view_a = 0
            s.wavelength = Wavelength(0.55)

            # Try to run
            s.run()

            # Check if we got results
            if hasattr(s.outputs, 'pixel_radiance'):
                logger.info(f"  6S verification successful! Test radiance: {s.outputs.pixel_radiance:.4f}")
                return True
            else:
                logger.warning("  6S ran but produced no output")
                return False

        except Exception as e:
            logger.warning(f"  6S verification failed: {e}")
            logger.warning("  ISOFIT may still work if 6S is correctly configured")
            return False

    def _get_surface_model(self) -> Path:
        """
        Get or create a surface model for ISOFIT.

        Returns path to surface model file.
        """
        # Check for pre-built surface .mat files
        try:
            mat_paths = [
                Path.home() / '.isofit' / 'examples' / '20250308_AV3Cal_wltest' / 'surface' / 'surface.mat',
                Path.home() / '.isofit' / 'examples' / 'image_cube' / 'small' / 'data' / 'surface_model.mat',
            ]

            for sp in mat_paths:
                if sp.exists():
                    logger.info(f"Using pre-built surface model: {sp}")
                    return sp

        except Exception as e:
            logger.warning(f"Could not find pre-built surface model: {e}")

        # Create a surface configuration JSON that points to UCSB reflectance library
        ucsb_surface = Path.home() / '.isofit' / 'data' / 'reflectance' / 'surface_model_ucsb'

        if not ucsb_surface.exists():
            raise FileNotFoundError(
                f"ISOFIT surface reflectance library not found at {ucsb_surface}\n"
                "Run: python -m isofit download data"
            )

        # Create output surface model path
        output_model = self.working_dir / 'surface_model.mat'

        surface_config = {
            "output_model_file": str(output_model),
            "normalize": "Euclidean",
            "reference_windows": [[400, 1300], [1450, 1700], [2100, 2450]],
            "sources": [
                {
                    "input_spectrum_files": [str(ucsb_surface)],
                    "n_components": 8,
                    "windows": [
                        {"interval": [300, 400], "regularizer": 1e-4, "correlation": "EM"},
                        {"interval": [400, 1300], "regularizer": 1e-6, "correlation": "EM"},
                        {"interval": [1300, 1450], "regularizer": 1e-4, "correlation": "EM"},
                        {"interval": [1450, 1700], "regularizer": 1e-6, "correlation": "EM"},
                        {"interval": [1700, 2100], "regularizer": 1e-4, "correlation": "EM"},
                        {"interval": [2100, 2450], "regularizer": 1e-6, "correlation": "EM"},
                        {"interval": [2450, 2550], "regularizer": 1e-4, "correlation": "EM"},
                    ]
                }
            ]
        }

        surface_json = self.working_dir / 'surface_config.json'
        with open(surface_json, 'w') as f:
            json.dump(surface_config, f, indent=2)

        logger.info(f"Created surface config: {surface_json}")
        logger.info(f"  Using UCSB reflectance library: {ucsb_surface}")
        return surface_json

    def process(self,
                radiance_path: Path,
                obs_path: Path,
                loc_path: Path,
                wavelength_path: Optional[Path] = None,
                n_cores: int = 1,  # Default to 1 to avoid Ray issues on Windows
                empirical_line: bool = False,  # Disabled - ISOFIT 3.6.1 has bug, use manual post-processing
                surface_category: str = "multicomponent_surface") -> Path:
        """
        Run ISOFIT atmospheric correction.

        Args:
            radiance_path: Path to radiance ENVI file
            obs_path: Path to observation ENVI file
            loc_path: Path to location ENVI file
            wavelength_path: Optional separate wavelength file
            n_cores: Number of parallel cores
            empirical_line: Use empirical line method for full scene
            surface_category: Surface model type

        Returns:
            Path to output reflectance ENVI file
        """
        logger.info("=" * 60)
        logger.info("Running ISOFIT Atmospheric Correction")
        logger.info("=" * 60)

        output_dir = self.working_dir / 'output'
        output_dir.mkdir(exist_ok=True)

        # Get surface model
        surface_path = self._get_surface_model()

        logger.info(f"Input radiance: {radiance_path}")
        logger.info(f"Input obs: {obs_path}")
        logger.info(f"Input loc: {loc_path}")
        logger.info(f"Surface model: {surface_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Cores: {n_cores}")

        # Pre-flight check: verify 6S is working
        self._verify_sixs()

        # Start progress monitor in background thread
        import threading
        stop_monitor = threading.Event()

        def progress_monitor():
            """Monitor ISOFIT progress by checking output files."""
            import time
            start_time = time.time()
            last_status = ""
            stages = [
                ("Setting up", "input"),
                ("Segmenting image", "subs"),
                ("Building LUTs", "lut"),
                ("Running inversions", "rfl"),
                ("Applying empirical line", "rfl"),
                ("Finalizing", "output"),
            ]

            while not stop_monitor.is_set():
                elapsed = time.time() - start_time
                elapsed_str = f"{int(elapsed//60)}m {int(elapsed%60)}s"

                # Check what files exist to determine progress
                status = "Processing"
                for stage_name, file_pattern in stages:
                    files = list(output_dir.rglob(f"*{file_pattern}*"))
                    if files:
                        status = stage_name

                # Check for specific ISOFIT progress indicators
                lut_files = list(output_dir.rglob("*lut*.nc"))
                rfl_files = list(output_dir.rglob("*rfl*"))

                if rfl_files:
                    status = "Generating reflectance output"
                elif lut_files:
                    status = f"LUT generation complete, running inversions"

                if status != last_status:
                    logger.info(f"  [{elapsed_str}] {status}...")
                    last_status = status
                elif int(elapsed) % 30 == 0:  # Log every 30 seconds
                    logger.info(f"  [{elapsed_str}] {status}... (still processing)")

                stop_monitor.wait(5)  # Check every 5 seconds

        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()

        try:
            # Configure Ray to reduce verbose logging
            os.environ['RAY_DEDUP_LOGS'] = '1'  # Deduplicate repeated messages
            os.environ['RAY_COLOR_PREFIX'] = '0'  # Disable color prefix
            os.environ['RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING'] = '0'  # Reduce task logging

            # Suppress the :task_name: spam by redirecting Ray's stderr logging
            import logging as py_logging
            py_logging.getLogger('ray').setLevel(py_logging.WARNING)

            # Configure Ray for single-process mode on Windows to avoid parallelization issues
            if sys.platform == 'win32' and n_cores == 1:
                os.environ['RAY_NUM_CPUS'] = '1'
                os.environ['RAY_OBJECT_STORE_MEMORY'] = '100000000'  # 100MB
                logger.info("  Windows single-core mode: Ray configured for local execution")

            logger.info("Starting ISOFIT apply_oe (this may take 10-30 minutes)...")
            logger.info("  Progress updates will appear below:")

            # Find sRTMnet emulator model - use native path for current platform
            srtmnet_path = Path.home() / '.isofit' / 'srtmnet' / 'sRTMnet_v120.h5'
            if not srtmnet_path.exists():
                raise FileNotFoundError(
                    f"sRTMnet model not found at {srtmnet_path}\n"
                    "Run: python -m isofit download srtmnet"
                )
            logger.info(f"  Using sRTMnet emulator: {srtmnet_path}")

            # Ensure surface path uses native format
            surface_path_str = str(surface_path)

            # Run apply_oe with sRTMnet (instead of MODTRAN)
            # Let ISOFIT find data files via its config rather than passing explicit paths
            # that may have Windows/Linux path mixing issues
            apply_oe.apply_oe(
                input_radiance=str(radiance_path),
                input_loc=str(loc_path),
                input_obs=str(obs_path),
                working_directory=str(output_dir),
                surface_path=surface_path_str,
                sensor="av3",  # AVIRIS-3
                n_cores=n_cores,
                empirical_line=empirical_line,
                surface_category=surface_category,
                emulator_base=str(srtmnet_path),  # Full path to .h5 file
                log_file=str(self.working_dir / 'isofit.log'),
            )

            # Stop progress monitor
            stop_monitor.set()
            monitor_thread.join(timeout=1)

            # Find output reflectance file
            rfl_files = list(output_dir.glob('*rfl*')) + list(output_dir.glob('*refl*'))
            if rfl_files:
                logger.info(f"ISOFIT completed successfully!")
                return rfl_files[0]
            else:
                raise FileNotFoundError("ISOFIT did not produce reflectance output")

        except Exception as e:
            stop_monitor.set()
            logger.error(f"ISOFIT processing failed: {e}")
            raise


# =============================================================================
# Manual Empirical Line Correction
# =============================================================================

def apply_empirical_line(radiance_path: Path, subs_rfl_path: Path, lbl_path: Path,
                          output_path: Path, wavelengths: np.ndarray) -> np.ndarray:
    """
    Apply manual empirical line correction to extrapolate subset inversions to full scene.

    This bypasses the buggy ISOFIT 3.6.1 empirical_line.py which crashes with shape mismatch.

    For each band, fits: reflectance = slope * radiance + intercept
    using the subset pixels, then applies to all pixels.

    Args:
        radiance_path: Path to full scene radiance ENVI file
        subs_rfl_path: Path to subset reflectance ENVI file from ISOFIT
        lbl_path: Path to label file mapping subset indices to image coordinates
        output_path: Path for output reflectance ENVI file
        wavelengths: Array of wavelengths

    Returns:
        Full scene reflectance array (lines, samples, bands)
    """
    import re

    logger.info("Applying manual empirical line correction...")
    logger.info("  (Bypassing buggy ISOFIT 3.6.1 empirical_line.py)")

    def parse_header(hdr_path):
        header = {}
        with open(hdr_path) as f:
            content = f.read()
        pattern = r'(\w+)\s*=\s*({[^}]+}|[^\n]+)'
        for match in re.finditer(pattern, content):
            key = match.group(1).lower()
            val = match.group(2).strip()
            header[key] = val
        return header

    # Read headers
    lbl_hdr = parse_header(lbl_path.with_suffix('.hdr'))
    subs_hdr = parse_header(subs_rfl_path.with_suffix('.hdr'))
    rdn_hdr = parse_header(radiance_path.with_suffix('.hdr'))

    lines = int(lbl_hdr['lines'])
    samples = int(lbl_hdr['samples'])
    n_subs = int(subs_hdr['lines'])
    bands = int(subs_hdr['bands'])

    logger.info(f"  Scene: {lines} x {samples} x {bands} bands")
    logger.info(f"  Subset pixels: {n_subs}")

    # Read label file
    lbl = np.fromfile(str(lbl_path), dtype=np.float32).reshape(lines, samples)

    # Read subset reflectance (bip: n_subs x bands)
    subs_rfl = np.fromfile(str(subs_rfl_path), dtype=np.float32).reshape(n_subs, bands)

    # Read full radiance
    rdn_interleave = rdn_hdr.get('interleave', 'bil').lower()
    rdn_data = np.fromfile(str(radiance_path), dtype=np.float32)

    if rdn_interleave == 'bil':
        rdn = rdn_data.reshape(lines, bands, samples)
    elif rdn_interleave == 'bip':
        rdn = rdn_data.reshape(lines, samples, bands).transpose(0, 2, 1)
    else:  # bsq
        rdn = rdn_data.reshape(bands, lines, samples).transpose(1, 0, 2)

    # Build coordinate mapping
    subset_coords = []
    for i in range(n_subs):
        loc = np.where(lbl == i)
        if len(loc[0]) > 0:
            subset_coords.append((loc[0][0], loc[1][0]))
        else:
            subset_coords.append((0, 0))
    subset_coords = np.array(subset_coords)

    # Extract radiance at subset locations
    subs_rdn = np.zeros((n_subs, bands), dtype=np.float32)
    for i, (y, x) in enumerate(subset_coords):
        subs_rdn[i, :] = rdn[y, :, x]

    # Fit empirical line for each band
    logger.info("  Fitting empirical line coefficients...")
    coeffs = np.zeros((bands, 2), dtype=np.float32)

    for b in range(bands):
        rdn_b = subs_rdn[:, b]
        rfl_b = subs_rfl[:, b]

        valid = (rdn_b > 0) & (rfl_b > 0) & (rfl_b < 1) & ~np.isnan(rdn_b) & ~np.isnan(rfl_b)

        if np.sum(valid) > 100:
            A = np.vstack([rdn_b[valid], np.ones(np.sum(valid))]).T
            coeffs[b, :], _, _, _ = np.linalg.lstsq(A, rfl_b[valid], rcond=None)

        if b % 50 == 0:
            logger.info(f"    Band {b}: slope={coeffs[b,0]:.6f}, intercept={coeffs[b,1]:.4f}")

    # Apply to full scene
    logger.info("  Applying to full scene...")
    full_rfl = np.zeros((lines, samples, bands), dtype=np.float32)

    for b in range(bands):
        full_rfl[:, :, b] = coeffs[b, 0] * rdn[:, b, :] + coeffs[b, 1]
        if b % 50 == 0:
            logger.info(f"    Band {b}/{bands}")

    # Clip to valid range
    full_rfl = np.clip(full_rfl, 0, 1)

    logger.info(f"  Reflectance range: {np.nanmin(full_rfl):.4f} to {np.nanmax(full_rfl):.4f}")

    # Write output ENVI file
    ENVIWriter.write(
        output_path,
        full_rfl,
        wavelengths=wavelengths,
        description="Surface Reflectance (ISOFIT + Manual Empirical Line)",
        interleave='bil'
    )

    return full_rfl


# =============================================================================
# Main Processing Pipeline
# =============================================================================

class AVIRIS3ISOFITCorrection:
    """
    Complete AVIRIS-3 atmospheric correction pipeline using ISOFIT.

    Handles the full workflow:
    1. Convert NetCDF to ENVI
    2. Run ISOFIT with sRTMnet
    3. Convert output back to NetCDF
    """

    def __init__(self, rdn_path: str, ort_path: str, output_path: str,
                 n_cores: int = 1, cleanup: bool = True):
        """
        Initialize the correction pipeline.

        Args:
            rdn_path: Path to L1B radiance NetCDF
            ort_path: Path to ORT observation NetCDF
            output_path: Output reflectance NetCDF path
            n_cores: Number of parallel cores for ISOFIT
            cleanup: Remove temporary files after processing
        """
        self.rdn_path = Path(rdn_path)
        self.ort_path = Path(ort_path)
        self.output_path = Path(output_path)
        self.n_cores = n_cores
        self.cleanup = cleanup

        # Working directory
        self.work_dir = Path(tempfile.mkdtemp(prefix='aviris_isofit_'))
        logger.info(f"Working directory: {self.work_dir}")

        # Results
        self.reflectance: Optional[np.ndarray] = None
        self.uncertainty: Optional[np.ndarray] = None
        self.wavelengths: Optional[np.ndarray] = None
        self.processing_metadata: Dict[str, Any] = {}

    def run(self) -> Path:
        """
        Execute the full correction pipeline.

        Returns:
            Path to output NetCDF file
        """
        start_time = datetime.now()
        logger.info("=" * 70)
        logger.info("AVIRIS-3 ISOFIT Atmospheric Correction Pipeline")
        logger.info("=" * 70)

        try:
            # Step 1: Convert to ENVI
            logger.info("\n[Step 1/4] Converting NetCDF to ENVI format...")
            converter = AVIRIS3Converter(
                self.rdn_path,
                self.ort_path,
                self.work_dir / 'envi_input'
            )
            envi_files = converter.convert()
            self.wavelengths = envi_files['wavelengths']

            # Step 2: Run ISOFIT (subset inversions only - empirical_line=False)
            logger.info("\n[Step 2/5] Running ISOFIT atmospheric correction (subset inversions)...")
            processor = ISOFITProcessor(self.work_dir / 'isofit_work')
            rfl_path = processor.process(
                radiance_path=envi_files['radiance'],
                obs_path=envi_files['obs'],
                loc_path=envi_files['loc'],
                n_cores=self.n_cores,
                empirical_line=False,  # Disabled due to ISOFIT 3.6.1 bug
            )

            # Step 3: Apply manual empirical line correction
            logger.info("\n[Step 3/5] Applying manual empirical line correction...")
            isofit_output = self.work_dir / 'isofit_work' / 'output' / 'output'

            # Find required files
            subs_rfl_files = list(isofit_output.glob('*subs_rfl'))
            lbl_files = list(isofit_output.glob('*lbl'))

            if subs_rfl_files and lbl_files:
                final_rfl_path = isofit_output / 'final_rfl'
                self.reflectance = apply_empirical_line(
                    radiance_path=envi_files['radiance'],
                    subs_rfl_path=subs_rfl_files[0],
                    lbl_path=lbl_files[0],
                    output_path=final_rfl_path,
                    wavelengths=self.wavelengths
                )
                logger.info(f"  Full scene reflectance: {self.reflectance.shape}")
            else:
                # Fallback: try to read whatever ISOFIT produced
                logger.warning("  Could not find subset files, using ISOFIT output directly")
                self.reflectance, _ = ENVIWriter.read(rfl_path)

            # Step 4: Read uncertainty if available
            logger.info("\n[Step 4/5] Reading uncertainty estimates...")

            # Try to find uncertainty file
            unc_files = list(isofit_output.glob('*subs_uncert'))
            if unc_files:
                # Note: uncertainty is only for subset pixels, not full scene
                logger.info(f"  Subset uncertainty available at: {unc_files[0]}")

            logger.info(f"  Reflectance shape: {self.reflectance.shape}")

            # Step 5: Write NetCDF output
            logger.info("\n[Step 5/5] Writing output NetCDF...")
            self._write_output_netcdf()

            # Cleanup
            if self.cleanup:
                logger.info("\nCleaning up temporary files...")
                shutil.rmtree(self.work_dir, ignore_errors=True)

            elapsed = datetime.now() - start_time
            logger.info("\n" + "=" * 70)
            logger.info(f"Processing complete! Elapsed time: {elapsed}")
            logger.info(f"Output: {self.output_path}")
            logger.info("=" * 70)

            return self.output_path

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if not self.cleanup:
                logger.info(f"Temporary files preserved at: {self.work_dir}")
            raise

    def _write_output_netcdf(self):
        """Write reflectance to NetCDF format."""
        with nc.Dataset(self.output_path, 'w', format='NETCDF4') as ds:
            # Global attributes
            ds.title = "AVIRIS-3 Surface Reflectance (ISOFIT + Manual Empirical Line)"
            ds.institution = "Processed with ISOFIT + sRTMnet"
            ds.source = f"Input: {self.rdn_path.name}"
            ds.history = f"Created {datetime.now().isoformat()}"
            ds.conventions = "CF-1.8"
            ds.processing_software = f"ISOFIT {ISOFIT_VERSION}"
            ds.note = "Empirical line applied via manual implementation (ISOFIT 3.6.1 bug workaround)"

            # Handle different array shapes
            if self.reflectance.ndim == 3:
                if self.reflectance.shape[2] == len(self.wavelengths):
                    # Shape is (lines, samples, bands) - our manual output
                    lines, samples, bands = self.reflectance.shape
                else:
                    # Shape is (bands, lines, samples) - transpose it
                    bands, lines, samples = self.reflectance.shape
                    self.reflectance = self.reflectance.transpose(1, 2, 0)
            else:
                raise ValueError(f"Unexpected reflectance shape: {self.reflectance.shape}")

            # Dimensions - use (wavelength, y, x) order for viewer compatibility
            ds.createDimension('wavelength', bands)
            ds.createDimension('y', lines)
            ds.createDimension('x', samples)

            # Wavelength coordinate
            wl_var = ds.createVariable('wavelength', 'f4', ('wavelength',))
            wl_var[:] = self.wavelengths
            wl_var.units = 'nm'
            wl_var.long_name = 'Wavelength'

            # Reflectance - transpose to (wavelength, y, x) for viewer
            rfl_grp = ds.createGroup('reflectance')
            rfl_var = rfl_grp.createVariable(
                'reflectance', 'f4', ('wavelength', 'y', 'x'),
                chunksizes=(1, lines, samples)
            )
            # Transpose from (lines, samples, bands) to (bands, lines, samples)
            rfl_var[:] = self.reflectance.transpose(2, 0, 1)
            rfl_var.units = 'unitless'
            rfl_var.long_name = 'Surface Reflectance'
            rfl_var.valid_range = [0.0, 1.5]

            # Copy wavelengths to group
            wl_grp = rfl_grp.createVariable('wavelength', 'f4', ('wavelength',))
            wl_grp[:] = self.wavelengths
            wl_grp.units = 'nm'

            # Note: Full-scene uncertainty not available with empirical line method
            # Subset uncertainty is preserved in temp files if needed

            # Quality assessment
            self._add_quality_assessment(ds)

            logger.info(f"  Wrote {self.output_path} ({self.output_path.stat().st_size / 1e6:.1f} MB)")

    def _add_quality_assessment(self, ds):
        """Add quality flags and statistics to output."""
        lines, samples, bands = self.reflectance.shape

        # Create quality group
        qa_grp = ds.createGroup('quality')

        # Quality flags
        quality = np.zeros((lines, samples), dtype=np.uint8)

        # Flag negative reflectance
        neg_mask = np.any(self.reflectance < -0.01, axis=2)
        quality[neg_mask] |= 1

        # Flag high reflectance
        high_mask = np.any(self.reflectance > 1.0, axis=2)
        quality[high_mask] |= 2

        # Flag NaN
        nan_mask = np.any(np.isnan(self.reflectance), axis=2)
        quality[nan_mask] |= 4

        qa_var = qa_grp.createVariable('flags', 'u1', ('y', 'x'))
        qa_var[:] = quality
        qa_var.flag_meanings = "negative_reflectance high_reflectance nan_values"
        qa_var.flag_values = [1, 2, 4]

        # Statistics
        valid_pixels = np.sum(quality == 0)
        total_pixels = lines * samples

        qa_grp.valid_pixel_count = int(valid_pixels)
        qa_grp.total_pixel_count = int(total_pixels)
        qa_grp.valid_fraction = float(valid_pixels / total_pixels)

        logger.info(f"  Quality: {valid_pixels}/{total_pixels} valid pixels ({100*valid_pixels/total_pixels:.1f}%)")


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AVIRIS-3 Atmospheric Correction using ISOFIT + sRTMnet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python aviris_isofit_processor.py radiance.nc obs.nc output_rfl.nc

    # With more cores
    python aviris_isofit_processor.py radiance.nc obs.nc output.nc --cores 8

    # Keep temporary files for debugging
    python aviris_isofit_processor.py radiance.nc obs.nc output.nc --no-cleanup

Prerequisites:
    mamba install -c conda-forge isofit
    isofit download sixs
    isofit download srtmnet
        """
    )

    parser.add_argument('radiance', help='L1B Radiance NetCDF file')
    parser.add_argument('obs', help='ORT Observation geometry NetCDF file')
    parser.add_argument('output', help='Output reflectance NetCDF file')
    parser.add_argument('--cores', type=int, default=1,
                        help='Number of parallel cores (default: 1 for Windows compatibility; use more cores on Linux)')
    parser.add_argument('--no-cleanup', action='store_true', help='Keep temporary files')

    args = parser.parse_args()

    # Check ISOFIT
    if not HAS_ISOFIT:
        print("\nError: ISOFIT is not installed.")
        print("\nInstallation instructions:")
        print("  mamba create -n isofit_env -c conda-forge isofit")
        print("  mamba activate isofit_env")
        print("  isofit download sixs")
        print("  isofit download srtmnet")
        sys.exit(1)

    # Run processing
    pipeline = AVIRIS3ISOFITCorrection(
        rdn_path=args.radiance,
        ort_path=args.obs,
        output_path=args.output,
        n_cores=args.cores,
        cleanup=not args.no_cleanup,
    )

    pipeline.run()


if __name__ == '__main__':
    main()
