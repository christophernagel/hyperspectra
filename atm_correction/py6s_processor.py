"""
Py6S-based atmospheric correction for AVIRIS-3.

This module wraps the aviris_atm_correction_v2.py with:
- Configurable paths via config.yaml
- Auto-detected memory limits
- Clean Python API

For direct CLI usage, use aviris_atm_correction_v2.py directly.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent to path for legacy import (aviris_tools/ contains aviris_atm_correction_v2.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from aviris_tools.utils.config import get_config
from aviris_tools.utils.memory import MemoryManager, get_available_memory

logger = logging.getLogger(__name__)


class Py6SProcessor:
    """
    Py6S-based atmospheric correction processor.

    Converts AVIRIS-3 L1B radiance to surface reflectance using
    6S radiative transfer modeling with scene-derived atmospheric parameters.
    """

    def __init__(self,
                 radiance_path: str,
                 obs_path: str,
                 output_path: str,
                 aerosol_model: Optional[str] = None,
                 altitude_km: Optional[float] = None,
                 validate: bool = True,
                 coastal_correction: bool = True,
                 estimate_uncertainty: bool = True):
        """
        Initialize processor.

        Args:
            radiance_path: Path to L1B radiance NetCDF
            obs_path: Path to observation geometry NetCDF
            output_path: Output reflectance NetCDF path
            aerosol_model: Aerosol type (continental, maritime, urban, desert)
            altitude_km: Sensor altitude in km (auto-detect if None)
            validate: Run validation checks on output
            coastal_correction: Apply coastal/adjacency correction
            estimate_uncertainty: Estimate per-pixel uncertainty
        """
        self.radiance_path = Path(radiance_path)
        self.obs_path = Path(obs_path)
        self.output_path = Path(output_path)

        # Get config
        config = get_config()

        # Apply config defaults
        self.aerosol_model = aerosol_model or config.get('atmospheric', 'aerosol_model', default='continental')
        self.altitude_km = altitude_km or config.get('atmospheric', 'default_altitude_km', default=8.5)
        self.validate = validate
        self.coastal_correction = coastal_correction
        self.estimate_uncertainty = estimate_uncertainty

        # Memory management
        self.memory_manager = MemoryManager(
            limit_gb=config.get('memory', 'limit_gb')
        )

        # Validate inputs exist
        if not self.radiance_path.exists():
            raise FileNotFoundError(f"Radiance file not found: {self.radiance_path}")
        if not self.obs_path.exists():
            raise FileNotFoundError(f"Observation file not found: {self.obs_path}")

        logger.info(f"Py6SProcessor initialized")
        logger.info(f"  Radiance: {self.radiance_path.name}")
        logger.info(f"  Observation: {self.obs_path.name}")
        logger.info(f"  Output: {self.output_path}")
        logger.info(f"  Aerosol model: {self.aerosol_model}")
        logger.info(f"  Memory limit: {self.memory_manager.limit_gb:.1f} GB")

    def run(self, use_6s: bool = True) -> Path:
        """
        Execute atmospheric correction.

        Args:
            use_6s: Use Py6S radiative transfer (True) or empirical method (False)

        Returns:
            Path to output file
        """
        # Import the legacy processor
        try:
            # Apply config patches before importing
            self._patch_legacy_config()

            from aviris_atm_correction_v2 import AVIRISL2Processor

            processor = AVIRISL2Processor(
                str(self.radiance_path),
                str(self.obs_path),
                sensor_altitude_km=self.altitude_km,
                aerosol_model=self.aerosol_model,
                coastal_correction=self.coastal_correction,
                estimate_uncertainty=self.estimate_uncertainty
            )

            try:
                processor.process(str(self.output_path), use_6s=use_6s)
            finally:
                processor.close()

            # Run validation if requested
            if self.validate:
                self._validate_output()

            return self.output_path

        except ImportError as e:
            logger.error(f"Could not import legacy processor: {e}")
            logger.error("Ensure aviris_atm_correction_v2.py is in the path")
            raise

    def _patch_legacy_config(self):
        """Patch legacy module constants with config values."""
        import aviris_atm_correction_v2 as legacy

        config = get_config()

        # Patch memory limit
        legacy.MEMORY_LIMIT_GB = self.memory_manager.limit_gb

        # Patch 6S path if configured
        sixs_path = config.get('paths', 'sixs_executable')
        if sixs_path:
            legacy.SIXS_PATH = sixs_path
            legacy.HAS_PY6S = True

        logger.debug("Patched legacy config with current settings")

    def _validate_output(self):
        """Validate output reflectance values."""
        import netCDF4 as nc
        import numpy as np

        logger.info("Validating output...")

        with nc.Dataset(self.output_path) as ds:
            # Find reflectance variable
            if 'reflectance' in ds.groups:
                rfl = ds.groups['reflectance'].variables['reflectance'][:]
            else:
                rfl = ds.variables['reflectance'][:]

            # Check ranges
            valid_mask = np.isfinite(rfl)
            if np.any(valid_mask):
                rfl_valid = rfl[valid_mask]
                min_val = np.min(rfl_valid)
                max_val = np.max(rfl_valid)
                mean_val = np.mean(rfl_valid)

                logger.info(f"  Reflectance range: {min_val:.4f} to {max_val:.4f}")
                logger.info(f"  Mean reflectance: {mean_val:.4f}")

                # Warnings
                if min_val < -0.05:
                    logger.warning(f"  Significant negative reflectance detected: {min_val:.4f}")
                if max_val > 1.0:
                    logger.warning(f"  Reflectance > 1.0 detected: {max_val:.4f}")
                if mean_val < 0.01:
                    logger.warning(f"  Very low mean reflectance - check atmospheric parameters")


def process_file(radiance_path: str,
                 obs_path: str,
                 output_path: str,
                 aerosol: str = 'continental',
                 validate: bool = True) -> Path:
    """
    Convenience function for atmospheric correction.

    Args:
        radiance_path: Input radiance file
        obs_path: Input observation file
        output_path: Output reflectance file
        aerosol: Aerosol model (continental, maritime, urban, desert)
        validate: Run validation on output

    Returns:
        Path to output file
    """
    processor = Py6SProcessor(
        radiance_path=radiance_path,
        obs_path=obs_path,
        output_path=output_path,
        aerosol_model=aerosol,
        validate=validate
    )
    return processor.run()


# CLI entry point
def main():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='AVIRIS-3 Atmospheric Correction (Py6S)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('radiance', help='Input L1B radiance NetCDF')
    parser.add_argument('obs', help='Input observation NetCDF')
    parser.add_argument('output', help='Output reflectance NetCDF')

    parser.add_argument('--aerosol', '-a',
                        choices=['maritime', 'continental', 'urban', 'desert'],
                        default=None, help='Aerosol model (default: from config)')
    parser.add_argument('--altitude', type=float, default=None,
                        help='Sensor altitude km (default: auto-detect)')
    parser.add_argument('--simple', action='store_true',
                        help='Use simplified empirical method (no 6S)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip output validation')
    parser.add_argument('--no-coastal', action='store_true',
                        help='Disable coastal correction')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    processor = Py6SProcessor(
        radiance_path=args.radiance,
        obs_path=args.obs,
        output_path=args.output,
        aerosol_model=args.aerosol,
        altitude_km=args.altitude,
        validate=not args.no_validate,
        coastal_correction=not args.no_coastal
    )

    processor.run(use_6s=not args.simple)


if __name__ == '__main__':
    main()
