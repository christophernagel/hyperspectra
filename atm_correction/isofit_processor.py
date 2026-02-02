"""
ISOFIT-based atmospheric correction for AVIRIS-3.

This module wraps the aviris_isofit_processor.py with:
- Configurable paths via config.yaml
- Auto-detected memory limits
- Clean Python API

ISOFIT uses optimal estimation with sRTMnet neural network emulator
for physics-based atmospheric correction.

Note: ISOFIT processing should be run from WSL/Linux due to shell script
generation for 6S LUT building.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional

# Add parent to path for legacy import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aviris_tools.utils.config import get_config
from aviris_tools.utils.memory import MemoryManager

logger = logging.getLogger(__name__)


class ISOFITProcessor:
    """
    ISOFIT-based atmospheric correction processor.

    Uses NASA/JPL ISOFIT with sRTMnet for optimal estimation
    atmospheric correction. Includes workaround for ISOFIT 3.6.1
    empirical line bug.
    """

    def __init__(self,
                 radiance_path: str,
                 obs_path: str,
                 output_path: str,
                 n_cores: Optional[int] = None,
                 cleanup: bool = True):
        """
        Initialize processor.

        Args:
            radiance_path: Path to L1B radiance NetCDF
            obs_path: Path to observation geometry NetCDF
            output_path: Output reflectance NetCDF path
            n_cores: Number of parallel cores (auto-detect if None)
            cleanup: Remove temporary files after processing
        """
        self.radiance_path = Path(radiance_path)
        self.obs_path = Path(obs_path)
        self.output_path = Path(output_path)

        # Get config
        config = get_config()

        # Apply config defaults
        self.n_cores = n_cores or config.get('processing', 'n_cores', default=4)
        self.cleanup = cleanup

        # Memory management
        self.memory_manager = MemoryManager(
            limit_gb=config.get('memory', 'limit_gb')
        )

        # Check for sRTMnet
        self.srtmnet_path = config.get('paths', 'srtmnet_model')
        if not self.srtmnet_path:
            default_path = Path.home() / '.isofit' / 'srtmnet' / 'sRTMnet_v120.h5'
            if default_path.exists():
                self.srtmnet_path = str(default_path)

        # Validate inputs
        if not self.radiance_path.exists():
            raise FileNotFoundError(f"Radiance file not found: {self.radiance_path}")
        if not self.obs_path.exists():
            raise FileNotFoundError(f"Observation file not found: {self.obs_path}")

        # Check platform
        if sys.platform == 'win32':
            logger.warning("ISOFIT works best in WSL/Linux due to shell script generation")
            logger.warning("Consider running from WSL: wsl bash -c 'conda activate isofit_env && ...'")

        logger.info(f"ISOFITProcessor initialized")
        logger.info(f"  Radiance: {self.radiance_path.name}")
        logger.info(f"  Observation: {self.obs_path.name}")
        logger.info(f"  Output: {self.output_path}")
        logger.info(f"  Cores: {self.n_cores}")
        if self.srtmnet_path:
            logger.info(f"  sRTMnet: {Path(self.srtmnet_path).name}")

    def run(self) -> Path:
        """
        Execute ISOFIT atmospheric correction.

        This runs:
        1. NetCDF to ENVI conversion
        2. ISOFIT optimal estimation (subset inversions)
        3. Manual empirical line extrapolation (bypasses ISOFIT bug)
        4. Output to NetCDF

        Returns:
            Path to output file
        """
        try:
            from aviris_isofit_processor import AVIRIS3ISOFITCorrection

            pipeline = AVIRIS3ISOFITCorrection(
                rdn_path=str(self.radiance_path),
                ort_path=str(self.obs_path),
                output_path=str(self.output_path),
                n_cores=self.n_cores,
                cleanup=self.cleanup
            )

            return Path(pipeline.run())

        except ImportError as e:
            logger.error(f"Could not import ISOFIT processor: {e}")
            logger.error("Ensure aviris_isofit_processor.py is in the path")
            logger.error("And that ISOFIT is installed: conda install -c conda-forge isofit")
            raise

    def check_dependencies(self) -> dict:
        """
        Check ISOFIT dependencies.

        Returns:
            Dict with dependency status
        """
        status = {
            'isofit': False,
            'srtmnet': False,
            'sixs': False,
            'ray': False,
        }

        try:
            import isofit
            status['isofit'] = True
            status['isofit_version'] = getattr(isofit, '__version__', 'unknown')
        except ImportError:
            pass

        try:
            import ray
            status['ray'] = True
        except ImportError:
            pass

        # Check sRTMnet
        if self.srtmnet_path and Path(self.srtmnet_path).exists():
            status['srtmnet'] = True
            status['srtmnet_path'] = self.srtmnet_path

        # Check 6S
        config = get_config()
        sixs_path = config.sixs_path
        if sixs_path and Path(sixs_path).exists():
            status['sixs'] = True
            status['sixs_path'] = sixs_path

        return status


def process_file(radiance_path: str,
                 obs_path: str,
                 output_path: str,
                 n_cores: int = 4) -> Path:
    """
    Convenience function for ISOFIT atmospheric correction.

    Args:
        radiance_path: Input radiance file
        obs_path: Input observation file
        output_path: Output reflectance file
        n_cores: Number of parallel cores

    Returns:
        Path to output file
    """
    processor = ISOFITProcessor(
        radiance_path=radiance_path,
        obs_path=obs_path,
        output_path=output_path,
        n_cores=n_cores
    )
    return processor.run()


# CLI entry point
def main():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='AVIRIS-3 Atmospheric Correction (ISOFIT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: ISOFIT processing is recommended to run from WSL/Linux.
The processor generates shell scripts for 6S that require bash.

Example (from WSL):
    conda activate isofit_env
    python -m aviris_tools.atm_correction.isofit_processor \\
        radiance.nc obs.nc output.nc --cores 4
        """
    )

    parser.add_argument('radiance', help='Input L1B radiance NetCDF')
    parser.add_argument('obs', help='Input observation NetCDF')
    parser.add_argument('output', help='Output reflectance NetCDF')

    parser.add_argument('--cores', '-c', type=int, default=None,
                        help='Number of parallel cores (default: auto-detect)')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Keep temporary files for debugging')
    parser.add_argument('--check', action='store_true',
                        help='Check dependencies and exit')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    processor = ISOFITProcessor(
        radiance_path=args.radiance,
        obs_path=args.obs,
        output_path=args.output,
        n_cores=args.cores,
        cleanup=not args.no_cleanup
    )

    if args.check:
        status = processor.check_dependencies()
        print("\nDependency Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        return

    processor.run()


if __name__ == '__main__':
    main()
