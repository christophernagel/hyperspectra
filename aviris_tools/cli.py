"""
Command-line interface for AVIRIS-3 tools.

Provides unified CLI access to all processing and visualization tools.

Usage:
    aviris-process radiance.nc obs.nc output.nc --method py6s
    aviris-view output.nc
"""

import sys
import logging
import argparse
from pathlib import Path


def process_cli():
    """Main processing CLI."""
    parser = argparse.ArgumentParser(
        prog='aviris-process',
        description='AVIRIS-3 Atmospheric Correction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  py6s    - Py6S radiative transfer (default, runs on Windows)
  isofit  - NASA ISOFIT + sRTMnet (requires WSL/Linux)

Examples:
  aviris-process radiance.nc obs.nc output.nc
  aviris-process radiance.nc obs.nc output.nc --method isofit --cores 4
  aviris-process radiance.nc obs.nc output.nc --aerosol urban
        """
    )

    parser.add_argument('radiance', help='Input L1B radiance NetCDF')
    parser.add_argument('obs', help='Input observation NetCDF')
    parser.add_argument('output', help='Output reflectance NetCDF')

    parser.add_argument('--method', '-m', choices=['py6s', 'isofit'],
                        default='py6s', help='Correction method (default: py6s)')
    parser.add_argument('--aerosol', '-a',
                        choices=['maritime', 'continental', 'urban', 'desert'],
                        default=None, help='Aerosol model')
    parser.add_argument('--altitude', type=float, default=None,
                        help='Sensor altitude km')
    parser.add_argument('--cores', '-c', type=int, default=None,
                        help='Parallel cores (ISOFIT only)')
    parser.add_argument('--simple', action='store_true',
                        help='Use simplified empirical method (Py6S only)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip output validation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    logger = logging.getLogger(__name__)

    # Validate inputs
    radiance_path = Path(args.radiance)
    obs_path = Path(args.obs)

    if not radiance_path.exists():
        logger.error(f"Radiance file not found: {radiance_path}")
        sys.exit(1)
    if not obs_path.exists():
        logger.error(f"Observation file not found: {obs_path}")
        sys.exit(1)

    # Run appropriate processor
    if args.method == 'py6s':
        from aviris_tools.atm_correction.py6s_processor import Py6SProcessor

        processor = Py6SProcessor(
            radiance_path=str(radiance_path),
            obs_path=str(obs_path),
            output_path=args.output,
            aerosol_model=args.aerosol,
            altitude_km=args.altitude,
            validate=not args.no_validate
        )
        processor.run(use_6s=not args.simple)

    elif args.method == 'isofit':
        from aviris_tools.atm_correction.isofit_processor import ISOFITProcessor

        processor = ISOFITProcessor(
            radiance_path=str(radiance_path),
            obs_path=str(obs_path),
            output_path=args.output,
            n_cores=args.cores
        )
        processor.run()

    logger.info(f"Output written to: {args.output}")


def view_cli():
    """Viewer CLI."""
    parser = argparse.ArgumentParser(
        prog='aviris-view',
        description='AVIRIS-3 Hyperspectral Viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('filepath', nargs='?', default=None,
                        help='NetCDF file to open')
    parser.add_argument('--simple', action='store_true',
                        help='Use matplotlib instead of napari')
    parser.add_argument('--scale', type=float, default=None,
                        help='UI scale factor')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )

    import os
    if args.scale:
        os.environ['QT_SCALE_FACTOR'] = str(args.scale)

    if args.simple:
        from aviris_tools.viewer.main import launch_viewer_simple
        if args.filepath:
            launch_viewer_simple(args.filepath)
        else:
            print("Error: --simple requires a filepath")
            sys.exit(1)
    else:
        from aviris_tools.viewer.main import launch_viewer
        launch_viewer(args.filepath)


def config_cli():
    """Configuration management CLI."""
    parser = argparse.ArgumentParser(
        prog='aviris-config',
        description='AVIRIS-3 Tools Configuration',
    )

    parser.add_argument('--show', action='store_true',
                        help='Show current configuration')
    parser.add_argument('--init', action='store_true',
                        help='Create config file template')
    parser.add_argument('--check', action='store_true',
                        help='Check dependencies')

    args = parser.parse_args()

    from aviris_tools.utils.config import get_config

    config = get_config()

    if args.show:
        import yaml
        print(yaml.dump(config._config, default_flow_style=False))

    elif args.init:
        config_path = Path.home() / '.aviris_tools' / 'config.yaml'
        if config_path.exists():
            print(f"Config already exists: {config_path}")
        else:
            config.save(config_path)
            print(f"Created config: {config_path}")

    elif args.check:
        print("Dependency Check")
        print("=" * 40)

        # Memory
        from aviris_tools.utils.memory import get_total_memory, get_available_memory
        print(f"Total Memory: {get_total_memory():.1f} GB")
        print(f"Available: {get_available_memory():.1f} GB")
        print(f"Config Limit: {config.memory_limit_gb:.1f} GB")
        print()

        # 6S
        print(f"6S Executable: {config.sixs_path or 'Not found'}")

        # sRTMnet
        srtmnet = config.get('paths', 'srtmnet_model')
        print(f"sRTMnet Model: {srtmnet or 'Not found'}")
        print()

        # Python packages
        packages = ['numpy', 'scipy', 'netCDF4', 'napari', 'Py6S', 'isofit']
        print("Python Packages:")
        for pkg in packages:
            try:
                mod = __import__(pkg)
                version = getattr(mod, '__version__', 'installed')
                print(f"  {pkg}: {version}")
            except ImportError:
                print(f"  {pkg}: NOT INSTALLED")

    else:
        parser.print_help()


def main():
    """Main entry point - dispatch to appropriate CLI."""
    if len(sys.argv) < 2:
        print("AVIRIS-3 Processing Tools")
        print()
        print("Commands:")
        print("  aviris-process  - Atmospheric correction")
        print("  aviris-view     - Hyperspectral viewer")
        print("  aviris-config   - Configuration management")
        print()
        print("Use --help with any command for details.")
        sys.exit(0)

    # Simple dispatch based on script name
    script_name = Path(sys.argv[0]).stem
    if 'process' in script_name:
        process_cli()
    elif 'view' in script_name:
        view_cli()
    elif 'config' in script_name:
        config_cli()
    else:
        # Default to process
        process_cli()


if __name__ == '__main__':
    main()
