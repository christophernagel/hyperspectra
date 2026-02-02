"""
Configuration management for AVIRIS-3 tools.

Supports:
    - YAML configuration files
    - Environment variable overrides
    - Auto-detection of system paths
    - Site-specific parameter tuning
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    # Paths - auto-detected if not specified
    'paths': {
        'sixs_executable': None,  # Auto-detect
        'srtmnet_model': None,    # Auto-detect
        'isofit_data': None,      # Auto-detect
        'output_dir': None,       # Current directory
    },

    # Memory management
    'memory': {
        'limit_gb': None,         # Auto-detect (use 80% of available)
        'chunk_size_mb': 512,
        'cache_size_mb': 1024,
    },

    # Atmospheric correction defaults
    'atmospheric': {
        'aerosol_model': 'continental',  # continental, maritime, urban
        'default_altitude_km': 8.5,      # AVIRIS-3 typical (King Air B-200)
        'water_vapor_bands': [870, 940, 1000],
        'aod_reference_band': 550,
    },

    # Spectral bands to mask (atmospheric absorption)
    'absorption_bands': {
        'water_vapor_1': [1350, 1450],
        'water_vapor_2': [1800, 1950],
        'o2_absorption': [760, 770],
    },

    # Processing options
    'processing': {
        'n_cores': None,          # Auto-detect
        'use_gpu': False,
        'validate_output': True,
        'preserve_temp_files': False,
    },

    # Viewer settings
    'viewer': {
        'default_rgb_bands': [640, 550, 460],  # nm
        'colormap': 'viridis',
        'qt_scale_factor': 1.0,
    },
}


class Config:
    """Configuration manager with auto-detection and validation."""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config = DEFAULT_CONFIG.copy()
        self._load_config_file()
        self._apply_env_overrides()
        self._auto_detect_paths()
        self._auto_detect_resources()

    def _load_config_file(self):
        """Load configuration from YAML file if present."""
        config_paths = [
            Path.home() / '.aviris_tools' / 'config.yaml',
            Path.home() / '.config' / 'aviris_tools' / 'config.yaml',
            Path.cwd() / 'aviris_config.yaml',
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        user_config = yaml.safe_load(f)
                    self._merge_config(user_config)
                    logger.info(f"Loaded config from: {config_path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {config_path}: {e}")

    def _merge_config(self, user_config: Dict):
        """Deep merge user config into default config."""
        def merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge(base[key], value)
                else:
                    base[key] = value
        merge(self._config, user_config)

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mapping = {
            'AVIRIS_SIXS_PATH': ('paths', 'sixs_executable'),
            'AVIRIS_SRTMNET_PATH': ('paths', 'srtmnet_model'),
            'AVIRIS_MEMORY_LIMIT': ('memory', 'limit_gb'),
            'AVIRIS_N_CORES': ('processing', 'n_cores'),
            'SIXS_DIR': ('paths', 'sixs_executable'),  # ISOFIT convention
        }

        for env_var, config_path in env_mapping.items():
            if env_var in os.environ:
                section, key = config_path
                value = os.environ[env_var]
                # Type conversion
                if key in ['limit_gb']:
                    value = float(value)
                elif key in ['n_cores']:
                    value = int(value)
                self._config[section][key] = value
                logger.debug(f"Config override from {env_var}: {section}.{key} = {value}")

    def _auto_detect_paths(self):
        """Auto-detect executable and data paths."""
        # 6S executable
        if not self._config['paths']['sixs_executable']:
            sixs_path = self._find_sixs()
            if sixs_path:
                self._config['paths']['sixs_executable'] = str(sixs_path)
                logger.info(f"Auto-detected 6S: {sixs_path}")

        # sRTMnet model
        if not self._config['paths']['srtmnet_model']:
            srtmnet_path = Path.home() / '.isofit' / 'srtmnet' / 'sRTMnet_v120.h5'
            if srtmnet_path.exists():
                self._config['paths']['srtmnet_model'] = str(srtmnet_path)
                logger.info(f"Auto-detected sRTMnet: {srtmnet_path}")

    def _find_sixs(self) -> Optional[Path]:
        """Find 6S executable on system."""
        # Check PATH first
        import shutil
        for name in ['sixsV2.1', 'sixs', 'sixsV2.1.exe', 'sixs.exe']:
            path = shutil.which(name)
            if path:
                return Path(path)

        # Check common locations
        search_paths = [
            # Conda environments
            Path(sys.prefix) / 'bin' / 'sixsV2.1',
            Path(sys.prefix) / 'Library' / 'bin' / 'sixsV2.1',
            Path(sys.prefix) / 'Scripts' / 'sixsV2.1',
            # ISOFIT download location
            Path.home() / '.isofit' / 'sixs' / 'sixsV2.1',
            # Linux/WSL
            Path('/usr/local/bin/sixsV2.1'),
        ]

        # Add .exe variants for Windows
        if sys.platform == 'win32':
            search_paths = [p.with_suffix('.exe') if not p.suffix else p for p in search_paths]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _auto_detect_resources(self):
        """Auto-detect system resources."""
        # Memory
        if not self._config['memory']['limit_gb']:
            try:
                import psutil
                total_gb = psutil.virtual_memory().total / (1024**3)
                self._config['memory']['limit_gb'] = round(total_gb * 0.8, 1)
                logger.info(f"Auto-detected memory limit: {self._config['memory']['limit_gb']} GB")
            except ImportError:
                self._config['memory']['limit_gb'] = 8.0  # Conservative fallback

        # CPU cores
        if not self._config['processing']['n_cores']:
            self._config['processing']['n_cores'] = os.cpu_count() or 4

    def get(self, *keys, default=None):
        """Get nested config value."""
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys_and_value):
        """Set nested config value."""
        *keys, value = keys_and_value
        target = self._config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value

    def save(self, path: Optional[Path] = None):
        """Save current config to YAML file."""
        if path is None:
            path = Path.home() / '.aviris_tools' / 'config.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
        logger.info(f"Saved config to: {path}")

    @property
    def sixs_path(self) -> Optional[str]:
        return self._config['paths']['sixs_executable']

    @property
    def memory_limit_gb(self) -> float:
        return self._config['memory']['limit_gb']

    @property
    def n_cores(self) -> int:
        return self._config['processing']['n_cores']

    @property
    def aerosol_model(self) -> str:
        return self._config['atmospheric']['aerosol_model']

    def __repr__(self):
        return f"Config({self._config})"


# Singleton accessor
def get_config() -> Config:
    """Get the global configuration instance."""
    return Config()
