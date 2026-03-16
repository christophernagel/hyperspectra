"""
Tests for Py6S atmospheric correction processor.
"""

import types
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from aviris_tools.atm_correction.py6s_processor import (
    Py6SProcessor,
    _import_legacy_module,
    _LEGACY_MODULE,
    process_file,
)


# ── _import_legacy_module ──────────────────────────────────────────

class TestImportLegacyModule:
    def test_raises_when_missing(self, tmp_path):
        """ImportError when legacy file does not exist."""
        with patch(
            "aviris_tools.atm_correction.py6s_processor._LEGACY_MODULE",
            tmp_path / "nonexistent.py",
        ):
            with pytest.raises(ImportError, match="Legacy module not found"):
                _import_legacy_module()

    def test_imports_valid_module(self, tmp_path):
        """Successfully imports a valid Python module."""
        mod_path = tmp_path / "aviris_atm_correction_v2.py"
        mod_path.write_text("MEMORY_LIMIT_GB = 4\nHAS_PY6S = False\n")

        with patch(
            "aviris_tools.atm_correction.py6s_processor._LEGACY_MODULE",
            mod_path,
        ):
            mod = _import_legacy_module()
            assert hasattr(mod, "MEMORY_LIMIT_GB")
            assert mod.MEMORY_LIMIT_GB == 4


# ── Py6SProcessor.__init__ ─────────────────────────────────────────

class TestPy6SProcessorInit:
    def test_raises_missing_radiance(self, tmp_path):
        """FileNotFoundError when radiance file missing."""
        obs = tmp_path / "obs.nc"
        obs.touch()
        with pytest.raises(FileNotFoundError, match="Radiance file not found"):
            Py6SProcessor(
                radiance_path=str(tmp_path / "missing.nc"),
                obs_path=str(obs),
                output_path=str(tmp_path / "out.nc"),
            )

    def test_raises_missing_obs(self, tmp_path):
        """FileNotFoundError when observation file missing."""
        rad = tmp_path / "rad.nc"
        rad.touch()
        with pytest.raises(FileNotFoundError, match="Observation file not found"):
            Py6SProcessor(
                radiance_path=str(rad),
                obs_path=str(tmp_path / "missing.nc"),
                output_path=str(tmp_path / "out.nc"),
            )

    def test_init_stores_paths(self, tmp_path):
        """Paths are stored correctly on successful init."""
        rad = tmp_path / "rad.nc"
        obs = tmp_path / "obs.nc"
        rad.touch()
        obs.touch()
        p = Py6SProcessor(
            radiance_path=str(rad),
            obs_path=str(obs),
            output_path=str(tmp_path / "out.nc"),
        )
        assert p.radiance_path == rad
        assert p.obs_path == obs
        assert p.output_path == tmp_path / "out.nc"


# ── _patch_legacy_config ───────────────────────────────────────────

class TestPatchLegacyConfig:
    def test_patches_memory_limit(self, tmp_path):
        """Memory limit is patched onto legacy module."""
        rad = tmp_path / "rad.nc"
        obs = tmp_path / "obs.nc"
        rad.touch()
        obs.touch()
        proc = Py6SProcessor(
            radiance_path=str(rad),
            obs_path=str(obs),
            output_path=str(tmp_path / "out.nc"),
        )
        legacy = types.ModuleType("fake_legacy")
        legacy.MEMORY_LIMIT_GB = 0
        legacy.SIXS_PATH = None
        legacy.HAS_PY6S = False

        proc._patch_legacy_config(legacy)

        assert legacy.MEMORY_LIMIT_GB == proc.memory_manager.limit_gb

    def test_patches_sixs_path(self, tmp_path):
        """6S path is patched when configured."""
        rad = tmp_path / "rad.nc"
        obs = tmp_path / "obs.nc"
        rad.touch()
        obs.touch()
        proc = Py6SProcessor(
            radiance_path=str(rad),
            obs_path=str(obs),
            output_path=str(tmp_path / "out.nc"),
        )
        legacy = types.ModuleType("fake_legacy")
        legacy.MEMORY_LIMIT_GB = 0
        legacy.SIXS_PATH = None
        legacy.HAS_PY6S = False

        with patch(
            "aviris_tools.atm_correction.py6s_processor.get_config"
        ) as mock_cfg:
            mock_cfg.return_value.get.side_effect = lambda *a, **kw: (
                "/usr/bin/sixs" if a == ("paths", "sixs_executable") else None
            )
            proc._patch_legacy_config(legacy)

        assert legacy.SIXS_PATH == "/usr/bin/sixs"
        assert legacy.HAS_PY6S is True


# ── _validate_output ───────────────────────────────────────────────

class TestValidateOutput:
    @pytest.fixture()
    def _make_nc(self, tmp_path):
        """Helper to create a minimal NetCDF with reflectance."""
        import netCDF4 as nc

        def _create(values):
            out = tmp_path / "rfl.nc"
            with nc.Dataset(str(out), "w") as ds:
                grp = ds.createGroup("reflectance")
                grp.createDimension("x", len(values))
                var = grp.createVariable("reflectance", "f4", ("x",))
                var[:] = values
            return out

        return _create

    def test_warns_negative(self, tmp_path, _make_nc, caplog):
        """Warning when reflectance has significant negatives."""
        out = _make_nc([-0.1, 0.3, 0.5])
        rad = tmp_path / "rad.nc"
        obs = tmp_path / "obs.nc"
        rad.touch()
        obs.touch()
        proc = Py6SProcessor(
            radiance_path=str(rad),
            obs_path=str(obs),
            output_path=str(out),
        )
        import logging
        with caplog.at_level(logging.WARNING):
            proc._validate_output()
        assert "negative reflectance" in caplog.text.lower()

    def test_warns_above_one(self, tmp_path, _make_nc, caplog):
        """Warning when reflectance > 1.0."""
        out = _make_nc([0.3, 0.5, 1.5])
        rad = tmp_path / "rad.nc"
        obs = tmp_path / "obs.nc"
        rad.touch()
        obs.touch()
        proc = Py6SProcessor(
            radiance_path=str(rad),
            obs_path=str(obs),
            output_path=str(out),
        )
        import logging
        with caplog.at_level(logging.WARNING):
            proc._validate_output()
        assert "reflectance > 1.0" in caplog.text.lower()

    def test_warns_low_mean(self, tmp_path, _make_nc, caplog):
        """Warning when mean reflectance is very low."""
        out = _make_nc([0.001, 0.002, 0.003])
        rad = tmp_path / "rad.nc"
        obs = tmp_path / "obs.nc"
        rad.touch()
        obs.touch()
        proc = Py6SProcessor(
            radiance_path=str(rad),
            obs_path=str(obs),
            output_path=str(out),
        )
        import logging
        with caplog.at_level(logging.WARNING):
            proc._validate_output()
        assert "very low mean" in caplog.text.lower()


# ── process_file convenience ───────────────────────────────────────

class TestProcessFile:
    def test_wires_to_processor(self, tmp_path):
        """process_file creates Py6SProcessor and calls run()."""
        rad = tmp_path / "rad.nc"
        obs = tmp_path / "obs.nc"
        rad.touch()
        obs.touch()
        with patch.object(Py6SProcessor, "run", return_value=tmp_path / "out.nc") as mock_run:
            result = process_file(
                str(rad), str(obs), str(tmp_path / "out.nc"),
                aerosol="maritime", validate=False,
            )
            mock_run.assert_called_once()
            assert result == tmp_path / "out.nc"
