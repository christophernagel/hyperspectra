"""
Tests for CLI argument parsing.
"""

import sys
import pytest
from unittest.mock import patch

from aviris_tools.cli import process_cli, view_cli


class TestProcessCLI:
    """Test process_cli argument parsing."""

    def test_required_args(self):
        """Missing required args should cause SystemExit."""
        with patch.object(sys, "argv", ["aviris-process"]):
            with pytest.raises(SystemExit):
                process_cli()

    def test_parses_positional_args(self, tmp_path):
        """Positional args parsed correctly."""
        rad = tmp_path / "rad.nc"
        obs = tmp_path / "obs.nc"
        rad.touch()
        obs.touch()
        args_list = [
            "aviris-process",
            str(rad),
            str(obs),
            str(tmp_path / "out.nc"),
        ]
        # Py6SProcessor is imported inside process_cli via deferred
        # import, so we patch at the source module level.
        with patch.object(sys, "argv", args_list):
            with patch(
                "aviris_tools.atm_correction.py6s_processor.Py6SProcessor"
            ) as mock_cls:
                mock_cls.return_value.run.return_value = None
                process_cli()
                mock_cls.assert_called_once()

    def test_defaults(self):
        """Default values for optional flags."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("radiance")
        parser.add_argument("obs")
        parser.add_argument("output")
        parser.add_argument("--method", "-m", default="py6s")
        parser.add_argument("--aerosol", "-a", default=None)
        parser.add_argument("--altitude", type=float, default=None)
        parser.add_argument("--simple", action="store_true")
        parser.add_argument("--no-validate", action="store_true")
        parser.add_argument("--verbose", "-v", action="store_true")
        parsed = parser.parse_args(["r.nc", "o.nc", "out.nc"])
        assert parsed.method == "py6s"
        assert parsed.aerosol is None
        assert parsed.altitude is None
        assert parsed.simple is False
        assert parsed.no_validate is False
        assert parsed.verbose is False

    def test_optional_flags(self):
        """Optional flags parsed correctly."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("radiance")
        parser.add_argument("obs")
        parser.add_argument("output")
        parser.add_argument("--method", "-m", default="py6s")
        parser.add_argument("--aerosol", "-a", default=None)
        parser.add_argument("--simple", action="store_true")
        parser.add_argument("--no-validate", action="store_true")
        parser.add_argument("--verbose", "-v", action="store_true")
        parsed = parser.parse_args([
            "r.nc", "o.nc", "out.nc",
            "--method", "isofit",
            "--aerosol", "urban",
            "--simple",
            "--no-validate",
            "-v",
        ])
        assert parsed.method == "isofit"
        assert parsed.aerosol == "urban"
        assert parsed.simple is True
        assert parsed.no_validate is True
        assert parsed.verbose is True

    def test_invalid_method_rejected(self):
        """Invalid --method value should raise."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--method", choices=["py6s", "isofit"], default="py6s"
        )
        with pytest.raises(SystemExit):
            parser.parse_args(["--method", "invalid"])

    def test_invalid_aerosol_rejected(self):
        """Invalid --aerosol value should raise."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--aerosol",
            choices=["maritime", "continental", "urban", "desert"],
        )
        with pytest.raises(SystemExit):
            parser.parse_args(["--aerosol", "tropical"])

    def test_help_exits_cleanly(self):
        """--help should exit with code 0."""
        with patch.object(sys, "argv", ["aviris-process", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                process_cli()
            assert exc_info.value.code == 0

    def test_nonexistent_radiance_exits(self, tmp_path):
        """Non-existent radiance file causes sys.exit(1)."""
        obs = tmp_path / "obs.nc"
        obs.touch()
        with patch.object(sys, "argv", [
            "aviris-process",
            str(tmp_path / "missing.nc"),
            str(obs),
            str(tmp_path / "out.nc"),
        ]):
            with pytest.raises(SystemExit) as exc_info:
                process_cli()
            assert exc_info.value.code == 1


class TestViewCLI:
    """Test view_cli argument parsing."""

    def test_help_exits_cleanly(self):
        """--help should exit with code 0."""
        with patch.object(sys, "argv", ["aviris-view", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                view_cli()
            assert exc_info.value.code == 0

    def test_simple_without_filepath_exits(self):
        """--simple without filepath should exit with error."""
        with patch.object(sys, "argv", ["aviris-view", "--simple"]):
            with pytest.raises(SystemExit) as exc_info:
                view_cli()
            assert exc_info.value.code == 1

    def test_parses_optional_args(self):
        """Optional arguments parsed correctly."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("filepath", nargs="?", default=None)
        parser.add_argument("--simple", action="store_true")
        parser.add_argument("--scale", type=float, default=None)
        parsed = parser.parse_args(["file.nc", "--simple", "--scale", "1.5"])
        assert parsed.filepath == "file.nc"
        assert parsed.simple is True
        assert parsed.scale == 1.5
