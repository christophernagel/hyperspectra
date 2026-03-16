"""
Tests for USGS spectral library parser.
"""

import json
import pytest
import numpy as np
from pathlib import Path

from aviris_tools.reference_spectra.usgs_parser import (
    USGSSpectrum,
    USGSLibraryParser,
    load_resampled_library,
    load_wavelengths_file,
    match_spectrum,
    parse_usgs_ascii,
    WAVELENGTHS_D1,
    WAVELENGTHS_D2,
)


# ── USGSSpectrum ───────────────────────────────────────────────────

class TestUSGSSpectrum:
    @pytest.fixture()
    def flat_spectrum(self):
        """Flat 0.5 reflectance across ASD range."""
        wl_um = np.linspace(0.35, 2.5, 2151)
        refl = np.full(2151, 0.5)
        return USGSSpectrum("flat", wl_um, refl)

    def test_wavelength_conversion(self, flat_spectrum):
        """wavelengths_nm should be wavelengths_um * 1000."""
        np.testing.assert_allclose(
            flat_spectrum.wavelengths_nm,
            flat_spectrum.wavelengths_um * 1000,
        )

    def test_valid_mask_all_good(self, flat_spectrum):
        """All values > -1e30 should be valid."""
        assert flat_spectrum.valid_mask.all()

    def test_valid_mask_bad_bands(self):
        """USGS sentinel -1.23e34 should be masked."""
        wl_um = np.linspace(0.35, 2.5, 10)
        refl = np.full(10, 0.5)
        refl[3] = -1.23e34
        refl[7] = -1.23e34
        spec = USGSSpectrum("bad", wl_um, refl)
        assert spec.valid_mask.sum() == 8
        assert not spec.valid_mask[3]
        assert not spec.valid_mask[7]

    def test_resample_preserves_flat(self, flat_spectrum):
        """Resampling a flat spectrum should stay flat."""
        target = np.linspace(400, 2400, 100)
        resampled = flat_spectrum.resample(target)
        valid = ~np.isnan(resampled)
        np.testing.assert_allclose(resampled[valid], 0.5, atol=1e-6)

    def test_resample_nan_outside_range(self, flat_spectrum):
        """Values outside spectrum range should be NaN."""
        target = np.array([100.0, 500.0, 3000.0])
        resampled = flat_spectrum.resample(target)
        assert np.isnan(resampled[0])
        assert np.isfinite(resampled[1])
        assert np.isnan(resampled[2])

    def test_to_d1_shape(self, flat_spectrum):
        """to_d1 returns array matching WAVELENGTHS_D1 length."""
        d1 = flat_spectrum.to_d1()
        assert len(d1) == len(WAVELENGTHS_D1)

    def test_to_d2_shape(self, flat_spectrum):
        """to_d2 returns array matching WAVELENGTHS_D2 length."""
        d2 = flat_spectrum.to_d2()
        assert len(d2) == len(WAVELENGTHS_D2)

    def test_get_coverage(self, flat_spectrum):
        """Coverage should span the valid wavelength range in nm."""
        lo, hi = flat_spectrum.get_coverage()
        assert lo == pytest.approx(350.0, abs=1)
        assert hi == pytest.approx(2500.0, abs=1)

    def test_get_coverage_all_bad(self):
        """Coverage returns NaN for all-bad spectrum."""
        wl_um = np.linspace(0.35, 2.5, 5)
        refl = np.full(5, -1.23e34)
        spec = USGSSpectrum("allbad", wl_um, refl)
        lo, hi = spec.get_coverage()
        assert np.isnan(lo)
        assert np.isnan(hi)


# ── load_wavelengths_file ──────────────────────────────────────────

class TestLoadWavelengthsFile:
    def test_reads_header_and_values(self, tmp_path):
        """Skips header line, reads float values."""
        wl_file = tmp_path / "wavelengths.txt"
        lines = ["ASD wavelengths header\n"]
        lines += [f"{0.35 + i * 0.001:.6f}\n" for i in range(100)]
        wl_file.write_text("".join(lines))
        wl = load_wavelengths_file(wl_file)
        assert len(wl) == 100
        assert wl[0] == pytest.approx(0.35, abs=1e-4)


# ── parse_usgs_ascii ──────────────────────────────────────────────

class TestParseUSGSAscii:
    def test_parses_valid_file(self, tmp_path):
        """Parses a synthetic USGS ASCII file."""
        wl_um = np.linspace(0.35, 2.5, 50)
        refl_values = np.linspace(0.1, 0.6, 50)

        f = tmp_path / "splib07a_TestMaterial_ASDFRa_AREF.txt"
        lines = ["splib07a Record=1: TestMaterial GDS001\n"]
        lines += [f"{v:.6f}\n" for v in refl_values]
        f.write_text("".join(lines))

        spec = parse_usgs_ascii(f, wl_um)
        assert spec is not None
        assert len(spec.reflectance) == 50
        assert "TestMaterial" in spec.name or "Test" in spec.name

    def test_too_few_points_returns_none(self, tmp_path):
        """Files with < 10 data points return None."""
        wl_um = np.linspace(0.35, 2.5, 5)
        f = tmp_path / "short.txt"
        lines = ["header\n"] + ["0.5\n"] * 5
        f.write_text("".join(lines))
        assert parse_usgs_ascii(f, wl_um) is None

    def test_length_mismatch_truncates(self, tmp_path):
        """Small length mismatch is tolerated by truncation."""
        wl_um = np.linspace(0.35, 2.5, 50)
        f = tmp_path / "mismatch.txt"
        lines = ["header\n"] + ["0.3\n"] * 55  # 5 extra
        f.write_text("".join(lines))
        spec = parse_usgs_ascii(f, wl_um)
        assert spec is not None
        assert len(spec.reflectance) == 50

    def test_large_mismatch_returns_none(self, tmp_path):
        """Large length mismatch returns None."""
        wl_um = np.linspace(0.35, 2.5, 50)
        f = tmp_path / "big_mismatch.txt"
        lines = ["header\n"] + ["0.3\n"] * 100
        f.write_text("".join(lines))
        assert parse_usgs_ascii(f, wl_um) is None


# ── match_spectrum ─────────────────────────────────────────────────

class TestMatchSpectrum:
    @pytest.fixture()
    def library(self):
        """Minimal library with known spectra."""
        wl = np.linspace(400, 1700, 131)
        return {
            "wavelengths": wl,
            "spectra": {
                "flat_high": {
                    "reflectance": np.full(131, 0.8),
                    "category": "test",
                },
                "flat_low": {
                    "reflectance": np.full(131, 0.2),
                    "category": "test",
                },
                "slope_up": {
                    "reflectance": np.linspace(0.1, 0.9, 131),
                    "category": "test",
                },
            },
        }

    def test_exact_match_sam(self, library):
        """Identical spectrum → SAM angle ≈ 0."""
        unknown = np.full(131, 0.8)
        matches = match_spectrum(unknown, library, top_n=3, method="sam")
        best_name, best_score = matches[0]
        assert best_name == "flat_high"
        assert best_score < 0.01  # Near zero angle

    def test_exact_match_correlation(self, library):
        """Identical spectrum → correlation ≈ 1."""
        unknown = np.linspace(0.1, 0.9, 131)
        matches = match_spectrum(
            unknown, library, top_n=3, method="correlation"
        )
        best_name, best_score = matches[0]
        assert best_name == "slope_up"
        assert best_score > 0.99

    def test_top_n_respected(self, library):
        """Returns at most top_n results."""
        unknown = np.full(131, 0.5)
        matches = match_spectrum(unknown, library, top_n=2)
        assert len(matches) == 2

    def test_rejects_mismatched_length(self, library):
        """Wrong-length spectrum should raise or produce bad results."""
        unknown = np.full(50, 0.5)
        # np.dot will fail or produce incorrect results
        # depending on implementation. Match still runs
        # but results are meaningless — we just verify no crash.
        # The library has 131-channel spectra vs 50-channel unknown.
        # numpy will raise ValueError on dot product mismatch.
        with pytest.raises((ValueError, IndexError)):
            match_spectrum(unknown, library, method="sam")


# ── load_resampled_library ─────────────────────────────────────────

class TestLoadResampledLibrary:
    def test_round_trip(self, tmp_path):
        """Write JSON, load it, verify structure."""
        wl = np.linspace(400, 1700, 131).tolist()
        data = {
            "metadata": {
                "wavelengths_nm": wl,
                "n_channels": 131,
                "source": "test",
            },
            "spectra": {
                "material_a": {
                    "reflectance": [0.5] * 131,
                    "category": "test",
                },
            },
        }
        json_path = tmp_path / "lib.json"
        json_path.write_text(json.dumps(data))
        lib = load_resampled_library(str(json_path))

        assert len(lib["wavelengths"]) == 131
        assert "material_a" in lib["spectra"]
        refl = lib["spectra"]["material_a"]["reflectance"]
        np.testing.assert_allclose(refl, 0.5)

    def test_negative_values_masked(self, tmp_path):
        """Negative reflectance values (sentinel) become NaN."""
        data = {
            "metadata": {"wavelengths_nm": [400, 500, 600]},
            "spectra": {
                "m": {"reflectance": [0.5, -1.0, 0.3], "category": "t"},
            },
        }
        json_path = tmp_path / "lib.json"
        json_path.write_text(json.dumps(data))
        lib = load_resampled_library(str(json_path))
        refl = lib["spectra"]["m"]["reflectance"]
        assert np.isnan(refl[1])
        assert refl[0] == pytest.approx(0.5)


# ── USGSLibraryParser ─────────────────────────────────────────────

class TestUSGSLibraryParser:
    def test_fallback_wavelengths(self, tmp_path):
        """Uses default wavelengths when file not found."""
        parser = USGSLibraryParser(str(tmp_path))
        assert len(parser.asd_wavelengths) == 2151

    def test_categorize_material(self, tmp_path):
        """Material categorization matches keywords."""
        parser = USGSLibraryParser(str(tmp_path))
        assert parser._categorize_material("polyethylene film") == "plastic"
        assert parser._categorize_material("cotton fabric") == "fabric"
        assert parser._categorize_material("concrete block") == "construction"
        assert parser._categorize_material("unknown xyz") is None

    def test_get_summary_empty(self, tmp_path):
        """Summary on empty parser."""
        parser = USGSLibraryParser(str(tmp_path))
        summary = parser.get_summary()
        assert summary["total_spectra"] == 0
