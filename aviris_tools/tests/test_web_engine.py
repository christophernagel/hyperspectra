"""
Tests for web viewer engine and figure builders.
"""

import numpy as np
import pytest
from pathlib import Path

netCDF4 = pytest.importorskip("netCDF4")
go = pytest.importorskip("plotly.graph_objects")


def _create_nc(path, n_rows=10, n_cols=10, n_bands=50):
    """Create minimal L2 NetCDF."""
    wavelengths = np.linspace(400, 2500, n_bands).astype(np.float32)
    base = np.where(wavelengths < 700, 0.05, 0.45).astype(np.float32)
    rng = np.random.RandomState(42)
    cube = (
        base[np.newaxis, np.newaxis, :]
        + rng.uniform(-0.02, 0.02, (n_rows, n_cols, n_bands)).astype(
            np.float32
        )
    )
    cube = np.clip(cube, 0, 1)
    cube_brc = cube.transpose(2, 0, 1)

    with netCDF4.Dataset(str(path), "w") as ds:
        grp = ds.createGroup("reflectance")
        grp.createDimension("wavelength", n_bands)
        grp.createDimension("y", n_rows)
        grp.createDimension("x", n_cols)
        wl_var = grp.createVariable("wavelength", "f4", ("wavelength",))
        wl_var[:] = wavelengths
        rfl_var = grp.createVariable(
            "reflectance", "f4", ("wavelength", "y", "x")
        )
        rfl_var[:] = cube_brc

    return wavelengths, cube


@pytest.fixture()
def engine_loaded(tmp_path):
    """WebEngine with a loaded synthetic file."""
    from aviris_tools.web.engine import WebEngine

    nc_path = tmp_path / "test.nc"
    wavelengths, cube = _create_nc(nc_path)
    engine = WebEngine()
    engine.load_file(nc_path)
    yield engine, wavelengths, cube
    engine.data_loader.close()


# ── WebEngine.calculate_index ─────────────────────────────────────

class TestCalculateIndex:
    def test_nd_index_shape(self, engine_loaded):
        """Normalized difference index returns (rows, cols) float."""
        engine, _, _ = engine_loaded
        data, idx_def = engine.calculate_index("NDVI")
        assert data.shape == (10, 10)
        assert data.dtype == np.float32

    def test_ratio_index_shape(self, engine_loaded):
        """Ratio index returns correct shape."""
        engine, _, _ = engine_loaded
        data, idx_def = engine.calculate_index("Iron Oxide")
        assert data is not None
        assert data.shape == (10, 10)

    def test_continuum_index_shape(self, engine_loaded):
        """Continuum-removed index returns correct shape."""
        engine, _, _ = engine_loaded
        data, idx_def = engine.calculate_index("Clay (General)")
        assert data is not None
        assert data.shape == (10, 10)

    def test_ndvi_positive_for_vegetation(self, engine_loaded):
        """Vegetation-like spectrum should yield positive NDVI."""
        engine, _, _ = engine_loaded
        data, _ = engine.calculate_index("NDVI")
        assert np.nanmean(data) > 0.3


# ── WebEngine.get_pixel_spectrum ──────────────────────────────────

class TestGetPixelSpectrum:
    def test_returns_correct_pixel(self, engine_loaded):
        """Spectrum at (r, c) matches expected values."""
        engine, wavelengths, cube = engine_loaded
        spec = engine.get_pixel_spectrum(0, 0)
        assert spec is not None
        assert len(spec["values"]) == len(wavelengths)
        np.testing.assert_allclose(
            spec["values"], cube[0, 0, :], atol=1e-4
        )

    def test_out_of_bounds_returns_none(self, engine_loaded):
        """Out-of-bounds pixel returns None."""
        engine, _, _ = engine_loaded
        assert engine.get_pixel_spectrum(100, 100) is None
        assert engine.get_pixel_spectrum(-1, 0) is None

    def test_no_data_returns_none(self):
        """get_pixel_spectrum with no loaded file returns None."""
        from aviris_tools.web.engine import WebEngine

        engine = WebEngine()
        assert engine.get_pixel_spectrum(0, 0) is None


# ── WebEngine.get_rgb_composite ───────────────────────────────────

class TestGetRGBComposite:
    def test_shape_and_dtype(self, engine_loaded):
        """RGB composite is uint8 (rows, cols, 3)."""
        engine, _, _ = engine_loaded
        rgb = engine.get_rgb_composite(640, 550, 470)
        assert rgb.shape == (10, 10, 3)
        assert rgb.dtype == np.uint8

    def test_values_in_range(self, engine_loaded):
        """All RGB values are 0-255."""
        engine, _, _ = engine_loaded
        rgb = engine.get_rgb_composite(640, 550, 470)
        assert rgb.min() >= 0
        assert rgb.max() <= 255


# ── WebEngine.get_roi_spectrum ────────────────────────────────────

class TestGetROISpectrum:
    def test_roi_returns_mean_std(self, engine_loaded):
        """ROI spectrum has mean, std, and pixel count."""
        engine, wavelengths, _ = engine_loaded
        roi = engine.get_roi_spectrum(2, 5, 3, 7)
        assert roi is not None
        assert len(roi["mean_values"]) == len(wavelengths)
        assert len(roi["std_values"]) == len(wavelengths)
        assert roi["pixel_count"] == 3 * 4  # 3 rows × 4 cols

    def test_empty_roi_returns_none(self, engine_loaded):
        """Zero-area ROI returns None."""
        engine, _, _ = engine_loaded
        assert engine.get_roi_spectrum(5, 5, 3, 3) is None  # 0 rows
        assert engine.get_roi_spectrum(5, 3, 5, 5) is None  # 0 rows

    def test_roi_with_mask(self, engine_loaded):
        """ROI with boolean mask selects subset."""
        engine, wavelengths, _ = engine_loaded
        mask = np.zeros((3, 4), dtype=bool)
        mask[0, 0] = True
        mask[1, 2] = True
        roi = engine.get_roi_spectrum(2, 5, 3, 7, mask=mask)
        assert roi is not None
        assert roi["pixel_count"] == 2


# ── WebEngine.robust_percentile ───────────────────────────────────

class TestRobustPercentile:
    def test_returns_two_values(self, engine_loaded):
        """Default returns [low, high] percentiles."""
        engine, _, _ = engine_loaded
        data = np.random.uniform(0, 1, (100, 100)).astype(np.float32)
        p = engine.robust_percentile(data)
        assert len(p) == 2
        assert p[0] < p[1]

    def test_all_nan_returns_default(self, engine_loaded):
        """All-NaN data returns [0, 1]."""
        engine, _, _ = engine_loaded
        data = np.full((10, 10), np.nan, dtype=np.float32)
        p = engine.robust_percentile(data)
        assert p == [0.0, 1.0]


# ── WebEngine.scan_directory ──────────────────────────────────────

class TestScanDirectory:
    def test_finds_nc_files(self, tmp_path):
        """Finds .nc files in a directory."""
        from aviris_tools.web.engine import WebEngine

        (tmp_path / "a.nc").touch()
        (tmp_path / "b.nc").touch()
        (tmp_path / "c.txt").touch()
        files = WebEngine.scan_directory(tmp_path)
        assert len(files) == 2
        assert all(f["title"].endswith(".nc") for f in files)

    def test_nonexistent_dir(self):
        """Non-existent directory returns empty list."""
        from aviris_tools.web.engine import WebEngine

        assert WebEngine.scan_directory("/no/such/dir") == []


# ── Figure builders ───────────────────────────────────────────────

class TestFigureBuilders:
    def test_make_band_figure(self):
        """make_band_figure returns a go.Figure."""
        from aviris_tools.web.figures import make_band_figure

        data = np.random.uniform(0, 1, (10, 10))
        fig = make_band_figure(data, 550.0)
        assert isinstance(fig, go.Figure)

    def test_make_rgb_figure(self):
        """make_rgb_figure returns a go.Figure."""
        from aviris_tools.web.figures import make_rgb_figure

        rgb = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        fig = make_rgb_figure(rgb)
        assert isinstance(fig, go.Figure)

    def test_make_index_figure(self):
        """make_index_figure returns a go.Figure."""
        from aviris_tools.web.figures import make_index_figure

        data = np.random.uniform(-1, 1, (10, 10))
        fig = make_index_figure(data, "NDVI", "RdYlGn", [-1, 1])
        assert isinstance(fig, go.Figure)

    def test_make_spectral_figure(self):
        """make_spectral_figure returns a go.Figure."""
        from aviris_tools.web.figures import make_spectral_figure

        spectra = [
            {
                "wavelengths": list(range(400, 900, 10)),
                "values": [0.3] * 50,
                "label": "test",
                "color": "#FF0000",
            }
        ]
        fig = make_spectral_figure(spectra)
        assert isinstance(fig, go.Figure)

    def test_make_empty_figure(self):
        """make_empty_figure returns a go.Figure."""
        from aviris_tools.web.figures import make_empty_figure

        fig = make_empty_figure("test message")
        assert isinstance(fig, go.Figure)

    def test_band_figure_with_geo(self):
        """Band figure with lat/lon axes."""
        from aviris_tools.web.figures import make_band_figure

        data = np.random.uniform(0, 1, (10, 10))
        lat = np.linspace(34.0, 34.1, 10)
        lon = np.linspace(-118.0, -117.9, 10)
        fig = make_band_figure(data, 550.0, lat_axis=lat, lon_axis=lon)
        assert isinstance(fig, go.Figure)

    def test_spectral_figure_with_roi(self):
        """Spectral figure with ROI envelope."""
        from aviris_tools.web.figures import make_spectral_figure

        wl = list(range(400, 900, 10))
        roi = [
            {
                "wavelengths": wl,
                "mean_values": [0.3] * len(wl),
                "std_values": [0.05] * len(wl),
                "label": "ROI",
                "color": "#00FF00",
            }
        ]
        fig = make_spectral_figure([], roi_spectra=roi)
        assert isinstance(fig, go.Figure)
        # Should have 3 traces: upper, lower, mean
        assert len(fig.data) >= 3


# ── Edge cases ────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_band_file(self, tmp_path):
        """Engine handles a single-band file."""
        from aviris_tools.web.engine import WebEngine

        nc_path = tmp_path / "single.nc"
        _create_nc(nc_path, n_bands=1)
        engine = WebEngine()
        info = engine.load_file(nc_path)
        assert info["n_bands"] == 1
        spec = engine.get_pixel_spectrum(0, 0)
        assert spec is not None
        assert len(spec["values"]) == 1
        engine.data_loader.close()
