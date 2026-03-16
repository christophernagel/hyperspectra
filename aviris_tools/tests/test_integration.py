"""
Integration tests for the L1B → L2 → viewer pipeline.

Uses synthetic NetCDF data to verify end-to-end data flow.
"""

import numpy as np
import pytest
from pathlib import Path

netCDF4 = pytest.importorskip("netCDF4")


def _create_synthetic_nc(path, n_rows=10, n_cols=10, n_bands=50):
    """Create a minimal NetCDF mimicking AVIRIS-3 L2 reflectance."""
    wavelengths = np.linspace(400, 2500, n_bands).astype(np.float32)
    # Vegetation-like spectrum: low visible, high NIR
    base_spectrum = np.where(
        wavelengths < 700, 0.05, 0.45
    ).astype(np.float32)
    # Spatially: each pixel is base_spectrum + small variation
    rng = np.random.RandomState(42)
    cube = (
        base_spectrum[np.newaxis, np.newaxis, :]
        + rng.uniform(-0.02, 0.02, (n_rows, n_cols, n_bands)).astype(
            np.float32
        )
    )
    cube = np.clip(cube, 0, 1)
    # NetCDF expects (bands, rows, cols) for the data_loader
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
def synthetic_nc(tmp_path):
    """Fixture providing a synthetic L2 NetCDF file."""
    nc_path = tmp_path / "synthetic_l2.nc"
    wavelengths, cube = _create_synthetic_nc(nc_path)
    return nc_path, wavelengths, cube


# ── Data loader auto-detection ────────────────────────────────────

class TestDataLoaderAutoDetect:
    def test_nc_extension_loads(self, synthetic_nc):
        """NetCDF extension triggers LazyHyperspectralData."""
        from aviris_tools.viewer.data_loader import load_hyperspectral

        nc_path, wavelengths, _ = synthetic_nc
        loader = load_hyperspectral(nc_path)
        assert loader.n_bands == len(wavelengths)
        assert loader.n_rows == 10
        assert loader.n_cols == 10
        assert loader.data_type == "reflectance"
        loader.close()

    def test_unsupported_extension_raises(self, tmp_path):
        """Unsupported extension raises ValueError."""
        from aviris_tools.viewer.data_loader import load_hyperspectral

        bad = tmp_path / "file.tif"
        bad.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_hyperspectral(bad)


# ── LazyHyperspectralData → index calculation ─────────────────────

class TestLoaderToIndex:
    def test_band_retrieval(self, synthetic_nc):
        """get_interpolated_band returns correct shape."""
        from aviris_tools.viewer.data_loader import load_hyperspectral

        nc_path, _, _ = synthetic_nc
        loader = load_hyperspectral(nc_path)
        band = loader.get_interpolated_band(860.0)
        assert band.shape == (10, 10)
        assert np.all(np.isfinite(band))
        loader.close()

    def test_pixel_spectrum_extraction(self, synthetic_nc):
        """get_cube_region returns full spectrum at a pixel."""
        from aviris_tools.viewer.data_loader import load_hyperspectral

        nc_path, wavelengths, cube = synthetic_nc
        loader = load_hyperspectral(nc_path)
        region = loader.get_cube_region(slice(5, 6), slice(3, 4))
        spectrum = region[:, 0, 0]
        assert len(spectrum) == len(wavelengths)
        # Values should be close to the original cube
        np.testing.assert_allclose(
            spectrum, cube[5, 3, :], atol=1e-5
        )
        loader.close()

    def test_ndvi_from_loader(self, synthetic_nc):
        """NDVI computed from loaded bands has correct shape and range."""
        from aviris_tools.viewer.data_loader import load_hyperspectral

        nc_path, _, _ = synthetic_nc
        loader = load_hyperspectral(nc_path)
        nir = loader.get_interpolated_band(860.0).astype(np.float64)
        red = loader.get_interpolated_band(650.0).astype(np.float64)
        denom = nir + red
        ndvi = np.divide(
            nir - red, denom,
            out=np.zeros_like(nir), where=denom != 0,
        )
        assert ndvi.shape == (10, 10)
        # Vegetation-like spectrum: high NIR, low red → positive NDVI
        assert np.mean(ndvi) > 0.3
        loader.close()


# ── WebEngine round-trip ──────────────────────────────────────────

class TestWebEngineRoundTrip:
    def test_load_and_info(self, synthetic_nc):
        """WebEngine.load_file returns correct metadata."""
        from aviris_tools.web.engine import WebEngine

        nc_path, wavelengths, _ = synthetic_nc
        engine = WebEngine()
        info = engine.load_file(nc_path)
        assert info["n_bands"] == len(wavelengths)
        assert info["n_rows"] == 10
        assert info["n_cols"] == 10
        assert info["data_type"] == "reflectance"
        engine.data_loader.close()

    def test_calculate_index(self, synthetic_nc):
        """calculate_index returns correct shape for NDVI."""
        from aviris_tools.web.engine import WebEngine

        nc_path, _, _ = synthetic_nc
        engine = WebEngine()
        engine.load_file(nc_path)
        data, idx_def = engine.calculate_index("NDVI")
        assert data is not None
        assert data.shape == (10, 10)
        assert idx_def["type"] == "nd"
        engine.data_loader.close()

    def test_get_pixel_spectrum(self, synthetic_nc):
        """get_pixel_spectrum returns wavelengths and values."""
        from aviris_tools.web.engine import WebEngine

        nc_path, wavelengths, cube = synthetic_nc
        engine = WebEngine()
        engine.load_file(nc_path)
        spec = engine.get_pixel_spectrum(5, 3)
        assert spec is not None
        assert len(spec["wavelengths"]) == len(wavelengths)
        assert len(spec["values"]) == len(wavelengths)
        # Values should match the cube
        np.testing.assert_allclose(
            spec["values"], cube[5, 3, :], atol=1e-4
        )
        engine.data_loader.close()

    def test_get_rgb_composite(self, synthetic_nc):
        """get_rgb_composite returns uint8 (rows, cols, 3)."""
        from aviris_tools.web.engine import WebEngine

        nc_path, _, _ = synthetic_nc
        engine = WebEngine()
        engine.load_file(nc_path)
        rgb = engine.get_rgb_composite(640, 550, 470)
        assert rgb.shape == (10, 10, 3)
        assert rgb.dtype == np.uint8
        engine.data_loader.close()

    def test_unknown_index_returns_none(self, synthetic_nc):
        """Unknown index name returns (None, None)."""
        from aviris_tools.web.engine import WebEngine

        nc_path, _, _ = synthetic_nc
        engine = WebEngine()
        engine.load_file(nc_path)
        data, idx_def = engine.calculate_index("DOES_NOT_EXIST")
        assert data is None
        assert idx_def is None
        engine.data_loader.close()
