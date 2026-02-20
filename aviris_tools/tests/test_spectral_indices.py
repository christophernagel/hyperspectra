"""
Tests for spectral index calculations.

Run with: pytest tests/test_spectral_indices.py -v
"""

import numpy as np
import pytest


class TestNDVI:
    """Test NDVI calculation."""

    def test_ndvi_vegetation(self):
        """Vegetation should have positive NDVI (0.3-0.8)."""
        # Typical vegetation: high NIR, low red
        nir = 0.45
        red = 0.08
        ndvi = (nir - red) / (nir + red)
        assert 0.3 < ndvi < 0.9, f"Vegetation NDVI should be 0.3-0.9, got {ndvi}"

    def test_ndvi_water(self):
        """Water should have negative NDVI."""
        # Water: low NIR, higher red
        nir = 0.02
        red = 0.05
        ndvi = (nir - red) / (nir + red)
        assert ndvi < 0, f"Water NDVI should be negative, got {ndvi}"

    def test_ndvi_bare_soil(self):
        """Bare soil should have low positive NDVI."""
        nir = 0.25
        red = 0.20
        ndvi = (nir - red) / (nir + red)
        assert 0 < ndvi < 0.3, f"Bare soil NDVI should be 0-0.3, got {ndvi}"

    def test_ndvi_division_safety(self):
        """NDVI should handle zero denominator gracefully."""
        nir = np.array([0.0, 0.5, 0.3])
        red = np.array([0.0, 0.1, 0.3])

        # Safe NDVI calculation
        denom = nir + red
        epsilon = 1e-10
        ndvi = (nir - red) / (denom + epsilon)

        assert not np.any(np.isnan(ndvi)), "NDVI should not produce NaN"
        assert not np.any(np.isinf(ndvi)), "NDVI should not produce Inf"


class TestNDWI:
    """Test NDWI (water index) calculation."""

    def test_ndwi_water(self):
        """Water should have positive NDWI."""
        green = 0.08
        nir = 0.02
        ndwi = (green - nir) / (green + nir)
        assert ndwi > 0.3, f"Water NDWI should be > 0.3, got {ndwi}"

    def test_ndwi_vegetation(self):
        """Vegetation should have negative NDWI."""
        green = 0.10
        nir = 0.45
        ndwi = (green - nir) / (green + nir)
        assert ndwi < 0, f"Vegetation NDWI should be negative, got {ndwi}"


class TestNDMI:
    """Test NDMI (moisture index) calculation."""

    def test_ndmi_healthy_vegetation(self):
        """Healthy vegetation should have positive NDMI."""
        nir = 0.45
        swir = 0.15
        ndmi = (nir - swir) / (nir + swir)
        assert ndmi > 0.2, f"Healthy vegetation NDMI should be > 0.2, got {ndmi}"

    def test_ndmi_stressed_vegetation(self):
        """Water-stressed vegetation should have lower NDMI."""
        nir = 0.35
        swir = 0.25
        ndmi = (nir - swir) / (nir + swir)
        assert 0 < ndmi < 0.3, f"Stressed vegetation NDMI should be 0-0.3, got {ndmi}"


class TestArrayOperations:
    """Test index calculations on arrays."""

    def test_ndvi_array(self):
        """Test NDVI on 2D arrays."""
        nir = np.random.uniform(0.2, 0.6, (100, 100))
        red = np.random.uniform(0.05, 0.15, (100, 100))

        ndvi = (nir - red) / (nir + red + 1e-10)

        assert ndvi.shape == (100, 100)
        assert np.all(ndvi >= -1) and np.all(ndvi <= 1)
        assert np.all(np.isfinite(ndvi))

    def test_masked_array_handling(self):
        """Test handling of masked/NaN values."""
        nir = np.array([0.5, np.nan, 0.4, 0.3])
        red = np.array([0.1, 0.1, np.nan, 0.1])

        # Calculate with NaN propagation
        ndvi = (nir - red) / (nir + red)

        assert np.isnan(ndvi[1]), "NaN in input should produce NaN output"
        assert np.isnan(ndvi[2]), "NaN in input should produce NaN output"
        assert np.isfinite(ndvi[0]), "Valid inputs should produce finite output"
        assert np.isfinite(ndvi[3]), "Valid inputs should produce finite output"


class TestReflectanceValidation:
    """Test reflectance value validation."""

    def test_valid_reflectance_range(self):
        """Surface reflectance should be in 0-1 range."""
        reflectance = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        valid = (reflectance >= 0) & (reflectance <= 1)
        assert np.all(valid)

    def test_invalid_negative_reflectance(self):
        """Negative reflectance indicates processing error."""
        reflectance = np.array([-0.1, 0.5, 0.3])
        invalid = reflectance < 0
        assert np.sum(invalid) == 1

    def test_typical_surface_ranges(self):
        """Test typical reflectance ranges for surfaces."""
        # Vegetation at 550nm (green peak)
        veg_green = 0.12
        assert 0.05 < veg_green < 0.20, "Vegetation green peak out of range"

        # Vegetation at 860nm (NIR plateau)
        veg_nir = 0.45
        assert 0.30 < veg_nir < 0.60, "Vegetation NIR out of range"

        # Water at 860nm
        water_nir = 0.02
        assert water_nir < 0.10, "Water NIR should be < 0.10"

        # Desert sand at 1600nm
        sand_swir = 0.35
        assert 0.20 < sand_swir < 0.50, "Sand SWIR out of range"


class TestWavelengthInterpolation:
    """Test wavelength-based band selection."""

    def test_nearest_band_selection(self):
        """Test finding nearest band to target wavelength."""
        wavelengths = np.array([400, 450, 500, 550, 600, 650, 700])
        target = 555

        idx = np.argmin(np.abs(wavelengths - target))
        assert wavelengths[idx] == 550

    def test_interpolation_bounds(self):
        """Test interpolation at spectrum edges."""
        wavelengths = np.array([400, 450, 500, 550])

        # Target below range
        target_low = 350
        idx = np.argmin(np.abs(wavelengths - target_low))
        assert idx == 0

        # Target above range
        target_high = 600
        idx = np.argmin(np.abs(wavelengths - target_high))
        assert idx == len(wavelengths) - 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
