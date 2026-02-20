"""
Tests for utility modules.

Run with: pytest tests/test_utils.py -v
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path


class TestMemoryUtils:
    """Test memory management utilities."""

    def test_available_memory_positive(self):
        """Available memory should be positive."""
        from aviris_tools.utils.memory import get_available_memory
        mem = get_available_memory()
        assert mem > 0, "Available memory should be positive"

    def test_total_memory_positive(self):
        """Total memory should be positive."""
        from aviris_tools.utils.memory import get_total_memory
        mem = get_total_memory()
        assert mem > 0, "Total memory should be positive"

    def test_array_memory_estimate(self):
        """Test array memory estimation."""
        from aviris_tools.utils.memory import estimate_array_memory

        # 1000x1000 float32 = 4MB
        shape = (1000, 1000)
        mem_gb = estimate_array_memory(shape, np.float32)
        expected_gb = 4 / 1024  # 4MB in GB
        assert abs(mem_gb - expected_gb) < 0.001

        # 1000x1000x100 float32 = ~381MB (1000*1000*100*4 bytes)
        shape = (1000, 1000, 100)
        mem_gb = estimate_array_memory(shape, np.float32)
        expected_gb = (1000 * 1000 * 100 * 4) / (1024**3)
        assert abs(mem_gb - expected_gb) < 0.001

    def test_memory_manager_fits_check(self):
        """Test memory manager fits_in_memory check."""
        from aviris_tools.utils.memory import MemoryManager

        mm = MemoryManager(limit_gb=1.0)

        # Small array should fit
        assert mm.fits_in_memory((100, 100, 10))

        # Huge array should not fit
        assert not mm.fits_in_memory((10000, 10000, 1000))

    def test_chunk_iteration(self):
        """Test chunk iteration generates valid indices."""
        from aviris_tools.utils.memory import MemoryManager

        mm = MemoryManager(limit_gb=0.1)  # Small limit to force chunking
        shape = (1000, 1000, 100)

        chunks = list(mm.iterate_chunks(shape, chunk_dim=0))

        # Should have multiple chunks
        assert len(chunks) >= 1

        # First chunk starts at 0
        assert chunks[0][0] == 0

        # Last chunk ends at shape[0]
        assert chunks[-1][1] == shape[0]

        # Chunks should be contiguous
        for i in range(len(chunks) - 1):
            assert chunks[i][1] == chunks[i+1][0]


class TestConfig:
    """Test configuration management."""

    def test_config_singleton(self):
        """Config should be a singleton."""
        from aviris_tools.utils.config import Config

        c1 = Config()
        c2 = Config()
        assert c1 is c2

    def test_config_defaults(self):
        """Config should have sensible defaults."""
        from aviris_tools.utils.config import get_config

        config = get_config()

        # Memory should be auto-detected
        assert config.memory_limit_gb > 0

        # N cores should be auto-detected
        assert config.n_cores >= 1

        # Aerosol model should have default
        assert config.aerosol_model in ['continental', 'maritime', 'urban', 'desert']

    def test_config_get_nested(self):
        """Test nested config access."""
        from aviris_tools.utils.config import get_config

        config = get_config()

        # Get nested value
        chunk_size = config.get('memory', 'chunk_size_mb')
        assert chunk_size is not None
        assert isinstance(chunk_size, (int, float))

        # Get with default
        missing = config.get('nonexistent', 'key', default='fallback')
        assert missing == 'fallback'


class TestENVIIO:
    """Test ENVI format I/O."""

    def test_write_read_roundtrip(self):
        """Test that write->read produces identical data."""
        from aviris_tools.utils.envi_io import ENVIWriter, ENVIReader

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test'

            # Create test data
            data = np.random.rand(50, 60, 10).astype(np.float32)
            wavelengths = np.linspace(400, 900, 10)

            # Write
            ENVIWriter.write(filepath, data, wavelengths=wavelengths)

            # Read back
            reader = ENVIReader(filepath)
            read_data = reader.read()

            # Compare
            assert read_data.shape == data.shape
            np.testing.assert_array_almost_equal(read_data, data, decimal=5)

    def test_wavelengths_preserved(self):
        """Test that wavelengths are preserved in header."""
        from aviris_tools.utils.envi_io import ENVIWriter, ENVIReader

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test'

            data = np.random.rand(10, 10, 5).astype(np.float32)
            wavelengths = np.array([450, 550, 650, 750, 850])

            ENVIWriter.write(filepath, data, wavelengths=wavelengths)

            reader = ENVIReader(filepath)
            read_wl = reader.wavelengths

            np.testing.assert_array_almost_equal(read_wl, wavelengths, decimal=2)

    def test_interleave_formats(self):
        """Test different interleave formats."""
        from aviris_tools.utils.envi_io import ENVIWriter, ENVIReader

        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.rand(20, 30, 5).astype(np.float32)

            for interleave in ['bil', 'bip', 'bsq']:
                filepath = Path(tmpdir) / f'test_{interleave}'
                ENVIWriter.write(filepath, data, interleave=interleave)

                reader = ENVIReader(filepath)
                assert reader.interleave == interleave

                read_data = reader.read()
                np.testing.assert_array_almost_equal(read_data, data, decimal=5)


class TestAtmosphericBands:
    """Test atmospheric absorption band handling."""

    def test_water_vapor_bands_defined(self):
        """Ensure water vapor bands are in expected range."""
        bands = [
            (1350, 1450),  # Strong H2O
            (1800, 1950),  # Strong H2O
        ]

        for low, high in bands:
            assert 1000 < low < 2500
            assert 1000 < high < 2500
            assert low < high

    def test_mask_absorption_bands(self):
        """Test masking of absorption bands."""
        wavelengths = np.arange(400, 2500, 10)
        data = np.ones((10, 10, len(wavelengths)))

        # Define absorption regions
        absorption = [(1350, 1450), (1800, 1950)]

        # Create mask
        mask = np.zeros(len(wavelengths), dtype=bool)
        for low, high in absorption:
            mask |= (wavelengths >= low) & (wavelengths <= high)

        # Apply mask (set to NaN)
        data[:, :, mask] = np.nan

        # Check masked regions are NaN
        assert np.all(np.isnan(data[:, :, mask]))
        assert not np.any(np.isnan(data[:, :, ~mask]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
