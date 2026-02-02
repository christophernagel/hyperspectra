"""
ENVI format I/O utilities.

Handles reading and writing ENVI binary format files with headers,
which is required for ISOFIT integration.
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ENVI data type mapping
ENVI_DTYPE_MAP = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    6: np.complex64,
    9: np.complex128,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}

DTYPE_TO_ENVI = {v: k for k, v in ENVI_DTYPE_MAP.items()}


class ENVIReader:
    """Read ENVI format hyperspectral files."""

    def __init__(self, filepath: Path):
        """
        Initialize reader.

        Args:
            filepath: Path to ENVI file (with or without .hdr extension)
        """
        self.filepath = Path(filepath)
        self.header_path = self._find_header()
        self.binary_path = self._find_binary()
        self.header = self._parse_header()

    def _find_header(self) -> Path:
        """Find the header file."""
        if self.filepath.suffix == '.hdr':
            return self.filepath
        hdr_path = self.filepath.with_suffix('.hdr')
        if hdr_path.exists():
            return hdr_path
        # Try adding .hdr
        hdr_path = Path(str(self.filepath) + '.hdr')
        if hdr_path.exists():
            return hdr_path
        raise FileNotFoundError(f"Cannot find header for {self.filepath}")

    def _find_binary(self) -> Path:
        """Find the binary data file."""
        if self.filepath.suffix == '.hdr':
            binary = self.filepath.with_suffix('')
        else:
            binary = self.filepath.with_suffix('')
        if binary.exists():
            return binary
        # Try original path
        if self.filepath.exists() and self.filepath.suffix != '.hdr':
            return self.filepath
        raise FileNotFoundError(f"Cannot find binary data for {self.filepath}")

    def _parse_header(self) -> Dict[str, Any]:
        """Parse ENVI header file."""
        header = {}
        with open(self.header_path, 'r') as f:
            content = f.read()

        # Match key = value pairs, handling multi-line braced values
        pattern = r'(\w+)\s*=\s*({[^}]+}|[^\n]+)'
        for match in re.finditer(pattern, content, re.DOTALL):
            key = match.group(1).strip().lower()
            value = match.group(2).strip()

            # Remove braces from lists
            if value.startswith('{') and value.endswith('}'):
                value = value[1:-1].strip()

            header[key] = value

        return header

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return (lines, samples, bands) shape."""
        return (
            int(self.header.get('lines', 0)),
            int(self.header.get('samples', 0)),
            int(self.header.get('bands', 1))
        )

    @property
    def dtype(self) -> np.dtype:
        """Return numpy dtype."""
        dtype_code = int(self.header.get('data type', 4))
        return np.dtype(ENVI_DTYPE_MAP.get(dtype_code, np.float32))

    @property
    def interleave(self) -> str:
        """Return interleave type (bil, bip, bsq)."""
        return self.header.get('interleave', 'bil').lower()

    @property
    def wavelengths(self) -> Optional[np.ndarray]:
        """Return wavelength array if present."""
        wl_str = self.header.get('wavelength', '')
        if wl_str:
            try:
                return np.array([float(x.strip()) for x in wl_str.split(',')])
            except ValueError:
                return None
        return None

    def read(self) -> np.ndarray:
        """
        Read entire file into memory.

        Returns:
            Array with shape (lines, samples, bands)
        """
        lines, samples, bands = self.shape
        data = np.fromfile(str(self.binary_path), dtype=self.dtype)

        # Reshape based on interleave
        if self.interleave == 'bil':
            data = data.reshape(lines, bands, samples).transpose(0, 2, 1)
        elif self.interleave == 'bip':
            data = data.reshape(lines, samples, bands)
        else:  # bsq
            data = data.reshape(bands, lines, samples).transpose(1, 2, 0)

        return data

    def read_band(self, band_idx: int) -> np.ndarray:
        """
        Read a single band efficiently.

        Args:
            band_idx: Band index (0-based)

        Returns:
            2D array (lines, samples)
        """
        lines, samples, bands = self.shape
        itemsize = self.dtype.itemsize

        if self.interleave == 'bsq':
            # Band sequential - can read directly
            offset = band_idx * lines * samples * itemsize
            data = np.fromfile(str(self.binary_path), dtype=self.dtype,
                             count=lines * samples, offset=offset)
            return data.reshape(lines, samples)
        else:
            # Need to read full file for BIL/BIP
            full = self.read()
            return full[:, :, band_idx]


class ENVIWriter:
    """Write ENVI format hyperspectral files."""

    @staticmethod
    def write(filepath: Path, data: np.ndarray,
              wavelengths: Optional[np.ndarray] = None,
              fwhm: Optional[np.ndarray] = None,
              description: str = "",
              interleave: str = "bil") -> Path:
        """
        Write array to ENVI format.

        Args:
            filepath: Output path (without extension)
            data: 3D array (lines, samples, bands) or 2D (lines, samples)
            wavelengths: Band wavelengths in nm
            fwhm: Full width half maximum for each band
            description: File description
            interleave: Data interleave (bil, bip, bsq)

        Returns:
            Path to written binary file
        """
        filepath = Path(filepath)

        # Ensure 3D
        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        lines, samples, bands = data.shape
        dtype = data.dtype

        # Get ENVI dtype code
        dtype_code = DTYPE_TO_ENVI.get(dtype.type, 4)

        # Convert if needed
        if dtype.type not in DTYPE_TO_ENVI:
            data = data.astype(np.float32)
            dtype_code = 4

        # Write binary data
        binary_path = filepath.with_suffix('')
        data = np.ascontiguousarray(data)

        if interleave == 'bil':
            out_data = np.ascontiguousarray(data.transpose(0, 2, 1))
        elif interleave == 'bip':
            out_data = data
        else:  # bsq
            out_data = np.ascontiguousarray(data.transpose(2, 0, 1))

        out_data.tofile(str(binary_path))

        # Write header
        header_path = filepath.with_suffix('.hdr')
        header_lines = [
            "ENVI",
            f"description = {{{description}}}",
            f"samples = {samples}",
            f"lines = {lines}",
            f"bands = {bands}",
            f"header offset = 0",
            f"file type = ENVI Standard",
            f"data type = {dtype_code}",
            f"interleave = {interleave}",
            f"byte order = 0",
        ]

        if wavelengths is not None and len(wavelengths) == bands:
            wl_str = ", ".join(f"{w:.4f}" for w in wavelengths)
            header_lines.append(f"wavelength = {{{wl_str}}}")
            header_lines.append("wavelength units = nanometers")

        if fwhm is not None and len(fwhm) == bands:
            fwhm_str = ", ".join(f"{f:.4f}" for f in fwhm)
            header_lines.append(f"fwhm = {{{fwhm_str}}}")

        with open(header_path, 'w') as f:
            f.write('\n'.join(header_lines))

        logger.info(f"Wrote ENVI: {binary_path} ({lines}x{samples}x{bands})")
        return binary_path
