"""
Memory management utilities for large hyperspectral datasets.

Features:
    - Auto-detection of available system memory
    - Intelligent chunking for large arrays
    - Memory-mapped file support
    - Garbage collection helpers
"""

import os
import gc
import logging
import numpy as np
from typing import Tuple, Optional, Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def get_available_memory() -> float:
    """
    Get available system memory in GB.

    Returns:
        Available memory in GB, or conservative estimate if detection fails.
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        logger.debug(f"Available memory: {available_gb:.1f} GB")
        return available_gb
    except ImportError:
        logger.warning("psutil not installed, using conservative memory estimate")
        return 4.0  # Conservative fallback


def get_total_memory() -> float:
    """Get total system memory in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return 8.0


def estimate_array_memory(shape: Tuple[int, ...], dtype=np.float32) -> float:
    """
    Estimate memory usage of an array in GB.

    Args:
        shape: Array dimensions
        dtype: Numpy dtype

    Returns:
        Estimated memory in GB
    """
    itemsize = np.dtype(dtype).itemsize
    total_bytes = np.prod(shape) * itemsize
    return total_bytes / (1024**3)


class MemoryManager:
    """
    Manages memory for large dataset processing.

    Handles chunking, caching, and cleanup for hyperspectral cubes
    that may exceed available RAM.
    """

    def __init__(self, limit_gb: Optional[float] = None, safety_factor: float = 0.8):
        """
        Initialize memory manager.

        Args:
            limit_gb: Memory limit in GB (auto-detect if None)
            safety_factor: Fraction of available memory to use (default 0.8)
        """
        if limit_gb is None:
            limit_gb = get_available_memory() * safety_factor

        self.limit_gb = limit_gb
        self.safety_factor = safety_factor
        self._cached_arrays = {}
        logger.info(f"MemoryManager initialized with {limit_gb:.1f} GB limit")

    def fits_in_memory(self, shape: Tuple[int, ...], dtype=np.float32) -> bool:
        """Check if array fits in available memory."""
        required = estimate_array_memory(shape, dtype)
        available = get_available_memory() * self.safety_factor
        return required < available

    def calculate_chunk_size(self, shape: Tuple[int, ...], dtype=np.float32,
                             chunk_dim: int = 0) -> int:
        """
        Calculate optimal chunk size for a given dimension.

        Args:
            shape: Full array shape
            dtype: Array dtype
            chunk_dim: Dimension to chunk along (default 0)

        Returns:
            Number of slices per chunk
        """
        available_gb = min(self.limit_gb, get_available_memory() * self.safety_factor)

        # Calculate memory per slice along chunk_dim
        slice_shape = list(shape)
        slice_shape[chunk_dim] = 1
        slice_gb = estimate_array_memory(tuple(slice_shape), dtype)

        if slice_gb == 0:
            return shape[chunk_dim]

        # How many slices fit in memory (with buffer for processing)
        max_slices = int(available_gb / slice_gb / 2)  # /2 for processing headroom
        chunk_size = max(1, min(max_slices, shape[chunk_dim]))

        logger.debug(f"Chunk size for shape {shape}: {chunk_size} along dim {chunk_dim}")
        return chunk_size

    def iterate_chunks(self, shape: Tuple[int, ...], chunk_dim: int = 0,
                       chunk_size: Optional[int] = None) -> Generator[Tuple[int, int], None, None]:
        """
        Generate chunk indices for iterating over large arrays.

        Args:
            shape: Full array shape
            chunk_dim: Dimension to chunk along
            chunk_size: Override automatic chunk size

        Yields:
            (start_idx, end_idx) tuples for each chunk
        """
        if chunk_size is None:
            chunk_size = self.calculate_chunk_size(shape, chunk_dim=chunk_dim)

        total = shape[chunk_dim]
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            yield start, end

    @contextmanager
    def processing_context(self, description: str = "processing"):
        """
        Context manager for memory-intensive operations.

        Runs garbage collection before and after.
        """
        logger.debug(f"Starting {description}, available: {get_available_memory():.1f} GB")
        gc.collect()

        try:
            yield
        finally:
            gc.collect()
            logger.debug(f"Finished {description}, available: {get_available_memory():.1f} GB")

    def clear_cache(self):
        """Clear cached arrays and run garbage collection."""
        self._cached_arrays.clear()
        gc.collect()
        logger.debug("Cleared memory cache")


def chunk_array(array: np.ndarray, chunk_size_mb: float = 512,
                axis: int = 0) -> Generator[Tuple[np.ndarray, int], None, None]:
    """
    Iterate over array in memory-efficient chunks.

    Args:
        array: Input array
        chunk_size_mb: Target chunk size in MB
        axis: Axis to chunk along

    Yields:
        (chunk_array, start_index) tuples
    """
    chunk_bytes = chunk_size_mb * 1024 * 1024
    slice_size = array[0].nbytes if axis == 0 else array[:, 0].nbytes
    n_slices = max(1, int(chunk_bytes / slice_size))

    total = array.shape[axis]
    for start in range(0, total, n_slices):
        end = min(start + n_slices, total)
        if axis == 0:
            yield array[start:end], start
        elif axis == 1:
            yield array[:, start:end], start
        elif axis == 2:
            yield array[:, :, start:end], start
        else:
            # Generic slicing
            slices = [slice(None)] * array.ndim
            slices[axis] = slice(start, end)
            yield array[tuple(slices)], start
