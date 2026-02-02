"""
Pytest configuration and fixtures for AVIRIS validation tests.
"""

import pytest


def pytest_addoption(parser):
    """Add command line options for validation tests."""
    parser.addoption(
        "--l2-path", action="store", default=None,
        help="Path to L2 reflectance NetCDF file for validation"
    )
