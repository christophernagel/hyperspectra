"""
Atmospheric correction modules for AVIRIS-3.

Available processors:
    - Py6SProcessor: 6S radiative transfer based correction
    - ISOFITProcessor: NASA/JPL ISOFIT + sRTMnet correction
"""

__all__ = ['Py6SProcessor', 'ISOFITProcessor']

# Lazy imports to avoid loading heavy dependencies
def __getattr__(name):
    if name == 'Py6SProcessor':
        from aviris_tools.atm_correction.py6s_processor import Py6SProcessor
        return Py6SProcessor
    elif name == 'ISOFITProcessor':
        from aviris_tools.atm_correction.isofit_processor import ISOFITProcessor
        return ISOFITProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
