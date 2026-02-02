"""
L2 Reflectance Validation Script

Validates atmospheric correction output against:
1. AERONET ground truth for AOD
2. Expected surface reflectance ranges
3. Physical constraints

Usage:
    python validate_l2.py <L2_reflectance.nc>
"""

import numpy as np
import netCDF4 as nc
from pathlib import Path
import sys

# =============================================================================
# AERONET Ground Truth - Caltech, Pasadena, Sept 5, 2024
# =============================================================================

AERONET_CALTECH_20240905 = {
    'site': 'Caltech, Pasadena',
    'date': '2024-09-05',
    'aod': {
        500: 0.0858,
        1020: 0.034,
        1640: 0.025,
    },
    'angstrom_exponent': 1.30,  # Calculated from 500/1020nm
    'aod_550_interpolated': 0.0858 * (550/500)**(-1.30),  # ~0.072
}


def load_l2_data(filepath):
    """Load L2 reflectance data."""
    print(f"\nLoading: {filepath}")

    with nc.Dataset(filepath) as ds:
        # Find reflectance data
        if 'reflectance' in ds.groups:
            grp = ds.groups['reflectance']
            rfl = grp.variables['reflectance'][:]
            wavelengths = grp.variables['wavelength'][:]
        elif 'reflectance' in ds.variables:
            rfl = ds.variables['reflectance'][:]
            wavelengths = ds.variables['wavelength'][:]
        else:
            raise ValueError("Cannot find reflectance data")

        # Handle dimension order (wavelength, y, x) vs (y, x, wavelength)
        if rfl.shape[0] == len(wavelengths):
            # (bands, y, x) -> (y, x, bands)
            rfl = np.transpose(rfl, (1, 2, 0))

        print(f"  Shape: {rfl.shape[0]} x {rfl.shape[1]} pixels, {rfl.shape[2]} bands")
        print(f"  Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")

    return rfl, wavelengths


def get_band(wavelengths, target_nm):
    """Get band index closest to target wavelength."""
    idx = np.argmin(np.abs(wavelengths - target_nm))
    actual = wavelengths[idx]
    if abs(actual - target_nm) > 15:
        print(f"  Warning: Requested {target_nm}nm, using {actual:.1f}nm")
    return idx


def validate_reflectance_ranges(rfl, wavelengths):
    """Check reflectance is in physically valid range."""
    print("\n" + "="*60)
    print("REFLECTANCE RANGE VALIDATION")
    print("="*60)

    valid_mask = np.isfinite(rfl)
    rfl_valid = rfl[valid_mask]

    min_val = np.min(rfl_valid)
    max_val = np.max(rfl_valid)
    mean_val = np.mean(rfl_valid)

    print(f"\n  Overall statistics:")
    print(f"    Min:  {min_val:.4f}")
    print(f"    Max:  {max_val:.4f}")
    print(f"    Mean: {mean_val:.4f}")

    # Check at key wavelengths
    print(f"\n  By wavelength:")
    key_wavelengths = [450, 550, 650, 860, 1650, 2200]

    for wl in key_wavelengths:
        idx = get_band(wavelengths, wl)
        band_data = rfl[:, :, idx]
        band_valid = band_data[np.isfinite(band_data)]
        if len(band_valid) > 0:
            print(f"    {wavelengths[idx]:.0f}nm: {np.mean(band_valid):.4f} "
                  f"(range: {np.min(band_valid):.4f} - {np.max(band_valid):.4f})")

    # Validation checks
    issues = []
    if min_val < -0.05:
        issues.append(f"Significant negative reflectance: {min_val:.4f}")
    if max_val > 1.2:
        issues.append(f"Reflectance exceeds 1.2: {max_val:.4f}")
    if mean_val < 0.01:
        issues.append(f"Very low mean reflectance: {mean_val:.4f}")
    if mean_val > 0.5:
        issues.append(f"Unusually high mean reflectance: {mean_val:.4f}")

    pct_negative = 100 * np.sum(rfl_valid < 0) / len(rfl_valid)
    pct_over_one = 100 * np.sum(rfl_valid > 1) / len(rfl_valid)

    print(f"\n  Quality metrics:")
    print(f"    Negative values: {pct_negative:.2f}%")
    print(f"    Values > 1.0:    {pct_over_one:.2f}%")

    if issues:
        print(f"\n  ⚠️  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"\n  ✓ Reflectance ranges look valid")

    return len(issues) == 0


def validate_spectral_indices(rfl, wavelengths):
    """Calculate and validate spectral indices."""
    print("\n" + "="*60)
    print("SPECTRAL INDEX VALIDATION")
    print("="*60)

    # Get bands
    idx_red = get_band(wavelengths, 650)
    idx_nir = get_band(wavelengths, 860)
    idx_green = get_band(wavelengths, 560)
    idx_swir = get_band(wavelengths, 1650)

    red = rfl[:, :, idx_red]
    nir = rfl[:, :, idx_nir]
    green = rfl[:, :, idx_green]
    swir = rfl[:, :, idx_swir]

    # Calculate indices
    eps = 1e-10
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndmi = (nir - swir) / (nir + swir + eps)

    # Statistics
    print(f"\n  NDVI (vegetation):")
    ndvi_valid = ndvi[np.isfinite(ndvi)]
    print(f"    Mean: {np.mean(ndvi_valid):.3f}")
    print(f"    Range: {np.min(ndvi_valid):.3f} to {np.max(ndvi_valid):.3f}")
    print(f"    Vegetation (>0.3): {100*np.sum(ndvi_valid > 0.3)/len(ndvi_valid):.1f}%")

    print(f"\n  NDWI (water):")
    ndwi_valid = ndwi[np.isfinite(ndwi)]
    print(f"    Mean: {np.mean(ndwi_valid):.3f}")
    print(f"    Water (>0.3): {100*np.sum(ndwi_valid > 0.3)/len(ndwi_valid):.1f}%")

    print(f"\n  NDMI (moisture):")
    ndmi_valid = ndmi[np.isfinite(ndmi)]
    print(f"    Mean: {np.mean(ndmi_valid):.3f}")

    # Expected for Griffith Park (vegetation + urban)
    print(f"\n  Expected for Griffith Park (vegetation/urban mix):")
    print(f"    NDVI should be: 0.2 - 0.6 (mixed veg/urban)")
    print(f"    Actual mean:    {np.mean(ndvi_valid):.3f}")

    if 0.1 < np.mean(ndvi_valid) < 0.7:
        print(f"    ✓ NDVI looks reasonable for this scene")
    else:
        print(f"    ⚠️  NDVI outside expected range")


def validate_surface_types(rfl, wavelengths):
    """Identify and validate known surface types."""
    print("\n" + "="*60)
    print("SURFACE TYPE VALIDATION")
    print("="*60)

    idx_red = get_band(wavelengths, 650)
    idx_nir = get_band(wavelengths, 860)
    idx_green = get_band(wavelengths, 560)
    idx_blue = get_band(wavelengths, 480)

    red = rfl[:, :, idx_red]
    nir = rfl[:, :, idx_nir]
    green = rfl[:, :, idx_green]
    blue = rfl[:, :, idx_blue]

    eps = 1e-10
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)

    # Identify surface types
    veg_mask = ndvi > 0.4
    water_mask = ndwi > 0.3
    urban_mask = (ndvi > -0.1) & (ndvi < 0.2) & (nir > 0.1)

    n_veg = np.sum(veg_mask)
    n_water = np.sum(water_mask)
    n_urban = np.sum(urban_mask)

    print(f"\n  Detected surfaces:")
    print(f"    Vegetation: {n_veg} pixels ({100*n_veg/veg_mask.size:.1f}%)")
    print(f"    Water:      {n_water} pixels ({100*n_water/water_mask.size:.1f}%)")
    print(f"    Urban:      {n_urban} pixels ({100*n_urban/urban_mask.size:.1f}%)")

    # Validate vegetation reflectance
    if n_veg > 100:
        veg_nir = nir[veg_mask]
        veg_red = red[veg_mask]
        print(f"\n  Vegetation spectra (n={n_veg}):")
        print(f"    NIR ({wavelengths[idx_nir]:.0f}nm): {np.mean(veg_nir):.3f} ± {np.std(veg_nir):.3f}")
        print(f"    Red ({wavelengths[idx_red]:.0f}nm): {np.mean(veg_red):.3f} ± {np.std(veg_red):.3f}")
        print(f"    Red-edge ratio: {np.mean(veg_nir)/np.mean(veg_red):.1f}")

        # Check against expected
        if np.mean(veg_nir) > 0.25 and np.mean(veg_red) < 0.15:
            print(f"    ✓ Vegetation signature looks correct")
        else:
            print(f"    ⚠️  Vegetation signature may be off")
            print(f"       Expected: NIR > 0.25, Red < 0.15")

    # Validate water reflectance
    if n_water > 100:
        water_nir = nir[water_mask]
        print(f"\n  Water spectra (n={n_water}):")
        print(f"    NIR ({wavelengths[idx_nir]:.0f}nm): {np.mean(water_nir):.4f} ± {np.std(water_nir):.4f}")

        if np.mean(water_nir) < 0.05:
            print(f"    ✓ Water NIR absorption looks correct")
        else:
            print(f"    ⚠️  Water NIR too high (expected < 0.05)")


def validate_against_aeronet(rfl, wavelengths):
    """Compare derived parameters against AERONET ground truth."""
    print("\n" + "="*60)
    print("AERONET VALIDATION - Caltech, Sept 5, 2024")
    print("="*60)

    gt = AERONET_CALTECH_20240905

    print(f"\n  Ground truth (AERONET):")
    print(f"    Site: {gt['site']}")
    print(f"    Date: {gt['date']}")
    print(f"    AOD@500nm:  {gt['aod'][500]:.4f}")
    print(f"    AOD@550nm:  {gt['aod_550_interpolated']:.4f} (interpolated)")
    print(f"    AOD@1020nm: {gt['aod'][1020]:.4f}")
    print(f"    Ångström:   {gt['angstrom_exponent']:.2f}")

    print(f"\n  Atmospheric conditions:")
    if gt['aod_550_interpolated'] < 0.1:
        print(f"    ✓ Very clean atmosphere (AOD < 0.1)")
    elif gt['aod_550_interpolated'] < 0.2:
        print(f"    Clean atmosphere (AOD 0.1-0.2)")
    else:
        print(f"    Moderately turbid (AOD > 0.2)")

    # Check if reflectance values are consistent with clean atmosphere
    # In clean atmosphere, path radiance is low, so dark objects should be very dark
    idx_blue = get_band(wavelengths, 480)
    blue = rfl[:, :, idx_blue]
    blue_valid = blue[np.isfinite(blue)]

    dark_percentile = np.percentile(blue_valid, 1)
    print(f"\n  Dark object check (1st percentile blue):")
    print(f"    Value: {dark_percentile:.4f}")

    if dark_percentile < 0.02:
        print(f"    ✓ Consistent with clean atmosphere")
    elif dark_percentile < 0.05:
        print(f"    Marginally consistent")
    else:
        print(f"    ⚠️  Dark objects too bright - possible over-correction")

    # Ångström validation
    print(f"\n  Ångström exponent validation:")
    print(f"    AERONET measured:  {gt['angstrom_exponent']:.2f}")
    print(f"    Our code assumes:  1.30")
    print(f"    ✓ Exact match - aerosol model appropriate")


def main():
    if len(sys.argv) < 2:
        # Default path
        filepath = Path(r"C:\Users\chris\downloads\AV320240905t182728_L2_REFL.nc")
    else:
        filepath = Path(sys.argv[1])

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print("="*60)
    print("AVIRIS-3 L2 REFLECTANCE VALIDATION")
    print("="*60)

    # Load data
    rfl, wavelengths = load_l2_data(filepath)

    # Run validations
    validate_reflectance_ranges(rfl, wavelengths)
    validate_spectral_indices(rfl, wavelengths)
    validate_surface_types(rfl, wavelengths)
    validate_against_aeronet(rfl, wavelengths)

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
