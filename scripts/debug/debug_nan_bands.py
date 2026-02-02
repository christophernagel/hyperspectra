"""Debug script to analyze NaN bands in ISOFIT vs Simple L2 outputs."""

import netCDF4 as nc
import numpy as np
from pathlib import Path

def analyze_l2_file(filepath, name):
    """Analyze a single L2 file for NaN bands."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")

    ds = nc.Dataset(filepath)

    # Get reflectance and wavelengths
    if 'reflectance' in ds.groups:
        rfl = ds.groups['reflectance'].variables['reflectance'][:]
        wl = ds.groups['reflectance'].variables['wavelength'][:]
    else:
        rfl = ds.variables['reflectance'][:]
        wl = ds.variables['wavelength'][:]

    wl = np.array(wl).flatten()
    print(f"Shape: {rfl.shape}")
    print(f"Wavelength range: {wl[0]:.1f} - {wl[-1]:.1f} nm")

    # Determine band axis
    if rfl.shape[0] == len(wl):
        band_axis = 0
    else:
        band_axis = 2

    # Find bands with all NaN
    all_nan_bands = []
    mostly_nan_bands = []

    for i in range(len(wl)):
        if band_axis == 0:
            band_data = rfl[i, :, :]
        else:
            band_data = rfl[:, :, i]

        nan_frac = np.sum(np.isnan(band_data)) / band_data.size

        if nan_frac == 1.0:
            all_nan_bands.append((i, wl[i]))
        elif nan_frac > 0.5:
            mostly_nan_bands.append((i, wl[i], nan_frac))

    print(f"\nBands with 100% NaN: {len(all_nan_bands)}")

    if all_nan_bands:
        # Group into contiguous ranges
        ranges = []
        start_idx, start_wl = all_nan_bands[0]
        end_idx, end_wl = start_idx, start_wl

        for idx, w in all_nan_bands[1:]:
            if idx == end_idx + 1:
                end_idx, end_wl = idx, w
            else:
                ranges.append((start_idx, end_idx, start_wl, end_wl))
                start_idx, start_wl = idx, w
                end_idx, end_wl = idx, w
        ranges.append((start_idx, end_idx, start_wl, end_wl))

        print("NaN band ranges:")
        for si, ei, sw, ew in ranges:
            print(f"  Bands {si}-{ei}: {sw:.1f} - {ew:.1f} nm ({ei-si+1} bands)")

    # Check SWIR region specifically
    print(f"\n--- SWIR Region (2000-2500nm) ---")
    swir_mask = (wl >= 2000) & (wl <= 2500)
    swir_indices = np.where(swir_mask)[0]

    if len(swir_indices) > 0:
        print(f"SWIR bands: {len(swir_indices)} ({wl[swir_indices[0]]:.1f} - {wl[swir_indices[-1]]:.1f} nm)")

        swir_valid = 0
        swir_nan = 0
        for i in swir_indices:
            if band_axis == 0:
                band_data = rfl[i, :, :]
            else:
                band_data = rfl[:, :, i]

            if np.all(np.isnan(band_data)):
                swir_nan += 1
            else:
                swir_valid += 1

        print(f"Valid SWIR bands: {swir_valid}")
        print(f"All-NaN SWIR bands: {swir_nan}")

    # Check specific mineral index bands
    print(f"\n--- Mineral Index Bands ---")
    targets = [
        (2120, "Clay shoulder"),
        (2200, "Clay center"),
        (2250, "Clay shoulder"),
        (2330, "Carbonate center"),
        (1730, "Hydrocarbon"),
    ]

    for target, desc in targets:
        idx = np.argmin(np.abs(wl - target))

        if band_axis == 0:
            band_data = rfl[idx, :, :]
        else:
            band_data = rfl[:, :, idx]

        nan_frac = np.sum(np.isnan(band_data)) / band_data.size
        valid_data = band_data[~np.isnan(band_data)]

        if len(valid_data) > 0:
            print(f"  {target}nm ({desc}): band {idx}, actual {wl[idx]:.1f}nm")
            print(f"    NaN: {nan_frac*100:.1f}%, range: {valid_data.min():.4f} - {valid_data.max():.4f}")
        else:
            print(f"  {target}nm ({desc}): band {idx}, actual {wl[idx]:.1f}nm - 100% NaN")

    ds.close()
    return all_nan_bands, wl


def main():
    # Files to analyze
    files = [
        (r"C:\Users\chris\Downloads\AV320230926t201618_L2_REFL.nc", "ISOFIT L2"),
        (r"C:\Users\chris\Downloads\AV320230926t201618_003_L2_REFL_simple.nc", "Simple Method L2"),
    ]

    results = {}
    for filepath, name in files:
        if Path(filepath).exists():
            nan_bands, wl = analyze_l2_file(filepath, name)
            results[name] = {'nan_bands': nan_bands, 'wavelengths': wl}
        else:
            print(f"\nFile not found: {filepath}")

    # Compare
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")

        isofit_nan = set(b[0] for b in results['ISOFIT L2']['nan_bands'])
        simple_nan = set(b[0] for b in results['Simple Method L2']['nan_bands'])

        only_isofit = isofit_nan - simple_nan
        only_simple = simple_nan - isofit_nan
        both_nan = isofit_nan & simple_nan

        wl = results['ISOFIT L2']['wavelengths']

        print(f"\nNaN in ISOFIT only: {len(only_isofit)} bands")
        if only_isofit and len(only_isofit) <= 50:
            wl_only_isofit = [wl[i] for i in sorted(only_isofit)]
            print(f"  Wavelengths: {wl_only_isofit[0]:.1f} - {wl_only_isofit[-1]:.1f} nm")

        print(f"\nNaN in Simple only: {len(only_simple)} bands")
        if only_simple:
            wl_only_simple = [wl[i] for i in sorted(only_simple)]
            print(f"  Wavelengths: {min(wl_only_simple):.1f} - {max(wl_only_simple):.1f} nm")

        print(f"\nNaN in both: {len(both_nan)} bands")
        if both_nan:
            wl_both = [wl[i] for i in sorted(both_nan)]
            print(f"  Wavelengths: {min(wl_both):.1f} - {max(wl_both):.1f} nm")


if __name__ == '__main__':
    main()
