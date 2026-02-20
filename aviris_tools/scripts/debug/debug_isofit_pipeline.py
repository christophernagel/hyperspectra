"""
Debug script to trace NaN values through the ISOFIT pipeline.

Run this after processing to check intermediate files, or use as a template
for adding diagnostics to the main processor.
"""

import numpy as np
from pathlib import Path
import sys

def check_envi_file(filepath, name):
    """Check an ENVI file for NaN/zero bands."""
    filepath = Path(filepath)
    hdr_path = filepath.with_suffix('.hdr')

    if not filepath.exists():
        print(f"  {name}: File not found")
        return None

    # Parse header
    header = {}
    with open(hdr_path) as f:
        content = f.read()

    import re
    for match in re.finditer(r'(\w+)\s*=\s*({[^}]+}|[^\n]+)', content):
        key = match.group(1).lower()
        val = match.group(2).strip()
        if val.startswith('{'):
            val = val[1:-1]
        header[key] = val

    lines = int(header.get('lines', 0))
    samples = int(header.get('samples', 0))
    bands = int(header.get('bands', 1))
    interleave = header.get('interleave', 'bil').lower()

    # Read data
    data = np.fromfile(str(filepath), dtype=np.float32)

    if interleave == 'bil':
        data = data.reshape(lines, bands, samples)
    elif interleave == 'bip':
        data = data.reshape(lines, samples, bands).transpose(0, 2, 1)
    else:
        data = data.reshape(bands, lines, samples).transpose(1, 0, 2)

    # Analyze
    print(f"\n  {name}: {lines}x{samples}x{bands}")

    nan_bands = []
    zero_bands = []

    for b in range(bands):
        band_data = data[:, b, :]
        nan_frac = np.sum(np.isnan(band_data)) / band_data.size
        zero_frac = np.sum(band_data == 0) / band_data.size

        if nan_frac > 0.9:
            nan_bands.append(b)
        elif zero_frac > 0.9:
            zero_bands.append(b)

    print(f"    NaN bands (>90%): {len(nan_bands)}")
    if nan_bands:
        print(f"      Indices: {nan_bands[:10]}{'...' if len(nan_bands) > 10 else ''}")

    print(f"    Zero bands (>90%): {len(zero_bands)}")
    if zero_bands:
        print(f"      Indices: {zero_bands[:10]}{'...' if len(zero_bands) > 10 else ''}")

    # Get wavelengths if available
    if 'wavelength' in header:
        wl = np.array([float(x.strip()) for x in header['wavelength'].split(',')])

        # Check SWIR
        swir_nan = [b for b in nan_bands if 2000 <= wl[b] <= 2500]
        swir_zero = [b for b in zero_bands if 2000 <= wl[b] <= 2500]

        if swir_nan:
            print(f"    SWIR NaN bands: {len(swir_nan)} ({wl[swir_nan[0]]:.0f}-{wl[swir_nan[-1]]:.0f}nm)")
        if swir_zero:
            print(f"    SWIR Zero bands: {len(swir_zero)} ({wl[swir_zero[0]]:.0f}-{wl[swir_zero[-1]]:.0f}nm)")

    return {'nan_bands': nan_bands, 'zero_bands': zero_bands, 'shape': data.shape}


def find_isofit_outputs(work_dir):
    """Find all ISOFIT output files in working directory."""
    work_dir = Path(work_dir)

    output_dir = work_dir / 'output' / 'output'
    if not output_dir.exists():
        output_dir = work_dir / 'output'

    print(f"\nSearching in: {output_dir}")

    files_to_check = [
        ('*_rdn', 'Input radiance'),
        ('*subs_rfl', 'Subset reflectance (ISOFIT output)'),
        ('*subs_uncert', 'Subset uncertainty'),
        ('*lbl', 'Subset labels'),
        ('*rfl', 'Full reflectance'),
        ('*final_rfl', 'Final reflectance'),
    ]

    results = {}
    for pattern, name in files_to_check:
        matches = list(output_dir.glob(pattern))
        if matches:
            # Skip .hdr files
            matches = [m for m in matches if not str(m).endswith('.hdr')]
            if matches:
                for match in matches[:1]:  # Check first match
                    result = check_envi_file(match, f"{name} ({match.name})")
                    if result:
                        results[name] = result

    return results


def analyze_empirical_line_coefficients(work_dir, wavelengths=None):
    """
    Analyze empirical line coefficients to find problematic bands.

    This would require modifying apply_empirical_line() to save coefficients.
    """
    print("\n" + "="*60)
    print("EMPIRICAL LINE COEFFICIENT ANALYSIS")
    print("="*60)

    # Look for coefficient file if it exists
    coeff_file = Path(work_dir) / 'empirical_coeffs.npy'
    if coeff_file.exists():
        coeffs = np.load(coeff_file)
        print(f"Loaded coefficients: {coeffs.shape}")

        # Find bands with zero coefficients
        zero_slope = np.where(coeffs[:, 0] == 0)[0]
        zero_both = np.where((coeffs[:, 0] == 0) & (coeffs[:, 1] == 0))[0]

        print(f"Bands with zero slope: {len(zero_slope)}")
        print(f"Bands with zero slope+intercept: {len(zero_both)}")

        if wavelengths is not None and len(zero_both) > 0:
            print(f"Zero bands wavelengths: {wavelengths[zero_both[:10]]}")
    else:
        print("No coefficient file found.")
        print("Modify apply_empirical_line() to save coefficients for debugging:")
        print("  np.save(work_dir / 'empirical_coeffs.npy', coeffs)")


def main():
    """Main debug entry point."""
    print("="*60)
    print("ISOFIT PIPELINE DEBUG")
    print("="*60)

    # Check for recent temp directories
    import tempfile
    temp_base = Path(tempfile.gettempdir())

    isofit_dirs = list(temp_base.glob('aviris_isofit_*'))
    isofit_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    if isofit_dirs:
        print(f"\nFound {len(isofit_dirs)} ISOFIT work directories")
        print(f"Most recent: {isofit_dirs[0]}")

        # Check most recent
        work_dir = isofit_dirs[0] / 'isofit_work'
        if work_dir.exists():
            results = find_isofit_outputs(work_dir)
            analyze_empirical_line_coefficients(work_dir)
        else:
            print(f"Work directory not found: {work_dir}")
    else:
        print("\nNo ISOFIT work directories found in temp.")
        print("Run ISOFIT with --no-cleanup to preserve files for debugging.")

    # Also check for specific known outputs
    print("\n" + "="*60)
    print("CHECKING L2 OUTPUT FILES")
    print("="*60)

    l2_files = [
        r"C:\Users\chris\Downloads\AV320230926t201618_L2_REFL.nc",
        r"C:\Users\chris\Downloads\AV320230926t201618_L2_REFL_isofit.nc",
    ]

    import netCDF4 as nc

    for f in l2_files:
        if Path(f).exists():
            print(f"\n{Path(f).name}:")
            try:
                ds = nc.Dataset(f)
                if 'reflectance' in ds.groups:
                    rfl = ds.groups['reflectance'].variables['reflectance'][:]
                    wl = ds.groups['reflectance'].variables['wavelength'][:]
                else:
                    rfl = ds.variables['reflectance'][:]
                    wl = ds.variables['wavelength'][:]

                wl = np.array(wl).flatten()

                # Check 2100-2300nm range
                swir_mask = (wl >= 2100) & (wl <= 2300)
                swir_idx = np.where(swir_mask)[0]

                nan_in_swir = 0
                for idx in swir_idx:
                    band = rfl[idx] if rfl.shape[0] == len(wl) else rfl[:, :, idx]
                    if np.all(np.isnan(band)):
                        nan_in_swir += 1

                print(f"  SWIR 2100-2300nm: {len(swir_idx)} bands, {nan_in_swir} all-NaN")

                # Check processing metadata
                if hasattr(ds, 'processing_software'):
                    print(f"  Processor: {ds.processing_software}")
                if hasattr(ds, 'note'):
                    print(f"  Note: {ds.note}")

                ds.close()
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == '__main__':
    main()
