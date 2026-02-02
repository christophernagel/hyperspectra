#!/usr/bin/env python3
"""
AVIRIS-3 Atmospheric Correction Verification Script

Validates L2 reflectance products against known ground truth and
compares Simple vs Full (6S) correction methods.

Author: AVIRIS Verification Suite
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    print("ERROR: netCDF4 required. Install with: pip install netCDF4")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SPECTRAL INDEX CALCULATIONS
# =============================================================================

def find_band_index(wavelengths: np.ndarray, target_nm: float, tolerance: float = 15.0) -> int:
    """Find the band index closest to a target wavelength."""
    idx = np.argmin(np.abs(wavelengths - target_nm))
    if np.abs(wavelengths[idx] - target_nm) > tolerance:
        logger.warning(f"No band within {tolerance}nm of {target_nm}nm (nearest: {wavelengths[idx]:.1f}nm)")
    return idx


def calculate_ndvi(reflectance: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Vegetation Index.
    NDVI = (NIR - Red) / (NIR + Red)
    """
    red_idx = find_band_index(wavelengths, 670)
    nir_idx = find_band_index(wavelengths, 860)

    red = reflectance[red_idx]
    nir = reflectance[nir_idx]

    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
        ndvi = np.clip(ndvi, -1, 1)

    return ndvi


def calculate_ndwi(reflectance: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Water Index.
    NDWI = (Green - NIR) / (Green + NIR)
    """
    green_idx = find_band_index(wavelengths, 560)
    nir_idx = find_band_index(wavelengths, 860)

    green = reflectance[green_idx]
    nir = reflectance[nir_idx]

    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - nir) / (green + nir)
        ndwi = np.clip(ndwi, -1, 1)

    return ndwi


def calculate_clay_index(reflectance: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Calculate Clay Mineral Index.
    Uses absorption feature near 2200nm characteristic of clay minerals.
    """
    # Shoulder wavelengths
    short_idx = find_band_index(wavelengths, 2120)
    center_idx = find_band_index(wavelengths, 2200)
    long_idx = find_band_index(wavelengths, 2260)

    short = reflectance[short_idx]
    center = reflectance[center_idx]
    long = reflectance[long_idx]

    # Continuum-removed absorption depth
    with np.errstate(divide='ignore', invalid='ignore'):
        continuum = (short + long) / 2
        clay_idx = (continuum - center) / continuum
        clay_idx = np.clip(clay_idx, 0, 1)

    return clay_idx


def calculate_alunite_index(reflectance: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """
    Calculate Alunite Index.
    Uses absorption feature at 2170nm characteristic of alunite.
    """
    short_idx = find_band_index(wavelengths, 2120)
    center_idx = find_band_index(wavelengths, 2170)
    long_idx = find_band_index(wavelengths, 2220)

    short = reflectance[short_idx]
    center = reflectance[center_idx]
    long = reflectance[long_idx]

    with np.errstate(divide='ignore', invalid='ignore'):
        continuum = (short + long) / 2
        alunite_idx = (continuum - center) / continuum
        alunite_idx = np.clip(alunite_idx, 0, 1)

    return alunite_idx


def calculate_kaolinite_doublet(reflectance: np.ndarray, wavelengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Kaolinite Doublet depths at 2165nm and 2205nm.
    Returns tuple of (depth_2165, depth_2205).
    """
    # Shoulders
    short_idx = find_band_index(wavelengths, 2120)
    long_idx = find_band_index(wavelengths, 2260)

    # Doublet centers
    center1_idx = find_band_index(wavelengths, 2165)
    center2_idx = find_band_index(wavelengths, 2205)

    short = reflectance[short_idx]
    long = reflectance[long_idx]
    center1 = reflectance[center1_idx]
    center2 = reflectance[center2_idx]

    with np.errstate(divide='ignore', invalid='ignore'):
        # Linear continuum
        wl_short = wavelengths[short_idx]
        wl_long = wavelengths[long_idx]
        wl_c1 = wavelengths[center1_idx]
        wl_c2 = wavelengths[center2_idx]

        # Interpolate continuum at each center
        frac1 = (wl_c1 - wl_short) / (wl_long - wl_short)
        frac2 = (wl_c2 - wl_short) / (wl_long - wl_short)

        cont1 = short + frac1 * (long - short)
        cont2 = short + frac2 * (long - short)

        depth1 = (cont1 - center1) / cont1
        depth2 = (cont2 - center2) / cont2

        depth1 = np.clip(depth1, 0, 1)
        depth2 = np.clip(depth2, 0, 1)

    return depth1, depth2


# =============================================================================
# DATA LOADING
# =============================================================================

def load_l2_data(filepath: Path) -> Dict:
    """Load L2 reflectance data from NetCDF."""
    logger.info(f"Loading: {filepath.name}")

    with nc.Dataset(filepath) as ds:
        # Find reflectance variable
        if 'reflectance' in ds.groups:
            rfl_grp = ds.groups['reflectance']
            reflectance = rfl_grp.variables['reflectance'][:]
            wavelengths = rfl_grp.variables['wavelength'][:]
        else:
            reflectance = ds.variables['reflectance'][:]
            wavelengths = ds.variables['wavelength'][:]

        # Get shape info
        n_bands, n_rows, n_cols = reflectance.shape

        # Load uncertainty if available
        uncertainty = None
        if 'uncertainty' in ds.groups:
            unc_grp = ds.groups['uncertainty']
            if 'uncertainty' in unc_grp.variables:
                uncertainty = unc_grp.variables['uncertainty'][:]

        # Load quality flags if available
        quality = None
        if 'quality' in ds.groups:
            qual_grp = ds.groups['quality']
            if 'quality_flag' in qual_grp.variables:
                quality = qual_grp.variables['quality_flag'][:]

        logger.info(f"  Shape: {n_bands} bands x {n_rows} x {n_cols} pixels")
        logger.info(f"  Wavelength range: {wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm")

        return {
            'reflectance': reflectance,
            'wavelengths': wavelengths,
            'uncertainty': uncertainty,
            'quality': quality,
            'shape': (n_bands, n_rows, n_cols)
        }


# =============================================================================
# QUALITY METRICS
# =============================================================================

def calculate_quality_metrics(data: Dict) -> Dict:
    """Calculate basic quality metrics for reflectance data."""
    rfl = data['reflectance']

    # Flatten for statistics
    valid_mask = np.isfinite(rfl)
    rfl_valid = rfl[valid_mask]

    metrics = {
        'min': float(np.min(rfl_valid)) if len(rfl_valid) > 0 else np.nan,
        'max': float(np.max(rfl_valid)) if len(rfl_valid) > 0 else np.nan,
        'mean': float(np.mean(rfl_valid)) if len(rfl_valid) > 0 else np.nan,
        'std': float(np.std(rfl_valid)) if len(rfl_valid) > 0 else np.nan,
        'negative_pct': float(np.sum(rfl_valid < 0) / len(rfl_valid) * 100) if len(rfl_valid) > 0 else 0,
        'over_one_pct': float(np.sum(rfl_valid > 1.0) / len(rfl_valid) * 100) if len(rfl_valid) > 0 else 0,
        'nan_pct': float(np.sum(~valid_mask) / rfl.size * 100)
    }

    return metrics


def calculate_surface_fractions(data: Dict) -> Dict:
    """Calculate surface type fractions based on spectral indices."""
    rfl = data['reflectance']
    wl = data['wavelengths']

    ndvi = calculate_ndvi(rfl, wl)
    ndwi = calculate_ndwi(rfl, wl)

    # Use valid pixels only
    valid = np.isfinite(ndvi) & np.isfinite(ndwi)
    total_valid = np.sum(valid)

    if total_valid == 0:
        return {'vegetation': 0, 'water': 0, 'bare': 0}

    fractions = {
        'vegetation': float(np.sum((ndvi > 0.3) & valid) / total_valid * 100),
        'water': float(np.sum((ndwi > 0.3) & valid) / total_valid * 100),
        'bare': float(np.sum((ndvi <= 0.3) & (ndwi <= 0.3) & valid) / total_valid * 100)
    }

    return fractions


# =============================================================================
# SITE-SPECIFIC VALIDATION
# =============================================================================

def validate_cuprite(data: Dict) -> Dict:
    """
    Cuprite-specific validation against known mineral locations.
    """
    rfl = data['reflectance']
    wl = data['wavelengths']

    # Calculate mineral indices
    alunite = calculate_alunite_index(rfl, wl)
    kaol_2165, kaol_2205 = calculate_kaolinite_doublet(rfl, wl)
    clay = calculate_clay_index(rfl, wl)

    # Find best mineral signatures
    valid = np.isfinite(alunite) & np.isfinite(kaol_2165)

    results = {
        'alunite_index': {
            'max': float(np.nanmax(alunite)),
            'mean': float(np.nanmean(alunite)),
            'p95': float(np.nanpercentile(alunite[valid], 95)) if np.any(valid) else np.nan
        },
        'kaolinite_2165': {
            'max': float(np.nanmax(kaol_2165)),
            'mean': float(np.nanmean(kaol_2165)),
            'p95': float(np.nanpercentile(kaol_2165[valid], 95)) if np.any(valid) else np.nan
        },
        'kaolinite_2205': {
            'max': float(np.nanmax(kaol_2205)),
            'mean': float(np.nanmean(kaol_2205)),
            'p95': float(np.nanpercentile(kaol_2205[valid], 95)) if np.any(valid) else np.nan
        },
        'clay_index': {
            'max': float(np.nanmax(clay)),
            'mean': float(np.nanmean(clay)),
            'p95': float(np.nanpercentile(clay[valid], 95)) if np.any(valid) else np.nan
        }
    }

    # Check for clear absorption features (validation criteria)
    results['validation'] = {
        'alunite_2170_clear': results['alunite_index']['p95'] > 0.05,
        'kaolinite_doublet_clear': (results['kaolinite_2165']['p95'] > 0.03 and
                                     results['kaolinite_2205']['p95'] > 0.03)
    }

    return results


def validate_santa_barbara(data: Dict) -> Dict:
    """
    Santa Barbara (coastal) specific validation.
    """
    rfl = data['reflectance']
    wl = data['wavelengths']

    ndvi = calculate_ndvi(rfl, wl)
    ndwi = calculate_ndwi(rfl, wl)

    # Find NIR band for water validation
    nir_idx = find_band_index(wl, 860)
    nir_rfl = rfl[nir_idx]

    # Water pixels (NDWI > 0.3)
    water_mask = ndwi > 0.3
    veg_mask = ndvi > 0.3

    results = {
        'water': {
            'fraction_pct': float(np.sum(water_mask) / water_mask.size * 100),
            'nir_mean': float(np.nanmean(nir_rfl[water_mask])) if np.any(water_mask) else np.nan,
            'nir_max': float(np.nanmax(nir_rfl[water_mask])) if np.any(water_mask) else np.nan,
            'nir_p95': float(np.nanpercentile(nir_rfl[water_mask], 95)) if np.any(water_mask) else np.nan
        },
        'vegetation': {
            'fraction_pct': float(np.sum(veg_mask) / veg_mask.size * 100),
            'ndvi_mean': float(np.nanmean(ndvi[veg_mask])) if np.any(veg_mask) else np.nan,
            'ndvi_max': float(np.nanmax(ndvi[veg_mask])) if np.any(veg_mask) else np.nan
        }
    }

    # Validation criteria
    results['validation'] = {
        'water_nir_low': results['water']['nir_mean'] < 0.05 if not np.isnan(results['water']['nir_mean']) else False,
        'vegetation_ndvi_reasonable': (0.3 <= results['vegetation']['ndvi_mean'] <= 0.8) if not np.isnan(results['vegetation']['ndvi_mean']) else False
    }

    return results


def validate_san_juan(data: Dict) -> Dict:
    """
    San Juan general validation.
    """
    rfl = data['reflectance']
    wl = data['wavelengths']

    ndvi = calculate_ndvi(rfl, wl)
    ndwi = calculate_ndwi(rfl, wl)

    veg_mask = ndvi > 0.3

    results = {
        'vegetation': {
            'fraction_pct': float(np.sum(veg_mask) / veg_mask.size * 100),
            'ndvi_mean': float(np.nanmean(ndvi[veg_mask])) if np.any(veg_mask) else np.nan,
            'ndvi_max': float(np.nanmax(ndvi[veg_mask])) if np.any(veg_mask) else np.nan
        },
        'overall': {
            'ndvi_mean': float(np.nanmean(ndvi)),
            'ndwi_mean': float(np.nanmean(ndwi))
        }
    }

    return results


# =============================================================================
# METHOD COMPARISON
# =============================================================================

def compare_methods(simple_data: Dict, full_data: Dict) -> Dict:
    """Compare Simple vs Full method results."""
    simple_rfl = simple_data['reflectance']
    full_rfl = full_data['reflectance']

    # Ensure same shape
    if simple_rfl.shape != full_rfl.shape:
        logger.error("Shape mismatch between Simple and Full outputs")
        return {}

    # Calculate difference statistics
    diff = full_rfl - simple_rfl
    valid = np.isfinite(diff)

    results = {
        'mean_abs_diff': float(np.mean(np.abs(diff[valid]))),
        'max_abs_diff': float(np.max(np.abs(diff[valid]))),
        'mean_diff': float(np.mean(diff[valid])),
        'std_diff': float(np.std(diff[valid]))
    }

    # Band-by-band correlation
    n_bands = simple_rfl.shape[0]
    correlations = []
    for b in range(n_bands):
        simple_band = simple_rfl[b].flatten()
        full_band = full_rfl[b].flatten()
        valid_band = np.isfinite(simple_band) & np.isfinite(full_band)
        if np.sum(valid_band) > 100:
            corr = np.corrcoef(simple_band[valid_band], full_band[valid_band])[0, 1]
            correlations.append(corr)

    results['mean_correlation'] = float(np.mean(correlations)) if correlations else np.nan
    results['min_correlation'] = float(np.min(correlations)) if correlations else np.nan

    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(results: Dict, output_path: Optional[Path] = None) -> str:
    """Generate verification report."""
    lines = []
    lines.append("=" * 60)
    lines.append("AVIRIS-3 Atmospheric Correction Verification Report")
    lines.append("=" * 60)
    lines.append("")

    for site, site_results in results.items():
        lines.append(f"\nSITE: {site.upper()}")
        lines.append("-" * 40)

        for method in ['simple', 'full']:
            if method not in site_results:
                continue

            method_data = site_results[method]
            lines.append(f"\nMethod: {method.capitalize()}")

            # Quality metrics
            if 'quality_metrics' in method_data:
                qm = method_data['quality_metrics']
                lines.append(f"  Reflectance: min={qm['min']:.4f}, max={qm['max']:.4f}, mean={qm['mean']:.4f}")
                lines.append(f"  Negative pixels: {qm['negative_pct']:.2f}%")
                lines.append(f"  Values > 1.0: {qm['over_one_pct']:.2f}%")

            # Surface fractions
            if 'surface_fractions' in method_data:
                sf = method_data['surface_fractions']
                lines.append(f"  Vegetation: {sf['vegetation']:.1f}%")
                lines.append(f"  Water: {sf['water']:.1f}%")
                lines.append(f"  Bare/mineral: {sf['bare']:.1f}%")

            # Site-specific validation
            if 'site_validation' in method_data:
                sv = method_data['site_validation']

                if 'alunite_index' in sv:
                    lines.append(f"  Alunite Index p95: {sv['alunite_index']['p95']:.3f}")
                if 'kaolinite_2165' in sv:
                    lines.append(f"  Kaolinite 2165nm p95: {sv['kaolinite_2165']['p95']:.3f}")
                if 'kaolinite_2205' in sv:
                    lines.append(f"  Kaolinite 2205nm p95: {sv['kaolinite_2205']['p95']:.3f}")
                if 'water' in sv:
                    lines.append(f"  Water NIR mean: {sv['water']['nir_mean']:.4f}")
                if 'vegetation' in sv and 'ndvi_mean' in sv['vegetation']:
                    lines.append(f"  Vegetation NDVI mean: {sv['vegetation']['ndvi_mean']:.3f}")

        # Method comparison
        if 'comparison' in site_results:
            comp = site_results['comparison']
            lines.append(f"\nMethod Comparison (Full - Simple):")
            lines.append(f"  Mean absolute difference: {comp['mean_abs_diff']:.4f}")
            lines.append(f"  Mean correlation (R): {comp['mean_correlation']:.4f}")
            lines.append(f"  Min correlation: {comp['min_correlation']:.4f}")

        lines.append("")

    # Summary section
    lines.append("\n" + "=" * 60)
    lines.append("VALIDATION SUMMARY")
    lines.append("=" * 60)

    all_pass = True
    for site, site_results in results.items():
        for method in ['simple', 'full']:
            if method not in site_results:
                continue
            method_data = site_results[method]

            # Check validation criteria
            qm = method_data.get('quality_metrics', {})
            if qm.get('negative_pct', 0) > 1.0:
                lines.append(f"  [WARNING] {site}/{method}: Excessive negative values ({qm['negative_pct']:.1f}%)")
                all_pass = False

            sv = method_data.get('site_validation', {})
            if 'validation' in sv:
                for check, passed in sv['validation'].items():
                    if not passed:
                        lines.append(f"  [WARNING] {site}/{method}: {check} - FAILED")
                        all_pass = False

    if all_pass:
        lines.append("  [OK] All validation checks passed")

    lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.write_text(report)
        logger.info(f"Report saved to: {output_path}")

    return report


# =============================================================================
# MAIN
# =============================================================================

def verify_file(filepath: Path, site: str) -> Dict:
    """Verify a single L2 file."""
    data = load_l2_data(filepath)

    results = {
        'quality_metrics': calculate_quality_metrics(data),
        'surface_fractions': calculate_surface_fractions(data)
    }

    # Site-specific validation
    if site == 'cuprite':
        results['site_validation'] = validate_cuprite(data)
    elif site == 'sb':
        results['site_validation'] = validate_santa_barbara(data)
    elif site == 'sj':
        results['site_validation'] = validate_san_juan(data)

    return results, data


def main():
    parser = argparse.ArgumentParser(
        description='Verify AVIRIS-3 atmospheric correction outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all C:\\Users\\chris\\downloads\\*_verify.nc
  %(prog)s --site cuprite C:\\Users\\chris\\downloads\\cuprite_L2_*.nc
  %(prog)s cuprite_L2_simple_verify.nc cuprite_L2_full_verify.nc --compare
        """
    )

    parser.add_argument('files', nargs='*', help='L2 NetCDF files to verify')
    parser.add_argument('--all', action='store_true',
                        help='Process all *_verify.nc files in specified pattern')
    parser.add_argument('--site', choices=['sb', 'cuprite', 'sj'],
                        help='Override site detection')
    parser.add_argument('--compare', action='store_true',
                        help='Compare simple vs full methods')
    parser.add_argument('--output', '-o', type=Path,
                        help='Output report file')
    parser.add_argument('--json', type=Path,
                        help='Output JSON results file')

    args = parser.parse_args()

    # Collect files
    if args.all and args.files:
        import glob
        files = []
        for pattern in args.files:
            files.extend([Path(f) for f in glob.glob(pattern)])
    else:
        files = [Path(f) for f in args.files]

    if not files:
        logger.error("No files specified")
        parser.print_help()
        return 1

    # Group files by site
    site_files = {'sb': {}, 'cuprite': {}, 'sj': {}}

    for f in files:
        fname = f.name.lower()

        # Detect site
        if args.site:
            site = args.site
        elif 'sb_' in fname or fname.startswith('sb'):
            site = 'sb'
        elif 'cuprite' in fname:
            site = 'cuprite'
        elif 'sj_' in fname or fname.startswith('sj'):
            site = 'sj'
        else:
            logger.warning(f"Could not determine site for {f.name}, skipping")
            continue

        # Detect method
        if '_simple_' in fname or fname.endswith('_simple.nc'):
            method = 'simple'
        elif '_full_' in fname or fname.endswith('_full.nc'):
            method = 'full'
        else:
            method = 'unknown'

        site_files[site][method] = f

    # Process each site
    all_results = {}
    all_data = {}

    for site, methods in site_files.items():
        if not methods:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing site: {site.upper()}")
        logger.info(f"{'='*60}")

        all_results[site] = {}
        all_data[site] = {}

        for method, filepath in methods.items():
            logger.info(f"\nMethod: {method}")
            results, data = verify_file(filepath, site)
            all_results[site][method] = results
            all_data[site][method] = data

        # Compare methods if both available
        if 'simple' in all_data[site] and 'full' in all_data[site]:
            logger.info("\nComparing Simple vs Full methods...")
            all_results[site]['comparison'] = compare_methods(
                all_data[site]['simple'],
                all_data[site]['full']
            )

    # Generate report
    report = generate_report(all_results, args.output)
    print("\n" + report)

    # Save JSON if requested
    if args.json:
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            return obj

        json_results = convert_for_json(all_results)
        args.json.write_text(json.dumps(json_results, indent=2))
        logger.info(f"JSON results saved to: {args.json}")

    return 0


if __name__ == '__main__':
    exit(main())
