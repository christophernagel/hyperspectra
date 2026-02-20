"""Simple debug script - writes output to file."""
import netCDF4 as nc
import numpy as np

output = []

def log(msg):
    print(msg)
    output.append(msg)

try:
    # ISOFIT file
    log("=== ISOFIT L2 ===")
    ds = nc.Dataset(r"C:\Users\chris\Downloads\AV320230926t201618_L2_REFL.nc")

    if 'reflectance' in ds.groups:
        rfl = ds.groups['reflectance'].variables['reflectance'][:]
        wl = ds.groups['reflectance'].variables['wavelength'][:]
    else:
        rfl = ds.variables['reflectance'][:]
        wl = ds.variables['wavelength'][:]

    wl = np.array(wl).flatten()
    log(f"Shape: {rfl.shape}, Bands: {len(wl)}")

    # Count NaN bands
    nan_count = 0
    nan_ranges = []
    in_nan_range = False
    range_start = 0

    for i in range(len(wl)):
        band = rfl[i, :, :] if rfl.shape[0] == len(wl) else rfl[:, :, i]
        is_nan = np.all(np.isnan(band))

        if is_nan:
            nan_count += 1
            if not in_nan_range:
                in_nan_range = True
                range_start = i
        else:
            if in_nan_range:
                nan_ranges.append((range_start, i-1, wl[range_start], wl[i-1]))
                in_nan_range = False

    if in_nan_range:
        nan_ranges.append((range_start, len(wl)-1, wl[range_start], wl[-1]))

    log(f"Total NaN bands: {nan_count}")
    log("NaN ranges:")
    for si, ei, sw, ew in nan_ranges:
        log(f"  {si}-{ei}: {sw:.1f}-{ew:.1f} nm ({ei-si+1} bands)")

    # Check clay bands
    log("\nClay index bands:")
    for target in [2120, 2200, 2250]:
        idx = np.argmin(np.abs(wl - target))
        band = rfl[idx, :, :] if rfl.shape[0] == len(wl) else rfl[:, :, idx]
        nan_pct = 100 * np.sum(np.isnan(band)) / band.size
        log(f"  {target}nm (band {idx}, {wl[idx]:.1f}nm): {nan_pct:.1f}% NaN")

    ds.close()

    # Simple file
    log("\n=== SIMPLE L2 ===")
    ds = nc.Dataset(r"C:\Users\chris\Downloads\AV320230926t201618_003_L2_REFL_simple.nc")

    if 'reflectance' in ds.groups:
        rfl = ds.groups['reflectance'].variables['reflectance'][:]
        wl = ds.groups['reflectance'].variables['wavelength'][:]
    else:
        rfl = ds.variables['reflectance'][:]
        wl = ds.variables['wavelength'][:]

    wl = np.array(wl).flatten()
    log(f"Shape: {rfl.shape}, Bands: {len(wl)}")

    nan_count = 0
    nan_ranges = []
    in_nan_range = False

    for i in range(len(wl)):
        band = rfl[i, :, :] if rfl.shape[0] == len(wl) else rfl[:, :, i]
        is_nan = np.all(np.isnan(band))

        if is_nan:
            nan_count += 1
            if not in_nan_range:
                in_nan_range = True
                range_start = i
        else:
            if in_nan_range:
                nan_ranges.append((range_start, i-1, wl[range_start], wl[i-1]))
                in_nan_range = False

    if in_nan_range:
        nan_ranges.append((range_start, len(wl)-1, wl[range_start], wl[-1]))

    log(f"Total NaN bands: {nan_count}")
    if nan_ranges:
        log("NaN ranges:")
        for si, ei, sw, ew in nan_ranges:
            log(f"  {si}-{ei}: {sw:.1f}-{ew:.1f} nm ({ei-si+1} bands)")

    log("\nClay index bands:")
    for target in [2120, 2200, 2250]:
        idx = np.argmin(np.abs(wl - target))
        band = rfl[idx, :, :] if rfl.shape[0] == len(wl) else rfl[:, :, idx]
        nan_pct = 100 * np.sum(np.isnan(band)) / band.size
        valid = band[~np.isnan(band)]
        if len(valid) > 0:
            log(f"  {target}nm (band {idx}): {nan_pct:.1f}% NaN, range: {valid.min():.4f}-{valid.max():.4f}")
        else:
            log(f"  {target}nm (band {idx}): 100% NaN")

    ds.close()

except Exception as e:
    log(f"Error: {e}")
    import traceback
    log(traceback.format_exc())

# Save output
with open(r"C:\Users\chris\aviris_tools\debug_output.txt", "w") as f:
    f.write("\n".join(output))

print("\nOutput saved to debug_output.txt")
