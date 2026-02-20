"""Check actual reflectance values in clay bands."""
import netCDF4 as nc
import numpy as np

files = [
    (r"C:\Users\chris\Downloads\AV320230926t201618_L2_REFL.nc", "ISOFIT"),
    (r"C:\Users\chris\Downloads\AV320230926t201618_003_L2_REFL_simple.nc", "Simple"),
]

for filepath, name in files:
    print(f"\n{'='*50}")
    print(f"{name} L2")
    print('='*50)

    ds = nc.Dataset(filepath)

    if 'reflectance' in ds.groups:
        rfl = ds.groups['reflectance'].variables['reflectance'][:]
        wl = ds.groups['reflectance'].variables['wavelength'][:]
    else:
        rfl = ds.variables['reflectance'][:]
        wl = ds.variables['wavelength'][:]

    wl = np.array(wl).flatten()

    # Check clay bands
    print("\nClay Index Bands Statistics:")
    for target in [2120, 2200, 2250]:
        idx = np.argmin(np.abs(wl - target))
        band = rfl[idx, :, :] if rfl.shape[0] == len(wl) else rfl[:, :, idx]

        valid = band[~np.isnan(band)]

        print(f"\n  {target}nm (band {idx}, actual {wl[idx]:.1f}nm):")
        print(f"    Min:    {valid.min():.6f}")
        print(f"    Max:    {valid.max():.6f}")
        print(f"    Mean:   {valid.mean():.6f}")
        print(f"    Median: {np.median(valid):.6f}")
        print(f"    Std:    {valid.std():.6f}")
        print(f"    % zeros: {100 * np.sum(valid == 0) / len(valid):.1f}%")
        print(f"    % < 0.01: {100 * np.sum(valid < 0.01) / len(valid):.1f}%")

    # Check a sample pixel spectrum in SWIR
    print("\n\nSample pixel (center) SWIR spectrum (2000-2400nm):")
    cy, cx = rfl.shape[1] // 2, rfl.shape[2] // 2

    swir_mask = (wl >= 2000) & (wl <= 2400)
    swir_idx = np.where(swir_mask)[0]

    print(f"  Wavelength | Reflectance")
    print(f"  -----------|------------")
    for idx in swir_idx[::5]:  # Every 5th band
        val = rfl[idx, cy, cx] if rfl.shape[0] == len(wl) else rfl[cy, cx, idx]
        print(f"  {wl[idx]:7.1f}nm | {val:.6f}")

    ds.close()

print("\n\nConclusion:")
print("If ISOFIT values are all ~0 while Simple has realistic values (0.1-0.4),")
print("then ISOFIT's empirical line extrapolation failed due to zero subset inputs.")
