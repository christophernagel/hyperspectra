"""Check ISOFIT LUT and subs_rfl files."""
import numpy as np
from pathlib import Path

base = Path(r"C:\Users\chris\AppData\Local\Temp\aviris_isofit_d33ck_k9\isofit_work\output")

# Check LUT
print("=== LUT Check ===")
lut_path = base / "lut_full" / "lut.nc"
if lut_path.exists():
    import netCDF4 as nc
    ds = nc.Dataset(lut_path)
    print(f"LUT variables: {list(ds.variables.keys())}")
    for var in ds.variables:
        data = ds.variables[var][:]
        print(f"  {var}: shape={data.shape}, min={np.nanmin(data):.4f}, max={np.nanmax(data):.4f}")
    ds.close()
else:
    print(f"LUT not found: {lut_path}")

# Check 6S LUT
sixs_lut = base / "lut_full" / "6S.lut.nc"
if sixs_lut.exists():
    ds = nc.Dataset(sixs_lut)
    print(f"\n6S LUT variables: {list(ds.variables.keys())}")
    for var in list(ds.variables.keys())[:5]:
        data = ds.variables[var][:]
        print(f"  {var}: shape={data.shape}, min={np.nanmin(data):.4f}, max={np.nanmax(data):.4f}")
    ds.close()

# Check subset reflectance
print("\n=== Subset Reflectance ===")
subs_rfl = base / "output" / "AV320230926t201618_subs_rfl"
if subs_rfl.exists():
    data = np.fromfile(str(subs_rfl), dtype=np.float32)
    print(f"Size: {len(data)}")
    print(f"Min: {data.min():.6f}")
    print(f"Max: {data.max():.6f}")
    print(f"Mean: {data.mean():.6f}")
    print(f"Non-zero count: {np.sum(data != 0)}")
    print(f"Unique values: {len(np.unique(data))}")
else:
    print(f"subs_rfl not found")

# Check state vector
print("\n=== State Vector ===")
subs_state = base / "output" / "AV320230926t201618_subs_state"
if subs_state.exists():
    data = np.fromfile(str(subs_state), dtype=np.float32)
    print(f"Size: {len(data)}")
    print(f"Min: {data.min():.6f}")
    print(f"Max: {data.max():.6f}")
    # State vector should have H2O and AOT values
    # If it's all zeros, the inversion failed
    if data.max() == 0:
        print("WARNING: State vector is all zeros - inversion failed!")
    else:
        print("State vector has non-zero values")
