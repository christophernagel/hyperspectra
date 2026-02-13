# HyperspecI Indoor Scene Adaptation - Technical Notes

## Overview

This document summarizes the technical research and implementation decisions for adapting HyperSpectra to support the HyperspecI indoor hyperspectral dataset.

## Dataset Specifications

| Property | D1 (HyperspecI-V1) | D2 (HyperspecI-V2) |
|----------|--------------------|--------------------|
| Spectral Range | 400-1000nm | 400-1700nm |
| Channels | 61 | 131 |
| Resolution | 960 x 1230 px | 640 x 660 px |
| Interval | 10nm | 10nm |
| File Size | ~288 MB | ~222 MB |
| File Format | HDF5 (.h5) | HDF5 (.h5) |

**Source:** [bianlab/Hyperspectral-imaging-dataset](https://github.com/bianlab/Hyperspectral-imaging-dataset)

---

## Critical Issue: SRNet Reconstruction Artifacts

The HyperspecI data is **not raw sensor output**. Every spectral profile has been computationally reconstructed by SRNet from 16 broadband filter measurements.

### Known Failure Modes

1. **Spectral smoothing**: Sharp absorption dips get flattened. A narrow 2nm protein peak at 1510nm may appear as a 10nm-wide slope.

2. **Metamerism**: Materials with different spectra but similar 16-filter responses are reconstructed as nearly identical. Red plastic and red-dyed cotton may be indistinguishable.

3. **Boundary artifacts**: Edge pixels between materials can have hallucinated spectral features from neither material.

### Implications

- Material matching should show **similarity scores**, not definitive identifications
- Sharp absorption features in reference libraries may not match reconstructed data
- Edge pixels are unreliable for spectral analysis

**Reference:** [Limitations of Data-Driven Spectral Reconstruction](https://arxiv.org/abs/2403.17844)

---

## Wavelength Coverage Issues

### D2 Cuts Off at 1700nm - Not 1730nm

The primary lipid C-H absorption at **1730nm is outside the D2 spectral range**.

**Solution:** Use the 1210nm C-H second overtone as a proxy (12x weaker but within range), or the 1690nm CH3 peak at the spectral edge.

### Index Availability Matrix

| Index | Key Wavelengths | D1 (400-1000nm) | D2 (400-1700nm) |
|-------|-----------------|-----------------|-----------------|
| Chlorophyll | 680, 750nm | Full | Full |
| Moisture (VNIR) | 970nm | Edge (noisy) | Full |
| Moisture (SWIR) | 1430nm | Out of range | Full |
| Protein | 1500nm | Out of range | Full |
| Lipid (2nd OT) | 1210nm | Out of range | Full |
| Lipid (1st OT) | 1730nm | Out of range | **Out of range** |
| Cellulose | 1490nm | Out of range | Full |
| Aromatic | 1670nm | Out of range | Full |

---

## Verified Absorption Band Positions

From [Wilson et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4370890/) "Review of short-wave infrared spectroscopy":

### Water (O-H)
- 970nm: 2nd overtone (weak)
- 1180nm: O-H overtone
- 1430nm: 1st overtone (60x stronger than 970nm)
- 1940nm: O-H combo (outside D2 range)

### Lipid (C-H)
- 920nm: 2nd overtone
- 1210nm: 2nd overtone (12x stronger than 930nm)
- 1730nm: 1st overtone (80x stronger than 930nm) **— outside D2 range**

### Protein/Collagen (N-H, C-H)
- 1200nm: C-H 2nd overtone
- 1500nm: CH2 combo band
- 1690nm: CH3 1st overtone

---

## Spectral Library Sources

### Verified Reference Data

**USGS Spectral Library Version 7**
- Chapter A: Artificial/manmade materials (plastics, fabrics, paints, roofing)
- Coverage: 0.2-200 µm
- Format: ASCII text files, SPECPR format
- Download: https://doi.org/10.5066/F7RR1WDJ

**ECOSTRESS Spectral Library**
- 3,400+ spectra including man-made materials
- Coverage: VIS/SWIR (0.35-2.5µm) and TIR
- URL: https://speclib.jpl.nasa.gov/

### Current Implementation Status

The indoor material library in `viewer/constants.py` has:
- **Verified wavelength positions** from published NIR/SWIR spectroscopy literature
- **Approximate absorption depths** (need replacement with USGS measured values)

### USGS Library Export (Completed)

Exported 263 spectra from USGS v7 Chapter A (Artificial Materials) to:
- `reference_spectra/usgs_indoor_d1.json` (548 KB, 61 channels)
- `reference_spectra/usgs_indoor_d2.json` (963 KB, 131 channels)

**Material categories:**
| Category | Count | Examples |
|----------|-------|----------|
| construction | 25 | Concrete, asphalt, brick, aluminum |
| fabric | 7 | Cotton, burlap, nylon carpet |
| paint | 17 | Cadmium red/yellow/orange, cobalt blue, umber |
| plastic | 34 | Nylon, PVC, polyester, rubber |
| wood | 11 | Cedar shake, cardboard, plywood |
| other | 169 | Various artificial materials |

**Usage:**
```python
from aviris_tools.reference_spectra.usgs_parser import load_resampled_library
library = load_resampled_library("reference_spectra/usgs_indoor_d2.json")
wavelengths = library['wavelengths']  # 131 channels, 400-1700nm
cotton = library['spectra']['Cotton Fabric GDS437 White']['reflectance']
```

---

## Memory Management

### Single Scene Strategy

Both D1 (~288 MB) and D2 (~222 MB) scenes fit in RAM. **Load the full cube into numpy** rather than using h5py lazy slicing:

```python
# Recommended: load once, access fast
with h5py.File(path, 'r') as f:
    cube = np.array(f[key][:])  # Full load into RAM

# NOT recommended for interactive use:
# h5py per-pixel access is ~1000x slower than numpy indexing
```

### When Memory Becomes an Issue

- Multiple scenes: Use one loader at a time, close before loading next
- Batch processing: Use dask.array for lazy chunked access

---

## H5 File Structure (Unconfirmed)

The exact HDF5 key names are **not documented** in the bianlab repository. Use the exploration function when data arrives:

```python
from aviris_tools.viewer.h5_loader import explore_h5
explore_h5("path/to/scene.h5")
```

Expected structure based on conventions:
- Datacube: largest 3D array (auto-detected)
- Wavelengths: may be stored as attribute or separate dataset, or implied from specs
- RGB reference: may be stored as separate dataset

---

## Existing HSI Viewers

None of the existing tools handle the HyperspecI use case:

| Tool | Format | Indoor Materials | Web/Interactive |
|------|--------|------------------|-----------------|
| Spectronon (Resonon) | ENVI | No | Desktop only |
| HyperGUI | Various | No | R/Shiny (slow) |
| OpenSpectra | ENVI | No | Desktop |
| SPy | ENVI | No | Library only |

HyperSpectra fills a genuine gap: Python-native H5 loader with indoor material indices.

---

## Implementation Checklist

### Completed
- [x] H5 loader with auto-detection (`viewer/h5_loader.py`)
- [x] Wavelength range validation
- [x] Database type detection (D1/D2)
- [x] Indoor material signatures (wavelengths verified)
- [x] Indoor spectral indices with range requirements
- [x] Unified `load_hyperspectral()` factory function
- [x] H5 file exploration utility

### Before Dataset Arrives
- [x] Download USGS v7 Chapter A spectra (5.1GB complete library)
- [x] Write parser for USGS splib07a ASCII format
- [x] Resample to 10nm intervals (D1: 61ch, D2: 131ch)
- [ ] Replace approximate absorption depths with measured values

### After Dataset Arrives
- [ ] Run `explore_h5()` on first file to confirm structure
- [ ] Verify RGB alignment matches datacube spatial dimensions
- [ ] Test wavelength array generation
- [ ] Validate index calculations against known materials

### UI Integration (Future)
- [ ] Add indoor index selector to viewer control panel
- [ ] Grey out indices outside loaded wavelength range
- [ ] Add reconstruction quality warning to spectral plots
- [ ] Add material similarity matching (not definitive ID)
