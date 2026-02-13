# Changelog

All notable changes to Hyperspectra are documented here.

---

## [Unreleased] - 2026-02-11

### Added

**HyperspecI Indoor Scene Support**
- `viewer/h5_loader.py` - HDF5 loader for HyperspecI dataset with auto-detection of datacube structure
- `viewer/data_loader.py` - Added `load_hyperspectral()` factory function for unified format handling
- `viewer/illumination.py` - Illumination normalization module for indoor scenes (flat-field, white reference detection)
- `docs/hyperspecI_adaptation_notes.md` - Technical documentation on dataset specs, SRNet artifacts, wavelength coverage

**USGS Spectral Library Integration**
- `reference_spectra/usgs_parser.py` - Parser for USGS Spectral Library v7 (splib07a format)
- `reference_spectra/usgs_indoor_d2.json` - 263 artificial material spectra (400-1700nm, 131 channels)
- `reference_spectra/usgs_minerals_d2.json` - 448 mineral spectra (400-1700nm, 131 channels)
- Material categories: construction (25), fabric (7), paint (17), plastic (34), wood (11), minerals (448)

**Material Matching API**
- `viewer/material_matching.py` - `MaterialMatcher` class for spectral matching against reference libraries
- Matching methods: SAM (spectral angle), correlation, euclidean distance
- Automatic wavelength resampling between data and library grids

**Indoor Spectral Indices**
- Added 9 indoor-specific indices to `viewer/constants.py`:
  - Chlorophyll (680/750nm)
  - Moisture VNIR (970nm) and SWIR (1430nm)
  - Protein (1500nm)
  - Lipid 2nd overtone (1210nm) and edge (1690nm)
  - Cellulose (1490nm)
  - Polymer (1210/1500nm ratio)
  - Aromatic (1670nm)
- Each index includes wavelength range validation for D1/D2 compatibility

**Performance**
- Added `preload()` / `get_full_cube()` methods to both loaders for fast interactive access
- 23x speedup for pixel access (145ms → 6ms) when cube is preloaded into RAM

### Changed

- `viewer/__init__.py` - Exposed new classes: `LazyH5Data`, `MaterialMatcher`, `quick_match`
- `viewer/constants.py` - Added `INDOOR_INDEX_DEFINITIONS` and `INDOOR_MATERIAL_SIGNATURES` with verified absorption band positions from Wilson et al. 2015

### Technical Notes

**HyperspecI Dataset Limitations**
- Data is SRNet-reconstructed from 16 broadband filters, not raw sensor output
- Known artifacts: spectral smoothing, metamerism, boundary hallucination
- D2 range (400-1700nm) excludes primary lipid absorption at 1730nm
- Material matching returns similarity scores, not definitive identifications

**Wavelength Coverage**
| Index | D1 (400-1000nm) | D2 (400-1700nm) |
|-------|-----------------|-----------------|
| Chlorophyll | Full | Full |
| Moisture VNIR | Edge | Full |
| Moisture SWIR | Out | Full |
| Protein | Out | Full |
| Lipid (1730nm) | Out | Out |

---

## [1.0.0] - 2026-02-02

### Initial Release

- AVIRIS-3 L1B/L2 NetCDF support
- Atmospheric correction pipeline (Py6S, ISOFIT, empirical)
- Napari-based interactive viewer
- 23 spectral indices (vegetation, minerals, hydrocarbons)
- ROI spectral analysis with peak matching
- Learning suite (atmospheric/sensor simulation)
- CLI tools: `aviris-view`, `aviris-process`, `aviris-config`
