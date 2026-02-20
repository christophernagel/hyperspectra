# Hyperspectra
<img width="2894" height="2000" alt="Screenshot 2026-02-02 133817" src="https://github.com/user-attachments/assets/07d8c14f-775a-42ec-b09e-2793c3be1436" />

A desktop application for processing, visualizing, and analyzing airborne hyperspectral imagery. Built on Napari with a custom Qt interface for interactive spectral exploration of AVIRIS-3 data.


---

## What This Does

Hyperspectra processes AVIRIS-3 Level 1B radiance and observation data into Level 2 surface reflectance with uncertainty classification, then provides an interactive environment for exploring the imagery through RGB composites, spectral band indices, ROI analysis, and a learning suite grounded in radiative transfer physics.

The tool covers the full signal chain: from raw at-sensor radiance through atmospheric correction to material identification — with each step inspectable and interactive.

---

## Who This Is For

Researchers, students, and practitioners interested in:

- Understanding hyperspectral imagery and how spectral information encodes material properties
- Learning the radiometric processing chain from source radiance to surface reflectance
- Exploring spectral indices for vegetation health, mineral identification, and material detection
- Performing ROI-based spectral analysis with interactive signature extraction
- Investigating the physics of atmospheric interference and correction methods

---

## Features

### Atmospheric Correction Pipeline

Converts L1B radiance to L2 surface reflectance using observation geometry from AVIRIS-3 OBS files.

| Method | Description | Use Case |
|--------|-------------|----------|
| **Empirical (Simple)** | Fast band-ratio correction with solar angle normalization | VNIR indices, previews |
| **Py6S Radiative Transfer** | Scene-average correction using 6S atmospheric model | Research-grade reflectance |
| **ISOFIT Optimal Estimation** | Per-pixel atmospheric state retrieval via Bayesian inversion | Highest accuracy |

The processor performs scene-based water vapor retrieval from the 940nm absorption band, estimates aerosol optical depth from dark targets, and outputs reflectance with per-pixel uncertainty classification.

```
Processing Chain:
L1B At-Sensor Radiance → Atmospheric Correction → L2 Surface Reflectance
```

**Usage:**
```bash
# Basic
aviris-process radiance.nc obs.nc output.nc

# With options
aviris-process radiance.nc obs.nc output.nc --aerosol maritime --verbose
```

---

### Interactive Viewer
<img width="2917" height="2009" alt="Screenshot 2026-02-02 134328" src="https://github.com/user-attachments/assets/a44d6e64-dced-497c-b934-8287f3bb8816" />

Napari-based image viewer with controls for hyperspectral workflows.

**RGB Composites:**

| Composite | R / G / B (nm) | Purpose |
|-----------|----------------|---------|
| True Color | 640 / 550 / 470 | Natural appearance |
| Color Infrared | 860 / 650 / 550 | Vegetation (appears red) |
| SWIR Geology | 2200 / 1650 / 850 | Mineral mapping |
| Vegetation Stress | 750 / 705 / 550 | Red edge chlorophyll sensitivity |
| Urban | 2200 / 860 / 650 | Built environment |

Custom composites from any three wavelengths in the 380–2500nm range.

**Spectral Indices (23 pre-defined):**
<img width="2912" height="1955" alt="Screenshot 2026-02-02 134602" src="https://github.com/user-attachments/assets/093baa47-8e01-47d1-9f7d-ad5218b02d48" />

| Category | Indices |
|----------|---------|
| Vegetation | NDVI, NDRE, NDWI, NDMI |
| Clay Minerals | Clay, Kaolinite, Alunite, Smectite, Illite |
| Carbonates | Carbonate, Calcite, Dolomite, Chlorite |
| Iron Oxides | Iron Oxide, Ferric Iron, Ferrous Iron |
| Hydrocarbons | Hydrocarbon, HC 2310, Methane, Oil Slick |
| Agriculture | Protein, Cellulose, Lignin |

Custom index calculator supports any two wavelengths with ratio or normalized difference formulas.

---

### ROI Spectral Analysis
<img width="2916" height="2010" alt="Screenshot 2026-02-02 134910" src="https://github.com/user-attachments/assets/15e791ad-4458-4ccf-bea9-295b6e80a87b" />

Draw rectangle, polygon, or ellipse regions on any layer. Extracts full 284-band spectrum with mean ± standard deviation. Multiple ROIs can be compared on the same plot.

**Peak Matching (14 reference materials):**
- Clay: Kaolinite, Montmorillonite, Illite
- Carbonate: Calcite, Dolomite, Gypsum
- Iron: Hematite, Goethite, Jarosite
- Organic: Chlorophyll, Dry Vegetation
- Hydrocarbon: Crude Oil, Asphalt, Plastics

Export spectra to CSV for external analysis.

---

### Learning Suite

Interactive walkthrough of the electromagnetic signal chain from source to sensor:

1. **Solar Irradiance** — Source spectrum and 5778K blackbody characteristics
2. **Surface Interaction** — Electronic transitions (VNIR), vibrational overtones (SWIR)
3. **Atmospheric Interference** — Rayleigh scattering, Mie/aerosol scattering, molecular absorption (H₂O, CO₂, O₃, O₂)
4. **Sensor Response** — Pushbroom geometry, grating dispersion, detector noise sources
5. **Radiative Transfer** — Forward model and the inverse problem of atmospheric correction

The learning suite includes:
- **Atmospheric simulator** with adjustable water vapor, aerosol optical depth, solar/view geometry
- **Sensor simulator** with grating physics, quantum efficiency, shot/read/dark noise
- **Forward model** connecting surface reflectance to at-sensor digital numbers
- **Interactive dashboard** for parameter exploration (requires Dash)

```python
from hsi_toolkit import ForwardModel, SceneParameters, quick_demo

# Run demonstration
quick_demo()

# Manual simulation
model = ForwardModel()
targets = model.generate_test_targets()

scene = SceneParameters(
    surface_reflectance=targets['vegetation'],
    solar_zenith_deg=30,
    pwv_cm=1.5,
    aod_550=0.1
)
result = model.simulate(scene, add_noise=True)

# Educational content
print(model.explain_forward_model())
```

Launch interactive dashboard:
```bash
python -c "from hsi_toolkit import launch_dashboard; launch_dashboard()"
```

---

## Data Requirements

**Input:** AVIRIS-3 Level 1B data from [NASA/JPL AVIRIS Data Portal](https://aviris.jpl.nasa.gov/)

| File | Description | Required |
|------|-------------|----------|
| `*_L1B_RDN_*.nc` | At-sensor radiance (284 bands, 380–2500nm) | Yes |
| `*_L1B_OBS_*.nc` | Observation geometry (solar/sensor angles, elevation) | For atmospheric correction |

**Coverage:** US regions with AVIRIS-3 flight lines. Tested on Santa Barbara coastal, Cuprite NV (mineral validation site), Southern California.

**USGS Spectral Library (optional):** For material matching, download [USGS Spectral Library Version 7](https://crustal.usgs.gov/speclab/QueryAll07a.php) and place it in `aviris_tools/reference_spectra/usgs_v7/`. This is not included in the repository due to size.

---

## Installation

**Requirements:** Python 3.9+

### Option A: Conda (recommended — includes viewer, Py6S, and all dependencies)

```bash
git clone https://github.com/christophernagel/hyperspectra.git
cd hyperspectra
conda env create -f environment.yml
conda activate hyperspectra
pip install -e .
```

For Py6S atmospheric correction, also install the 6S executable:
```bash
conda install -c conda-forge sixs
```

### Option B: Pip (lightweight — core only, add extras as needed)

```bash
git clone https://github.com/christophernagel/hyperspectra.git
cd hyperspectra
pip install -e .
```

Extras:
```bash
pip install -e ".[viewer]"      # Napari viewer + PyQt5
pip install -e ".[py6s]"        # Py6S atmospheric correction
pip install -e ".[dashboard]"   # Dash/Plotly interactive dashboard
pip install -e ".[all]"         # Everything
```

---

## CLI Commands

### `aviris-view` — Hyperspectral Viewer
```bash
aviris-view [filepath] [--scale FACTOR]
```

### `aviris-process` — Atmospheric Correction
```bash
aviris-process <radiance.nc> <obs.nc> <output.nc> [options]

Options:
  --method, -m      py6s (default) or isofit
  --aerosol, -a     maritime, continental, urban, desert
  --simple          Use simplified empirical method
  --verbose, -v     Verbose output
```

### `aviris-config` — Configuration
```bash
aviris-config --check   # Check dependencies
aviris-config --show    # Show current config
```

---

## Project Structure

```
hyperspectra/
├── aviris_tools/                    # Main package
│   ├── cli.py                       # Command line interface
│   ├── atm_correction/              # Atmospheric correction
│   │   ├── py6s_processor.py        # Py6S radiative transfer
│   │   └── isofit_processor.py      # ISOFIT optimal estimation
│   ├── indices/                     # Spectral indices
│   │   ├── vegetation.py            # NDVI, NDRE, etc.
│   │   ├── minerals.py              # Clay, carbonate, iron
│   │   ├── hydrocarbons.py          # Hydrocarbon detection
│   │   └── spectral_library.py      # Reference spectra
│   ├── viewer/                      # Viewer components
│   │   ├── app.py                   # HyperspectralViewer class
│   │   ├── data_loader.py           # Lazy NetCDF loading
│   │   └── constants.py             # Spectral features, references
│   ├── utils/                       # Utilities
│   │   ├── envi_io.py               # ENVI format I/O
│   │   ├── memory.py                # Memory management
│   │   └── config.py                # Configuration
│   ├── reference_spectra/           # Spectral reference data
│   └── tests/                       # Test suite
│
├── hsi_toolkit/                     # Learning suite (independent package)
│   ├── atmosphere/                  # Atmospheric RT simulation
│   ├── sensor/                      # Sensor physics
│   ├── forward_model/               # Imaging chain
│   └── visualization/               # Interactive tools
│
├── legacy_scripts/                  # Standalone processing scripts
│   ├── aviris_atm_correction_v2.py  # L1B → L2 correction
│   ├── aviris_isofit_processor.py   # ISOFIT processing
│   └── hyperspectral_viewer_v4.py   # Original Napari viewer
│
├── pyproject.toml
├── environment.yml
└── requirements.txt
```

---

## Limitations

- **SWIR mineral indices require reflectance.** Running on radiance produces washed-out results due to atmospheric absorption in the 2000–2500nm region.
- **Memory scales with scene size.** Lazy loading helps, but large flight lines can require 8GB+ RAM.
- **No georeferencing.** Operates in pixel coordinates.
- **Single-scene processing.** Each file processed independently.

---

## Atmospheric Bands

The viewer flags wavelengths in atmospheric absorption regions:

| Region | Wavelength | Severity |
|--------|------------|----------|
| O₂ A-band | 760–770 nm | Moderate |
| H₂O | 1350–1450 nm | Severe |
| H₂O | 1800–1950 nm | Severe |
| CO₂ | 2500–2600 nm | Moderate |

Avoid using these wavelengths for indices.

---

## Background

This project began as a learning exercise to understand the full signal chain of imaging spectroscopy — from the physical resonance of materials through atmospheric interference to sensor capture and data processing. The atmospheric correction pipeline was built from radiative transfer principles rather than relying on black-box solutions, with the goal of understanding each step of the radiance-to-reflectance conversion rather than simply executing it.

---

## Acknowledgments & References

### Software

| Component | Software | Reference |
|-----------|----------|-----------|
| Atmospheric Correction | [Py6S](https://py6s.readthedocs.io/) | Wilson, R.T. (2013). Py6S: A Python interface to the 6S radiative transfer model. *Computers & Geosciences*, 51, 166-171. |
| Radiative Transfer | [6S](http://6s.ltdri.org/) | Vermote, E.F., et al. (1997). Second Simulation of the Satellite Signal in the Solar Spectrum (6S). *IEEE TGRS*, 35(3), 675-686. |
| Optimal Estimation | [ISOFIT](https://github.com/isofit/isofit) | Thompson, D.R., et al. (2018). Optimal estimation for imaging spectrometer atmospheric correction. *Remote Sensing of Environment*, 216, 355-373. |
| Viewer | [Napari](https://napari.org/) | Napari contributors (2019). napari: a multi-dimensional image viewer for Python. |

### Scientific Foundations

**Atmospheric Physics (hsi_toolkit):**
- Rayleigh scattering: Bucholtz, A. (1995). Rayleigh-scattering calculations for the terrestrial atmosphere. *Applied Optics*, 34(15), 2765-2773.
- Gas absorption: HITRAN molecular spectroscopic database. Rothman, L.S., et al. (2013). *JQSRT*, 130, 4-50.
- Aerosol models: Shettle, E.P. & Fenn, R.W. (1979). Models for the aerosols of the lower atmosphere. *AFGL-TR-79-0214*.
- Solar spectrum: Gueymard, C.A. (2004). The sun's total and spectral irradiance. *Solar Energy*, 76(4), 423-453.

**Sensor Physics (hsi_toolkit):**
- Detector noise: Janesick, J.R. (2001). *Scientific Charge-Coupled Devices*. SPIE Press.
- Grating dispersion: Palmer, C. & Loewen, E. (2005). *Diffraction Grating Handbook*. Newport Corporation.
- Pushbroom imaging: Mouroulis, P. & Green, R.O. (2018). Review of high fidelity imaging spectrometer design. *Optical Engineering*, 57(4).

**Spectral Indices:**
- NDVI: Rouse, J.W., et al. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *NASA SP-351*, 309-317.
- Mineral indices: Clark, R.N., et al. (1990). High spectral resolution reflectance spectroscopy of minerals. *JGR*, 95(B8), 12653-12680.
- USGS Spectral Library: Kokaly, R.F., et al. (2017). USGS Spectral Library Version 7. *USGS Data Series 1035*.


---

## License

MIT License — see LICENSE file.

## Author

Christopher Nagel
