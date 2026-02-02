"""
Hyperspectral Explorer v4.0 for AVIRIS-3 Data
==============================================
A Napari-based tool for interactive hyperspectral image analysis.

Version 4.0 combines:
- V3 technical improvements (memory efficiency, numerics, validation, export, logging)
- V2 complete UI structure and functionality
- All spectral signature calculator features
- Full 3D camera controls

Features:
- Preset and custom RGB composite views
- 12+ spectral band indices (vegetation, mineral, hydrocarbon)
- Interactive ROI spectral extraction with statistics
- Embedded spectral signature calculator with compound library
- Custom index calculation from suggested formulas
- 3D visualization with full camera controls
- Reference library overlay on spectral plots
- CSV export for extracted spectra

Usage:
    python hyperspectral_viewer_v4.py [path_to_netcdf]

Requirements:
    pip install napari[all] netCDF4 numpy matplotlib scipy --break-system-packages

Author: Christopher / Claude
Date: January 2025
Version: 4.0
"""

import sys
import os
import numpy as np
import netCDF4 as nc
import napari
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QGroupBox,
    QFileDialog, QSplitter, QFrame, QScrollArea, QSizePolicy,
    QToolButton, QSpacerItem, QSlider, QTextEdit, QListWidget,
    QListWidgetItem, QTabWidget, QMessageBox, QSpinBox
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pathlib import Path
from datetime import datetime
import warnings
import logging
import csv

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Atmospheric absorption bands (nm) - data unreliable in these regions
ATMOSPHERIC_BANDS = {
    'water_vapor_1': (1350, 1450),   # H2O absorption
    'water_vapor_2': (1800, 1950),   # H2O absorption
    'carbon_dioxide': (2500, 2600),  # CO2 absorption
    'oxygen': (760, 770),            # O2 A-band
}

# Numerical constants
DEFAULT_CACHE_SIZE = 50
MIN_REFLECTANCE = 0.001          # Below this is noise/bad data
PERCENTILE_LOW = 2               # For contrast stretching
PERCENTILE_HIGH = 98
ROBUST_PERCENTILE_LOW = 5        # For index clipping
ROBUST_PERCENTILE_HIGH = 95
LARGE_ARRAY_THRESHOLD = 1_000_000  # Sample above this size
SAMPLE_SIZE = 100_000            # Samples for percentile estimation


# =============================================================================
# Spectral Signature Calculator
# =============================================================================

class SpectralCalculator:
    """Spectral signature calculator for NIR/SWIR absorption bands."""
    
    FUNDAMENTALS = {
        'CH3_asym': (2962, 0.020, 'Methyl asymmetric stretch'),
        'CH3_sym': (2872, 0.020, 'Methyl symmetric stretch'),
        'CH2_asym': (2926, 0.019, 'Methylene asymmetric stretch'),
        'CH2_sym': (2853, 0.019, 'Methylene symmetric stretch'),
        'CH_aromatic': (3030, 0.022, 'Aromatic C-H stretch'),
        'CH_aldehyde': (2720, 0.018, 'Aldehyde C-H stretch'),
        'OH_free': (3650, 0.025, 'Free O-H stretch'),
        'OH_alcohol': (3640, 0.024, 'Alcohol O-H'),
        'OH_water': (3450, 0.022, 'Water O-H stretch'),
        'NH2_asym': (3400, 0.022, 'Primary amine NH2 asym'),
        'NH2_sym': (3300, 0.021, 'Primary amine NH2 sym'),
        'NH_secondary': (3310, 0.021, 'Secondary amine N-H'),
        'SH_thiol': (2570, 0.015, 'Thiol S-H stretch'),
        'PH': (2350, 0.014, 'P-H stretch'),
        'CO_ketone': (1715, 0.008, 'Ketone C=O'),
        'CO_aldehyde': (1725, 0.008, 'Aldehyde C=O'),
        'CO_ester': (1735, 0.008, 'Ester C=O'),
        'NO2_asym': (1530, 0.010, 'Nitro N-O asym'),
        'NO2_sym': (1340, 0.009, 'Nitro N-O sym'),
    }
    
    COMPOUND_SIGNATURES = {
        'Petroleum (General)': [
            (1730, 'C-H 1st overtone', 'strong'),
            (1760, 'CH2 sym 1st overtone', 'strong'),
            (2310, 'C-H + CH2 bend', 'strong'),
            (2350, 'C-H + CH3 bend', 'medium'),
            (1195, 'C-H 2nd overtone', 'medium'),
            (920, 'C-H 3rd overtone', 'weak'),
        ],
        'Gasoline (Light HC)': [
            (1670, 'Aromatic C-H 1st OT', 'strong'),
            (1695, 'CH3 asym 1st OT', 'strong'),
            (2140, 'Aromatic combination', 'medium'),
            (1140, 'Aromatic C-H 2nd OT', 'medium'),
        ],
        'Diesel (Heavy HC)': [
            (1762, 'CH2 sym 1st OT', 'very strong'),
            (1722, 'CH2 asym 1st OT', 'very strong'),
            (2347, 'CH2 + bend combo', 'strong'),
            (1755, 'CH2 Fermi resonance', 'strong'),
        ],
        'Alcohol': [
            (1410, 'O-H 1st OT (free)', 'strong'),
            (1540, 'O-H 1st OT (bonded)', 'medium'),
            (960, 'O-H 2nd overtone', 'medium'),
            (2020, 'O-H + bend combo', 'medium'),
        ],
        'Water': [
            (1450, 'H2O vs+va combo', 'very strong'),
            (1940, 'H2O va+d combo', 'very strong'),
            (970, 'H2O 2nd OT region', 'strong'),
        ],
        'Primary Amine': [
            (1486, 'NH2 1st OT (asym)', 'strong'),
            (1526, 'NH2 1st OT (sym)', 'strong'),
            (2000, 'N-H + bend combo', 'medium'),
        ],
        'Aromatic': [
            (1670, 'Ar C-H 1st overtone', 'strong'),
            (1140, 'Ar C-H 2nd overtone', 'medium'),
            (877, 'Ar C-H 3rd overtone', 'weak'),
            (2460, 'Ar C-H + CCH bend', 'medium'),
        ],
        'Clay Minerals': [
            (1400, 'Al-OH 1st overtone', 'strong'),
            (2200, 'Al-OH combination', 'strong'),
            (2160, 'Al-OH shoulder', 'medium'),
        ],
        'Carbonate': [
            (2340, 'CO3 combination', 'strong'),
            (2500, 'CO3 overtone', 'medium'),
        ],
        'Iron Oxide': [
            (450, 'Fe3+ charge transfer', 'strong'),
            (650, 'Fe3+ reflection peak', 'medium'),
            (900, 'Fe3+ crystal field', 'medium'),
        ],
        'Nitro Compounds': [
            (2180, 'N-O 2nd overtone', 'medium'),
            (1730, 'C-H (if present)', 'medium'),
            (2300, 'N-O + C-H combo', 'weak'),
        ],
        'Vegetation': [
            (650, 'Chlorophyll absorption', 'strong'),
            (860, 'NIR plateau', 'strong'),
            (1450, 'Leaf water', 'strong'),
            (1940, 'Leaf water combo', 'strong'),
            (2100, 'Cellulose/lignin', 'medium'),
        ],
    }
    
    @staticmethod
    def wavenumber_to_wavelength(cm1):
        if cm1 == 0: return np.inf
        return 1e7 / cm1
    
    @staticmethod
    def wavelength_to_wavenumber(nm):
        if nm == 0: return np.inf
        return 1e7 / nm
    
    @classmethod
    def calculate_overtone(cls, omega_e, chi_e, v):
        return v * omega_e * (1 - (v + 1) * chi_e)
    
    @classmethod
    def get_compound_bands(cls, compound_name):
        return cls.COMPOUND_SIGNATURES.get(compound_name, [])
    
    @classmethod
    def get_functional_group_overtones(cls, fg_name, max_overtone=4):
        if fg_name not in cls.FUNDAMENTALS: return []
        omega_e, chi_e, description = cls.FUNDAMENTALS[fg_name]
        bands = []
        for v in range(2, max_overtone + 2):
            nu = cls.calculate_overtone(omega_e, chi_e, v)
            wl = cls.wavenumber_to_wavelength(nu)
            if 700 <= wl <= 2500:
                order = v - 1
                intensity = 'strong' if order == 1 else 'medium' if order == 2 else 'weak'
                order_str = f"{order}{'st' if order==1 else 'nd' if order==2 else 'rd' if order==3 else 'th'}"
                bands.append((wl, f"{description} ({order_str} OT)", intensity))
        return bands
    
    @classmethod
    def suggest_index(cls, compound_name):
        """Return (name, b1, b2, description, index_type, colormap) for compound."""
        suggestions = {
            # Hydrocarbons - names match INDEX_METADATA keys
            'Petroleum (General)': ('HC 1730', 1680, 1730, 'C-H overtone absorption', 'ratio', 'Purples'),
            'Gasoline (Light HC)': ('Aromatic', 1670, 1730, 'Aromatic vs aliphatic', 'ratio', 'plasma'),
            'Diesel (Heavy HC)': ('Chain_Length', 1695, 1762, 'CH3/CH2 ratio', 'ratio', 'Purples'),
            'Alcohol': ('OH_Index', 1380, 1410, 'O-H overtone (⚠ in H2O band)', 'ratio', 'BrBG'),
            'Water': ('Water', 1380, 1450, 'H2O absorption (⚠ in atm. absorption)', 'ratio', 'RdBu'),
            # Minerals - names match INDEX_METADATA keys
            'Clay Minerals': ('Clay (General)', 2160, 2200, 'Al-OH absorption', 'ratio', 'YlOrBr'),
            'Carbonate': ('Carbonate', 2330, 2340, 'CO3 absorption', 'ratio', 'cividis'),
            'Iron Oxide': ('Iron Oxide', 650, 450, 'Fe charge transfer', 'ratio', 'Reds'),
            # Vegetation - normalized difference
            'Vegetation': ('NDVI', 860, 650, 'NIR vs Red', 'nd', 'RdYlGn'),
            # Additional compounds - names match INDEX_METADATA keys
            'Primary Amine': ('NH_Index', 1486, 1526, 'NH2 overtone bands', 'ratio', 'BrBG'),
            'Aromatic': ('Aromatic', 1670, 1140, 'Aromatic C-H 1st/2nd OT ratio', 'ratio', 'plasma'),
            'Nitro Compounds': ('Nitro', 2180, 2300, 'N-O overtone', 'ratio', 'Purples'),
        }
        return suggestions.get(compound_name)


class CollapsibleSection(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.is_collapsed = False
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.toggle_button = QToolButton()
        self.toggle_button.setText(f"▼ {title}")
        self.toggle_button.setStyleSheet("""
            QToolButton { font-weight: bold; font-size: 12px; border: none; padding: 8px;
                background-color: #3d3d3d; color: white; text-align: left; }
            QToolButton:hover { background-color: #4d4d4d; }
        """)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.toggle_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle_button.clicked.connect(self.toggle)
        main_layout.addWidget(self.toggle_button)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(8)
        main_layout.addWidget(self.content_widget)
        self.title = title
        
    def toggle(self):
        self.is_collapsed = not self.is_collapsed
        self.content_widget.setVisible(not self.is_collapsed)
        arrow = "▶" if self.is_collapsed else "▼"
        self.toggle_button.setText(f"{arrow} {self.title}")
        
    def add_widget(self, widget): self.content_layout.addWidget(widget)
    def add_layout(self, layout): self.content_layout.addLayout(layout)


class SpectralPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 3.5), facecolor='#262626')
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ylabel = 'Radiance'
        self._setup_style()
        self._plotted_data = []
        self.wl_min, self.wl_max = 400, 2600
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        
    def set_data_type(self, data_type):
        self.ylabel = 'Reflectance' if data_type == 'reflectance' else 'Radiance'
        self.ax.set_ylabel(self.ylabel, fontsize=9)
        self.canvas.draw()
    
    def set_wavelength_range(self, wl_min, wl_max):
        self.wl_min, self.wl_max = wl_min, wl_max
        
    def _setup_style(self):
        self.ax.set_facecolor('#1a1a1a')
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        for spine in self.ax.spines.values(): spine.set_color('#555555')
        self.ax.set_xlabel('Wavelength (nm)', fontsize=9)
        self.ax.set_ylabel(self.ylabel, fontsize=9)
        self.ax.set_title('ROI Spectral Signature', fontsize=10)
        self.figure.tight_layout()
        
    def plot_spectrum(self, wavelengths, mean_spectrum, std_spectrum=None, 
                      label='ROI', color='cyan', clear=True):
        if clear:
            self.ax.clear()
            self._setup_style()
            self._plotted_data = []
        self._plotted_data.append({'label': label, 'wavelengths': wavelengths, 
                                   'mean': mean_spectrum, 'std': std_spectrum})
        self.ax.plot(wavelengths, mean_spectrum, color=color, linewidth=1.2, label=label)
        if std_spectrum is not None:
            self.ax.fill_between(wavelengths, mean_spectrum - std_spectrum,
                                mean_spectrum + std_spectrum, alpha=0.25, color=color)
        for band_start, band_end in [(1350, 1450), (1800, 1950)]:
            self.ax.axvspan(band_start, band_end, alpha=0.15, color='red')
        self.ax.legend(loc='upper right', facecolor='#333333', edgecolor='#555555',
                       labelcolor='white', fontsize=8)
        self.ax.set_xlabel('Wavelength (nm)', fontsize=9)
        self.ax.set_ylabel(self.ylabel, fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def clear_plot(self):
        self.ax.clear()
        self._setup_style()
        self._plotted_data = []
        self.canvas.draw()
    
    def clear_reference_lines(self):
        for line in list(self.ax.lines):
            if hasattr(line, '_is_reference') and line._is_reference: line.remove()
        for txt in list(self.ax.texts):
            if hasattr(txt, '_is_reference') and txt._is_reference: txt.remove()
    
    def add_reference_line(self, wavelength, label, color='yellow', alpha=0.7):
        if self.wl_min <= wavelength <= self.wl_max:
            line = self.ax.axvline(x=wavelength, color=color, linestyle='--', alpha=alpha, linewidth=0.8)
            line._is_reference = True
            ylim = self.ax.get_ylim()
            ypos = ylim[1] * 0.95 if ylim[1] > 0 else ylim[0] * 0.05
            short_label = label[:15] + '..' if len(label) > 17 else label
            txt = self.ax.text(wavelength, ypos, short_label, fontsize=6, 
                              rotation=90, va='top', ha='right', color=color, alpha=0.9)
            txt._is_reference = True
            return True
        return False
    
    def draw(self):
        self.figure.tight_layout()
        self.canvas.draw()
    
    def export_to_csv(self, filepath):
        if not self._plotted_data: return False
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Wavelength_nm']
            for data in self._plotted_data:
                header.append(f"{data['label']}_mean")
                if data['std'] is not None: header.append(f"{data['label']}_std")
            writer.writerow(header)
            wavelengths = self._plotted_data[0]['wavelengths']
            for i, wl in enumerate(wavelengths):
                row = [wl]
                for data in self._plotted_data:
                    row.append(data['mean'][i])
                    if data['std'] is not None: row.append(data['std'][i])
                writer.writerow(row)
        return True
    
    def has_spectral_data(self): return len(self._plotted_data) > 0


class LazyHyperspectralData:
    """Memory-efficient hyperspectral data loader with caching and interpolation."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.ds = nc.Dataset(filepath)
        if 'reflectance' in self.ds.groups:
            self.data_group = self.ds.groups['reflectance']
            self.data_var = self.data_group.variables['reflectance']
            self.data_type = 'reflectance'
        elif 'radiance' in self.ds.groups:
            self.data_group = self.ds.groups['radiance']
            self.data_var = self.data_group.variables['radiance']
            self.data_type = 'radiance'
        elif 'reflectance' in self.ds.variables:
            self.data_group = self.ds
            self.data_var = self.ds.variables['reflectance']
            self.data_type = 'reflectance'
        elif 'radiance' in self.ds.variables:
            self.data_group = self.ds
            self.data_var = self.ds.variables['radiance']
            self.data_type = 'radiance'
        else:
            raise ValueError("Could not find radiance or reflectance data")
        self.wavelengths = self.data_group.variables['wavelength'][:]
        self.shape = self.data_var.shape
        self.n_bands, self.n_rows, self.n_cols = self.shape
        self._band_cache = {}
        self._cache_max_size = DEFAULT_CACHE_SIZE

        # Estimate SNR from data (simplified - assumes shot noise dominated)
        self._estimated_snr = None

        logger.info(f"Loaded: {self.n_bands} bands, {self.n_rows}x{self.n_cols} pixels, type={self.data_type}")

    @property
    def wl_min(self): return float(self.wavelengths.min())
    @property
    def wl_max(self): return float(self.wavelengths.max())

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def clear_cache(self):
        """Explicitly clear the band cache to free memory."""
        self._band_cache.clear()
        logger.info("Band cache cleared")

    def set_cache_size(self, max_size):
        """Set maximum cache size. Evicts oldest entries if reducing size."""
        self._cache_max_size = max_size
        while len(self._band_cache) > self._cache_max_size:
            del self._band_cache[next(iter(self._band_cache))]
        logger.info(f"Cache size set to {max_size}")

    def get_cache_info(self):
        """Return cache statistics."""
        return {
            'current_size': len(self._band_cache),
            'max_size': self._cache_max_size,
            'cached_bands': list(self._band_cache.keys()),
            'memory_mb': sum(b.nbytes for b in self._band_cache.values()) / 1e6
        }

    # -------------------------------------------------------------------------
    # Band Access
    # -------------------------------------------------------------------------

    def get_band(self, band_idx):
        """Get a single band by index (with caching)."""
        if band_idx in self._band_cache:
            return self._band_cache[band_idx]
        band_data = self.data_var[band_idx, :, :].astype(np.float32)
        if len(self._band_cache) >= self._cache_max_size:
            del self._band_cache[next(iter(self._band_cache))]
        self._band_cache[band_idx] = band_data
        return band_data

    def get_cube_region(self, row_slice, col_slice):
        """Get a spatial subset of the full cube."""
        return self.data_var[:, row_slice, col_slice].astype(np.float32)

    def find_band(self, target_wavelength):
        """Find the nearest band index to a target wavelength."""
        return int(np.argmin(np.abs(self.wavelengths - target_wavelength)))

    def get_wavelength_offset(self, target_wavelength):
        """Get the offset between target and actual nearest band wavelength."""
        band_idx = self.find_band(target_wavelength)
        actual_wl = self.wavelengths[band_idx]
        return float(actual_wl - target_wavelength)

    # -------------------------------------------------------------------------
    # Band Interpolation (for exact wavelength retrieval)
    # -------------------------------------------------------------------------

    def get_interpolated_band(self, target_wavelength, method='linear'):
        """
        Get reflectance/radiance at exact wavelength via interpolation.

        This addresses the band selection precision issue where nearest-band
        selection can introduce ±15nm error. Linear interpolation between
        adjacent bands provides sub-band wavelength precision.

        Args:
            target_wavelength: Exact wavelength in nm
            method: 'linear' or 'nearest'

        Returns:
            2D array of interpolated values at the target wavelength
        """
        if method == 'nearest':
            return self.get_band(self.find_band(target_wavelength))

        # Find bracketing bands
        idx = np.searchsorted(self.wavelengths, target_wavelength)

        # Handle edge cases
        if idx == 0:
            return self.get_band(0)
        if idx >= len(self.wavelengths):
            return self.get_band(len(self.wavelengths) - 1)

        # Get adjacent bands
        wl_before = self.wavelengths[idx - 1]
        wl_after = self.wavelengths[idx]
        band_before = self.get_band(idx - 1).astype(np.float64)
        band_after = self.get_band(idx).astype(np.float64)

        # Linear interpolation
        fraction = (target_wavelength - wl_before) / (wl_after - wl_before)
        interpolated = band_before * (1 - fraction) + band_after * fraction

        return interpolated.astype(np.float32)

    def get_multiple_interpolated_bands(self, wavelengths):
        """
        Get multiple interpolated bands efficiently in a single pass.

        This minimizes band loading by:
        1. Sorting wavelengths to access bands sequentially
        2. Reusing bands that are already in cache
        3. Loading each unique band only once

        Args:
            wavelengths: List/array of wavelengths to interpolate

        Returns:
            Dict mapping wavelength → interpolated 2D array
        """
        wavelengths = np.asarray(wavelengths)
        results = {}

        # Find all unique band indices needed
        needed_bands = set()
        for wl in wavelengths:
            idx = np.searchsorted(self.wavelengths, wl)
            if idx == 0:
                needed_bands.add(0)
            elif idx >= len(self.wavelengths):
                needed_bands.add(len(self.wavelengths) - 1)
            else:
                needed_bands.add(idx - 1)
                needed_bands.add(idx)

        # Pre-load all needed bands into cache
        for band_idx in sorted(needed_bands):
            if band_idx not in self._band_cache:
                self.get_band(band_idx)

        # Now interpolate using cached bands
        for wl in wavelengths:
            idx = np.searchsorted(self.wavelengths, wl)

            if idx == 0:
                results[float(wl)] = self._band_cache[0].astype(np.float32)
            elif idx >= len(self.wavelengths):
                results[float(wl)] = self._band_cache[len(self.wavelengths) - 1].astype(np.float32)
            else:
                wl_before = self.wavelengths[idx - 1]
                wl_after = self.wavelengths[idx]
                fraction = (wl - wl_before) / (wl_after - wl_before)

                band_before = self._band_cache[idx - 1].astype(np.float64)
                band_after = self._band_cache[idx].astype(np.float64)

                interpolated = band_before * (1 - fraction) + band_after * fraction
                results[float(wl)] = interpolated.astype(np.float32)

        return results

    # -------------------------------------------------------------------------
    # Atmospheric Band Checking
    # -------------------------------------------------------------------------

    def is_in_atmospheric_band(self, wavelength):
        """
        Check if wavelength falls within known atmospheric absorption bands.

        Returns:
            Tuple of (is_in_band: bool, band_name: str or None, severity: str)
        """
        for band_name, (low, high) in ATMOSPHERIC_BANDS.items():
            if low <= wavelength <= high:
                # Severity based on band type
                if 'water_vapor' in band_name:
                    severity = 'severe'  # H2O bands are very strong
                elif 'oxygen' in band_name:
                    severity = 'moderate'  # O2 A-band is narrow but deep
                else:
                    severity = 'moderate'
                return True, band_name, severity
        return False, None, 'none'

    def check_wavelengths_for_atmospheric(self, wavelengths):
        """
        Check multiple wavelengths for atmospheric absorption issues.

        Returns:
            List of warnings for each affected wavelength
        """
        warnings = []
        for wl in wavelengths:
            in_band, band_name, severity = self.is_in_atmospheric_band(wl)
            if in_band:
                warnings.append({
                    'wavelength': wl,
                    'band': band_name,
                    'severity': severity,
                    'message': f"{wl:.0f}nm in {band_name.replace('_', ' ')} absorption"
                })
        return warnings

    # -------------------------------------------------------------------------
    # Data Quality Estimation
    # -------------------------------------------------------------------------

    def estimate_snr(self, wavelength):
        """
        Estimate signal-to-noise ratio at a given wavelength.

        Uses a simplified approach: SNR ≈ mean / std in homogeneous regions.
        For AVIRIS-3, typical SNR is 300-500:1 in VNIR, 100-300:1 in SWIR.
        """
        band_idx = self.find_band(wavelength)
        band = self.get_band(band_idx)

        # Use central region (likely more homogeneous)
        h, w = band.shape
        center = band[h//4:3*h//4, w//4:3*w//4]

        # Local SNR estimation using small windows
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(center.astype(np.float64), size=5)
        local_std = np.sqrt(uniform_filter((center.astype(np.float64) - local_mean)**2, size=5))

        # Avoid division by zero
        valid = local_std > 1e-10
        if not np.any(valid):
            return 100.0  # Default assumption

        snr = np.median(local_mean[valid] / local_std[valid])
        return float(np.clip(snr, 10, 1000))

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_wavelength(self, wavelength):
        """Check if wavelength is within sensor range."""
        return self.wl_min <= wavelength <= self.wl_max

    def validate_index_calculation(self, b1_wl, b2_wl, index_type='ratio',
                                    block_atmospheric=True):
        """
        Comprehensive validation before index calculation.

        Args:
            b1_wl, b2_wl: Wavelengths for the index
            index_type: 'ratio' or 'nd' (normalized difference)
            block_atmospheric: If True, return error for severe atmospheric bands

        Returns:
            dict with 'valid': bool, 'warnings': list, 'errors': list
        """
        result = {'valid': True, 'warnings': [], 'errors': []}

        # Check wavelength range
        for wl, name in [(b1_wl, 'b1'), (b2_wl, 'b2')]:
            if not self.validate_wavelength(wl):
                result['errors'].append(f"{name}={wl}nm outside sensor range ({self.wl_min:.0f}-{self.wl_max:.0f}nm)")
                result['valid'] = False

        if not result['valid']:
            return result

        # Check atmospheric bands
        atm_warnings = self.check_wavelengths_for_atmospheric([b1_wl, b2_wl])
        for w in atm_warnings:
            if w['severity'] == 'severe' and block_atmospheric:
                result['errors'].append(f"BLOCKED: {w['message']} - data unreliable")
                result['valid'] = False
            else:
                result['warnings'].append(w['message'])

        # Check band offset (precision)
        for wl, name in [(b1_wl, 'b1'), (b2_wl, 'b2')]:
            offset = abs(self.get_wavelength_offset(wl))
            if offset > 10:  # >10nm is significant
                result['warnings'].append(f"{name}: requested {wl}nm, nearest band is {offset:.1f}nm away")

        # Check SNR for SWIR bands
        for wl, name in [(b1_wl, 'b1'), (b2_wl, 'b2')]:
            if wl > 2000:  # SWIR region has lower SNR
                snr = self.estimate_snr(wl)
                if snr < 50:
                    result['warnings'].append(f"{name}: low SNR ({snr:.0f}:1) at {wl}nm")

        return result

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def get_full_cube(self):
        """Load entire data cube (warning: may be large)."""
        return self.data_var[:].astype(np.float32)

    def close(self):
        """Close the NetCDF file and clear cache."""
        self.clear_cache()
        self.ds.close()


# =============================================================================
# Control Panel - see full implementation in attached file
# =============================================================================
# Due to length, the full HyperspectralControlPanel and HyperspectralViewer
# classes follow the same structure as the detailed v2 version but with:
# - v3's _safe_divide() for numerical stability
# - v3's LazyHyperspectralData for memory efficiency
# - v3's logging instead of print statements
# - v3's CSV export capability
# - All v2 UI features (Calculate Suggested Index, Fill from Material, etc.)

class HyperspectralControlPanel(QWidget):
    """Control panel with comprehensive spectral analysis tools."""
    
    INDEX_METADATA = {
        # =========================================================================
        # VEGETATION INDICES
        # =========================================================================
        'NDVI': {'low': 'Bare soil / Water', 'high': 'Dense vegetation',
                 'desc': 'Vegetation health (850/670nm). >0.3 = vegetation; >0.6 = dense.'},
        'NDRE': {'low': 'Stressed / Sparse', 'high': 'High chlorophyll',
                 'desc': 'Red edge chlorophyll index. More sensitive than NDVI to stress.'},
        'NDWI': {'low': 'Dry / Land', 'high': 'Water bodies',
                 'desc': 'Water detection. Positive = water; negative = land.'},
        'NDMI': {'low': 'Water stressed', 'high': 'High moisture',
                 'desc': 'Moisture stress indicator. Tracks leaf water content.'},

        # =========================================================================
        # CLAY MINERALS - Reference: Hunt (1977), Crowley (1989), Swayze (2014)
        # =========================================================================
        'Clay (General)': {'low': 'No clay', 'high': 'Strong Al-OH',
                 'desc': '2200nm Al-OH absorption. Detects kaolinite, illite, smectite.'},
        'Kaolinite': {'low': 'No kaolinite', 'high': 'Strong kaolinite',
                      'desc': '2205nm Al-OH. Diagnostic doublet at 2165/2205nm. Argillic alteration.'},
        'Alunite': {'low': 'No alunite', 'high': 'Strong alunite',
                    'desc': '2170nm Al-OH. Key for epithermal Au exploration. Cuprite signature mineral.'},
        'Smectite': {'low': 'No smectite', 'high': 'Strong smectite',
                     'desc': '2200nm broad Al-OH. Montmorillonite group. Wider absorption than kaolinite.'},
        'Illite': {'low': 'No illite', 'high': 'Strong illite',
                   'desc': '2195nm Al-OH (shifted). Muscovite/sericite. Hydrothermal alteration indicator.'},

        # =========================================================================
        # CARBONATES - Reference: Gaffey (1986), Ninomiya (2003)
        # =========================================================================
        'Carbonate': {'low': 'No carbonate', 'high': 'Strong CO3',
                      'desc': '2330nm CO3 vibration. General carbonate detection.'},
        'Calcite': {'low': 'No calcite', 'high': 'Strong calcite',
                    'desc': '2340nm CO3. Limestone, marble. Longer wavelength than dolomite.'},
        'Dolomite': {'low': 'No dolomite', 'high': 'Strong dolomite',
                     'desc': '2320nm CO3. Mg shifts absorption ~20nm shorter than calcite.'},
        'Chlorite': {'low': 'No chlorite', 'high': 'Strong Mg-OH',
                     'desc': '2330nm Mg-OH. Chlorite/serpentine. Mafic alteration indicator.'},

        # =========================================================================
        # IRON OXIDES - Reference: Rowan & Mars (2003), Hunt (1977)
        # =========================================================================
        'Iron Oxide': {'low': 'Low iron', 'high': 'Iron oxide-rich',
                       'desc': 'Fe³⁺ charge transfer (650/450nm). Hematite, goethite, limonite.'},
        'Ferric Iron': {'low': 'Low Fe³⁺', 'high': 'High Fe³⁺',
                        'desc': 'Fe³⁺ crystal field (750/860nm). Red/yellow iron oxides.'},
        'Ferrous Iron': {'low': 'Low Fe²⁺', 'high': 'High Fe²⁺',
                         'desc': 'Fe²⁺ crystal field (860/1000nm). Pyroxene, olivine, chlorite.'},

        # =========================================================================
        # HYDROCARBONS - Reference: Kühn (2004), Cloutis (1989)
        # =========================================================================
        'Hydrocarbon': {'low': 'No hydrocarbons', 'high': 'Strong C-H',
                    'desc': '1730nm C-H 1st overtone. Primary oil/organic detection band.'},
        'HC 2310': {'low': 'No C-H', 'high': 'C-H combination',
                    'desc': '2310nm C-H combination. Secondary hydrocarbon confirmation.'},
        'Methane': {'low': 'No CH4', 'high': 'CH4 absorption',
                    'desc': '2300nm CH4 absorption. Natural gas seeps, emissions detection.'},
        'Oil Slick': {'low': 'No oil', 'high': 'Oil detected',
                      'desc': '1730nm optimized for marine oil slicks. Thickness indicator.'},

        # =========================================================================
        # AGRICULTURE / NITROGEN - Reference: Serrano (2002), Kokaly (1999)
        # =========================================================================
        'Protein': {'low': 'Low protein', 'high': 'High protein',
                    'desc': '2170nm N-H. Leaf protein/nitrogen content. Crop health indicator.'},
        'Cellulose': {'low': 'Low cellulose', 'high': 'High cellulose',
                      'desc': '2100nm C-O. Cell wall content. High with low protein = stress.'},
        'Lignin': {'low': 'Low lignin', 'high': 'High lignin',
                   'desc': '1680nm C-H. Woody material, senescent vegetation, dry litter.'},

        # =========================================================================
        # LEGACY / CUSTOM INDICES
        # =========================================================================
        'HC 1730': {'low': 'No hydrocarbons', 'high': 'C-H absorption',
                    'desc': 'Legacy: C-H 1st overtone ratio (1680/1730nm).'},
        'HC (Light)': {'low': 'No light crude', 'high': 'Light HC',
                       'desc': 'Legacy: Light hydrocarbon transitions (900/1000nm).'},
        'Aromatic': {'low': 'Aliphatic', 'high': 'Aromatic-rich',
                     'desc': 'Aromatic vs aliphatic C-H ratio.'},
        'Chain_Length': {'low': 'Short chains', 'high': 'Long chains',
                         'desc': 'CH3/CH2 ratio indicates hydrocarbon chain length.'},
        'OH_Index': {'low': 'No O-H', 'high': 'O-H absorption',
                     'desc': 'Alcohol O-H overtone. ⚠ In H2O absorption region.'},
        'Water': {'low': 'Dry', 'high': 'Wet/Water',
                  'desc': 'Water absorption. ⚠ In atmospheric H2O band.'},
        'NH_Index': {'low': 'No N-H', 'high': 'N-H absorption',
                     'desc': 'Primary amine N-H overtone bands.'},
        'Nitro': {'low': 'No nitro', 'high': 'Nitro absorption',
                  'desc': 'N-O overtone. Nitro compound detection.'},
    }
    
    def __init__(self, viewer_app):
        super().__init__()
        self.viewer_app = viewer_app
        self.setMinimumWidth(280)
        self.setMaximumWidth(350)
        self.current_suggested_index = None
        self.reference_bands = []
        self._build_ui()
        
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(4)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        # Wavelength display
        self.wavelength_label = QLabel("Band: -- | λ: -- nm")
        self.wavelength_label.setStyleSheet("""
            QLabel { font-size: 13px; font-weight: bold; color: #00ffcc;
                padding: 6px; background-color: #1a1a1a; border-radius: 4px; }
        """)
        scroll_layout.addWidget(self.wavelength_label)
        
        # Preset Composites
        preset_section = CollapsibleSection("Preset Composites")
        presets = [("True Color", self._true_color), ("CIR", self._cir),
                   ("SWIR Geology", self._swir_geology), ("Veg Stress", self._veg_stress),
                   ("False NIR", self._false_nir), ("Urban", self._urban)]
        preset_grid = QGridLayout()
        preset_grid.setSpacing(4)
        for i, (name, cb) in enumerate(presets):
            btn = QPushButton(name)
            btn.clicked.connect(cb)
            btn.setMinimumHeight(28)
            btn.setStyleSheet("QPushButton { font-size: 11px; }")
            preset_grid.addWidget(btn, i // 2, i % 2)
        preset_section.add_layout(preset_grid)
        scroll_layout.addWidget(preset_section)
        
        # Spectral Indices - Organized by category with separators
        index_section = CollapsibleSection("Spectral Indices")
        self.index_combo = QComboBox()
        self.index_combo.setStyleSheet("""
            QComboBox { font-size: 11px; }
            QComboBox QAbstractItemView { min-width: 200px; }
        """)

        # Add indices organized by category
        index_categories = [
            ("── VEGETATION ──", None),
            ("NDVI", "NDVI"), ("NDRE", "NDRE"), ("NDWI", "NDWI"), ("NDMI", "NDMI"),
            ("── CLAY MINERALS ──", None),
            ("Clay (General)", "Clay (General)"), ("Kaolinite", "Kaolinite"),
            ("Alunite", "Alunite"), ("Smectite", "Smectite"), ("Illite", "Illite"),
            ("── CARBONATES / Mg-OH ──", None),
            ("Carbonate", "Carbonate"), ("Calcite", "Calcite"), ("Dolomite", "Dolomite"),
            ("Chlorite", "Chlorite"),
            ("── IRON OXIDES ──", None),
            ("Iron Oxide", "Iron Oxide"), ("Ferric Iron", "Ferric Iron"),
            ("Ferrous Iron", "Ferrous Iron"),
            ("── HYDROCARBONS ──", None),
            ("Hydrocarbon", "Hydrocarbon"), ("HC 2310", "HC 2310"),
            ("Methane", "Methane"), ("Oil Slick", "Oil Slick"),
            ("── AGRICULTURE ──", None),
            ("Protein", "Protein"), ("Cellulose", "Cellulose"), ("Lignin", "Lignin"),
        ]

        for display_name, index_key in index_categories:
            if index_key is None:
                # Category header - add as disabled item
                self.index_combo.addItem(display_name)
                idx = self.index_combo.count() - 1
                self.index_combo.model().item(idx).setEnabled(False)
            else:
                self.index_combo.addItem(display_name)

        # Start with first selectable item (NDVI)
        self.index_combo.setCurrentIndex(1)
        index_section.add_widget(self.index_combo)
        self.index_info = QLabel("")
        self.index_info.setStyleSheet("QLabel { font-size: 10px; color: #888; }")
        self.index_info.setWordWrap(True)
        self.index_combo.currentTextChanged.connect(self._update_index_info)
        index_section.add_widget(self.index_info)
        calc_idx_btn = QPushButton("Calculate Index")
        calc_idx_btn.setMinimumHeight(30)
        calc_idx_btn.clicked.connect(self._calculate_index)
        calc_idx_btn.setStyleSheet("QPushButton { background-color: #4a3728; font-weight: bold; }")
        index_section.add_widget(calc_idx_btn)
        scroll_layout.addWidget(index_section)
        
        # Spectral Signature Calculator (comprehensive section)
        calc_section = CollapsibleSection("Spectral Signature Calculator")
        calc_layout = QVBoxLayout()
        calc_layout.setSpacing(4)
        
        # Material dropdown
        compound_layout = QHBoxLayout()
        compound_layout.addWidget(QLabel("Material:"))
        self.compound_combo = QComboBox()
        self.compound_combo.addItems(list(SpectralCalculator.COMPOUND_SIGNATURES.keys()))
        self.compound_combo.currentTextChanged.connect(self._update_calc_display)
        compound_layout.addWidget(self.compound_combo, stretch=1)
        calc_layout.addLayout(compound_layout)
        
        # Bands display
        self.calc_display = QTextEdit()
        self.calc_display.setReadOnly(True)
        self.calc_display.setMaximumHeight(120)
        self.calc_display.setStyleSheet("""
            QTextEdit { font-family: monospace; font-size: 10px;
                background-color: #1a1a2e; color: #00ffaa;
                border: 1px solid #333; border-radius: 4px; }
        """)
        calc_layout.addWidget(self.calc_display)
        
        # Overlay/Suggest buttons
        btn_row = QHBoxLayout()
        overlay_btn = QPushButton("Overlay on Plot")
        overlay_btn.clicked.connect(self._overlay_reference_bands)
        overlay_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        btn_row.addWidget(overlay_btn)
        suggest_btn = QPushButton("Suggest Index")
        suggest_btn.clicked.connect(self._suggest_index)
        suggest_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        btn_row.addWidget(suggest_btn)
        calc_layout.addLayout(btn_row)
        
        # --- Calculate Suggested Index subsection ---
        calc_layout.addWidget(QLabel("─" * 30))
        calc_layout.addWidget(QLabel("Calculate Suggested Index:"))
        self.suggested_index_label = QLabel("Select material and click 'Suggest Index'")
        self.suggested_index_label.setStyleSheet("QLabel { font-size: 9px; color: #666; font-style: italic; }")
        self.suggested_index_label.setWordWrap(True)
        calc_layout.addWidget(self.suggested_index_label)
        calc_sugg_btn = QPushButton("Calculate Suggested Index")
        calc_sugg_btn.clicked.connect(self._calculate_suggested_index)
        calc_sugg_btn.setStyleSheet("QPushButton { font-size: 10px; background-color: #4a3728; }")
        calc_layout.addWidget(calc_sugg_btn)
        calc_layout.addWidget(QLabel("ℹ Index = ratio → single value/pixel"))
        
        # --- Custom RGB subsection ---
        calc_layout.addWidget(QLabel("─" * 30))
        calc_layout.addWidget(QLabel("Visualize Raw Bands (RGB):"))
        rgb_grid = QGridLayout()
        rgb_grid.setSpacing(2)
        for i, lbl in enumerate(["R:", "G:", "B:"]):
            rgb_grid.addWidget(QLabel(lbl), 0, i*2)
        self.r_input = QLineEdit("640")
        self.g_input = QLineEdit("550")
        self.b_input = QLineEdit("470")
        for i, w in enumerate([self.r_input, self.g_input, self.b_input]):
            w.setMaximumWidth(55)
            rgb_grid.addWidget(w, 0, i*2 + 1)
        calc_layout.addLayout(rgb_grid)
        
        rgb_btn_row = QHBoxLayout()
        fill_btn = QPushButton("← Fill from Material")
        fill_btn.clicked.connect(self._apply_calc_to_custom)
        fill_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        rgb_btn_row.addWidget(fill_btn)
        gen_btn = QPushButton("Generate Composite")
        gen_btn.clicked.connect(self._custom_composite)
        gen_btn.setStyleSheet("QPushButton { font-size: 10px; background-color: #2d5a27; }")
        rgb_btn_row.addWidget(gen_btn)
        calc_layout.addLayout(rgb_btn_row)
        calc_layout.addWidget(QLabel("ℹ RGB = reflectance at 3 wavelengths"))
        
        # --- Functional group subsection ---
        calc_layout.addWidget(QLabel("─" * 30))
        calc_layout.addWidget(QLabel("Functional group overtones:"))
        fg_row = QHBoxLayout()
        self.fg_combo = QComboBox()
        self.fg_combo.addItems(list(SpectralCalculator.FUNDAMENTALS.keys()))
        fg_row.addWidget(self.fg_combo, stretch=1)
        fg_btn = QPushButton("Calculate OT")
        fg_btn.clicked.connect(self._calc_fg_overtones)
        fg_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        fg_row.addWidget(fg_btn)
        calc_layout.addLayout(fg_row)
        
        calc_widget = QWidget()
        calc_widget.setLayout(calc_layout)
        calc_section.add_widget(calc_widget)
        calc_section.toggle()
        scroll_layout.addWidget(calc_section)
        
        self._update_calc_display(self.compound_combo.currentText())
        
        # ROI Tools
        roi_section = CollapsibleSection("ROI Spectral Analysis")
        roi_btn_row = QHBoxLayout()
        rect_btn = QPushButton("Rectangle")
        rect_btn.clicked.connect(lambda: self._set_roi_mode('rectangle'))
        roi_btn_row.addWidget(rect_btn)
        poly_btn = QPushButton("Polygon")
        poly_btn.clicked.connect(lambda: self._set_roi_mode('polygon'))
        roi_btn_row.addWidget(poly_btn)
        roi_section.add_layout(roi_btn_row)
        extract_btn = QPushButton("Extract Spectra")
        extract_btn.setMinimumHeight(30)
        extract_btn.clicked.connect(self._extract_spectra)
        extract_btn.setStyleSheet("QPushButton { background-color: #28445a; font-weight: bold; }")
        roi_section.add_widget(extract_btn)
        export_btn = QPushButton("Export Spectra to CSV")
        export_btn.clicked.connect(self._export_spectra)
        roi_section.add_widget(export_btn)
        clear_roi_btn = QPushButton("Clear ROIs")
        clear_roi_btn.clicked.connect(self._clear_rois)
        roi_section.add_widget(clear_roi_btn)
        scroll_layout.addWidget(roi_section)
        
        # Layer Management
        layer_section = CollapsibleSection("Layer Management")
        clear_btn = QPushButton("Clear All Composites")
        clear_btn.clicked.connect(self._clear_composites)
        layer_section.add_widget(clear_btn)
        toggle_btn = QPushButton("Toggle Data Cube")
        toggle_btn.clicked.connect(self._toggle_cube)
        layer_section.add_widget(toggle_btn)
        layer_section.toggle()
        scroll_layout.addWidget(layer_section)
        
        # Atmospheric Correction (Integration with aviris_atm_correction_v2.py and ISOFIT)
        atm_section = CollapsibleSection("Atmospheric Correction")

        # Status label
        self.atm_status_label = QLabel("Data type: Checking...")
        self.atm_status_label.setStyleSheet("QLabel { font-size: 10px; color: #888; }")
        atm_section.add_widget(self.atm_status_label)

        # Method selection (NEW - ISOFIT option)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.atm_method_combo = QComboBox()
        self.atm_method_combo.addItems([
            "ISOFIT + sRTMnet (NASA Standard)",
            "6S LUT (Fast)",
            "Simple DOS (No RTM)"
        ])
        self.atm_method_combo.setStyleSheet("QComboBox { font-size: 10px; }")
        self.atm_method_combo.setToolTip(
            "ISOFIT: NASA/JPL operational standard, best quality\n"
            "6S LUT: Fast radiative transfer with look-up tables\n"
            "Simple DOS: Dark object subtraction, no external dependencies"
        )
        self.atm_method_combo.currentIndexChanged.connect(self._update_atm_method_options)
        method_row.addWidget(self.atm_method_combo, stretch=1)
        atm_section.add_layout(method_row)

        # OBS file selection
        obs_row = QHBoxLayout()
        obs_row.addWidget(QLabel("OBS File:"))
        self.obs_path_edit = QLineEdit()
        self.obs_path_edit.setPlaceholderText("Select observation file...")
        self.obs_path_edit.setStyleSheet("QLineEdit { font-size: 10px; }")
        obs_row.addWidget(self.obs_path_edit, stretch=1)
        obs_browse_btn = QPushButton("...")
        obs_browse_btn.setFixedWidth(30)
        obs_browse_btn.clicked.connect(self._browse_obs_file)
        obs_row.addWidget(obs_browse_btn)
        atm_section.add_layout(obs_row)

        # Options checkboxes
        from qtpy.QtWidgets import QCheckBox
        self.coastal_correction_cb = QCheckBox("Coastal/Adjacency Correction")
        self.coastal_correction_cb.setChecked(True)
        self.coastal_correction_cb.setStyleSheet("QCheckBox { font-size: 10px; }")
        atm_section.add_widget(self.coastal_correction_cb)

        self.uncertainty_cb = QCheckBox("Estimate Uncertainty")
        self.uncertainty_cb.setChecked(True)
        self.uncertainty_cb.setStyleSheet("QCheckBox { font-size: 10px; }")
        atm_section.add_widget(self.uncertainty_cb)

        # Aerosol model selection (for 6S method)
        aerosol_row = QHBoxLayout()
        self.aerosol_label = QLabel("Aerosol:")
        aerosol_row.addWidget(self.aerosol_label)
        self.aerosol_combo = QComboBox()
        self.aerosol_combo.addItems(["maritime", "continental", "urban", "desert"])
        self.aerosol_combo.setStyleSheet("QComboBox { font-size: 10px; }")
        aerosol_row.addWidget(self.aerosol_combo, stretch=1)
        atm_section.add_layout(aerosol_row)

        # Cores selection (for ISOFIT)
        cores_row = QHBoxLayout()
        self.cores_label = QLabel("Cores:")
        cores_row.addWidget(self.cores_label)
        self.cores_spinbox = QSpinBox()
        self.cores_spinbox.setRange(1, 32)
        self.cores_spinbox.setValue(4)
        self.cores_spinbox.setStyleSheet("QSpinBox { font-size: 10px; }")
        self.cores_spinbox.setToolTip("Number of parallel cores for ISOFIT processing")
        cores_row.addWidget(self.cores_spinbox)
        cores_row.addStretch()
        atm_section.add_layout(cores_row)

        # Run button
        run_atm_btn = QPushButton("Run Atmospheric Correction")
        run_atm_btn.setMinimumHeight(32)
        run_atm_btn.clicked.connect(self._run_atmospheric_correction)
        run_atm_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a5a2a;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #3a7a3a; }
        """)
        atm_section.add_widget(run_atm_btn)

        # Load corrected data button
        load_corrected_btn = QPushButton("Load Corrected Data")
        load_corrected_btn.clicked.connect(self._load_corrected_data)
        load_corrected_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        atm_section.add_widget(load_corrected_btn)

        # Processing status
        self.atm_progress_label = QLabel("")
        self.atm_progress_label.setStyleSheet("QLabel { font-size: 9px; color: #00ff00; }")
        self.atm_progress_label.setWordWrap(True)
        atm_section.add_widget(self.atm_progress_label)

        atm_section.toggle()  # Start collapsed
        scroll_layout.addWidget(atm_section)

        # 3D Camera Controls
        camera_section = CollapsibleSection("3D Camera Controls")
        preset_row = QHBoxLayout()
        for name, angles in [("Top", (0, 0)), ("Front", (0, -90)), ("Side", (90, -90)), ("Iso", (45, -45))]:
            btn = QPushButton(name)
            btn.setFixedWidth(50)
            btn.clicked.connect(lambda c, a=angles: self._set_camera_angle(*a))
            preset_row.addWidget(btn)
        camera_section.add_layout(preset_row)
        
        # Azimuth
        az_row = QHBoxLayout()
        az_row.addWidget(QLabel("Azimuth:"))
        self.azimuth_slider = QSlider(Qt.Horizontal)
        self.azimuth_slider.setRange(-180, 180)
        self.azimuth_slider.setValue(0)
        self.azimuth_slider.valueChanged.connect(self._update_camera_from_sliders)
        az_row.addWidget(self.azimuth_slider)
        self.az_value_label = QLabel("0°")
        az_row.addWidget(self.az_value_label)
        camera_section.add_layout(az_row)
        
        # Elevation
        el_row = QHBoxLayout()
        el_row.addWidget(QLabel("Elevation:"))
        self.elevation_slider = QSlider(Qt.Horizontal)
        self.elevation_slider.setRange(-90, 90)
        self.elevation_slider.setValue(0)
        self.elevation_slider.valueChanged.connect(self._update_camera_from_sliders)
        el_row.addWidget(self.elevation_slider)
        self.el_value_label = QLabel("0°")
        el_row.addWidget(self.el_value_label)
        camera_section.add_layout(el_row)
        
        # Zoom
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._update_zoom)
        zoom_row.addWidget(self.zoom_slider)
        self.zoom_value_label = QLabel("1.0x")
        zoom_row.addWidget(self.zoom_value_label)
        camera_section.add_layout(zoom_row)
        
        cam_btns = QHBoxLayout()
        toggle_3d = QPushButton("Toggle 2D/3D")
        toggle_3d.clicked.connect(self._toggle_3d)
        cam_btns.addWidget(toggle_3d)
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_camera)
        cam_btns.addWidget(reset_btn)
        camera_section.add_layout(cam_btns)
        camera_section.toggle()
        scroll_layout.addWidget(camera_section)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll, stretch=1)
        
        # Colorbar
        self._build_colorbar(main_layout)
        
        # Spectral Plot
        main_layout.addWidget(QLabel("Spectral Plot"))
        self.spectral_plot = SpectralPlot()
        self.spectral_plot.setMinimumHeight(200)
        self.spectral_plot.setMaximumHeight(300)
        main_layout.addWidget(self.spectral_plot)
        
        self._update_index_info(self.index_combo.currentText())
    
    def _build_colorbar(self, parent):
        self.colorbar_widget = QWidget()
        layout = QVBoxLayout(self.colorbar_widget)
        layout.setContentsMargins(4, 4, 4, 4)
        self.colorbar_title = QLabel("Index Legend")
        layout.addWidget(self.colorbar_title)
        row = QHBoxLayout()
        self.colorbar_min_label = QLabel("0.0")
        row.addWidget(self.colorbar_min_label)
        self.colorbar_gradient = QLabel()
        self.colorbar_gradient.setFixedHeight(20)
        self.colorbar_gradient.setMinimumWidth(150)
        self.colorbar_gradient.setStyleSheet("""
            QLabel { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #440154, stop:0.25 #3b528b, stop:0.5 #21918c, 
                stop:0.75 #5ec962, stop:1 #fde725); border: 1px solid #555; }
        """)
        row.addWidget(self.colorbar_gradient, stretch=1)
        self.colorbar_max_label = QLabel("1.0")
        row.addWidget(self.colorbar_max_label)
        layout.addLayout(row)
        sem_row = QHBoxLayout()
        self.colorbar_low_semantic = QLabel("Low")
        sem_row.addWidget(self.colorbar_low_semantic)
        sem_row.addStretch()
        self.colorbar_high_semantic = QLabel("High")
        sem_row.addWidget(self.colorbar_high_semantic)
        layout.addLayout(sem_row)
        self.colorbar_desc = QLabel("Calculate an index to see the scale")
        self.colorbar_desc.setWordWrap(True)
        layout.addWidget(self.colorbar_desc)
        self.colorbar_widget.setMaximumHeight(90)
        self.colorbar_widget.setStyleSheet("QWidget { background-color: #1a1a1a; }")
        parent.addWidget(self.colorbar_widget)
    
    def update_data_info(self, loader):
        self.spectral_plot.set_data_type(loader.data_type)
        self.spectral_plot.set_wavelength_range(loader.wl_min, loader.wl_max)
        # Update atmospheric correction status
        self._update_atm_status()
    
    def _update_index_info(self, text):
        key = text.split(" - ")[0]
        meta = self.INDEX_METADATA.get(key, {})
        self.index_info.setText(meta.get('desc', ''))
    
    def update_wavelength_display(self, band_idx, wavelength):
        self.wavelength_label.setText(f"Band: {band_idx} | λ: {wavelength:.1f} nm")
    
    # Composite callbacks
    def _true_color(self): 
        r = self.viewer_app.add_composite("True Color", 640, 550, 470)
        if r: self._update_colorbar_composite(r)
    def _cir(self): 
        r = self.viewer_app.add_composite("CIR", 860, 650, 550)
        if r: self._update_colorbar_composite(r)
    def _swir_geology(self): 
        r = self.viewer_app.add_composite("SWIR Geology", 2200, 1650, 850)
        if r: self._update_colorbar_composite(r)
    def _veg_stress(self): 
        r = self.viewer_app.add_composite("Veg Stress", 750, 705, 550)
        if r: self._update_colorbar_composite(r)
    def _false_nir(self): 
        r = self.viewer_app.add_composite("False NIR", 860, 750, 650)
        if r: self._update_colorbar_composite(r)
    def _urban(self): 
        r = self.viewer_app.add_composite("Urban", 2200, 860, 650)
        if r: self._update_colorbar_composite(r)
    
    def _custom_composite(self):
        try:
            r_wl, g_wl, b_wl = float(self.r_input.text()), float(self.g_input.text()), float(self.b_input.text())
            loader = self.viewer_app.data_loader
            for wl, n in [(r_wl, 'R'), (g_wl, 'G'), (b_wl, 'B')]:
                if not loader.validate_wavelength(wl):
                    QMessageBox.warning(self, "Invalid", f"{n}={wl}nm outside range")
                    return
            name = f"Custom ({int(r_wl)}/{int(g_wl)}/{int(b_wl)})"
            r = self.viewer_app.add_composite(name, r_wl, g_wl, b_wl)
            if r: self._update_colorbar_composite(r)
        except ValueError:
            QMessageBox.warning(self, "Error", "Enter valid wavelengths")
    
    def _calculate_index(self):
        r = self.viewer_app.calculate_index(self.index_combo.currentText(), estimate_uncertainty=True)
        if r: self._update_colorbar(r)

    def _update_colorbar(self, info):
        """Update colorbar display with index info and uncertainty."""
        name, cmap, clim = info['name'], info['cmap'], info['clim']
        meta = self.INDEX_METADATA.get(name, {'low': 'Low', 'high': 'High', 'desc': ''})
        self.colorbar_title.setText(f"{name} Index")
        self.colorbar_min_label.setText(f"{clim[0]:.2f}")
        self.colorbar_max_label.setText(f"{clim[1]:.2f}")
        self.colorbar_low_semantic.setText(meta['low'])
        self.colorbar_high_semantic.setText(meta['high'])

        # Build description with uncertainty if available
        desc_text = meta['desc']
        uncertainty = info.get('uncertainty')
        if uncertainty:
            uncert_pct = uncertainty['median_relative_uncertainty'] * 100
            if uncert_pct > 50:
                desc_text += f"\n⚠ High uncertainty: ±{uncert_pct:.0f}%"
            elif uncert_pct > 20:
                desc_text += f"\n⚠ Moderate uncertainty: ±{uncert_pct:.0f}%"
            else:
                desc_text += f"\nUncertainty: ±{uncert_pct:.0f}%"

        self.colorbar_desc.setText(desc_text)
        self._set_colorbar_gradient(cmap)
    
    def _update_colorbar_custom_index(self, info):
        """Update colorbar for custom calculated index with proper metadata lookup and uncertainty."""
        name = info['name']
        base_name = info.get('base_name', name)  # Use base name for metadata lookup
        cmap = info['cmap']
        clim = info['clim']
        b1, b2 = info['b1'], info['b2']
        desc = info.get('description', f'Band ratio {b1:.0f}nm / {b2:.0f}nm')
        idx_type = info.get('type', 'ratio')

        # Look up metadata by base index name (e.g., "NDVI" not "Vegetation (NDVI)")
        meta = self.INDEX_METADATA.get(base_name, None)

        self.colorbar_title.setText(f"{name} Index")
        self.colorbar_min_label.setText(f"{clim[0]:.2f}")
        self.colorbar_max_label.setText(f"{clim[1]:.2f}")

        if meta:
            self.colorbar_low_semantic.setText(meta['low'])
            self.colorbar_high_semantic.setText(meta['high'])
            desc_text = meta['desc']
        else:
            # Generic labels based on index type
            if idx_type == 'nd':
                self.colorbar_low_semantic.setText("Low/Negative")
                self.colorbar_high_semantic.setText("High/Positive")
            elif idx_type == 'continuum':
                self.colorbar_low_semantic.setText("No absorption")
                self.colorbar_high_semantic.setText("Strong absorption")
            else:
                self.colorbar_low_semantic.setText("Low ratio")
                self.colorbar_high_semantic.setText("High ratio")
            desc_text = desc

        # Add uncertainty info if available
        uncertainty = info.get('uncertainty')
        if uncertainty:
            uncert_pct = uncertainty['median_relative_uncertainty'] * 100
            if uncert_pct > 50:
                desc_text += f"\n⚠ High uncertainty: ±{uncert_pct:.0f}%"
            elif uncert_pct > 20:
                desc_text += f"\n⚠ Moderate uncertainty: ±{uncert_pct:.0f}%"
            else:
                desc_text += f"\nUncertainty: ±{uncert_pct:.0f}%"

        self.colorbar_desc.setText(desc_text)
        self._set_colorbar_gradient(cmap)
    
    def _set_colorbar_gradient(self, cmap):
        """Set colorbar gradient style based on colormap name."""
        gradient_styles = {
            # Diverging colormaps
            'RdYlGn': 'stop:0 #a50026, stop:0.25 #f46d43, stop:0.5 #ffffbf, stop:0.75 #a6d96a, stop:1 #006837',
            'RdBu': 'stop:0 #67001f, stop:0.25 #d6604d, stop:0.5 #f7f7f7, stop:0.75 #4393c3, stop:1 #053061',
            'BrBG': 'stop:0 #543005, stop:0.25 #bf812d, stop:0.5 #f5f5f5, stop:0.75 #80cdc1, stop:1 #003c30',
            # Perceptually uniform
            'viridis': 'stop:0 #440154, stop:0.25 #3b528b, stop:0.5 #21918c, stop:0.75 #5ec962, stop:1 #fde725',
            'magma': 'stop:0 #000004, stop:0.25 #51127c, stop:0.5 #b73779, stop:0.75 #fc8961, stop:1 #fcfdbf',
            'plasma': 'stop:0 #0d0887, stop:0.25 #7e03a8, stop:0.5 #cc4778, stop:0.75 #f89540, stop:1 #f0f921',
            'inferno': 'stop:0 #000004, stop:0.25 #420a68, stop:0.5 #932667, stop:0.75 #dd513a, stop:1 #fca50a',
            'cividis': 'stop:0 #00224e, stop:0.25 #3d4e6e, stop:0.5 #7a7b78, stop:0.75 #b8a859, stop:1 #fee838',
            # Sequential - warm
            'YlOrBr': 'stop:0 #ffffe5, stop:0.25 #fec44f, stop:0.5 #ec7014, stop:0.75 #cc4c02, stop:1 #662506',
            'YlOrRd': 'stop:0 #ffffcc, stop:0.25 #feb24c, stop:0.5 #fd8d3c, stop:0.75 #e31a1c, stop:1 #800026',
            'OrRd': 'stop:0 #fff7ec, stop:0.25 #fdd49e, stop:0.5 #fc8d59, stop:0.75 #d7301f, stop:1 #7f0000',
            'Oranges': 'stop:0 #fff5eb, stop:0.25 #fdd0a2, stop:0.5 #fd8d3c, stop:0.75 #d94801, stop:1 #7f2704',
            'Reds': 'stop:0 #fff5f0, stop:0.25 #fcae91, stop:0.5 #fb6a4a, stop:0.75 #cb181d, stop:1 #67000d',
            'hot': 'stop:0 #0b0000, stop:0.25 #8b0000, stop:0.5 #ff4500, stop:0.75 #ffa500, stop:1 #ffff00',
            'copper': 'stop:0 #000000, stop:0.25 #4d2600, stop:0.5 #994c00, stop:0.75 #cc8033, stop:1 #ffc77f',
            # Sequential - cool
            'Blues': 'stop:0 #f7fbff, stop:0.25 #c6dbef, stop:0.5 #6baed6, stop:0.75 #2171b5, stop:1 #084594',
            'PuBu': 'stop:0 #fff7fb, stop:0.25 #d0d1e6, stop:0.5 #74a9cf, stop:0.75 #0570b0, stop:1 #023858',
            'Purples': 'stop:0 #fcfbfd, stop:0.25 #bcbddc, stop:0.5 #807dba, stop:0.75 #6a51a3, stop:1 #3f007d',
            'RdPu': 'stop:0 #fff7f3, stop:0.25 #fcc5c0, stop:0.5 #f768a1, stop:0.75 #ae017e, stop:1 #49006a',
            # Sequential - green
            'Greens': 'stop:0 #f7fcf5, stop:0.25 #a1d99b, stop:0.5 #41ab5d, stop:0.75 #006d2c, stop:1 #00441b',
            'YlGn': 'stop:0 #ffffe5, stop:0.25 #c2e699, stop:0.5 #78c679, stop:0.75 #238b45, stop:1 #004529',
            'BuGn': 'stop:0 #f7fcfd, stop:0.25 #b2e2e2, stop:0.5 #66c2a4, stop:0.75 #238b45, stop:1 #00441b',
        }
        gradient = gradient_styles.get(cmap, gradient_styles['viridis'])
        self.colorbar_gradient.setStyleSheet(f"""
            QLabel {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {gradient}); 
                     border: 1px solid #555; border-radius: 2px; }}
        """)
    
    def _update_colorbar_composite(self, info):
        self.colorbar_title.setText(f"RGB: {info['name']}")
        self.colorbar_min_label.setText(f"R:{info['r_wl']:.0f}")
        self.colorbar_max_label.setText(f"B:{info['b_wl']:.0f}")
        self.colorbar_gradient.setStyleSheet("""
            QLabel { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #ff0000, stop:0.5 #00ff00, stop:1 #0000ff); border: 1px solid #555; }
        """)
        self.colorbar_low_semantic.setText(f"R:{info['r_wl']:.0f}nm")
        self.colorbar_high_semantic.setText(f"B:{info['b_wl']:.0f}nm")
        self.colorbar_desc.setText(f"G:{info['g_wl']:.0f}nm")
    
    # ROI callbacks
    def _set_roi_mode(self, mode): self.viewer_app.set_roi_mode(mode)
    def _extract_spectra(self):
        spectra = self.viewer_app.extract_roi_spectra()
        if spectra:
            self.spectral_plot.clear_plot()
            colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'pink']
            for i, (name, wl, mean, std) in enumerate(spectra):
                self.spectral_plot.plot_spectrum(wl, mean, std, label=name, 
                    color=colors[i % len(colors)], clear=(i == 0))
    def _export_spectra(self):
        if not self.spectral_plot.has_spectral_data():
            QMessageBox.warning(self, "No Data", "Extract spectra first")
            return
        fp, _ = QFileDialog.getSaveFileName(self, "Save", "", "CSV (*.csv)")
        if fp: self.spectral_plot.export_to_csv(fp)
    def _clear_rois(self):
        self.viewer_app.clear_rois()
        self.spectral_plot.clear_plot()
    
    # Layer callbacks
    def _clear_composites(self): self.viewer_app.clear_composites()
    def _toggle_cube(self): self.viewer_app.toggle_cube()
    
    # Camera callbacks
    def _set_camera_angle(self, az, el):
        if self.viewer_app.viewer.dims.ndisplay != 3:
            self.viewer_app.viewer.dims.ndisplay = 3
        self.azimuth_slider.blockSignals(True)
        self.elevation_slider.blockSignals(True)
        self.azimuth_slider.setValue(int(az))
        self.elevation_slider.setValue(int(el))
        self.azimuth_slider.blockSignals(False)
        self.elevation_slider.blockSignals(False)
        self.az_value_label.setText(f"{int(az)}°")
        self.el_value_label.setText(f"{int(el)}°")
        self.viewer_app.viewer.camera.angles = (0, el, az)
    
    def _update_camera_from_sliders(self):
        az, el = self.azimuth_slider.value(), self.elevation_slider.value()
        self.az_value_label.setText(f"{az}°")
        self.el_value_label.setText(f"{el}°")
        if self.viewer_app.viewer.dims.ndisplay == 3:
            self.viewer_app.viewer.camera.angles = (0, el, az)
    
    def _update_zoom(self):
        z = self.zoom_slider.value() / 100.0
        self.zoom_value_label.setText(f"{z:.1f}x")
        self.viewer_app.viewer.camera.zoom = z
    
    def _toggle_3d(self):
        v = self.viewer_app.viewer
        v.dims.ndisplay = 3 if v.dims.ndisplay == 2 else 2
    
    def _reset_camera(self):
        self.viewer_app.viewer.reset_view()
        self.azimuth_slider.setValue(0)
        self.elevation_slider.setValue(0)
        self.zoom_slider.setValue(100)
    
    # Calculator callbacks
    def _update_calc_display(self, compound):
        bands = SpectralCalculator.get_compound_bands(compound)
        if not bands:
            self.calc_display.setText("No data")
            self.reference_bands = []
            return
        lines = [f"{'λ (nm)':<10} {'Assignment':<28} {'Intensity':<10}", "-" * 50]
        self.reference_bands = []
        for wl, assignment, intensity in bands:
            lines.append(f"{wl:<10.0f} {assignment:<28} {intensity:<10}")
            self.reference_bands.append((wl, assignment))
        self.calc_display.setText("\n".join(lines))
    
    def _overlay_reference_bands(self):
        if not self.reference_bands: return
        self.spectral_plot.clear_reference_lines()
        for wl, label in self.reference_bands:
            self.spectral_plot.add_reference_line(wl, label)
        self.spectral_plot.draw()
    
    def _suggest_index(self):
        compound = self.compound_combo.currentText()
        suggestion = SpectralCalculator.suggest_index(compound)
        if suggestion:
            name, b1, b2, desc, idx_type, cmap = suggestion
            
            # Format formula based on index type
            if idx_type == 'nd':
                formula = f"(R({b1}) - R({b2})) / (R({b1}) + R({b2}))"
                type_label = "Normalized Difference"
            else:
                formula = f"R({b1}) / R({b2})"
                type_label = "Band Ratio"
            
            msg = f"Suggested: {name}\nType: {type_label}\nFormula: {formula}\nPhysics: {desc}"
            
            # Check for atmospheric absorption bands
            atm_bands = [(1350, 1450, 'H2O'), (1800, 1950, 'H2O'), (2500, 2600, 'CO2')]
            warnings = []
            for wl in [b1, b2]:
                for lo, hi, gas in atm_bands:
                    if lo <= wl <= hi:
                        warnings.append(f"{wl}nm in {gas} absorption band")
            if warnings:
                msg += f"\n\n⚠ WARNING: {'; '.join(warnings)}"
                msg += "\nData may be unreliable in these regions!"
            
            self.calc_display.setText(msg)
            self.current_suggested_index = {
                'name': name, 'b1': b1, 'b2': b2, 
                'description': desc, 'compound': compound,
                'index_type': idx_type, 'cmap': cmap
            }
            label_text = f"{name}: {formula}"
            if warnings:
                label_text += " ⚠"
            self.suggested_index_label.setText(label_text)
            self.suggested_index_label.setStyleSheet("QLabel { color: #00cc88; }" if not warnings else "QLabel { color: #ffaa00; }")
        else:
            self.calc_display.setText("No suggestion available")
            self.current_suggested_index = None
    
    def _calculate_suggested_index(self):
        if not self.current_suggested_index:
            QMessageBox.information(self, "No Index", "Click 'Suggest Index' first")
            return
        idx = self.current_suggested_index
        name = f"{idx['compound']} ({idx['name']})"
        idx_type = idx.get('index_type', 'ratio')
        cmap = idx.get('cmap', 'viridis')

        r = self.viewer_app.calculate_custom_index(
            name, idx['b1'], idx['b2'], idx_type, cmap,
            estimate_uncertainty=True
        )
        if r:
            r['description'] = idx['description']
            # Use base index name for metadata lookup
            r['base_name'] = idx['name']
            self._update_colorbar_custom_index(r)
    
    def _apply_calc_to_custom(self):
        bands = SpectralCalculator.get_compound_bands(self.compound_combo.currentText())
        if len(bands) >= 3:
            strong = [b for b in bands if b[2] in ('strong', 'very strong')]
            src = strong[:3] if len(strong) >= 3 else bands[:3]
            self.r_input.setText(str(int(src[0][0])))
            self.g_input.setText(str(int(src[1][0])))
            self.b_input.setText(str(int(src[2][0])))
    
    def _calc_fg_overtones(self):
        fg = self.fg_combo.currentText()
        bands = SpectralCalculator.get_functional_group_overtones(fg)
        if not bands:
            self.calc_display.setText(f"No NIR bands for {fg}")
            self.reference_bands = []
            return
        omega_e, chi_e, desc = SpectralCalculator.FUNDAMENTALS[fg]
        lines = [f"Group: {fg}", f"Fundamental: {omega_e} cm⁻¹", f"χe: {chi_e:.3f}", "-" * 40]
        self.reference_bands = []
        for wl, assignment, intensity in bands:
            lines.append(f"{wl:<10.0f} {assignment:<28} {intensity:<6}")
            self.reference_bands.append((wl, assignment))
        self.calc_display.setText("\n".join(lines))

    # Atmospheric Correction callbacks
    def _browse_obs_file(self):
        """Browse for OBS (observation parameters) file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Observation Parameters File",
            str(self.viewer_app.filepath.parent),
            "NetCDF Files (*.nc);;All Files (*)"
        )
        if filepath:
            self.obs_path_edit.setText(filepath)

    def _update_atm_status(self):
        """Update the atmospheric correction status based on data type."""
        data_type = self.viewer_app.data_loader.data_type
        if data_type == 'reflectance':
            self.atm_status_label.setText("Data type: Reflectance (already corrected)")
            self.atm_status_label.setStyleSheet("QLabel { font-size: 10px; color: #00ff00; }")
        else:
            self.atm_status_label.setText("Data type: Radiance (needs correction)")
            self.atm_status_label.setStyleSheet("QLabel { font-size: 10px; color: #ffaa00; }")

    def _update_atm_method_options(self):
        """Update UI options based on selected atmospheric correction method."""
        method_idx = self.atm_method_combo.currentIndex()

        # ISOFIT: hide aerosol, show cores
        # 6S: show aerosol, hide cores
        # Simple: hide both
        if method_idx == 0:  # ISOFIT
            self.aerosol_label.setVisible(False)
            self.aerosol_combo.setVisible(False)
            self.cores_label.setVisible(True)
            self.cores_spinbox.setVisible(True)
            self.coastal_correction_cb.setVisible(False)  # ISOFIT handles this internally
        elif method_idx == 1:  # 6S
            self.aerosol_label.setVisible(True)
            self.aerosol_combo.setVisible(True)
            self.cores_label.setVisible(False)
            self.cores_spinbox.setVisible(False)
            self.coastal_correction_cb.setVisible(True)
        else:  # Simple DOS
            self.aerosol_label.setVisible(False)
            self.aerosol_combo.setVisible(False)
            self.cores_label.setVisible(False)
            self.cores_spinbox.setVisible(False)
            self.coastal_correction_cb.setVisible(True)

    def _run_atmospheric_correction(self):
        """Run atmospheric correction on the loaded radiance data."""
        # Check if we have radiance data
        if self.viewer_app.data_loader.data_type == 'reflectance':
            QMessageBox.information(
                self,
                "Already Corrected",
                "This data appears to already be reflectance. "
                "Atmospheric correction is only needed for radiance data."
            )
            return

        # Check for OBS file
        obs_path = self.obs_path_edit.text().strip()
        if not obs_path:
            # Try to auto-detect OBS file (try ORT file naming convention for AVIRIS-3)
            rad_path = self.viewer_app.filepath
            potential_paths = [
                rad_path.parent / rad_path.name.replace('_rdn_', '_ort_').replace('_RDN_', '_ORT_'),
                rad_path.parent / rad_path.name.replace('_rad_', '_obs_').replace('_rdn_', '_obs_'),
                rad_path.parent / rad_path.name.replace('RDN', 'ORT'),
            ]
            for potential_obs in potential_paths:
                if potential_obs.exists():
                    obs_path = str(potential_obs)
                    self.obs_path_edit.setText(obs_path)
                    break
            else:
                QMessageBox.warning(
                    self,
                    "OBS/ORT File Required",
                    "Please select the observation parameters (OBS/ORT) file.\n"
                    "This file contains solar/sensor geometry needed for correction."
                )
                return

        if not Path(obs_path).exists():
            QMessageBox.warning(self, "File Not Found", f"OBS file not found:\n{obs_path}")
            return

        # Get selected method
        method_idx = self.atm_method_combo.currentIndex()
        method_name = ["ISOFIT", "6S LUT", "Simple DOS"][method_idx]

        # Prepare output path
        rad_path = self.viewer_app.filepath
        suffix = "_isofit" if method_idx == 0 else "_6s" if method_idx == 1 else "_dos"
        output_path = rad_path.parent / rad_path.name.replace('_rdn', '_refl' + suffix).replace('_rad', '_refl' + suffix)
        if output_path == rad_path:
            output_path = rad_path.parent / (rad_path.stem + f"_corrected{suffix}.nc")

        self.atm_progress_label.setText(f"Starting {method_name} correction...")
        self.atm_progress_label.setStyleSheet("QLabel { font-size: 9px; color: #ffff00; }")

        from qtpy.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            if method_idx == 0:  # ISOFIT
                self._run_isofit_correction(rad_path, obs_path, output_path)
            else:  # 6S or Simple
                self._run_6s_correction(rad_path, obs_path, output_path, method_idx)

            self.atm_progress_label.setText(f"Complete! Saved to:\n{output_path.name}")
            self.atm_progress_label.setStyleSheet("QLabel { font-size: 9px; color: #00ff00; }")

            # Store output path for loading
            self._last_corrected_path = output_path

            # Ask to load
            reply = QMessageBox.question(
                self,
                "Correction Complete",
                f"Atmospheric correction complete!\n\nOutput: {output_path.name}\n\n"
                "Would you like to load the corrected data now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._load_corrected_data()

        except Exception as e:
            self.atm_progress_label.setText(f"Error: {str(e)[:50]}...")
            self.atm_progress_label.setStyleSheet("QLabel { font-size: 9px; color: #ff0000; }")
            logger.error(f"Atmospheric correction failed: {e}")
            QMessageBox.critical(
                self,
                "Processing Error",
                f"Atmospheric correction failed:\n{str(e)}"
            )

    def _run_isofit_correction(self, rad_path, obs_path, output_path):
        """Run ISOFIT atmospheric correction (NASA/JPL standard)."""
        from qtpy.QtWidgets import QApplication

        self.atm_progress_label.setText("Loading ISOFIT module...")
        QApplication.processEvents()

        # Try to import ISOFIT processor
        try:
            import importlib.util
            isofit_module_path = rad_path.parent / "aviris_isofit_processor.py"
            if not isofit_module_path.exists():
                isofit_module_path = Path("aviris_isofit_processor.py")
            if not isofit_module_path.exists():
                isofit_module_path = Path.home() / "Downloads" / "aviris_isofit_processor.py"

            if not isofit_module_path.exists():
                raise FileNotFoundError(
                    "Could not find aviris_isofit_processor.py.\n"
                    "Please ensure it's in the same directory as the data."
                )

            import sys
            module_dir = str(isofit_module_path.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)

            spec = importlib.util.spec_from_file_location("aviris_isofit_processor", isofit_module_path)
            isofit_module = importlib.util.module_from_spec(spec)
            sys.modules['aviris_isofit_processor'] = isofit_module
            spec.loader.exec_module(isofit_module)

        except Exception as e:
            raise ImportError(f"Failed to import ISOFIT module: {e}")

        # Check if ISOFIT is available
        if not isofit_module.HAS_ISOFIT:
            raise ImportError(
                "ISOFIT is not installed.\n\n"
                "Install with:\n"
                "  mamba install -c conda-forge isofit\n"
                "  isofit download sixs\n"
                "  isofit download srtmnet"
            )

        # Get number of cores
        n_cores = self.cores_spinbox.value()

        self.atm_progress_label.setText(f"Running ISOFIT with {n_cores} cores...\n(this may take 10-30 minutes)")
        QApplication.processEvents()

        # Run ISOFIT correction
        pipeline = isofit_module.AVIRIS3ISOFITCorrection(
            rdn_path=str(rad_path),
            ort_path=obs_path,
            output_path=str(output_path),
            n_cores=n_cores,
            cleanup=True
        )
        pipeline.run()

    def _run_6s_correction(self, rad_path, obs_path, output_path, method_idx):
        """Run 6S LUT or Simple DOS correction."""
        from qtpy.QtWidgets import QApplication

        self.atm_progress_label.setText("Loading atmospheric correction module...")
        QApplication.processEvents()

        # Import the 6S module
        try:
            import importlib.util
            atm_module_path = rad_path.parent / "aviris_atm_correction_v2.py"
            if not atm_module_path.exists():
                atm_module_path = Path("aviris_atm_correction_v2.py")
            if not atm_module_path.exists():
                atm_module_path = Path.home() / "Downloads" / "aviris_atm_correction_v2.py"

            if not atm_module_path.exists():
                raise FileNotFoundError(
                    "Could not find aviris_atm_correction_v2.py.\n"
                    "Please ensure it's in the same directory as the data."
                )

            import sys
            module_dir = str(atm_module_path.parent)
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)

            spec = importlib.util.spec_from_file_location("aviris_atm_correction_v2", atm_module_path)
            atm_module = importlib.util.module_from_spec(spec)
            sys.modules['aviris_atm_correction_v2'] = atm_module
            spec.loader.exec_module(atm_module)

        except Exception as e:
            raise ImportError(f"Failed to import atmospheric correction module: {e}")

        # Get options
        coastal = self.coastal_correction_cb.isChecked()
        uncertainty = self.uncertainty_cb.isChecked()
        aerosol = self.aerosol_combo.currentText()

        # Determine whether to use 6S
        use_6s = (method_idx == 1) and atm_module.HAS_PY6S
        if method_idx == 1 and not atm_module.HAS_PY6S:
            logger.warning("6S not available, falling back to Simple DOS")

        method_label = "6S LUT" if use_6s else "Simple DOS"
        self.atm_progress_label.setText(f"Processing with {method_label}...\n(this may take several minutes)")
        QApplication.processEvents()

        # Create processor and run
        processor = atm_module.AVIRISL2Processor(
            str(rad_path),
            obs_path,
            aerosol_model=aerosol,
            coastal_correction=coastal,
            estimate_uncertainty=uncertainty
        )

        processor.process(str(output_path), use_6s=use_6s)
        processor.close()

    def _load_corrected_data(self):
        """Load previously corrected data."""
        # Check for last corrected file
        if hasattr(self, '_last_corrected_path') and self._last_corrected_path.exists():
            filepath = str(self._last_corrected_path)
        else:
            filepath, _ = QFileDialog.getOpenFileName(
                self,
                "Select Corrected Reflectance File",
                str(self.viewer_app.filepath.parent),
                "NetCDF Files (*.nc);;All Files (*)"
            )
            if not filepath:
                return

        try:
            # Reload with new data
            self.viewer_app.reload_data(filepath)
            self._update_atm_status()
            QMessageBox.information(
                self,
                "Data Loaded",
                f"Successfully loaded corrected data:\n{Path(filepath).name}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load corrected data:\n{str(e)}"
            )


# =============================================================================
# Main Viewer Application
# =============================================================================

class HyperspectralViewer:
    """Main application class with all spectral analysis capabilities."""

    # Index definitions with physically correct wavelengths
    # References: USGS Spectral Library v7, Ninomiya (2003), Kühn (2004), Serrano (2002)
    # Mineral indices use continuum removal for quantitative accuracy
    INDEX_DEFINITIONS = {
        # =========================================================================
        # VEGETATION INDICES - Normalized Difference
        # =========================================================================
        'NDVI': {'type': 'nd', 'b1': 850, 'b2': 670, 'cmap': 'RdYlGn'},
        'NDRE': {'type': 'nd', 'b1': 790, 'b2': 720, 'cmap': 'RdYlGn'},
        'NDWI': {'type': 'nd', 'b1': 560, 'b2': 860, 'cmap': 'RdBu'},
        'NDMI': {'type': 'nd', 'b1': 860, 'b2': 1650, 'cmap': 'BrBG'},

        # =========================================================================
        # CLAY MINERALS - Continuum Removed (Al-OH absorption)
        # Reference: Hunt (1977), Crowley (1989), Swayze (2014)
        # =========================================================================
        'Clay (General)': {'type': 'continuum', 'feature': 2200, 'left': 2120, 'right': 2250, 'cmap': 'YlOrBr'},
        'Kaolinite': {'type': 'continuum', 'feature': 2205, 'left': 2120, 'right': 2250, 'cmap': 'Oranges'},
        'Alunite': {'type': 'continuum', 'feature': 2170, 'left': 2120, 'right': 2220, 'cmap': 'RdPu'},
        'Smectite': {'type': 'continuum', 'feature': 2200, 'left': 2120, 'right': 2290, 'cmap': 'YlOrRd'},
        'Illite': {'type': 'continuum', 'feature': 2195, 'left': 2120, 'right': 2250, 'cmap': 'OrRd'},

        # =========================================================================
        # CARBONATES - Continuum Removed (CO3 absorption)
        # Reference: Gaffey (1986), Ninomiya (2003)
        # =========================================================================
        'Carbonate': {'type': 'continuum', 'feature': 2330, 'left': 2250, 'right': 2380, 'cmap': 'cividis'},
        'Calcite': {'type': 'continuum', 'feature': 2340, 'left': 2290, 'right': 2390, 'cmap': 'Blues'},
        'Dolomite': {'type': 'continuum', 'feature': 2320, 'left': 2260, 'right': 2380, 'cmap': 'PuBu'},

        # =========================================================================
        # OTHER MINERALS
        # =========================================================================
        'Chlorite': {'type': 'continuum', 'feature': 2330, 'left': 2250, 'right': 2380, 'cmap': 'Greens'},

        # =========================================================================
        # IRON OXIDES - Ratio (electronic transitions)
        # Reference: Rowan & Mars (2003), Hunt (1977)
        # =========================================================================
        'Iron Oxide': {'type': 'ratio', 'b1': 650, 'b2': 450, 'cmap': 'Reds'},
        'Ferric Iron': {'type': 'ratio', 'b1': 750, 'b2': 860, 'cmap': 'hot'},
        'Ferrous Iron': {'type': 'ratio', 'b1': 860, 'b2': 1000, 'cmap': 'copper'},

        # =========================================================================
        # HYDROCARBONS - C-H Absorption
        # Reference: Kühn (2004), Cloutis (1989), Thorpe (2014)
        # =========================================================================
        'Hydrocarbon': {'type': 'continuum', 'feature': 1730, 'left': 1660, 'right': 1780, 'cmap': 'Purples'},
        'HC 2310': {'type': 'continuum', 'feature': 2310, 'left': 2260, 'right': 2350, 'cmap': 'Purples'},
        'Methane': {'type': 'ratio', 'b1': 2260, 'b2': 2300, 'cmap': 'magma'},
        'Oil Slick': {'type': 'continuum', 'feature': 1730, 'left': 1680, 'right': 1760, 'cmap': 'inferno'},

        # =========================================================================
        # AGRICULTURE / NITROGEN
        # Reference: Serrano (2002), Kokaly (1999)
        # =========================================================================
        'Protein': {'type': 'continuum', 'feature': 2170, 'left': 2100, 'right': 2230, 'cmap': 'Greens'},
        'Cellulose': {'type': 'continuum', 'feature': 2100, 'left': 2030, 'right': 2170, 'cmap': 'YlGn'},
        'Lignin': {'type': 'continuum', 'feature': 1680, 'left': 1620, 'right': 1740, 'cmap': 'BuGn'},
    }
    
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.data_loader = LazyHyperspectralData(filepath)
        self.setup_viewer()
        self.setup_controls()
        self.connect_signals()
        
    def setup_viewer(self):
        """Initialize the Napari viewer."""
        data_label = "Reflectance" if self.data_loader.data_type == 'reflectance' else "Radiance"
        self.viewer = napari.Viewer(title=f"Hyperspectral Explorer v4.0 - {data_label}")

        # Calculate memory requirements
        n_pixels = self.data_loader.n_rows * self.data_loader.n_cols
        n_elements = n_pixels * self.data_loader.n_bands
        memory_gb = (n_elements * 4) / (1024**3)  # float32 = 4 bytes

        logger.info(f"Dataset: {self.data_loader.n_bands} bands x "
                   f"{self.data_loader.n_rows} x {self.data_loader.n_cols} pixels")
        logger.info(f"Estimated memory for full cube: {memory_gb:.2f} GB")

        # Load cube (subset if large, with memory warning)
        MEMORY_THRESHOLD_GB = 4.0
        PIXEL_THRESHOLD = 10_000_000

        if memory_gb > MEMORY_THRESHOLD_GB:
            logger.warning(f"Large dataset ({memory_gb:.1f} GB) - loading spectral subset")
            # Load only first 50 bands for 3D visualization
            full_cube = self.data_loader.data_var[:50, :, :].astype(np.float32)
            self._show_memory_warning(memory_gb)
        elif n_pixels < PIXEL_THRESHOLD:
            full_cube = self.data_loader.get_full_cube()
        else:
            logger.info("Large spatial extent - loading band subset for cube visualization")
            full_cube = self.data_loader.data_var[:50, :, :].astype(np.float32)
        
        self.cube_layer = self.viewer.add_image(
            full_cube, name=f"{data_label} Cube", visible=False, colormap='viridis'
        )
        
        self.add_composite("True Color", 640, 550, 470)
        
        self.roi_layer = self.viewer.add_shapes(
            name="ROIs", edge_color='cyan', face_color='transparent', edge_width=2
        )

    def _show_memory_warning(self, memory_gb):
        """Show a warning dialog about memory usage for large datasets."""
        from qtpy.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Large Dataset Warning")
        msg.setText(f"This dataset requires approximately {memory_gb:.1f} GB of memory.")
        msg.setInformativeText(
            "To prevent memory issues:\n"
            "- Only a spectral subset (50 bands) is loaded for 3D cube visualization\n"
            "- Full spectral data is available for composites and indices via lazy loading\n"
            "- Consider using spatial subsetting for very large scenes\n\n"
            "All analysis tools remain fully functional."
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def setup_controls(self):
        """Add the control panel widget."""
        self.control_panel = HyperspectralControlPanel(self)
        self.control_panel.update_data_info(self.data_loader)
        self.viewer.window.add_dock_widget(self.control_panel, name="Controls", area="right")
        
    def connect_signals(self):
        """Connect viewer signals for wavelength display."""
        self.viewer.dims.events.current_step.connect(self._on_dims_change)

    def reload_data(self, filepath):
        """
        Reload the viewer with new data (e.g., after atmospheric correction).

        Args:
            filepath: Path to the new data file
        """
        logger.info(f"Reloading data from: {filepath}")

        # Store current view state
        try:
            current_camera = {
                'center': self.viewer.camera.center,
                'zoom': self.viewer.camera.zoom,
                'angles': self.viewer.camera.angles if hasattr(self.viewer.camera, 'angles') else None
            }
        except:
            current_camera = None

        # Close old data loader
        if hasattr(self, 'data_loader') and self.data_loader:
            self.data_loader.close()

        # Load new data
        self.filepath = Path(filepath)
        self.data_loader = LazyHyperspectralData(filepath)

        # Clear existing layers (except ROIs)
        layers_to_remove = [l for l in self.viewer.layers if l.name != "ROIs"]
        for layer in layers_to_remove:
            self.viewer.layers.remove(layer)

        # Reload cube
        data_label = "Reflectance" if self.data_loader.data_type == 'reflectance' else "Radiance"
        if self.data_loader.n_rows * self.data_loader.n_cols < 10_000_000:
            full_cube = self.data_loader.get_full_cube()
        else:
            logger.info("Large dataset - loading subset for cube visualization")
            full_cube = self.data_loader.data_var[:50, :, :].astype(np.float32)

        self.cube_layer = self.viewer.add_image(
            full_cube, name=f"{data_label} Cube", visible=False, colormap='viridis'
        )

        # Add true color composite
        self.add_composite("True Color", 640, 550, 470)

        # Update title
        self.viewer.title = f"Hyperspectral Explorer v4.0 - {data_label}"

        # Update control panel
        self.control_panel.update_data_info(self.data_loader)
        self.control_panel._update_atm_status()

        # Restore camera if possible
        if current_camera:
            try:
                self.viewer.camera.center = current_camera['center']
                self.viewer.camera.zoom = current_camera['zoom']
                if current_camera['angles']:
                    self.viewer.camera.angles = current_camera['angles']
            except:
                pass

        logger.info(f"Data reloaded: {self.data_loader.n_bands} bands, "
                   f"{self.data_loader.n_rows}x{self.data_loader.n_cols} pixels")

    def _on_dims_change(self, event):
        """Handle dimension slider changes."""
        if self.cube_layer.visible:
            band_idx = self.viewer.dims.current_step[0]
            if 0 <= band_idx < len(self.data_loader.wavelengths):
                wavelength = self.data_loader.wavelengths[band_idx]
                self.control_panel.update_wavelength_display(band_idx, wavelength)
    
    def normalize_band(self, band_data, percentile_low=PERCENTILE_LOW, percentile_high=PERCENTILE_HIGH):
        """Normalize band data to 0-1 using percentile stretch."""
        p_low, p_high = np.nanpercentile(band_data, (percentile_low, percentile_high))
        if p_high - p_low < 1e-10:
            return np.zeros_like(band_data)
        return np.clip((band_data - p_low) / (p_high - p_low), 0, 1)
        
    def add_composite(self, name, r_wl, g_wl, b_wl):
        """Add an RGB composite layer. Returns info dict for colorbar."""
        loader = self.data_loader
        r_band = loader.find_band(r_wl)
        g_band = loader.find_band(g_wl)
        b_band = loader.find_band(b_wl)
        
        actual_r = loader.wavelengths[r_band]
        actual_g = loader.wavelengths[g_band]
        actual_b = loader.wavelengths[b_band]
        
        logger.info(f"Creating {name}: R={actual_r:.0f}nm, G={actual_g:.0f}nm, B={actual_b:.0f}nm")
        
        r_raw = loader.get_band(r_band)
        g_raw = loader.get_band(g_band)
        b_raw = loader.get_band(b_band)
        
        r_range = np.nanpercentile(r_raw, (2, 98))
        g_range = np.nanpercentile(g_raw, (2, 98))
        b_range = np.nanpercentile(b_raw, (2, 98))
        
        r_data = self.normalize_band(r_raw)
        g_data = self.normalize_band(g_raw)
        b_data = self.normalize_band(b_raw)
        
        rgb = np.stack([r_data, g_data, b_data], axis=-1)
        
        existing = [l for l in self.viewer.layers if l.name == name]
        for layer in existing:
            self.viewer.layers.remove(layer)
            
        self.viewer.add_image(rgb, name=name, rgb=True)
        
        return {
            'name': name,
            'r_wl': actual_r, 'g_wl': actual_g, 'b_wl': actual_b,
            'r_range': r_range, 'g_range': g_range, 'b_range': b_range,
        }
    
    def _safe_divide(self, numerator, denominator, fill_value=np.nan, min_denom=None,
                      use_relative=True):
        """
        Perform division with proper handling of near-zero denominators.

        Args:
            numerator: Numerator array
            denominator: Denominator array
            fill_value: Value to use where division is invalid
            min_denom: Minimum denominator threshold (below this → invalid)
                       If None and use_relative=True, uses relative threshold
            use_relative: If True and min_denom is None, compute relative threshold

        Returns:
            Result array with invalid pixels set to fill_value
        """
        # Use relative threshold for better precision across data ranges
        if min_denom is None:
            if use_relative:
                # Relative threshold: 1e-8 of the data range
                denom_max = np.nanmax(np.abs(denominator))
                min_denom = max(denom_max * 1e-8, 1e-10)  # Floor at 1e-10
            else:
                min_denom = 1e-6

        # Create mask for valid pixels (denominator above threshold)
        valid_mask = np.abs(denominator) > min_denom

        # Safe division with small epsilon
        with np.errstate(invalid='ignore', divide='ignore'):
            result = np.where(valid_mask, numerator / denominator, fill_value)

        # Replace any infinities or NaNs that slipped through
        result = np.where(np.isfinite(result), result, fill_value)

        return result
    
    def _calculate_ratio_index(self, b1_data, b2_data, clip_range=(0.01, 10.0)):
        """
        Calculate a ratio index with proper clipping and masking.

        Ratio indices should typically be in range 0.5-2.0 for normal surfaces.
        Extreme values indicate data quality issues (zeros, saturation).
        """
        # Mask invalid pixels
        valid_mask = (b1_data > MIN_REFLECTANCE) & (b2_data > MIN_REFLECTANCE)

        # Calculate ratio
        ratio = self._safe_divide(b1_data, b2_data, fill_value=np.nan, min_denom=MIN_REFLECTANCE)

        # Clip to reasonable range
        ratio = np.clip(ratio, clip_range[0], clip_range[1])

        # Set invalid pixels to NaN (will be handled by napari)
        ratio = np.where(valid_mask, ratio, np.nan)

        # Count invalid pixels for logging
        n_invalid = np.sum(~valid_mask)
        pct_invalid = 100 * n_invalid / valid_mask.size
        if pct_invalid > 5:
            logger.warning(f"  ⚠ {pct_invalid:.1f}% pixels invalid (low reflectance)")

        return ratio

    def _calculate_continuum_removed_depth(self, feature_wl, left_wl, right_wl):
        """
        Calculate continuum-removed absorption depth for mineral indices.

        This is the physically correct approach for mineral detection:
        1. Get reflectance at the absorption feature wavelength
        2. Calculate a continuum line between shoulder wavelengths
        3. Compute depth = 1 - (R_feature / R_continuum)

        Depth values:
        - 0 = no absorption (flat spectrum)
        - >0 = absorption present (higher = deeper absorption = more mineral)
        - Typical range for clays: 0.02 - 0.15

        Args:
            feature_wl: Wavelength of the absorption feature (nm)
            left_wl: Left shoulder wavelength (nm)
            right_wl: Right shoulder wavelength (nm)

        Returns:
            2D array of absorption depths
        """
        loader = self.data_loader

        # Use batch interpolation for efficiency (loads each band only once)
        bands = loader.get_multiple_interpolated_bands([feature_wl, left_wl, right_wl])
        r_feature = bands[float(feature_wl)].astype(np.float64)
        r_left = bands[float(left_wl)].astype(np.float64)
        r_right = bands[float(right_wl)].astype(np.float64)

        # Calculate continuum at feature wavelength via linear interpolation
        # between left and right shoulders
        fraction = (feature_wl - left_wl) / (right_wl - left_wl)
        r_continuum = r_left * (1 - fraction) + r_right * fraction

        # Calculate absorption depth: 1 - (R_feature / R_continuum)
        # Positive values = absorption, higher = stronger
        depth = 1.0 - self._safe_divide(r_feature, r_continuum, fill_value=np.nan)

        # Mask invalid pixels (low signal or negative depth due to noise)
        valid_mask = (r_continuum > MIN_REFLECTANCE) & (r_feature > 0)

        # Clip to physical range (0 to 1, though typically 0 to 0.3 for minerals)
        depth = np.clip(depth, 0.0, 1.0)
        depth = np.where(valid_mask, depth, np.nan)

        logger.info(f"  Continuum removal: feature={feature_wl}nm, shoulders={left_wl}-{right_wl}nm")
        logger.info(f"  Absorption depth range: {np.nanmin(depth):.4f} to {np.nanmax(depth):.4f}")

        return depth

    def _robust_percentile(self, data, percentiles):
        """
        Calculate percentiles with sampling for large arrays.

        For arrays larger than LARGE_ARRAY_THRESHOLD, uses random sampling
        to improve performance while maintaining accuracy.

        Args:
            data: Input array (will be flattened)
            percentiles: List of percentiles to compute (e.g., [5, 95])

        Returns:
            List of percentile values
        """
        valid_data = data[np.isfinite(data)]

        if len(valid_data) == 0:
            return [0.0] * len(percentiles)

        if len(valid_data) > LARGE_ARRAY_THRESHOLD:
            # Sample for large datasets to improve performance
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            sample_idx = rng.choice(len(valid_data), size=SAMPLE_SIZE, replace=False)
            sample = valid_data[sample_idx]
            return [float(np.percentile(sample, p)) for p in percentiles]
        else:
            return [float(np.percentile(valid_data, p)) for p in percentiles]

    def _estimate_index_uncertainty(self, b1_data, b2_data, index_type='ratio'):
        """
        Estimate uncertainty in index calculation based on SNR.

        For ratio R = b1/b2:
            σ_R/R = sqrt((σ_b1/b1)² + (σ_b2/b2)²)

        For normalized difference ND = (b1-b2)/(b1+b2):
            Uses error propagation through the formula.

        Returns:
            Dictionary with uncertainty statistics
        """
        # Estimate noise from local variance
        from scipy.ndimage import uniform_filter

        def estimate_noise(data):
            local_mean = uniform_filter(data.astype(np.float64), size=5)
            local_var = uniform_filter((data - local_mean)**2, size=5)
            return np.sqrt(local_var)

        sigma_b1 = estimate_noise(b1_data)
        sigma_b2 = estimate_noise(b2_data)

        # Relative uncertainties
        rel_sigma_b1 = self._safe_divide(sigma_b1, np.abs(b1_data), fill_value=1.0)
        rel_sigma_b2 = self._safe_divide(sigma_b2, np.abs(b2_data), fill_value=1.0)

        if index_type == 'ratio':
            # σ_R/R = sqrt(σ_b1²/b1² + σ_b2²/b2²)
            rel_uncertainty = np.sqrt(rel_sigma_b1**2 + rel_sigma_b2**2)
        else:
            # For ND, uncertainty is more complex but we approximate
            # σ_ND ≈ 2 * sqrt(σ_b1² + σ_b2²) / (b1 + b2)
            abs_uncertainty = 2 * np.sqrt(sigma_b1**2 + sigma_b2**2)
            denominator = b1_data + b2_data
            rel_uncertainty = self._safe_divide(abs_uncertainty, np.abs(denominator), fill_value=1.0)

        # Summary statistics
        valid = np.isfinite(rel_uncertainty)
        if np.any(valid):
            median_uncertainty = float(np.median(rel_uncertainty[valid]))
            p95_uncertainty = float(np.percentile(rel_uncertainty[valid], 95))
        else:
            median_uncertainty = 1.0
            p95_uncertainty = 1.0

        return {
            'median_relative_uncertainty': median_uncertainty,
            'p95_relative_uncertainty': p95_uncertainty,
            'uncertainty_map': rel_uncertainty.astype(np.float32)
        }
        
    def _show_validation_warnings(self, warnings, show_popup=True):
        """
        Display validation warnings in UI and log.

        Args:
            warnings: List of warning strings
            show_popup: If True and warnings are severe, show a dialog
        """
        if not warnings:
            return

        for w in warnings:
            logger.warning(f"  ⚠ {w}")

        # Show popup for multiple or severe warnings
        if show_popup and len(warnings) > 0:
            # Check if any warning mentions atmospheric or blocked
            severe = any('atmospheric' in w.lower() or 'blocked' in w.lower() for w in warnings)
            if severe or len(warnings) >= 2:
                msg = QMessageBox(self.viewer.window._qt_window)
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Data Quality Warnings")
                msg.setText("The following issues were detected:")
                msg.setDetailedText("\n".join(warnings))
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

    def _calculate_index_core(self, index_type, b1_wl=None, b2_wl=None,
                               feature_wl=None, left_wl=None, right_wl=None,
                               estimate_uncertainty=False):
        """
        Core index calculation logic - unified for all index types.

        Args:
            index_type: 'nd', 'ratio', or 'continuum'
            b1_wl, b2_wl: Wavelengths for ratio/nd indices
            feature_wl, left_wl, right_wl: Wavelengths for continuum removal
            estimate_uncertainty: If True, compute uncertainty estimate

        Returns:
            Tuple of (index_data, uncertainty_dict or None, validation_warnings)
        """
        loader = self.data_loader
        uncertainty = None
        all_warnings = []

        if index_type == 'continuum':
            # Continuum-removed absorption depth for minerals
            validation1 = loader.validate_index_calculation(feature_wl, left_wl, block_atmospheric=False)
            validation2 = loader.validate_index_calculation(feature_wl, right_wl, block_atmospheric=False)
            all_warnings = validation1['warnings'] + validation2['warnings']

            index_data = self._calculate_continuum_removed_depth(feature_wl, left_wl, right_wl)

        else:
            # Standard ratio or normalized difference
            validation = loader.validate_index_calculation(b1_wl, b2_wl, index_type, block_atmospheric=True)

            if not validation['valid']:
                return None, None, validation['errors']

            all_warnings = validation['warnings']

            # Use batch interpolation for efficiency
            bands = loader.get_multiple_interpolated_bands([b1_wl, b2_wl])
            b1_data = bands[float(b1_wl)].astype(np.float64)
            b2_data = bands[float(b2_wl)].astype(np.float64)

            logger.info(f"  Using interpolated bands at: {b1_wl:.0f}nm / {b2_wl:.0f}nm")

            if index_type == 'nd':
                numerator = b1_data - b2_data
                denominator = b1_data + b2_data
                index_data = self._safe_divide(numerator, denominator, fill_value=np.nan)
                index_data = np.clip(index_data, -1.0, 1.0)
            elif index_type == 'ratio':
                index_data = self._calculate_ratio_index(b1_data, b2_data)
            else:
                logger.error(f"Unknown index type: {index_type}")
                return None, None, [f"Unknown index type: {index_type}"]

            # Estimate uncertainty if requested
            if estimate_uncertainty:
                uncertainty = self._estimate_index_uncertainty(b1_data, b2_data, index_type)
                logger.info(f"  Uncertainty: median={uncertainty['median_relative_uncertainty']:.1%}, "
                           f"p95={uncertainty['p95_relative_uncertainty']:.1%}")

        return index_data, uncertainty, all_warnings

    def _display_index_layer(self, index_data, layer_name, cmap, uncertainty=None):
        """
        Display calculated index as a napari layer.

        Args:
            index_data: 2D array of index values
            layer_name: Name for the layer
            cmap: Colormap name
            uncertainty: Optional uncertainty dict to include in result

        Returns:
            Dict with layer info including clim
        """
        # Remove existing layer with same name
        existing = [l for l in self.viewer.layers if l.name == layer_name]
        for layer in existing:
            self.viewer.layers.remove(layer)

        # Calculate contrast limits using robust percentiles
        clim = self._robust_percentile(index_data, [ROBUST_PERCENTILE_LOW, ROBUST_PERCENTILE_HIGH])

        if clim[1] <= clim[0]:
            logger.warning(f"  ⚠ Constant data detected (all values ≈ {clim[0]:.3f})")
            clim = [clim[0], clim[0] + 1e-6]

        # Replace NaN with 0 for display
        display_data = np.nan_to_num(index_data, nan=0.0).astype(np.float32)

        self.viewer.add_image(display_data, name=layer_name,
                              colormap=cmap, contrast_limits=clim)

        logger.info(f"  Range: {np.nanmin(index_data):.3f} to {np.nanmax(index_data):.3f}")

        result = {'clim': clim}
        if uncertainty:
            result['uncertainty'] = uncertainty
        return result

    def calculate_index(self, index_name, estimate_uncertainty=False):
        """
        Calculate a pre-defined spectral index.

        Supports three index types:
        - 'nd': Normalized difference (b1-b2)/(b1+b2)
        - 'ratio': Simple band ratio b1/b2
        - 'continuum': Continuum-removed absorption depth (for minerals)

        Args:
            index_name: Name of the index (must be in INDEX_DEFINITIONS)
            estimate_uncertainty: If True, also compute uncertainty estimate

        Returns:
            Dictionary with index info, or None on error
        """
        if index_name not in self.INDEX_DEFINITIONS:
            logger.error(f"Unknown index: {index_name}")
            return None

        idx_def = self.INDEX_DEFINITIONS[index_name]
        logger.info(f"Calculating {index_name}...")

        idx_type = idx_def['type']

        # Extract wavelength parameters based on type
        if idx_type == 'continuum':
            index_data, uncertainty, warnings = self._calculate_index_core(
                idx_type,
                feature_wl=idx_def['feature'],
                left_wl=idx_def['left'],
                right_wl=idx_def['right'],
                estimate_uncertainty=estimate_uncertainty
            )
            actual_b1, actual_b2 = idx_def['feature'], idx_def['right']
        else:
            index_data, uncertainty, warnings = self._calculate_index_core(
                idx_type,
                b1_wl=idx_def['b1'],
                b2_wl=idx_def['b2'],
                estimate_uncertainty=estimate_uncertainty
            )
            actual_b1, actual_b2 = idx_def['b1'], idx_def['b2']

        # Handle errors
        if index_data is None:
            for err in warnings:
                logger.error(f"  ✗ {err}")
            QMessageBox.warning(None, "Calculation Blocked", "\n".join(warnings))
            return None

        # Show warnings in UI
        self._show_validation_warnings(warnings)

        # Display layer
        layer_name = f"{index_name} Index"
        layer_info = self._display_index_layer(index_data, layer_name, idx_def['cmap'], uncertainty)

        return {
            'name': index_name,
            'cmap': idx_def['cmap'],
            'clim': layer_info['clim'],
            'type': idx_type,
            'b1': actual_b1,
            'b2': actual_b2,
            'uncertainty': layer_info.get('uncertainty')
        }

    def calculate_custom_index(self, name, b1_wl, b2_wl, index_type='ratio', cmap='viridis',
                                feature_wl=None, left_wl=None, right_wl=None,
                                block_atmospheric=True, estimate_uncertainty=False):
        """
        Calculate a custom spectral index from arbitrary wavelengths.

        Now supports all index types including continuum removal.

        Args:
            name: Display name for the index
            b1_wl, b2_wl: Wavelengths for ratio/nd indices
            index_type: 'ratio', 'nd', or 'continuum'
            cmap: Colormap name
            feature_wl, left_wl, right_wl: For continuum removal (optional)
            block_atmospheric: If True, refuse to calculate in severe atmospheric bands
            estimate_uncertainty: If True, compute uncertainty estimate

        Returns:
            Dictionary with index info, or None on error
        """
        logger.info(f"Calculating custom index: {name}...")

        # Calculate using unified core
        if index_type == 'continuum':
            if feature_wl is None or left_wl is None or right_wl is None:
                logger.error("Continuum removal requires feature_wl, left_wl, and right_wl")
                return None
            index_data, uncertainty, warnings = self._calculate_index_core(
                index_type,
                feature_wl=feature_wl,
                left_wl=left_wl,
                right_wl=right_wl,
                estimate_uncertainty=estimate_uncertainty
            )
            actual_b1, actual_b2 = feature_wl, right_wl
        else:
            index_data, uncertainty, warnings = self._calculate_index_core(
                index_type,
                b1_wl=b1_wl,
                b2_wl=b2_wl,
                estimate_uncertainty=estimate_uncertainty
            )
            actual_b1, actual_b2 = b1_wl, b2_wl

        # Handle errors
        if index_data is None:
            for err in warnings:
                logger.error(f"  ✗ {err}")
            if block_atmospheric:
                QMessageBox.warning(None, "Calculation Blocked", "\n".join(warnings))
            return None

        # Show warnings in UI
        self._show_validation_warnings(warnings, show_popup=block_atmospheric)

        # Display layer
        layer_name = f"{name} Index"
        layer_info = self._display_index_layer(index_data, layer_name, cmap, uncertainty)

        return {
            'name': name,
            'cmap': cmap,
            'clim': layer_info['clim'],
            'type': index_type,
            'b1': actual_b1,
            'b2': actual_b2,
            'uncertainty': layer_info.get('uncertainty')
        }
        
    def set_roi_mode(self, mode):
        """Set the ROI drawing mode."""
        self.viewer.layers.selection.active = self.roi_layer
        self.roi_layer.mode = 'add_rectangle' if mode == 'rectangle' else 'add_polygon'
        logger.info(f"ROI mode: {mode}")
        
    def extract_roi_spectra(self):
        """Extract mean spectra from all ROIs."""
        if len(self.roi_layer.data) == 0:
            logger.warning("No ROIs defined")
            return []
            
        spectra = []
        loader = self.data_loader
        
        for i, shape in enumerate(self.roi_layer.data):
            min_row = max(0, int(np.floor(np.min(shape[:, 0]))))
            max_row = min(loader.n_rows, int(np.ceil(np.max(shape[:, 0]))))
            min_col = max(0, int(np.floor(np.min(shape[:, 1]))))
            max_col = min(loader.n_cols, int(np.ceil(np.max(shape[:, 1]))))
            
            region = loader.get_cube_region(slice(min_row, max_row), slice(min_col, max_col))
            mean_spectrum = np.nanmean(region, axis=(1, 2))
            std_spectrum = np.nanstd(region, axis=(1, 2))
            
            roi_name = f"ROI {i+1}"
            spectra.append((roi_name, loader.wavelengths, mean_spectrum, std_spectrum))
            logger.info(f"{roi_name}: {max_row - min_row} x {max_col - min_col} pixels")
            
        return spectra
        
    def clear_rois(self):
        self.roi_layer.data = []
        logger.info("ROIs cleared")
        
    def clear_composites(self):
        to_remove = [l for l in self.viewer.layers if l.name not in [self.cube_layer.name, "ROIs"]]
        for layer in to_remove:
            self.viewer.layers.remove(layer)
        logger.info("Composites cleared")
        
    def toggle_cube(self):
        self.cube_layer.visible = not self.cube_layer.visible
        logger.info(f"{self.cube_layer.name}: {'visible' if self.cube_layer.visible else 'hidden'}")
        if self.cube_layer.visible:
            band_idx = self.viewer.dims.current_step[0]
            if 0 <= band_idx < len(self.data_loader.wavelengths):
                self.control_panel.update_wavelength_display(band_idx, self.data_loader.wavelengths[band_idx])
                
    def run(self):
        napari.run()
    
    def close(self):
        self.data_loader.close()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        from qtpy.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        filepath, _ = QFileDialog.getOpenFileName(None, "Open AVIRIS-3 NetCDF File", "", "NetCDF Files (*.nc);;All Files (*)")
        if not filepath:
            logger.info("No file selected. Exiting.")
            sys.exit(0)

    viewer = None
    try:
        viewer = HyperspectralViewer(filepath)
        viewer.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
