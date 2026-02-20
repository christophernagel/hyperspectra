"""
USGS Spectral Library Version 7 Parser

Parses ASCII spectral files from USGS v7 and resamples to HyperspecI wavelength grids.

USGS splib07a ASCII format:
- Wavelengths stored in separate file (e.g., splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt)
- Spectrum files: single column of reflectance values (0-1 scale)
- First line is metadata header (e.g., "splib07a Record=18587: Cotton_Fabric GDS437 White")
- Bad bands marked as -1.23e34

File naming convention:
- splib07a_<Material>_<SampleID>_<Spectrometer>_<RefType>.txt
- ASDFRa = ASD FieldSpec standard resolution (0.35-2.5µm, 2151 channels)
- BECKa = Beckman (0.2-3.0µm)
- NIC4a = Nicolet FTIR (1.12-216µm)
- AREF = Absolute reflectance, RREF = Relative reflectance

Output: JSON files with spectra resampled to:
- D1: 400-1000nm, 10nm intervals, 61 channels
- D2: 400-1700nm, 10nm intervals, 131 channels

Usage:
    python usgs_parser.py path/to/usgs_splib07/ output_dir/

    # Or use programmatically:
    from usgs_parser import USGSLibraryParser
    parser = USGSLibraryParser("path/to/usgs_splib07")
    parser.parse_chapter_a()
    parser.export_resampled("output.json", target="D2")
"""

import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# HyperspecI wavelength grids
WAVELENGTHS_D1 = np.linspace(400, 1000, 61)  # nm
WAVELENGTHS_D2 = np.linspace(400, 1700, 131)  # nm

# ASD wavelength grid (350-2500nm, 1nm intervals, 2151 channels)
# Loaded from file or generated
ASD_WAVELENGTHS_UM = np.linspace(0.35, 2.5, 2151)  # Default, can be overridden


class USGSSpectrum:
    """Single spectrum from USGS library."""

    def __init__(self, name: str, wavelengths_um: np.ndarray, reflectance: np.ndarray,
                 metadata: Optional[Dict] = None):
        """
        Parameters
        ----------
        name : str
            Material/sample name
        wavelengths_um : np.ndarray
            Wavelengths in micrometers
        reflectance : np.ndarray
            Reflectance values (0-1 scale)
        metadata : dict, optional
            Additional metadata (chapter, description, etc.)
        """
        self.name = name
        self.wavelengths_um = wavelengths_um
        self.wavelengths_nm = wavelengths_um * 1000  # Convert to nm
        self.reflectance = reflectance
        self.metadata = metadata or {}

        # Mask bad bands (USGS uses -1.23e34 for bad data)
        self.valid_mask = reflectance > -1e30

    def resample(self, target_wavelengths_nm: np.ndarray) -> np.ndarray:
        """
        Resample spectrum to target wavelength grid.

        Uses linear interpolation. Returns NaN for wavelengths outside
        the original spectrum's range.

        Parameters
        ----------
        target_wavelengths_nm : np.ndarray
            Target wavelengths in nanometers

        Returns
        -------
        np.ndarray
            Resampled reflectance values
        """
        # Use only valid data points
        valid_wl = self.wavelengths_nm[self.valid_mask]
        valid_refl = self.reflectance[self.valid_mask]

        if len(valid_wl) < 2:
            return np.full_like(target_wavelengths_nm, np.nan)

        # Interpolate
        resampled = np.interp(
            target_wavelengths_nm,
            valid_wl,
            valid_refl,
            left=np.nan,
            right=np.nan
        )

        return resampled

    def to_d1(self) -> np.ndarray:
        """Resample to HyperspecI D1 grid (400-1000nm, 61 channels)."""
        return self.resample(WAVELENGTHS_D1)

    def to_d2(self) -> np.ndarray:
        """Resample to HyperspecI D2 grid (400-1700nm, 131 channels)."""
        return self.resample(WAVELENGTHS_D2)

    def get_coverage(self) -> Tuple[float, float]:
        """Return wavelength coverage in nm."""
        valid_wl = self.wavelengths_nm[self.valid_mask]
        if len(valid_wl) == 0:
            return (np.nan, np.nan)
        return (float(valid_wl.min()), float(valid_wl.max()))


def load_wavelengths_file(filepath: Path) -> np.ndarray:
    """
    Load wavelengths from a USGS wavelength file.

    Format: Single column of wavelengths in micrometers, first line is header.

    Parameters
    ----------
    filepath : Path
        Path to wavelength file (e.g., splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt)

    Returns
    -------
    np.ndarray
        Wavelengths in micrometers
    """
    wavelengths = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:  # Skip header
                continue
            if not line:
                continue
            try:
                wl = float(line)
                wavelengths.append(wl)
            except ValueError:
                continue
    return np.array(wavelengths)


def parse_usgs_ascii(filepath: Path, wavelengths_um: np.ndarray) -> Optional[USGSSpectrum]:
    """
    Parse a single USGS splib07a ASCII spectrum file.

    Format: Single column of reflectance values
    - Line 1: Header with metadata (e.g., "splib07a Record=18587: Cotton_Fabric GDS437 White")
    - Lines 2+: Reflectance values (fraction, 0-1)
    - Bad bands: -1.23e34

    Parameters
    ----------
    filepath : Path
        Path to .txt ASCII file
    wavelengths_um : np.ndarray
        Wavelength array in micrometers (from separate wavelength file)

    Returns
    -------
    USGSSpectrum or None if parsing fails
    """
    try:
        reflectances = []
        header = None

        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                if i == 0:
                    # First line is header with metadata
                    header = line
                    continue

                try:
                    refl = float(line)
                    reflectances.append(refl)
                except ValueError:
                    continue

        if len(reflectances) < 10:
            logger.warning(f"Too few data points in {filepath}")
            return None

        # Validate length matches wavelength array
        if len(reflectances) != len(wavelengths_um):
            logger.warning(f"Length mismatch in {filepath}: {len(reflectances)} vs {len(wavelengths_um)} wavelengths")
            # Truncate or pad if close
            if abs(len(reflectances) - len(wavelengths_um)) <= 10:
                min_len = min(len(reflectances), len(wavelengths_um))
                reflectances = reflectances[:min_len]
                wavelengths_um = wavelengths_um[:min_len]
            else:
                return None

        # Extract material name from header or filename
        name = filepath.stem
        # Remove prefix
        name = re.sub(r'^splib07a_', '', name)
        # Remove spectrometer suffix
        name = re.sub(r'_ASD\w*_\w+$', '', name)
        name = re.sub(r'_BECK\w*_\w+$', '', name)
        name = re.sub(r'_NIC\w*_\w+$', '', name)
        # Clean up
        name = name.replace('_', ' ')

        return USGSSpectrum(
            name=name,
            wavelengths_um=wavelengths_um.copy(),
            reflectance=np.array(reflectances),
            metadata={
                'source_file': str(filepath),
                'header': header,
            }
        )

    except Exception as e:
        logger.error(f"Failed to parse {filepath}: {e}")
        return None


class USGSLibraryParser:
    """Parser for USGS Spectral Library v7 (splib07a original resolution data)."""

    # Materials of interest for indoor scenes (Chapter A)
    INDOOR_MATERIALS = {
        # Plastics
        'plastic': ['polyethylene', 'polypropylene', 'pvc', 'polystyrene',
                    'nylon', 'acrylic', 'abs', 'pmma', 'pet', 'hdpe', 'ldpe'],
        # Textiles
        'fabric': ['cotton', 'polyester', 'wool', 'silk', 'rayon',
                   'fabric', 'cloth', 'textile', 'burlap'],
        # Paints and coatings
        'paint': ['paint', 'pigment', 'coating', 'enamel', 'latex',
                  'cadmium', 'cobalt', 'umber', 'alizarin', 'cerulean',
                  'ultramarine', 'viridian', 'ochre', 'sienna'],
        # Wood and paper
        'wood': ['wood', 'plywood', 'mdf', 'particle', 'paper', 'cardboard',
                 'cedar', 'shake'],
        # Construction
        'construction': ['concrete', 'asphalt', 'brick', 'tile', 'glass',
                        'aluminum', 'steel', 'roofing', 'shingle', 'tar',
                        'cinder', 'block', 'girder'],
        # Other indoor
        'other': ['rubber', 'leather', 'vinyl', 'foam', 'insulation']
    }

    def __init__(self, library_path: str):
        """
        Initialize parser with path to extracted USGS library.

        Parameters
        ----------
        library_path : str
            Path to extracted usgs_splib07 directory
        """
        self.library_path = Path(library_path)
        self.spectra: Dict[str, USGSSpectrum] = {}

        # Load ASD wavelength file
        self.asd_wavelengths = self._load_asd_wavelengths()

    def _load_asd_wavelengths(self) -> np.ndarray:
        """Load the ASD wavelength file (350-2500nm, 2151 channels)."""
        wl_file = self.library_path / "ASCIIdata" / "ASCIIdata_splib07a" / \
                  "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"

        if wl_file.exists():
            wavelengths = load_wavelengths_file(wl_file)
            logger.info(f"Loaded {len(wavelengths)} wavelengths from ASD file")
            return wavelengths
        else:
            # Generate default
            logger.warning(f"Wavelength file not found at {wl_file}, using default")
            return np.linspace(0.35, 2.5, 2151)

    def find_chapter_a_files(self, spectrometer: str = 'ASD') -> List[Path]:
        """
        Find ASCII spectrum files for Chapter A (Artificial Materials).

        Parameters
        ----------
        spectrometer : str
            Spectrometer type: 'ASD' (350-2500nm), 'BECK' (200-3000nm), or 'all'

        Returns
        -------
        List of paths to ASCII spectrum files
        """
        chapter_dir = self.library_path / "ASCIIdata" / "ASCIIdata_splib07a" / \
                      "ChapterA_ArtificialMaterials"

        if not chapter_dir.exists():
            logger.error(f"Chapter A directory not found: {chapter_dir}")
            return []

        # Find all .txt files
        all_files = list(chapter_dir.glob("*.txt"))

        if spectrometer == 'all':
            files = all_files
        elif spectrometer == 'ASD':
            # Filter for ASD spectrometer (best coverage for HyperspecI)
            files = [f for f in all_files if '_ASD' in f.name and '_AREF' in f.name]
        elif spectrometer == 'BECK':
            files = [f for f in all_files if '_BECK' in f.name and '_AREF' in f.name]
        else:
            files = all_files

        logger.info(f"Found {len(files)} {spectrometer} files in Chapter A")
        return files

    def parse_chapter_a(self, include_all: bool = False) -> int:
        """
        Parse Chapter A (Artificial materials) ASD spectra.

        Parameters
        ----------
        include_all : bool
            If True, include all materials. If False, filter for indoor-relevant.

        Returns
        -------
        int : Number of spectra successfully parsed
        """
        files = self.find_chapter_a_files(spectrometer='ASD')

        count = 0
        for filepath in files:
            spectrum = parse_usgs_ascii(filepath, self.asd_wavelengths)
            if spectrum is None:
                continue

            name_lower = spectrum.name.lower()
            category = self._categorize_material(name_lower)

            if include_all or category:
                spectrum.metadata['category'] = category or 'other'
                # Use clean name as key (avoid duplicates)
                key = spectrum.name
                if key in self.spectra:
                    # Append sample ID to make unique
                    key = f"{key} ({filepath.stem[-10:]})"
                self.spectra[key] = spectrum
                count += 1

        logger.info(f"Parsed {count} spectra from Chapter A")
        return count

    def _categorize_material(self, name: str) -> Optional[str]:
        """Categorize material by name matching."""
        for category, keywords in self.INDOOR_MATERIALS.items():
            for keyword in keywords:
                if keyword in name:
                    return category
        return None

    def parse_all_indoor(self, include_all_chapter_a: bool = True) -> int:
        """
        Parse spectra from all chapters that might be indoor-relevant.

        Parameters
        ----------
        include_all_chapter_a : bool
            If True, include all Chapter A materials (recommended).
            If False, only include materials matching INDOOR_MATERIALS keywords.

        Returns
        -------
        int : Total number of spectra parsed
        """
        total = 0

        # Primary: Chapter A (Artificial) - include all since they're all manmade
        total += self.parse_chapter_a(include_all=include_all_chapter_a)

        logger.info(f"Total spectra parsed: {total}")
        return total

    def find_chapter_files(self, chapter: str, spectrometer: str = 'ASD') -> List[Path]:
        """
        Find ASCII spectrum files for any chapter.

        Parameters
        ----------
        chapter : str
            Chapter name: 'A' (Artificial), 'M' (Minerals), 'V' (Vegetation),
            'S' (Soils), 'C' (Coatings), 'L' (Liquids), 'O' (Organics)
        spectrometer : str
            'ASD' (350-2500nm), 'BECK' (200-3000nm), or 'all'

        Returns
        -------
        List of paths to ASCII spectrum files
        """
        chapter_names = {
            'A': 'ChapterA_ArtificialMaterials',
            'M': 'ChapterM_Minerals',
            'V': 'ChapterV_Vegetation',
            'S': 'ChapterS_SoilsAndMixtures',
            'C': 'ChapterC_Coatings',
            'L': 'ChapterL_Liquids',
            'O': 'ChapterO_OrganicCompounds',
        }

        chapter_dir = self.library_path / "ASCIIdata" / "ASCIIdata_splib07a" / \
                      chapter_names.get(chapter, f'Chapter{chapter}')

        if not chapter_dir.exists():
            logger.error(f"Chapter directory not found: {chapter_dir}")
            return []

        all_files = list(chapter_dir.glob("*.txt"))

        if spectrometer == 'all':
            files = all_files
        elif spectrometer == 'ASD':
            files = [f for f in all_files if '_ASD' in f.name and '_AREF' in f.name]
        elif spectrometer == 'BECK':
            files = [f for f in all_files if '_BECK' in f.name and '_AREF' in f.name]
        else:
            files = all_files

        logger.info(f"Found {len(files)} {spectrometer} files in Chapter {chapter}")
        return files

    def parse_chapter(self, chapter: str, category: Optional[str] = None) -> int:
        """
        Parse spectra from any chapter.

        Parameters
        ----------
        chapter : str
            Chapter letter (A, M, V, S, C, L, O)
        category : str, optional
            Category to assign (default: derived from chapter)

        Returns
        -------
        int : Number of spectra parsed
        """
        default_categories = {
            'A': 'artificial',
            'M': 'mineral',
            'V': 'vegetation',
            'S': 'soil',
            'C': 'coating',
            'L': 'liquid',
            'O': 'organic',
        }

        files = self.find_chapter_files(chapter, spectrometer='ASD')
        cat = category or default_categories.get(chapter, 'unknown')

        count = 0
        for filepath in files:
            spectrum = parse_usgs_ascii(filepath, self.asd_wavelengths)
            if spectrum is None:
                continue

            spectrum.metadata['category'] = cat
            key = spectrum.name
            if key in self.spectra:
                key = f"{key} ({filepath.stem[-10:]})"
            self.spectra[key] = spectrum
            count += 1

        logger.info(f"Parsed {count} spectra from Chapter {chapter}")
        return count

    def export_resampled(self, output_path: str, target: str = 'D2') -> None:
        """
        Export resampled spectra to JSON.

        Parameters
        ----------
        output_path : str
            Output JSON file path
        target : str
            'D1' (400-1000nm) or 'D2' (400-1700nm)
        """
        if target == 'D1':
            wavelengths = WAVELENGTHS_D1
        else:
            wavelengths = WAVELENGTHS_D2

        output = {
            'metadata': {
                'source': 'USGS Spectral Library Version 7',
                'citation': 'Kokaly et al., 2017, USGS Data Series 1035',
                'doi': 'https://doi.org/10.3133/ds1035',
                'target_grid': target,
                'wavelengths_nm': wavelengths.tolist(),
                'n_channels': len(wavelengths),
            },
            'spectra': {}
        }

        for name, spectrum in self.spectra.items():
            resampled = spectrum.resample(wavelengths)

            # Skip if too many NaN values
            valid_fraction = np.sum(~np.isnan(resampled)) / len(resampled)
            if valid_fraction < 0.5:
                logger.warning(f"Skipping {name}: only {valid_fraction:.0%} valid")
                continue

            # Replace NaN with -1 for JSON (will be masked on load)
            resampled_clean = np.where(np.isnan(resampled), -1, resampled)

            coverage = spectrum.get_coverage()
            output['spectra'][name] = {
                'reflectance': resampled_clean.tolist(),
                'category': spectrum.metadata.get('category', 'unknown'),
                'coverage_nm': coverage,
                'source_file': spectrum.metadata.get('source_file', ''),
            }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Exported {len(output['spectra'])} spectra to {output_path}")

    def get_summary(self) -> Dict:
        """Return summary statistics of parsed spectra."""
        categories = {}
        for name, spectrum in self.spectra.items():
            cat = spectrum.metadata.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        return {
            'total_spectra': len(self.spectra),
            'by_category': categories,
            'materials': list(self.spectra.keys())
        }


def load_resampled_library(json_path: str) -> Dict:
    """
    Load a resampled spectral library from JSON.

    Returns dict with 'wavelengths' and 'spectra' (name -> reflectance array)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    wavelengths = np.array(data['metadata']['wavelengths_nm'])

    spectra = {}
    for name, spec_data in data['spectra'].items():
        refl = np.array(spec_data['reflectance'])
        # Mask bad values
        refl = np.where(refl < 0, np.nan, refl)
        spectra[name] = {
            'reflectance': refl,
            'category': spec_data.get('category', 'unknown'),
        }

    return {
        'wavelengths': wavelengths,
        'spectra': spectra,
        'metadata': data['metadata']
    }


def match_spectrum(unknown: np.ndarray, library: Dict, top_n: int = 5,
                   method: str = 'sam') -> List[Tuple[str, float]]:
    """
    Match an unknown spectrum against the reference library.

    Parameters
    ----------
    unknown : np.ndarray
        Unknown spectrum to match (same length as library wavelengths)
    library : Dict
        Loaded library from load_resampled_library()
    top_n : int
        Number of top matches to return
    method : str
        Matching method: 'sam' (spectral angle), 'correlation', 'euclidean'

    Returns
    -------
    List of (material_name, similarity_score) tuples, sorted by best match.
    For 'sam', lower is better. For 'correlation', higher is better.

    Note
    ----
    HyperspecI reconstructed spectra may not match reference spectra well
    due to SRNet smoothing and reconstruction artifacts. Use these results
    as similarity indicators, not definitive identifications.
    """
    matches = []

    # Normalize unknown spectrum
    unknown_valid = np.nan_to_num(unknown, nan=0)
    unknown_norm = unknown_valid / (np.linalg.norm(unknown_valid) + 1e-10)

    for name, spec_data in library['spectra'].items():
        ref = spec_data['reflectance']
        ref_valid = np.nan_to_num(ref, nan=0)
        ref_norm = ref_valid / (np.linalg.norm(ref_valid) + 1e-10)

        if method == 'sam':
            # Spectral Angle Mapper (radians, lower = more similar)
            dot_product = np.clip(np.dot(unknown_norm, ref_norm), -1, 1)
            angle = np.arccos(dot_product)
            matches.append((name, angle))
        elif method == 'correlation':
            # Pearson correlation (higher = more similar)
            corr = np.corrcoef(unknown_valid, ref_valid)[0, 1]
            if np.isnan(corr):
                corr = 0
            matches.append((name, corr))
        else:  # euclidean
            # Euclidean distance (lower = more similar)
            dist = np.linalg.norm(unknown_valid - ref_valid)
            matches.append((name, dist))

    # Sort by similarity
    if method == 'correlation':
        matches.sort(key=lambda x: -x[1])  # Higher is better
    else:
        matches.sort(key=lambda x: x[1])   # Lower is better

    return matches[:top_n]


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if len(sys.argv) < 2:
        print("Usage: python usgs_parser.py <usgs_splib07_path> [output_dir]")
        print("\nParses USGS Spectral Library v7 and exports resampled spectra")
        print("for HyperspecI D1 (400-1000nm) and D2 (400-1700nm) grids.")
        print("\nExpected directory structure:")
        print("  usgs_splib07/")
        print("    ASCIIdata/")
        print("      ASCIIdata_splib07a/")
        print("        ChapterA_ArtificialMaterials/")
        print("        splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt")
        sys.exit(1)

    library_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '.'

    print(f"Parsing USGS library from: {library_path}")
    parser = USGSLibraryParser(library_path)

    # Parse all Chapter A materials (include_all=True since they're all manmade)
    parser.parse_all_indoor(include_all_chapter_a=True)

    summary = parser.get_summary()
    print(f"\n{'='*50}")
    print(f"Parsed {summary['total_spectra']} spectra:")
    for cat, count in sorted(summary['by_category'].items()):
        print(f"  {cat}: {count}")

    if summary['total_spectra'] > 0:
        print(f"\nSample materials:")
        for name in list(summary['materials'])[:10]:
            print(f"  - {name}")
        if len(summary['materials']) > 10:
            print(f"  ... and {len(summary['materials']) - 10} more")

    # Export both grids
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    parser.export_resampled(output_dir / 'usgs_indoor_d1.json', target='D1')
    parser.export_resampled(output_dir / 'usgs_indoor_d2.json', target='D2')

    print(f"\nExported to {output_dir}")
