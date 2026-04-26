"""
Volumetric LED display preview and export.

Extracts sub-cubes from hyperspectral data and renders them as 3D voxel
visualizations simulating a physical LED strand display.

Physical model:
  - A grid of vertical LED strands hanging from a ceiling frame
  - X axis = wavelength (depth — the axis you look into)
  - Y axis = spatial column
  - Z axis = spatial row (vertical, hanging down → mapped upward for display)
  - Each voxel's color = spectral color of its wavelength band
  - Each voxel's brightness = reflectance intensity

Orientation is configurable via axis_map.
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wavelength → visible color mapping
# ---------------------------------------------------------------------------

def wavelength_to_rgb(wl_nm):
    """Convert wavelength in nm to (R, G, B) tuple [0–255].

    Attempt to display the visible range (380-780nm) as actual spectral color,
    and map NIR/SWIR to a warm false-color palette.
    """
    wl = float(wl_nm)

    # Visible spectrum approximation (Dan Bruton's algorithm)
    if 380 <= wl < 440:
        r = -(wl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= wl < 490:
        r = 0.0
        g = (wl - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wl < 510:
        r = 0.0
        g = 1.0
        b = -(wl - 510) / (510 - 490)
    elif 510 <= wl < 580:
        r = (wl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wl < 645:
        r = 1.0
        g = -(wl - 645) / (645 - 580)
        b = 0.0
    elif 645 <= wl <= 780:
        r = 1.0
        g = 0.0
        b = 0.0
    elif 780 < wl <= 1100:
        # Near-IR: fade from red to deep red/maroon
        t = (wl - 780) / (1100 - 780)
        r = 1.0 - 0.4 * t
        g = 0.0
        b = 0.1 * t
    elif 1100 < wl <= 1800:
        # SWIR-1: deep magenta range
        t = (wl - 1100) / (1800 - 1100)
        r = 0.6 - 0.2 * t
        g = 0.0
        b = 0.1 + 0.3 * t
    elif 1800 < wl <= 2500:
        # SWIR-2: into deep violet/indigo
        t = (wl - 1800) / (2500 - 1800)
        r = 0.4 - 0.2 * t
        g = 0.0
        b = 0.4 + 0.2 * t
    else:
        r, g, b = 0.3, 0.3, 0.3

    # Intensity falloff at edges of visible range
    if 380 <= wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / (420 - 380)
    elif 700 < wl <= 780:
        factor = 0.3 + 0.7 * (780 - wl) / (780 - 700)
    elif 380 <= wl <= 700:
        factor = 1.0
    else:
        factor = 0.8  # NIR/SWIR — keep reasonably bright

    r = int(np.clip(r * factor * 255, 0, 255))
    g = int(np.clip(g * factor * 255, 0, 255))
    b = int(np.clip(b * factor * 255, 0, 255))
    return (r, g, b)


def build_wavelength_colormap(wavelengths):
    """Build an array of RGB colors for each wavelength band."""
    return np.array([wavelength_to_rgb(w) for w in wavelengths], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Sub-cube extraction
# ---------------------------------------------------------------------------

def extract_subcube(filepath, row_start, col_start, size=32,
                    rows=None, cols=None,
                    include_nan_bands=True):
    """Extract a spatial sub-cube from a NetCDF/HDF5 hyperspectral file.

    Parameters
    ----------
    size : int — square extraction size (used if rows/cols not given)
    rows : int — number of rows to extract (rod length)
    cols : int — number of columns to extract (rod grid width)

    Returns
    -------
    cube : ndarray, shape (n_bands, rows, cols)
    wavelengths : ndarray, shape (n_bands,)
    valid_mask : ndarray, shape (n_bands,) — True where data is valid
    """
    import h5py

    n_rows_extract = rows if rows is not None else size
    n_cols_extract = cols if cols is not None else size

    with h5py.File(filepath, 'r') as f:
        refl = f['reflectance/reflectance']
        wl = f['reflectance/wavelength'][:]
        n_bands, n_rows, n_cols = refl.shape

        # Clamp bounds
        r0 = max(0, min(row_start, n_rows - n_rows_extract))
        c0 = max(0, min(col_start, n_cols - n_cols_extract))
        r1 = r0 + n_rows_extract
        c1 = c0 + n_cols_extract

        cube = refl[:, r0:r1, c0:c1]

    # Identify valid vs atmospheric absorption bands
    nan_frac = np.isnan(cube).mean(axis=(1, 2))
    valid_mask = nan_frac < 0.5

    if not include_nan_bands:
        cube = cube[valid_mask]
        wl = wl[valid_mask]
        valid_mask = np.ones(len(wl), dtype=bool)

    # Fill remaining NaNs with 0
    cube = np.nan_to_num(cube, nan=0.0)

    logger.info(
        f"Extracted subcube: {cube.shape} from row={r0} col={c0}, "
        f"{valid_mask.sum()}/{len(valid_mask)} valid bands"
    )
    return cube, wl, valid_mask


# ---------------------------------------------------------------------------
# 3D voxel figure
# ---------------------------------------------------------------------------

# Axis mapping presets
#
# Physical model: vertical LED strands in a grid.
#   - Each strand hangs vertically (Z axis in plotly = bar length)
#   - Wavelength passes ACROSS bars (one horizontal grid axis)
#   - The other horizontal grid axis = second spatial dimension
#   - Coordinates use band index for wavelength so all axes are
#     comparable scale (not raw nm which dwarfs spatial dims)
#
ORIENTATIONS = {
    'strand_horizontal_wl': {
        # Bars hang in Z (row). Wavelength across X. Depth in Y (col).
        # Central view: looking along Y, you see row×wavelength face.
        'x': 'band',
        'y': 'col',
        'z': 'row',
        'labels': {'x': 'Band (wavelength →)', 'y': 'Column (depth)', 'z': 'Row (bar length)'},
    },
    'strand_vertical_wl': {
        # Bars hang in Z (row). Col across X. Wavelength into depth Y.
        # Central view: looking along Y, you see row×col spatial face.
        'x': 'col',
        'y': 'band',
        'z': 'row',
        'labels': {'x': 'Column', 'y': 'Band (wavelength →)', 'z': 'Row (bar length)'},
    },
    'profile_view': {
        # Bars hang in Z (col). Row across X. Wavelength into depth Y.
        'x': 'row',
        'y': 'band',
        'z': 'col',
        'labels': {'x': 'Row', 'y': 'Band (wavelength →)', 'z': 'Column (bar length)'},
    },
}


def _apply_colormap(val_norm, cmap='plasma'):
    """Map normalized value [0,1] to (R,G,B) [0,255] using matplotlib cmap."""
    import matplotlib.cm as cm
    cmap_fn = cm.get_cmap(cmap)
    rgba = cmap_fn(val_norm)
    return int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)


# Pre-build a LUT for speed (256 entries)
def _build_cmap_lut(cmap='plasma', n=256):
    import matplotlib.cm as cm
    cmap_fn = cm.get_cmap(cmap)
    lut = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        rgba = cmap_fn(i / (n - 1))
        lut[i] = [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)]
    return lut


def build_volumetric_figure(cube, wavelengths, valid_mask,
                            orientation='strand_horizontal_wl',
                            brightness_percentile=(2, 98),
                            nan_band_style='dim',
                            marker_size=2,
                            opacity_scale=1.0,
                            colormap='plasma'):
    """Build an interactive 3D scatter plot simulating the LED volume.

    Parameters
    ----------
    cube : ndarray, shape (n_bands, rows, cols)
    wavelengths : ndarray, shape (n_bands,)
    valid_mask : ndarray, shape (n_bands,) — True for real data bands
    orientation : str — key into ORIENTATIONS or custom dict
    brightness_percentile : tuple — percentile stretch for reflectance
    nan_band_style : str — 'dim' (show faintly), 'gap' (skip), 'marker' (distinct color)
    marker_size : float
    opacity_scale : float — global opacity multiplier (0–1)
    colormap : str — matplotlib colormap for reflectance values
    """
    n_bands, n_rows, n_cols = cube.shape

    if isinstance(orientation, str):
        orient = ORIENTATIONS[orientation]
    else:
        orient = orientation

    # Percentile stretch for reflectance → color mapping
    valid_data = cube[valid_mask]
    if valid_data.size > 0:
        flat = valid_data[valid_data > 0]
        vmin, vmax = np.percentile(flat, brightness_percentile)
    else:
        vmin, vmax = 0.0, 1.0
    vrange = max(vmax - vmin, 1e-6)

    # Build colormap LUT
    cmap_lut = _build_cmap_lut(colormap)

    # Vectorized: normalize entire cube at once
    cube_norm = np.clip((cube - vmin) / vrange, 0, 1)

    # Build coordinate and color arrays
    xs, ys, zs = [], [], []
    colors = []
    hover = []

    for b in range(n_bands):
        is_valid = valid_mask[b]

        if not is_valid and nan_band_style == 'gap':
            continue

        band_norm = cube_norm[b]

        for r in range(n_rows):
            for c in range(n_cols):
                val_norm = band_norm[r, c]

                # Map axes
                coords = {'band': b, 'row': r, 'col': c}
                xs.append(coords[orient['x']])
                ys.append(coords[orient['y']])
                zs.append(coords[orient['z']])

                if not is_valid:
                    if nan_band_style == 'dim':
                        colors.append(f'rgba(40, 40, 40, {0.06 * opacity_scale:.3f})')
                    else:  # 'marker'
                        colors.append(f'rgba(255, 0, 0, {0.12 * opacity_scale:.3f})')
                else:
                    # Color from reflectance through colormap
                    lut_idx = int(val_norm * 255)
                    cr, cg, cb = cmap_lut[lut_idx]
                    # Alpha: low reflectance = more transparent
                    alpha = (0.08 + 0.92 * val_norm) * opacity_scale
                    colors.append(f'rgba({cr}, {cg}, {cb}, {alpha:.3f})')

                hover.append(
                    f'λ={wavelengths[b]:.0f}nm  row={r} col={c}<br>'
                    f'refl={cube[b, r, c]:.4f}  {"valid" if is_valid else "atm. absorption"}'
                )

    fig = go.Figure(data=[go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=colors,
        ),
        hovertext=hover,
        hoverinfo='text',
    )])

    # Build wavelength tick labels for whichever axis carries band index
    wl_tick_step = max(1, n_bands // 10)
    wl_tickvals = list(range(0, n_bands, wl_tick_step))
    wl_ticktext = [f'{wavelengths[i]:.0f}nm' for i in wl_tickvals]

    def _axis_config(axis_key):
        """Build axis dict; add wavelength ticks if this axis is 'band'."""
        cfg = dict(color='white', gridcolor='#222')
        if orient[axis_key] == 'band':
            cfg['tickvals'] = wl_tickvals
            cfg['ticktext'] = wl_ticktext
        return cfg

    # Aspect ratio: make band axis length proportional to spatial dims
    # so bars look like bars, not stretched filaments
    aspect_map = {orient['x']: n_bands, orient['y']: n_bands, orient['z']: n_bands}
    aspect_map[orient['x']] = {'band': n_bands, 'row': n_rows, 'col': n_cols}[orient['x']]
    aspect_map[orient['y']] = {'band': n_bands, 'row': n_rows, 'col': n_cols}[orient['y']]
    aspect_map[orient['z']] = {'band': n_bands, 'row': n_rows, 'col': n_cols}[orient['z']]
    max_dim = max(aspect_map.values())
    aspect = dict(
        x=aspect_map[orient['x']] / max_dim,
        y=aspect_map[orient['y']] / max_dim,
        z=aspect_map[orient['z']] / max_dim,
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=orient['labels']['x'],
            yaxis_title=orient['labels']['y'],
            zaxis_title=orient['labels']['z'],
            bgcolor='black',
            xaxis=_axis_config('x'),
            yaxis=_axis_config('y'),
            zaxis=_axis_config('z'),
            aspectmode='manual',
            aspectratio=aspect,
        ),
        paper_bgcolor='black',
        font=dict(color='white'),
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text=f'Volumetric Preview — {n_cols}×{n_rows}×{n_bands} '
                 f'({wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm)',
            font=dict(size=14),
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------

def export_frame_sequence(cube, wavelengths, valid_mask, output_dir,
                          fmt='png', colormap=None):
    """Export each wavelength band as a 2D image frame.

    Each frame is a spatial slice colored by wavelength with reflectance
    as brightness. Suitable for driving LED matrix hardware or animation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    wl_colors = build_wavelength_colormap(wavelengths)
    vmin, vmax = np.percentile(cube[valid_mask][cube[valid_mask] > 0], [2, 98])
    vrange = max(vmax - vmin, 1e-6)

    manifest = []
    for b in range(cube.shape[0]):
        band = cube[b]
        brightness = np.clip((band - vmin) / vrange, 0, 1)

        # Tint by wavelength color
        wr, wg, wb = wl_colors[b] / 255.0
        frame = np.zeros((*band.shape, 3), dtype=np.float32)
        frame[:, :, 0] = brightness * wr
        frame[:, :, 1] = brightness * wg
        frame[:, :, 2] = brightness * wb

        if not valid_mask[b]:
            frame *= 0.15  # dim atmospheric bands

        fname = f'band_{b:03d}_{wavelengths[b]:.0f}nm.{fmt}'
        fpath = output_dir / fname

        plt.imsave(str(fpath), np.clip(frame, 0, 1))
        manifest.append({
            'band': int(b),
            'wavelength_nm': float(wavelengths[b]),
            'valid': bool(valid_mask[b]),
            'file': fname,
        })

    # Write manifest
    with open(output_dir / 'manifest.json', 'w') as f:
        json.dump({
            'n_bands': cube.shape[0],
            'spatial_size': [cube.shape[1], cube.shape[2]],
            'wavelength_range': [float(wavelengths[0]), float(wavelengths[-1])],
            'frames': manifest,
        }, f, indent=2)

    logger.info(f"Exported {len(manifest)} frames to {output_dir}")
    return manifest


# ---------------------------------------------------------------------------
# Quick preview helper
# ---------------------------------------------------------------------------

def preview_subcube(filepath, row_start=340, col_start=1100, size=32,
                    rows=None, cols=None,
                    orientation='strand_horizontal_wl',
                    include_nan_bands=True,
                    nan_band_style='dim',
                    marker_size=2,
                    opacity_scale=1.0,
                    colormap='plasma',
                    output_html=None):
    """One-shot: extract subcube → build figure → optionally save HTML."""
    cube, wl, valid = extract_subcube(
        filepath, row_start, col_start, size=size,
        rows=rows, cols=cols,
        include_nan_bands=include_nan_bands,
    )

    fig = build_volumetric_figure(
        cube, wl, valid,
        orientation=orientation,
        nan_band_style=nan_band_style,
        marker_size=marker_size,
        opacity_scale=opacity_scale,
        colormap=colormap,
    )

    if output_html:
        fig.write_html(str(output_html), include_plotlyjs=True)
        logger.info(f"Saved preview to {output_html}")

    return fig, cube, wl, valid
