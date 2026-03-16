"""
Plotly-based 3D figure builders for hyperspectral data cube visualization.

Provides interactive 3D surface slicing and volume rendering using Plotly's
go.Surface and go.Isosurface traces.
"""

import numpy as np
import plotly.graph_objects as go

from .figures import get_plot_layout, CMAP_TO_PLOTLY


def build_cube_slices_figure(
    engine,
    slice_x_index=None,
    slice_y_index=None,
    slice_z_index=None,
    slice_x_show=True,
    slice_y_show=True,
    slice_z_show=True,
    downsample=2,
    colormap="Viridis",
    opacity=0.9,
    dark=True,
):
    """
    Build a 3D figure with up to 3 orthogonal slice planes through the cube.

    Axes:
        X = columns (spatial), Y = rows (spatial), Z = bands (spectral/wavelength)

    Parameters
    ----------
    engine : WebEngine
        Data engine with loaded file.
    slice_x_index : int or None
        Column index for the YZ slice plane.
    slice_y_index : int or None
        Row index for the XZ slice plane.
    slice_z_index : int or None
        Band index for the XY slice plane.
    downsample : int
        Spatial/spectral decimation factor.
    colormap : str
        Plotly colorscale name.
    opacity : float
        Surface opacity (0-1).
    dark : bool
        Dark or light theme.

    Returns
    -------
    go.Figure
    """
    if engine.data_loader is None:
        return _empty_3d("No data loaded", dark)

    dl = engine.data_loader
    n_bands, n_rows, n_cols = dl.n_bands, dl.n_rows, dl.n_cols
    wavelengths = dl.wavelengths

    # Compute global color range from a quick sample
    sample_band = engine.get_single_band(float(wavelengths[n_bands // 2]))
    vmin, vmax = engine.robust_percentile(sample_band)

    traces = []

    # --- XY slice (at band z_index) ---
    if slice_z_show and slice_z_index is not None:
        z_idx = min(int(slice_z_index), n_bands - 1)
        data, meta = engine.get_slice_2d("band", z_idx, downsample)
        if data is not None:
            nr, nc = data.shape
            cols = np.arange(nc) * downsample
            rows = np.arange(nr) * downsample
            X, Y = np.meshgrid(cols, rows)
            Z = np.full_like(X, float(z_idx), dtype=np.float32)

            traces.append(go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=data,
                colorscale=colormap,
                cmin=vmin, cmax=vmax,
                opacity=opacity,
                showscale=True,
                colorbar=dict(title="Reflectance", thickness=20, len=0.6, x=1.02),
                name=f"Spatial slice at {wavelengths[z_idx]:.0f} nm (band {z_idx})",
                hovertemplate=(
                    "Col: %{x}<br>Row: %{y}<br>Band: %{z:.0f}"
                    "<br>Value: %{surfacecolor:.4f}<extra></extra>"
                ),
            ))

    # --- XZ slice (at row y_index) ---
    if slice_y_show and slice_y_index is not None:
        y_idx = min(int(slice_y_index), n_rows - 1)
        data, meta = engine.get_slice_2d("row", y_idx, downsample)
        if data is not None:
            nb, nc = data.shape
            cols = np.arange(nc) * downsample
            bands = np.arange(0, n_bands, downsample)[:nb]
            X, Z = np.meshgrid(cols, bands)
            Y = np.full_like(X, float(y_idx), dtype=np.float32)

            traces.append(go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=data,
                colorscale=colormap,
                cmin=vmin, cmax=vmax,
                opacity=opacity,
                showscale=False,
                name=f"Col–wavelength section at row {y_idx}",
                hovertemplate=(
                    "Col: %{x}<br>Row: %{y:.0f}<br>Band: %{z:.0f}"
                    "<br>Value: %{surfacecolor:.4f}<extra></extra>"
                ),
            ))

    # --- YZ slice (at col x_index) ---
    if slice_x_show and slice_x_index is not None:
        x_idx = min(int(slice_x_index), n_cols - 1)
        data, meta = engine.get_slice_2d("col", x_idx, downsample)
        if data is not None:
            nb, nr = data.shape
            rows = np.arange(nr) * downsample
            bands = np.arange(0, n_bands, downsample)[:nb]
            Y, Z = np.meshgrid(rows, bands)
            X = np.full_like(Y, float(x_idx), dtype=np.float32)

            traces.append(go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=data,
                colorscale=colormap,
                cmin=vmin, cmax=vmax,
                opacity=opacity,
                showscale=False,
                name=f"Row–wavelength section at col {x_idx}",
                hovertemplate=(
                    "Col: %{x:.0f}<br>Row: %{y}<br>Band: %{z:.0f}"
                    "<br>Value: %{surfacecolor:.4f}<extra></extra>"
                ),
            ))

    # --- Wireframe bounding box ---
    traces.append(_wireframe_box(n_cols - 1, n_rows - 1, n_bands - 1, dark))

    if not traces:
        return _empty_3d("Enable at least one slice plane", dark)

    fig = go.Figure(data=traces)

    # Build Z-axis tick labels with wavelengths
    n_ticks = min(8, n_bands)
    tick_indices = np.linspace(0, n_bands - 1, n_ticks, dtype=int)
    tick_labels = [f"{wavelengths[i]:.0f} nm" for i in tick_indices]

    fig.update_layout(
        scene=dict(
            xaxis_title="Column (px)",
            yaxis_title="Row (px)",
            zaxis_title="Wavelength",
            zaxis=dict(
                tickvals=tick_indices.tolist(),
                ticktext=tick_labels,
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=n_rows / n_cols, z=0.5),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        **get_plot_layout(dark),
    )

    return fig


def _wireframe_box(max_x, max_y, max_z, dark):
    """Create a wireframe bounding box as a Scatter3d trace."""
    color = "rgba(255,255,255,0.3)" if dark else "rgba(0,0,0,0.3)"
    # 12 edges of a box
    x = [0, max_x, max_x, 0, 0, None,
         0, max_x, max_x, 0, 0, None,
         0, 0, None, max_x, max_x, None,
         0, 0, None, max_x, max_x, None]
    y = [0, 0, max_y, max_y, 0, None,
         0, 0, max_y, max_y, 0, None,
         0, 0, None, 0, 0, None,
         max_y, max_y, None, max_y, max_y, None]
    z = [0, 0, 0, 0, 0, None,
         max_z, max_z, max_z, max_z, max_z, None,
         0, max_z, None, 0, max_z, None,
         0, max_z, None, 0, max_z, None]
    return go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        line=dict(color=color, width=2),
        showlegend=False,
        hoverinfo="skip",
    )


def _empty_3d(message, dark=True):
    """Return an empty 3D figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        scene=dict(
            xaxis_title="Column",
            yaxis_title="Row",
            zaxis_title="Band",
        ),
        annotations=[dict(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="white" if dark else "black"),
        )],
        **get_plot_layout(dark),
    )
    return fig
