"""
Plotly figure factory functions for the web viewer.
"""

import numpy as np
import plotly.graph_objects as go

from aviris_tools.viewer.constants import ATMOSPHERIC_BANDS, REFERENCE_SIGNATURES

# Matplotlib colormap names -> Plotly colorscale names
CMAP_TO_PLOTLY = {
    "RdYlGn": "RdYlGn",
    "RdBu": "RdBu",
    "BrBG": "BrBG",
    "viridis": "Viridis",
    "magma": "Magma",
    "plasma": "Plasma",
    "inferno": "Inferno",
    "cividis": "Cividis",
    "YlOrBr": "YlOrBr",
    "YlOrRd": "YlOrRd",
    "OrRd": "OrRd",
    "Oranges": "Oranges",
    "Reds": "Reds",
    "hot": "Hot",
    "Blues": "Blues",
    "PuBu": "PuBu",
    "Purples": "Purples",
    "RdPu": "RdPu",
    "Greens": "Greens",
    "YlGn": "YlGn",
    "BuGn": "BuGn",
    "copper": "YlOrBr",  # Closest to matplotlib copper (brown gradient)
}

_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#1e1e1e",
    plot_bgcolor="#1e1e1e",
    margin=dict(l=40, r=40, t=40, b=40),
    font=dict(size=11),
)

_LIGHT_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    margin=dict(l=40, r=40, t=40, b=40),
    font=dict(size=11),
)


def get_plot_layout(dark=True):
    """Return the Plotly layout theme dict."""
    return dict(_DARK_LAYOUT) if dark else dict(_LIGHT_LAYOUT)


def _to_rgba(hex_color, alpha):
    """Convert '#FF6B6B' to 'rgba(255,107,107,0.15)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _geo_tick_overrides(n_rows, n_cols, lat_axis, lon_axis, n_ticks=6):
    """Build tick overrides that map pixel indices to lat/lon labels."""
    row_idx = np.linspace(0, n_rows - 1, n_ticks, dtype=int)
    col_idx = np.linspace(0, n_cols - 1, n_ticks, dtype=int)

    x_ax = dict(
        title="Longitude",
        tickvals=col_idx.tolist(),
        ticktext=[f"{lon_axis[i]:.3f}" for i in col_idx],
    )
    y_ax = dict(
        title="Latitude",
        tickvals=row_idx.tolist(),
        ticktext=[f"{lat_axis[i]:.3f}" for i in row_idx],
        autorange="reversed",
        scaleanchor="x",
    )
    return x_ax, y_ax


def make_band_figure(band_data, wavelength, colorscale="Viridis",
                     lat_axis=None, lon_axis=None):
    """Single band as heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=band_data,
            colorscale=colorscale,
            colorbar=dict(title="Value", thickness=15),
        )
    )

    if lat_axis is not None and lon_axis is not None:
        n_rows, n_cols = band_data.shape
        x_ax, y_ax = _geo_tick_overrides(n_rows, n_cols, lat_axis, lon_axis)
    else:
        x_ax = {}
        y_ax = dict(scaleanchor="x", autorange="reversed")

    fig.update_layout(
        title=f"Band {wavelength:.0f} nm",
        xaxis=x_ax,
        yaxis=y_ax,
        **_DARK_LAYOUT,
    )
    return fig


def make_rgb_figure(rgb_array, title="RGB Composite",
                    lat_axis=None, lon_axis=None):
    """RGB composite (uint8 array) as go.Image."""
    fig = go.Figure(data=go.Image(z=rgb_array))

    if lat_axis is not None and lon_axis is not None:
        n_rows, n_cols = rgb_array.shape[:2]
        x_ax, y_ax = _geo_tick_overrides(n_rows, n_cols, lat_axis, lon_axis)
        # Image y-axis goes top-down natively, don't reverse
        y_ax.pop("autorange", None)
    else:
        x_ax = {}
        y_ax = dict(scaleanchor="x")

    fig.update_layout(
        title=title,
        xaxis=x_ax,
        yaxis=y_ax,
        **_DARK_LAYOUT,
    )
    return fig


def make_index_figure(index_data, index_name, cmap, clim,
                      lat_axis=None, lon_axis=None):
    """Index map as heatmap with colorscale."""
    plotly_cmap = CMAP_TO_PLOTLY.get(cmap, "Viridis")
    fig = go.Figure(
        data=go.Heatmap(
            z=np.nan_to_num(index_data, nan=0.0),
            colorscale=plotly_cmap,
            zmin=clim[0],
            zmax=clim[1],
            colorbar=dict(title=index_name, thickness=15),
        )
    )

    if lat_axis is not None and lon_axis is not None:
        n_rows, n_cols = index_data.shape
        x_ax, y_ax = _geo_tick_overrides(n_rows, n_cols, lat_axis, lon_axis)
    else:
        x_ax = {}
        y_ax = dict(scaleanchor="x", autorange="reversed")

    fig.update_layout(
        title=f"{index_name} Index",
        xaxis=x_ax,
        yaxis=y_ax,
        **_DARK_LAYOUT,
    )
    return fig


_ATM_LABELS = {
    "water_vapor_1": "H₂O",
    "water_vapor_2": "H₂O",
    "carbon_dioxide": "CO₂",
    "oxygen": "O₂",
}


def make_spectral_figure(spectra_list, data_type="reflectance", dark=True,
                         roi_spectra=None, match_materials=None):
    """
    Spectral plot from a list of spectrum dicts.

    Each dict in spectra_list has: wavelengths, values, label, color
    Each dict in roi_spectra has: wavelengths, mean_values, std_values, label, color
    match_materials: list of material names to annotate absorption features for
    """
    fig = go.Figure()

    # ROI envelopes first (so pixel lines draw on top)
    for roi in (roi_spectra or []):
        wl = roi["wavelengths"]
        mean = np.array(roi["mean_values"])
        std = np.array(roi["std_values"])
        color = roi["color"]
        label = roi["label"]
        fill_color = _to_rgba(color, 0.15)

        upper = (mean + std).tolist()
        lower = (mean - std).tolist()

        # Upper bound (invisible, anchor for fill)
        fig.add_trace(go.Scatter(
            x=wl, y=upper,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        # Lower bound with fill to upper
        fig.add_trace(go.Scatter(
            x=wl, y=lower,
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=fill_color,
            showlegend=False, hoverinfo="skip",
        ))
        # Mean line (dashed, in legend)
        fig.add_trace(go.Scatter(
            x=wl, y=mean.tolist(),
            mode="lines", name=label,
            line=dict(color=color, width=2, dash="dash"),
        ))

    # Pixel spectra
    for spec in spectra_list:
        fig.add_trace(
            go.Scatter(
                x=spec["wavelengths"],
                y=spec["values"],
                mode="lines",
                name=spec["label"],
                line=dict(color=spec["color"], width=1.5),
            )
        )

    # Atmospheric absorption band shading + labels
    for band_name, (start, end) in ATMOSPHERIC_BANDS.items():
        opacity = 0.15 if "water" in band_name else 0.1
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=f"rgba(255,80,80,{opacity})",
            line_width=0,
            layer="below",
        )
        label = _ATM_LABELS.get(band_name, band_name)
        fig.add_annotation(
            x=(start + end) / 2,
            y=1.0,
            yref="paper",
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=8, color="rgba(255,80,80,0.7)"),
            yanchor="top",
        )

    # Material absorption feature annotations
    for mat_name in (match_materials or []):
        # Look up in REFERENCE_SIGNATURES (case-insensitive partial match)
        ref_key = None
        mat_lower = mat_name.lower()
        for key in REFERENCE_SIGNATURES:
            if key.lower() in mat_lower or mat_lower in key.lower():
                ref_key = key
                break
        if ref_key is None:
            continue

        features = REFERENCE_SIGNATURES[ref_key]
        for wl_nm, depth, feature_name in features:
            fig.add_vline(
                x=wl_nm,
                line=dict(color="rgba(255,255,0,0.5)", width=1, dash="dot"),
                layer="below",
            )
            fig.add_annotation(
                x=wl_nm,
                y=0.0,
                yref="paper",
                text=f"{feature_name}",
                showarrow=True,
                arrowhead=0,
                arrowcolor="rgba(255,255,0,0.4)",
                ax=0,
                ay=20,
                font=dict(size=7, color="yellow" if dark else "#886600"),
                bgcolor="rgba(0,0,0,0.5)" if dark else "rgba(255,255,255,0.7)",
                borderpad=1,
            )

    legend_bg = "rgba(0,0,0,0.5)" if dark else "rgba(255,255,255,0.7)"
    fig.update_layout(
        xaxis_title="Wavelength (nm)",
        yaxis_title=data_type.title(),
        legend=dict(x=0.7, y=0.95, bgcolor=legend_bg),
        height=250,
        **get_plot_layout(dark),
    )
    return fig


def make_empty_figure(message="No data loaded", dark=True):
    """Empty placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#888" if dark else "#666"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=400,
        **get_plot_layout(dark),
    )
    return fig
