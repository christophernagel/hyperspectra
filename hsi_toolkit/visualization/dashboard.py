"""
Interactive Dashboard for HSI Toolkit

Provides a web-based interactive interface for exploring
hyperspectral imaging concepts.

Requires: dash, plotly (install with: pip install dash plotly)
"""

import numpy as np
from typing import Optional, Dict, Any, Callable
import warnings

# Try to import Dash components
try:
    from dash import Dash, dcc, html, callback, Input, Output, State
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    warnings.warn("Dash not installed. Install with: pip install dash plotly")


def launch_dashboard(port: int = 8050, debug: bool = False):
    """
    Launch the interactive HSI learning dashboard.

    Args:
        port: Port to run the server on
        debug: Enable debug mode
    """
    if not DASH_AVAILABLE:
        print("Dash is not installed. Install with: pip install dash plotly")
        print("Falling back to matplotlib interactive mode...")
        _launch_matplotlib_interactive()
        return

    app = create_dashboard_app()
    print(f"\nStarting HSI Learning Dashboard at http://localhost:{port}")
    print("Press Ctrl+C to stop\n")
    app.run(debug=debug, port=port)


def create_dashboard_app() -> 'Dash':
    """
    Create the Dash application.

    Returns:
        Dash app instance
    """
    if not DASH_AVAILABLE:
        raise ImportError("Dash not available. Install with: pip install dash plotly")

    app = Dash(__name__, title="HSI Learning Toolkit")

    # Import models here to avoid circular imports
    from ..atmosphere.atmosphere_simulator import AtmosphereSimulator, AtmosphericState
    from ..sensor.sensor_simulator import SensorSimulator
    from ..forward_model.forward_model import ForwardModel

    # Initialize models with default wavelengths
    wavelengths = np.linspace(380, 2500, 224)

    # Shared absorption band definitions for all plots
    absorption_bands = [
        # H2O bands
        {'x0': 710, 'x1': 740, 'label': 'H₂O', 'color': 'rgba(0,100,255,0.15)'},
        {'x0': 810, 'x1': 840, 'label': 'H₂O', 'color': 'rgba(0,100,255,0.15)'},
        {'x0': 900, 'x1': 980, 'label': 'H₂O 940', 'color': 'rgba(0,100,255,0.2)'},
        {'x0': 1100, 'x1': 1180, 'label': 'H₂O 1140', 'color': 'rgba(0,100,255,0.2)'},
        {'x0': 1330, 'x1': 1480, 'label': 'H₂O 1380', 'color': 'rgba(0,100,255,0.3)'},
        {'x0': 1800, 'x1': 1970, 'label': 'H₂O 1880', 'color': 'rgba(0,100,255,0.3)'},
        # O2 bands
        {'x0': 682, 'x1': 695, 'label': 'O₂-B', 'color': 'rgba(255,165,0,0.2)'},
        {'x0': 755, 'x1': 775, 'label': 'O₂-A', 'color': 'rgba(255,165,0,0.25)'},
        # CO2 bands
        {'x0': 1995, 'x1': 2080, 'label': 'CO₂', 'color': 'rgba(128,0,128,0.2)'},
        {'x0': 2560, 'x1': 2600, 'label': 'CO₂', 'color': 'rgba(128,0,128,0.15)'},
    ]

    def add_absorption_bands(fig, show_labels=True, row=None, col=None):
        """Add atmospheric absorption band shading to a plotly figure."""
        for band in absorption_bands:
            if row is not None and col is not None:
                # For subplots
                fig.add_vrect(
                    x0=band['x0'], x1=band['x1'],
                    fillcolor=band['color'],
                    layer='below', line_width=0,
                    row=row, col=col
                )
            else:
                fig.add_vrect(
                    x0=band['x0'], x1=band['x1'],
                    fillcolor=band['color'],
                    layer='below', line_width=0,
                )

    # App layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("Hyperspectral Imaging Learning Dashboard",
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            html.P("Interactive exploration of the complete imaging chain",
                  style={'textAlign': 'center', 'color': '#7f8c8d'})
        ], style={'padding': '20px', 'backgroundColor': '#ecf0f1'}),

        # Tab navigation
        dcc.Tabs([
            # Tab 1: Atmospheric Effects
            dcc.Tab(label='Atmospheric Effects', children=[
                html.Div([
                    html.H3("Explore Atmospheric Transmittance and Path Radiance"),

                    html.Div([
                        # Controls
                        html.Div([
                            html.H4("Atmospheric Parameters"),

                            # PWV with tooltip
                            html.Div([
                                html.Label("Precipitable Water Vapor (cm) ", style={'fontWeight': 'bold'}),
                                html.Span("ⓘ", id='pwv-info', style={'cursor': 'pointer', 'color': '#3498db'}),
                            ]),
                            html.Details([
                                html.Summary("What is PWV?", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                html.Div([
                                    html.P("Total column water vapor if condensed to liquid. Creates absorption bands at 720, 940, 1140, 1380, and 1880 nm.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P(["Typical values: 0.5 cm (dry), 2-3 cm (humid), 5+ cm (tropical). ",
                                            html.A("Learn more →", href="https://en.wikipedia.org/wiki/Precipitable_water", target="_blank")],
                                           style={'fontSize': '11px', 'margin': '5px 0'})
                                ], style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                            ], style={'marginBottom': '5px'}),
                            dcc.Slider(id='pwv-slider', min=0.1, max=5.0, step=0.1,
                                      value=1.5, marks={i: str(i) for i in range(6)}),

                            # AOD with tooltip
                            html.Div([
                                html.Label("Aerosol Optical Depth (550nm) ", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                                html.Span("ⓘ", style={'cursor': 'pointer', 'color': '#3498db'}),
                            ]),
                            html.Details([
                                html.Summary("What is AOD?", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                html.Div([
                                    html.P("Measure of aerosol extinction in the atmospheric column. Higher values = hazier conditions.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P(["Clean: <0.1 | Typical: 0.1-0.3 | Hazy: 0.3-0.5 | Polluted: >0.5. ",
                                            html.A("AERONET data →", href="https://aeronet.gsfc.nasa.gov/", target="_blank")],
                                           style={'fontSize': '11px', 'margin': '5px 0'})
                                ], style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                            ], style={'marginBottom': '5px'}),
                            dcc.Slider(id='aod-slider', min=0.01, max=0.5, step=0.01,
                                      value=0.1, marks={0: '0', 0.25: '0.25', 0.5: '0.5'}),

                            # SZA with tooltip
                            html.Div([
                                html.Label("Solar Zenith Angle (°) ", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                                html.Span("ⓘ", style={'cursor': 'pointer', 'color': '#3498db'}),
                            ]),
                            html.Details([
                                html.Summary("What is SZA?", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                html.Div([
                                    html.P("Angle between sun and vertical (zenith). 0°=overhead, 90°=horizon. Affects path length through atmosphere.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P(["Airmass = 1/cos(SZA). At 60°, light travels 2× the vertical path. ",
                                            html.A("Solar geometry →", href="https://en.wikipedia.org/wiki/Solar_zenith_angle", target="_blank")],
                                           style={'fontSize': '11px', 'margin': '5px 0'})
                                ], style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                            ], style={'marginBottom': '5px'}),
                            dcc.Slider(id='sza-slider', min=0, max=80, step=5,
                                      value=30, marks={i: str(i) for i in range(0, 81, 20)}),

                            # Aerosol Type with tooltip
                            html.Div([
                                html.Label("Aerosol Type ", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                                html.Span("ⓘ", style={'cursor': 'pointer', 'color': '#3498db'}),
                            ]),
                            html.Details([
                                html.Summary("About aerosol types", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                html.Div([
                                    html.P("Different aerosol compositions have different scattering properties (Ångström exponent α):",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.Ul([
                                        html.Li("Maritime: Sea salt, large particles (α≈0.8)", style={'fontSize': '10px'}),
                                        html.Li("Continental: Mixed dust/sulfates (α≈1.3)", style={'fontSize': '10px'}),
                                        html.Li("Urban: Combustion soot, small particles (α≈1.8)", style={'fontSize': '10px'}),
                                        html.Li("Desert: Mineral dust, very large (α≈0.3)", style={'fontSize': '10px'}),
                                    ], style={'margin': '5px 0', 'paddingLeft': '20px'}),
                                    html.P([html.A("Shettle & Fenn models →", href="https://www.spiedigitallibrary.org/conference-proceedings-of-spie/0277/0000/Models-For-The-Aerosols-Of-The-Lower-Atmosphere-And-The/10.1117/12.931927.short", target="_blank")],
                                           style={'fontSize': '11px', 'margin': '5px 0'})
                                ], style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                            ], style={'marginBottom': '5px'}),
                            dcc.Dropdown(id='aerosol-dropdown',
                                        options=[
                                            {'label': 'Continental', 'value': 'continental'},
                                            {'label': 'Maritime', 'value': 'maritime'},
                                            {'label': 'Urban', 'value': 'urban'},
                                            {'label': 'Desert', 'value': 'desert'}
                                        ],
                                        value='continental'),

                        ], style={'width': '30%', 'padding': '20px', 'display': 'inline-block',
                                 'verticalAlign': 'top'}),

                        # Plots with chart tooltips
                        html.Div([
                            # Transmittance chart header
                            html.Details([
                                html.Summary("About this chart: Atmospheric Transmittance",
                                           style={'color': '#2c3e50', 'fontWeight': 'bold', 'fontSize': '13px', 'cursor': 'pointer'}),
                                html.Div([
                                    html.P("Shows the fraction of light that passes through the atmosphere at each wavelength (0 = fully absorbed, 1 = fully transmitted).",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P("How to read: Dips indicate absorption bands where gases absorb light. Shaded regions mark known absorption features (H₂O, O₂, CO₂). The red 'Total' line is the combined effect.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P("Key insight: Wavelengths in deep absorption bands (e.g., 1380nm, 1880nm) are unusable for surface analysis.",
                                           style={'fontSize': '11px', 'margin': '5px 0', 'fontStyle': 'italic'})
                                ], style={'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px', 'marginBottom': '10px'})
                            ]),
                            dcc.Graph(id='transmittance-plot'),

                            # Path Radiance chart header
                            html.Details([
                                html.Summary("About this chart: Path Radiance",
                                           style={'color': '#2c3e50', 'fontWeight': 'bold', 'fontSize': '13px', 'cursor': 'pointer'}),
                                html.Div([
                                    html.P("Shows light scattered by the atmosphere toward the sensor without hitting the ground - the 'haze' that reduces image contrast.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P("How to read: Higher values mean more atmospheric interference. Blue/green wavelengths have highest path radiance due to λ⁻⁴ Rayleigh scattering.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P("Key insight: This unwanted signal must be subtracted during atmospheric correction to recover true surface reflectance.",
                                           style={'fontSize': '11px', 'margin': '5px 0', 'fontStyle': 'italic'})
                                ], style={'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px', 'marginBottom': '10px'})
                            ]),
                            dcc.Graph(id='path-radiance-plot')
                        ], style={'width': '65%', 'display': 'inline-block'})

                    ], style={'display': 'flex'})
                ], style={'padding': '20px'})
            ]),

            # Tab 2: Sensor Response
            dcc.Tab(label='Sensor Physics', children=[
                html.Div([
                    html.H3("Sensor Detection Chain"),

                    html.Div([
                        # Controls
                        html.Div([
                            html.H4("Sensor Parameters"),

                            # Read Noise with tooltip
                            html.Div([
                                html.Label("Read Noise (electrons) ", style={'fontWeight': 'bold'}),
                                html.Span("ⓘ", style={'cursor': 'pointer', 'color': '#3498db'}),
                            ]),
                            html.Details([
                                html.Summary("What is read noise?", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                html.Div([
                                    html.P("Electronics noise introduced during readout of the detector. Constant per pixel read, independent of signal level.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P(["Typical values: 5-20 e⁻ (scientific CCDs), 20-50 e⁻ (standard). Dominates SNR at low light. ",
                                            html.A("CCD noise →", href="https://en.wikipedia.org/wiki/Image_noise#Read_noise", target="_blank")],
                                           style={'fontSize': '11px', 'margin': '5px 0'})
                                ], style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                            ], style={'marginBottom': '5px'}),
                            dcc.Slider(id='read-noise-slider', min=5, max=100, step=5,
                                      value=20, marks={5: '5', 50: '50', 100: '100'}),

                            # Dark Current with tooltip
                            html.Div([
                                html.Label("Dark Current (e⁻/s) ", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                                html.Span("ⓘ", style={'cursor': 'pointer', 'color': '#3498db'}),
                            ]),
                            html.Details([
                                html.Summary("What is dark current?", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                html.Div([
                                    html.P("Thermal electrons generated in the detector even without light. Accumulates linearly with integration time.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P(["Doubles every ~6-7°C. Cooled detectors (-40°C) can have <1 e⁻/s. Room temp: 100-1000 e⁻/s. ",
                                            html.A("Dark current physics →", href="https://en.wikipedia.org/wiki/Dark_current_(physics)", target="_blank")],
                                           style={'fontSize': '11px', 'margin': '5px 0'})
                                ], style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                            ], style={'marginBottom': '5px'}),
                            dcc.Slider(id='dark-slider', min=10, max=500, step=10,
                                      value=100, marks={10: '10', 250: '250', 500: '500'}),

                            # Integration Time with tooltip
                            html.Div([
                                html.Label("Integration Time (ms) ", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                                html.Span("ⓘ", style={'cursor': 'pointer', 'color': '#3498db'}),
                            ]),
                            html.Details([
                                html.Summary("What is integration time?", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                html.Div([
                                    html.P("Exposure duration for each frame. Longer = more signal but also more dark current and potential saturation.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P(["AVIRIS-3 uses ~8-16 ms. Constrained by aircraft speed and desired spatial resolution. ",
                                            html.A("Pushbroom timing →", href="https://en.wikipedia.org/wiki/Push_broom_scanner", target="_blank")],
                                           style={'fontSize': '11px', 'margin': '5px 0'})
                                ], style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                            ], style={'marginBottom': '5px'}),
                            dcc.Slider(id='integration-slider', min=1, max=50, step=1,
                                      value=10, marks={1: '1', 25: '25', 50: '50'}),

                            # Scene Radiance Level with tooltip
                            html.Div([
                                html.Label("Scene Radiance Level ", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                                html.Span("ⓘ", style={'cursor': 'pointer', 'color': '#3498db'}),
                            ]),
                            html.Details([
                                html.Summary("About radiance levels", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                html.Div([
                                    html.P("Simulates different illumination conditions affecting the signal reaching the detector.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.Ul([
                                        html.Li("Low: Shadows, north-facing slopes, water", style={'fontSize': '10px'}),
                                        html.Li("Medium: Typical vegetated/soil surfaces", style={'fontSize': '10px'}),
                                        html.Li("High: Bright targets, sun glint, white surfaces", style={'fontSize': '10px'}),
                                    ], style={'margin': '5px 0', 'paddingLeft': '20px'}),
                                    html.P("Higher radiance → more signal electrons → better SNR (until saturation).",
                                           style={'fontSize': '11px', 'margin': '5px 0'})
                                ], style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                            ], style={'marginBottom': '5px'}),
                            dcc.Dropdown(id='radiance-level-dropdown',
                                        options=[
                                            {'label': 'Low (shadow)', 'value': 0.3},
                                            {'label': 'Medium', 'value': 1.0},
                                            {'label': 'High (bright)', 'value': 3.0}
                                        ],
                                        value=1.0),

                        ], style={'width': '30%', 'padding': '20px', 'display': 'inline-block',
                                 'verticalAlign': 'top'}),

                        # Plots with chart tooltips
                        html.Div([
                            # SNR chart header
                            html.Details([
                                html.Summary("About this chart: Signal-to-Noise Ratio",
                                           style={'color': '#2c3e50', 'fontWeight': 'bold', 'fontSize': '13px', 'cursor': 'pointer'}),
                                html.Div([
                                    html.P("Shows the ratio of signal to noise at each wavelength. Higher SNR = cleaner data, better ability to detect subtle spectral features.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P("How to read: Log scale on Y-axis. Blue dashed line (SNR=100) is a typical quality target. Red dashed line (SNR=10) is the minimum for usable data.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P("Key insight: SNR varies with wavelength because signal strength and detector sensitivity vary across the spectrum.",
                                           style={'fontSize': '11px', 'margin': '5px 0', 'fontStyle': 'italic'})
                                ], style={'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px', 'marginBottom': '10px'})
                            ]),
                            dcc.Graph(id='snr-plot'),

                            # Noise breakdown chart header
                            html.Details([
                                html.Summary("About this chart: Noise Components",
                                           style={'color': '#2c3e50', 'fontWeight': 'bold', 'fontSize': '13px', 'cursor': 'pointer'}),
                                html.Div([
                                    html.P("Breaks down the three main noise sources in a detector: shot noise (signal-dependent), dark noise (thermal), and read noise (electronics).",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P("How to read: Taller bars = more noise. Total noise combines as √(shot² + dark² + read²). The dominant source limits your SNR.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P("Key insight: At high signal, shot noise dominates. At low signal or long exposures, dark and read noise become significant.",
                                           style={'fontSize': '11px', 'margin': '5px 0', 'fontStyle': 'italic'})
                                ], style={'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px', 'marginBottom': '10px'})
                            ]),
                            dcc.Graph(id='noise-breakdown-plot')
                        ], style={'width': '65%', 'display': 'inline-block'})

                    ], style={'display': 'flex'})
                ], style={'padding': '20px'})
            ]),

            # Tab 3: Forward Model
            dcc.Tab(label='Forward Model', children=[
                html.Div([
                    html.H3("Complete Imaging Chain: Surface → Sensor"),

                    html.Div([
                        # Controls
                        html.Div([
                            # Surface Type with tooltip
                            html.Div([
                                html.Label("Surface Type ", style={'fontWeight': 'bold'}),
                                html.Span("ⓘ", style={'cursor': 'pointer', 'color': '#3498db'}),
                            ]),
                            html.Details([
                                html.Summary("About surface types", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                html.Div([
                                    html.P("Select a material with characteristic spectral reflectance:",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.Ul([
                                        html.Li("Vegetation: Red edge at 700nm, chlorophyll absorption", style={'fontSize': '10px'}),
                                        html.Li("Soil: Gradual increase toward SWIR, iron features", style={'fontSize': '10px'}),
                                        html.Li("Water: Very low reflectance, drops off in NIR", style={'fontSize': '10px'}),
                                        html.Li("Concrete: Flat, moderate reflectance ~30-40%", style={'fontSize': '10px'}),
                                        html.Li("Panels: Calibration targets with known reflectance", style={'fontSize': '10px'}),
                                    ], style={'margin': '5px 0', 'paddingLeft': '20px'}),
                                ], style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                            ], style={'marginBottom': '5px'}),
                            dcc.Dropdown(id='surface-dropdown',
                                        options=[
                                            {'label': 'Vegetation', 'value': 'vegetation'},
                                            {'label': 'Soil', 'value': 'soil'},
                                            {'label': 'Water', 'value': 'water'},
                                            {'label': 'Concrete', 'value': 'concrete'},
                                            {'label': 'White Panel (95%)', 'value': 'white'},
                                            {'label': 'Gray Panel (25%)', 'value': 'gray'}
                                        ],
                                        value='vegetation'),

                            # Atmosphere section
                            html.H4("Atmosphere", style={'marginTop': '20px'}),

                            # PWV with tooltip
                            html.Div([
                                html.Label("PWV (cm) ", style={'fontWeight': 'bold'}),
                                html.Span("ⓘ", style={'cursor': 'pointer', 'color': '#3498db'}),
                            ]),
                            html.Details([
                                html.Summary("Precipitable Water Vapor", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                html.Div([
                                    html.P("Column water vapor in the atmosphere. Higher values create deeper absorption bands at 940, 1140, 1380, and 1880 nm.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                ], style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                            ], style={'marginBottom': '5px'}),
                            dcc.Slider(id='fm-pwv-slider', min=0.5, max=4.0, step=0.5,
                                      value=1.5, marks={0.5: '0.5', 2: '2', 4: '4'}),

                            # AOD with tooltip
                            html.Div([
                                html.Label("AOD (550nm) ", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                                html.Span("ⓘ", style={'cursor': 'pointer', 'color': '#3498db'}),
                            ]),
                            html.Details([
                                html.Summary("Aerosol Optical Depth", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                html.Div([
                                    html.P("Aerosol loading in the atmosphere. Higher values increase path radiance (haze) and reduce surface signal, especially at shorter wavelengths.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                ], style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                            ], style={'marginBottom': '5px'}),
                            dcc.Slider(id='fm-aod-slider', min=0.05, max=0.3, step=0.05,
                                      value=0.1, marks={0.05: '0.05', 0.15: '0.15', 0.3: '0.3'}),

                            # Display options with tooltip
                            html.Div([
                                html.Label("Display Options ", style={'fontWeight': 'bold', 'marginTop': '15px'}),
                                html.Span("ⓘ", style={'cursor': 'pointer', 'color': '#3498db'}),
                            ]),
                            html.Details([
                                html.Summary("About display options", style={'color': '#7f8c8d', 'fontSize': '12px'}),
                                html.Div([
                                    html.Ul([
                                        html.Li("Path radiance: Atmospheric contribution to sensor signal", style={'fontSize': '10px'}),
                                        html.Li("Apparent reflectance: What you'd get without atmospheric correction", style={'fontSize': '10px'}),
                                        html.Li("Sensor noise: Adds realistic detector noise to the signal", style={'fontSize': '10px'}),
                                    ], style={'margin': '5px 0', 'paddingLeft': '20px'}),
                                ], style={'padding': '5px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px'})
                            ], style={'marginBottom': '5px'}),
                            dcc.Checklist(id='display-options',
                                         options=[
                                             {'label': 'Show path radiance', 'value': 'path'},
                                             {'label': 'Show apparent reflectance', 'value': 'apparent'},
                                             {'label': 'Add sensor noise', 'value': 'noise'}
                                         ],
                                         value=['path'])

                        ], style={'width': '30%', 'padding': '20px', 'display': 'inline-block',
                                 'verticalAlign': 'top'}),

                        # Plots with chart tooltips
                        html.Div([
                            # Chain plot header
                            html.Details([
                                html.Summary("About this chart: Imaging Chain",
                                           style={'color': '#2c3e50', 'fontWeight': 'bold', 'fontSize': '13px', 'cursor': 'pointer'}),
                                html.Div([
                                    html.P("Top panel: True surface reflectance (green) vs apparent reflectance (red dashed) - the difference is atmospheric distortion.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P("Bottom panel: At-sensor radiance (blue) with path radiance contribution (red fill). Path radiance is the 'floor' that must be removed.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P("Key insight: Compare apparent vs true reflectance to see why atmospheric correction matters.",
                                           style={'fontSize': '11px', 'margin': '5px 0', 'fontStyle': 'italic'})
                                ], style={'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px', 'marginBottom': '10px'})
                            ]),
                            dcc.Graph(id='chain-plot'),

                            # Components plot header
                            html.Details([
                                html.Summary("About this chart: Signal Composition",
                                           style={'color': '#2c3e50', 'fontWeight': 'bold', 'fontSize': '13px', 'cursor': 'pointer'}),
                                html.Div([
                                    html.P("Shows the percentage of sensor signal coming from path radiance (red) vs actual surface reflection (green) at each wavelength.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P("How to read: At blue wavelengths, path radiance can dominate. At SWIR wavelengths, nearly all signal is from the surface.",
                                           style={'fontSize': '11px', 'margin': '5px 0'}),
                                    html.P("Key insight: Atmospheric correction is most critical where path radiance percentage is high.",
                                           style={'fontSize': '11px', 'margin': '5px 0', 'fontStyle': 'italic'})
                                ], style={'padding': '8px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px', 'marginBottom': '10px'})
                            ]),
                            dcc.Graph(id='components-plot')
                        ], style={'width': '65%', 'display': 'inline-block'})

                    ], style={'display': 'flex'})
                ], style={'padding': '20px'})
            ]),

            # Tab 4: Educational Content
            dcc.Tab(label='Learn', children=[
                html.Div([
                    html.H3("Understanding Hyperspectral Imaging"),

                    dcc.Tabs([
                        dcc.Tab(label='The Imaging Chain', children=[
                            html.Div([
                                # Left: Text content
                                html.Div([
                                    dcc.Markdown('''
## The Hyperspectral Imaging Chain

### From Surface to Sensor

1. **Solar Illumination**
   - Sun provides broadband illumination
   - ~6000K blackbody with Fraunhofer absorption lines
   - Varies with solar angle (cos(θ) effect)

2. **Downward Atmospheric Path**
   - Gas absorption (H₂O, O₃, O₂, CO₂)
   - Rayleigh scattering (λ⁻⁴ dependence)
   - Aerosol scattering (larger particles)
   - Some light scattered away, some reaches surface

3. **Surface Interaction**
   - Reflection depends on material properties
   - Lambertian assumption: L = ρ × E / π
   - Real surfaces may be non-Lambertian

4. **Upward Atmospheric Path**
   - More absorption and scattering
   - Path radiance added (atmosphere glows)
   - Adjacent pixels can affect each other

5. **Sensor Detection**
   - Optics collect and focus light
   - Grating disperses by wavelength
   - Detector converts photons to electrons
   - Noise added, signal digitized
''')
                                ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '20px'}),

                                # Right: Diagram
                                html.Div([
                                    html.H4("Signal Flow Diagram", style={'textAlign': 'center', 'color': '#2c3e50'}),
                                    html.Div([
                                        html.Div("☀️ SUN", style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f39c12', 'borderRadius': '8px', 'marginBottom': '5px', 'fontWeight': 'bold'}),
                                        html.Div("↓ Solar Irradiance (E₀)", style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px'}),
                                        html.Div("ATMOSPHERE", style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#85c1e9', 'borderRadius': '8px', 'margin': '5px 0', 'fontWeight': 'bold'}),
                                        html.Div("↓ Transmittance (T↓) + Scattering", style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px'}),
                                        html.Div("SURFACE (ρ)", style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#27ae60', 'borderRadius': '8px', 'margin': '5px 0', 'fontWeight': 'bold', 'color': 'white'}),
                                        html.Div("↓ Reflected: L = ρ × E / π", style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px'}),
                                        html.Div("ATMOSPHERE", style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#85c1e9', 'borderRadius': '8px', 'margin': '5px 0', 'fontWeight': 'bold'}),
                                        html.Div("↓ T↑ + Path Radiance (Lₚ)", style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px'}),
                                        html.Div("✈️ SENSOR", style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': '#3498db', 'borderRadius': '8px', 'margin': '5px 0', 'fontWeight': 'bold', 'color': 'white'}),
                                        html.Div("L_sensor = Lₚ + T↑ × ρ × E_down / π", style={'textAlign': 'center', 'color': '#2c3e50', 'fontSize': '12px', 'marginTop': '10px', 'fontFamily': 'monospace', 'backgroundColor': '#ecf0f1', 'padding': '8px', 'borderRadius': '4px'}),
                                    ], style={'padding': '15px', 'backgroundColor': '#fafafa', 'borderRadius': '8px', 'border': '1px solid #ddd'})
                                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
                            ], style={'display': 'flex'})
                        ]),

                        dcc.Tab(label='Atmospheric Effects', children=[
                            html.Div([
                                # Left: Text content
                                html.Div([
                                    dcc.Markdown('''
## Atmospheric Effects on Hyperspectral Data

### Absorption Features

Major atmospheric gases create distinct absorption bands:

| Gas | Wavelengths (nm) | Effect |
|-----|------------------|--------|
| O₂  | 688, 762 | Deep narrow bands |
| H₂O | 720, 820, 940, 1140, 1380, 1880 | Wide bands, PWV dependent |
| CO₂ | 2010, 2060 | Moderate bands |
| O₃  | 300-340, 550-650 | UV absorption, Chappuis band |

### Scattering

**Rayleigh Scattering** (molecules):
- τ ∝ λ⁻⁴
- Causes blue sky, red sunsets
- Makes short wavelengths appear hazy

**Aerosol Scattering** (particles):
- τ ∝ λ⁻α (α typically 1-2)
- Depends on particle size distribution
- Maritime, continental, urban types differ

### Path Radiance

Light scattered by atmosphere toward sensor without
hitting the ground. Creates a "haze" that reduces contrast.

- Strongest at short wavelengths (blue/green)
- Minimal at SWIR wavelengths
- Must be removed for accurate surface reflectance
''')
                                ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '20px'}),

                                # Right: Absorption bands diagram
                                html.Div([
                                    html.H4("Atmospheric Windows & Absorption", style={'textAlign': 'center', 'color': '#2c3e50'}),
                                    html.Div([
                                        # Wavelength scale
                                        html.Div([
                                            html.Span("400", style={'position': 'absolute', 'left': '0%'}),
                                            html.Span("700", style={'position': 'absolute', 'left': '15%'}),
                                            html.Span("1000", style={'position': 'absolute', 'left': '30%'}),
                                            html.Span("1400", style={'position': 'absolute', 'left': '50%'}),
                                            html.Span("1900", style={'position': 'absolute', 'left': '75%'}),
                                            html.Span("2500 nm", style={'position': 'absolute', 'right': '0%'}),
                                        ], style={'position': 'relative', 'height': '20px', 'fontSize': '10px', 'color': '#7f8c8d'}),

                                        # Transmittance bar with absorption bands
                                        html.Div([
                                            html.Div(style={'position': 'absolute', 'left': '0%', 'width': '100%', 'height': '100%', 'backgroundColor': '#a8e6cf'}),
                                            # O2 bands
                                            html.Div("O₂", style={'position': 'absolute', 'left': '14%', 'width': '2%', 'height': '100%', 'backgroundColor': '#ff9f43', 'fontSize': '9px', 'textAlign': 'center', 'color': 'white'}),
                                            html.Div("O₂", style={'position': 'absolute', 'left': '18%', 'width': '2%', 'height': '100%', 'backgroundColor': '#ff9f43', 'fontSize': '9px', 'textAlign': 'center', 'color': 'white'}),
                                            # H2O bands
                                            html.Div("H₂O", style={'position': 'absolute', 'left': '26%', 'width': '4%', 'height': '100%', 'backgroundColor': '#74b9ff', 'fontSize': '9px', 'textAlign': 'center'}),
                                            html.Div("H₂O", style={'position': 'absolute', 'left': '36%', 'width': '4%', 'height': '100%', 'backgroundColor': '#74b9ff', 'fontSize': '9px', 'textAlign': 'center'}),
                                            html.Div("H₂O", style={'position': 'absolute', 'left': '47%', 'width': '8%', 'height': '100%', 'backgroundColor': '#0984e3', 'fontSize': '9px', 'textAlign': 'center', 'color': 'white'}),
                                            html.Div("H₂O", style={'position': 'absolute', 'left': '70%', 'width': '10%', 'height': '100%', 'backgroundColor': '#0984e3', 'fontSize': '9px', 'textAlign': 'center', 'color': 'white'}),
                                            # CO2 bands
                                            html.Div("CO₂", style={'position': 'absolute', 'left': '82%', 'width': '5%', 'height': '100%', 'backgroundColor': '#a29bfe', 'fontSize': '9px', 'textAlign': 'center'}),
                                        ], style={'position': 'relative', 'height': '40px', 'borderRadius': '4px', 'overflow': 'hidden', 'border': '1px solid #ddd'}),

                                        # Legend
                                        html.Div([
                                            html.Span("■", style={'color': '#a8e6cf'}), " High transmittance  ",
                                            html.Span("■", style={'color': '#74b9ff'}), " H₂O absorption  ",
                                            html.Span("■", style={'color': '#ff9f43'}), " O₂ absorption  ",
                                            html.Span("■", style={'color': '#a29bfe'}), " CO₂ absorption",
                                        ], style={'fontSize': '11px', 'marginTop': '10px', 'textAlign': 'center'}),

                                        # Scattering diagram
                                        html.H5("Scattering Wavelength Dependence", style={'marginTop': '20px', 'textAlign': 'center'}),
                                        html.Div([
                                            html.Div([
                                                html.Div("Blue (450nm)", style={'color': '#3498db', 'fontWeight': 'bold'}),
                                                html.Div("████████████████", style={'color': '#3498db', 'fontFamily': 'monospace'}),
                                                html.Div("Strong scattering", style={'fontSize': '10px', 'color': '#7f8c8d'}),
                                            ], style={'marginBottom': '5px'}),
                                            html.Div([
                                                html.Div("Green (550nm)", style={'color': '#27ae60', 'fontWeight': 'bold'}),
                                                html.Div("████████████", style={'color': '#27ae60', 'fontFamily': 'monospace'}),
                                                html.Div("Moderate", style={'fontSize': '10px', 'color': '#7f8c8d'}),
                                            ], style={'marginBottom': '5px'}),
                                            html.Div([
                                                html.Div("Red (700nm)", style={'color': '#e74c3c', 'fontWeight': 'bold'}),
                                                html.Div("████████", style={'color': '#e74c3c', 'fontFamily': 'monospace'}),
                                                html.Div("Less scattering", style={'fontSize': '10px', 'color': '#7f8c8d'}),
                                            ], style={'marginBottom': '5px'}),
                                            html.Div([
                                                html.Div("SWIR (2000nm)", style={'color': '#95a5a6', 'fontWeight': 'bold'}),
                                                html.Div("███", style={'color': '#95a5a6', 'fontFamily': 'monospace'}),
                                                html.Div("Minimal scattering", style={'fontSize': '10px', 'color': '#7f8c8d'}),
                                            ]),
                                        ], style={'padding': '10px', 'backgroundColor': '#fafafa', 'borderRadius': '4px'})
                                    ], style={'padding': '15px', 'backgroundColor': '#fafafa', 'borderRadius': '8px', 'border': '1px solid #ddd'})
                                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
                            ], style={'display': 'flex'})
                        ]),

                        dcc.Tab(label='Sensor Physics', children=[
                            html.Div([
                                # Left: Text content
                                html.Div([
                                    dcc.Markdown('''
## Imaging Spectrometer Physics

### Pushbroom Scanning

The sensor has a linear detector array:
- One dimension = spatial (cross-track)
- Perpendicular motion builds along-track dimension
- Each pixel captures full spectrum simultaneously

### Grating Dispersion

Light separated by wavelength using diffraction:

```
Grating equation: d × sin(θ) = m × λ

where:
  d = groove spacing
  θ = diffraction angle
  m = order (usually 1)
  λ = wavelength
```

### Detector Noise Sources

1. **Shot noise**: √N (Poisson statistics)
   - Fundamental limit from quantum nature of light
   - Dominates at high signal levels

2. **Read noise**: Constant per read
   - Electronics noise in readout circuit
   - Dominates at low signal levels

3. **Dark current**: Thermal electrons
   - Accumulates with integration time
   - Reduced by cooling detector

### Signal-to-Noise Ratio

```
SNR = S / √(S + D + R²)

where:
  S = signal electrons
  D = dark current electrons
  R = read noise electrons
```
''')
                                ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '20px'}),

                                # Right: Sensor diagrams
                                html.Div([
                                    html.H4("Pushbroom Sensor Concept", style={'textAlign': 'center', 'color': '#2c3e50'}),
                                    html.Div([
                                        # Pushbroom diagram
                                        html.Div([
                                            html.Pre('''
     Flight Direction →
     ==================

     ┌─────────────────┐
     │    AIRCRAFT     │
     │   ┌─────────┐   │
     │   │ SENSOR  │   │
     │   └────┬────┘   │
     └────────┼────────┘
              │ FOV
         ╱    │    ╲
        ╱     │     ╲
       ╱      │      ╲
      ╱       │       ╲
     ╱        │        ╲
    ══════════════════════  ← Scan Line
    ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■   (cross-track pixels)

    Ground moves under sensor
    to build 2D image
''', style={'fontFamily': 'monospace', 'fontSize': '11px', 'backgroundColor': '#f8f9fa', 'padding': '10px', 'borderRadius': '4px'})
                                        ]),

                                        # Noise sources diagram
                                        html.H5("Noise Budget", style={'marginTop': '15px', 'textAlign': 'center'}),
                                        html.Div([
                                            html.Div([
                                                html.Div("Shot Noise", style={'display': 'inline-block', 'width': '30%', 'fontWeight': 'bold'}),
                                                html.Div("σ = √S", style={'display': 'inline-block', 'width': '25%', 'fontFamily': 'monospace'}),
                                                html.Div("████████", style={'display': 'inline-block', 'color': '#3498db', 'fontFamily': 'monospace'}),
                                            ], style={'marginBottom': '5px'}),
                                            html.Div([
                                                html.Div("Dark Noise", style={'display': 'inline-block', 'width': '30%', 'fontWeight': 'bold'}),
                                                html.Div("σ = √(D×t)", style={'display': 'inline-block', 'width': '25%', 'fontFamily': 'monospace'}),
                                                html.Div("████", style={'display': 'inline-block', 'color': '#9b59b6', 'fontFamily': 'monospace'}),
                                            ], style={'marginBottom': '5px'}),
                                            html.Div([
                                                html.Div("Read Noise", style={'display': 'inline-block', 'width': '30%', 'fontWeight': 'bold'}),
                                                html.Div("σ = R", style={'display': 'inline-block', 'width': '25%', 'fontFamily': 'monospace'}),
                                                html.Div("██", style={'display': 'inline-block', 'color': '#e74c3c', 'fontFamily': 'monospace'}),
                                            ], style={'marginBottom': '10px'}),
                                            html.Div([
                                                html.Div("Total Noise", style={'display': 'inline-block', 'width': '30%', 'fontWeight': 'bold'}),
                                                html.Div("σ = √(S+D+R²)", style={'display': 'inline-block', 'width': '25%', 'fontFamily': 'monospace'}),
                                                html.Div("██████████", style={'display': 'inline-block', 'color': '#2c3e50', 'fontFamily': 'monospace'}),
                                            ]),
                                        ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px', 'fontSize': '12px'}),

                                    ], style={'padding': '15px', 'backgroundColor': '#fafafa', 'borderRadius': '8px', 'border': '1px solid #ddd'})
                                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
                            ], style={'display': 'flex'})
                        ])
                    ])
                ], style={'padding': '20px'})
            ])

        ], style={'padding': '10px'})

    ], style={'fontFamily': 'Arial, sans-serif'})

    # Callbacks
    @app.callback(
        [Output('transmittance-plot', 'figure'),
         Output('path-radiance-plot', 'figure')],
        [Input('pwv-slider', 'value'),
         Input('aod-slider', 'value'),
         Input('sza-slider', 'value'),
         Input('aerosol-dropdown', 'value')]
    )
    def update_atmospheric_plots(pwv, aod, sza, aerosol_type):
        from ..atmosphere.aerosol_scattering import AerosolType

        atm = AtmosphereSimulator(wavelengths)

        # Set the aerosol type based on dropdown selection
        aerosol_type_map = {
            'continental': AerosolType.CONTINENTAL,
            'maritime': AerosolType.MARITIME,
            'urban': AerosolType.URBAN,
            'desert': AerosolType.DESERT
        }
        atm.aerosol.set_aerosol_type(aerosol_type_map.get(aerosol_type, AerosolType.CONTINENTAL))

        # Get atmospheric effects
        airmass = 1.0 / np.cos(np.radians(sza)) if sza < 85 else 10.0
        T_gas = atm.gas_absorption.transmission(pwv_cm=pwv, airmass=airmass)
        T_rayleigh = atm.rayleigh.transmission(airmass=airmass)
        T_aerosol = atm.aerosol.transmission(aod_550=aod, airmass=airmass)
        T_total = T_gas * T_rayleigh * T_aerosol

        # Transmittance plot
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=wavelengths, y=T_gas, name='Gas absorption',
                                  line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=wavelengths, y=T_rayleigh, name='Rayleigh',
                                  line=dict(color='orange')))
        fig1.add_trace(go.Scatter(x=wavelengths, y=T_aerosol, name='Aerosol',
                                  line=dict(color='green')))
        fig1.add_trace(go.Scatter(x=wavelengths, y=T_total, name='Total',
                                  line=dict(color='red', width=2)))

        # Add atmospheric absorption band shading
        for band in absorption_bands:
            fig1.add_vrect(
                x0=band['x0'], x1=band['x1'],
                fillcolor=band['color'],
                layer='below', line_width=0,
            )
            # Add label at top of band
            fig1.add_annotation(
                x=(band['x0'] + band['x1']) / 2,
                y=1.02,
                text=band['label'],
                showarrow=False,
                font=dict(size=9, color='#555'),
                yref='y'
            )

        fig1.update_layout(
            title=f'Atmospheric Transmittance (PWV={pwv}cm, AOD={aod}, SZA={sza}°, {aerosol_type.title()})',
            xaxis_title='Wavelength (nm)',
            yaxis_title='Transmittance',
            yaxis=dict(range=[0, 1.08]),
            legend=dict(x=0.02, y=0.15)
        )

        # Path radiance plot
        solar_irr = atm.solar.irradiance()
        L_path = atm.rayleigh.path_radiance(solar_irr, sza, 0)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=wavelengths, y=L_path,
                                  fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.3)',
                                  line=dict(color='#3498db'), name='Rayleigh path radiance'))

        # Add same absorption band markers to path radiance plot
        for band in absorption_bands:
            fig2.add_vrect(
                x0=band['x0'], x1=band['x1'],
                fillcolor=band['color'],
                layer='below', line_width=0,
            )

        # Add annotation explaining λ^-4 dependence
        fig2.add_annotation(
            x=600, y=L_path[np.argmin(np.abs(wavelengths - 600))] * 0.7,
            text="Rayleigh: τ ∝ λ⁻⁴<br>(blue light scatters more)",
            showarrow=True,
            arrowhead=2,
            ax=80, ay=-40,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.8)'
        )

        fig2.update_layout(
            title=f'Atmospheric Path Radiance (SZA={sza}°)',
            xaxis_title='Wavelength (nm)',
            yaxis_title='Path Radiance (W/m²/sr/nm)',
            legend=dict(x=0.7, y=0.9)
        )

        return fig1, fig2

    @app.callback(
        [Output('snr-plot', 'figure'),
         Output('noise-breakdown-plot', 'figure')],
        [Input('read-noise-slider', 'value'),
         Input('dark-slider', 'value'),
         Input('integration-slider', 'value'),
         Input('radiance-level-dropdown', 'value')]
    )
    def update_sensor_plots(read_noise, dark_current, integration_ms, radiance_level):
        from ..sensor.detector_model import DetectorModel, NoiseModel

        noise_model = NoiseModel(
            read_noise_e=read_noise,
            dark_current_e_s=dark_current
        )
        detector = DetectorModel(noise_model)

        integration_s = integration_ms / 1000

        # Realistic spectral signal model based on:
        # 1. Solar irradiance (approximated: peaks ~500nm, decreases into SWIR)
        # 2. Atmospheric transmittance (dips at absorption bands)
        # 3. Detector quantum efficiency (peaks in visible/NIR, drops in SWIR)
        # 4. Surface reflectance (assume ~30% gray target)

        # Solar irradiance relative curve (normalized, peaks in visible)
        solar_relative = np.exp(-((wavelengths - 500)**2) / (600**2))
        # Add SWIR dropoff
        solar_relative *= np.where(wavelengths > 1000,
                                   np.exp(-(wavelengths - 1000) / 1500), 1.0)

        # Atmospheric transmittance with absorption bands
        transmittance = np.ones_like(wavelengths)
        # H2O absorption bands (approximate depths)
        for center, width, depth in [(720, 30, 0.15), (820, 30, 0.15),
                                      (940, 60, 0.4), (1140, 60, 0.35),
                                      (1380, 100, 0.85), (1880, 120, 0.9)]:
            transmittance *= 1 - depth * np.exp(-((wavelengths - center)**2) / (width**2))
        # O2 bands
        for center, width, depth in [(688, 10, 0.3), (762, 15, 0.5)]:
            transmittance *= 1 - depth * np.exp(-((wavelengths - center)**2) / (width**2))

        # Detector QE (peaks around 700nm, drops in blue and SWIR)
        qe_relative = 0.4 + 0.6 * np.exp(-((wavelengths - 700)**2) / (500**2))
        qe_relative *= np.where(wavelengths > 1800,
                                np.exp(-(wavelengths - 1800) / 400), 1.0)

        # Combined spectral response
        spectral_response = solar_relative * transmittance * qe_relative
        spectral_response = spectral_response / spectral_response.max()  # Normalize

        # Base signal rate (electrons/second at peak)
        # Typical AVIRIS: ~10^4 to 10^5 electrons for a 30% reflectance target
        base_rate = 100000 * radiance_level  # electrons per second at peak
        signal = spectral_response * base_rate * integration_s

        # Noise components (standard CCD/CMOS noise model)
        shot_noise = np.sqrt(signal)  # Poisson statistics
        dark_electrons = dark_current * integration_s
        dark_noise = np.sqrt(dark_electrons)  # Also Poisson
        read = read_noise  # Gaussian, constant

        # Total noise and SNR: noise variances add
        # SNR = S / sqrt(S + D*t + R^2) - the CCD equation
        total_noise = np.sqrt(signal + dark_electrons + read**2)
        snr = signal / total_noise

        # SNR plot
        fig1 = go.Figure()

        # Add atmospheric absorption bands with labels
        for band in absorption_bands:
            fig1.add_vrect(
                x0=band['x0'], x1=band['x1'],
                fillcolor=band['color'],
                layer='below', line_width=0,
            )
            # Add label at top using paper coordinates for y
            fig1.add_annotation(
                x=(band['x0'] + band['x1']) / 2,
                y=1.02,
                yref='paper',
                text=band['label'],
                showarrow=False,
                font=dict(size=8, color='#555'),
            )

        fig1.add_trace(go.Scatter(x=wavelengths, y=snr, name='SNR',
                                  line=dict(color='#27ae60', width=2)))

        # Add labeled reference lines with annotations positioned using paper coordinates
        fig1.add_shape(type='line', x0=380, x1=2500, y0=100, y1=100,
                      line=dict(color='#3498db', width=1.5, dash='dash'))
        fig1.add_annotation(
            xref='paper', yref='y',
            x=1.0, y=100,
            text="◀ Target SNR = 100",
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=10, color='#3498db'),
        )

        fig1.add_shape(type='line', x0=380, x1=2500, y0=10, y1=10,
                      line=dict(color='#e74c3c', width=1.5, dash='dash'))
        fig1.add_annotation(
            xref='paper', yref='y',
            x=1.0, y=10,
            text="◀ Minimum SNR = 10",
            showarrow=False,
            xanchor='left',
            yanchor='middle',
            font=dict(size=10, color='#e74c3c'),
        )

        fig1.update_layout(
            title=f'Signal-to-Noise Ratio',
            xaxis_title='Wavelength (nm)',
            yaxis_title='SNR',
            yaxis_type='log',
            yaxis=dict(range=[0, 3]),  # 1 to 1000 on log scale
            margin=dict(r=120)  # Extra right margin for labels
        )

        # Noise breakdown - show all components
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=['Shot Noise', 'Dark Noise', 'Read Noise', 'Total'],
            y=[np.mean(shot_noise), dark_noise, read, np.mean(total_noise)],
            marker_color=['#3498db', '#9b59b6', '#e74c3c', '#2c3e50'],
            text=[f'{np.mean(shot_noise):.1f}', f'{dark_noise:.1f}', f'{read:.1f}', f'{np.mean(total_noise):.1f}'],
            textposition='outside'
        ))

        fig2.update_layout(
            title=f'Noise Components (t={integration_ms}ms, dark={dark_current} e⁻/s)',
            yaxis_title='Noise (electrons RMS)',
            showlegend=False
        )

        return fig1, fig2

    @app.callback(
        [Output('chain-plot', 'figure'),
         Output('components-plot', 'figure')],
        [Input('surface-dropdown', 'value'),
         Input('fm-pwv-slider', 'value'),
         Input('fm-aod-slider', 'value'),
         Input('display-options', 'value')]
    )
    def update_forward_model_plots(surface_type, pwv, aod, display_options):
        model = ForwardModel(wavelengths)
        targets = model.generate_test_targets()

        # Get reflectance
        if surface_type == 'white':
            refl = np.ones_like(wavelengths) * 0.95
        elif surface_type == 'gray':
            refl = np.ones_like(wavelengths) * 0.25
        else:
            refl = targets[surface_type]

        from ..forward_model.forward_model import SceneParameters
        scene = SceneParameters(
            surface_reflectance=refl,
            pwv_cm=pwv,
            aod_550=aod
        )

        result = model.simulate(scene, add_noise='noise' in display_options,
                               return_intermediates=True)

        # Get radiance - add visible noise if enabled
        at_sensor_radiance = result['at_sensor_radiance'].copy()
        apparent_refl = result['apparent_reflectance'].copy()

        if 'noise' in display_options:
            # Add realistic sensor noise to the radiance for visualization
            # Noise model: SNR proportional to sqrt(signal) (shot noise dominated)
            # Higher radiance = higher SNR, noise std = signal / SNR
            signal_level = at_sensor_radiance / (at_sensor_radiance.max() + 1e-10)
            # SNR ranges from ~30 at low signal to ~150 at high signal
            snr = 30 + 120 * np.sqrt(signal_level)
            noise_std = at_sensor_radiance / (snr + 1e-10)
            np.random.seed(42)  # Reproducible noise
            noise = np.random.normal(0, noise_std)
            at_sensor_radiance = at_sensor_radiance + noise

            # Also add noise to apparent reflectance (proportional noise)
            refl_noise_std = apparent_refl / (snr + 1e-10)
            apparent_refl = apparent_refl + np.random.normal(0, refl_noise_std)

        # Main chain plot
        fig1 = make_subplots(rows=2, cols=1,
                            subplot_titles=['Surface Reflectance', 'At-Sensor Radiance'],
                            vertical_spacing=0.15)

        # Add atmospheric absorption bands to both subplots using shapes
        for band in absorption_bands:
            # Top subplot (reflectance) - uses xaxis/yaxis (x/y)
            fig1.add_shape(
                type='rect',
                x0=band['x0'], x1=band['x1'],
                y0=0, y1=1,
                yref='y domain',
                fillcolor=band['color'],
                layer='below', line_width=0,
                xref='x',
            )
            # Bottom subplot (radiance) - uses xaxis2/yaxis2 (x2/y2)
            fig1.add_shape(
                type='rect',
                x0=band['x0'], x1=band['x1'],
                y0=0, y1=1,
                yref='y2 domain',
                fillcolor=band['color'],
                layer='below', line_width=0,
                xref='x2',
            )

        # Add band labels to top subplot only (to avoid clutter)
        for band in absorption_bands:
            fig1.add_annotation(
                x=(band['x0'] + band['x1']) / 2,
                y=1.0,
                yref='y domain',
                text=band['label'],
                showarrow=False,
                font=dict(size=8, color='#555'),
                xref='x',
            )

        fig1.add_trace(go.Scatter(x=wavelengths, y=refl, name='True Reflectance',
                                  line=dict(color='#27ae60')),
                      row=1, col=1)

        if 'apparent' in display_options:
            fig1.add_trace(go.Scatter(x=wavelengths, y=apparent_refl,
                                      name='Apparent Reflectance',
                                      line=dict(color='#e74c3c', dash='dash')),
                          row=1, col=1)

        fig1.add_trace(go.Scatter(x=wavelengths, y=at_sensor_radiance,
                                  name='Total Radiance' + (' (with noise)' if 'noise' in display_options else ''),
                                  line=dict(color='#3498db')),
                      row=2, col=1)

        if 'path' in display_options and result.get('path_radiance') is not None:
            fig1.add_trace(go.Scatter(x=wavelengths, y=result['path_radiance'],
                                      name='Path Radiance', fill='tozeroy',
                                      fillcolor='rgba(231, 76, 60, 0.3)',
                                      line=dict(color='#e74c3c')),
                          row=2, col=1)

        fig1.update_layout(height=650, showlegend=True)
        fig1.update_xaxes(title_text='Wavelength (nm)', row=2, col=1)
        fig1.update_yaxes(title_text='Reflectance', row=1, col=1)
        fig1.update_yaxes(title_text='Radiance (W/m²/sr/nm)', row=2, col=1)

        # Components plot
        total_rad = result['at_sensor_radiance']
        path_rad = result.get('path_radiance', np.zeros_like(wavelengths))
        surface_rad = total_rad - path_rad

        fig2 = go.Figure()

        # Add atmospheric absorption bands with labels
        for band in absorption_bands:
            fig2.add_vrect(
                x0=band['x0'], x1=band['x1'],
                fillcolor=band['color'],
                layer='below', line_width=0,
            )
            # Add label at top
            fig2.add_annotation(
                x=(band['x0'] + band['x1']) / 2,
                y=102,
                text=band['label'],
                showarrow=False,
                font=dict(size=8, color='#555'),
            )

        fig2.add_trace(go.Scatter(x=wavelengths, y=path_rad / total_rad * 100,
                                  name='Path Radiance %',
                                  fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.5)'))
        fig2.add_trace(go.Scatter(x=wavelengths, y=np.ones_like(wavelengths) * 100,
                                  name='Surface Signal %',
                                  fill='tonexty', fillcolor='rgba(39, 174, 96, 0.5)'))

        fig2.update_layout(
            title='Signal Composition',
            xaxis_title='Wavelength (nm)',
            yaxis_title='Fraction (%)',
            yaxis=dict(range=[0, 108])
        )

        return fig1, fig2

    return app


def _launch_matplotlib_interactive():
    """Fallback interactive mode using matplotlib."""
    import matplotlib.pyplot as plt

    print("\n" + "="*60)
    print("HSI Learning Toolkit - Matplotlib Interactive Mode")
    print("="*60)
    print("\nUse the following commands to explore:")
    print("  from hsi_toolkit import AtmosphereSimulator, SensorSimulator")
    print("  from hsi_toolkit.visualization import HSIVisualizer")
    print()
    print("Example:")
    print("  viz = HSIVisualizer()")
    print("  atm = AtmosphereSimulator()")
    print("  fig = viz.plot_atmospheric_transmittance(...)")
    print("  plt.show()")
    print()
    print("For full interactive dashboard, install Dash:")
    print("  pip install dash plotly")
    print("="*60 + "\n")
