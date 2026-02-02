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
    app.run_server(debug=debug, port=port)


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

                            html.Label("Precipitable Water Vapor (cm)"),
                            dcc.Slider(id='pwv-slider', min=0.1, max=5.0, step=0.1,
                                      value=1.5, marks={i: str(i) for i in range(6)}),

                            html.Label("Aerosol Optical Depth (550nm)"),
                            dcc.Slider(id='aod-slider', min=0.01, max=0.5, step=0.01,
                                      value=0.1, marks={0: '0', 0.25: '0.25', 0.5: '0.5'}),

                            html.Label("Solar Zenith Angle (°)"),
                            dcc.Slider(id='sza-slider', min=0, max=80, step=5,
                                      value=30, marks={i: str(i) for i in range(0, 81, 20)}),

                            html.Label("Aerosol Type"),
                            dcc.Dropdown(id='aerosol-dropdown',
                                        options=[
                                            {'label': 'Continental', 'value': 'continental'},
                                            {'label': 'Maritime', 'value': 'maritime'},
                                            {'label': 'Urban', 'value': 'urban'},
                                            {'label': 'Desert', 'value': 'desert'}
                                        ],
                                        value='continental'),

                        ], style={'width': '25%', 'padding': '20px', 'display': 'inline-block',
                                 'verticalAlign': 'top'}),

                        # Plots
                        html.Div([
                            dcc.Graph(id='transmittance-plot'),
                            dcc.Graph(id='path-radiance-plot')
                        ], style={'width': '70%', 'display': 'inline-block'})

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

                            html.Label("Read Noise (electrons)"),
                            dcc.Slider(id='read-noise-slider', min=5, max=100, step=5,
                                      value=20, marks={5: '5', 50: '50', 100: '100'}),

                            html.Label("Dark Current (e-/s)"),
                            dcc.Slider(id='dark-slider', min=10, max=500, step=10,
                                      value=100, marks={10: '10', 250: '250', 500: '500'}),

                            html.Label("Integration Time (ms)"),
                            dcc.Slider(id='integration-slider', min=1, max=50, step=1,
                                      value=10, marks={1: '1', 25: '25', 50: '50'}),

                            html.Label("Scene Radiance Level"),
                            dcc.Dropdown(id='radiance-level-dropdown',
                                        options=[
                                            {'label': 'Low (shadow)', 'value': 0.3},
                                            {'label': 'Medium', 'value': 1.0},
                                            {'label': 'High (bright)', 'value': 3.0}
                                        ],
                                        value=1.0),

                        ], style={'width': '25%', 'padding': '20px', 'display': 'inline-block',
                                 'verticalAlign': 'top'}),

                        # Plots
                        html.Div([
                            dcc.Graph(id='snr-plot'),
                            dcc.Graph(id='noise-breakdown-plot')
                        ], style={'width': '70%', 'display': 'inline-block'})

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
                            html.H4("Surface Type"),
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

                            html.H4("Atmosphere"),
                            html.Label("PWV (cm)"),
                            dcc.Slider(id='fm-pwv-slider', min=0.5, max=4.0, step=0.5,
                                      value=1.5, marks={0.5: '0.5', 2: '2', 4: '4'}),

                            html.Label("AOD (550nm)"),
                            dcc.Slider(id='fm-aod-slider', min=0.05, max=0.3, step=0.05,
                                      value=0.1, marks={0.05: '0.05', 0.15: '0.15', 0.3: '0.3'}),

                            html.H4("Display"),
                            dcc.Checklist(id='display-options',
                                         options=[
                                             {'label': 'Show path radiance', 'value': 'path'},
                                             {'label': 'Show apparent reflectance', 'value': 'apparent'},
                                             {'label': 'Add sensor noise', 'value': 'noise'}
                                         ],
                                         value=['path'])

                        ], style={'width': '25%', 'padding': '20px', 'display': 'inline-block',
                                 'verticalAlign': 'top'}),

                        # Plots
                        html.Div([
                            dcc.Graph(id='chain-plot'),
                            dcc.Graph(id='components-plot')
                        ], style={'width': '70%', 'display': 'inline-block'})

                    ], style={'display': 'flex'})
                ], style={'padding': '20px'})
            ]),

            # Tab 4: Educational Content
            dcc.Tab(label='Learn', children=[
                html.Div([
                    html.H3("Understanding Hyperspectral Imaging"),

                    dcc.Tabs([
                        dcc.Tab(label='The Imaging Chain', children=[
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
                        ]),

                        dcc.Tab(label='Atmospheric Effects', children=[
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
                        ]),

                        dcc.Tab(label='Sensor Physics', children=[
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
        atm = AtmosphereSimulator(wavelengths)

        state = AtmosphericState(
            solar_zenith_deg=sza,
            pwv_cm=pwv,
            aod_550=aod,
            aerosol_type=aerosol_type
        )

        # Get atmospheric effects
        T_gas = atm.gas_absorption.combined_transmission(wavelengths, pwv, 0.34)
        T_rayleigh = atm.rayleigh.transmission(wavelengths, sza, 0)
        T_aerosol = atm.aerosol.transmission(wavelengths)
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

        fig1.update_layout(
            title=f'Atmospheric Transmittance (PWV={pwv}cm, AOD={aod}, SZA={sza}°)',
            xaxis_title='Wavelength (nm)',
            yaxis_title='Transmittance',
            yaxis=dict(range=[0, 1.05]),
            legend=dict(x=0.7, y=0.1)
        )

        # Path radiance plot
        L_path = atm.rayleigh.path_radiance(wavelengths, sza, 0)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=wavelengths, y=L_path,
                                  fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.3)',
                                  line=dict(color='#3498db')))
        fig2.update_layout(
            title='Atmospheric Path Radiance',
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

        # Simulate typical signal
        base_signal = 5000 * radiance_level  # electrons
        signal = np.linspace(0.1 * base_signal, 2 * base_signal, len(wavelengths))

        # Calculate SNR
        snr = detector.calculate_snr(signal, integration_s)

        # Noise components
        shot_noise = np.sqrt(signal)
        dark_noise = np.sqrt(dark_current * integration_s)
        read = read_noise

        # SNR plot
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=wavelengths, y=snr, name='SNR',
                                  line=dict(color='#27ae60', width=2)))
        fig1.add_hline(y=100, line_dash='dash', line_color='blue',
                      annotation_text='Target SNR=100')
        fig1.add_hline(y=10, line_dash='dash', line_color='red',
                      annotation_text='Minimum SNR=10')

        fig1.update_layout(
            title='Signal-to-Noise Ratio',
            xaxis_title='Wavelength (nm)',
            yaxis_title='SNR',
            yaxis_type='log',
            yaxis=dict(range=[0.5, 3])
        )

        # Noise breakdown
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Shot', x=['Shot', 'Dark', 'Read'],
                             y=[np.mean(shot_noise), dark_noise, read],
                             marker_color=['#3498db', '#9b59b6', '#e74c3c']))

        fig2.update_layout(
            title=f'Noise Components (t={integration_ms}ms)',
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

        # Main chain plot
        fig1 = make_subplots(rows=2, cols=1,
                            subplot_titles=['Surface Reflectance', 'At-Sensor Radiance'])

        fig1.add_trace(go.Scatter(x=wavelengths, y=refl, name='True Reflectance',
                                  line=dict(color='#27ae60')),
                      row=1, col=1)

        if 'apparent' in display_options:
            fig1.add_trace(go.Scatter(x=wavelengths, y=result['apparent_reflectance'],
                                      name='Apparent Reflectance',
                                      line=dict(color='#e74c3c', dash='dash')),
                          row=1, col=1)

        fig1.add_trace(go.Scatter(x=wavelengths, y=result['at_sensor_radiance'],
                                  name='Total Radiance',
                                  line=dict(color='#3498db')),
                      row=2, col=1)

        if 'path' in display_options and result.get('path_radiance') is not None:
            fig1.add_trace(go.Scatter(x=wavelengths, y=result['path_radiance'],
                                      name='Path Radiance', fill='tozeroy',
                                      fillcolor='rgba(231, 76, 60, 0.3)',
                                      line=dict(color='#e74c3c')),
                          row=2, col=1)

        fig1.update_layout(height=600, showlegend=True)
        fig1.update_xaxes(title_text='Wavelength (nm)', row=2, col=1)
        fig1.update_yaxes(title_text='Reflectance', row=1, col=1)
        fig1.update_yaxes(title_text='Radiance (W/m²/sr/nm)', row=2, col=1)

        # Components plot
        total_rad = result['at_sensor_radiance']
        path_rad = result.get('path_radiance', np.zeros_like(wavelengths))
        surface_rad = total_rad - path_rad

        fig2 = go.Figure()
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
            yaxis=dict(range=[0, 100])
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
