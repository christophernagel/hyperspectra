"""
Main trame application for the hyperspectral web viewer.
"""

import logging
import re
import threading

import plotly.graph_objects as go
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as v3, plotly as trame_plotly

from .engine import WebEngine
from .layers import LayerManager
from .rois import ROIManager
from .ui.processing import build_processing_panel
from .ui.view_3d import build_3d_controls
from .figures import (
    make_spectral_figure,
    make_empty_figure,
    get_plot_layout,
    _geo_tick_overrides,
    CMAP_TO_PLOTLY,
)
from .plotly_3d import build_cube_slices_figure
from aviris_tools.viewer.constants import (
    COMPOSITE_PRESETS,
    INDEX_DEFINITIONS,
    INDEX_METADATA,
)

# Conditional VTK imports
try:
    from trame.widgets import vtk as trame_vtk
    from .vtk_3d import VtkCubeViewer
    HAS_VTK = True
except ImportError:
    HAS_VTK = False

logger = logging.getLogger(__name__)

SPECTRUM_COLORS = ["cyan", "magenta", "yellow", "lime", "orange"]
_DRAGMODE_MAP = {
    "none": "pan",
    "rect": "drawrect",
    "freehand": "drawopenpath",
    "polygon": "drawclosedpath",
}


class HyperspectralWebApp:
    def __init__(self, server=None, data_dir="/data"):
        self.server = server or get_server(client_type="vue3")
        self.state = self.server.state
        self.ctrl = self.server.controller
        self.engine = WebEngine()
        self.data_dir = data_dir

        self._pixel_spectra = []
        self._main_fig_widget = None
        self._spectral_fig_widget = None
        self._3d_plotly_widget = None
        self._vtk_view_widget = None
        self._vtk_viewer = None

        self._init_state()

        # Managers (created after state init so they can reference state)
        self.layer_mgr = LayerManager(
            self.state, on_change=self._rebuild_main_figure
        )
        self.roi_mgr = ROIManager(
            self.state, self.engine, on_change=self._on_roi_changed
        )

        self._register_callbacks()
        self._build_ui()

    # =================================================================
    # State
    # =================================================================

    def _init_state(self):
        self.state.trame__title = "Hyperspectral Viewer"

        # File browser
        self.state.file_list = WebEngine.scan_directory(self.data_dir)
        self.state.current_file = None
        self.state.file_info = {}

        # Band slider
        self.state.current_wavelength = 550
        self.state.wl_min = 380
        self.state.wl_max = 2500

        # Composites
        self.state.composite_preset = None
        self.state.custom_r = 640
        self.state.custom_g = 550
        self.state.custom_b = 470

        # Indices
        index_items = [{"title": name, "value": name} for name in INDEX_DEFINITIONS]
        self.state.index_items = index_items
        self.state.selected_index = "NDVI"
        self.state.index_description = INDEX_METADATA.get("NDVI", {}).get("desc", "")

        # Layout
        self.state.image_height = 85
        self.state.active_tab = "view"
        self.state.dark_mode = True

        # Layers (serializable summary for Vue)
        self.state.layer_list_ui = []

        # ROIs
        self.state.roi_list_ui = []
        self.state.roi_draw_mode = "none"
        self.state.index_meta_low = ""
        self.state.index_meta_high = ""

        # 3D Viewer
        self.state.view_mode = "2d"  # "2d" or "3d"
        self.state.render_3d_mode = "plotly"  # "plotly" or "vtk"
        self.state.slice_x_index = 0
        self.state.slice_y_index = 0
        self.state.slice_z_index = 0
        self.state.slice_x_show = True
        self.state.slice_y_show = True
        self.state.slice_z_show = True
        self.state.slice_x_max = 100
        self.state.slice_y_max = 100
        self.state.slice_z_max = 100
        self.state.spatial_downsample = 2
        self.state.spectral_downsample = 4
        self.state.colormap_3d = "Viridis"
        self.state.opacity_3d = 0.9
        self.state.cube_3d_loaded = False
        self.state.cube_3d_loading = False

        # L1 Processing
        self.state.l1_file_list = WebEngine.scan_directory(self.data_dir)
        self.state.l1_radiance_file = None
        self.state.l1_obs_file = None
        self.state.l1_output_path = ""
        self.state.l1_aerosol = "continental"
        self.state.l1_altitude = "8.5"
        self.state.l1_use_6s = True
        self.state.l1_validate = True
        self.state.l1_processing = False
        self.state.l1_progress = 0  # 0-100
        self.state.l1_progress_label = ""
        self.state.l1_log = ""

    # =================================================================
    # Callbacks
    # =================================================================

    def _register_callbacks(self):
        self.state.change("current_file")(self._on_file_changed)
        self.state.change("current_wavelength")(self._on_wavelength_changed)
        self.state.change("selected_index")(self._on_index_selected)
        self.state.change("dark_mode")(self._on_dark_mode_changed)
        self.state.change("roi_draw_mode")(self._on_roi_draw_mode_changed)
        self.ctrl.apply_composite = self._apply_composite
        self.ctrl.apply_custom_rgb = self._apply_custom_rgb
        self.ctrl.calculate_index = self._calculate_index
        self.ctrl.on_image_click = self._on_image_click
        self.ctrl.on_image_double_click = self._on_image_double_click
        self.ctrl.on_image_relayout = self.roi_mgr.on_image_relayout
        self.ctrl.clear_spectra = self._clear_spectra
        self.ctrl.run_atm_correction = self._run_atm_correction

        # Layer controls
        self.ctrl.remove_layer = self.layer_mgr.remove
        self.ctrl.clear_all_layers = self.layer_mgr.clear_all
        self.ctrl.set_layer_visible = self.layer_mgr.set_visible
        self.ctrl.set_layer_opacity = self.layer_mgr.set_opacity
        self.ctrl.move_layer_up = lambda lid: self.layer_mgr.move(int(lid), 1)
        self.ctrl.move_layer_down = lambda lid: self.layer_mgr.move(int(lid), -1)

        # ROI controls
        self.ctrl.remove_roi = self.roi_mgr.remove
        self.ctrl.clear_all_rois = self.roi_mgr.clear_all
        self.ctrl.set_roi_visible = self.roi_mgr.set_visible

        # 3D controls
        self.state.change("view_mode")(self._on_view_mode_changed)
        self.ctrl.load_3d_cube = self._load_3d_cube
        self.ctrl.on_3d_click = self._on_3d_click
        self.state.change("slice_x_index")(self._on_3d_slice_changed)
        self.state.change("slice_y_index")(self._on_3d_slice_changed)
        self.state.change("slice_z_index")(self._on_3d_slice_changed)
        self.state.change("slice_x_show")(self._on_3d_slice_changed)
        self.state.change("slice_y_show")(self._on_3d_slice_changed)
        self.state.change("slice_z_show")(self._on_3d_slice_changed)
        self.state.change("colormap_3d")(self._on_3d_slice_changed)
        self.state.change("opacity_3d")(self._on_3d_slice_changed)

    # ----- File / wavelength / index callbacks -----

    def _on_file_changed(self, current_file, **kwargs):
        if not current_file:
            return
        try:
            info = self.engine.load_file(current_file)
            self.state.file_info = info
            self.state.wl_min = int(info["wl_min"])
            self.state.wl_max = int(info["wl_max"])
            self.state.current_wavelength = 550
            self._pixel_spectra.clear()
            self.layer_mgr.reset()
            self.roi_mgr.reset()
            self.state.cube_3d_loaded = False
            band = self.engine.get_single_band(550.0)
            self.layer_mgr.add("Band 550 nm", "band", band,
                               colorscale="Viridis", colorbar_title="Value")
            self._update_spectral_figure()
            logger.info(f"Loaded {info['filename']}")
        except Exception as e:
            logger.error(f"Failed to load {current_file}: {e}")

    def _on_wavelength_changed(self, current_wavelength, **kwargs):
        if self.engine.data_loader is None:
            return
        wl = float(current_wavelength)
        band = self.engine.get_single_band(wl)
        # Update existing band layer in-place, or add one
        for layer in self.layer_mgr.layers:
            if layer["layer_type"] == "band":
                layer["data"] = band
                layer["name"] = f"Band {wl:.0f} nm"
                self.layer_mgr.sync_ui()
                self._rebuild_main_figure()
                return
        self.layer_mgr.add(f"Band {wl:.0f} nm", "band", band,
                           colorscale="Viridis", colorbar_title="Value")

    def _on_index_selected(self, selected_index, **kwargs):
        meta = INDEX_METADATA.get(selected_index, {})
        self.state.index_description = meta.get("desc", "")
        self.state.index_meta_low = meta.get("low", "")
        self.state.index_meta_high = meta.get("high", "")

    def _on_dark_mode_changed(self, dark_mode, **kwargs):
        self._rebuild_main_figure()
        self._update_spectral_figure()

    def _on_roi_draw_mode_changed(self, roi_draw_mode, **kwargs):
        self._rebuild_main_figure()

    # ----- Composite / index actions (now add layers) -----

    def _apply_composite(self, name):
        if self.engine.data_loader is None or name not in COMPOSITE_PRESETS:
            return
        p = COMPOSITE_PRESETS[name]
        r_wl, g_wl, b_wl = p["r"], p["g"], p["b"]
        rgb = self.engine.get_rgb_composite(r_wl, g_wl, b_wl)
        self.layer_mgr.add(f"{name} (R={r_wl}, G={g_wl}, B={b_wl})", "rgb", rgb)

    def _apply_custom_rgb(self):
        if self.engine.data_loader is None:
            return
        r_wl = float(self.state.custom_r)
        g_wl = float(self.state.custom_g)
        b_wl = float(self.state.custom_b)
        rgb = self.engine.get_rgb_composite(r_wl, g_wl, b_wl)
        self.layer_mgr.add(
            f"Custom (R={r_wl:.0f}, G={g_wl:.0f}, B={b_wl:.0f})",
            "rgb", rgb,
        )

    def _calculate_index(self):
        if self.engine.data_loader is None:
            return
        idx_name = self.state.selected_index
        data, idx_def = self.engine.calculate_index(idx_name)
        if data is None:
            return
        clim = self.engine.robust_percentile(data)
        meta = INDEX_METADATA.get(idx_name, {})
        self.layer_mgr.add(
            f"{idx_name} Index", "index", data,
            colorscale=idx_def["cmap"],
            zmin=clim[0], zmax=clim[1],
            colorbar_title=idx_name,
            colorbar_low=meta.get("low", ""),
            colorbar_high=meta.get("high", ""),
        )

    # ----- Click / spectra -----

    def _on_image_click(self, event_data):
        if self.engine.data_loader is None or not event_data:
            return
        points = event_data.get("points", [])
        if not points:
            return

        col = int(points[0].get("x", 0))
        row = int(points[0].get("y", 0))

        if self.engine._has_coords:
            r = min(row, len(self.engine.lat_axis) - 1)
            c = min(col, len(self.engine.lon_axis) - 1)
            label = f"({self.engine.lat_axis[r]:.4f}, {self.engine.lon_axis[c]:.4f})"
        else:
            label = f"({row}, {col})"

        spectrum = self.engine.get_pixel_spectrum(row, col)
        if spectrum is None:
            return

        color = SPECTRUM_COLORS[len(self._pixel_spectra) % len(SPECTRUM_COLORS)]
        self._pixel_spectra.append({
            "wavelengths": spectrum["wavelengths"],
            "values": spectrum["values"],
            "label": label,
            "color": color,
        })
        if len(self._pixel_spectra) > 5:
            self._pixel_spectra = self._pixel_spectra[-5:]

        self._update_spectral_figure()

    def _clear_spectra(self):
        self._pixel_spectra.clear()
        self._update_spectral_figure()

    def _on_image_double_click(self):
        """Reset zoom on double-click."""
        self._rebuild_main_figure()

    def _on_roi_changed(self):
        """Callback for ROI manager — rebuild both figures."""
        self._rebuild_main_figure()
        self._update_spectral_figure()

    # ----- 3D view callbacks -----

    def _on_view_mode_changed(self, view_mode, **kwargs):
        """Toggle between 2D and 3D main view."""
        if view_mode == "3d" and self.state.cube_3d_loaded:
            self._update_3d_plotly()

    def _load_3d_cube(self):
        """Load and prepare the 3D cube data."""
        if self.engine.data_loader is None:
            return

        self.state.cube_3d_loading = True
        dl = self.engine.data_loader

        # Update slice slider maxima
        self.state.slice_x_max = dl.n_cols - 1
        self.state.slice_y_max = dl.n_rows - 1
        self.state.slice_z_max = dl.n_bands - 1

        # Set initial slice positions to midpoints
        self.state.slice_x_index = dl.n_cols // 2
        self.state.slice_y_index = dl.n_rows // 2
        self.state.slice_z_index = dl.n_bands // 2

        # Load VTK cube if VTK available and vtk mode is active
        if HAS_VTK:
            spatial = int(self.state.spatial_downsample)
            spectral = int(self.state.spectral_downsample)
            cube, wavelengths = self.engine.get_downsampled_cube(spatial, spectral)
            if cube is not None:
                if self._vtk_viewer is None:
                    self._vtk_viewer = VtkCubeViewer()
                self._vtk_viewer.load_cube(cube, wavelengths)
                logger.info("VTK cube loaded")

        self.state.cube_3d_loaded = True
        self.state.cube_3d_loading = False

        # Update the 3D view
        if self.state.view_mode == "3d":
            self._update_3d_plotly()
            if HAS_VTK and self._vtk_view_widget is not None:
                self._vtk_view_widget.update()

    def _on_3d_slice_changed(self, **kwargs):
        """Regenerate 3D view when slice parameters change."""
        if self.state.view_mode != "3d" or not self.state.cube_3d_loaded:
            return
        if self.state.render_3d_mode == "plotly":
            self._update_3d_plotly()
        elif self.state.render_3d_mode == "vtk" and self._vtk_viewer is not None:
            self._vtk_viewer.set_slice_position("x", int(self.state.slice_x_index))
            self._vtk_viewer.set_slice_position("y", int(self.state.slice_y_index))
            self._vtk_viewer.set_slice_position("z", int(self.state.slice_z_index))
            self._vtk_viewer.set_slice_visible("x", self.state.slice_x_show)
            self._vtk_viewer.set_slice_visible("y", self.state.slice_y_show)
            self._vtk_viewer.set_slice_visible("z", self.state.slice_z_show)
            if self._vtk_view_widget is not None:
                self._vtk_view_widget.update()

    def _update_3d_plotly(self):
        """Regenerate the Plotly 3D figure."""
        if self._3d_plotly_widget is None or self.engine.data_loader is None:
            return

        fig = build_cube_slices_figure(
            self.engine,
            slice_x_index=int(self.state.slice_x_index) if self.state.slice_x_show else None,
            slice_y_index=int(self.state.slice_y_index) if self.state.slice_y_show else None,
            slice_z_index=int(self.state.slice_z_index) if self.state.slice_z_show else None,
            slice_x_show=self.state.slice_x_show,
            slice_y_show=self.state.slice_y_show,
            slice_z_show=self.state.slice_z_show,
            downsample=int(self.state.spatial_downsample),
            colormap=self.state.colormap_3d,
            opacity=float(self.state.opacity_3d),
            dark=self.state.dark_mode,
        )
        self._3d_plotly_widget.update(fig)

    def _on_3d_click(self, event_data):
        """Handle click on 3D surface to extract spectrum."""
        if self.engine.data_loader is None or not event_data:
            return
        points = event_data.get("points", [])
        if not points:
            return

        pt = points[0]
        # Plotly Surface: x=col, y=row in our coordinate system
        col = int(pt.get("x", 0))
        row = int(pt.get("y", 0))

        col = min(col, self.engine.data_loader.n_cols - 1)
        row = min(row, self.engine.data_loader.n_rows - 1)

        label = f"3D ({row}, {col})"
        spectrum = self.engine.get_pixel_spectrum(row, col)
        if spectrum is None:
            return

        color = SPECTRUM_COLORS[len(self._pixel_spectra) % len(SPECTRUM_COLORS)]
        self._pixel_spectra.append({
            "wavelengths": spectrum["wavelengths"],
            "values": spectrum["values"],
            "label": label,
            "color": color,
        })
        if len(self._pixel_spectra) > 5:
            self._pixel_spectra = self._pixel_spectra[-5:]
        self._update_spectral_figure()

    # =================================================================
    # Figure compositing
    # =================================================================

    def _rebuild_main_figure(self):
        """Composite all visible layers into a single Plotly figure."""
        if self._main_fig_widget is None:
            return

        dark = self.state.dark_mode
        visible = [ly for ly in self.layer_mgr.layers if ly["visible"]]
        if not visible:
            self._main_fig_widget.update(
                make_empty_figure("No visible layers", dark=dark)
            )
            return

        composite = self.layer_mgr.composite(dark)
        fig = go.Figure(data=go.Image(z=composite))

        # Axis setup
        n_rows, n_cols = composite.shape[:2]
        lat = self.engine.lat_axis
        lon = self.engine.lon_axis

        if lat is not None and lon is not None:
            x_ax, y_ax = _geo_tick_overrides(n_rows, n_cols, lat, lon)
            y_ax.pop("autorange", None)  # Image y goes top-down natively
        else:
            x_ax = {}
            y_ax = dict(scaleanchor="x")

        title = " + ".join(ly["name"] for ly in visible[:3])
        if len(visible) > 3:
            title += f" (+{len(visible) - 3} more)"

        # Dragmode for ROI drawing
        dm = _DRAGMODE_MAP.get(self.state.roi_draw_mode, "pan")

        # Add colorbar for the top-most visible band/index layer
        cbar_layer = None
        for ly in reversed(visible):
            if ly["layer_type"] in ("band", "index") and ly["zmin"] is not None:
                cbar_layer = ly
                break
        if cbar_layer is not None:
            zmin = cbar_layer["zmin"]
            zmax = cbar_layer["zmax"]
            low_label = cbar_layer.get("colorbar_low", "")
            high_label = cbar_layer.get("colorbar_high", "")
            plotly_cmap = CMAP_TO_PLOTLY.get(cbar_layer["colorscale"] or "viridis", "Viridis")
            # Use plain-language tick labels if available, otherwise numeric
            if low_label and high_label:
                cbar_dict = dict(
                    title=cbar_layer.get("colorbar_title", ""),
                    thickness=32, len=0.72,
                    tickvals=[zmin, zmax],
                    ticktext=[low_label, high_label],
                    tickfont=dict(size=9),
                )
            else:
                cbar_dict = dict(
                    title=cbar_layer.get("colorbar_title", ""),
                    thickness=32, len=0.72,
                )
            fig.add_trace(go.Heatmap(
                z=[[zmin, zmax]],
                colorscale=plotly_cmap,
                zmin=zmin, zmax=zmax,
                colorbar=cbar_dict,
                showscale=True,
                opacity=0,
                hoverinfo="skip",
            ))

        fig.update_layout(
            title=title,
            xaxis=x_ax,
            yaxis=y_ax,
            dragmode=dm,
            newshape=dict(
                line=dict(color="cyan", width=2, dash="dash"),
                fillcolor="rgba(0,255,255,0.1)",
            ),
            **get_plot_layout(dark),
        )

        # Re-add stored ROI shapes
        shapes = []
        for roi in self.roi_mgr.rois:
            sd = roi["shape_data"]
            color = roi["color"]
            if roi["visible"]:
                line_kw = dict(color=color, width=2, dash="solid")
                fill_opacity = 0.1
            else:
                line_kw = dict(color=color, width=1, dash="dot")
                fill_opacity = 0.03
            shape = dict(sd)  # copy
            shape["line"] = line_kw
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            shape["fillcolor"] = f"rgba({r}, {g}, {b}, {fill_opacity})"
            shapes.append(shape)
        if shapes:
            fig.update_layout(shapes=shapes)

        self._main_fig_widget.update(fig)

    def _update_spectral_figure(self):
        if self._spectral_fig_widget is None:
            return
        dark = self.state.dark_mode

        # Build ROI spectra list from visible ROIs
        roi_spectra = []
        for roi in self.roi_mgr.rois:
            if roi["visible"]:
                roi_spectra.append({
                    "wavelengths": roi["wavelengths"],
                    "mean_values": roi["mean_spectrum"],
                    "std_values": roi["std_spectrum"],
                    "label": roi["name"],
                    "color": roi["color"],
                })

        # Collect top match material names from visible ROIs
        match_materials = []
        for roi in self.roi_mgr.rois:
            if roi["visible"] and roi["matches"]:
                match_materials.append(roi["matches"][0]["name"])

        has_data = self._pixel_spectra or roi_spectra
        if not has_data:
            fig = make_empty_figure("Click a pixel to see its spectrum", dark=dark)
        else:
            data_type = "reflectance"
            if self.engine.data_loader:
                data_type = self.engine.data_loader.data_type
            logger.info(
                f"Spectral update: {len(self._pixel_spectra)} pixel, "
                f"{len(roi_spectra)} ROI spectra"
            )
            fig = make_spectral_figure(
                self._pixel_spectra, data_type, dark=dark,
                roi_spectra=roi_spectra,
                match_materials=match_materials,
            )
        self._spectral_fig_widget.update(fig)

    # =================================================================
    # Public API (for notebook access)
    # =================================================================

    @property
    def data_loader(self):
        return self.engine.data_loader

    @property
    def current_spectra(self):
        return list(self._pixel_spectra)

    @property
    def layers(self):
        """Get layer summary list."""
        return self.layer_mgr.get_summary()

    def get_current_image(self):
        """Get the topmost visible layer's data as numpy array."""
        return self.layer_mgr.get_topmost_visible()

    def load_file(self, filepath):
        self.state.current_file = str(filepath)

    def set_composite(self, name):
        self._apply_composite(name)

    def calculate_and_get_index(self, index_name):
        data, _ = self.engine.calculate_index(index_name)
        return data

    # =================================================================
    # L1 Processing
    # =================================================================

    def _run_atm_correction(self):
        if self.state.l1_processing:
            return

        radiance = self.state.l1_radiance_file
        obs = self.state.l1_obs_file
        output = self.state.l1_output_path

        if not radiance or not obs or not output:
            self.state.l1_log = "Error: Please select radiance file, obs file, and output path.\n"
            return

        self.state.l1_processing = True
        self.state.l1_progress = 0
        self.state.l1_progress_label = "Initializing..."
        self.state.l1_log = "Starting atmospheric correction...\n"

        def _run():
            handler = _StateLogHandler(self.state)
            handler.setLevel(logging.DEBUG)

            # Attach handler to loggers we need to capture.
            # Must add to root logger AND set its level before
            # the legacy module's basicConfig() runs.
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            if root_logger.level > logging.INFO:
                root_logger.setLevel(logging.INFO)

            try:
                from aviris_tools.atm_correction.py6s_processor import Py6SProcessor

                processor = Py6SProcessor(
                    radiance_path=radiance,
                    obs_path=obs,
                    output_path=output,
                    aerosol_model=self.state.l1_aerosol,
                    altitude_km=float(self.state.l1_altitude),
                    validate=self.state.l1_validate,
                )
                result = processor.run(use_6s=self.state.l1_use_6s)

                with self.server.state as state:
                    state.l1_log += f"\nComplete! Output: {result}\n"
                    state.l1_progress = 100
                    state.l1_progress_label = "Complete!"
                    state.l1_processing = False
                    state.file_list = WebEngine.scan_directory(self.data_dir)
                    state.l1_file_list = state.file_list

            except Exception as e:
                with self.server.state as state:
                    state.l1_log += f"\nERROR: {e}\n"
                    state.l1_processing = False

            finally:
                root_logger.removeHandler(handler)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    # =================================================================
    # UI
    # =================================================================

    def _build_ui(self):
        with SinglePageWithDrawerLayout(
            self.server,
            vuetify_config={"theme": {"defaultTheme": "dark"}},
        ) as layout:
            self.ui = layout
            layout.title.set_text("Hyperspectral Viewer")

            # --- Toolbar ---
            with layout.toolbar:
                v3.VSpacer()
                v3.VChip(
                    v_if="file_info.filename",
                    text=("file_info.filename",),
                    prepend_icon="mdi-file",
                    size="small",
                    variant="tonal",
                    color="primary",
                    classes="mr-2",
                )
                _info_tpl = (
                    "`${file_info.n_bands} bands"
                    " | ${file_info.n_rows}\u00d7${file_info.n_cols}"
                    " | ${file_info.data_type}`"
                )
                v3.VChip(
                    v_if="file_info.n_bands",
                    text=(_info_tpl,),
                    size="small",
                    variant="tonal",
                    classes="mr-2",
                )
                # 2D/3D toggle
                with v3.VBtnToggle(
                    v_model=("view_mode", "2d"),
                    mandatory=True,
                    density="compact",
                    classes="mx-2",
                    color="primary",
                ):
                    v3.VBtn(
                        icon="mdi-image",
                        value="2d",
                        size="small",
                    )
                    v3.VBtn(
                        icon="mdi-cube-outline",
                        value="3d",
                        size="small",
                    )

                _dark_toggle = (
                    "dark_mode = !dark_mode;"
                    " $vuetify.theme.global.name ="
                    " dark_mode ? 'dark' : 'light'"
                )
                v3.VBtn(
                    icon=("dark_mode ? 'mdi-weather-night'"
                          " : 'mdi-weather-sunny'",),
                    click=_dark_toggle,
                    size="small",
                    variant="text",
                )

            # --- Drawer (sidebar) ---
            with layout.drawer as drawer:
                drawer.width = 340

                with v3.VTabs(v_model=("active_tab", "view"), density="compact", grow=True):
                    v3.VTab(value="view", text="View", prepend_icon="mdi-eye")
                    v3.VTab(value="3d", text="3D", prepend_icon="mdi-cube-outline")
                    v3.VTab(value="layers", text="Layers", prepend_icon="mdi-layers")
                    v3.VTab(value="process", text="Process", prepend_icon="mdi-cog")

                v3.VDivider()

                with v3.VWindow(v_model="active_tab"):
                    with v3.VWindowItem(value="view"):
                        with v3.VContainer(fluid=True, classes="pa-2"):
                            self._build_file_section()
                            self._build_band_section()
                            self._build_composite_section()
                            self._build_index_section()
                            self._build_spectral_section()

                    with v3.VWindowItem(value="3d"):
                        with v3.VContainer(fluid=True, classes="pa-2"):
                            build_3d_controls(self.state, self.ctrl)

                    with v3.VWindowItem(value="layers"):
                        with v3.VContainer(fluid=True, classes="pa-2"):
                            self._build_layers_section()

                    with v3.VWindowItem(value="process"):
                        with v3.VContainer(fluid=True, classes="pa-2"):
                            build_processing_panel(
                                self.state, self.ctrl, self.data_dir
                            )

            # --- Main content ---
            with layout.content:
                with v3.VContainer(fluid=True, classes="fill-height pa-1"):
                    with v3.VCard(style="flex: 1;"):
                        # 2D Plotly view
                        self._main_fig_widget = trame_plotly.Figure(
                            figure=make_empty_figure("Select a data file to begin"),
                            display_mode_bar="hover",
                            style="width: 100%; height: 92vh;",
                            v_show=("view_mode === '2d'",),
                            click=(self.ctrl.on_image_click, "[$event]"),
                            double_click=self.ctrl.on_image_double_click,
                            relayout=(self.ctrl.on_image_relayout, "[$event]"),
                        )

                        # 3D Plotly view
                        self._3d_plotly_widget = trame_plotly.Figure(
                            figure=make_empty_figure(
                                "Go to the 3D tab and click Load 3D Cube"
                            ),
                            display_mode_bar="hover",
                            style="width: 100%; height: 92vh;",
                            v_show=(
                                "view_mode === '3d' && render_3d_mode === 'plotly'",
                            ),
                            click=(self.ctrl.on_3d_click, "[$event]"),
                        )

                        # VTK 3D view (conditional on VTK availability)
                        if HAS_VTK:
                            # Lazy-init VtkCubeViewer for render window
                            if self._vtk_viewer is None:
                                self._vtk_viewer = VtkCubeViewer()
                            self._vtk_view_widget = trame_vtk.VtkLocalView(
                                self._vtk_viewer.render_window,
                                style="width: 100%; height: 92vh;",
                                v_show=(
                                    "view_mode === '3d' && render_3d_mode === 'vtk'",
                                ),
                            )

    # ----- Sidebar sections -----

    def _build_file_section(self):
        with v3.VCard(variant="outlined", classes="mb-2"):
            v3.VCardTitle("Data File", classes="text-subtitle-2 pa-2")
            with v3.VCardText(classes="pa-2 pt-0"):
                v3.VSelect(
                    v_model=("current_file",),
                    items=("file_list",),
                    item_title="title",
                    item_value="value",
                    label="Select file",
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                    clearable=True,
                )

    def _build_band_section(self):
        with v3.VCard(variant="outlined", classes="mb-2"):
            v3.VCardTitle("Single Band", classes="text-subtitle-2 pa-2")
            with v3.VCardText(classes="pa-2 pt-0"):
                v3.VSlider(
                    v_model=("current_wavelength", 550),
                    min=("wl_min",),
                    max=("wl_max",),
                    step=1,
                    label="nm",
                    thumb_label="always",
                    hide_details=True,
                    density="compact",
                    color="primary",
                )

    def _build_composite_section(self):
        with v3.VCard(variant="outlined", classes="mb-2"):
            v3.VCardTitle("RGB Composites", classes="text-subtitle-2 pa-2")
            with v3.VCardText(classes="pa-2 pt-0"):
                with v3.VRow(dense=True, classes="mb-1"):
                    for name in COMPOSITE_PRESETS:
                        with v3.VCol(cols=4, classes="pa-1"):
                            v3.VBtn(
                                name,
                                click=(self.ctrl.apply_composite, f"['{name}']"),
                                size="x-small",
                                variant="tonal",
                                block=True,
                            )

                v3.VDivider(classes="my-2")

                with v3.VRow(dense=True):
                    with v3.VCol(cols=4):
                        v3.VTextField(
                            v_model=("custom_r", 640),
                            label="R",
                            type="number",
                            density="compact",
                            variant="outlined",
                            hide_details=True,
                        )
                    with v3.VCol(cols=4):
                        v3.VTextField(
                            v_model=("custom_g", 550),
                            label="G",
                            type="number",
                            density="compact",
                            variant="outlined",
                            hide_details=True,
                        )
                    with v3.VCol(cols=4):
                        v3.VTextField(
                            v_model=("custom_b", 470),
                            label="B",
                            type="number",
                            density="compact",
                            variant="outlined",
                            hide_details=True,
                        )
                v3.VBtn(
                    "Apply Custom RGB",
                    click=self.ctrl.apply_custom_rgb,
                    size="small",
                    variant="tonal",
                    color="primary",
                    block=True,
                    classes="mt-2",
                )

    def _build_index_section(self):
        with v3.VCard(variant="outlined", classes="mb-2"):
            v3.VCardTitle("Spectral Indices", classes="text-subtitle-2 pa-2")
            with v3.VCardText(classes="pa-2 pt-0"):
                v3.VSelect(
                    v_model=("selected_index", "NDVI"),
                    items=("index_items",),
                    item_title="title",
                    item_value="value",
                    label="Index",
                    density="compact",
                    variant="outlined",
                    hide_details=True,
                )
                v3.VAlert(
                    v_if="index_description",
                    text=("index_description",),
                    type="info",
                    variant="tonal",
                    density="compact",
                    classes="mt-2",
                )
                v3.VBtn(
                    "Calculate Index",
                    click=self.ctrl.calculate_index,
                    size="small",
                    variant="tonal",
                    color="secondary",
                    block=True,
                    classes="mt-2",
                )

    def _build_spectral_section(self):
        with v3.VCard(variant="outlined", classes="mb-2"):
            v3.VCardTitle("Spectra", classes="text-subtitle-2 pa-2")
            with v3.VCardText(classes="pa-2 pt-0"):
                # Index metadata labels
                with v3.VRow(
                    dense=True, classes="mb-2",
                    v_if="index_meta_low || index_meta_high",
                ):
                    with v3.VCol(cols=6):
                        v3.VChip(
                            v_if="index_meta_low",
                            text=("index_meta_low",),
                            prepend_icon="mdi-arrow-down",
                            color="error",
                            size="x-small",
                            variant="tonal",
                            label=True,
                        )
                    with v3.VCol(cols=6):
                        v3.VChip(
                            v_if="index_meta_high",
                            text=("index_meta_high",),
                            prepend_icon="mdi-arrow-up",
                            color="success",
                            size="x-small",
                            variant="tonal",
                            label=True,
                        )

                # ROI drawing mode
                v3.VLabel("ROI Drawing", classes="text-caption mb-1")
                with v3.VBtnToggle(
                    v_model=("roi_draw_mode", "none"),
                    density="compact",
                    variant="outlined",
                    divided=True,
                    mandatory=True,
                    classes="mb-2",
                ):
                    v3.VBtn(icon="mdi-cursor-default", value="none", size="small")
                    v3.VBtn(icon="mdi-rectangle-outline", value="rect", size="small")
                    v3.VBtn(icon="mdi-draw", value="freehand", size="small")
                    v3.VBtn(icon="mdi-vector-polygon", value="polygon", size="small")

                # ROI list (compact, inline)
                with v3.VSheet(
                    v_if="roi_list_ui.length > 0",
                    classes="mb-2 pa-2",
                    rounded=True,
                    border=True,
                ):
                    with v3.VRow(
                        dense=True, no_gutters=True,
                        classes="mb-1",
                        align="center",
                    ):
                        with v3.VCol(cols=8):
                            v3.VLabel(
                                text=("`ROIs (${roi_list_ui.length})`",),
                                classes="text-subtitle-2 font-weight-medium",
                            )
                        with v3.VCol(cols=4, classes="text-right"):
                            v3.VBtn(
                                "Clear",
                                click=self.ctrl.clear_all_rois,
                                size="x-small",
                                variant="text",
                                color="error",
                                density="compact",
                            )

                    with v3.VDivider(classes="mb-1"):
                        pass

                    # Each ROI as a small card
                    _roi_bg = (
                        "dark_mode"
                        " ? 'background: rgba(255,255,255,0.04)'"
                        " : 'background: rgba(0,0,0,0.03)'"
                    )
                    with v3.VSheet(
                        v_for="roi in roi_list_ui",
                        key="roi.id",
                        classes="pa-2 mb-1",
                        rounded=True,
                        style=(_roi_bg,),
                    ):
                        with v3.VRow(dense=True, align="center", no_gutters=True):
                            with v3.VCol(cols="auto", classes="mr-2"):
                                v3.VIcon(
                                    "mdi-circle",
                                    size="12",
                                    color=("roi.color",),
                                )
                            with v3.VCol():
                                v3.VLabel(
                                    text=("roi.name",),
                                    classes="text-body-2 font-weight-medium",
                                )
                            with v3.VCol(cols="auto", classes="mx-1"):
                                v3.VChip(
                                    text=("`${roi.pixel_count} px`",),
                                    size="x-small",
                                    variant="tonal",
                                    label=True,
                                )
                            with v3.VCol(cols="auto"):
                                v3.VSwitch(
                                    model_value=("roi.visible",),
                                    density="compact",
                                    hide_details=True,
                                    color="primary",
                                    change=(
                                        self.ctrl.set_roi_visible,
                                        "[roi.id, !roi.visible]",
                                    ),
                                )
                            with v3.VCol(cols="auto"):
                                v3.VBtn(
                                    icon="mdi-close",
                                    size="x-small",
                                    variant="text",
                                    color="error",
                                    density="compact",
                                    click=(self.ctrl.remove_roi, "[roi.id]"),
                                )
                        # Material match line
                        v3.VChip(
                            v_if="roi.top_match",
                            text=("`${roi.top_match} (SAM: ${roi.top_score.toFixed(4)})`",),
                            size="x-small",
                            variant="tonal",
                            color="info",
                            prepend_icon="mdi-molecule",
                            classes="mt-1 ml-5",
                        )

                # Spectral signature plot (in sidebar)
                self._spectral_fig_widget = trame_plotly.Figure(
                    figure=make_empty_figure(
                        "Click a pixel or draw an ROI", dark=True
                    ),
                    display_mode_bar=False,
                    style="width: 100%; height: 250px;",
                )

                v3.VBtn(
                    "Clear Spectra",
                    click=self.ctrl.clear_spectra,
                    size="small",
                    variant="outlined",
                    block=True,
                    prepend_icon="mdi-delete",
                    classes="mt-1",
                )

    def _build_layers_section(self):
        with v3.VCard(variant="outlined", classes="mb-2"):
            with v3.VCardTitle(classes="text-subtitle-2 pa-2 d-flex align-center"):
                v3.VLabel("Layers")
                v3.VSpacer()
                v3.VBtn(
                    "Clear All",
                    click=self.ctrl.clear_all_layers,
                    size="x-small",
                    variant="text",
                    color="error",
                    prepend_icon="mdi-delete-sweep",
                )

            with v3.VCardText(classes="pa-2 pt-0"):
                # Empty state
                v3.VAlert(
                    v_if="layer_list_ui.length === 0",
                    text="No layers. Use the View tab to add composites or indices.",
                    type="info",
                    variant="tonal",
                    density="compact",
                )

                # Layer list
                _layer_border = (
                    "dark_mode"
                    " ? 'border-bottom: 1px solid rgba(255,255,255,0.05)'"
                    " : 'border-bottom: 1px solid rgba(0,0,0,0.08)'"
                )
                with v3.VList(density="compact", v_if="layer_list_ui.length > 0"):
                    with v3.VListItem(
                        v_for="(layer, idx) in layer_list_ui",
                        key="layer.id",
                        classes="px-0 py-1",
                        style=(_layer_border,),
                    ):
                        # Row 1: visibility + name + type icon + delete
                        with v3.VRow(dense=True, align="center", no_gutters=True):
                            with v3.VCol(cols=2):
                                v3.VSwitch(
                                    model_value=("layer.visible",),
                                    density="compact",
                                    hide_details=True,
                                    color="primary",
                                    change=(
                                        self.ctrl.set_layer_visible,
                                        "[layer.id, !layer.visible]",
                                    ),
                                )
                            with v3.VCol(cols=7):
                                v3.VChip(
                                    text=("layer.name",),
                                    size="x-small",
                                    variant="tonal",
                                    label=True,
                                    prepend_icon=(
                                        "layer.layer_type === 'rgb' ? 'mdi-image' : "
                                        "layer.layer_type === 'index' ? 'mdi-chart-box' : "
                                        "'mdi-gradient-horizontal'",
                                    ),
                                )
                            with v3.VCol(cols=3, classes="text-right"):
                                v3.VBtn(
                                    icon="mdi-arrow-up",
                                    size="x-small",
                                    variant="text",
                                    density="compact",
                                    click=(self.ctrl.move_layer_up, "[layer.id]"),
                                )
                                v3.VBtn(
                                    icon="mdi-arrow-down",
                                    size="x-small",
                                    variant="text",
                                    density="compact",
                                    click=(self.ctrl.move_layer_down, "[layer.id]"),
                                )
                                v3.VBtn(
                                    icon="mdi-close",
                                    size="x-small",
                                    variant="text",
                                    color="error",
                                    density="compact",
                                    click=(self.ctrl.remove_layer, "[layer.id]"),
                                )

                        # Row 2: opacity slider
                        with v3.VRow(dense=True, no_gutters=True, classes="mt-0"):
                            with v3.VCol(cols=12):
                                v3.VSlider(
                                    model_value=("layer.opacity",),
                                    min=0,
                                    max=1,
                                    step=0.05,
                                    density="compact",
                                    hide_details=True,
                                    color="white",
                                    thumb_size=12,
                                    track_size=2,
                                    end=(
                                        self.ctrl.set_layer_opacity,
                                        "[layer.id, $event]",
                                    ),
                                    prepend_icon="mdi-opacity",
                                )


class _StateLogHandler(logging.Handler):
    """Routes log messages to trame state for live display."""

    # Stage markers in the legacy script logs → (label, progress %)
    _STAGE_MARKERS = [
        ("ATMOSPHERIC PARAMETER RETRIEVAL", "Retrieving atmospheric parameters", 10),
        ("WATER VAPOR RETRIEVAL", "Retrieving water vapor", 15),
        ("AOD RETRIEVAL", "Retrieving aerosol optical depth", 20),
        ("LOOK-UP TABLE GENERATION", "Generating look-up tables", 30),
        ("APPLYING EMPIRICAL ATMOSPHERIC CORRECTION", "Applying atmospheric correction", 40),
        ("ADJACENCY EFFECT CORRECTION", "Correcting adjacency effects", 70),
        ("PROCESSING COMPLETE", "Complete!", 100),
    ]

    _TILE_RE = re.compile(r"Processed\s+(\d+)/(\d+)\s+tiles")

    def __init__(self, state):
        super().__init__()
        self._state = state
        self.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"
        ))

    def emit(self, record):
        msg = self.format(record)
        raw = record.getMessage()
        try:
            with self._state:
                self._state.l1_log += msg + "\n"

                for marker, label, pct in self._STAGE_MARKERS:
                    if marker in raw:
                        self._state.l1_progress = pct
                        self._state.l1_progress_label = label
                        break

                m = self._TILE_RE.search(raw)
                if m:
                    done, total = int(m.group(1)), int(m.group(2))
                    tile_pct = 40 + int(30 * done / max(total, 1))
                    self._state.l1_progress = tile_pct
                    self._state.l1_progress_label = f"Processing tiles ({done}/{total})"
        except Exception:
            pass
