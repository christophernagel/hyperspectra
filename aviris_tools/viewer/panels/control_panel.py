"""
Hyperspectral Viewer Control Panel

Structure:
- Scrollable top section: Visualization, Spectral Indices, ROI Analysis, Advanced
- Persistent bottom section: Colorbar display + Spectral plot (always visible)

Workflow: Explore → Target → Map → Extract → Export
"""

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QFrame,
    QScrollArea, QSizePolicy, QFileDialog, QMessageBox,
    QSlider, QSpinBox, QListWidget, QListWidgetItem, QTextEdit
)
from qtpy.QtCore import Qt

from ..widgets import CollapsibleSection, SpectralPlot
from ..constants import (
    INDEX_DEFINITIONS, INDEX_METADATA, COLORBAR_GRADIENTS,
    COMPOSITE_PRESETS, SPECTRAL_FEATURES, REFERENCE_SIGNATURES,
    ATMOSPHERIC_BANDS
)


class HyperspectralControlPanel(QWidget):
    """
    Control panel with persistent feedback section at bottom.

    Layout:
    - Top (scrollable): Controls for visualization, indices, ROI
    - Bottom (fixed): Colorbar + Spectral plot (always visible)
    """

    def __init__(self, viewer_app):
        super().__init__()
        self.viewer_app = viewer_app
        self.setMinimumWidth(300)
        self.setMaximumWidth(380)
        self.current_suggested_index = None
        self._build_ui()

        # Connect to layer events for auto-refresh
        if hasattr(self.viewer_app, 'viewer'):
            try:
                self.viewer_app.viewer.layers.events.inserted.connect(self._refresh_layers)
                self.viewer_app.viewer.layers.events.removed.connect(self._refresh_layers)
            except Exception:
                pass  # Ignore if events not available

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # =====================================================================
        # SCROLLABLE TOP SECTION
        # =====================================================================
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

        # Build sections
        self._build_visualization_section(scroll_layout)
        self._build_indices_section(scroll_layout)
        self._build_roi_section(scroll_layout)
        self._build_validation_section(scroll_layout)
        self._build_advanced_section(scroll_layout)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll, stretch=1)

        # =====================================================================
        # PERSISTENT BOTTOM SECTION (Colorbar + Spectral Plot)
        # =====================================================================
        self._build_persistent_section(main_layout)

    # =========================================================================
    # SECTION BUILDERS
    # =========================================================================

    def _build_visualization_section(self, parent_layout):
        """Build Visualization section with presets and Smart RGB Builder."""
        viz_section = CollapsibleSection("Visualization")

        # Preset buttons in grid
        preset_label = QLabel("Quick Presets:")
        preset_label.setStyleSheet("QLabel { color: #aaa; font-weight: bold; }")
        viz_section.add_widget(preset_label)

        preset_grid = QGridLayout()
        preset_grid.setSpacing(4)
        presets = list(COMPOSITE_PRESETS.keys())
        for i, name in enumerate(presets):
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, n=name: self._apply_preset(n))
            btn.setMinimumHeight(26)
            btn.setStyleSheet("QPushButton { font-size: 10px; }")
            preset_grid.addWidget(btn, i // 2, i % 2)
        viz_section.add_layout(preset_grid)

        # Smart RGB Builder
        rgb_label = QLabel("Smart RGB Builder:")
        rgb_label.setStyleSheet("QLabel { color: #aaa; font-weight: bold; margin-top: 10px; }")
        viz_section.add_widget(rgb_label)

        rgb_info = QLabel("Select spectral features for each band:")
        rgb_info.setStyleSheet("QLabel { color: #666; font-size: 9px; }")
        viz_section.add_widget(rgb_info)

        # Build feature dropdowns for R, G, B
        self.rgb_combos = {}
        self.rgb_info_labels = {}

        for band, color in [('R', '#ff6666'), ('G', '#66ff66'), ('B', '#6666ff')]:
            row = QHBoxLayout()
            label = QLabel(f"{band}:")
            label.setStyleSheet(f"QLabel {{ color: {color}; font-weight: bold; }}")
            label.setFixedWidth(20)
            row.addWidget(label)

            combo = QComboBox()
            combo.setStyleSheet("QComboBox { font-size: 10px; }")
            combo.addItem("Select feature...")

            # Group features by category
            categories = {}
            for name, (wl, cat, desc) in SPECTRAL_FEATURES.items():
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append((name, wl))

            for cat in ['visible', 'vegetation', 'iron', 'clay', 'carbonate', 'mgoh', 'hydrocarbon', 'water']:
                if cat in categories:
                    combo.addItem(f"── {cat.upper()} ──")
                    idx = combo.count() - 1
                    combo.model().item(idx).setEnabled(False)
                    for name, wl in sorted(categories[cat], key=lambda x: x[1]):
                        combo.addItem(f"{name} ({wl}nm)")

            combo.addItem("── CUSTOM ──")
            combo.model().item(combo.count() - 1).setEnabled(False)
            combo.addItem("Custom wavelength...")

            combo.currentTextChanged.connect(lambda t, b=band: self._on_rgb_feature_changed(b, t))
            self.rgb_combos[band] = combo
            row.addWidget(combo, stretch=1)

            # Manual input
            manual = QLineEdit()
            manual.setPlaceholderText("nm")
            manual.setFixedWidth(50)
            manual.setStyleSheet("QLineEdit { font-size: 10px; }")
            self.rgb_combos[f'{band}_manual'] = manual
            row.addWidget(manual)

            viz_section.add_layout(row)

            # Info label for this band
            info_label = QLabel("")
            info_label.setStyleSheet("QLabel { color: #555; font-size: 9px; margin-left: 25px; }")
            info_label.setWordWrap(True)
            self.rgb_info_labels[band] = info_label
            viz_section.add_widget(info_label)

        # Apply button
        apply_btn = QPushButton("Apply Smart RGB")
        apply_btn.clicked.connect(self._apply_smart_rgb)
        apply_btn.setMinimumHeight(28)
        apply_btn.setStyleSheet("QPushButton { background-color: #2a4a3a; font-weight: bold; }")
        viz_section.add_widget(apply_btn)

        parent_layout.addWidget(viz_section)

    def _build_indices_section(self, parent_layout):
        """Build Spectral Indices section."""
        index_section = CollapsibleSection("Spectral Indices")

        # Index dropdown organized by category
        self.index_combo = QComboBox()
        self.index_combo.setStyleSheet("QComboBox { font-size: 11px; }")

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
                self.index_combo.addItem(display_name)
                idx = self.index_combo.count() - 1
                self.index_combo.model().item(idx).setEnabled(False)
            else:
                self.index_combo.addItem(display_name)

        self.index_combo.setCurrentIndex(1)
        self.index_combo.currentTextChanged.connect(self._update_index_info)
        index_section.add_widget(self.index_combo)

        # Index info label
        self.index_info = QLabel("")
        self.index_info.setStyleSheet("QLabel { font-size: 10px; color: #888; }")
        self.index_info.setWordWrap(True)
        index_section.add_widget(self.index_info)

        # Calculate button
        calc_btn = QPushButton("Calculate Index")
        calc_btn.setMinimumHeight(28)
        calc_btn.clicked.connect(self._calculate_index)
        calc_btn.setStyleSheet("QPushButton { background-color: #4a3728; font-weight: bold; }")
        index_section.add_widget(calc_btn)

        # Custom Index Calculator
        custom_label = QLabel("Custom Index:")
        custom_label.setStyleSheet("QLabel { color: #aaa; font-weight: bold; margin-top: 10px; }")
        index_section.add_widget(custom_label)

        custom_info = QLabel("Enter two wavelengths (nm) for ratio or normalized difference:")
        custom_info.setStyleSheet("QLabel { color: #666; font-size: 9px; }")
        custom_info.setWordWrap(True)
        index_section.add_widget(custom_info)

        # Wavelength inputs
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel("λ1:"))
        self.custom_wl1 = QSpinBox()
        self.custom_wl1.setRange(350, 2600)
        self.custom_wl1.setValue(860)
        self.custom_wl1.setSuffix(" nm")
        self.custom_wl1.setStyleSheet("QSpinBox { font-size: 10px; }")
        wl_layout.addWidget(self.custom_wl1)

        wl_layout.addWidget(QLabel("λ2:"))
        self.custom_wl2 = QSpinBox()
        self.custom_wl2.setRange(350, 2600)
        self.custom_wl2.setValue(650)
        self.custom_wl2.setSuffix(" nm")
        self.custom_wl2.setStyleSheet("QSpinBox { font-size: 10px; }")
        wl_layout.addWidget(self.custom_wl2)
        index_section.add_layout(wl_layout)

        # Index type and calculate
        type_layout = QHBoxLayout()
        self.custom_type = QComboBox()
        self.custom_type.addItems(["Ratio (λ1/λ2)", "Norm Diff (λ1-λ2)/(λ1+λ2)"])
        self.custom_type.setStyleSheet("QComboBox { font-size: 10px; }")
        type_layout.addWidget(self.custom_type)

        custom_calc_btn = QPushButton("Calculate")
        custom_calc_btn.clicked.connect(self._calculate_custom_index)
        custom_calc_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        type_layout.addWidget(custom_calc_btn)
        index_section.add_layout(type_layout)

        parent_layout.addWidget(index_section)
        self._update_index_info(self.index_combo.currentText())

    def _build_roi_section(self, parent_layout):
        """Build ROI Analysis section with reference overlay."""
        roi_section = CollapsibleSection("ROI Analysis")

        # ROI creation tools
        roi_label = QLabel("Create ROI:")
        roi_label.setStyleSheet("QLabel { color: #aaa; font-weight: bold; }")
        roi_section.add_widget(roi_label)

        roi_btn_layout = QHBoxLayout()
        for mode, label in [('rectangle', 'Rect'), ('polygon', 'Poly'), ('ellipse', 'Ellipse')]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, m=mode: self._set_roi_mode(m))
            btn.setStyleSheet("QPushButton { font-size: 10px; }")
            roi_btn_layout.addWidget(btn)
        roi_section.add_layout(roi_btn_layout)

        # Extract and clear buttons
        action_layout = QHBoxLayout()
        extract_btn = QPushButton("Extract Spectra")
        extract_btn.clicked.connect(self._extract_roi_spectra)
        extract_btn.setMinimumHeight(28)
        extract_btn.setStyleSheet("QPushButton { background-color: #28445a; font-weight: bold; }")
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_rois)
        action_layout.addWidget(extract_btn)
        action_layout.addWidget(clear_btn)
        roi_section.add_layout(action_layout)

        # Reference overlay for peak matching
        ref_label = QLabel("Reference Overlay:")
        ref_label.setStyleSheet("QLabel { color: #aaa; font-weight: bold; margin-top: 8px; }")
        roi_section.add_widget(ref_label)

        ref_layout = QHBoxLayout()
        self.reference_combo = QComboBox()
        self.reference_combo.setStyleSheet("QComboBox { font-size: 10px; }")
        self.reference_combo.addItem("Select reference...")
        for ref_name in REFERENCE_SIGNATURES.keys():
            self.reference_combo.addItem(ref_name)
        ref_layout.addWidget(self.reference_combo, stretch=1)

        overlay_btn = QPushButton("Overlay")
        overlay_btn.clicked.connect(self._overlay_reference)
        overlay_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        ref_layout.addWidget(overlay_btn)
        roi_section.add_layout(ref_layout)

        # Match suggestion
        match_btn = QPushButton("Identify Material (Peak Matching)")
        match_btn.clicked.connect(self._identify_material)
        match_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        roi_section.add_widget(match_btn)

        self.match_result = QLabel("")
        self.match_result.setStyleSheet("QLabel { color: #00cc88; font-size: 10px; }")
        self.match_result.setWordWrap(True)
        roi_section.add_widget(self.match_result)

        # Export button
        export_btn = QPushButton("Export Spectra to CSV")
        export_btn.clicked.connect(self._export_spectra)
        export_btn.setStyleSheet("QPushButton { font-size: 10px; margin-top: 8px; }")
        roi_section.add_widget(export_btn)

        parent_layout.addWidget(roi_section)

    def _build_advanced_section(self, parent_layout):
        """Build Advanced section (collapsed by default)."""
        advanced_section = CollapsibleSection("Advanced", collapsed=True)

        # Layer management
        layer_label = QLabel("Layers:")
        layer_label.setStyleSheet("QLabel { color: #aaa; font-weight: bold; }")
        advanced_section.add_widget(layer_label)

        layer_btn_layout = QHBoxLayout()
        clear_comp_btn = QPushButton("Clear Composites")
        clear_comp_btn.clicked.connect(self._clear_composites)
        clear_comp_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        toggle_cube_btn = QPushButton("Toggle Cube")
        toggle_cube_btn.clicked.connect(self._toggle_cube)
        toggle_cube_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        layer_btn_layout.addWidget(clear_comp_btn)
        layer_btn_layout.addWidget(toggle_cube_btn)
        advanced_section.add_layout(layer_btn_layout)

        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(80)
        self.layer_list.setStyleSheet("QListWidget { font-size: 10px; }")
        self.layer_list.itemDoubleClicked.connect(self._toggle_layer)
        self.layer_list.setToolTip("Double-click to toggle visibility")
        advanced_section.add_widget(self.layer_list)

        layer_action_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_layers)
        refresh_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        toggle_btn = QPushButton("Toggle Selected")
        toggle_btn.clicked.connect(self._toggle_layer)
        toggle_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        layer_action_layout.addWidget(refresh_btn)
        layer_action_layout.addWidget(toggle_btn)
        advanced_section.add_layout(layer_action_layout)

        # 3D Camera
        camera_label = QLabel("3D Camera:")
        camera_label.setStyleSheet("QLabel { color: #aaa; font-weight: bold; margin-top: 8px; }")
        advanced_section.add_widget(camera_label)

        # Camera preset buttons
        cam_preset_layout = QHBoxLayout()
        for name, angles in [('Top', (0, 0)), ('Front', (0, 90)), ('Side', (90, 45)), ('3D', (45, 45))]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, a=angles: self._set_camera_preset(a))
            btn.setStyleSheet("QPushButton { font-size: 9px; }")
            cam_preset_layout.addWidget(btn)
        advanced_section.add_layout(cam_preset_layout)

        # Sliders
        for label, attr, range_vals in [('Elevation', 'elev', (0, 90, 30)), ('Azimuth', 'azim', (0, 360, 45))]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{label}:"))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(range_vals[0], range_vals[1])
            slider.setValue(range_vals[2])
            slider.valueChanged.connect(self._update_camera)
            setattr(self, f'{attr}_slider', slider)
            row.addWidget(slider)
            advanced_section.add_layout(row)

        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_camera)
        reset_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        advanced_section.add_widget(reset_btn)

        parent_layout.addWidget(advanced_section)

    def _build_validation_section(self, parent_layout):
        """Build Validation section with diagnostic tools."""
        val_section = CollapsibleSection("Validation", collapsed=True)

        # Quick checks row
        quick_label = QLabel("Quick Checks:")
        quick_label.setStyleSheet("QLabel { color: #aaa; font-weight: bold; }")
        val_section.add_widget(quick_label)

        row1 = QHBoxLayout()
        data_qual_btn = QPushButton("Data Quality")
        data_qual_btn.clicked.connect(self._check_data_quality)
        data_qual_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        data_qual_btn.setToolTip("Check NaN%, negative values, data range")
        row1.addWidget(data_qual_btn)

        snr_btn = QPushButton("SNR")
        snr_btn.clicked.connect(self._check_snr)
        snr_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        snr_btn.setToolTip("Estimate signal-to-noise at key wavelengths")
        row1.addWidget(snr_btn)

        atm_btn = QPushButton("Atm Bands")
        atm_btn.clicked.connect(self._check_atmospheric)
        atm_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        atm_btn.setToolTip("Check bands in atmospheric absorption")
        row1.addWidget(atm_btn)
        val_section.add_layout(row1)

        layer_stats_btn = QPushButton("Layer Stats")
        layer_stats_btn.clicked.connect(self._check_layer_stats)
        layer_stats_btn.setStyleSheet("QPushButton { font-size: 10px; }")
        layer_stats_btn.setToolTip("Statistics on all index layers")
        val_section.add_widget(layer_stats_btn)

        # Full report button
        full_btn = QPushButton("Generate Full Validation Report")
        full_btn.clicked.connect(self._full_validation_report)
        full_btn.setMinimumHeight(28)
        full_btn.setStyleSheet("QPushButton { background-color: #3a3a5a; font-weight: bold; font-size: 10px; }")
        val_section.add_widget(full_btn)

        # Results display
        self.validation_output = QTextEdit()
        self.validation_output.setReadOnly(True)
        self.validation_output.setMaximumHeight(150)
        self.validation_output.setStyleSheet("""
            QTextEdit {
                font-family: monospace;
                font-size: 9px;
                background-color: #1a1a1a;
                color: #00ff88;
                border: 1px solid #333;
            }
        """)
        self.validation_output.setPlaceholderText("Validation results will appear here...")
        val_section.add_widget(self.validation_output)

        # Copy to clipboard button
        copy_btn = QPushButton("Copy Results")
        copy_btn.clicked.connect(self._copy_validation_results)
        copy_btn.setStyleSheet("QPushButton { font-size: 9px; }")
        val_section.add_widget(copy_btn)

        parent_layout.addWidget(val_section)

    def _build_persistent_section(self, parent_layout):
        """Build persistent bottom section with colorbar and spectral plot."""
        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("QFrame { color: #444; }")
        parent_layout.addWidget(separator)

        # Colorbar
        colorbar_widget = QWidget()
        colorbar_layout = QVBoxLayout(colorbar_widget)
        colorbar_layout.setContentsMargins(4, 4, 4, 4)
        colorbar_layout.setSpacing(2)

        self.colorbar_title = QLabel("Ready")
        self.colorbar_title.setStyleSheet("QLabel { font-weight: bold; color: white; font-size: 11px; }")
        colorbar_layout.addWidget(self.colorbar_title)

        bar_layout = QHBoxLayout()
        self.colorbar_min_label = QLabel("0.0")
        self.colorbar_min_label.setStyleSheet("QLabel { color: #aaa; font-size: 9px; }")
        self.colorbar_gradient = QLabel()
        self.colorbar_gradient.setFixedHeight(16)
        self.colorbar_gradient.setStyleSheet("""
            QLabel { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #440154, stop:0.25 #3b528b, stop:0.5 #21918c,
                stop:0.75 #5ec962, stop:1 #fde725);
                border: 1px solid #555; border-radius: 2px; }
        """)
        self.colorbar_max_label = QLabel("1.0")
        self.colorbar_max_label.setStyleSheet("QLabel { color: #aaa; font-size: 9px; }")
        bar_layout.addWidget(self.colorbar_min_label)
        bar_layout.addWidget(self.colorbar_gradient, stretch=1)
        bar_layout.addWidget(self.colorbar_max_label)
        colorbar_layout.addLayout(bar_layout)

        semantic_layout = QHBoxLayout()
        self.colorbar_low_semantic = QLabel("")
        self.colorbar_low_semantic.setStyleSheet("QLabel { color: #666; font-size: 9px; }")
        self.colorbar_high_semantic = QLabel("")
        self.colorbar_high_semantic.setStyleSheet("QLabel { color: #666; font-size: 9px; }")
        self.colorbar_high_semantic.setAlignment(Qt.AlignRight)
        semantic_layout.addWidget(self.colorbar_low_semantic)
        semantic_layout.addWidget(self.colorbar_high_semantic)
        colorbar_layout.addLayout(semantic_layout)

        self.colorbar_desc = QLabel("")
        self.colorbar_desc.setStyleSheet("QLabel { color: #555; font-size: 9px; }")
        self.colorbar_desc.setWordWrap(True)
        colorbar_layout.addWidget(self.colorbar_desc)

        parent_layout.addWidget(colorbar_widget)

        # Spectral plot
        self.spectral_plot = SpectralPlot()
        self.spectral_plot.setMinimumHeight(160)
        self.spectral_plot.setMaximumHeight(200)
        parent_layout.addWidget(self.spectral_plot)

    # =========================================================================
    # VISUALIZATION CALLBACKS
    # =========================================================================

    def _apply_preset(self, preset_name):
        preset = COMPOSITE_PRESETS.get(preset_name)
        if preset and hasattr(self.viewer_app, 'add_composite'):
            result = self.viewer_app.add_composite(
                preset_name, preset['r'], preset['g'], preset['b']
            )
            if result:
                self._update_colorbar_composite(result)

    def _on_rgb_feature_changed(self, band, text):
        """Update info label when RGB feature selection changes."""
        if text.startswith("──") or text == "Select feature..." or text == "Custom wavelength...":
            self.rgb_info_labels[band].setText("")
            return

        # Extract wavelength from text like "Clay Al-OH (2200nm)"
        import re
        match = re.search(r'\((\d+)nm\)', text)
        if match:
            wl = int(match.group(1))
            # Find the feature description
            for name, (feat_wl, cat, desc) in SPECTRAL_FEATURES.items():
                if feat_wl == wl and name in text:
                    self.rgb_info_labels[band].setText(desc)
                    self.rgb_combos[f'{band}_manual'].setText(str(wl))
                    break

    def _apply_smart_rgb(self):
        """Apply the Smart RGB composite."""
        wavelengths = {}
        for band in ['R', 'G', 'B']:
            # Try manual input first
            manual_text = self.rgb_combos[f'{band}_manual'].text().strip()
            if manual_text:
                try:
                    wavelengths[band] = float(manual_text)
                except ValueError:
                    QMessageBox.warning(self, "Error", f"Invalid wavelength for {band} band")
                    return
            else:
                QMessageBox.warning(self, "Missing", f"Select or enter wavelength for {band} band")
                return

        # Validate wavelengths
        if hasattr(self.viewer_app, 'data_loader'):
            loader = self.viewer_app.data_loader
            for band, wl in wavelengths.items():
                if not loader.validate_wavelength(wl):
                    QMessageBox.warning(self, "Invalid",
                        f"{band}={wl}nm outside sensor range ({loader.wl_min:.0f}-{loader.wl_max:.0f}nm)")
                    return

        name = f"Smart ({int(wavelengths['R'])}/{int(wavelengths['G'])}/{int(wavelengths['B'])})"
        if hasattr(self.viewer_app, 'add_composite'):
            result = self.viewer_app.add_composite(
                name, wavelengths['R'], wavelengths['G'], wavelengths['B']
            )
            if result:
                self._update_colorbar_composite(result)

    def _update_colorbar_composite(self, info):
        self.colorbar_title.setText(f"RGB: {info['name']}")
        self.colorbar_min_label.setText(f"R:{info['r_wl']:.0f}")
        self.colorbar_max_label.setText(f"B:{info['b_wl']:.0f}")
        self.colorbar_gradient.setStyleSheet("""
            QLabel { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #ff0000, stop:0.5 #00ff00, stop:1 #0000ff);
                border: 1px solid #555; }
        """)
        self.colorbar_low_semantic.setText(f"R:{info['r_wl']:.0f}nm")
        self.colorbar_high_semantic.setText(f"B:{info['b_wl']:.0f}nm")
        self.colorbar_desc.setText(f"G:{info['g_wl']:.0f}nm")

    # =========================================================================
    # INDEX CALLBACKS
    # =========================================================================

    def _update_index_info(self, index_name):
        key = index_name.strip()
        meta = INDEX_METADATA.get(key, {})
        self.index_info.setText(meta.get('desc', ''))

    def _calculate_index(self):
        index_name = self.index_combo.currentText().strip()
        if index_name.startswith("──"):
            return

        if hasattr(self.viewer_app, 'calculate_index'):
            result = self.viewer_app.calculate_index(index_name, estimate_uncertainty=True)
            if result:
                self._update_colorbar_index(result)

    def _calculate_custom_index(self):
        """Calculate custom ratio or normalized difference index."""
        wl1 = self.custom_wl1.value()
        wl2 = self.custom_wl2.value()
        idx_type = 'ratio' if self.custom_type.currentIndex() == 0 else 'nd'

        if hasattr(self.viewer_app, 'calculate_custom_index'):
            name = f"Custom {idx_type.upper()} ({wl1}/{wl2})"
            result = self.viewer_app.calculate_custom_index(
                name=name,
                b1_wl=wl1,
                b2_wl=wl2,
                index_type=idx_type,
                cmap='viridis'
            )
            if result:
                self._update_colorbar_custom(result)

    def _update_colorbar_custom(self, info):
        """Update colorbar for custom index."""
        self.colorbar.set_colormap(info['cmap'])
        self.colorbar.set_range(info['clim'][0], info['clim'][1])
        idx_type = info['type'].upper()
        self.colorbar.set_labels(f"Low ({idx_type})", f"High ({idx_type})")
        self.colorbar.set_title(info['name'])

    def _update_colorbar_index(self, info):
        name = info['name']
        cmap = info['cmap']
        clim = info['clim']

        meta = INDEX_METADATA.get(name, {'low': 'Low', 'high': 'High', 'desc': ''})

        self.colorbar_title.setText(f"{name} Index")
        self.colorbar_min_label.setText(f"{clim[0]:.2f}")
        self.colorbar_max_label.setText(f"{clim[1]:.2f}")
        self.colorbar_low_semantic.setText(meta['low'])
        self.colorbar_high_semantic.setText(meta['high'])

        desc_text = meta['desc']
        uncertainty = info.get('uncertainty')
        if uncertainty:
            uncert_pct = uncertainty['median_relative_uncertainty'] * 100
            if uncert_pct > 50:
                desc_text += f" | ⚠ High uncertainty: ±{uncert_pct:.0f}%"
            elif uncert_pct > 20:
                desc_text += f" | ⚠ Moderate: ±{uncert_pct:.0f}%"
        self.colorbar_desc.setText(desc_text)

        self._set_colorbar_gradient(cmap)

    def _set_colorbar_gradient(self, cmap):
        gradient = COLORBAR_GRADIENTS.get(cmap, COLORBAR_GRADIENTS['viridis'])
        self.colorbar_gradient.setStyleSheet(f"""
            QLabel {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {gradient});
                     border: 1px solid #555; border-radius: 2px; }}
        """)

    # =========================================================================
    # ROI CALLBACKS
    # =========================================================================

    def _set_roi_mode(self, mode):
        if hasattr(self.viewer_app, 'set_roi_mode'):
            self.viewer_app.set_roi_mode(mode)

    def _clear_rois(self):
        if hasattr(self.viewer_app, 'clear_rois'):
            self.viewer_app.clear_rois()
        self.spectral_plot.clear_plot()
        self.match_result.setText("")

    def _extract_roi_spectra(self):
        if hasattr(self.viewer_app, 'extract_roi_spectra'):
            results = self.viewer_app.extract_roi_spectra()
            if results:
                colors = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'pink', 'red', 'green']
                self.spectral_plot.clear_plot()
                for i, result in enumerate(results):
                    self.spectral_plot.plot_spectrum(
                        result['wavelengths'],
                        result['mean'],
                        result.get('std'),
                        label=result.get('label', f'ROI {i+1}'),
                        color=colors[i % len(colors)],
                        clear=False
                    )

    def _overlay_reference(self):
        """Overlay reference absorption bands on spectral plot."""
        ref_name = self.reference_combo.currentText()
        if ref_name == "Select reference..." or ref_name not in REFERENCE_SIGNATURES:
            return

        self.spectral_plot.clear_reference_lines()
        signature = REFERENCE_SIGNATURES[ref_name]
        for wl, depth, feature in signature:
            self.spectral_plot.add_reference_line(wl, f"{feature}", color='yellow')
        self.spectral_plot.draw()

    def _identify_material(self):
        """Identify material by matching ROI spectrum to reference signatures."""
        if not self.spectral_plot.has_spectral_data():
            self.match_result.setText("Extract ROI spectra first")
            return

        # Get the first ROI's spectrum
        data = self.spectral_plot._plotted_data[0]
        wavelengths = np.array(data['wavelengths'])
        spectrum = np.array(data['mean'])

        # Normalize spectrum
        spectrum_norm = (spectrum - np.nanmin(spectrum)) / (np.nanmax(spectrum) - np.nanmin(spectrum) + 1e-10)

        # Score each reference
        scores = {}
        for ref_name, signature in REFERENCE_SIGNATURES.items():
            score = 0
            matched = 0
            for wl, expected_depth, feature in signature:
                # Find nearest wavelength
                idx = np.argmin(np.abs(wavelengths - wl))
                if abs(wavelengths[idx] - wl) < 20:  # Within 20nm
                    # Check for absorption (dip) at this wavelength
                    # Compare to neighbors
                    start = max(0, idx - 5)
                    end = min(len(spectrum_norm), idx + 6)
                    local_mean = np.nanmean(spectrum_norm[start:end])
                    local_val = spectrum_norm[idx]

                    # If there's a dip (absorption), score higher
                    if local_val < local_mean:
                        absorption_depth = local_mean - local_val
                        score += absorption_depth * expected_depth
                        matched += 1

            if matched > 0:
                scores[ref_name] = score / len(signature)  # Normalize by signature length

        if not scores:
            self.match_result.setText("No clear matches found")
            return

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_matches = sorted_scores[:3]

        result_text = "Best matches:\n"
        for i, (name, score) in enumerate(top_matches):
            confidence = min(100, score * 200)  # Scale to percentage
            result_text += f"  {i+1}. {name} ({confidence:.0f}%)\n"

        self.match_result.setText(result_text)
        self.match_result.setStyleSheet("QLabel { color: #00cc88; font-size: 10px; }")

    def _export_spectra(self):
        if not self.spectral_plot.has_spectral_data():
            QMessageBox.information(self, "No Data", "Extract spectra first")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Save", "", "CSV (*.csv)")
        if filepath:
            self.spectral_plot.export_to_csv(filepath)

    # =========================================================================
    # ADVANCED CALLBACKS
    # =========================================================================

    def _clear_composites(self):
        if hasattr(self.viewer_app, 'viewer'):
            to_remove = []
            for layer in self.viewer_app.viewer.layers:
                if layer.name not in ['Reflectance Cube', 'Radiance Cube', 'ROIs']:
                    to_remove.append(layer)
            for layer in to_remove:
                self.viewer_app.viewer.layers.remove(layer)

    def _toggle_cube(self):
        if hasattr(self.viewer_app, 'cube_layer'):
            self.viewer_app.cube_layer.visible = not self.viewer_app.cube_layer.visible

    def _refresh_layers(self, event=None):
        """Refresh the layer list. Can be called directly or from napari events."""
        self.layer_list.clear()
        if hasattr(self.viewer_app, 'viewer'):
            for layer in self.viewer_app.viewer.layers:
                item = QListWidgetItem(layer.name)
                item.setCheckState(Qt.Checked if layer.visible else Qt.Unchecked)
                self.layer_list.addItem(item)

    def _toggle_layer(self, item=None):
        """Toggle layer visibility. Can be called from button or double-click."""
        current = item if item else self.layer_list.currentItem()
        if current and hasattr(self.viewer_app, 'viewer'):
            layer_name = current.text()
            for layer in self.viewer_app.viewer.layers:
                if layer.name == layer_name:
                    layer.visible = not layer.visible
                    current.setCheckState(Qt.Checked if layer.visible else Qt.Unchecked)
                    break

    def _set_camera_preset(self, angles):
        if hasattr(self.viewer_app, 'viewer'):
            az, el = angles
            self.azim_slider.setValue(az)
            self.elev_slider.setValue(el)
            try:
                self.viewer_app.viewer.dims.ndisplay = 3
                self.viewer_app.viewer.camera.angles = (el, az, 0)
            except Exception:
                pass

    def _update_camera(self):
        if hasattr(self.viewer_app, 'viewer'):
            elev = self.elev_slider.value()
            azim = self.azim_slider.value()
            try:
                self.viewer_app.viewer.camera.angles = (elev, azim, 0)
            except Exception:
                pass

    def _reset_camera(self):
        if hasattr(self.viewer_app, 'viewer'):
            self.viewer_app.viewer.reset_view()
            self.azim_slider.setValue(45)
            self.elev_slider.setValue(30)

    # =========================================================================
    # VALIDATION CALLBACKS
    # =========================================================================

    def _check_data_quality(self):
        """Check data quality: NaN%, negative values, range."""
        if not hasattr(self.viewer_app, 'data_loader'):
            self.validation_output.setText("No data loaded")
            return

        loader = self.viewer_app.data_loader
        output = ["=== DATA QUALITY CHECK ===\n"]
        output.append(f"Dataset: {loader.n_bands} bands × {loader.n_rows} × {loader.n_cols} px")
        output.append(f"Data type: {loader.data_type}")
        output.append(f"Wavelength range: {loader.wl_min:.0f} - {loader.wl_max:.0f} nm\n")

        # Sample a few bands for quality check
        test_bands = [0, loader.n_bands // 4, loader.n_bands // 2,
                      3 * loader.n_bands // 4, loader.n_bands - 1]

        output.append("Band    λ(nm)    NaN%    Neg%    Min       Max")
        output.append("-" * 55)

        for band_idx in test_bands:
            band = loader.get_band(band_idx)
            wl = loader.wavelengths[band_idx]
            nan_pct = np.isnan(band).sum() / band.size * 100
            neg_pct = np.nansum(band < 0) / band.size * 100

            # Handle all-NaN bands gracefully
            valid_data = band[~np.isnan(band)]
            if len(valid_data) > 0:
                bmin = float(np.min(valid_data))
                bmax = float(np.max(valid_data))
                output.append(f"{band_idx:4d}    {wl:6.0f}   {nan_pct:5.1f}   {neg_pct:5.1f}   {bmin:8.4f}  {bmax:8.4f}")
            else:
                output.append(f"{band_idx:4d}    {wl:6.0f}   {nan_pct:5.1f}   {neg_pct:5.1f}   [all NaN]")

        self.validation_output.setText("\n".join(output))

    def _check_snr(self):
        """Estimate SNR at key wavelengths."""
        if not hasattr(self.viewer_app, 'data_loader'):
            self.validation_output.setText("No data loaded")
            return

        loader = self.viewer_app.data_loader
        output = ["=== SNR ESTIMATION ===\n"]
        output.append("Wavelength    SNR      Quality")
        output.append("-" * 35)

        # Key wavelengths across VNIR and SWIR
        test_wavelengths = [550, 650, 860, 1050, 1650, 2000, 2200]

        for wl in test_wavelengths:
            if loader.wl_min <= wl <= loader.wl_max:
                snr = loader.estimate_snr(wl)
                if snr > 200:
                    quality = "Excellent"
                elif snr > 100:
                    quality = "Good"
                elif snr > 50:
                    quality = "Moderate"
                else:
                    quality = "Poor"
                output.append(f"  {wl:4.0f} nm     {snr:5.0f}:1   {quality}")

        output.append("\nTypical AVIRIS-3 SNR:")
        output.append("  VNIR (400-1000nm): 300-500:1")
        output.append("  SWIR (1000-2500nm): 100-300:1")

        self.validation_output.setText("\n".join(output))

    def _check_atmospheric(self):
        """Check bands in atmospheric absorption regions."""
        if not hasattr(self.viewer_app, 'data_loader'):
            self.validation_output.setText("No data loaded")
            return

        loader = self.viewer_app.data_loader
        output = ["=== ATMOSPHERIC BAND CHECK ===\n"]

        # Group bad bands by absorption type
        bad_bands = {'water_vapor_1': [], 'water_vapor_2': [],
                     'carbon_dioxide': [], 'oxygen': []}

        for i, wl in enumerate(loader.wavelengths):
            in_atm, name, severity = loader.is_in_atmospheric_band(wl)
            if in_atm and name in bad_bands:
                bad_bands[name].append((i, wl))

        total_bad = sum(len(v) for v in bad_bands.values())
        output.append(f"Total bands in absorption: {total_bad} / {loader.n_bands}")
        output.append(f"Percentage affected: {total_bad / loader.n_bands * 100:.1f}%\n")

        absorption_info = {
            'water_vapor_1': ('H₂O (1350-1450nm)', 'SEVERE - data unreliable'),
            'water_vapor_2': ('H₂O (1800-1950nm)', 'SEVERE - data unreliable'),
            'carbon_dioxide': ('CO₂ (2500-2600nm)', 'MODERATE - edge effects'),
            'oxygen': ('O₂ A-band (760-770nm)', 'MODERATE - narrow feature'),
        }

        for key, bands in bad_bands.items():
            if bands:
                name, severity = absorption_info[key]
                output.append(f"{name}")
                output.append(f"  Status: {severity}")
                output.append(f"  Bands: {len(bands)} ({bands[0][0]}-{bands[-1][0]})")
                output.append(f"  Range: {bands[0][1]:.0f}-{bands[-1][1]:.0f} nm\n")

        self.validation_output.setText("\n".join(output))

    def _check_layer_stats(self):
        """Get statistics on all index layers."""
        if not hasattr(self.viewer_app, 'viewer'):
            self.validation_output.setText("No viewer available")
            return

        output = ["=== LAYER STATISTICS ===\n"]
        output.append("Layer                    Min      Max      Mean     Std")
        output.append("-" * 60)

        for layer in self.viewer_app.viewer.layers:
            # Skip non-image layers (Shapes, Points, etc.)
            layer_type = type(layer).__name__
            if layer_type in ('Shapes', 'Points', 'Labels', 'Vectors', 'Tracks'):
                name = layer.name[:22].ljust(22)
                output.append(f"{name}  [{layer_type} layer - no image stats]")
                continue

            if hasattr(layer, 'data') and layer.data is not None:
                data = layer.data
                # Ensure data is a numpy array with dimensions
                if not isinstance(data, np.ndarray):
                    continue
                if data.ndim >= 2:
                    name = layer.name[:22].ljust(22)
                    try:
                        valid_data = data[~np.isnan(data)] if np.issubdtype(data.dtype, np.floating) else data.ravel()
                        if len(valid_data) > 0:
                            bmin = float(np.min(valid_data))
                            bmax = float(np.max(valid_data))
                            bmean = float(np.mean(valid_data))
                            bstd = float(np.std(valid_data))
                            output.append(f"{name}  {bmin:8.3f} {bmax:8.3f} {bmean:8.3f} {bstd:8.3f}")
                        else:
                            output.append(f"{name}  [all NaN]")
                    except Exception as e:
                        output.append(f"{name}  [error: {str(e)[:20]}]")

        self.validation_output.setText("\n".join(output))

    def _full_validation_report(self):
        """Generate comprehensive validation report."""
        if not hasattr(self.viewer_app, 'data_loader'):
            self.validation_output.setText("No data loaded")
            return

        loader = self.viewer_app.data_loader
        output = ["=" * 50]
        output.append("      FULL VALIDATION REPORT")
        output.append("=" * 50 + "\n")

        # Basic info
        output.append("DATASET INFO")
        output.append("-" * 30)
        output.append(f"Type: {loader.data_type}")
        output.append(f"Dimensions: {loader.n_bands} × {loader.n_rows} × {loader.n_cols}")
        output.append(f"Wavelengths: {loader.wl_min:.0f} - {loader.wl_max:.0f} nm")
        memory_mb = (loader.n_bands * loader.n_rows * loader.n_cols * 4) / 1e6
        output.append(f"Est. memory: {memory_mb:.1f} MB\n")

        # Data quality summary
        output.append("DATA QUALITY SUMMARY")
        output.append("-" * 30)
        test_bands = [0, loader.n_bands // 2, loader.n_bands - 1]
        total_nan = 0
        total_neg = 0
        for band_idx in test_bands:
            band = loader.get_band(band_idx)
            total_nan += np.isnan(band).sum()
            total_neg += (band < 0).sum()

        pixels_checked = len(test_bands) * loader.n_rows * loader.n_cols
        output.append(f"NaN values: {total_nan / pixels_checked * 100:.2f}%")
        output.append(f"Negative values: {total_neg / pixels_checked * 100:.2f}%\n")

        # SNR summary
        output.append("SNR SUMMARY")
        output.append("-" * 30)
        snr_vnir = loader.estimate_snr(650)
        snr_swir = loader.estimate_snr(2000) if loader.wl_max > 2000 else None
        output.append(f"VNIR (650nm): {snr_vnir:.0f}:1")
        if snr_swir:
            output.append(f"SWIR (2000nm): {snr_swir:.0f}:1")

        if snr_vnir > 200:
            output.append("Quality: EXCELLENT")
        elif snr_vnir > 100:
            output.append("Quality: GOOD")
        else:
            output.append("Quality: MODERATE")
        output.append("")

        # Atmospheric bands
        output.append("ATMOSPHERIC ABSORPTION")
        output.append("-" * 30)
        bad_count = 0
        for wl in loader.wavelengths:
            in_atm, _, _ = loader.is_in_atmospheric_band(wl)
            if in_atm:
                bad_count += 1
        output.append(f"Affected bands: {bad_count} ({bad_count/loader.n_bands*100:.1f}%)")
        output.append("Use caution with indices using:")
        output.append("  • 1350-1450nm (H₂O)")
        output.append("  • 1800-1950nm (H₂O)")
        output.append("")

        # Layers summary
        output.append("ACTIVE LAYERS")
        output.append("-" * 30)
        for layer in self.viewer_app.viewer.layers:
            vis = "✓" if layer.visible else "○"
            output.append(f"  {vis} {layer.name}")

        output.append("\n" + "=" * 50)
        output.append("Report generated successfully")

        self.validation_output.setText("\n".join(output))

    def _copy_validation_results(self):
        """Copy validation results to clipboard."""
        from qtpy.QtWidgets import QApplication
        text = self.validation_output.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            # Brief feedback
            current = self.validation_output.toPlainText()
            self.validation_output.setText("Copied to clipboard!\n\n" + current)

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def update_wavelength_display(self, band_idx, wavelength):
        self.wavelength_label.setText(f"Band: {band_idx} | λ: {wavelength:.1f} nm")

    def set_data_type(self, data_type):
        self.spectral_plot.set_data_type(data_type)

    def set_wavelength_range(self, wl_min, wl_max):
        self.spectral_plot.set_wavelength_range(wl_min, wl_max)
