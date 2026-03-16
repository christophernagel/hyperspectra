"""
3D view sidebar controls for the hyperspectral web viewer.
"""

from trame.widgets import vuetify3 as v3


def build_3d_controls(state, ctrl):
    """Build the 3D view controls card for the sidebar."""

    # --- View Mode Toggle ---
    with v3.VCard(variant="outlined", classes="mb-2"):
        v3.VCardTitle("View Mode", classes="text-subtitle-2 pa-2")
        with v3.VCardText(classes="pa-2 pt-0"):
            with v3.VBtnToggle(
                v_model=("view_mode", "2d"),
                density="compact",
                mandatory=True,
                classes="mb-2",
                color="primary",
            ):
                v3.VBtn(value="2d", text="2D Map", prepend_icon="mdi-image", size="small")
                v3.VBtn(value="3d", text="3D Cube", prepend_icon="mdi-cube-outline", size="small")

    # --- 3D Controls (only visible in 3D mode) ---
    with v3.VCard(
        variant="outlined", classes="mb-2",
        v_show="view_mode === '3d'",
    ):
        v3.VCardTitle("3D Controls", classes="text-subtitle-2 pa-2")
        with v3.VCardText(classes="pa-2 pt-0"):

            # Renderer selector
            v3.VLabel("Renderer", classes="text-caption mb-1")
            with v3.VBtnToggle(
                v_model=("render_3d_mode", "plotly"),
                density="compact",
                mandatory=True,
                classes="mb-3",
                color="secondary",
            ):
                v3.VBtn(value="plotly", text="Plotly", size="small")
                v3.VBtn(value="vtk", text="VTK", size="small")

            v3.VDivider(classes="mb-2")

            # Resolution controls
            v3.VLabel("Resolution", classes="text-caption mb-1")
            with v3.VRow(dense=True, classes="mb-2"):
                with v3.VCol(cols=6):
                    v3.VSelect(
                        v_model=("spatial_downsample", 2),
                        items=([
                            {"title": "Full", "value": 1},
                            {"title": "1/2", "value": 2},
                            {"title": "1/4", "value": 4},
                        ],),
                        label="Spatial",
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                    )
                with v3.VCol(cols=6):
                    v3.VSelect(
                        v_model=("spectral_downsample", 4),
                        items=([
                            {"title": "Full", "value": 1},
                            {"title": "1/2", "value": 2},
                            {"title": "1/4", "value": 4},
                            {"title": "1/8", "value": 8},
                        ],),
                        label="Spectral",
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                    )

            v3.VDivider(classes="mb-2")

            # Slice plane controls
            v3.VLabel("Slice Planes", classes="text-caption mb-1")

            # X slice (column → row–wavelength cross-section)
            with v3.VRow(dense=True, align="center", classes="mb-1"):
                with v3.VCol(cols=2):
                    v3.VCheckbox(
                        v_model=("slice_x_show", True),
                        density="compact",
                        hide_details=True,
                        color="red",
                    )
                with v3.VCol(cols=10):
                    v3.VSlider(
                        v_model=("slice_x_index", 0),
                        min=0,
                        max=("slice_x_max", 100),
                        step=1,
                        label="Col",
                        density="compact",
                        hide_details=True,
                        thumb_label=True,
                        color="red",
                        disabled=("!slice_x_show",),
                    )
                    v3.VLabel(
                        "Row–wavelength section",
                        classes="text-caption text-disabled mt-n2 ml-10",
                    )

            # Y slice (row → col–wavelength cross-section)
            with v3.VRow(dense=True, align="center", classes="mb-1"):
                with v3.VCol(cols=2):
                    v3.VCheckbox(
                        v_model=("slice_y_show", True),
                        density="compact",
                        hide_details=True,
                        color="green",
                    )
                with v3.VCol(cols=10):
                    v3.VSlider(
                        v_model=("slice_y_index", 0),
                        min=0,
                        max=("slice_y_max", 100),
                        step=1,
                        label="Row",
                        density="compact",
                        hide_details=True,
                        thumb_label=True,
                        color="green",
                        disabled=("!slice_y_show",),
                    )
                    v3.VLabel(
                        "Col–wavelength section",
                        classes="text-caption text-disabled mt-n2 ml-10",
                    )

            # Z slice (band/wavelength → spatial image)
            with v3.VRow(dense=True, align="center", classes="mb-1"):
                with v3.VCol(cols=2):
                    v3.VCheckbox(
                        v_model=("slice_z_show", True),
                        density="compact",
                        hide_details=True,
                        color="blue",
                    )
                with v3.VCol(cols=10):
                    v3.VSlider(
                        v_model=("slice_z_index", 0),
                        min=0,
                        max=("slice_z_max", 100),
                        step=1,
                        label="Band",
                        density="compact",
                        hide_details=True,
                        thumb_label=True,
                        color="blue",
                        disabled=("!slice_z_show",),
                    )
                    v3.VLabel(
                        "Spatial image at wavelength",
                        classes="text-caption text-disabled mt-n2 ml-10",
                    )

            v3.VDivider(classes="mb-2")

            # Colormap
            v3.VSelect(
                v_model=("colormap_3d", "Viridis"),
                items=([
                    "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
                    "RdYlGn", "RdBu",
                ],),
                label="Colormap",
                density="compact",
                variant="outlined",
                hide_details=True,
                classes="mb-2",
            )

            # Opacity
            v3.VSlider(
                v_model=("opacity_3d", 0.9),
                min=0.1,
                max=1.0,
                step=0.05,
                label="Opacity",
                density="compact",
                hide_details=True,
                thumb_label=True,
                classes="mb-2",
            )

            # Load button
            v3.VBtn(
                "Load 3D Cube",
                click=ctrl.load_3d_cube,
                color="primary",
                size="small",
                variant="tonal",
                block=True,
                prepend_icon="mdi-cube-scan",
                loading=("cube_3d_loading", False),
            )
