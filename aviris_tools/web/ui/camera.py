"""Live camera capture panel for the hyperspectral web viewer."""
from trame.widgets import vuetify3 as v3, plotly as trame_plotly

from ..figures import make_empty_figure


def build_camera_panel(state, ctrl):
    """Build the camera tab content. Returns the preview Figure widget so
    the app can update it from the worker callback."""

    preview_widget = None

    # ---- Connection ----
    with v3.VCard(variant="outlined", classes="mb-2"):
        v3.VCardTitle("Camera", classes="text-subtitle-2 pa-2")
        with v3.VCardText(classes="pa-2 pt-0"):
            with v3.VRow(dense=True, classes="mb-2", align="center"):
                with v3.VCol(cols=7):
                    v3.VChip(
                        text=("camera_status_label",),
                        color=("camera_status_color",),
                        size="small",
                        prepend_icon=("camera_status_icon",),
                        variant="tonal",
                    )
                with v3.VCol(cols=5, classes="text-right"):
                    v3.VBtn(
                        v_show="!camera_connected",
                        text="Connect",
                        click=ctrl.camera_connect,
                        color="primary",
                        size="small",
                        variant="tonal",
                        prepend_icon="mdi-power-plug",
                    )
                    v3.VBtn(
                        v_show="camera_connected",
                        text="Disconnect",
                        click=ctrl.camera_disconnect,
                        color="error",
                        size="small",
                        variant="tonal",
                        prepend_icon="mdi-power-plug-off",
                    )
            v3.VLabel(
                v_show="camera_connected",
                text=(
                    "`${camera_device_model || ''}"
                    " | SN ${camera_device_serial || ''}`",
                ),
                classes="text-caption text-disabled d-block",
            )
            v3.VLabel(
                v_show="camera_error",
                text=("camera_error",),
                classes="text-caption text-error d-block mt-1",
                style="white-space: pre-wrap;",
            )

    # ---- Preview ----
    with v3.VCard(variant="outlined", classes="mb-2"):
        v3.VCardTitle("Preview", classes="text-subtitle-2 pa-2")
        with v3.VCardText(classes="pa-1 pt-0"):
            preview_widget = trame_plotly.Figure(
                figure=make_empty_figure("Connect & start preview"),
                display_mode_bar=False,
                style="width: 100%; height: 280px;",
            )
            with v3.VRow(dense=True, classes="mt-1", align="center"):
                with v3.VCol(cols=7):
                    v3.VBtn(
                        v_show="!camera_previewing",
                        text="Start Preview",
                        click=ctrl.camera_start_preview,
                        color="success",
                        size="small",
                        variant="tonal",
                        block=True,
                        prepend_icon="mdi-play",
                        disabled=("!camera_connected",),
                    )
                    v3.VBtn(
                        v_show="camera_previewing",
                        text="Stop Preview",
                        click=ctrl.camera_stop_preview,
                        color="warning",
                        size="small",
                        variant="tonal",
                        block=True,
                        prepend_icon="mdi-stop",
                    )
                with v3.VCol(cols=5):
                    v3.VLabel(
                        text=("`${camera_fps.toFixed(1)} fps`",),
                        classes="text-caption d-block text-right",
                    )

    # ---- Exposure / Gain ----
    with v3.VCard(variant="outlined", classes="mb-2"):
        v3.VCardTitle("Exposure / Gain", classes="text-subtitle-2 pa-2")
        with v3.VCardText(classes="pa-2 pt-0"):
            v3.VLabel("Exposure (µs)", classes="text-caption")
            v3.VSlider(
                v_model=("camera_exposure_us", 10000),
                min=("camera_exposure_min", 1),
                max=("camera_exposure_max", 1000000),
                step=("camera_exposure_inc", 1),
                density="compact",
                hide_details=True,
                thumb_label=True,
                disabled=("!camera_connected",),
                classes="mb-1",
            )
            v3.VTextField(
                v_model=("camera_exposure_us",),
                type="number",
                density="compact",
                variant="outlined",
                hide_details=True,
                disabled=("!camera_connected",),
                classes="mb-3",
            )
            v3.VLabel("Gain (dB)", classes="text-caption")
            v3.VSlider(
                v_model=("camera_gain_db", 0.0),
                min=("camera_gain_min", 0.0),
                max=("camera_gain_max", 24.0),
                step=("camera_gain_inc", 0.1),
                density="compact",
                hide_details=True,
                thumb_label=True,
                disabled=("!camera_connected",),
            )

    # ---- ROI ----
    with v3.VCard(variant="outlined", classes="mb-2"):
        v3.VCardTitle("ROI", classes="text-subtitle-2 pa-2")
        with v3.VCardText(classes="pa-2 pt-0"):
            with v3.VRow(dense=True):
                with v3.VCol(cols=6):
                    v3.VTextField(
                        v_model=("camera_roi_width", 2464),
                        label="Width",
                        type="number",
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                        disabled=("!camera_connected",),
                    )
                with v3.VCol(cols=6):
                    v3.VTextField(
                        v_model=("camera_roi_height", 2056),
                        label="Height",
                        type="number",
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                        disabled=("!camera_connected",),
                    )
            with v3.VRow(dense=True, classes="mt-1"):
                with v3.VCol(cols=6):
                    v3.VTextField(
                        v_model=("camera_roi_offsetX", 0),
                        label="Offset X",
                        type="number",
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                        disabled=("!camera_connected",),
                    )
                with v3.VCol(cols=6):
                    v3.VTextField(
                        v_model=("camera_roi_offsetY", 0),
                        label="Offset Y",
                        type="number",
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                        disabled=("!camera_connected",),
                    )
            with v3.VRow(dense=True, classes="mt-2"):
                with v3.VCol(cols=6):
                    v3.VBtn(
                        text="Apply",
                        click=ctrl.camera_apply_roi,
                        color="primary",
                        size="small",
                        variant="tonal",
                        block=True,
                        disabled=("!camera_connected",),
                    )
                with v3.VCol(cols=6):
                    v3.VBtn(
                        text="Full Sensor",
                        click=ctrl.camera_reset_roi,
                        size="small",
                        variant="text",
                        block=True,
                        disabled=("!camera_connected",),
                    )

    # ---- Format ----
    with v3.VCard(variant="outlined", classes="mb-2"):
        v3.VCardTitle("Format", classes="text-subtitle-2 pa-2")
        with v3.VCardText(classes="pa-2 pt-0"):
            v3.VSelect(
                v_model=("camera_format", "XI_MONO8"),
                items=([
                    {"title": "8-bit Mono",  "value": "XI_MONO8"},
                    {"title": "16-bit Mono", "value": "XI_MONO16"},
                    {"title": "8-bit Raw",   "value": "XI_RAW8"},
                    {"title": "16-bit Raw",  "value": "XI_RAW16"},
                ],),
                label="Image Format",
                density="compact",
                variant="outlined",
                hide_details=True,
                disabled=("!camera_connected",),
                classes="mb-2",
            )
            v3.VSelect(
                v_model=("camera_bit_depth", "XI_BPP_8"),
                items=([
                    {"title": "8 bit",  "value": "XI_BPP_8"},
                    {"title": "10 bit", "value": "XI_BPP_10"},
                    {"title": "12 bit", "value": "XI_BPP_12"},
                    {"title": "14 bit", "value": "XI_BPP_14"},
                    {"title": "16 bit", "value": "XI_BPP_16"},
                ],),
                label="Sensor Bit Depth",
                density="compact",
                variant="outlined",
                hide_details=True,
                disabled=("!camera_connected",),
            )
            v3.VBtn(
                text="Apply Format",
                click=ctrl.camera_apply_format,
                color="primary",
                size="small",
                variant="tonal",
                block=True,
                disabled=("!camera_connected",),
                classes="mt-2",
            )

    # ---- Status ----
    with v3.VCard(variant="outlined", classes="mb-2"):
        v3.VCardTitle("Status", classes="text-subtitle-2 pa-2")
        with v3.VCardText(classes="pa-2 pt-0"):
            v3.VLabel(
                text=("`Board temp:  ${camera_board_temp_label}`",),
                classes="text-caption d-block",
            )
            v3.VLabel(
                text=("`Bandwidth:   ${camera_bandwidth_label}`",),
                classes="text-caption d-block",
            )
            v3.VLabel(
                text=("`Frame:       #${camera_last_nframe}`",),
                classes="text-caption d-block",
            )

    # ---- Capture ----
    with v3.VCard(variant="outlined", classes="mb-2"):
        v3.VCardTitle("Capture", classes="text-subtitle-2 pa-2")
        with v3.VCardText(classes="pa-2 pt-0"):
            v3.VTextField(
                v_model=("camera_save_dir", "/data/captures"),
                label="Output Directory",
                density="compact",
                variant="outlined",
                hide_details=True,
                classes="mb-2",
            )
            v3.VTextField(
                v_model=("camera_save_prefix", "cap"),
                label="Filename Prefix",
                density="compact",
                variant="outlined",
                hide_details=True,
                classes="mb-2",
            )
            with v3.VRow(dense=True):
                with v3.VCol(cols=7):
                    v3.VBtn(
                        text="Snap 1",
                        click=ctrl.camera_snap_one,
                        color="success",
                        size="small",
                        variant="tonal",
                        block=True,
                        prepend_icon="mdi-camera",
                        disabled=("!camera_connected",),
                    )
                with v3.VCol(cols=5):
                    v3.VTextField(
                        v_model=("camera_burst_count", 5),
                        type="number",
                        density="compact",
                        variant="outlined",
                        hide_details=True,
                        disabled=("!camera_connected",),
                    )
            v3.VBtn(
                text="Snap N",
                click=ctrl.camera_snap_n,
                color="success",
                size="small",
                variant="tonal",
                block=True,
                prepend_icon="mdi-camera-burst",
                disabled=("!camera_connected",),
                classes="mt-2",
            )
            v3.VLabel(
                text=("camera_save_log",),
                classes="text-caption text-disabled d-block mt-2",
                style="white-space: pre-wrap; font-family: monospace;",
            )

    return preview_widget
