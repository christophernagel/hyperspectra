"""
L1 atmospheric correction processing panel for the web viewer.
"""

from trame.widgets import vuetify3 as v3


def build_processing_panel(state, ctrl, data_dir="/data"):
    """Build the L1 atmospheric correction interface in the sidebar."""

    with v3.VCard(variant="outlined", classes="mb-2"):
        v3.VCardTitle("Atmospheric Correction", classes="text-subtitle-2 pa-2")
        with v3.VCardText(classes="pa-2 pt-0"):
            v3.VAlert(
                text="Convert L1B radiance to L2 surface reflectance",
                type="info",
                variant="tonal",
                density="compact",
                classes="mb-3",
            )

            # Radiance file
            v3.VSelect(
                v_model=("l1_radiance_file",),
                items=("l1_file_list",),
                item_title="title",
                item_value="value",
                label="Radiance File (L1B RDN)",
                density="compact",
                variant="outlined",
                hide_details=True,
                clearable=True,
                classes="mb-2",
            )

            # Observation file
            v3.VSelect(
                v_model=("l1_obs_file",),
                items=("l1_file_list",),
                item_title="title",
                item_value="value",
                label="Observation File (OBS)",
                density="compact",
                variant="outlined",
                hide_details=True,
                clearable=True,
                classes="mb-2",
            )

            # Output path
            v3.VTextField(
                v_model=("l1_output_path", ""),
                label="Output Path",
                density="compact",
                variant="outlined",
                hide_details=True,
                placeholder=f"{data_dir}/output_L2_REFL.nc",
                classes="mb-2",
            )

            v3.VDivider(classes="my-2")

            # Aerosol model
            v3.VSelect(
                v_model=("l1_aerosol", "continental"),
                items=(["continental", "maritime", "urban", "desert"],),
                label="Aerosol Model",
                density="compact",
                variant="outlined",
                hide_details=True,
                classes="mb-2",
            )

            # Altitude
            v3.VTextField(
                v_model=("l1_altitude", "8.5"),
                label="Sensor Altitude (km)",
                type="number",
                density="compact",
                variant="outlined",
                hide_details=True,
                classes="mb-2",
            )

            # Options
            v3.VCheckbox(
                v_model=("l1_use_6s", True),
                label="Use Py6S radiative transfer",
                density="compact",
                hide_details=True,
            )
            v3.VCheckbox(
                v_model=("l1_validate", True),
                label="Validate output",
                density="compact",
                hide_details=True,
            )

            # Run button
            v3.VBtn(
                "Run Correction",
                click=ctrl.run_atm_correction,
                color="warning",
                block=True,
                classes="mt-3",
                prepend_icon="mdi-play",
                loading=("l1_processing",),
                disabled=("l1_processing",),
            )

    # Log output
    with v3.VCard(variant="outlined", classes="mb-2"):
        v3.VCardTitle("Processing Log", classes="text-subtitle-2 pa-2")
        with v3.VCardText(classes="pa-2 pt-0"):
            v3.VTextarea(
                v_model=("l1_log", ""),
                readonly=True,
                rows=12,
                density="compact",
                variant="outlined",
                hide_details=True,
                auto_grow=False,
                style="font-family: monospace; font-size: 11px; overflow-y: auto;",
            )
            v3.VProgressLinear(
                v_show="l1_processing",
                model_value=("l1_progress", 0),
                color="warning",
                height=20,
                classes="mt-1",
                rounded=True,
            )
            v3.VLabel(
                v_show="l1_processing",
                text=("l1_progress_label",),
                classes="text-caption text-center mt-1 d-block",
            )
