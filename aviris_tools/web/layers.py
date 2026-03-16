"""
Layer management for the hyperspectral web viewer.

Extracted from HyperspectralWebApp to separate concerns.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Module-level cache: cmap_name → uint8 LUT (256, 4)
_LUT_CACHE = {}


class LayerManager:
    """Manages a stack of image layers (bands, RGB composites, indices)."""

    def __init__(self, state, on_change=None):
        self._layers = []
        self._id_counter = 0
        self._state = state
        self._on_change = on_change or (lambda: None)
        self._composite_cache = (None, None)  # (fingerprint, result)

    @property
    def layers(self):
        return self._layers

    def reset(self):
        """Clear all layers and reset counter."""
        self._layers.clear()
        self._id_counter = 0

    def _next_id(self):
        self._id_counter += 1
        return self._id_counter

    def add(self, name, layer_type, data,
            colorscale=None, zmin=None, zmax=None, colorbar_title=None,
            colorbar_low="", colorbar_high=""):
        layer = {
            "id": self._next_id(),
            "name": name,
            "layer_type": layer_type,
            "visible": True,
            "opacity": 0.7 if layer_type == "index" else 1.0,
            "data": data,
            "colorscale": colorscale,
            "zmin": zmin,
            "zmax": zmax,
            "colorbar_title": colorbar_title,
            "colorbar_low": colorbar_low,
            "colorbar_high": colorbar_high,
        }
        self._layers.append(layer)
        self.sync_ui()
        self._on_change()

    def remove(self, layer_id):
        layer_id = int(layer_id)
        self._layers = [
            ly for ly in self._layers if ly["id"] != layer_id
        ]
        self.sync_ui()
        self._on_change()

    def clear_all(self):
        self._layers.clear()
        self.sync_ui()
        self._on_change()

    def set_visible(self, layer_id, visible):
        layer_id = int(layer_id)
        for ly in self._layers:
            if ly["id"] == layer_id:
                ly["visible"] = bool(visible)
                break
        self.sync_ui()
        self._on_change()

    def set_opacity(self, layer_id, opacity):
        layer_id = int(layer_id)
        for ly in self._layers:
            if ly["id"] == layer_id:
                ly["opacity"] = max(0.0, min(1.0, float(opacity)))
                break
        self._on_change()

    def move(self, layer_id, direction):
        for i, ly in enumerate(self._layers):
            if ly["id"] == layer_id:
                new_i = i + direction
                if 0 <= new_i < len(self._layers):
                    self._layers[i], self._layers[new_i] = (
                        self._layers[new_i], self._layers[i]
                    )
                break
        self.sync_ui()
        self._on_change()

    def sync_ui(self):
        """Push serializable layer list to trame state."""
        self._state.layer_list_ui = [
            {
                "id": ly["id"],
                "name": ly["name"],
                "layer_type": ly["layer_type"],
                "visible": ly["visible"],
                "opacity": ly["opacity"],
            }
            for ly in reversed(self._layers)
        ]
        has_index = any(
            ly["layer_type"] == "index" and ly["visible"]
            for ly in self._layers
        )
        if not has_index:
            self._state.index_meta_low = ""
            self._state.index_meta_high = ""

    # -----------------------------------------------------------------
    # Colormap + compositing
    # -----------------------------------------------------------------

    @staticmethod
    def apply_colormap(data, cmap_name, zmin, zmax):
        """Apply a matplotlib colormap to 2D data → RGBA uint8 (H,W,4)."""
        arr = np.nan_to_num(data, nan=0.0).astype(np.float32)
        if zmin is None:
            zmin = float(np.nanmin(arr))
        if zmax is None:
            zmax = float(np.nanmax(arr))
        if zmax <= zmin:
            zmax = zmin + 1.0

        # Cached LUT lookup
        lut = _LUT_CACHE.get(cmap_name)
        if lut is None:
            import matplotlib.cm as cm
            try:
                cmap = cm.get_cmap(cmap_name)
            except (ValueError, KeyError):
                cmap = cm.get_cmap("viridis")
            lut = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
            _LUT_CACHE[cmap_name] = lut

        idx = np.clip(
            ((arr - zmin) / (zmax - zmin) * 255), 0, 255
        ).astype(np.uint8)
        return lut[idx]

    def _composite_fingerprint(self, dark_mode):
        """Build a lightweight cache key for the current layer state."""
        return tuple(
            (ly["id"], ly["visible"], ly["opacity"], id(ly["data"]),
             ly["colorscale"], ly["zmin"], ly["zmax"])
            for ly in self._layers
        ) + (("dark", dark_mode),)

    def composite(self, dark_mode):
        """Composite visible layers bottom-to-top → uint8 RGB (H,W,3)."""
        visible = [ly for ly in self._layers if ly["visible"]]
        if not visible:
            return None

        fp = self._composite_fingerprint(dark_mode)
        cached_fp, cached_result = self._composite_cache
        if fp == cached_fp and cached_result is not None:
            return cached_result

        first = visible[0]
        if first["layer_type"] == "rgb":
            n_rows, n_cols = first["data"].shape[:2]
        else:
            n_rows, n_cols = first["data"].shape

        bg = 1.0 if not dark_mode else 0.0
        canvas = np.full((n_rows, n_cols, 4), bg, dtype=np.float64)
        canvas[:, :, 3] = 1.0

        for layer in visible:
            alpha = layer["opacity"]
            if layer["layer_type"] == "rgb":
                src = layer["data"].astype(np.float64) / 255.0
                canvas[:, :, :3] = (
                    src * alpha + canvas[:, :, :3] * (1 - alpha)
                )
            else:
                rgba = self.apply_colormap(
                    layer["data"], layer["colorscale"] or "viridis",
                    layer["zmin"], layer["zmax"],
                )
                src = rgba.astype(np.float64) / 255.0
                src_alpha = (src[:, :, 3] * alpha)[:, :, np.newaxis]
                inv_alpha = 1.0 - src_alpha
                canvas[:, :, :3] = (
                    src[:, :, :3] * src_alpha
                    + canvas[:, :, :3] * inv_alpha
                )

        result = (np.clip(canvas[:, :, :3], 0, 1) * 255).astype(np.uint8)
        self._composite_cache = (fp, result)
        return result

    # -----------------------------------------------------------------
    # Public helpers
    # -----------------------------------------------------------------

    def get_summary(self):
        """Serializable layer summary for public API."""
        return [
            {
                "id": ly["id"], "name": ly["name"],
                "type": ly["layer_type"],
                "visible": ly["visible"], "opacity": ly["opacity"],
            }
            for ly in self._layers
        ]

    def get_topmost_visible(self):
        """Return data array of the topmost visible layer, or None."""
        for ly in reversed(self._layers):
            if ly["visible"]:
                return ly["data"]
        return None
