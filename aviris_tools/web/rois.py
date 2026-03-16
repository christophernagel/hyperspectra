"""
ROI (Region of Interest) management for the hyperspectral web viewer.

Extracted from HyperspectralWebApp to separate concerns.
"""

import re
import logging

import numpy as np

logger = logging.getLogger(__name__)

ROI_COLORS = [
    "#FF6B6B", "#4ECDC4", "#FFE66D", "#A8E6CF",
    "#FF8B94", "#B39DDB", "#81D4FA", "#FFAB91",
]


class ROIManager:
    """Manages drawn ROIs, spectrum extraction, and material matching."""

    def __init__(self, state, engine, on_change=None):
        self._rois = []
        self._id_counter = 0
        self._state = state
        self._engine = engine
        self._on_change = on_change or (lambda: None)

    @property
    def rois(self):
        return self._rois

    def reset(self):
        """Clear all ROIs and reset counter."""
        self._rois.clear()
        self._id_counter = 0
        self._state.roi_list_ui = []
        self._state.roi_draw_mode = "none"

    def _next_id(self):
        self._id_counter += 1
        return self._id_counter

    # -----------------------------------------------------------------
    # Shape capture from Plotly relayout events
    # -----------------------------------------------------------------

    def on_image_relayout(self, relayout_data):
        """Capture newly drawn shapes from Plotly relayout events."""
        if not relayout_data or self._engine.data_loader is None:
            return

        logger.debug(f"relayout keys: {list(relayout_data.keys())}")

        # Method 1: Plotly sends full shapes array
        shapes = relayout_data.get("shapes", None)
        if shapes is not None:
            existing_count = len(self._rois)
            if len(shapes) > existing_count:
                for shape in shapes[existing_count:]:
                    self.create_from_shape(shape)
                return

        # Method 2: individual shape[N].key entries
        shape_props = {}
        for key, val in relayout_data.items():
            m = re.match(r"shapes\[(\d+)\]\.(\w+)", key)
            if m:
                idx = int(m.group(1))
                prop = m.group(2)
                shape_props.setdefault(idx, {})[prop] = val

        if shape_props:
            existing_count = len(self._rois)
            for idx in sorted(shape_props.keys()):
                if idx >= existing_count:
                    logger.info(
                        f"New shape from relayout: idx={idx}, "
                        f"props={shape_props[idx]}"
                    )
                    self.create_from_shape(shape_props[idx])

    def create_from_shape(self, shape_dict):
        """Parse a Plotly shape dict, extract spectrum, match materials."""
        shape_type = shape_dict.get("type", "")
        logger.info(
            f"Creating ROI from shape: type={shape_type}, "
            f"keys={list(shape_dict.keys())}"
        )
        mask = None

        if shape_type == "rect":
            x0 = int(round(shape_dict.get("x0", 0)))
            x1 = int(round(shape_dict.get("x1", 0)))
            y0 = int(round(shape_dict.get("y0", 0)))
            y1 = int(round(shape_dict.get("y1", 0)))
            col_min, col_max = min(x0, x1), max(x0, x1)
            row_min, row_max = min(y0, y1), max(y0, y1)
            roi_type = "rect"
        elif shape_type == "path":
            path_str = shape_dict.get("path", "")
            vertices = self.parse_svg_path(path_str)
            if not vertices or len(vertices) < 3:
                return
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            col_min, col_max = int(min(xs)), int(max(xs)) + 1
            row_min, row_max = int(min(ys)), int(max(ys)) + 1
            mask = self.polygon_to_mask(
                vertices, row_min, row_max, col_min, col_max
            )
            roi_type = "polygon"
        else:
            return

        row_min = max(0, row_min)
        col_min = max(0, col_min)
        row_max = min(self._engine.data_loader.n_rows, row_max)
        col_max = min(self._engine.data_loader.n_cols, col_max)

        result = self._engine.get_roi_spectrum(
            row_min, row_max, col_min, col_max, mask=mask
        )
        if result is None:
            return

        matches = self._engine.match_roi_spectrum(result["mean_values"])

        roi_id = self._next_id()
        color = ROI_COLORS[(roi_id - 1) % len(ROI_COLORS)]
        roi = {
            "id": roi_id,
            "name": f"ROI {roi_id}",
            "color": color,
            "roi_type": roi_type,
            "visible": True,
            "shape_data": shape_dict,
            "row_min": row_min,
            "row_max": row_max,
            "col_min": col_min,
            "col_max": col_max,
            "mask": mask,
            "pixel_count": result["pixel_count"],
            "wavelengths": result["wavelengths"],
            "mean_spectrum": result["mean_values"],
            "std_spectrum": result["std_values"],
            "matches": matches,
        }
        self._rois.append(roi)

        self._state.roi_draw_mode = "none"
        self.sync_ui()
        self._on_change()

    # -----------------------------------------------------------------
    # SVG / polygon helpers
    # -----------------------------------------------------------------

    @staticmethod
    def parse_svg_path(path_str):
        """Parse Plotly SVG path 'M10,20L30,40L50,60Z' → [(x,y), ...]."""
        coords = re.findall(
            r"[ML]\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)", path_str
        )
        return [(float(x), float(y)) for x, y in coords]

    @staticmethod
    def polygon_to_mask(vertices, row_min, row_max, col_min, col_max):
        """Rasterize polygon vertices to boolean mask within bounding box."""
        from matplotlib.path import Path as MplPath

        n_rows = row_max - row_min
        n_cols = col_max - col_min
        if n_rows <= 0 or n_cols <= 0:
            return None

        cols = np.arange(col_min, col_max) + 0.5
        rows = np.arange(row_min, row_max) + 0.5
        cc, rr = np.meshgrid(cols, rows)
        points = np.stack([cc.ravel(), rr.ravel()], axis=1)

        path = MplPath(vertices)
        mask = path.contains_points(points).reshape(n_rows, n_cols)
        return mask

    # -----------------------------------------------------------------
    # CRUD operations
    # -----------------------------------------------------------------

    def remove(self, roi_id):
        roi_id = int(roi_id)
        self._rois = [r for r in self._rois if r["id"] != roi_id]
        self.sync_ui()
        self._on_change()

    def clear_all(self):
        self._rois.clear()
        self.sync_ui()
        self._on_change()

    def set_visible(self, roi_id, visible):
        roi_id = int(roi_id)
        for r in self._rois:
            if r["id"] == roi_id:
                r["visible"] = bool(visible)
                break
        self.sync_ui()
        self._on_change()

    def sync_ui(self):
        """Push serializable ROI list to trame state."""
        self._state.roi_list_ui = [
            {
                "id": r["id"],
                "name": r["name"],
                "color": r["color"],
                "roi_type": r["roi_type"],
                "visible": r["visible"],
                "pixel_count": r["pixel_count"],
                "top_match": (
                    r["matches"][0]["name"] if r["matches"] else ""
                ),
                "top_score": (
                    round(r["matches"][0]["score"], 4)
                    if r["matches"] else 0
                ),
            }
            for r in self._rois
        ]
