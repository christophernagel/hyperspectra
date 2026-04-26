"""Camera engine for the hyperspectral web viewer.

Thread-safe wrapper around xiapi.Camera. Owns the device, the parameter
state, the acquisition state machine, and the preview worker. UI-free:
all interaction with trame happens via callbacks the app provides.

Bring-up notes baked in:
- ximea is imported lazily so the viewer still loads with no camera installed.
- Default buffer policy is XI_BP_SAFE with a 4-buffer / 64 MiB ring. The
  default ring (~52 buffers, 256 MiB) crashes on the non-zerocopy path
  until cma=300M is on the kernel cmdline.
- Frames are decoded from img.get_image_data_raw() because
  get_image_data_numpy() segfaults on this xiapi build.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tifffile

from .camera_worker import CameraWorker

logger = logging.getLogger(__name__)

DEFAULT_FORMAT = "XI_MONO8"
SAFE_BUFFER_QUEUE_SIZE = 4
SAFE_ACQ_BUFFER_SIZE = 64 * 1024 * 1024


class CameraNotConnectedError(RuntimeError):
    pass


class CameraEngine:
    def __init__(self) -> None:
        self._cam = None
        self._img = None
        self._lock = threading.RLock()
        self._acquiring = False
        self._worker: Optional[CameraWorker] = None
        self._device_info: dict = {}
        self._capabilities: dict = {}

    @property
    def is_connected(self) -> bool:
        return self._cam is not None

    @property
    def is_previewing(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    # ----- lifecycle -----

    def connect(self) -> dict:
        with self._lock:
            if self.is_connected:
                return self.snapshot()

            try:
                from ximea import xiapi
            except ImportError as exc:
                raise CameraNotConnectedError(
                    "ximea Python bindings not installed"
                ) from exc

            cam = xiapi.Camera()
            cam.open_device()
            try:
                cam.set_imgdataformat(DEFAULT_FORMAT)
                cam.set_buffer_policy("XI_BP_SAFE")
                cam.set_buffers_queue_size(SAFE_BUFFER_QUEUE_SIZE)
                cam.set_acq_buffer_size(SAFE_ACQ_BUFFER_SIZE)
                # Sensible defaults so the very first preview frame is visible
                # and doesn't blow past USB by free-running at the prior
                # session's microsecond exposure.
                try:
                    cam.set_exposure(10_000)  # 10 ms
                except Exception:
                    logger.warning("could not set default exposure")
                try:
                    cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT")
                    cam.set_framerate(30.0)
                except Exception:
                    logger.warning("could not enable framerate limit; free-running")
            except Exception:
                cam.close_device()
                raise

            self._cam = cam
            self._img = xiapi.Image()
            self._device_info = {
                "model": cam.get_device_name().decode(errors="replace"),
                "serial": cam.get_device_sn().decode(errors="replace"),
                "api_version": getattr(__import__("ximea"), "__version__", "?"),
            }
            self._refresh_capabilities()
            logger.info("XIMEA camera connected: %s", self._device_info)
            return self.snapshot()

    def disconnect(self) -> None:
        self.stop_preview()
        with self._lock:
            if self._cam is None:
                return
            try:
                if self._acquiring:
                    self._cam.stop_acquisition()
                    self._acquiring = False
                self._cam.close_device()
            except Exception:
                logger.exception("error during camera disconnect")
            finally:
                self._cam = None
                self._img = None
                self._device_info = {}
                self._capabilities = {}

    # ----- capabilities -----

    def _refresh_capabilities(self) -> None:
        cam = self._cam
        if cam is None:
            return

        def _try(getter, default=None):
            try:
                return getter()
            except Exception:
                return default

        self._capabilities = {
            "exposure": {
                "min": _try(cam.get_exposure_minimum, 1),
                "max": _try(cam.get_exposure_maximum, 1_000_000),
                "inc": _try(cam.get_exposure_increment, 1),
                "value": _try(cam.get_exposure, 10_000),
            },
            "gain": {
                "min": _try(cam.get_gain_minimum, 0.0),
                "max": _try(cam.get_gain_maximum, 24.0),
                "inc": _try(cam.get_gain_increment, 0.1),
                "value": _try(cam.get_gain, 0.0),
            },
            "width": {
                "min": _try(cam.get_width_minimum, 8),
                "max": _try(cam.get_width_maximum, 2464),
                "inc": _try(cam.get_width_increment, 16),
                "value": _try(cam.get_width, 2464),
            },
            "height": {
                "min": _try(cam.get_height_minimum, 2),
                "max": _try(cam.get_height_maximum, 2056),
                "inc": _try(cam.get_height_increment, 2),
                "value": _try(cam.get_height, 2056),
            },
            "offsetX": {
                "min": _try(cam.get_offsetX_minimum, 0),
                "max": _try(cam.get_offsetX_maximum, 0),
                "inc": _try(cam.get_offsetX_increment, 16),
                "value": _try(cam.get_offsetX, 0),
            },
            "offsetY": {
                "min": _try(cam.get_offsetY_minimum, 0),
                "max": _try(cam.get_offsetY_maximum, 0),
                "inc": _try(cam.get_offsetY_increment, 2),
                "value": _try(cam.get_offsetY, 0),
            },
            "format": _try(lambda: cam.get_imgdataformat(), DEFAULT_FORMAT),
            "bit_depth": {
                "min": _try(cam.get_sensor_bit_depth_minimum, 8),
                "max": _try(cam.get_sensor_bit_depth_maximum, 12),
                "value": _try(cam.get_sensor_bit_depth, 8),
            },
        }

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "connected": self.is_connected,
                "previewing": self.is_previewing,
                "device": dict(self._device_info),
                "capabilities": json.loads(json.dumps(self._capabilities)),
                "acquiring": self._acquiring,
            }

    # ----- live parameter changes -----

    def set_exposure_us(self, exposure_us: int) -> int:
        with self._lock:
            self._require_connected()
            self._cam.set_exposure(int(exposure_us))
            actual = self._cam.get_exposure()
            self._capabilities["exposure"]["value"] = actual
            return actual

    def set_gain_db(self, gain_db: float) -> float:
        with self._lock:
            self._require_connected()
            self._cam.set_gain(float(gain_db))
            actual = self._cam.get_gain()
            self._capabilities["gain"]["value"] = actual
            return actual

    # ----- stateful changes (acquisition stop/restart) -----

    def set_roi(self, width: int, height: int, offsetX: int, offsetY: int) -> dict:
        with self._lock:
            self._require_connected()
            was_acquiring = self._acquiring
            if was_acquiring:
                self._cam.stop_acquisition()
                self._acquiring = False

            # Zero offsets first so (offset + size) can't transiently exceed
            # sensor bounds and get the call rejected.
            self._cam.set_offsetX(0)
            self._cam.set_offsetY(0)
            self._cam.set_width(int(width))
            self._cam.set_height(int(height))
            self._cam.set_offsetX(int(offsetX))
            self._cam.set_offsetY(int(offsetY))

            actual = {
                "width": self._cam.get_width(),
                "height": self._cam.get_height(),
                "offsetX": self._cam.get_offsetX(),
                "offsetY": self._cam.get_offsetY(),
            }
            for k, v in actual.items():
                self._capabilities[k]["value"] = v

            if was_acquiring:
                self._cam.start_acquisition()
                self._acquiring = True
            return actual

    def set_format(
        self, imgdataformat: str, sensor_bit_depth: Optional[str] = None
    ) -> dict:
        # sensor_bit_depth is an xiapi enum string (e.g. "XI_BPP_8"), not an int.
        with self._lock:
            self._require_connected()
            was_acquiring = self._acquiring
            if was_acquiring:
                self._cam.stop_acquisition()
                self._acquiring = False

            self._cam.set_imgdataformat(imgdataformat)
            if sensor_bit_depth:
                try:
                    self._cam.set_sensor_bit_depth(sensor_bit_depth)
                except Exception:
                    logger.warning(
                        "set_sensor_bit_depth(%s) rejected by camera",
                        sensor_bit_depth,
                    )

            self._capabilities["format"] = imgdataformat
            try:
                self._capabilities["bit_depth"]["value"] = (
                    self._cam.get_sensor_bit_depth()
                )
            except Exception:
                pass

            if was_acquiring:
                self._cam.start_acquisition()
                self._acquiring = True
            return {
                "format": imgdataformat,
                "bit_depth": self._capabilities["bit_depth"]["value"],
            }

    # ----- acquisition -----

    def grab_one(self) -> np.ndarray:
        with self._lock:
            self._require_connected()
            was_acquiring = self._acquiring
            if not was_acquiring:
                self._cam.start_acquisition()
                self._acquiring = True
            try:
                self._cam.get_image(self._img)
                return self._image_to_numpy(self._img)
            finally:
                if not was_acquiring:
                    self._cam.stop_acquisition()
                    self._acquiring = False

    def grab_for_worker(self, timeout_ms: int = 5000) -> tuple[np.ndarray, dict]:
        with self._lock:
            self._require_connected()
            if not self._acquiring:
                self._cam.start_acquisition()
                self._acquiring = True
            self._cam.get_image(self._img, timeout=timeout_ms)
            arr = self._image_to_numpy(self._img)
            return arr, {
                "nframe": int(self._img.nframe),
                "width": int(self._img.width),
                "height": int(self._img.height),
            }

    @staticmethod
    def _image_to_numpy(img) -> np.ndarray:
        # img.frm: 0=MONO8, 1=MONO16, 2=RGB24, 3=RGB32, 5=RAW8, 6=RAW16
        frm = int(img.frm)
        h, w = int(img.height), int(img.width)
        raw = img.get_image_data_raw()
        if frm in (0, 5):
            return np.frombuffer(raw, dtype=np.uint8, count=h * w).reshape(h, w).copy()
        if frm in (1, 6):
            return np.frombuffer(raw, dtype=np.uint16, count=h * w).reshape(h, w).copy()
        return img.get_image_data_numpy()

    # ----- status -----

    def read_status(self) -> dict:
        # get_available_bandwidth measures the link by stopping/restarting
        # acquisition and errors out if called while acquiring. We treat it
        # as a one-shot read taken when idle and let it stay None otherwise.
        with self._lock:
            if not self.is_connected:
                return {}
            try:
                temp = float(self._cam.get_sensor_board_temp())
            except Exception:
                temp = None
            bw_mbps = None
            if not self._acquiring:
                try:
                    bw_mbits = int(self._cam.get_available_bandwidth())
                    bw_mbps = round(bw_mbits / 8.0, 1)
                except Exception:
                    bw_mbps = None
            return {"board_temp_c": temp, "bandwidth_mbps": bw_mbps}

    # ----- preview -----

    def start_preview(
        self,
        on_frame: Callable[[np.ndarray, dict], None],
        on_status: Callable[[dict], None],
        on_error: Callable[[str], None],
        target_fps: float = 8.0,
    ) -> None:
        with self._lock:
            self._require_connected()
            if self.is_previewing:
                return
            self._worker = CameraWorker(
                engine=self,
                on_frame=on_frame,
                on_status=on_status,
                on_error=on_error,
                target_fps=target_fps,
            )
            self._worker.start()

    def stop_preview(self) -> None:
        worker = self._worker
        if worker is None:
            return
        worker.stop()
        worker.join(timeout=2.0)
        with self._lock:
            self._worker = None
            if self._acquiring and self._cam is not None:
                try:
                    self._cam.stop_acquisition()
                except Exception:
                    logger.exception("stop_acquisition failed")
                self._acquiring = False

    # ----- save -----

    def save_frame(
        self,
        arr: np.ndarray,
        output_dir: str,
        name_prefix: str = "cap",
        extra_meta: Optional[dict] = None,
    ) -> tuple[Path, Path]:
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        stem = f"{name_prefix}_{ts}"
        tif_path = out / f"{stem}.tif"
        json_path = out / f"{stem}_meta.json"

        tifffile.imwrite(str(tif_path), arr)

        with self._lock:
            caps = self._capabilities
            dev = self._device_info
            try:
                temp = (
                    float(self._cam.get_sensor_board_temp())
                    if self._cam is not None
                    else None
                )
            except Exception:
                temp = None

        meta = {
            "filename": tif_path.name,
            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "model": dev.get("model"),
            "serial": dev.get("serial"),
            "api_version": dev.get("api_version"),
            "exposure_us": caps.get("exposure", {}).get("value"),
            "gain_db": caps.get("gain", {}).get("value"),
            "roi": {
                "width": caps.get("width", {}).get("value"),
                "height": caps.get("height", {}).get("value"),
                "offsetX": caps.get("offsetX", {}).get("value"),
                "offsetY": caps.get("offsetY", {}).get("value"),
            },
            "format": caps.get("format"),
            "bit_depth": caps.get("bit_depth", {}).get("value"),
            "board_temp_c": temp,
        }
        if extra_meta:
            meta.update(extra_meta)
        json_path.write_text(json.dumps(meta, indent=2))
        return tif_path, json_path

    # ----- internals -----

    def _require_connected(self) -> None:
        if self._cam is None:
            raise CameraNotConnectedError("camera is not connected")
