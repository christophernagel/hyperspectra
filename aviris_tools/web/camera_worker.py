"""Preview pump for the camera engine.

Daemon thread that grabs frames at the camera's natural rate, decimates
them for the websocket, and forwards to the UI via callbacks at a
configurable target FPS. Status (temp/bandwidth) is refreshed less often
because those reads are not free.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

PREVIEW_MAX_DIM = 512
STATUS_INTERVAL_S = 5.0


class CameraWorker(threading.Thread):
    def __init__(
        self,
        engine,
        on_frame: Callable,
        on_status: Callable,
        on_error: Callable,
        target_fps: float = 8.0,
    ) -> None:
        super().__init__(daemon=True, name="CameraWorker")
        self._engine = engine
        self._on_frame = on_frame
        self._on_status = on_status
        self._on_error = on_error
        self._target_dt = 1.0 / max(target_fps, 0.1)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        last_push = 0.0
        last_status = 0.0
        frame_times: list[float] = []

        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                try:
                    arr, meta = self._engine.grab_for_worker(timeout_ms=5000)
                except Exception as exc:
                    logger.exception("camera grab failed")
                    self._on_error(str(exc))
                    return

                frame_times.append(t0)
                while frame_times and frame_times[0] < t0 - 2.0:
                    frame_times.pop(0)

                fps = 0.0
                if len(frame_times) >= 2:
                    fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

                now = time.monotonic()
                if now - last_push >= self._target_dt:
                    decimated = self._decimate(arr)
                    meta["fps"] = round(fps, 2)
                    try:
                        self._on_frame(decimated, meta)
                    except Exception:
                        logger.exception("on_frame callback raised")
                    last_push = now

                if now - last_status >= STATUS_INTERVAL_S:
                    try:
                        status = self._engine.read_status()
                        status["fps"] = round(fps, 2)
                        self._on_status(status)
                    except Exception:
                        logger.exception("on_status callback raised")
                    last_status = now
        finally:
            logger.info("CameraWorker exiting")

    @staticmethod
    def _decimate(arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        long_side = max(h, w)
        if long_side <= PREVIEW_MAX_DIM:
            return arr
        stride = int(np.ceil(long_side / PREVIEW_MAX_DIM))
        return arr[::stride, ::stride]
