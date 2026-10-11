"""The real-time loop that wires camera -> tracker -> gestures -> desktop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

import numpy as np

from nocursor.camera import Camera, CameraError
from nocursor.controller import SafeController
from nocursor.emergency import EmergencyStop, HotkeyWatcher
from nocursor.models import HandObservation
from nocursor.preview import Preview, StatusInfo
from nocursor.processor import GestureFrame, GestureProcessor
from nocursor.tracking import TrackingState

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_EMERGENCY = 2


class Tracker(Protocol):
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandObservation | None: ...
    def close(self) -> None: ...


class NoCursorApp:
    """Runs the control loop; every collaborator is injected so the loop is testable.

    ``run()`` always releases the camera, tracker, preview and hotkey watcher,
    whatever caused the loop to end.
    """

    def __init__(
        self,
        camera: Camera,
        tracker: Tracker,
        processor: GestureProcessor,
        controller: SafeController,
        preview: Preview,
        stop: EmergencyStop,
        hotkey_label: str,
        watcher: HotkeyWatcher | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._camera = camera
        self._tracker = tracker
        self._processor = processor
        self._controller = controller
        self._preview = preview
        self._stop = stop
        self._hotkey_label = hotkey_label
        self._watcher = watcher
        self._clock = clock
        self._fps = 0.0

    def run(self) -> int:
        try:
            if self._watcher is not None:
                self._watcher.start()
            self._camera.open()
            self._loop()
        except CameraError as exc:
            logger.error("camera error: %s", exc)
            return EXIT_ERROR
        except KeyboardInterrupt:
            logger.info("interrupted by Ctrl+C")
            return EXIT_OK
        except Exception:
            self._stop.trigger("unexpected error")
            logger.exception("unexpected error in main loop")
            return EXIT_ERROR
        finally:
            self._shutdown()
        return EXIT_EMERGENCY if self._stop.is_set() else EXIT_OK

    def _loop(self) -> None:
        previous = self._clock()
        while not self._stop.is_set():
            frame = self._camera.read()
            now = self._clock()
            self._update_fps(now - previous)
            previous = now

            if frame is None:  # dropped frame: treat as tracking lost, take no action
                observation = None
                gesture = self._processor.process(None, now)
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                observation = self._tracker.detect(frame, int(now * 1000))
                gesture = self._processor.process(observation, now)

            if not self._stop.is_set():  # the hotkey may have fired during inference
                self._apply(gesture)

            keep_going = self._preview.show(frame, observation, gesture, self._status())
            if not keep_going:
                self._stop.trigger("quit requested from preview window")

    def _apply(self, gesture: GestureFrame) -> None:
        if gesture.tracking is not TrackingState.TRACKING:
            return
        if gesture.cursor is not None:
            self._controller.move_to(*gesture.cursor)
        if gesture.click:
            self._controller.click()

    def _status(self) -> StatusInfo:
        if self._stop.is_set():
            control = "STOPPED"
        else:
            control = "LIVE" if self._controller.enabled else "DRY-RUN"
        return StatusInfo(control, self._fps, self._hotkey_label, self._stop.reason)

    def _update_fps(self, dt: float) -> None:
        if dt > 0:
            instant = 1.0 / dt
            self._fps = instant if self._fps == 0.0 else 0.9 * self._fps + 0.1 * instant

    def _shutdown(self) -> None:
        # Each step is isolated so one failure cannot skip the rest (esp. the camera).
        for name, step in (
            ("hotkey watcher", self._watcher.close if self._watcher else None),
            ("preview", self._preview.close),
            ("tracker", self._tracker.close),
            ("camera", self._camera.release),
        ):
            if step is None:
                continue
            try:
                step()
            except Exception:
                logger.exception("error closing %s", name)
