"""Webcam capture."""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import TracebackType
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """The camera could not be opened or has stopped delivering frames."""


def _default_capture_factory(index: int) -> Any:
    import cv2

    return cv2.VideoCapture(index)


class Camera:
    """OpenCV webcam wrapper.

    ``read()`` returns ``None`` for an occasional dropped frame and raises
    ``CameraError`` after ``max_read_failures`` consecutive failures (unplugged
    or claimed by another app). ``release()`` is idempotent.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        max_read_failures: int = 30,
        capture_factory: Callable[[int], Any] = _default_capture_factory,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._max_read_failures = max_read_failures
        self._capture_factory = capture_factory
        self._capture: Any = None
        self._failures = 0

    def open(self) -> None:
        if self._capture is not None:
            return
        try:
            capture = self._capture_factory(self._index)
        except Exception as exc:
            raise CameraError(f"could not open camera {self._index}: {exc}") from exc
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise CameraError(
                f"could not open camera {self._index}; is it connected and not in use "
                "by another application (or blocked by Windows camera privacy settings)?"
            )
        self._set_property(capture, "CAP_PROP_FRAME_WIDTH", self._width)
        self._set_property(capture, "CAP_PROP_FRAME_HEIGHT", self._height)
        self._set_property(capture, "CAP_PROP_BUFFERSIZE", 1)  # keep latency low
        self._capture = capture
        self._failures = 0

    def read(self) -> np.ndarray | None:
        """Return a BGR frame, or None if this read failed."""
        if self._capture is None:
            raise CameraError("camera is not open")
        try:
            ok, frame = self._capture.read()
        except Exception as exc:
            raise CameraError(f"camera read failed: {exc}") from exc
        if not ok or frame is None:
            self._failures += 1
            if self._failures >= self._max_read_failures:
                raise CameraError(f"camera stopped delivering frames ({self._failures} failures)")
            return None
        self._failures = 0
        return frame

    def release(self) -> None:
        capture, self._capture = self._capture, None
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            logger.exception("error while releasing camera")

    def __enter__(self) -> Camera:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    @staticmethod
    def _set_property(capture: Any, name: str, value: int) -> None:
        import cv2

        capture.set(getattr(cv2, name), value)
