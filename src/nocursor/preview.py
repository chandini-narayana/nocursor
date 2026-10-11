"""Camera preview window with hand landmarks and control status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from nocursor.gestures import PinchState
from nocursor.models import HAND_CONNECTIONS, INDEX_TIP, THUMB_TIP, HandObservation
from nocursor.processor import GestureFrame
from nocursor.tracking import TrackingState

WINDOW_NAME = "NoCursor"
_QUIT_KEYS = {27, ord("q"), ord("Q")}  # Esc, q


@dataclass(frozen=True)
class StatusInfo:
    """Non-gesture state the preview should display."""

    control: str  # "LIVE", "DRY-RUN" or "STOPPED"
    fps: float
    hotkey: str
    stop_reason: str | None = None


class Preview(Protocol):
    def show(
        self,
        frame_bgr: np.ndarray,
        observation: HandObservation | None,
        gesture: GestureFrame,
        status: StatusInfo,
    ) -> bool:
        """Render a frame. Returns False if the user asked to quit."""
        ...

    def close(self) -> None: ...


class NullPreview:
    """No-op preview for tests."""

    def show(
        self,
        frame_bgr: np.ndarray,
        observation: HandObservation | None,
        gesture: GestureFrame,
        status: StatusInfo,
    ) -> bool:
        return True

    def close(self) -> None:
        pass


class OpenCVPreview:
    """Mirrored preview window. Esc/Q or closing the window requests quit."""

    def __init__(self, active_margin: float) -> None:
        import cv2

        self._cv2 = cv2
        self._margin = active_margin
        self._opened = False

    def show(
        self,
        frame_bgr: np.ndarray,
        observation: HandObservation | None,
        gesture: GestureFrame,
        status: StatusInfo,
    ) -> bool:
        cv2 = self._cv2
        view = cv2.flip(frame_bgr, 1)  # mirror so on-screen motion matches the user's
        height, width = view.shape[:2]

        m = self._margin
        cv2.rectangle(
            view,
            (int(m * width), int(m * height)),
            (int((1 - m) * width), int((1 - m) * height)),
            (120, 120, 120),
            1,
        )
        if observation is not None:
            self._draw_hand(view, observation, gesture)
        self._draw_status(view, gesture, status)

        cv2.imshow(WINDOW_NAME, view)
        self._opened = True
        key = cv2.waitKey(1) & 0xFF
        if key in _QUIT_KEYS:
            return False
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1

    def close(self) -> None:
        if self._opened:
            self._cv2.destroyWindow(WINDOW_NAME)
            self._cv2.waitKey(1)
            self._opened = False

    def _draw_hand(self, view: np.ndarray, obs: HandObservation, gesture: GestureFrame) -> None:
        cv2 = self._cv2
        height, width = view.shape[:2]
        # The view is mirrored, so x is flipped.
        pts = [(int((1.0 - x) * width), int(y * height)) for x, y, _ in obs.landmarks]
        reliable = gesture.tracking is TrackingState.TRACKING
        color = (0, 220, 0) if reliable else (0, 165, 255)
        for a, b in HAND_CONNECTIONS:
            cv2.line(view, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(view, p, 3, (255, 255, 255), -1)
        pinched = gesture.pinch is PinchState.PINCHED
        cv2.circle(view, pts[INDEX_TIP], 9, (0, 0, 255) if pinched else (255, 200, 0), 2)
        cv2.circle(view, pts[THUMB_TIP], 9, (0, 0, 255) if pinched else (255, 200, 0), 2)

    def _draw_status(self, view: np.ndarray, gesture: GestureFrame, status: StatusInfo) -> None:
        cv2 = self._cv2
        pinch = gesture.pinch.value if gesture.pinch else "-"
        ratio = f"{gesture.pinch_ratio:.2f}" if gesture.pinch_ratio is not None else "-"
        lines = [
            (f"CONTROL: {status.control}", _control_color(status.control)),
            (f"TRACKING: {gesture.tracking.value}", _tracking_color(gesture.tracking)),
            (f"PINCH: {pinch}  (ratio {ratio})", (255, 255, 255)),
            (f"FPS: {status.fps:.0f}", (255, 255, 255)),
        ]
        if status.stop_reason:
            lines.append((f"STOPPED: {status.stop_reason}", (0, 0, 255)))
        for i, (text, color) in enumerate(lines):
            cv2.putText(
                view, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3
            )
            cv2.putText(view, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        hint = f"Stop: Esc/Q here, or {status.hotkey} anywhere"
        cv2.putText(
            view, hint, (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3
        )
        cv2.putText(
            view, hint, (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )


def _control_color(control: str) -> tuple[int, int, int]:
    return {"LIVE": (0, 220, 0), "DRY-RUN": (0, 215, 255)}.get(control, (0, 0, 255))


def _tracking_color(state: TrackingState) -> tuple[int, int, int]:
    return {
        TrackingState.TRACKING: (0, 220, 0),
        TrackingState.ACQUIRING: (0, 215, 255),
        TrackingState.UNCERTAIN: (0, 165, 255),
    }.get(state, (0, 0, 255))
