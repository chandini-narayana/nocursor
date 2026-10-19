"""Camera preview window: hand landmarks, the status HUD and keyboard commands.

What the HUD says is decided in ``nocursor.hud`` (pure and tested); this module only draws it.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol

import numpy as np

from nocursor.calibration import WizardState, WizardView
from nocursor.gestures import PinchState
from nocursor.hud import HudLine, StatusInfo, Tone, build_hud, help_lines
from nocursor.interaction import InteractionState
from nocursor.models import HAND_CONNECTIONS, INDEX_TIP, THUMB_TIP, HandObservation
from nocursor.processor import GestureFrame
from nocursor.tracking import TrackingState

__all__ = ["NullPreview", "OpenCVPreview", "Preview", "PreviewCommand", "StatusInfo"]

WINDOW_NAME = "NoCursor"
_QUIT_KEYS = {27, ord("q"), ord("Q")}  # Esc, q: always stop, in every mode
_HELP_KEYS = {ord("h"), ord("H"), ord("?")}  # toggles the help panel; never a command


class PreviewCommand(Enum):
    PAUSE_TOGGLE = "pause"  # P
    CALIBRATE = "calibrate"  # C
    CONFIRM = "confirm"  # Space / Enter: capture the current calibration corner
    CANCEL = "cancel"  # Backspace / X: cancel calibration
    RESET_CALIBRATION = "reset"  # R


_COMMAND_KEYS: dict[int, PreviewCommand] = {
    ord("p"): PreviewCommand.PAUSE_TOGGLE,
    ord("P"): PreviewCommand.PAUSE_TOGGLE,
    ord("c"): PreviewCommand.CALIBRATE,
    ord("C"): PreviewCommand.CALIBRATE,
    32: PreviewCommand.CONFIRM,
    13: PreviewCommand.CONFIRM,
    8: PreviewCommand.CANCEL,
    ord("x"): PreviewCommand.CANCEL,
    ord("X"): PreviewCommand.CANCEL,
    ord("r"): PreviewCommand.RESET_CALIBRATION,
    ord("R"): PreviewCommand.RESET_CALIBRATION,
}

_TONE_COLORS: dict[Tone, tuple[int, int, int]] = {  # BGR
    Tone.OK: (0, 220, 0),
    Tone.INFO: (255, 255, 255),
    Tone.PENDING: (0, 165, 255),
    Tone.ACTIVE: (60, 60, 255),
    Tone.SCROLL: (255, 255, 0),
    Tone.WARN: (0, 215, 255),
    Tone.ERROR: (60, 60, 255),
}


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

    def take_commands(self) -> tuple[PreviewCommand, ...]:
        """Keyboard commands pressed since the last call (quit keys are not included)."""
        ...

    def close(self) -> None: ...


class NullPreview:
    """No-op preview for tests and headless runs."""

    def show(
        self,
        frame_bgr: np.ndarray,
        observation: HandObservation | None,
        gesture: GestureFrame,
        status: StatusInfo,
    ) -> bool:
        return True

    def take_commands(self) -> tuple[PreviewCommand, ...]:
        return ()

    def close(self) -> None:
        pass


class OpenCVPreview:
    """Mirrored preview window. Esc/Q or closing the window requests quit."""

    def __init__(self, active_margin: float) -> None:
        import cv2

        self._cv2 = cv2
        self._margin = active_margin
        self._opened = False
        self._show_help = False
        self._commands: list[PreviewCommand] = []

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

        left, top, right, bottom = status.region or (
            self._margin, self._margin, 1 - self._margin, 1 - self._margin
        )  # fmt: skip
        cv2.rectangle(
            view,
            (int(left * width), int(top * height)),
            (int(right * width), int(bottom * height)),
            (255, 200, 0) if status.region else (120, 120, 120),
            1,
        )
        if observation is not None:
            self._draw_hand(view, observation, gesture)
        self._draw_status(view, gesture, status)
        if self._show_help:
            self._draw_help(view, status)

        cv2.imshow(WINDOW_NAME, view)
        self._opened = True
        key = cv2.waitKey(1) & 0xFF
        if key in _QUIT_KEYS:
            return False
        if key in _HELP_KEYS:
            self._show_help = not self._show_help
        command = _COMMAND_KEYS.get(key)
        if command is not None:
            self._commands.append(command)
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1

    def take_commands(self) -> tuple[PreviewCommand, ...]:
        commands, self._commands = tuple(self._commands), []
        return commands

    def close(self) -> None:
        if self._opened:
            self._cv2.destroyWindow(WINDOW_NAME)
            self._cv2.waitKey(1)
            self._opened = False

    # --------------------------------------------------------------- drawing
    def _draw_hand(self, view: np.ndarray, obs: HandObservation, gesture: GestureFrame) -> None:
        cv2 = self._cv2
        height, width = view.shape[:2]
        # The view is mirrored, so x is flipped.
        pts = [(int((1.0 - x) * width), int(y * height)) for x, y, _ in obs.landmarks]
        reliable = gesture.tracking is TrackingState.TRACKING
        color = (0, 220, 0) if reliable else (0, 165, 255)
        if gesture.state is InteractionState.SCROLL:
            color = (255, 255, 0)
        for a, b in HAND_CONNECTIONS:
            cv2.line(view, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(view, p, 3, (255, 255, 255), -1)
        # Fingertip rings: orange while a pinch is only a candidate, red once it is confirmed.
        pressed = gesture.state in (InteractionState.PRESSED, InteractionState.DRAGGING)
        pending = gesture.pinch_progress is not None or gesture.pinch is PinchState.CLOSING
        ring = (60, 60, 255) if pressed else (0, 165, 255) if pending else (255, 200, 0)
        cv2.circle(view, pts[INDEX_TIP], 9, ring, 2)
        cv2.circle(view, pts[THUMB_TIP], 9, ring, 2)

    def _draw_status(self, view: np.ndarray, gesture: GestureFrame, status: StatusInfo) -> None:
        hud = build_hud(gesture, status)
        height, width = view.shape[:2]
        rows = len(hud.details) + len(hud.notices)
        panel_bottom = 38 + 20 * rows + 6
        self._panel(view, (4, 4), (min(width - 4, 600), panel_bottom))
        self._line(view, hud.headline, (12, 28), 0.7)
        y = 54
        for line in (*hud.details, *hud.notices):
            self._line(view, line, (12, y), 0.5)
            y += 20

        y = panel_bottom + 18
        for text in status.diagnostics_lines:
            self._text(view, text, (10, y), 0.42, (255, 255, 0))
            y += 17

        self._text(view, hud.hint, (10, height - 10), 0.45, (255, 255, 255))
        if status.calibration is not None and status.calibration.state is WizardState.COLLECTING:
            self._draw_calibration(view, status.calibration)

    def _draw_help(self, view: np.ndarray, status: StatusInfo) -> None:
        lines = help_lines(status.hotkey, status.pause_hotkey, status.right_click)
        height, width = view.shape[:2]
        box_w, box_h = min(width - 40, 470), 22 * len(lines) + 16
        x0, y0 = (width - box_w) // 2, max(10, (height - box_h) // 2)
        self._panel(view, (x0, y0), (x0 + box_w, y0 + box_h), alpha=0.8)
        for i, text in enumerate(lines):
            heading = text.isupper() or text.startswith("CONTROLS")
            self._text(view, text, (x0 + 12, y0 + 26 + 22 * i), 0.5,
                       (0, 255, 255) if heading else (255, 255, 255))  # fmt: skip

    def _draw_calibration(self, view: np.ndarray, wizard: WizardView) -> None:
        cv2 = self._cv2
        height, width = view.shape[:2]
        self._text(view, wizard.title, (10, height - 70), 0.45, (0, 255, 255))
        self._text(view, wizard.instruction, (10, height - 50), 0.5, (255, 255, 255))
        if wizard.notice:
            self._text(view, wizard.notice, (10, height - 30), 0.5, (0, 165, 255))
        pts = [(int(x * width), int(y * height)) for x, y in wizard.captured]
        for p in pts:
            cv2.circle(view, p, 8, (0, 255, 0), -1)
        for a, b in zip(pts, pts[1:], strict=False):
            cv2.line(view, a, b, (0, 255, 0), 1)
        bar_w = int(120 * wizard.stability)
        cv2.rectangle(view, (width - 130, 14), (width - 10, 24), (255, 255, 255), 1)
        cv2.rectangle(
            view, (width - 130, 14), (width - 130 + bar_w, 24),
            (0, 220, 0) if wizard.stability >= 1.0 else (0, 165, 255), -1,
        )  # fmt: skip
        self._text(view, "hold still", (width - 130, 40), 0.4, (255, 255, 255))

    def _panel(
        self,
        view: np.ndarray,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        alpha: float = 0.55,
    ) -> None:
        """Darken a rectangle so text stays readable over any camera image."""
        cv2 = self._cv2
        x0, y0 = top_left
        x1, y1 = bottom_right
        x1, y1 = min(x1, view.shape[1]), min(y1, view.shape[0])
        roi = view[y0:y1, x0:x1]
        if roi.size:
            view[y0:y1, x0:x1] = cv2.addWeighted(roi, 1.0 - alpha, np.zeros_like(roi), alpha, 0)

    def _line(
        self, view: np.ndarray, line: HudLine, origin: tuple[int, int], scale: float
    ) -> None:
        self._text(view, line.text, origin, scale, _TONE_COLORS[line.tone])

    def _text(
        self,
        view: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
    ) -> None:
        cv2 = self._cv2
        cv2.putText(view, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
        cv2.putText(view, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
