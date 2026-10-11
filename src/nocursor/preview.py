"""Camera preview window with hand landmarks, control status and keyboard commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from nocursor.calibration import WizardState, WizardView
from nocursor.gestures import PinchState
from nocursor.interaction import InteractionState
from nocursor.models import HAND_CONNECTIONS, INDEX_TIP, THUMB_TIP, HandObservation
from nocursor.processor import GestureFrame
from nocursor.tracking import TrackingState

WINDOW_NAME = "NoCursor"
_QUIT_KEYS = {27, ord("q"), ord("Q")}  # Esc, q: always stop, in every mode


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


@dataclass(frozen=True)
class StatusInfo:
    """Non-gesture state the preview should display."""

    control: str  # "LIVE", "DRY-RUN", "PAUSED", "CALIBRATING" or "STOPPED"
    fps: float
    hotkey: str
    stop_reason: str | None = None
    message: str | None = None  # transient notice (calibration saved, paused, ...)
    fault: str | None = None  # persistent error, e.g. a failed button release
    region: tuple[float, float, float, float] | None = None  # calibrated (l, t, r, b), view coords
    calibration: WizardView | None = None
    pause_hotkey: str = ""
    diagnostics_lines: tuple[str, ...] = ()  # shown only with --diagnostics


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
    """No-op preview for tests."""

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

        cv2.imshow(WINDOW_NAME, view)
        self._opened = True
        key = cv2.waitKey(1) & 0xFF
        if key in _QUIT_KEYS:
            return False
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
        pinched = gesture.pinch is PinchState.PINCHED
        cv2.circle(view, pts[INDEX_TIP], 9, (0, 0, 255) if pinched else (255, 200, 0), 2)
        cv2.circle(view, pts[THUMB_TIP], 9, (0, 0, 255) if pinched else (255, 200, 0), 2)

    def _draw_status(self, view: np.ndarray, gesture: GestureFrame, status: StatusInfo) -> None:
        pinch = _PINCH_LABELS.get(gesture.pinch, "-")
        ratio = f"{gesture.pinch_ratio:.2f}" if gesture.pinch_ratio is not None else "-"
        not_armed = gesture.pinch is PinchState.REOPEN
        lines = [
            (f"CONTROL: {status.control}", _control_color(status.control)),
            (f"STATE: {gesture.state.value}", _state_color(gesture.state)),
            (f"TRACKING: {gesture.tracking.value}", _tracking_color(gesture.tracking)),
            (
                f"PINCH: {pinch}  (ratio {ratio})",
                (0, 165, 255) if not_armed else (255, 255, 255),
            ),
            (f"FPS: {status.fps:.0f}", (255, 255, 255)),
        ]
        if status.stop_reason:
            lines.append((f"STOPPED: {status.stop_reason}", (0, 0, 255)))
        if status.fault:
            lines.append((f"FAULT: {status.fault}", (0, 0, 255)))
        if status.message:
            lines.append((status.message, (0, 255, 255)))
        for i, (text, color) in enumerate(lines):
            self._text(view, text, (10, 24 + 22 * i), 0.6, color)

        height = view.shape[0]
        top = 24 + 22 * len(lines) + 10
        for i, text in enumerate(status.diagnostics_lines):
            self._text(view, text, (10, top + 17 * i), 0.42, (255, 255, 0))
        hint = f"Stop: Esc/Q here or {status.hotkey} anywhere"
        if status.pause_hotkey:
            hint += f" | Pause: P or {status.pause_hotkey} | Calibrate: C"
        self._text(view, hint, (10, height - 10), 0.45, (255, 255, 255))

        if status.control == "PAUSED":
            self._banner(view, "PAUSED - press P to resume, then show an open hand", (0, 215, 255))
        elif gesture.state in (InteractionState.SCROLL, InteractionState.SCROLL_PENDING):
            active = gesture.state is InteractionState.SCROLL
            self._banner(
                view, "SCROLL MODE" if active else "scroll pose detected...", (255, 255, 0)
            )
        if status.calibration is not None and status.calibration.state is WizardState.COLLECTING:
            self._draw_calibration(view, status.calibration)

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

    def _banner(self, view: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
        width = view.shape[1]
        self._text(view, text, (max(10, width // 2 - 8 * len(text) // 2), 140), 0.8, color)

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


_PINCH_LABELS: dict[PinchState | None, str] = {
    PinchState.OPEN: "open",
    PinchState.CLOSING: "closing",
    PinchState.PINCHED: "pinched",
    # The detector ignores pinches until it has seen the thumb and index clearly apart.
    PinchState.REOPEN: "NOT ARMED - spread thumb and index apart",
}


def _control_color(control: str) -> tuple[int, int, int]:
    return {
        "LIVE": (0, 220, 0),
        "DRY-RUN": (0, 215, 255),
        "PAUSED": (0, 215, 255),
        "CALIBRATING": (255, 200, 0),
    }.get(control, (0, 0, 255))


def _state_color(state: InteractionState) -> tuple[int, int, int]:
    if state in (InteractionState.PRESSED, InteractionState.DRAGGING):
        return (0, 0, 255)
    if state in (InteractionState.SCROLL, InteractionState.SCROLL_PENDING):
        return (255, 255, 0)
    if state is InteractionState.POINTER:
        return (0, 220, 0)
    return (0, 215, 255)


def _tracking_color(state: TrackingState) -> tuple[int, int, int]:
    return {
        TrackingState.TRACKING: (0, 220, 0),
        TrackingState.ACQUIRING: (0, 215, 255),
        TrackingState.UNCERTAIN: (0, 165, 255),
    }.get(state, (0, 0, 255))
