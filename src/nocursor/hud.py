"""What the preview says, as plain data: the text and tone of every line.

Keeping this separate from OpenCV drawing makes the wording testable. The key rule: a gesture
*candidate* (pinch pending, scroll pose seen) is never described as an action. "PRESSED" and
"DRAGGING" appear only when the controller really holds the button (or, in dry-run, is
simulating one), and a toast such as "Button pressed" appears only after the controller
accepted that event.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nocursor.calibration import WizardView
from nocursor.gestures import PinchState
from nocursor.interaction import InteractionState
from nocursor.processor import GestureFrame
from nocursor.tracking import TrackingState


class Tone(Enum):
    OK = "ok"  # everything fine, input available
    INFO = "info"
    PENDING = "pending"  # a candidate that has not triggered anything
    ACTIVE = "active"  # a confirmed press / drag
    SCROLL = "scroll"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class HudLine:
    text: str
    tone: Tone = Tone.INFO


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
    pinch_mode: str = ""  # "2D pinch ratio (default)" or "3D pinch ratio (experimental)"
    pinch_threshold: float | None = None
    perf_line: str | None = None  # shown only with --show-perf
    toast: str | None = None  # a short-lived note about an event the controller accepted
    button_held: bool = False  # the controller holds (or, in dry-run, simulates) a press
    right_click: bool = False  # --right-click is on (only changes the help panel)


@dataclass(frozen=True)
class Hud:
    headline: HudLine
    details: tuple[HudLine, ...]  # control, tracking, pinch, performance
    notices: tuple[HudLine, ...]  # toast, message, fault, stop reason
    hint: str


CONTROL_TEXT = {
    "LIVE": "LIVE (mouse input is sent)",
    "DRY-RUN": "DRY-RUN (no input is sent)",
    "PAUSED": "PAUSED (no input is sent)",
    "CALIBRATING": "CALIBRATING (no input is sent)",
    "STOPPED": "STOPPED",
}

TOAST_TEXT = {
    "press": "Button pressed",
    "click": "Released: click",
    "drag_end": "Released: drag finished",
    "right_click": "Right-clicked",
}


def toast_text(event: str, simulated: bool) -> str:
    base = TOAST_TEXT.get(event) or f"Scrolled {event.removeprefix('scroll ')}"
    return f"{base} (simulated)" if simulated else base


def build_hud(gesture: GestureFrame, status: StatusInfo) -> Hud:
    return Hud(
        headline=_headline(gesture, status),
        details=tuple(_details(gesture, status)),
        notices=tuple(_notices(status)),
        hint=_hint(status),
    )


def _headline(g: GestureFrame, s: StatusInfo) -> HudLine:
    simulated = " (simulated)" if s.control == "DRY-RUN" else ""
    if s.control == "STOPPED":
        reason = f" - {s.stop_reason}" if s.stop_reason else ""
        return HudLine(f"STOPPED{reason}", Tone.ERROR)
    if s.control == "CALIBRATING":
        wizard = s.calibration
        corner = f" - corner {wizard.step}/{wizard.total}" if wizard is not None else ""
        return HudLine(f"CALIBRATING{corner}", Tone.INFO)
    if s.control == "PAUSED":
        return HudLine("PAUSED - no input is sent. Press P to resume", Tone.WARN)

    if g.tracking is TrackingState.LOST:
        return HudLine("NO HAND - show one hand to the camera", Tone.WARN)
    if g.tracking is TrackingState.UNCERTAIN:
        why = g.tracking_note or "tracking is not reliable"
        return HudLine(f"HAND NOT USABLE - {why}", Tone.WARN)
    if g.tracking is TrackingState.ACQUIRING:
        progress = f" ({g.tracking_note})" if g.tracking_note else ""
        return HudLine(f"LOCKING ON{progress} - hold your hand steady", Tone.INFO)

    state = g.state
    if state is InteractionState.SCROLL:
        return HudLine("SCROLLING - move your hand up or down", Tone.SCROLL)
    if state is InteractionState.SCROLL_PENDING:
        return HudLine("SCROLL POSE - hold still to start scrolling", Tone.PENDING)
    if state in (InteractionState.PRESSED, InteractionState.DRAGGING):
        if not s.button_held:  # the gesture logic wanted a press, the controller has none
            return HudLine("PRESS NOT SENT - input is blocked", Tone.ERROR)
        verb = "PRESSED" if state is InteractionState.PRESSED else "DRAGGING"
        return HudLine(f"{verb} - button held{simulated}", Tone.ACTIVE)
    if g.right_progress is not None:  # a candidate, not an action
        done, needed = g.right_progress
        return HudLine(f"RIGHT-CLICK PENDING {done}/{needed} - not clicked yet", Tone.PENDING)
    if g.right_pinch is PinchState.PINCHED:
        return HudLine("RIGHT PINCH HELD - release, then pinch again", Tone.INFO)
    if g.pinch is PinchState.REOPEN or state is InteractionState.REARMING:
        return HudLine("NOT ARMED - spread thumb and index apart", Tone.WARN)
    if state is InteractionState.PINCH_PENDING:
        if g.pinch_progress is not None:
            done, needed = g.pinch_progress
            return HudLine(f"PINCH PENDING {done}/{needed} - not clicked yet", Tone.PENDING)
        if g.pinch is PinchState.PINCHED:  # held but refused: cooldown
            return HudLine("PINCH IGNORED - release, then pinch again", Tone.WARN)
        return HudLine("PINCH PENDING - not clicked yet", Tone.PENDING)
    if state is InteractionState.POINTER:
        return HudLine("POINTING", Tone.OK)
    return HudLine("SEARCHING", Tone.WARN)


def _details(g: GestureFrame, s: StatusInfo) -> list[HudLine]:
    lines = [
        HudLine(f"Control: {CONTROL_TEXT.get(s.control, s.control)}",
                {"LIVE": Tone.OK, "DRY-RUN": Tone.PENDING}.get(s.control, Tone.WARN)),
        HudLine(_tracking_text(g), _tracking_tone(g.tracking)),
    ]  # fmt: skip
    ratio = f"{g.pinch_ratio:.2f}" if g.pinch_ratio is not None else "-"
    threshold = f"  (pinch at <= {s.pinch_threshold:.2f})" if s.pinch_threshold is not None else ""
    mode = s.pinch_mode or "pinch ratio"
    lines.append(HudLine(f"{mode}: {ratio}{threshold}", Tone.INFO))
    lines.append(HudLine(s.perf_line or f"FPS {s.fps:.0f}", Tone.INFO))
    return lines


def _tracking_text(g: GestureFrame) -> str:
    t = g.tracking
    if t is TrackingState.TRACKING:
        return "Tracking: stable"
    if t is TrackingState.ACQUIRING:
        return f"Tracking: locking on {g.tracking_note or ''}".rstrip()
    if t is TrackingState.UNCERTAIN:
        return f"Tracking: unreliable - {g.tracking_note or 'not trusted this frame'}"
    return "Tracking: no hand"


def _tracking_tone(t: TrackingState) -> Tone:
    return {
        TrackingState.TRACKING: Tone.OK,
        TrackingState.ACQUIRING: Tone.INFO,
        TrackingState.UNCERTAIN: Tone.WARN,
    }.get(t, Tone.ERROR)


def _notices(s: StatusInfo) -> list[HudLine]:
    notices: list[HudLine] = []
    if s.toast:
        notices.append(HudLine(s.toast, Tone.ACTIVE))
    if s.fault:
        notices.append(HudLine(f"FAULT: {s.fault}", Tone.ERROR))
    if s.message:
        notices.append(HudLine(s.message, Tone.INFO))
    return notices


def _hint(s: StatusInfo) -> str:
    parts = [f"Stop: Esc/Q or {s.hotkey}"]
    if s.pause_hotkey:
        parts.append(f"Pause: P / {s.pause_hotkey}")
    parts += ["Calibrate: C", "Help: H"]
    return "  |  ".join(parts)


def help_lines(hotkey: str, pause_hotkey: str, right_click: bool = False) -> tuple[str, ...]:
    """The controls panel shown with the H key."""
    extra = (
        ("Thumb to middle tip (index, ring, little out): right-click",) if right_click else ()
    )
    return (
        "CONTROLS (press H to hide)",
        "Esc / Q ........... stop (always works)",
        f"{hotkey} ... stop from any window",
        f"P / {pause_hotkey or 'ctrl+alt+p'} ... pause / resume",
        "C .................. calibrate (4 corners)",
        "Space / Enter ...... capture a calibration corner",
        "Backspace / X ...... cancel calibration",
        "R .................. reset calibration",
        "GESTURES",
        "Index fingertip .... move the pointer",
        "Pinch + release .... click (hold ~0.2 s first)",
        "Pinch + move hand .. drag",
        "Index+middle up, ring+little folded: scroll",
        *extra,
    )
