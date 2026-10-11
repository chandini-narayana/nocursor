"""Read-only pinch diagnostics.

Nothing in this module changes gesture behaviour or sends input. The processor builds a
``PinchDiagnostics`` record from values it already computed; ``DiagnosticsReporter`` turns
records into short preview lines and a rate-limited log line.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from nocursor.gestures import PinchSnapshot
from nocursor.interaction import InteractionState
from nocursor.tracking import TrackingState

FINGER_LABELS = ("I", "M", "R", "P")  # index, middle, ring, little
MODE_2D = "2D pinch ratio (default)"
MODE_3D = "3D pinch ratio (experimental)"


def mode_label(depth_in_use: bool) -> str:
    """Which pinch calculation is active, in the wording shown to the user."""
    return MODE_3D if depth_in_use else MODE_2D


@dataclass(frozen=True)
class PinchDiagnostics:
    tracking: TrackingState
    state: InteractionState
    reason: str  # in plain words: what the pinch logic is doing / why it is not triggering
    ratio: float | None  # the ratio the detector actually used this frame
    ratio_2d: float | None  # image-plane thumb-index ratio
    ratio_3d: float | None  # depth-aware ratio
    depth_in_use: bool
    engage: float
    release: float
    armed: bool | None
    candidate_frames: int
    confirm_frames: int
    scroll_candidate: bool
    fingers: tuple[float, ...]  # extension ratios: index, middle, ring, little


def explain(
    *,
    tracking: TrackingState,
    tracking_reason: str | None,
    acquiring: str,
    paused: bool,
    state: InteractionState,
    ratio: float | None,
    ratio_2d: float | None,
    depth_in_use: bool,
    scroll_candidate: bool,
    snapshot: PinchSnapshot | None,
    clicked: bool,
) -> str:
    """The first applicable reason a pinch is not (yet) being accepted."""
    if paused:
        return "paused or calibrating: no input is sent"
    if tracking is TrackingState.LOST:
        return "no hand detected"
    if tracking is TrackingState.UNCERTAIN:
        return f"hand rejected: {tracking_reason}"
    if tracking is TrackingState.ACQUIRING:
        return f"acquiring tracking ({acquiring} stable frames)"
    if clicked:
        return "pinch ACCEPTED this frame: button press"
    if state is InteractionState.PRESSED:
        return "pinch active: button held (pointer frozen)"
    if state is InteractionState.DRAGGING:
        return "pinch active: dragging"
    if state is InteractionState.SCROLL:
        return "scroll mode: pinch is disabled until the thumb opens"
    if scroll_candidate:
        return "two-finger pose seen: scroll pending (pinch unaffected until scroll mode starts)"
    if snapshot is None or ratio is None:
        return "waiting for data"
    if not snapshot.armed:
        if ratio < snapshot.release:
            return (
                f"NOT ARMED: open thumb and index apart "
                f"(ratio {ratio:.2f} must reach {snapshot.release:.2f})"
            )
        return "arming on this open frame"
    if snapshot.pinched:
        if snapshot.cooldown_remaining_s > 0:
            return f"pinch refused by cooldown ({snapshot.cooldown_remaining_s:.2f}s left)"
        return "pinch held"
    if ratio > snapshot.engage:
        text = f"ratio {ratio:.2f} is above the pinch threshold {snapshot.engage:.2f}"
        if depth_in_use and ratio_2d is not None and ratio_2d <= snapshot.engage:
            text += (
                f" (the 2D ratio {ratio_2d:.2f} would pass: depth is inflating it; "
                "run without --pinch-depth)"
            )
        return text
    if snapshot.candidate_frames < snapshot.confirm_frames:
        return f"confirming pinch {snapshot.candidate_frames}/{snapshot.confirm_frames}"
    return "pointing"


class DiagnosticsReporter:
    """Formats diagnostics for the preview and logs at most one line per ``interval_s``."""

    def __init__(self, interval_s: float = 1.0, window_s: float = 3.0) -> None:
        if interval_s <= 0 or window_s <= 0:
            raise ValueError("interval_s and window_s must be > 0")
        self._interval_s = interval_s
        self._window_s = window_s
        self._ratios: deque[tuple[float, float]] = deque()
        self._last_log = float("-inf")

    def min_recent_ratio(self) -> float | None:
        return min((r for _, r in self._ratios), default=None)

    def observe(self, diag: PinchDiagnostics, now: float) -> None:
        if diag.ratio is not None:
            self._ratios.append((now, diag.ratio))
        while self._ratios and now - self._ratios[0][0] > self._window_s:
            self._ratios.popleft()

    def overlay_lines(self, diag: PinchDiagnostics) -> tuple[str, ...]:
        lowest = self.min_recent_ratio()
        if diag.ratio is None:
            ratio_line = "ratio: -  (no usable hand)"
        else:
            ratio_line = (
                f"ratio {diag.ratio:.2f} (measured: 2D {_fmt(diag.ratio_2d)}, "
                f"3D {_fmt(diag.ratio_3d)})"
            )
        armed = "-" if diag.armed is None else ("YES" if diag.armed else "NO")
        fingers = " ".join(
            f"{label}{value:.1f}" for label, value in zip(FINGER_LABELS, diag.fingers, strict=False)
        )
        return (
            f"mode: {mode_label(diag.depth_in_use)}",
            ratio_line,
            f"pinch <= {diag.engage:.2f}   release >= {diag.release:.2f}   "
            f"lowest in {self._window_s:g}s: {_fmt(lowest)}",
            f"armed: {armed}   confirm {diag.candidate_frames}/{diag.confirm_frames}",
            f"scroll candidate: {'yes' if diag.scroll_candidate else 'no'}   fingers {fingers}",
            f"state: {diag.state.value}   tracking: {diag.tracking.value}",
            f"why: {diag.reason}",
        )

    def log_line(self, diag: PinchDiagnostics, now: float) -> str | None:
        """A single-line summary, or None if one was already logged within the interval."""
        if now - self._last_log < self._interval_s:
            return None
        self._last_log = now
        armed = "-" if diag.armed is None else ("yes" if diag.armed else "no")
        return (
            f"diag mode=[{mode_label(diag.depth_in_use)}] tracking={diag.tracking.value} "
            f"state={diag.state.value} "
            f"ratio={_fmt(diag.ratio)} (2D {_fmt(diag.ratio_2d)}, 3D {_fmt(diag.ratio_3d)}) "
            f"thr={diag.engage:.2f} "
            f"rel={diag.release:.2f} armed={armed} "
            f"confirm={diag.candidate_frames}/{diag.confirm_frames} "
            f"scroll_candidate={'yes' if diag.scroll_candidate else 'no'} "
            f"lowest{self._window_s:g}s={_fmt(self.min_recent_ratio())} | {diag.reason}"
        )


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"
