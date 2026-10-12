"""Per-frame stage timing for the real-time loop.

Every duration is *software* time measured with ``time.perf_counter_ns`` inside this process.
The boundaries are fixed and documented, because they decide what a number means:

``frame_wait``      time spent inside ``camera.read()``. With a webcam this includes waiting
                    for the camera to deliver the next frame, so it reflects camera pacing.
``frame_interval``  time between two consecutive frame arrivals (the inverse of the frame rate).
``inference``       ``tracker.detect()``: MediaPipe hand-landmark inference (also tagged
                    ``inference_hand`` / ``inference_no_hand`` by whether a hand was found).
``gesture``         ``processor.process()``: tracking checks, mapping, smoothing, pinch/scroll
                    detection and the interaction state machine.
``dispatch``        applying the frame's actions through ``SafeController``. In dry-run this is
                    only the controller's own bookkeeping, not an operating-system call.
``preview``         ``preview.show()``: drawing and the window event poll (not part of latency).
``end_to_end``      from the moment the frame arrives (``camera.read()`` returned) to the end
                    of dispatch. This is software processing latency. It does NOT include the
                    camera's exposure/USB/driver delay, the display, or OS input handling, so
                    it is not camera-to-display (or camera-to-action) latency.
``end_to_end_action``  the same, for frames that carried at least one desktop action.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from nocursor.metrics import RollingWindow, StatSummary, summarize

SERIES = (
    "frame_wait", "frame_interval", "inference", "inference_hand", "inference_no_hand",
    "gesture", "dispatch", "preview", "end_to_end", "end_to_end_action",
)  # fmt: skip

TRACKING_STATE = "tracking"


class NullProfiler:
    """Does nothing, at the cost of a method call. Used when no timing was requested."""

    enabled = False

    def begin_wait(self) -> None:
        pass

    def frame_arrived(self, got_frame: bool = True) -> None:
        pass

    def mark(self) -> int:
        return 0

    def record_since(self, name: str, start: int, tag: str | None = None) -> None:
        pass

    def end_dispatch(self, had_actions: bool) -> None:
        pass

    def note_tracking(self, state: str) -> None:
        pass


@dataclass
class TrackingStats:
    frames_by_state: dict[str, int] = field(default_factory=dict)
    loss_events: int = 0  # transitions from stable tracking to anything else

    @property
    def frames(self) -> int:
        return sum(self.frames_by_state.values())

    @property
    def non_tracking_fraction(self) -> float:
        total = self.frames
        if total == 0:
            return 0.0
        return 1.0 - self.frames_by_state.get(TRACKING_STATE, 0) / total


class FrameProfiler(NullProfiler):
    """Collects stage durations in rolling windows and, optionally, the whole session.

    ``keep_session=True`` also keeps every sample (capped per series) for benchmark reports;
    the live ``--show-perf`` overlay uses only the rolling windows.
    """

    enabled = True

    def __init__(
        self,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        window: int = 300,
        keep_session: bool = False,
        session_cap: int = 1_000_000,
    ) -> None:
        self._clock = clock_ns
        self._window_size = window
        self._keep_session = keep_session
        self._session_cap = session_cap
        self._windows = {name: RollingWindow(window) for name in SERIES}
        self._session: dict[str, list[float]] = {name: [] for name in SERIES}
        self._wait_start: int | None = None
        self._arrival: int | None = None
        self._first_arrival: int | None = None
        self._last_arrival: int | None = None
        self.frames = 0
        self.dropped_frames = 0
        self.tracking = TrackingStats()
        self._previous_state: str | None = None

    # ----------------------------------------------------------- frame marks
    def begin_wait(self) -> None:
        self._wait_start = self._clock()

    def frame_arrived(self, got_frame: bool = True) -> None:
        now = self._clock()
        if self._wait_start is not None:
            self._record("frame_wait", now - self._wait_start)
        if self._last_arrival is not None:
            self._record("frame_interval", now - self._last_arrival)
        if self._first_arrival is None:
            self._first_arrival = now
        self._arrival = self._last_arrival = now
        self.frames += 1
        if not got_frame:
            self.dropped_frames += 1

    def mark(self) -> int:
        return self._clock()

    def record_since(self, name: str, start: int, tag: str | None = None) -> None:
        elapsed = self._clock() - start
        self._record(name, elapsed)
        if tag is not None:
            self._record(f"{name}_{tag}", elapsed)

    def end_dispatch(self, had_actions: bool) -> None:
        if self._arrival is None:
            return
        elapsed = self._clock() - self._arrival
        self._record("end_to_end", elapsed)
        if had_actions:
            self._record("end_to_end_action", elapsed)

    def note_tracking(self, state: str) -> None:
        counts = self.tracking.frames_by_state
        counts[state] = counts.get(state, 0) + 1
        if self._previous_state == TRACKING_STATE and state != TRACKING_STATE:
            self.tracking.loss_events += 1
        self._previous_state = state

    # --------------------------------------------------------------- results
    def _record(self, name: str, elapsed_ns: int) -> None:
        ms = elapsed_ns / 1e6
        self._windows[name].add(ms)
        if self._keep_session:
            samples = self._session[name]
            if len(samples) < self._session_cap:
                samples.append(ms)

    def window_summary(self, name: str) -> StatSummary | None:
        return self._windows[name].summary()

    def session_summary(self, name: str) -> StatSummary | None:
        return summarize(self._session[name])

    def window_fps(self) -> float | None:
        """Frame rate over the rolling window, from the mean frame interval."""
        summary = self.window_summary("frame_interval")
        if summary is None or summary.mean <= 0:
            return None
        return 1000.0 / summary.mean

    @property
    def elapsed_s(self) -> float:
        """Seconds from the first to the last frame arrival (0 with fewer than 2 frames)."""
        if self._first_arrival is None or self._last_arrival is None:
            return 0.0
        return (self._last_arrival - self._first_arrival) / 1e9

    def session_fps(self) -> float | None:
        elapsed = self.elapsed_s
        if elapsed <= 0 or self.frames < 2:
            return None
        return (self.frames - 1) / elapsed

    def loss_events_per_minute(self) -> float | None:
        elapsed = self.elapsed_s
        if elapsed <= 0:
            return None
        return self.tracking.loss_events / (elapsed / 60.0)

    def overlay_line(self) -> str | None:
        """One compact line for the preview, from the rolling window. None until data exists."""
        fps = self.window_fps()
        inference = self.window_summary("inference")
        e2e = self.window_summary("end_to_end")
        if fps is None or inference is None or e2e is None:
            return None
        p95 = "n/a" if e2e.p95 is None else f"{e2e.p95:.1f}"
        return (
            f"PERF {fps:.0f} fps | inference p50 {inference.p50:.1f} ms | "
            f"software e2e p50 {e2e.p50:.1f} / p95 {p95} ms (window {e2e.n})"
        )
