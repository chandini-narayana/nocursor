"""Pinch gesture detection with hysteresis and debouncing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from nocursor.models import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST, HandObservation

# Right-click only: the thumb must be nearer the middle fingertip than the index fingertip by
# at least this much (in hand sizes). Keeps the thumb-index pinch (left click) and the
# thumb-middle pinch (right click) from ever being the same pinch.
RIGHT_CLICK_INDEX_MARGIN = 0.10


def pinch_ratio(
    observation: HandObservation, use_depth: bool = False, tip: int = INDEX_TIP
) -> float:
    """Thumb-tip to ``tip`` distance divided by hand size (wrist to middle knuckle).

    ``tip`` is the index fingertip (the left-click pinch) unless the right-click pinch asks for the
    middle fingertip.

    The default is the image-plane (2D) ratio: on real webcams MediaPipe's depth (z)
    estimate disagrees between the two fingertips by enough to keep touching fingers
    above the pinch threshold. ``use_depth=True`` is the experimental 3D ratio, which
    additionally rejects fingertips that merely overlap in the image. A degenerate
    zero-size hand yields ``inf``, which the detector treats as "not a pinch".
    """
    if use_depth:
        size = observation.distance_3d(WRIST, MIDDLE_MCP)
        if size <= 0.0:
            return math.inf
        return observation.distance_3d(THUMB_TIP, tip) / size
    return pinch_ratio_2d(observation, tip)


def pinch_ratio_2d(observation: HandObservation, tip: int = INDEX_TIP) -> float:
    """The image-plane thumb-to-``tip`` ratio (ignores MediaPipe's z estimate)."""
    size = observation.hand_size
    if size <= 0.0:
        return math.inf
    return observation.distance(THUMB_TIP, tip) / size


class PinchState(Enum):
    OPEN = "open"
    CLOSING = "closing"  # fingers are close but the pinch is not (yet) confirmed
    PINCHED = "pinched"
    REOPEN = "reopen"  # waiting for the hand to be seen open before pinches count


@dataclass(frozen=True)
class PinchResult:
    state: PinchState
    clicked: bool  # True on exactly one frame per accepted pinch (the confirmed pinch-down)
    freeze_cursor: bool  # True while fingers are close, so the click lands on target
    held: bool = False  # a confirmed pinch is active, whether or not the cooldown accepted it
    released: bool = False  # True on exactly one frame, when an active pinch is confirmed ended


@dataclass(frozen=True)
class PinchSnapshot:
    """Read-only view of the detector's internal state, for diagnostics only."""

    armed: bool  # False: an open hand (ratio >= release) must be seen before a pinch counts
    pinched: bool
    candidate_frames: int  # consecutive frames at/below the activation threshold so far
    confirm_frames: int
    open_frames: int  # consecutive open frames toward confirming a release
    release_frames: int
    engage: float
    release: float
    cooldown_remaining_s: float


class PinchDetector:
    """Detects a thumb-index pinch from a hand-size-normalised distance ratio.

    Parameters:
    * ``threshold``: activation level. A frame counts toward a pinch when
      ``ratio <= threshold``. Lower is stricter.
    * ``release_factor``: the pinch ends, and the cursor unfreezes, once
      ``ratio >= threshold * release_factor``.
    * ``confirm_frames``: consecutive frames at or below ``threshold`` needed
      before a pinch is accepted (at 30 fps, 5 frames is about 0.17 s).
    * ``cooldown_s``: minimum time between two accepted pinches.
    * ``release_frames``: consecutive frames at or above the release level
      needed before an active pinch is confirmed ended. This stops a single
      glitchy frame from dropping a drag. The default of 1 releases immediately.

    Anti-misfire measures:
    * Hysteresis: between ``threshold`` and the release level an existing pinch
      is held and a new one cannot start, so jitter near the threshold cannot
      produce repeated pinches.
    * Confirmation: the ratio must stay at or below the threshold for
      ``confirm_frames`` consecutive frames.
    * One click per pinch: holding a pinch never repeats the click.
    * Cooldown: a pinch confirmed within ``cooldown_s`` of the previous click
      is consumed without clicking.
    * Re-arming: after construction or ``reset()`` (tracking loss) no click can
      fire until the fingers have been seen open.
    """

    def __init__(
        self,
        threshold: float = 0.20,
        release_factor: float = 1.5,
        confirm_frames: int = 5,
        cooldown_s: float = 0.4,
        release_frames: int = 1,
    ) -> None:
        if release_frames < 1:
            raise ValueError("release_frames must be >= 1")
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        if release_factor <= 1:
            raise ValueError("release_factor must be > 1")
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
        if cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")
        self._engage = threshold
        self._release = threshold * release_factor
        self._confirm_frames = confirm_frames
        self._cooldown_s = cooldown_s
        self._release_frames = release_frames
        self._last_click_time: float | None = None
        self.reset()

    def reset(self) -> None:
        """Drop pinch state (call whenever tracking is not reliable).

        The click cooldown deliberately survives a reset.
        """
        self._armed = False
        self._pinched = False
        self._candidate_frames = 0
        self._open_frames = 0

    def snapshot(self, now: float) -> PinchSnapshot:
        """Current internal state. Pure read: calling it never changes behaviour."""
        remaining = 0.0
        if self._last_click_time is not None:
            remaining = max(0.0, self._cooldown_s - (now - self._last_click_time))
        return PinchSnapshot(
            armed=self._armed,
            pinched=self._pinched,
            candidate_frames=self._candidate_frames,
            confirm_frames=self._confirm_frames,
            open_frames=self._open_frames,
            release_frames=self._release_frames,
            engage=self._engage,
            release=self._release,
            cooldown_remaining_s=remaining,
        )

    def update(self, ratio: float, now: float) -> PinchResult:
        """Feed one frame's thumb-index distance / hand size and a monotonic time."""
        if not math.isfinite(ratio) or ratio < 0:
            self.reset()
            return PinchResult(PinchState.REOPEN, clicked=False, freeze_cursor=True)

        close = ratio < self._release

        if not self._armed:
            if close:
                return PinchResult(PinchState.REOPEN, clicked=False, freeze_cursor=True)
            self._armed = True

        released = False
        if self._pinched:
            if close:
                self._open_frames = 0
                return PinchResult(PinchState.PINCHED, False, True, held=True)
            self._open_frames += 1
            if self._open_frames < self._release_frames:  # release not yet confirmed
                return PinchResult(PinchState.PINCHED, False, True, held=True)
            self._pinched = False
            self._open_frames = 0
            self._candidate_frames = 0
            released = True

        if ratio <= self._engage:
            self._candidate_frames += 1
            if self._candidate_frames >= self._confirm_frames:
                self._pinched = True
                clicked = self._cooldown_elapsed(now)
                if clicked:
                    self._last_click_time = now
                return PinchResult(PinchState.PINCHED, clicked, True, held=True)
            return PinchResult(PinchState.CLOSING, clicked=False, freeze_cursor=True)

        self._candidate_frames = 0
        state = PinchState.CLOSING if close else PinchState.OPEN
        return PinchResult(state, clicked=False, freeze_cursor=close, released=released)

    def _cooldown_elapsed(self, now: float) -> bool:
        return self._last_click_time is None or now - self._last_click_time >= self._cooldown_s
