"""Pinch gesture detection with hysteresis and debouncing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from nocursor.models import INDEX_TIP, MIDDLE_MCP, THUMB_TIP, WRIST, HandObservation


def pinch_ratio(observation: HandObservation) -> float:
    """Thumb-tip to index-tip distance divided by hand size, both measured in 3D.

    Using depth means fingertips that merely overlap in the image (thumb passing
    in front of or behind the index finger) do not look like a pinch. A degenerate
    zero-size hand yields ``inf``, which the detector treats as "not a pinch".
    """
    size = observation.distance_3d(WRIST, MIDDLE_MCP)
    if size <= 0.0:
        return math.inf
    return observation.distance_3d(THUMB_TIP, INDEX_TIP) / size


class PinchState(Enum):
    OPEN = "open"
    CLOSING = "closing"  # fingers are close but the pinch is not (yet) confirmed
    PINCHED = "pinched"
    REOPEN = "reopen"  # waiting for the hand to be seen open before pinches count


@dataclass(frozen=True)
class PinchResult:
    state: PinchState
    clicked: bool  # True on exactly one frame per accepted pinch
    freeze_cursor: bool  # True while fingers are close, so the click lands on target


class PinchDetector:
    """Detects a thumb-index pinch from a hand-size-normalised distance ratio.

    Parameters:
    * ``threshold``: activation level. A frame counts toward a pinch when
      ``ratio <= threshold``. Lower is stricter.
    * ``release_factor``: the pinch ends, and the cursor unfreezes, once
      ``ratio >= threshold * release_factor``.
    * ``confirm_frames``: consecutive frames at or below ``threshold`` needed
      before a pinch is accepted (at 30 fps, 5 frames is about 0.17 s).
    * ``cooldown_s``: minimum time between two clicks.

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
    ) -> None:
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
        self._last_click_time: float | None = None
        self.reset()

    def reset(self) -> None:
        """Drop pinch state (call whenever tracking is not reliable).

        The click cooldown deliberately survives a reset.
        """
        self._armed = False
        self._pinched = False
        self._candidate_frames = 0

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

        if self._pinched:
            if close:
                return PinchResult(PinchState.PINCHED, clicked=False, freeze_cursor=True)
            self._pinched = False
            self._candidate_frames = 0

        if ratio <= self._engage:
            self._candidate_frames += 1
            if self._candidate_frames >= self._confirm_frames:
                self._pinched = True
                clicked = self._cooldown_elapsed(now)
                if clicked:
                    self._last_click_time = now
                return PinchResult(PinchState.PINCHED, clicked=clicked, freeze_cursor=True)
            return PinchResult(PinchState.CLOSING, clicked=False, freeze_cursor=True)

        self._candidate_frames = 0
        state = PinchState.CLOSING if close else PinchState.OPEN
        return PinchResult(state, clicked=False, freeze_cursor=close)

    def _cooldown_elapsed(self, now: float) -> bool:
        return self._last_click_time is None or now - self._last_click_time >= self._cooldown_s
