"""Two-finger scroll: pose classification and movement-to-wheel-clicks conversion.

Pose geometry uses only MediaPipe landmarks already produced for pointing, and
only ratios of distances, so it does not depend on camera resolution or on how
far the hand is from the camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from nocursor.models import INDEX_TIP, MIDDLE_TIP, WRIST, HandObservation

# (knuckle, fingertip) landmark indices for the four non-thumb fingers.
_FINGERS = {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}

# Extension ratio = wrist->fingertip distance / wrist->knuckle distance (2D, aspect-corrected).
# A straight finger measures about 1.8-2.0; a fully curled one about 0.8-1.1. Anything
# in between is AMBIGUOUS and is never treated as the scroll pose. These defaults come
# from hand proportions, not from measurements on a real camera; verify on hardware.
EXTENDED_RATIO = 1.55
FOLDED_RATIO = 1.2


class FingerState(Enum):
    EXTENDED = "extended"
    FOLDED = "folded"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class FingerPose:
    index: FingerState
    middle: FingerState
    ring: FingerState
    pinky: FingerState

    @property
    def is_scroll_pose(self) -> bool:
        """Index and middle clearly extended, ring and little clearly folded (thumb ignored)."""
        return (
            self.index is FingerState.EXTENDED
            and self.middle is FingerState.EXTENDED
            and self.ring is FingerState.FOLDED
            and self.pinky is FingerState.FOLDED
        )

    @property
    def is_right_click_pose(self) -> bool:
        """Index, ring and little finger clearly extended; the middle finger is not judged.

        This is the pose the right-click pinch must be made in: an "OK sign" in which the thumb
        meets the middle fingertip. Requiring the other three fingers out is what separates it
        from the two natural resting shapes that also put the thumb against the middle finger:
        pointing (the other fingers curled) and the two-finger scroll pose (ring and little
        folded). The thumb is never judged.
        """
        return (
            self.index is FingerState.EXTENDED
            and self.ring is FingerState.EXTENDED
            and self.pinky is FingerState.EXTENDED
        )


def extension_ratio(observation: HandObservation, knuckle: int, tip: int) -> float:
    """Wrist-to-tip over wrist-to-knuckle distance; NaN for a degenerate hand."""
    base = observation.distance(WRIST, knuckle)
    if base <= 0.0:
        return math.nan
    return observation.distance(WRIST, tip) / base


def finger_ratios(observation: HandObservation) -> tuple[float, ...]:
    """Extension ratios for index, middle, ring and little finger (diagnostics)."""
    return tuple(
        extension_ratio(observation, knuckle, tip) for knuckle, tip in _FINGERS.values()
    )


def classify_finger(
    ratio: float, extended: float = EXTENDED_RATIO, folded: float = FOLDED_RATIO
) -> FingerState:
    if not math.isfinite(ratio):
        return FingerState.AMBIGUOUS
    if ratio >= extended:
        return FingerState.EXTENDED
    if ratio <= folded:
        return FingerState.FOLDED
    return FingerState.AMBIGUOUS


def classify_hand(
    observation: HandObservation,
    extended: float = EXTENDED_RATIO,
    folded: float = FOLDED_RATIO,
) -> FingerPose:
    states = {
        name: classify_finger(extension_ratio(observation, knuckle, tip), extended, folded)
        for name, (knuckle, tip) in _FINGERS.items()
    }
    return FingerPose(**states)


def scroll_signal(observation: HandObservation) -> float:
    """Vertical position (frame-height units, y grows downward) of the index/middle fingertips."""
    return float((observation.landmarks[INDEX_TIP, 1] + observation.landmarks[MIDDLE_TIP, 1]) / 2)


class ScrollTracker:
    """Converts smoothed vertical movement into rate-limited wheel clicks.

    Movement is measured from an anchor, in units of hand size (fixed when scrolling
    starts), so thresholds do not depend on resolution or camera distance:

    * Displacement smaller than ``dead_zone`` is ignored, so a stationary hand
      (landmark jitter) produces nothing.
    * Once it reaches the dead zone, the displacement is consumed (the anchor moves
      to the current position) and becomes ``displacement * speed`` pending clicks.
    * Output is limited to ``max_clicks`` per event and one event per
      ``interval_s``; the backlog is capped, so a flick followed by a stationary
      hand cannot keep scrolling.

    Hand moving up gives positive clicks (wheel up) unless ``invert`` is set.
    """

    def __init__(
        self,
        dead_zone: float = 0.10,
        speed: float = 8.0,
        interval_s: float = 0.05,
        max_clicks: int = 3,
        invert: bool = False,
    ) -> None:
        if dead_zone <= 0:
            raise ValueError("dead_zone must be > 0")
        if speed <= 0:
            raise ValueError("speed must be > 0")
        if interval_s < 0:
            raise ValueError("interval_s must be >= 0")
        if max_clicks < 1:
            raise ValueError("max_clicks must be >= 1")
        self._dead_zone = dead_zone
        self._speed = speed
        self._interval_s = interval_s
        self._max_clicks = max_clicks
        self._max_pending = 3.0 * max_clicks
        self._sign = -1 if invert else 1
        self.reset()

    def reset(self) -> None:
        self._anchor: float | None = None
        self._scale = 1.0
        self._pending = 0.0
        self._last_emit = -math.inf

    @property
    def active(self) -> bool:
        return self._anchor is not None

    def start(self, y: float, scale: float, now: float) -> None:
        """Begin a scroll session at vertical position ``y`` with hand size ``scale``."""
        if not (math.isfinite(y) and math.isfinite(scale) and scale > 0):
            self.reset()
            return
        self._anchor = y
        self._scale = scale
        self._pending = 0.0
        self._last_emit = now  # the first event waits one interval

    def update(self, y: float, now: float) -> int:
        """Feed the smoothed position; return signed wheel clicks to send (0 for none)."""
        if self._anchor is None or not math.isfinite(y):
            return 0
        displacement = (self._anchor - y) / self._scale  # positive = hand moved up
        if abs(displacement) >= self._dead_zone:
            self._pending += displacement * self._speed
            self._pending = max(-self._max_pending, min(self._max_pending, self._pending))
            self._anchor = y
        if now - self._last_emit < self._interval_s:
            return 0
        clicks = int(self._pending)  # truncates toward zero
        if clicks == 0:
            return 0
        clicks = max(-self._max_clicks, min(self._max_clicks, clicks))
        self._pending -= clicks
        self._last_emit = now
        return clicks * self._sign
