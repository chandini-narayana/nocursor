"""Decide whether hand tracking is reliable enough to act on."""

from __future__ import annotations

from enum import Enum

import numpy as np

from nocursor.models import HandObservation

_MIN_USABLE_HAND_SIZE = 1e-6


class TrackingState(Enum):
    LOST = "lost"  # no hand detected
    UNCERTAIN = "uncertain"  # hand detected but not trustworthy this frame
    ACQUIRING = "acquiring"  # trustworthy, but not yet for enough consecutive frames
    TRACKING = "tracking"  # reliable: desktop actions are allowed


class TrackingMonitor:
    """Classifies each frame; only ``TRACKING`` permits desktop actions.

    A frame is trustworthy when the hand's confidence is high enough, every
    landmark is finite and inside the camera frame, and the hand is not
    implausibly small. The hand must be trustworthy for ``stable_frames``
    consecutive frames before ``TRACKING`` is reported, and any bad frame
    restarts that count.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        min_hand_size: float = 0.05,
        stable_frames: int = 3,
    ) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        self._min_confidence = min_confidence
        self._min_hand_size = max(min_hand_size, _MIN_USABLE_HAND_SIZE)
        self._stable_frames = stable_frames
        self._good_streak = 0
        self._last_reason: str | None = None

    @property
    def stable_frames(self) -> int:
        return self._stable_frames

    @property
    def good_streak(self) -> int:
        """Consecutive trustworthy frames so far (diagnostics)."""
        return self._good_streak

    @property
    def last_reason(self) -> str | None:
        """Why the latest frame was not trusted (None if it was, or no hand). Diagnostics only."""
        return self._last_reason

    def reset(self) -> None:
        """Forget the stability streak; tracking must be re-established from scratch."""
        self._good_streak = 0

    def update(self, observation: HandObservation | None) -> TrackingState:
        self._last_reason = None
        if observation is None:
            self._good_streak = 0
            return TrackingState.LOST
        self._last_reason = self._untrusted_reason(observation)
        if self._last_reason is not None:
            self._good_streak = 0
            return TrackingState.UNCERTAIN
        self._good_streak += 1
        if self._good_streak < self._stable_frames:
            return TrackingState.ACQUIRING
        return TrackingState.TRACKING

    def _untrusted_reason(self, observation: HandObservation) -> str | None:
        """None if the frame is trustworthy, otherwise the first failed check."""
        xy = observation.landmarks[:, :2]
        if not np.all(np.isfinite(observation.landmarks)):
            return "non-finite landmarks"
        if observation.confidence < self._min_confidence:
            return f"confidence {observation.confidence:.2f} < {self._min_confidence:.2f}"
        if np.any(xy < 0.0) or np.any(xy > 1.0):
            return "a landmark is outside the camera frame (hand partly out of view)"
        if observation.hand_size < self._min_hand_size:
            return f"hand too small ({observation.hand_size:.3f} < {self._min_hand_size:.3f})"
        return None
