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

    def update(self, observation: HandObservation | None) -> TrackingState:
        if observation is None:
            self._good_streak = 0
            return TrackingState.LOST
        if not self._is_trustworthy(observation):
            self._good_streak = 0
            return TrackingState.UNCERTAIN
        self._good_streak += 1
        if self._good_streak < self._stable_frames:
            return TrackingState.ACQUIRING
        return TrackingState.TRACKING

    def _is_trustworthy(self, observation: HandObservation) -> bool:
        xy = observation.landmarks[:, :2]
        if not np.all(np.isfinite(observation.landmarks)):
            return False
        if observation.confidence < self._min_confidence:
            return False
        if np.any(xy < 0.0) or np.any(xy > 1.0):
            return False
        return observation.hand_size >= self._min_hand_size
