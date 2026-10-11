"""Shared data types for hand tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MediaPipe hand landmark indices used by NoCursor.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_MCP = 9
NUM_LANDMARKS = 21

# Skeleton edges for the preview overlay.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)  # fmt: skip


@dataclass(frozen=True)
class HandObservation:
    """One detected hand in one camera frame.

    ``landmarks`` is a (21, 3) array of MediaPipe-normalized coordinates in the
    *unmirrored* camera image (x right, y down, both in 0..1 when inside the
    frame). ``aspect_ratio`` is frame width / height, needed so distances in x
    and y are comparable.
    """

    landmarks: np.ndarray
    confidence: float
    aspect_ratio: float

    def __post_init__(self) -> None:
        if self.landmarks.shape != (NUM_LANDMARKS, 3):
            raise ValueError(
                f"landmarks must have shape ({NUM_LANDMARKS}, 3), got {self.landmarks.shape}"
            )
        if self.aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be positive")

    def distance(self, a: int, b: int) -> float:
        """Aspect-corrected 2D distance between two landmarks (units of frame height)."""
        delta = self.landmarks[a, :2] - self.landmarks[b, :2]
        return float(np.hypot(delta[0] * self.aspect_ratio, delta[1]))

    def distance_3d(self, a: int, b: int) -> float:
        """Aspect-corrected 3D distance (units of frame height).

        MediaPipe's z uses roughly the same scale as x, so it gets the same aspect
        correction. z is relative depth and coarser than x/y.
        """
        delta = self.landmarks[a] - self.landmarks[b]
        return float(np.sqrt((delta[0] * self.aspect_ratio) ** 2 + delta[1] ** 2
                             + (delta[2] * self.aspect_ratio) ** 2))

    @property
    def hand_size(self) -> float:
        """Wrist to middle-finger-knuckle distance; used to normalise other distances."""
        return self.distance(WRIST, MIDDLE_MCP)

    @property
    def index_tip(self) -> tuple[float, float]:
        return float(self.landmarks[INDEX_TIP, 0]), float(self.landmarks[INDEX_TIP, 1])
