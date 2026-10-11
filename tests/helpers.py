"""Synthetic hand observations for tests."""

from __future__ import annotations

import numpy as np

from nocursor.models import (
    INDEX_TIP,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    THUMB_TIP,
    WRIST,
    HandObservation,
)


def make_hand(
    index_xy: tuple[float, float] = (0.5, 0.5),
    pinch_ratio: float = 1.0,
    confidence: float = 0.9,
    hand_size: float = 0.2,
    aspect_ratio: float = 1.0,
    thumb_dz: float = 0.0,
) -> HandObservation:
    """Build a hand whose thumb-index image distance / hand size equals ``pinch_ratio``.

    With ``aspect_ratio=1`` the geometry is exact; the wrist/knuckle pair is placed
    vertically so ``hand_size`` is the knuckle distance regardless of aspect.
    ``thumb_dz`` offsets the thumb tip in depth (MediaPipe z) without changing its
    position in the image.
    """
    pts = np.full((NUM_LANDMARKS, 3), 0.5)
    pts[WRIST] = (0.5, 0.8, 0.0)
    pts[MIDDLE_MCP] = (0.5, 0.8 - hand_size, 0.0)
    pts[INDEX_TIP] = (index_xy[0], index_xy[1], 0.0)
    gap_x = pinch_ratio * hand_size / aspect_ratio
    pts[THUMB_TIP] = (index_xy[0] + gap_x, index_xy[1], thumb_dz)
    return HandObservation(pts, confidence, aspect_ratio)
