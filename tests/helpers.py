"""Synthetic hand observations for tests."""

from __future__ import annotations

import math

import numpy as np

from nocursor.models import (
    INDEX_TIP,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    THUMB_TIP,
    WRIST,
    HandObservation,
)

# Finger geometry for make_pose_hand: (knuckle landmark, tip landmark, fan angle in degrees,
# knuckle distance from the wrist in hand sizes).
_FINGER_GEOMETRY = {
    "index": (5, 8, -12.0, 0.95),
    "middle": (9, 12, 0.0, 1.00),
    "ring": (13, 16, 10.0, 0.95),
    "pinky": (17, 20, 22.0, 0.90),
}
# wrist->tip distance divided by wrist->knuckle distance
_EXTENSION = {"ext": 1.95, "fold": 0.9, "amb": 1.35}


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


def make_pose_hand(
    index: str = "ext",
    middle: str = "ext",
    ring: str = "fold",
    pinky: str = "fold",
    pinch_ratio: float = 1.0,
    shift: tuple[float, float] = (0.0, 0.0),
    hand_size: float = 0.2,
    confidence: float = 0.9,
    aspect_ratio: float = 1.0,
    depth: float = 0.0,
    thumb_dz: float = 0.0,
) -> HandObservation:
    """Build a hand with explicit finger states ("ext", "fold" or "amb").

    The defaults give the two-finger scroll pose with the thumb well away from the index
    finger. ``pinch_ratio`` is the thumb-index distance in hand sizes in the image plane;
    ``shift`` translates the whole hand in the image (frame fractions).

    Real-camera features that flat test hands lack: ``aspect_ratio`` (e.g. 4/3 for
    640x480), ``depth`` (landmarks get MediaPipe-style z that grows toward the camera with
    distance from the wrist, as when fingers point at the lens) and ``thumb_dz`` (the
    thumb tip's depth estimate disagreeing with the index tip's, in frame widths).
    """
    pts = np.full((NUM_LANDMARKS, 3), 0.0)
    wrist = np.array([0.5 + shift[0], 0.8 + shift[1]])
    pts[WRIST, :2] = wrist
    states = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    squash = np.array([1.0 / aspect_ratio, 1.0])  # keep distances isotropic in height units
    for name, (knuckle, tip, angle, radius) in _FINGER_GEOMETRY.items():
        direction = np.array([math.sin(math.radians(angle)), -math.cos(math.radians(angle))])
        ratio = _EXTENSION[states[name]]
        knuckle_pos = wrist + direction * radius * hand_size * squash
        tip_pos = wrist + direction * radius * ratio * hand_size * squash
        pts[knuckle, :2] = knuckle_pos
        pts[knuckle + 1, :2] = knuckle_pos + (tip_pos - knuckle_pos) * 0.4
        pts[knuckle + 2, :2] = knuckle_pos + (tip_pos - knuckle_pos) * 0.7
        pts[tip, :2] = tip_pos
        pts[knuckle, 2] = -depth * radius * 0.5
        pts[knuckle + 1, 2] = -depth * radius * (0.5 + 0.2 * ratio)
        pts[knuckle + 2, 2] = -depth * radius * (0.5 + 0.35 * ratio)
        pts[tip, 2] = -depth * radius * (0.5 + 0.5 * ratio)
    pts[1:5, :2] = wrist + np.array([0.04, -0.03]) * squash  # thumb chain; tip set below
    index_tip = pts[INDEX_TIP, :2]
    pts[THUMB_TIP, :2] = index_tip + np.array([pinch_ratio * hand_size / aspect_ratio, 0.0])
    pts[THUMB_TIP, 2] = pts[INDEX_TIP, 2] + thumb_dz
    return HandObservation(pts, confidence, aspect_ratio)
