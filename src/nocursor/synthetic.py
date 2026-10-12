"""Deterministic synthetic input for the software benchmark.

Nothing here comes from a camera. ``synthetic_frame`` is a fixed image with no hand in it, and
``scripted_hands`` produces landmark sets for a repeating, scripted session (pointing, a click,
a drag, a two-finger scroll, a tracking loss) so the gesture and dispatch code run through all
of their branches. Results measured on this input describe the software path only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import numpy as np

from nocursor.models import INDEX_TIP, NUM_LANDMARKS, THUMB_TIP, WRIST, HandObservation

SYNTHETIC_FPS = 30  # the virtual frame rate the gesture clock advances at

# knuckle, tip, fan angle (degrees), knuckle distance from the wrist in hand sizes
_FINGERS = {
    "index": (5, 8, -12.0, 0.95),
    "middle": (9, 12, 0.0, 1.00),
    "ring": (13, 16, 10.0, 0.95),
    "pinky": (17, 20, 22.0, 0.90),
}
_EXTENSION = {"ext": 1.95, "fold": 0.9}  # wrist->tip over wrist->knuckle distance
_HAND_SIZE = 0.2  # wrist to middle knuckle, as a fraction of frame height


def synthetic_frame(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A fixed BGR gradient with mild noise: representative image content, but no hand."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    base = np.stack([x + 0 * y, y + 0 * x, (x + y) / 2], axis=-1)
    noise = rng.normal(0.0, 6.0, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def build_hand(
    *,
    middle: str = "fold",
    ring: str = "fold",
    pinky: str = "fold",
    pinch_ratio: float = 1.0,
    shift: tuple[float, float] = (0.0, 0.0),
    aspect_ratio: float = 4 / 3,
) -> HandObservation:
    """A hand with the index finger extended and the other fingers as given.

    ``pinch_ratio`` is the thumb-index gap in hand sizes (image plane); ``shift`` moves the
    whole hand in the frame. Depth (z) is zero, which is all the 2D pinch ratio needs.
    """
    pts = np.zeros((NUM_LANDMARKS, 3))
    wrist = np.array([0.5 + shift[0], 0.8 + shift[1]])
    pts[WRIST, :2] = wrist
    squash = np.array([1.0 / aspect_ratio, 1.0])
    states = {"index": "ext", "middle": middle, "ring": ring, "pinky": pinky}
    for name, (knuckle, tip, angle, radius) in _FINGERS.items():
        direction = np.array([math.sin(math.radians(angle)), -math.cos(math.radians(angle))])
        ratio = _EXTENSION[states[name]]
        knuckle_pos = wrist + direction * radius * _HAND_SIZE * squash
        tip_pos = wrist + direction * radius * ratio * _HAND_SIZE * squash
        pts[knuckle, :2] = knuckle_pos
        pts[knuckle + 1, :2] = knuckle_pos + (tip_pos - knuckle_pos) * 0.4
        pts[knuckle + 2, :2] = knuckle_pos + (tip_pos - knuckle_pos) * 0.7
        pts[tip, :2] = tip_pos
    pts[1:5, :2] = wrist + np.array([0.04, -0.03]) * squash
    pts[THUMB_TIP, :2] = pts[INDEX_TIP, :2] + np.array([pinch_ratio * _HAND_SIZE / aspect_ratio, 0])
    return HandObservation(pts, confidence=0.95, aspect_ratio=aspect_ratio)


def _phases(aspect: float) -> list[tuple[int, Callable[[int], HandObservation | None]]]:
    """(frame count, builder(i)) for each phase of one scripted cycle."""

    def pointing(i: int) -> HandObservation:  # open hand, fingertip circling
        angle = i / 6
        return build_hand(shift=(0.06 * math.cos(angle), 0.05 * math.sin(angle)),
                          aspect_ratio=aspect)  # fmt: skip

    def pinch(i: int) -> HandObservation:  # thumb and index touching, hand still
        return build_hand(pinch_ratio=0.10, aspect_ratio=aspect)

    def drag(i: int) -> HandObservation:  # touching, hand sliding left in the image
        return build_hand(pinch_ratio=0.10, shift=(-0.004 * i, 0.0), aspect_ratio=aspect)

    def scroll_still(i: int) -> HandObservation:
        return build_hand(middle="ext", aspect_ratio=aspect)

    def scroll_up(i: int) -> HandObservation:
        return build_hand(middle="ext", shift=(0.0, -0.006 * i), aspect_ratio=aspect)

    def scroll_down(i: int) -> HandObservation:
        return build_hand(middle="ext", shift=(0.0, -0.18 + 0.006 * i), aspect_ratio=aspect)

    def lost(i: int) -> None:
        return None

    return [
        (14, pointing), (12, pinch), (10, pointing),  # lock on, click, release
        (6, pointing), (8, pinch), (30, drag), (8, pointing),  # drag
        (14, scroll_still), (30, scroll_up), (30, scroll_down), (6, pointing),  # scroll
        (12, lost),  # tracking loss, then the cycle repeats
    ]


def scripted_hands(frames: int, aspect: float = 4 / 3) -> Iterator[HandObservation | None]:
    """Yield ``frames`` landmark sets from the repeating scripted session."""
    phases = _phases(aspect)
    produced = 0
    while produced < frames:
        for count, builder in phases:
            for i in range(count):
                if produced >= frames:
                    return
                yield builder(i)
                produced += 1
