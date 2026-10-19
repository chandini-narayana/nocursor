"""Synthetic hand sequences for the dynamic-gesture tests.

Everything built here is SYNTHETIC: geometric stand-ins used to test software behaviour (feature
maths, validation, the recorder). It says nothing about how real hands or real cameras behave and
must never be used to claim gesture-recognition accuracy.

Paths are given in *view* terms: offsets in frame heights on the mirrored preview, x to the user's
right, y down. ``hand_at`` converts them to the raw, unmirrored landmark coordinates MediaPipe
would report, which is where the mirroring convention is exercised.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from helpers import make_pose_hand
from nocursor.dynamic.schema import DEFAULT_WINDOW_S, SequenceRecord
from nocursor.models import HandObservation

ViewPath = Callable[[float], tuple[float, float]]  # fraction of the take (0..1) -> view offset


def hand_at(
    offset: tuple[float, float],
    *,
    aspect: float = 4 / 3,
    hand_size: float = 0.2,
    index: str = "ext",
    middle: str = "ext",
    ring: str = "ext",
    pinky: str = "ext",
    confidence: float = 0.9,
) -> HandObservation:
    """A hand displaced by ``offset`` (frame heights) on the mirrored preview.

    The default is an open palm. The raw x shift is the negated, aspect-corrected view shift,
    because the preview mirrors the camera image.
    """
    shift = (-offset[0] / aspect, offset[1])
    return make_pose_hand(
        index, middle, ring, pinky, shift=shift, hand_size=hand_size, aspect_ratio=aspect,
        confidence=confidence,
    )  # fmt: skip


def swipe_path(direction: int, amplitude: float = 0.2) -> ViewPath:
    """Smooth sweep from one side to the other. direction +1 = the user's right, -1 = left."""

    def path(t: float) -> tuple[float, float]:
        eased = 3 * t**2 - 2 * t**3
        return (direction * amplitude * (2 * eased - 1), 0.0)

    return path


def circle_path(clockwise: bool = True, radius: float = 0.12) -> ViewPath:
    """One full loop starting at the top. Clockwise as seen on the mirrored preview (y down)."""

    def path(t: float) -> tuple[float, float]:
        angle = 2 * math.pi * t
        x = radius * math.sin(angle)
        return (x if clockwise else -x, -radius * math.cos(angle))

    return path


def still_path() -> ViewPath:
    return lambda t: (0.0, 0.0)


def make_sequence(
    path: ViewPath,
    *,
    label: str = "swipe_right",
    subtype: str | None = None,
    fps: float = 30.0,
    duration: float = DEFAULT_WINDOW_S,
    aspect: float = 4 / 3,
    hand_size: float = 0.2,
    sequence_id: str = "sess-0001",
    session_id: str = "sess",
    usable: np.ndarray | None = None,
    **pose: str,
) -> SequenceRecord:
    count = int(round(duration * fps)) + 1
    observations = [
        hand_at(path(i / (count - 1)), aspect=aspect, hand_size=hand_size, **pose)
        for i in range(count)
    ]
    landmarks = np.stack([o.landmarks for o in observations])
    flags = np.ones(count, dtype=bool) if usable is None else np.asarray(usable, dtype=bool)
    landmarks[~flags] = np.nan
    return SequenceRecord(
        sequence_id=sequence_id,
        session_id=session_id,
        label=label,
        subtype=subtype,
        t_ms=np.rint(np.arange(count) * 1000.0 / fps),
        landmarks=landmarks,
        confidence=np.where(flags, 0.9, 0.0),
        usable=flags,
        aspect_ratio=aspect,
    )
