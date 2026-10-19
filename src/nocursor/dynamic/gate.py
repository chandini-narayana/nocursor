"""The open-palm gate for streaming use: a window may only produce a gesture if it was an open palm.

The three target gestures are defined as open-palm movements (see ``pose.py``), but the classifiers
themselves never check the pose. In continuous use the gate is the explicit pose requirement:
the share of the window's trusted frames that are an open palm must reach ``min_palm_fraction``.

The pose is judged with the existing ``is_open_palm`` (all four non-thumb fingers clearly
extended; the thumb is ignored), so no new pose definition is introduced here. Per-frame flags are
computed once for a whole recording and then sliced per window, which gives exactly the number
``pose.open_palm_fraction`` returns for the same window (a test pins that).

Offline only: nothing here is connected to the mouse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nocursor.dynamic.pose import is_open_palm
from nocursor.dynamic.schema import SequenceRecord
from nocursor.models import HandObservation


@dataclass(frozen=True)
class GateConfig:
    """``min_palm_fraction`` of 0 switches the gate off; 1 needs every trusted frame to be open."""

    min_palm_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_palm_fraction <= 1.0):
            raise ValueError("min_palm_fraction must be within [0, 1]")

    @property
    def enabled(self) -> bool:
        return self.min_palm_fraction > 0.0


def frame_palm_flags(seq: SequenceRecord) -> np.ndarray:
    """Per frame: True where the frame is trusted and the hand is an open palm."""
    flags = np.zeros(seq.n_frames, dtype=bool)
    for i in np.flatnonzero(seq.usable):
        observation = HandObservation(seq.landmarks[i], float(seq.confidence[i]), seq.aspect_ratio)
        flags[i] = is_open_palm(observation)
    return flags


def window_palm_fraction(
    flags: np.ndarray, usable: np.ndarray, lo: int, hi: int
) -> float | None:  # fmt: skip
    """Open-palm share of the trusted frames in ``[lo, hi)``; None without a trusted frame."""
    trusted = int(usable[lo:hi].sum())
    if trusted == 0:
        return None
    return int(flags[lo:hi].sum()) / trusted


def gate_passes(fraction: float | None, config: GateConfig) -> bool:
    """A disabled gate passes everything. An enabled gate fails closed without a measurement."""
    if not config.enabled:
        return True
    if fraction is None:
        return False
    return fraction >= config.min_palm_fraction - 1e-12
