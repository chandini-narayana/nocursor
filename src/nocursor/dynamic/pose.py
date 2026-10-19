"""The open-palm pose that the dynamic gestures are performed in.

Definition: the index, middle, ring and little fingers are all clearly extended. The thumb is not
judged. "Extended" is NoCursor's existing finger measure (wrist-to-fingertip over wrist-to-knuckle
distance in the image plane, at least ``scroll.EXTENDED_RATIO``), the same one the two-finger
scroll pose uses, so the three poses stay consistent: pointing (index only), scrolling (index and
middle, ring and little folded) and open palm (all four).

Those ratio thresholds come from hand proportions, not from measurements on a particular camera.
"""

from __future__ import annotations

import numpy as np

from nocursor.dynamic.schema import SequenceRecord
from nocursor.models import HandObservation
from nocursor.scroll import FingerState, classify_hand


def is_open_palm(observation: HandObservation) -> bool:
    pose = classify_hand(observation)
    return all(
        state is FingerState.EXTENDED for state in (pose.index, pose.middle, pose.ring, pose.pinky)
    )


def open_palm_fraction(seq: SequenceRecord) -> float | None:
    """Share of the trusted frames in which the hand is an open palm; None with no trusted frame."""
    indices = np.flatnonzero(seq.usable)
    if indices.size == 0:
        return None
    hits = sum(
        is_open_palm(HandObservation(seq.landmarks[i], float(seq.confidence[i]), seq.aspect_ratio))
        for i in indices
    )
    return hits / indices.size
