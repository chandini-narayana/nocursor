"""The interface every gesture classifier (rule baseline, GRU) presents to the evaluation code.

Both systems take the same ``(steps, FEATURE_DIM)`` feature matrix and return a ``Prediction``, so
they are scored by the same code on the same samples. A prediction is one of the four class names
or ``REJECTED``: "not confident enough to act", which triggers nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from nocursor.dynamic.features import FeatureConfig, FeatureError, extract_features
from nocursor.dynamic.schema import CLASS_NAMES, NEGATIVE_LABEL, TARGET_CLASSES, SequenceRecord

REJECTED = "rejected"
PREDICTION_LABELS = (*CLASS_NAMES, REJECTED)


@dataclass(frozen=True)
class Prediction:
    label: str  # a class name, or REJECTED
    probabilities: tuple[float, ...] | None = None  # in CLASS_NAMES order (learned model only)
    note: str | None = None  # why it was rejected, when that was not low confidence
    valid: bool = True  # False when the model produced unusable output (always REJECTED)

    @property
    def effective_label(self) -> str:
        """What actually happens: a rejection triggers nothing, exactly like ``no_gesture``."""
        return NEGATIVE_LABEL if self.label == REJECTED else self.label

    @property
    def activates(self) -> bool:
        """True if this prediction would trigger a gesture."""
        return self.label in TARGET_CLASSES


class Classifier(Protocol):
    name: str

    def predict_features(self, features: np.ndarray) -> Prediction: ...


def predict_sequence(
    classifier: Classifier, seq: SequenceRecord, config: FeatureConfig
) -> Prediction | None:
    """Features plus prediction for one sequence; None if it cannot be turned into features."""
    try:
        features = extract_features(seq, config)
    except FeatureError:
        return None
    return classifier.predict_features(features)
