"""Metrics for gesture classifiers, with honest uncertainty.

Conventions (stated in every report):

* A prediction is a class name or ``rejected``. For the headline accuracy and F1 a rejection counts
  as ``no_gesture``, because it triggers nothing. The confusion matrix still has a ``rejected``
  column, and the number of true gestures lost to rejection is reported separately.
* A *false activation* is a take whose true label is ``no_gesture`` but that the system would act
  on (it predicted a target gesture and did not reject it).
* A *cross activation* is a true gesture that triggered the wrong gesture (a left swipe read as a
  right swipe): as harmful as a false activation and easy to miss in an accuracy figure.
* Intervals treat takes as independent. They are not: takes from one session share lighting,
  posture and habits, so the real uncertainty is larger than the intervals suggest.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from nocursor.dynamic.classifier import PREDICTION_LABELS, REJECTED, Prediction
from nocursor.dynamic.schema import CLASS_NAMES, NEGATIVE_LABEL, TARGET_CLASSES

Z_95 = 1.959964
N_CLASSES = len(CLASS_NAMES)


@dataclass(frozen=True)
class ScoredSample:
    true_label: str
    prediction: Prediction
    subtype: str | None = None
    sequence_id: str = ""
    flagged: bool = False


# ------------------------------------------------------------------------------ matrices
def confusion_matrix(samples: Sequence[ScoredSample]) -> np.ndarray:
    """Counts, rows = true class (CLASS_NAMES), columns = prediction (PREDICTION_LABELS)."""
    matrix = np.zeros((N_CLASSES, len(PREDICTION_LABELS)), dtype=int)
    for s in samples:
        matrix[CLASS_NAMES.index(s.true_label), PREDICTION_LABELS.index(s.prediction.label)] += 1
    return matrix


def effective_matrix(matrix: np.ndarray) -> np.ndarray:
    """Fold the ``rejected`` column into ``no_gesture`` (a rejection triggers nothing)."""
    folded = matrix[:, :N_CLASSES].copy()
    folded[:, CLASS_NAMES.index(NEGATIVE_LABEL)] += matrix[:, PREDICTION_LABELS.index(REJECTED)]
    return folded


def precision_recall_f1(matrix: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = int(matrix[i, i])
        support = int(matrix[i].sum())
        predicted = int(matrix[:, i].sum())
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else 0.0
        )
        out[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support,
                     "predicted": predicted, "true_positives": tp}  # fmt: skip
    return out


def macro_f1_of(matrix: np.ndarray) -> float:
    return float(np.mean([v["f1"] for v in precision_recall_f1(matrix).values()]))


def accuracy_of(matrix: np.ndarray) -> float:
    total = matrix.sum()
    return float(np.trace(matrix) / total) if total else 0.0


# --------------------------------------------------------------------------- intervals
def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """Wilson score interval for a proportion; None when there are no trials."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_interval(
    true_idx: np.ndarray, pred_idx: np.ndarray, statistic: Callable[[np.ndarray], float],
    seed: int, resamples: int = 2000,
) -> tuple[float, float] | None:  # fmt: skip
    """Percentile bootstrap (takes resampled with replacement) of a confusion-matrix statistic."""
    n = len(true_idx)
    if n < 2:
        return None
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for b in range(resamples):
        pick = rng.integers(0, n, n)
        matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        np.add.at(matrix, (true_idx[pick], pred_idx[pick]), 1)
        values[b] = statistic(matrix)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _poisson_cdf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return min(1.0, total)


def poisson_rate_interval(events: int, exposure: float, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Garwood) interval for an event rate: ``events`` observed over ``exposure`` units.

    With zero events the two-sided 95% upper bound is about 3.7 / exposure: two minutes without
    an event is still compatible with a true rate near 1.8 per minute.
    """
    if events < 0 or exposure <= 0:
        raise ValueError("events must be >= 0 and exposure > 0")

    def solve(target: Callable[[float], float]) -> float:
        low, high = 0.0, max(10.0, 10.0 * (events + 1))
        for _ in range(200):
            mid = (low + high) / 2
            if target(mid) > 0:
                high = mid
            else:
                low = mid
        return (low + high) / 2

    lower = 0.0
    if events > 0:  # lam such that P(X >= events) = alpha/2, i.e. P(X <= events-1) = 1 - alpha/2
        lower = solve(lambda lam: (1 - alpha / 2) - _poisson_cdf(events - 1, lam))
    upper = solve(lambda lam: alpha / 2 - _poisson_cdf(events, lam))  # P(X <= events) = alpha/2
    return lower / exposure, upper / exposure


# ------------------------------------------------------------------------------- scoring
def _interval(pair: tuple[float, float] | None) -> list[float] | None:
    return None if pair is None else [round(pair[0], 4), round(pair[1], 4)]


def score_samples(samples: Sequence[ScoredSample], seed: int = 0) -> dict:
    """Every metric the report shows, for one system on one set of samples."""
    raw = confusion_matrix(samples)
    eff = effective_matrix(raw)
    per_class = precision_recall_f1(eff)
    for name, row in per_class.items():
        i = CLASS_NAMES.index(name)
        row["recall_ci"] = _interval(wilson_interval(int(eff[i, i]), int(row["support"])))
        row["precision_ci"] = _interval(wilson_interval(int(eff[i, i]), int(row["predicted"])))

    n = len(samples)
    true_idx = np.array([CLASS_NAMES.index(s.true_label) for s in samples], dtype=int)
    pred_idx = np.array([CLASS_NAMES.index(s.prediction.effective_label) for s in samples],
                        dtype=int)  # fmt: skip

    negatives = [s for s in samples if s.true_label == NEGATIVE_LABEL]
    false_activations = [s for s in negatives if s.prediction.activates]
    by_subtype: dict[str, dict[str, int]] = {}
    for s in negatives:
        entry = by_subtype.setdefault(s.subtype or "unspecified", {"n": 0, "false_activations": 0})
        entry["n"] += 1
        entry["false_activations"] += int(s.prediction.activates)

    gestures = [s for s in samples if s.true_label in TARGET_CLASSES]
    cross = [s for s in gestures if s.prediction.activates and s.prediction.label != s.true_label]
    lost = [s for s in gestures if s.prediction.label == REJECTED]
    rejected = [s for s in samples if s.prediction.label == REJECTED]
    accuracy = accuracy_of(eff)
    macro = macro_f1_of(eff)
    return {
        "n": n,
        "class_support": {name: int(raw[i].sum()) for i, name in enumerate(CLASS_NAMES)},
        "confusion": {"rows_true": list(CLASS_NAMES), "columns_predicted": list(PREDICTION_LABELS),
                      "matrix": raw.tolist()},  # fmt: skip
        "confusion_effective": {"rows_true": list(CLASS_NAMES),
                                "columns_predicted": list(CLASS_NAMES),
                                "matrix": eff.tolist()},  # fmt: skip
        "accuracy": {"value": accuracy,
                     "ci95": _interval(wilson_interval(int(np.trace(eff)), n))},  # fmt: skip
        "macro_f1": {
            "value": macro,
            "ci95": _interval(bootstrap_interval(true_idx, pred_idx, macro_f1_of, seed)),
        },
        "per_class": per_class,
        "rejection": {
            "rejected": len(rejected),
            "rate": len(rejected) / n if n else None,
            "true_gestures_lost": len(lost),
            "true_gestures": len(gestures),
            "invalid_model_outputs": sum(1 for s in samples if not s.prediction.valid),
        },
        "false_activations": {
            "count": len(false_activations),
            "negatives": len(negatives),
            "rate": len(false_activations) / len(negatives) if negatives else None,
            "rate_ci95": _interval(wilson_interval(len(false_activations), len(negatives))),
            "by_subtype": by_subtype,
            "sequence_ids": [s.sequence_id for s in false_activations],
        },
        "cross_activations": {
            "count": len(cross), "gestures": len(gestures),
            "sequence_ids": [s.sequence_id for s in cross],
        },  # fmt: skip
    }
