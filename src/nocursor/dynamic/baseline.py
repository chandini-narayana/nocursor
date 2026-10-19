"""The rule-based baseline: a hand-written recogniser using the palm trajectory in the features.

It is the comparison the learned model has to beat, so it is built to be a fair opponent:

* It uses exactly the same versioned features as the GRU (the palm trajectory columns), so left
  and right keep the mirrored-preview convention: a negative x displacement is a swipe to the
  user's left, and a positive turning angle is clockwise as drawn on the preview (y down).
* It rejects by default. Anything that does not satisfy every criterion of a gesture is
  ``no_gesture``. It has no confidence, so it never returns ``rejected``.
* Two variants are reported. **rule_defaults** uses thresholds fixed below from the geometry of
  the gestures as described to the person recording (a swipe of about two palm-lengths in well
  under a second; a circle one to two palm-lengths across), written before any data was scored.
  **rule_tuned** adjusts them by a deterministic coordinate search (from the defaults and from the
  loosest setting, keeping the better) that maximises macro-F1 on the pooled *training and
  validation* takes only. It never sees the
  test sessions or the continuous stream.

Because the tuner reads training and validation together, it has slightly more data for choosing
its few numbers than the GRU has for gradient updates. That favours the baseline, not the model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from nocursor.dynamic.classifier import Prediction
from nocursor.dynamic.schema import CLASS_NAMES, NEGATIVE_LABEL
from nocursor.dynamic.scoring import macro_f1_of

TURN_MIN_STEP = 0.03  # hand sizes; shorter steps are jitter and have no meaningful direction
BASELINE_FILE = "baseline.json"


@dataclass(frozen=True)
class RuleParams:
    """Thresholds, in hand sizes (distance), hand sizes per second (speed) and radians (turning)."""

    swipe_min_dx: float = 1.0  # net horizontal displacement
    swipe_min_straightness: float = 0.6  # net displacement over path length
    swipe_max_dy_ratio: float = 0.6  # net vertical over net horizontal displacement
    swipe_min_peak_speed: float = 1.5  # peak horizontal speed
    swipe_max_turning: float = 2.0  # a swipe does not curl round
    circle_min_turning: float = 4.0  # about 230 degrees of clockwise turning
    circle_max_closure: float = 0.6  # a circle ends near where it started (net over path)
    circle_min_path: float = 2.5  # and is a real loop, not a tremor


# Search grids for the tuner. Small on purpose: few thresholds and coarse steps limit how far the
# baseline can overfit the data it is tuned on.
GRIDS: dict[str, tuple[float, ...]] = {
    "swipe_min_dx": (0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
    "swipe_min_straightness": (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    "swipe_max_dy_ratio": (0.3, 0.5, 0.75, 1.0, 1.5),
    "swipe_min_peak_speed": (0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
    "swipe_max_turning": (1.0, 1.5, 2.0, 3.0, 4.0),
    "circle_min_turning": (2.5, 3.0, 4.0, 4.5, 5.0, 5.5),
    "circle_max_closure": (0.3, 0.5, 0.7, 0.9, 1.2),
    "circle_min_path": (1.0, 1.5, 2.5, 3.5),
}


@dataclass(frozen=True)
class TrajectoryStats:
    net_dx: float  # signed: negative = the user's left on the mirrored preview
    net_dy: float  # signed: positive = down
    path: float
    straightness: float  # net displacement over path length (1 = straight line, 0 = closed loop)
    peak_speed_x: float
    turning: float  # signed total turning angle; positive = clockwise on the preview


def trajectory_stats(features: np.ndarray) -> TrajectoryStats:
    """Interpretable properties of the palm trajectory (feature columns 0-3)."""
    trajectory = features[:, 0:2]
    velocity = features[:, 2:4]
    steps = np.diff(trajectory, axis=0)
    path = float(np.linalg.norm(steps, axis=1).sum())
    net = trajectory[-1] - trajectory[0]
    significant = steps[np.linalg.norm(steps, axis=1) >= TURN_MIN_STEP]
    turning = 0.0
    for a, b in zip(significant[:-1], significant[1:], strict=True):
        turning += math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))
    return TrajectoryStats(
        net_dx=float(net[0]),
        net_dy=float(net[1]),
        path=path,
        straightness=float(np.linalg.norm(net)) / path if path > 1e-9 else 0.0,
        peak_speed_x=float(np.abs(velocity[:, 0]).max()),
        turning=turning,
    )


def classify_stats(stats: TrajectoryStats, params: RuleParams) -> str:
    """The decision rule. Clockwise circle first; then a straight, fast, mostly horizontal swipe."""
    if (
        stats.path >= params.circle_min_path
        and stats.turning >= params.circle_min_turning
        and stats.straightness <= params.circle_max_closure
    ):
        return "circle_clockwise"
    if (
        abs(stats.net_dx) >= params.swipe_min_dx
        and stats.straightness >= params.swipe_min_straightness
        and abs(stats.net_dy) <= params.swipe_max_dy_ratio * abs(stats.net_dx)
        and stats.peak_speed_x >= params.swipe_min_peak_speed
        and abs(stats.turning) <= params.swipe_max_turning
    ):
        return "swipe_right" if stats.net_dx > 0 else "swipe_left"
    return NEGATIVE_LABEL


class RuleBaseline:
    """A ``Classifier`` wrapping ``RuleParams``."""

    def __init__(self, params: RuleParams, name: str) -> None:
        self.params = params
        self.name = name

    def predict_features(self, features: np.ndarray) -> Prediction:
        return Prediction(label=classify_stats(trajectory_stats(features), self.params))


# ------------------------------------------------------------------------------- tuning
def _objective(stats: Sequence[TrajectoryStats], truth: np.ndarray, params: RuleParams) -> float:
    predicted = np.array([CLASS_NAMES.index(classify_stats(s, params)) for s in stats])
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    np.add.at(matrix, (truth, predicted), 1)
    return macro_f1_of(matrix)


def loosest_params() -> RuleParams:
    """The most permissive end of every grid: the second starting point of the search."""
    return RuleParams(
        swipe_min_dx=min(GRIDS["swipe_min_dx"]),
        swipe_min_straightness=min(GRIDS["swipe_min_straightness"]),
        swipe_max_dy_ratio=max(GRIDS["swipe_max_dy_ratio"]),
        swipe_min_peak_speed=min(GRIDS["swipe_min_peak_speed"]),
        swipe_max_turning=max(GRIDS["swipe_max_turning"]),
        circle_min_turning=min(GRIDS["circle_min_turning"]),
        circle_max_closure=max(GRIDS["circle_max_closure"]),
        circle_min_path=min(GRIDS["circle_min_path"]),
    )


@dataclass(frozen=True)
class TuneResult:
    params: RuleParams
    objective: float  # macro-F1 on the tuning data
    default_objective: float  # the same measure for the untuned defaults
    passes: int
    history: tuple[dict[str, Any], ...]
    start: str  # which starting point produced the result: "defaults", "loosest" or "given"


def _ascend(
    stats: Sequence[TrajectoryStats], truth: np.ndarray, start: RuleParams, max_passes: int
) -> tuple[RuleParams, float, int, list[dict[str, Any]]]:  # fmt: skip
    current = start
    best = _objective(stats, truth, current)
    history: list[dict[str, Any]] = []
    passes = 0
    for passes in range(1, max_passes + 1):
        changed = False
        for field in fields(RuleParams):
            name = field.name
            for value in GRIDS[name]:
                candidate = replace(current, **{name: value})
                score = _objective(stats, truth, candidate)
                if score > best + 1e-12:
                    history.append({"pass": passes, "parameter": name,
                                    "from": getattr(current, name), "to": value,
                                    "macro_f1": round(score, 6)})  # fmt: skip
                    current, best, changed = candidate, score, True
        if not changed:
            break
    return current, best, passes, history


def tune(
    features: Sequence[np.ndarray], labels: Sequence[str], start: RuleParams | None = None,
    max_passes: int = 8,
) -> TuneResult:  # fmt: skip
    """Coordinate search for the thresholds that maximise macro-F1 on the given takes.

    For each threshold in a fixed order it takes the grid value that most improves macro-F1 (ties
    keep the current value), until a full pass changes nothing. Because a gesture must satisfy
    several criteria at once, loosening one alone often changes no prediction, so a search from the
    defaults can stall on a plateau. The search therefore runs from two fixed starting points, the
    defaults and the loosest setting, and keeps the better result (the defaults win ties). With an
    explicit ``start`` it runs from that point only. Everything is deterministic.
    """
    if len(features) != len(labels) or not features:
        raise ValueError("need one label per feature matrix, and at least one sample")
    stats = [trajectory_stats(f) for f in features]
    truth = np.array([CLASS_NAMES.index(label) for label in labels])
    default_objective = _objective(stats, truth, RuleParams())
    starts = ([("given", start)] if start is not None
              else [("defaults", RuleParams()), ("loosest", loosest_params())])  # fmt: skip
    best: tuple[str, RuleParams, float, int, list[dict[str, Any]]] | None = None
    for name, point in starts:
        params, objective, passes, history = _ascend(stats, truth, point, max_passes)
        if best is None or objective > best[2] + 1e-12:
            best = (name, params, objective, passes, history)
    assert best is not None
    name, params, objective, passes, history = best
    return TuneResult(params, objective, default_objective, passes, tuple(history), name)


# ------------------------------------------------------------------------------ storage
def save_baseline(
    directory: Path, tuned: TuneResult, *, split_id: str, tuning_counts: dict[str, int],
    exclude_flagged: bool,
) -> Path:  # fmt: skip
    payload = {
        "split_id": split_id,
        "tuned_on": {"splits": ["train", "validation"], "class_counts": tuning_counts,
                     "flagged_takes_excluded": exclude_flagged},  # fmt: skip
        "rule_defaults": {"params": asdict(RuleParams()), "macro_f1_on_tuning_data":
                          tuned.default_objective},  # fmt: skip
        "rule_tuned": {"params": asdict(tuned.params), "macro_f1_on_tuning_data": tuned.objective,
                       "start": tuned.start, "passes": tuned.passes,
                       "changes": list(tuned.history)},  # fmt: skip
        "note": "Tuned on pooled training and validation takes only; never on the test split or "
                "the continuous stream.",  # fmt: skip
    }
    path = directory / BASELINE_FILE
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_baselines(directory: Path) -> tuple[RuleBaseline, RuleBaseline, dict[str, Any]]:
    """(defaults, tuned, raw metadata) from a run directory."""
    path = directory / BASELINE_FILE
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        defaults = RuleParams(**payload["rule_defaults"]["params"])
        tuned = RuleParams(**payload["rule_tuned"]["params"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc
    return RuleBaseline(defaults, "rule_defaults"), RuleBaseline(tuned, "rule_tuned"), payload
