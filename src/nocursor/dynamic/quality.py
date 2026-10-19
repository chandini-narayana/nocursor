"""Quality gates applied while recording, and label sanity review applied afterwards.

Two different things, kept apart on purpose:

* **Gates** reject a take that cannot be used at all (hand not tracked, wrong pose). They look at
  tracking quality and hand pose only, never at whether the motion "looks like" the label, so the
  dataset keeps hard, messy examples instead of only the ones a hand-written rule would accept.
* **Review** flags takes whose motion contradicts the label (for example a swipe labelled left
  that clearly went right). It only warns; nothing is deleted or relabelled automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nocursor.dynamic.features import (
    FeatureConfig,
    FeatureError,
    extract_features,
    longest_tracking_gap_s,
    motion_summary,
)
from nocursor.dynamic.pose import open_palm_fraction
from nocursor.dynamic.schema import KIND_STREAM, TARGET_CLASSES, SequenceRecord

DEFAULT_MIN_PALM_FRACTION = 0.8
ENTERING_LEAVING_MIN_USABLE = 0.3  # a hand that enters or leaves the view is lost part of the time

# Review thresholds, in hand sizes. Heuristics for flagging likely mistakes: warnings only, never
# used to filter, relabel or evaluate anything.
REVIEW_MIN_SWEEP = 0.75
REVIEW_MIN_CIRCLE_AREA = 0.2


@dataclass(frozen=True)
class QualityRules:
    min_usable_fraction: float
    max_gap_s: float | None  # None: dropouts are allowed
    min_open_palm_fraction: float | None  # None: no pose requirement


def rules_for(
    label: str,
    subtype: str | None,
    kind: str,
    config: FeatureConfig,
    min_palm_fraction: float = DEFAULT_MIN_PALM_FRACTION,
) -> QualityRules | None:
    """Gates for a take of this kind; None means only the structural checks apply."""
    if kind == KIND_STREAM:
        return None
    if label in TARGET_CLASSES:
        return QualityRules(config.min_usable_fraction, config.max_gap_s, min_palm_fraction)
    if subtype == "entering_leaving":
        return QualityRules(ENTERING_LEAVING_MIN_USABLE, None, None)
    return QualityRules(config.min_usable_fraction, None, None)


def check_quality(seq: SequenceRecord, rules: QualityRules | None) -> list[str]:
    """Why ``seq`` fails the recording gates; empty when it passes."""
    if rules is None:
        return []
    problems = []
    if seq.usable_fraction < rules.min_usable_fraction:
        problems.append(
            f"hand tracked reliably in only {seq.usable_fraction:.1%} of frames "
            f"(need {rules.min_usable_fraction:.1%}): keep the whole hand in view and well lit"
        )
    elif rules.max_gap_s is not None:
        gap = longest_tracking_gap_s((seq.t_ms - seq.t_ms[0]) / 1000.0, seq.usable)
        step = float(np.median(np.diff(seq.t_ms))) / 1000.0
        if gap > rules.max_gap_s and step > rules.max_gap_s / 2:
            problems.append(
                f"the camera delivered only about {1.0 / step:.0f} frames per second, too few "
                f"to follow a gesture (it needs a frame at least every {rules.max_gap_s:.2f} s)"
            )
        elif gap > rules.max_gap_s:
            problems.append(
                f"tracking was lost for {gap:.2f} s (limit {rules.max_gap_s:.2f} s): "
                "a very fast swipe can blur the hand; try a slightly slower one"
            )
    if rules.min_open_palm_fraction is not None:
        palm = open_palm_fraction(seq)
        if palm is None or palm < rules.min_open_palm_fraction:
            shown = "0%" if palm is None else f"{palm:.0%}"
            problems.append(
                f"open palm in only {shown} of tracked frames "
                f"(need {rules.min_open_palm_fraction:.0%}): keep all four fingers extended"
            )
    return problems


def review_label(seq: SequenceRecord, config: FeatureConfig | None = None) -> str | None:
    """A warning when the motion contradicts the label; None if fine or not reviewable."""
    if seq.kind == KIND_STREAM or seq.label not in TARGET_CLASSES:
        return None
    try:
        motion = motion_summary(extract_features(seq, config))
    except FeatureError:
        return None
    if seq.label in ("swipe_left", "swipe_right"):
        wants_left = seq.label == "swipe_left"
        went_left = motion.peak_dx < 0
        if abs(motion.peak_dx) < REVIEW_MIN_SWEEP:
            return f"sweep is small ({abs(motion.peak_dx):.2f} hand sizes) for a {seq.label}"
        if went_left != wants_left:
            went = "left" if went_left else "right"
            return f"labelled {seq.label} but the hand moved {went} on the mirrored preview"
        return None
    if abs(motion.signed_area) < REVIEW_MIN_CIRCLE_AREA:
        return f"loop is small or open (area {abs(motion.signed_area):.2f}) for a circle"
    if motion.signed_area < 0:
        return "labelled circle_clockwise but the loop is counter-clockwise on the mirrored preview"
    return None
