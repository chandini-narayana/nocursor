"""Controlled comparison of the streaming variants on one body of development evidence.

Every variant is scored on exactly the same recordings, the same replayed takes and the same
protocol, so the differences between them are the variants and nothing else:

* **systems**: the current GRU, any separately trained candidate model, and both rule baselines;
* **mechanisms** added on top of the open-palm gate and stabilizer: the settle gate (swipes
  must have stopped), an arming rule (a still open-palm hold first), both, or neither;
* **two replay protocols**: ``rest_padded`` holds the take's first and last frame still for a few
  seconds around it (a best case that is kind to arming), and ``own_lead_only`` replays each take
  with no synthetic lead-in, so arming has only the rest the person actually left before moving.

For each variant there is a **fixed reference setting declared in this file** (nothing tuned) and
an in-sample scan of a small **declared grid**. The scan is exploratory screening, not selection:
there is a single development recording of ordinary use, validation takes drove the GRU's early
stopping, and candidates are unreviewed unless a person has marked them.

Offline only: nothing here is connected to the mouse.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from nocursor.dynamic.arming import ArmingConfig
from nocursor.dynamic.classifier import Classifier
from nocursor.dynamic.features import FeatureConfig
from nocursor.dynamic.gate import GateConfig
from nocursor.dynamic.review import tally_candidates
from nocursor.dynamic.schema import NEGATIVE_LABEL, SequenceRecord
from nocursor.dynamic.stabilizer import StabilizerConfig
from nocursor.dynamic.stream_accept import build_review
from nocursor.dynamic.stream_eval import (
    DEV_MAX_DUPLICATE_FRACTION,
    DEV_MIN_CLASS_RECALL,
    DEV_MIN_POOLED_RECALL,
    DEV_SCREEN_MAX_CROSS,
    DEV_SCREEN_PER_TAKES,
    ConfigResult,
    DecisionCache,
    DecisionConfig,
    ExtraGates,
    Recording,
    SystemCorpus,
    build_replay_cases,
    build_system_corpus,
    compute_windows,
    evaluate_config,
    meets_dev_constraints,
    pareto_summary,
    percentile,
    rate_summary,
    replay_outcomes,
    select_config,
    summarize_recall,
)

COMPARE_VERSION = 1
GATES = (0.0, 0.8, 1.0)
TAUS = (0.5, 0.9, 0.99)
VOTES = ((1, 1), (3, 5), (8, 12))
COOLDOWNS = (2.5,)
QUIETS = (0, 3)
GUARDS = (1.0,)
SETTLE_SPEEDS = (None, 0.5, 1.0, 1.5)  # hand sizes per second over the last 0.3 s of a swipe
ARMINGS: tuple[ArmingConfig | None, ...] = (
    None, ArmingConfig(0.2), ArmingConfig(0.3), ArmingConfig(0.5))  # fmt: skip

# The fixed, untuned reference setting of each variant (declared before any comparison was run).
REF_GATE = 0.8
REF_TAU_CONFIDENCE = 0.9
REF_VOTE = (3, 5)
REF_COOLDOWN = 2.5
REF_QUIET = 3
REF_GUARD = 1.0
REF_SETTLE = 1.0
REF_ARMING = ArmingConfig(0.3)

MECHANISMS = ("base", "settle", "arming", "settle+arming")
PROTOCOLS: dict[str, dict[str, Any]] = {
    "rest_padded": {"lead_s": 2.0, "tail_s": 2.5,
                    "meaning": "first and last frame of each take held still for 2.0 s before "
                               "and 2.5 s after (best case for arming)"},  # fmt: skip
    "own_lead_only": {"lead_s": 0.0, "tail_s": 2.5,
                      "meaning": "no synthetic lead-in: arming only has the rest the person "
                                 "actually left before moving"},  # fmt: skip
}
PRIMARY_PROTOCOL = "rest_padded"
SHIFT_BIN_S = 0.4
SHIFT_RANGE_S = (-1.2, 1.2)
REVIEW_SEED = 20261017


def extras_for(mechanism: str) -> ExtraGates:
    return ExtraGates(REF_SETTLE if "settle" in mechanism else None,
                      REF_ARMING if "arming" in mechanism else None)  # fmt: skip


def reference_config(system: str, has_confidence: bool, mechanism: str) -> DecisionConfig:
    tau = REF_TAU_CONFIDENCE if has_confidence else TAUS[0]
    return DecisionConfig(
        system, GateConfig(REF_GATE),
        StabilizerConfig(tau, REF_VOTE[0], REF_VOTE[1], REF_COOLDOWN, REF_QUIET, REF_GUARD),
        extras_for(mechanism),
    )  # fmt: skip


def compare_grid(system: str, has_confidence: bool) -> list[DecisionConfig]:
    taus = TAUS if has_confidence else (TAUS[0],)
    return [
        DecisionConfig(system, GateConfig(g), StabilizerConfig(t, k, n, cd, q, guard),
                       ExtraGates(settle, arming))
        for g in GATES for t in taus for (k, n) in VOTES for cd in COOLDOWNS for q in QUIETS
        for guard in GUARDS for settle in SETTLE_SPEEDS for arming in ARMINGS
    ]  # fmt: skip


def arming_configs() -> tuple[ArmingConfig, ...]:
    """Every arming rule the comparison can ask for, including the reference one."""
    return tuple(dict.fromkeys([*(a for a in ARMINGS if a is not None), REF_ARMING]))


def compare_hash() -> str:
    payload = {
        "version": COMPARE_VERSION, "gates": GATES, "taus": TAUS, "votes": VOTES,
        "cooldowns": COOLDOWNS, "quiets": QUIETS, "guards": GUARDS, "settle": SETTLE_SPEEDS,
        "arming": [None if a is None else [a.hold_s, a.still_speed, a.max_age_s, a.drop_s]
                   for a in ARMINGS],  # fmt: skip
        "reference": [REF_GATE, REF_TAU_CONFIDENCE, REF_VOTE, REF_COOLDOWN, REF_QUIET, REF_GUARD,
                      REF_SETTLE, [REF_ARMING.hold_s, REF_ARMING.still_speed]],  # fmt: skip
        "protocols": {k: [v["lead_s"], v["tail_s"]] for k, v in PROTOCOLS.items()},
        "screening": [DEV_MIN_POOLED_RECALL, DEV_MIN_CLASS_RECALL, DEV_SCREEN_MAX_CROSS,
                      DEV_SCREEN_PER_TAKES, DEV_MAX_DUPLICATE_FRACTION],  # fmt: skip
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------------------ analysis
def shift_profile(corpus: SystemCorpus, truth: str) -> dict[str, dict[str, int]]:
    """How the model reads the windows of ``truth`` takes, by where the window sits on the gesture.

    The shift is the window's start minus the take's start (negative: the window starts before
    the gesture; about 0: the window holds the whole gesture; positive: the beginning is cut off).
    Only the unshifted alignment is used. A window the model itself rejects is ``rejected``.
    """
    low, high = SHIFT_RANGE_S
    profile: dict[str, dict[str, int]] = {}
    for case, outputs in corpus.cases:
        if case.truth != truth or case.offset_s != 0.0:
            continue
        for i, start in enumerate(case.windows.starts):
            if case.windows.features[i] is None or not outputs.valid[i]:
                continue
            shift = start - case.take_start_s
            if not (low <= shift < high):
                continue
            lo = low + SHIFT_BIN_S * int((shift - low) // SHIFT_BIN_S)
            label = outputs.labels[i]
            if outputs.confidences[i] < max(0.5, outputs.floor) and label != NEGATIVE_LABEL:
                label = "rejected"
            bin_name = f"{lo:+.1f}..{lo + SHIFT_BIN_S:+.1f}"
            row = profile.setdefault(bin_name, {})
            row[label] = row.get(label, 0) + 1
    return dict(sorted(profile.items()))


def cross_diagnosis(outcomes: Sequence[dict[str, Any]], take_seconds: float = 1.6
                    ) -> dict[str, Any]:  # fmt: skip
    """Which wrong gesture fired for which true gesture, and in which part of the take."""
    pairs: dict[str, int] = {}
    phases = {"first third": 0, "middle third": 0, "last third": 0, "after the take": 0}
    for row in outcomes:
        if row["outcome"] != "cross":
            continue
        for label, after in row.get("wrong", [])[:1]:
            key = f"{row['truth']} read as {label}"
            pairs[key] = pairs.get(key, 0) + 1
            fraction = after / take_seconds
            name = ("first third" if fraction < 1 / 3 else "middle third" if fraction < 2 / 3
                    else "last third" if fraction <= 1.0 else "after the take")  # fmt: skip
            phases[name] += 1
    return {"pairs": dict(sorted(pairs.items())), "decided_in_part_of_take": phases}


def latency_summary(outcomes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = [o["latency_s"] for o in outcomes if o.get("latency_s") is not None]
    return {"n": len(values), "p50_s": percentile(values, 50), "p95_s": percentile(values, 95),
            "meaning": "seconds from the end of the take to the decision (worst alignment)"}


class FloorOverride:
    """A model with its own rejection floor replaced, so the pipeline's ``tau`` is the only one.

    The floor of a trained model is chosen by the legacy rule "smallest value at which no
    validation negative triggers a gesture". For a candidate whose validation data now includes
    partial-window negatives that rule can run to the ceiling (0.999) and make the model reject
    nearly everything. This view puts the floor back to the primary model's value, which is how
    the primary is already treated, and is reported next to the as-saved candidate, never instead.
    """

    def __init__(self, inner: Classifier, floor: float, name: str) -> None:
        self.inner = inner
        self.threshold = floor
        self.name = name

    def predict_features(self, features: Any) -> Any:
        return self.inner.predict_features(features)


# ----------------------------------------------------------------------------- the run
Recalls = dict[str, dict[str, Any]]  # protocol -> recall summary
Outcomes = dict[str, list[dict[str, Any]]]  # protocol -> per-take outcomes


@dataclass
class SystemScores:
    """Every scored setting of one system: stream candidates plus recall under each protocol."""

    system: str
    has_confidence: bool
    corpus: SystemCorpus  # the primary protocol, with the development streams
    scored: list[tuple[ConfigResult, Recalls, Outcomes]]
    references: list[DecisionConfig]
    corpora: dict[str, SystemCorpus]


def score_system(
    name: str, classifier: Classifier, has_confidence: bool, stream_windows: Sequence[Any],
    cases_by_protocol: dict[str, list[Any]], log: Callable[[str], object] = print,
) -> SystemScores:  # fmt: skip
    """Score the declared grid and the reference settings for one system under every protocol."""
    corpora = {
        protocol: build_system_corpus(name, classifier,
                                      stream_windows if protocol == PRIMARY_PROTOCOL else [],
                                      cases)
        for protocol, cases in cases_by_protocol.items()
    }  # fmt: skip
    configs = compare_grid(name, has_confidence)
    references = [reference_config(name, has_confidence, m) for m in MECHANISMS]
    known = {c.hash() for c in configs}
    configs += [r for r in references if r.hash() not in known]
    caches: dict[str, DecisionCache] = {protocol: {} for protocol in corpora}
    log(f"  {name}: scoring {len(configs)} settings under {len(corpora)} protocols")
    scored = []
    for config in configs:
        base = evaluate_config(corpora[PRIMARY_PROTOCOL], config, caches[PRIMARY_PROTOCOL])
        recalls: Recalls = {PRIMARY_PROTOCOL: base.recall}
        outcomes: Outcomes = {PRIMARY_PROTOCOL: base.outcomes}
        for protocol, corpus in corpora.items():
            if protocol != PRIMARY_PROTOCOL:
                rows = replay_outcomes(corpus, config.gate, config.stab, config.extras,
                                       caches[protocol])  # fmt: skip
                outcomes[protocol], recalls[protocol] = rows, summarize_recall(rows)
        scored.append((base, recalls, outcomes))
    return SystemScores(name, has_confidence, corpora[PRIMARY_PROTOCOL], scored, references,
                        corpora)  # fmt: skip


def _row(result: ConfigResult, recalls: Recalls, outcomes: Outcomes) -> dict[str, Any]:
    return {
        "config_hash": result.config.hash(), **result.config.to_dict(),
        "candidates": result.candidates, "candidates_per_minute": result.per_minute,
        "protocols": {
            protocol: {
                "pooled_recall": r["pooled_recall"],
                "class_recall": {c: v["recall"] for c, v in r["per_class"].items()},
                "cross_activations": r["cross_activations"], "missed": r["missed"],
                "missed_by_reason": r["missed_by_reason"],
                "takes_with_duplicates": r["takes_with_duplicate_activations"],
                "negative_takes_with_activation": r["negative_takes_with_activation"],
                "latency": latency_summary(outcomes[protocol]),
                "cross_diagnosis": cross_diagnosis(outcomes[protocol]),
            }
            for protocol, r in recalls.items()
        },
    }  # fmt: skip


def reference_result(scores: SystemScores, mechanism: str
                     ) -> tuple[ConfigResult, Recalls, Outcomes]:  # fmt: skip
    reference = next(c for c in scores.references if c.extras.mechanism() == mechanism)
    return next(item for item in scores.scored if item[0].config.hash() == reference.hash())


def summarize_variant(
    scores: SystemScores, mechanism: str, review_counts: dict[str, int]
) -> dict[str, Any]:  # fmt: skip
    """The reference setting, the screening scan and the frontier for one system and mechanism."""
    result, recalls, outcomes = reference_result(scores, mechanism)
    members = [item for item in scores.scored if item[0].config.extras.mechanism() == mechanism]
    screening: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        views = [ConfigResult(r.config, r.streams, r.candidates, r.minutes, rec[protocol])
                 for r, rec, _ in members]  # fmt: skip
        best = select_config(views)
        screening[protocol] = {
            "settings_meeting_screening": sum(1 for v in views if meets_dev_constraints(v.recall)),
            "best_in_sample": None if best is None else {
                "config_hash": best.config.hash(), **best.config.to_dict(),
                "candidates": best.candidates, "candidates_per_minute": best.per_minute,
                "pooled_recall": best.recall["pooled_recall"],
                "class_recall": {c: v["recall"] for c, v in best.recall["per_class"].items()},
                "cross_activations": best.recall["cross_activations"]},  # fmt: skip
            "frontier": pareto_summary(views, 5),
        }
    return {
        "reference": _row(result, recalls, outcomes),
        "reference_rate": rate_summary(result.candidates, result.minutes),
        "review": review_counts, "grid_settings": len(members), "screening": screening,
    }  # fmt: skip


def candidate_rows(result: ConfigResult) -> list[dict[str, Any]]:
    return [
        {"recording_id": s["recording_id"], "time_s": t, "label": label}
        for s in result.streams for t, label in zip(s["times_s"], s["labels"], strict=True)
    ]


def build_stream_windows(streams: Sequence[Recording], config: FeatureConfig) -> list[Any]:
    return [compute_windows(r.sequence, config, recording_id=r.recording_id,
                            arming_configs=arming_configs()) for r in streams]  # fmt: skip


def build_protocol_cases(takes: Sequence[SequenceRecord], config: FeatureConfig
                         ) -> dict[str, list[Any]]:  # fmt: skip
    return {
        protocol: build_replay_cases(takes, config, lead_s=spec["lead_s"], tail_s=spec["tail_s"],
                                     arming_configs=arming_configs())
        for protocol, spec in PROTOCOLS.items()
    }  # fmt: skip


def export_review(
    reference_candidates: dict[str, list[dict[str, Any]]], window_s: float
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:  # fmt: skip
    """Blind review items for the candidates of every reference setting (names stay in the key)."""
    return build_review(reference_candidates, window_s, REVIEW_SEED)


def tally_for(variant: str, rows: list[dict[str, Any]], review: dict[str, Any] | None
              ) -> dict[str, int]:  # fmt: skip
    """Review outcome counts for one variant; with no review at all, everything is unreviewed."""
    if review is None:
        return {"candidates": len(rows), "false": 0, "genuine": 0, "ambiguous": 0,
                "unreviewed": len(rows), "not_in_review_set": len(rows)}  # fmt: skip
    return tally_candidates(rows, variant, review["items"], review["key"], review["marks"])
