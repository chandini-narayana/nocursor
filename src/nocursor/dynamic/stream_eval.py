"""Offline streaming evaluation: gate + temporal stabilisation, scored on long recordings.

The earlier ``evaluate`` counts flagged sliding windows on one reserved recording. This module
scores a *decision pipeline* (classifier -> open-palm gate -> stabilizer) in terms of discrete
activations, and measures what it costs in genuine-gesture recall:

* ``compute_windows`` cuts a recording into the same 1.6 s windows (stride 0.1 s) as ``evaluate``
  and caches, per window, its features and open-palm fraction, so a grid of decision settings can
  be scored without running the classifiers again.
* ``replay`` cases take an isolated, labelled take, hold its first and last frames still for a
  few seconds on each side, and slide the same windows over it at several alignments. A take only
  counts as recalled if it is recalled at *every* alignment (conservative), and as a cross
  activation if it triggers a wrong gesture at *any* alignment. The padding is cleaner than live
  use: the hand rests perfectly still before and after the movement.
* ``select_config`` / ``leave_one_recording_out`` choose a decision setting from development data
  only, with recording-level separation. Nothing here opens the sealed final data.

Everything is deterministic. Offline only: nothing is connected to the mouse.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nocursor.dynamic.arming import ArmingConfig, armed_series, motion_onset_step
from nocursor.dynamic.classifier import REJECTED, Classifier
from nocursor.dynamic.dataset import load_session
from nocursor.dynamic.evaluate import (
    MIN_PERSISTENT_WINDOWS,
    WINDOW_STRIDE_S,
    _events,
)
from nocursor.dynamic.features import (
    FeatureConfig,
    FeatureError,
    extract_features,
    failure_category,
)
from nocursor.dynamic.gate import GateConfig, frame_palm_flags, gate_passes, window_palm_fraction
from nocursor.dynamic.schema import (
    CLASS_NAMES,
    KIND_STREAM,
    MIN_FRAMES,
    NEGATIVE_LABEL,
    TARGET_CLASSES,
    SequenceRecord,
)
from nocursor.dynamic.scoring import _poisson_cdf, poisson_rate_interval, wilson_interval
from nocursor.dynamic.stabilizer import (
    Activation,
    StabilizerConfig,
    WindowDecision,
    run_stabilizer,
)

STRIDE_S = WINDOW_STRIDE_S
REPLAY_LEAD_S = 2.0
REPLAY_TAIL_S = 2.5
REPLAY_OFFSETS_S = (0.0, 0.025, 0.05, 0.075)

# The declared search grid. It is fixed here, before any new stream is recorded, so the number of
# settings tried is known and small. Changing it changes ``grid_hash()``.
GRID_VERSION = 1
GATE_GRID = (0.0, 0.5, 0.8, 0.9, 1.0)
TAU_GRID = (0.5, 0.7, 0.9, 0.95, 0.99)
VOTE_GRID = ((1, 1), (3, 5), (5, 8), (8, 12))
COOLDOWN_GRID = (1.6, 2.5, 4.0)
QUIET_GRID = (0, 3, 8)
GUARD_GRID = (0.0, 1.0)

# A fixed, untuned ladder that adds one mechanism at a time, to show what each contributes.
LADDER: tuple[tuple[str, GateConfig, StabilizerConfig], ...] = (
    ("L1 cooldown only", GateConfig(0.0), StabilizerConfig(tau=0.5, cooldown_s=1.6)),
    ("L2 + gate 0.8", GateConfig(0.8), StabilizerConfig(tau=0.5, cooldown_s=1.6)),
    ("L3 + confidence 0.9", GateConfig(0.8), StabilizerConfig(tau=0.9, cooldown_s=1.6)),
    ("L4 + vote 3 of 5", GateConfig(0.8), StabilizerConfig(tau=0.9, k=3, n=5, cooldown_s=1.6)),
    ("L5 + re-arm 3, guard 1 s, cooldown 2.5 s", GateConfig(0.8),
     StabilizerConfig(tau=0.9, k=3, n=5, cooldown_s=2.5, rearm_quiet=3, opposite_guard_s=1.0)),
)  # fmt: skip

# Development-time constraints used when *choosing* a setting. They are deliberately a little
# stricter than the final acceptance criteria, to leave a margin for generalisation.
DEV_MIN_POOLED_RECALL = 0.90
DEV_MIN_CLASS_RECALL = 0.85
# Exploratory screening only (not the final acceptance criterion, which is 2 per 90 takes): at most
# DEV_SCREEN_MAX_CROSS cross activations per DEV_SCREEN_PER_TAKES target takes. Raw counts are
# always reported next to it.
DEV_SCREEN_MAX_CROSS = 2
DEV_SCREEN_PER_TAKES = 45
DEV_MAX_DUPLICATE_FRACTION = 0.05


class SealedDataError(RuntimeError):
    """Development code was pointed at the sealed final data."""


def assert_not_sealed(path: Path) -> None:
    """Refuse any path inside ``.../streaming/final``: only ``accept`` may open it."""
    parts = [p.lower() for p in Path(path).resolve().parts]
    for a, b in zip(parts, parts[1:], strict=False):
        if a == "streaming" and b == "final":
            raise SealedDataError(f"{path} is sealed final data; only `stream-accept` opens it")


# ----------------------------------------------------------------------------- statistics
def poisson_upper_one_sided(events: int, exposure: float, confidence: float = 0.95) -> float:
    """Exact one-sided upper confidence bound on a rate: ``events`` seen over ``exposure``."""
    if events < 0 or exposure <= 0 or not (0.0 < confidence < 1.0):
        raise ValueError("events >= 0, exposure > 0 and 0 < confidence < 1 are required")
    low, high = 0.0, max(10.0, 10.0 * (events + 1))
    for _ in range(200):
        mid = (low + high) / 2
        if _poisson_cdf(events, mid) > 1.0 - confidence:  # still compatible: go higher
            low = mid
        else:
            high = mid
    return (low + high) / 2 / exposure


# ------------------------------------------------------------------------- extra gates
SWIPE_CLASSES = ("swipe_left", "swipe_right")
SETTLE_STEPS = 6  # of the 32 feature steps over 1.6 s: the last 0.3 s of the window


@dataclass(frozen=True)
class ExtraGates:
    """Experimental gates on top of the open-palm gate (both off by default).

    ``settle_speed``: a *swipe* only counts if the palm moved at most this fast (hand sizes per
    second) over the last 0.3 s of the window, because a swipe is defined as ending with a stop.
    Circles are exempt: recorded circles end moving (median about 2 hand sizes per second).
    ``arming``: an arming rule, see ``arming.py``.
    """

    settle_speed: float | None = None
    arming: ArmingConfig | None = None

    @property
    def active(self) -> bool:
        return self.settle_speed is not None or self.arming is not None

    def to_dict(self) -> dict[str, Any]:
        return {"settle_speed": self.settle_speed,
                "arming": None if self.arming is None else asdict(self.arming)}  # fmt: skip

    def mechanism(self) -> str:
        parts = (["settle"] if self.settle_speed is not None else []) + (
            ["arming"] if self.arming is not None else [])  # fmt: skip
        return "+".join(parts) or "base"


def window_end_speed(features: np.ndarray) -> float:
    """Palm speed over the last 0.3 s of a window, from its own features (hand sizes/s)."""
    return float(np.linalg.norm(features[-SETTLE_STEPS:, 2:4], axis=1).mean())


# ----------------------------------------------------------------------------- recordings
@dataclass(frozen=True)
class Recording:
    recording_id: str
    session_id: str
    sequence: SequenceRecord
    file: str
    file_sha256: str
    conditions: str
    created_utc: str

    @property
    def minutes(self) -> float:
        return self.sequence.duration_s / 60.0


def _session_files(paths: Sequence[Path], allow_sealed: bool = False) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not allow_sealed:
            assert_not_sealed(path)
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"not found: {path}")
    return files


def load_recordings(
    paths: Sequence[Path], *, allow_sealed: bool = False
) -> tuple[list[Recording], list[SequenceRecord]]:  # fmt: skip
    """Continuous-use recordings and labelled takes from session files or folders (development).

    ``allow_sealed`` exists only for reviewing the candidates of an acceptance run, whose items
    live in an ``acceptance`` folder; it is never used to tune.
    """
    streams: list[Recording] = []
    takes: list[SequenceRecord] = []
    for file in _session_files(paths, allow_sealed):
        session = load_session(file, strict=True)
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        for seq in session.sequences:
            if seq.kind == KIND_STREAM:
                streams.append(Recording(
                    seq.sequence_id, session.header.session_id, seq, file.name, digest,
                    session.header.conditions, session.header.created_utc,
                ))  # fmt: skip
            else:
                takes.append(seq)
    return streams, takes


# ------------------------------------------------------------------------------- windows
@dataclass
class RecordingWindows:
    """Every sliding window of one recording: features (or why not) and open-palm fraction."""

    recording_id: str
    duration_s: float
    window_s: float
    starts: list[float] = field(default_factory=list)
    features: list[np.ndarray | None] = field(default_factory=list)
    skip: list[str | None] = field(default_factory=list)
    palm_fraction: list[float | None] = field(default_factory=list)
    end_speed: list[float | None] = field(default_factory=list)
    armed: dict[ArmingConfig, list[bool]] = field(default_factory=dict)

    @property
    def minutes(self) -> float:
        return self.duration_s / 60.0

    @property
    def evaluated(self) -> int:
        return sum(1 for f in self.features if f is not None)


def slice_window(seq: SequenceRecord, t: np.ndarray, lo: int, hi: int, start: float
                 ) -> SequenceRecord:  # fmt: skip
    """The frames ``[lo, hi)`` of a recording as a window that starts at time zero."""
    return SequenceRecord(
        sequence_id=f"{seq.sequence_id}@{start:.3f}", session_id=seq.session_id,
        label=seq.label, subtype=seq.subtype, kind="take",
        t_ms=(t[lo:hi] - t[lo]) * 1000.0, landmarks=seq.landmarks[lo:hi],
        confidence=seq.confidence[lo:hi], usable=seq.usable[lo:hi],
        aspect_ratio=seq.aspect_ratio,
    )  # fmt: skip


def compute_windows(
    seq: SequenceRecord, config: FeatureConfig, *, stride_s: float = STRIDE_S,
    offset_s: float = 0.0, recording_id: str | None = None,
    arming_configs: Sequence[ArmingConfig] = (),
) -> RecordingWindows:  # fmt: skip
    t = (seq.t_ms - seq.t_ms[0]) / 1000.0
    flags = frame_palm_flags(seq)
    armed_frames = {cfg: armed_series(seq, flags, cfg) for cfg in arming_configs}
    out = RecordingWindows(recording_id or seq.sequence_id, float(t[-1]), config.window_s)
    out.armed = {cfg: [] for cfg in arming_configs}
    for start in np.arange(offset_s, t[-1] - config.window_s + 1e-9, stride_s):
        lo = int(np.searchsorted(t, start, side="left"))
        hi = int(np.searchsorted(t, start + config.window_s, side="right"))
        out.starts.append(float(start))
        out.palm_fraction.append(window_palm_fraction(flags, seq.usable, lo, hi))
        features = None
        if hi - lo < MIN_FRAMES:
            out.skip.append("too few frames")
        else:
            try:
                features = extract_features(slice_window(seq, t, lo, hi, float(start)), config)
                out.skip.append(None)
            except FeatureError as exc:
                out.skip.append(failure_category(str(exc)))
        out.features.append(features)
        out.end_speed.append(None if features is None else window_end_speed(features))
        for cfg, frames in armed_frames.items():
            step = None if features is None else motion_onset_step(features, cfg.still_speed)
            if step is None:
                out.armed[cfg].append(False)
                continue
            onset_t = float(start) + step * config.window_s / (config.steps - 1)
            frame = max(0, int(np.searchsorted(t, onset_t, side="right")) - 1)
            out.armed[cfg].append(bool(frames[frame]))
    return out


@dataclass
class SystemOutputs:
    """One classifier's raw output for every window of a recording."""

    system: str
    floor: float  # the model's own rejection threshold (0 for the rule baselines)
    labels: list[str]
    confidences: list[float]
    valid: list[bool]


def predict_windows(name: str, classifier: Classifier, windows: RecordingWindows) -> SystemOutputs:
    """The class the model leans towards and its probability, before any rejection threshold.

    Rule baselines have no confidence; theirs is 1.0.
    """
    floor = float(getattr(classifier, "threshold", 0.0))
    labels: list[str] = []
    confidences: list[float] = []
    valid: list[bool] = []
    for features in windows.features:
        if features is None:
            labels.append("")
            confidences.append(0.0)
            valid.append(True)
            continue
        prediction = classifier.predict_features(features)
        if not prediction.valid:
            labels.append(REJECTED)
            confidences.append(0.0)
            valid.append(False)
        elif prediction.probabilities:
            top = int(np.argmax(prediction.probabilities))
            labels.append(CLASS_NAMES[top])
            confidences.append(float(prediction.probabilities[top]))
            valid.append(True)
        else:
            labels.append(prediction.label)
            confidences.append(1.0)
            valid.append(True)
    return SystemOutputs(name, floor, labels, confidences, valid)


NO_EXTRAS = ExtraGates()


def decisions_for(
    windows: RecordingWindows, outputs: SystemOutputs, gate: GateConfig,
    extras: ExtraGates = NO_EXTRAS,
) -> list[WindowDecision]:  # fmt: skip
    if extras.arming is not None and extras.arming not in windows.armed:
        raise ValueError("these windows were not computed for that arming rule")
    armed = windows.armed[extras.arming] if extras.arming is not None else None
    decisions = []
    for i, start in enumerate(windows.starts):
        label = outputs.labels[i]
        speed = windows.end_speed[i]
        settled = (extras.settle_speed is None or label not in SWIPE_CLASSES
                   or (speed is not None and speed <= extras.settle_speed))  # fmt: skip
        decisions.append(WindowDecision(
            start, start + windows.window_s, windows.features[i] is not None, label,
            outputs.confidences[i], outputs.valid[i], gate_passes(windows.palm_fraction[i], gate),
            settled, True if armed is None else armed[i],
        ))  # fmt: skip
    return decisions


def legacy_counts(windows: RecordingWindows, outputs: SystemOutputs) -> dict[str, Any]:
    """The earlier ``evaluate`` measures (no gate, no stabilizer), for continuity with eval-001."""
    flagged = [
        (start, outputs.labels[i]) for i, start in enumerate(windows.starts)
        if windows.features[i] is not None and outputs.valid[i]
        and outputs.labels[i] in TARGET_CLASSES and outputs.confidences[i] >= outputs.floor
    ]  # fmt: skip
    events = _events(flagged)
    persistent = [e for e in events if e["windows"] >= MIN_PERSISTENT_WINDOWS]
    minutes = windows.minutes
    return {"flagged_windows": len(flagged), "events_any": len(events),
            "events_persistent": len(persistent),
            "events_any_per_minute": len(events) / minutes if minutes else None,
            "events_persistent_per_minute":
                len(persistent) / minutes if minutes else None}  # fmt: skip


# --------------------------------------------------------------------------------- replay
@dataclass
class ReplayCase:
    take_id: str
    truth: str
    subtype: str | None
    offset_s: float
    take_start_s: float
    take_end_s: float
    windows: RecordingWindows


def pad_take(
    seq: SequenceRecord, lead_s: float = REPLAY_LEAD_S, tail_s: float = REPLAY_TAIL_S
) -> tuple[SequenceRecord, float, float]:  # fmt: skip
    """The take with its first and last trusted frames held still before and after it.

    Returns (padded sequence, take start in seconds, take end in seconds) on the padded timeline.
    """
    trusted = np.flatnonzero(seq.usable)
    if trusted.size == 0:
        raise ValueError("a take with no trusted frame cannot be replayed")
    t = seq.t_ms - seq.t_ms[0]
    dt = float(np.median(np.diff(t)))
    n_lead, n_tail = int(round(lead_s * 1000 / dt)), int(round(tail_s * 1000 / dt))
    lead_t = -dt * np.arange(n_lead, 0, -1)
    tail_t = t[-1] + dt * np.arange(1, n_tail + 1)
    first, last = int(trusted[0]), int(trusted[-1])

    def hold(array: np.ndarray, source: int, count: int) -> np.ndarray:
        return np.repeat(array[source][None], count, axis=0)

    new_t = np.concatenate([lead_t, t, tail_t])
    padded = SequenceRecord(
        sequence_id=seq.sequence_id, session_id=seq.session_id, label=seq.label,
        subtype=seq.subtype, kind="take", t_ms=new_t - new_t[0],
        landmarks=np.concatenate([hold(seq.landmarks, first, n_lead), seq.landmarks,
                                  hold(seq.landmarks, last, n_tail)]),
        confidence=np.concatenate([hold(seq.confidence, first, n_lead), seq.confidence,
                                   hold(seq.confidence, last, n_tail)]),
        usable=np.concatenate([np.ones(n_lead, bool), seq.usable, np.ones(n_tail, bool)]),
        aspect_ratio=seq.aspect_ratio,
    )  # fmt: skip
    start = n_lead * dt / 1000.0
    return padded, start, start + float(t[-1]) / 1000.0


def build_replay_cases(
    takes: Sequence[SequenceRecord], config: FeatureConfig,
    offsets: Sequence[float] = REPLAY_OFFSETS_S, *, lead_s: float = REPLAY_LEAD_S,
    tail_s: float = REPLAY_TAIL_S, arming_configs: Sequence[ArmingConfig] = (),
) -> list[ReplayCase]:  # fmt: skip
    """Replay cases. ``lead_s=0`` replays a take with only its own recorded beginning."""
    cases: list[ReplayCase] = []
    for take in takes:
        try:
            padded, start, end = pad_take(take, lead_s, tail_s)
        except ValueError:
            continue
        for offset in offsets:
            windows = compute_windows(padded, config, offset_s=offset,
                                      recording_id=f"{take.sequence_id}+{offset:g}",
                                      arming_configs=arming_configs)  # fmt: skip
            cases.append(ReplayCase(take.sequence_id, take.label, take.subtype, offset, start,
                                    end, windows))  # fmt: skip
    return cases


def classify_replay(
    activations: Sequence[Activation], truth: str, take_start: float, take_end: float,
    window_s: float,
) -> dict[str, Any]:  # fmt: skip
    """Outcome of one replayed take.

    An activation belongs to the take if the window that completed it ended after the take began
    and no later than one window length after the take ended. Everything else is a stray
    activation in the held rest.
    """
    inside = [a for a in activations if take_start < a.time_s <= take_end + window_s + 1e-9]
    stray = len(activations) - len(inside)
    if truth == NEGATIVE_LABEL:
        return {"outcome": "false_activation" if activations else "quiet", "duplicates": 0,
                "stray": stray, "activations": len(activations)}  # fmt: skip
    wrong = [a for a in inside if a.label != truth]
    right = [a for a in inside if a.label == truth]
    if wrong:
        outcome = "cross"
    elif right:
        outcome = "correct"
    else:
        outcome = "missed"
    return {"outcome": outcome, "duplicates": max(0, len(right) - 1) if not wrong else 0,
            "stray": stray, "activations": len(activations),
            "wrong": [(a.label, round(a.time_s - take_start, 2)) for a in wrong],
            "latency_s": right[0].time_s - take_end if right and not wrong else None}  # fmt: skip


def _missed_reason(
    case: ReplayCase, outputs: SystemOutputs, gate: GateConfig, stab: StabilizerConfig,
    extras: ExtraGates = NO_EXTRAS,
) -> str:  # fmt: skip
    """Why a missed take was missed: the model, the gate, arming, settling or the stabilizer."""
    w = case.windows
    floor = max(stab.tau, outputs.floor)
    near = [i for i, s in enumerate(w.starts)
            if case.take_start_s < s + w.window_s <= case.take_end_s + w.window_s + 1e-9
            and w.features[i] is not None and outputs.valid[i]
            and outputs.labels[i] == case.truth and outputs.confidences[i] >= floor]  # fmt: skip
    if not near:
        return "model_or_confidence"
    near = [i for i in near if gate_passes(w.palm_fraction[i], gate)]
    if not near:
        return "gate"
    if extras.arming is not None:
        near = [i for i in near if w.armed[extras.arming][i]]
        if not near:
            return "arming"
    if extras.settle_speed is not None and case.truth in SWIPE_CLASSES:
        near = [i for i in near if (w.end_speed[i] or 0.0) <= extras.settle_speed]
        if not near:
            return "settle"
    return "stabilizer"


@dataclass(frozen=True)
class DecisionConfig:
    system: str
    gate: GateConfig
    stab: StabilizerConfig
    extras: ExtraGates = NO_EXTRAS

    def to_dict(self) -> dict[str, Any]:
        out = {"system": self.system, "gate": asdict(self.gate), "stabilizer": asdict(self.stab)}
        if self.extras.active:  # absent for the original pipeline, so its hashes are unchanged
            out["extras"] = self.extras.to_dict()
        return out

    def hash(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def order_key(self) -> tuple[float, ...]:
        """Deterministic tie-break among equally good settings: the more conservative first."""
        s = self.stab
        arming = 0.0 if self.extras.arming is None else self.extras.arming.hold_s
        settle = 1e9 if self.extras.settle_speed is None else self.extras.settle_speed
        return (-self.gate.min_palm_fraction, -s.tau, -s.k, -s.n, -s.cooldown_s, -s.rearm_quiet,
                -s.opposite_guard_s, settle, -arming)  # fmt: skip


@dataclass
class SystemCorpus:
    """One system's raw outputs over the whole development corpus."""

    name: str
    streams: list[tuple[RecordingWindows, SystemOutputs]]
    cases: list[tuple[ReplayCase, SystemOutputs]]


def build_system_corpus(
    name: str, classifier: Classifier, stream_windows_: Sequence[RecordingWindows],
    cases: Sequence[ReplayCase],
) -> SystemCorpus:  # fmt: skip
    return SystemCorpus(
        name,
        [(w, predict_windows(name, classifier, w)) for w in stream_windows_],
        [(c, predict_windows(name, classifier, c.windows)) for c in cases],
    )  # fmt: skip


DecisionCache = dict[tuple[Any, ...], list[WindowDecision]]


def _decisions(cache: DecisionCache | None, windows: RecordingWindows, outputs: SystemOutputs,
               gate: GateConfig, extras: ExtraGates) -> list[WindowDecision]:  # fmt: skip
    """Decisions depend only on the gates, so a grid can share them across the other settings."""
    if cache is None:
        return decisions_for(windows, outputs, gate, extras)
    key = (id(outputs), gate.min_palm_fraction, extras)
    if key not in cache:
        cache[key] = decisions_for(windows, outputs, gate, extras)
    return cache[key]


def replay_outcomes(corpus: SystemCorpus, gate: GateConfig, stab: StabilizerConfig,
                    extras: ExtraGates = NO_EXTRAS, cache: DecisionCache | None = None,
                    ) -> list[dict[str, Any]]:  # fmt: skip
    """One conservative outcome per take, combining all of its alignments."""
    per_take: dict[str, list[dict[str, Any]]] = {}
    for case, outputs in corpus.cases:
        activations = run_stabilizer(_decisions(cache, case.windows, outputs, gate, extras), stab,
                                     outputs.floor)  # fmt: skip
        result = classify_replay(activations, case.truth, case.take_start_s, case.take_end_s,
                                 case.windows.window_s)  # fmt: skip
        result.update(take_id=case.take_id, truth=case.truth, subtype=case.subtype,
                      offset_s=case.offset_s)  # fmt: skip
        if result["outcome"] == "missed":
            result["reason"] = _missed_reason(case, outputs, gate, stab, extras)
        per_take.setdefault(case.take_id, []).append(result)

    combined = []
    for take_id, results in per_take.items():
        outcomes = {r["outcome"] for r in results}
        if "cross" in outcomes:
            outcome = "cross"
        elif "false_activation" in outcomes:
            outcome = "false_activation"
        elif "missed" in outcomes:
            outcome = "missed"
        elif outcomes == {"quiet"}:
            outcome = "quiet"
        else:
            outcome = "correct"
        missed = [r for r in results if r["outcome"] == "missed"]
        latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
        combined.append({
            "take_id": take_id, "truth": results[0]["truth"], "subtype": results[0]["subtype"],
            "outcome": outcome, "duplicates": max(r["duplicates"] for r in results),
            "stray": max(r["stray"] for r in results),
            "reason": missed[0]["reason"] if missed else None,
            "wrong": next((r["wrong"] for r in results if r.get("wrong")), []),
            "latency_s": max(latencies) if latencies else None,  # worst alignment
        })  # fmt: skip
    return combined


def summarize_recall(outcomes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gestures = [o for o in outcomes if o["truth"] in TARGET_CLASSES]
    per_class: dict[str, dict[str, Any]] = {}
    for label in TARGET_CLASSES:
        rows = [o for o in gestures if o["truth"] == label]
        hit = sum(1 for o in rows if o["outcome"] == "correct")
        ci = wilson_interval(hit, len(rows))
        per_class[label] = {"n": len(rows), "correct": hit,
                            "recall": hit / len(rows) if rows else None,
                            "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None}  # fmt: skip
    correct = sum(1 for o in gestures if o["outcome"] == "correct")
    reasons: dict[str, int] = {}
    for o in gestures:
        if o["outcome"] == "missed":
            reasons[o["reason"]] = reasons.get(o["reason"], 0) + 1
    negatives = [o for o in outcomes if o["truth"] == NEGATIVE_LABEL]
    ci = wilson_interval(correct, len(gestures))
    return {
        "gestures": len(gestures), "correct": correct,
        "pooled_recall": correct / len(gestures) if gestures else None,
        "pooled_ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
        "per_class": per_class,
        "cross_activations": sum(1 for o in gestures if o["outcome"] == "cross"),
        "missed": sum(1 for o in gestures if o["outcome"] == "missed"),
        "missed_by_reason": reasons,
        "takes_with_duplicate_activations": sum(1 for o in gestures if o["duplicates"] > 0),
        "takes_with_stray_activations": sum(1 for o in gestures if o["stray"] > 0),
        "negative_takes": len(negatives),
        "negative_takes_with_activation": sum(
            1 for o in negatives if o["outcome"] == "false_activation"),  # fmt: skip
    }


# ----------------------------------------------------------------------------- scoring
@dataclass
class ConfigResult:
    config: DecisionConfig
    streams: list[dict[str, Any]]  # per recording: id, minutes, candidates, times
    candidates: int
    minutes: float
    recall: dict[str, Any]
    outcomes: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def per_minute(self) -> float | None:
        return self.candidates / self.minutes if self.minutes else None

    def summary(self) -> dict[str, Any]:
        r = self.recall
        return {"config_hash": self.config.hash()[:12], **self.config.to_dict(),
                "candidates": self.candidates, "candidates_per_minute": self.per_minute,
                "pooled_recall": r["pooled_recall"],
                "class_recall": {c: v["recall"] for c, v in r["per_class"].items()},
                "cross_activations": r["cross_activations"],
                "takes_with_duplicates": r["takes_with_duplicate_activations"]}  # fmt: skip


def evaluate_config(corpus: SystemCorpus, config: DecisionConfig,
                    cache: DecisionCache | None = None) -> ConfigResult:  # fmt: skip
    streams = []
    for windows, outputs in corpus.streams:
        activations = run_stabilizer(
            _decisions(cache, windows, outputs, config.gate, config.extras), config.stab,
            outputs.floor)  # fmt: skip
        streams.append({"recording_id": windows.recording_id, "minutes": windows.minutes,
                        "candidates": len(activations),
                        "times_s": [round(a.time_s, 2) for a in activations],
                        "labels": [a.label for a in activations]})  # fmt: skip
    outcomes = replay_outcomes(corpus, config.gate, config.stab, config.extras, cache)
    return ConfigResult(
        config, streams, sum(s["candidates"] for s in streams),
        sum(s["minutes"] for s in streams), summarize_recall(outcomes), outcomes,
    )  # fmt: skip


def grid_configs(system: str, has_confidence: bool) -> list[DecisionConfig]:
    """Every setting of the declared grid; rule baselines have no confidence, so no tau axis."""
    taus = TAU_GRID if has_confidence else (TAU_GRID[0],)
    return [
        DecisionConfig(system, GateConfig(g), StabilizerConfig(tau, k, n, cd, q, guard))
        for g in GATE_GRID for tau in taus for (k, n) in VOTE_GRID for cd in COOLDOWN_GRID
        for q in QUIET_GRID for guard in GUARD_GRID
    ]  # fmt: skip


def grid_hash() -> str:
    payload = {"version": GRID_VERSION, "gate": GATE_GRID, "tau": TAU_GRID, "vote": VOTE_GRID,
               "cooldown": COOLDOWN_GRID, "quiet": QUIET_GRID, "guard": GUARD_GRID,
               "dev_constraints": [DEV_MIN_POOLED_RECALL, DEV_MIN_CLASS_RECALL,
                                   DEV_SCREEN_MAX_CROSS, DEV_SCREEN_PER_TAKES,
                                   DEV_MAX_DUPLICATE_FRACTION]}  # fmt: skip
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def meets_dev_constraints(recall: dict[str, Any]) -> bool:
    n = recall["gestures"]
    if not n or recall["pooled_recall"] is None:
        return False
    if recall["pooled_recall"] < DEV_MIN_POOLED_RECALL:
        return False
    if any(v["recall"] is None or v["recall"] < DEV_MIN_CLASS_RECALL
           for v in recall["per_class"].values()):  # fmt: skip
        return False
    duplicates = recall["takes_with_duplicate_activations"]
    return (recall["cross_activations"] * DEV_SCREEN_PER_TAKES <= DEV_SCREEN_MAX_CROSS * n
            and duplicates <= DEV_MAX_DUPLICATE_FRACTION * n)


def pareto_front(results: Sequence[ConfigResult]) -> list[ConfigResult]:
    """Settings that no other setting beats on both candidates per minute and pooled recall."""
    scored = [(r.candidates, -(r.recall["pooled_recall"] or 0.0), r) for r in results]
    front = []
    for c, neg_recall, r in scored:
        dominated = any(
            c2 <= c and n2 <= neg_recall and (c2 < c or n2 < neg_recall)
            for c2, n2, _ in scored
        )  # fmt: skip
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: (r.candidates, -(r.recall["pooled_recall"] or 0.0)))


def pareto_summary(results: Sequence[ConfigResult], limit: int = 40) -> list[dict[str, Any]]:
    """The frontier with settings that give identical results collapsed into one row."""
    groups: dict[tuple[Any, ...], list[ConfigResult]] = {}
    for r in pareto_front(results):
        key = (r.candidates, r.recall["pooled_recall"], r.recall["cross_activations"])
        groups.setdefault(key, []).append(r)
    rows = []
    for members in groups.values():
        members.sort(key=lambda r: r.config.order_key())
        rows.append({**members[0].summary(), "equivalent_settings": len(members)})
    return rows[:limit]


def _best(results: Sequence[ConfigResult], recording_ids: Sequence[str]) -> ConfigResult | None:
    eligible = [r for r in results if meets_dev_constraints(r.recall)]
    if not eligible:
        return None

    def key(r: ConfigResult) -> tuple[Any, ...]:
        count = sum(s["candidates"] for s in r.streams if s["recording_id"] in recording_ids)
        return (count, -(r.recall["pooled_recall"] or 0.0), r.config.order_key())

    return min(eligible, key=key)


def select_config(results: Sequence[ConfigResult]) -> ConfigResult | None:
    """Fewest candidates over all development recordings among settings that keep recall."""
    ids = sorted({s["recording_id"] for r in results for s in r.streams})
    return _best(results, ids)


def leave_one_recording_out(results: Sequence[ConfigResult]) -> dict[str, Any]:
    """Pick a setting without each recording in turn and count candidates on the one left out.

    Needs at least two recordings. It estimates how well *the selection procedure* transfers to a
    recording it did not see; it is not an estimate for the final setting.
    """
    ids = sorted({s["recording_id"] for r in results for s in r.streams})
    if len(ids) < 2:
        raise ValueError("leave-one-recording-out needs at least two development recordings")
    folds = []
    for held in ids:
        rest = [i for i in ids if i != held]
        chosen = _best(results, rest)
        if chosen is None:
            folds.append({"held_out": held, "chosen": None})
            continue
        stream = next(s for s in chosen.streams if s["recording_id"] == held)
        folds.append({"held_out": held, "chosen_hash": chosen.config.hash()[:12],
                      "chosen": chosen.config.to_dict(),
                      "held_out_candidates": stream["candidates"],
                      "held_out_minutes": stream["minutes"],
                      "held_out_per_minute": stream["candidates"] / stream["minutes"]})  # fmt: skip
    done = [f for f in folds if f["chosen"] is not None]
    minutes = sum(f["held_out_minutes"] for f in done)
    count = sum(f["held_out_candidates"] for f in done)
    return {"folds": folds, "pooled_held_out_candidates": count, "pooled_held_out_minutes": minutes,
            "pooled_held_out_per_minute": count / minutes if minutes else None,
            "folds_without_a_feasible_setting": len(folds) - len(done),
            "distinct_settings_chosen": len({f["chosen_hash"] for f in done})}  # fmt: skip


def rate_summary(count: int, minutes: float) -> dict[str, Any]:
    low, high = poisson_rate_interval(count, minutes)
    return {"count": count, "minutes": round(minutes, 3), "per_minute": count / minutes,
            "ci95_two_sided_per_minute": [round(low, 4), round(high, 4)],
            "upper95_one_sided_per_minute":
                round(poisson_upper_one_sided(count, minutes), 4)}  # fmt: skip


def percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if len(values) else None

