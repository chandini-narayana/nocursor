"""Offline evaluation: both systems on the same held-out test takes, then the reserved stream.

This is the only code that opens the test split (``allow_test=True``) and the continuous stream.
It re-verifies the frozen split against the data first, records every evaluation in the run's
``eval_log.json`` (so a report can say how many times the test split has been scored), scores the
rule baseline (untuned and tuned) and the GRU on identical samples, repeats the scoring without the
flagged takes, optionally compares with a separately trained run that excluded those takes, and
slides a window over the stream to count false activations per minute.

Nothing here tunes anything: thresholds and weights come from the run directory as trained.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nocursor.dynamic.baseline import load_baselines
from nocursor.dynamic.classifier import Classifier
from nocursor.dynamic.dataset import load_dataset
from nocursor.dynamic.features import (
    FeatureConfig,
    FeatureError,
    extract_features,
    failure_category,
)
from nocursor.dynamic.model_io import META_FILE, load_classifier
from nocursor.dynamic.report import LIMITATIONS, render_markdown, write_confusion_png
from nocursor.dynamic.schema import MIN_FRAMES, SequenceRecord
from nocursor.dynamic.scoring import ScoredSample, poisson_rate_interval, score_samples
from nocursor.dynamic.splits import (
    SPLIT_FILE,
    SPLIT_TEST,
    SplitError,
    SplitItem,
    load_split_items,
    load_stream,
    verify_split,
)
from nocursor.metrics import summarize

EVAL_LOG = "eval_log.json"
WINDOW_STRIDE_S = 0.1
EVENT_MERGE_GAP_S = 1.0  # flagged windows closer than this, with the same class, are one event
MIN_PERSISTENT_WINDOWS = 3  # the "persistent" event count needs at least this many in a row
LATENCY_MIN_SAMPLES = 30
LATENCY_WARMUP = 5
SCORE_SEED = 0


class EvaluationError(RuntimeError):
    """The run cannot be evaluated (inconsistent files, wrong split, missing artifacts)."""


@dataclass
class System:
    name: str
    classifier: Classifier
    description: dict[str, Any]


# ------------------------------------------------------------------------------- loading
def load_systems(run_dir: Path) -> list[System]:
    """rule_defaults, rule_tuned and (if the run has one) the GRU, from a run directory."""
    systems: list[System] = []
    if (run_dir / "baseline.json").is_file():
        defaults, tuned, payload = load_baselines(run_dir)
        systems.append(System(defaults.name, defaults, {
            "kind": "rule baseline, thresholds fixed in advance",
            "parameters": payload["rule_defaults"]["params"]}))  # fmt: skip
        systems.append(System(tuned.name, tuned, {
            "kind": "rule baseline tuned on pooled training and validation takes",
            "parameters": payload["rule_tuned"]["params"],
            "tuning": {"macro_f1_on_tuning_data": payload["rule_tuned"]["macro_f1_on_tuning_data"],
                       "passes": payload["rule_tuned"]["passes"],
                       "changes": payload["rule_tuned"]["changes"],
                       "tuned_on": payload["tuned_on"]}}))  # fmt: skip
    if (run_dir / META_FILE).is_file():
        gru = load_classifier(run_dir)
        meta = gru.meta
        systems.append(System("gru", gru, {
            "kind": "GRU, NumPy inference from the saved artifact",
            "architecture": meta["architecture"], "rejection": meta["rejection"],
            "seed": meta["seed"], "training": meta["training"], "libraries": meta["libraries"],
            "weights_sha256": meta["weights_sha256"], "created_utc": meta["created_utc"]}))
    if not systems:
        raise EvaluationError(f"{run_dir.name} has neither baseline.json nor a model")
    return systems


def _load_run_split(run_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((run_dir / SPLIT_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{run_dir.name} has no readable {SPLIT_FILE}: {exc}") from exc


def _check_same_split(run_dir: Path, split: dict[str, Any], systems: list[System]) -> None:
    for system in systems:
        meta = getattr(system.classifier, "meta", None)
        if meta and (meta["split_id"] != split["split_id"]
                     or meta["dataset_manifest_hash"] != split["dataset_manifest_hash"]):
            raise EvaluationError(f"the model in {run_dir.name} was trained on a different split")


def _log_evaluation(run_dir: Path) -> tuple[str, int]:
    """Record this evaluation in the run's log before anything is scored; returns (id, number)."""
    path = run_dir / EVAL_LOG
    entries = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    now = datetime.now(timezone.utc)
    number = len(entries) + 1
    eval_id = f"eval-{number:03d}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    entries.append({"eval_id": eval_id, "utc": now.isoformat(timespec="seconds")})
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return eval_id, number


# -------------------------------------------------------------------------------- scoring
def predict_items(system: System, items: list[SplitItem]) -> list[ScoredSample]:
    return [
        ScoredSample(i.sequence.label, system.classifier.predict_features(i.features),
                     i.sequence.subtype, i.sequence.sequence_id, i.flagged)
        for i in items
    ]


def time_system(system: System, items: list[SplitItem], config: FeatureConfig) -> dict[str, Any]:
    """Software timing of feature extraction and inference on this machine, in milliseconds."""
    features_ms, inference_ms = [], []
    for round_ in range(2):  # the first round warms caches and is discarded
        for item in items:
            t0 = time.perf_counter()
            features = extract_features(item.sequence, config)
            t1 = time.perf_counter()
            system.classifier.predict_features(features)
            t2 = time.perf_counter()
            if round_ == 1:
                features_ms.append((t1 - t0) * 1000)
                inference_ms.append((t2 - t1) * 1000)
    total = [a + b for a, b in zip(features_ms, inference_ms, strict=True)]
    out: dict[str, Any] = {"unit": "ms", "warmup_discarded": len(items), "note":
                           "in-process software timing on this machine; not camera-to-action "
                           "latency"}  # fmt: skip
    for key, values in (("feature_extraction", features_ms), ("inference", inference_ms),
                        ("total", total)):  # fmt: skip
        stats = summarize(values)
        if stats is None or stats.n < LATENCY_MIN_SAMPLES:
            out[key] = "insufficient measurements"
        else:
            out[key] = stats.as_dict()
    return out


# ---------------------------------------------------------------------------------- stream
def stream_windows(
    seq: SequenceRecord, config: FeatureConfig, stride_s: float = WINDOW_STRIDE_S
) -> list[tuple[float, SequenceRecord | None]]:  # fmt: skip
    """Sliding windows over a long recording: (start in seconds, window or None if too short)."""
    t = (seq.t_ms - seq.t_ms[0]) / 1000.0
    out: list[tuple[float, SequenceRecord | None]] = []
    for start in np.arange(0.0, t[-1] - config.window_s + 1e-9, stride_s):
        lo = int(np.searchsorted(t, start, side="left"))
        hi = int(np.searchsorted(t, start + config.window_s, side="right"))
        if hi - lo < MIN_FRAMES:
            out.append((float(start), None))
            continue
        window = SequenceRecord(
            sequence_id=f"{seq.sequence_id}@{start:.1f}", session_id=seq.session_id,
            label=seq.label, subtype=seq.subtype, kind="take",
            t_ms=(t[lo:hi] - t[lo]) * 1000.0, landmarks=seq.landmarks[lo:hi],
            confidence=seq.confidence[lo:hi], usable=seq.usable[lo:hi],
            aspect_ratio=seq.aspect_ratio,
        )  # fmt: skip
        out.append((float(start), window))
    return out


def _events(flagged: list[tuple[float, str]]) -> list[dict[str, Any]]:
    """Group flagged windows (start, class) into events: same class, gaps within the merge limit."""
    events: list[dict[str, Any]] = []
    for start, label in flagged:
        last = events[-1] if events else None
        if last and last["label"] == label and start - last["last_start"] <= EVENT_MERGE_GAP_S:
            last["windows"] += 1
            last["last_start"] = start
        else:
            events.append({"label": label, "first_start": start, "last_start": start, "windows": 1})
    return events


def evaluate_stream(
    system: System, streams: list[SequenceRecord], config: FeatureConfig
) -> dict[str, Any]:  # fmt: skip
    """False activations on the reserved continuous recording of ordinary use."""
    total_s = sum(s.duration_s for s in streams)
    windows = evaluated = 0
    skipped: dict[str, int] = {}
    flagged: list[tuple[float, str]] = []
    for stream in streams:
        for start, window in stream_windows(stream, config):
            windows += 1
            reason = None
            if window is None:
                reason = "too few frames"
            else:
                try:
                    features = extract_features(window, config)
                except FeatureError as exc:
                    reason = failure_category(str(exc))
            if reason is not None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            evaluated += 1
            prediction = system.classifier.predict_features(features)
            if prediction.activates:
                flagged.append((start, prediction.label))
    events = _events(flagged)
    persistent = [e for e in events if e["windows"] >= MIN_PERSISTENT_WINDOWS]
    minutes = total_s / 60.0

    def rate(count: int) -> dict[str, Any]:
        low, high = poisson_rate_interval(count, minutes)
        return {"events": count, "per_minute": count / minutes,
                "ci95_per_minute": [round(low, 4), round(high, 4)]}  # fmt: skip

    return {
        "stream_seconds": round(total_s, 1),
        "window_seconds": config.window_s, "stride_seconds": WINDOW_STRIDE_S,
        "windows_total": windows, "windows_evaluated": evaluated,
        "windows_skipped": skipped,
        "fraction_evaluated": evaluated / windows if windows else None,
        "flagged_windows": len(flagged),
        "flagged_by_class": {c: sum(1 for _, label in flagged if label == c)
                             for c in sorted({label for _, label in flagged})},  # fmt: skip
        "events_any": rate(len(events)),
        "events_persistent": rate(len(persistent)),
        "persistent_definition": f"at least {MIN_PERSISTENT_WINDOWS} consecutive flagged windows",
        "merge_gap_seconds": EVENT_MERGE_GAP_S,
        "event_times_seconds": [round(e["first_start"], 1) for e in events],
        "denominator": "minutes of stream, including any windows skipped for unusable tracking",
        "caution": f"{total_s:.0f} s of one person's use is limited evidence. Zero events would "
                   "still be compatible with a true rate of roughly 3.7 divided by the minutes "
                   "observed; overlapping windows are not independent trials.",  # fmt: skip
    }


# ------------------------------------------------------------------------------ the run
def _provenance(run_dir: Path, split: dict[str, Any]) -> str:
    if (run_dir / META_FILE).is_file():
        return json.loads((run_dir / META_FILE).read_text(encoding="utf-8"))["data_provenance"]
    synthetic = any(r.get("provenance") == "synthetic" for r in split["sessions"])
    return "SYNTHETIC (software test only)" if synthetic else "recorded"


def _seed_runs(run_dir: Path) -> list[Path]:
    seeds = sorted((run_dir / "seeds").glob("seed-*")) if (run_dir / "seeds").is_dir() else []
    return [run_dir, *seeds] if (run_dir / META_FILE).is_file() else []


def _scores(system: System, items: list[SplitItem]) -> dict[str, Any]:
    samples = predict_items(system, items)
    return {"all": score_samples(samples, SCORE_SEED),
            "without_flagged": score_samples([s for s in samples if not s.flagged], SCORE_SEED),
            "flagged_takes": [{"sequence_id": s.sequence_id, "true_label": s.true_label,
                               "predicted": s.prediction.label} for s in samples if s.flagged]}


def evaluate_run(
    run_dir: Path, dataset_dir: Path, *, sensitivity_run: Path | None = None,
    include_stream: bool = True, log: Callable[[str], object] = print,
) -> Path:  # fmt: skip
    """Evaluate a run on its test split (and the reserved stream); returns the report folder."""
    split = _load_run_split(run_dir)
    systems = load_systems(run_dir)
    _check_same_split(run_dir, split, systems)
    config = FeatureConfig(**split["feature_config"])

    dataset = load_dataset(dataset_dir, strict=True)
    try:
        verify_split(split, dataset)
    except SplitError as exc:
        raise EvaluationError(str(exc)) from exc

    other: list[System] | None = None
    if sensitivity_run is not None:
        if _load_run_split(sensitivity_run)["split_id"] != split["split_id"]:
            raise EvaluationError("the sensitivity run used a different split")
        other = load_systems(sensitivity_run)

    eval_id, number = _log_evaluation(run_dir)
    items = load_split_items(split, dataset_dir, [SPLIT_TEST], allow_test=True)
    log(f"evaluating {run_dir.name} on {len(items)} test takes "
        f"(evaluation number {number} of this run)")  # fmt: skip

    results: dict[str, Any] = {}
    for system in systems:
        entry = {"description": system.description, **_scores(system, items),
                 "latency": time_system(system, items, config)}  # fmt: skip
        results[system.name] = entry
        s = entry["all"]
        log(f"  {system.name:<14} macro-F1 {s['macro_f1']['value']:.3f}  accuracy "
            f"{s['accuracy']['value']:.3f}  false activations "
            f"{s['false_activations']['count']}/{s['false_activations']['negatives']}  "
            f"rejected {s['rejection']['rejected']}")  # fmt: skip

    stability = None
    if len(_seed_runs(run_dir)) > 1:
        stability = []
        for directory in _seed_runs(run_dir):
            gru = load_classifier(directory)
            samples = predict_items(System("gru", gru, {}), items)
            scored = score_samples(samples, SCORE_SEED)
            stability.append({
                "seed": gru.meta["seed"], "is_final_model": directory == run_dir,
                "threshold": gru.threshold, "macro_f1": scored["macro_f1"]["value"],
                "accuracy": scored["accuracy"]["value"],
                "false_activations": scored["false_activations"]["count"],
                "gestures_lost_to_rejection": scored["rejection"]["true_gestures_lost"],
            })  # fmt: skip

    sensitivity = None
    if other is not None:
        sensitivity = {"retrained_run": sensitivity_run.name,  # type: ignore[union-attr]
                       "systems": {s.name: _scores(s, items) for s in other}}  # fmt: skip

    stream_results = None
    if include_stream:
        streams = load_stream(split, dataset_dir)
        stream_results = {s.name: evaluate_stream(s, streams, config) for s in systems}
        for name, r in stream_results.items():
            log(f"  stream {name:<14} events {r['events_any']['events']} "
                f"({r['events_any']['per_minute']:.2f}/min, 95% CI "
                f"{r['events_any']['ci95_per_minute']}) over {r['stream_seconds']} s")  # fmt: skip

    out_dir = run_dir / "eval" / eval_id
    out_dir.mkdir(parents=True)
    provenance = _provenance(run_dir, split)
    payload = {
        "report_version": 1, "eval_id": eval_id, "evaluation_number_of_this_run": number,
        "run": run_dir.name, "split_id": split["split_id"], "split_seed": split["seed"],
        "data_provenance": provenance,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "manifest_hash": split["dataset_manifest_hash"],
            "assignments": split["assignments"], "counts": split["counts"],
            "counts_before_exclusion": split["counts_before_exclusion"],
            "excluded_takes": split["excluded"], "flagged_takes": split["flagged"],
            "sessions": split["sessions"], "reserved_stream": split["reserved_stream"],
            "dates": split["dates"], "warnings": split["warnings"],
        },
        "test_takes_evaluated": len(items),
        "systems": results, "seed_stability": stability, "sensitivity": sensitivity,
        "stream": stream_results, "limitations": LIMITATIONS,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    png_notes = []
    for name, entry in results.items():
        note = write_confusion_png(out_dir / f"confusion_{name}.png",
                                   entry["all"]["confusion"], f"{name} on the test session")
        if note:
            png_notes.append(note)
            break
    payload["png_note"] = png_notes[0] if png_notes else None
    (out_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    log(f"report written to {out_dir}")
    return out_dir
