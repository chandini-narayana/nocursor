"""Freeze, one-shot acceptance and blind candidate review for the streaming pipeline.

The order of events is enforced by files, so that it cannot happen by accident:

1. ``create_freeze`` records the model, the decision settings, the acceptance criteria and a hash of
   every relevant source file in ``freeze.json``. It refuses to run if any sealed final recording
   already exists (the freeze must come first) and never overwrites a freeze.
2. The final recordings are made afterwards. ``accept_run`` accepts only sessions created after
   the freeze, checks the data requirements *before* it uses up the one attempt, writes a marker
   before it opens the data, and refuses to run twice. A failed or abandoned attempt needs a new
   final dataset after any change.
3. Candidate activations from every system are pooled, merged and shuffled into review items
   without the system's name. A person marks each ``false``, ``genuine`` or ``ambiguous``.
4. ``accept_finalize`` turns the marks into the verdict. Ambiguous candidates count as false for
   acceptance and are also reported separately.

This is tamper-evident, not tamper-proof: the code logs and refuses, but a person with file access
can still edit files. A pass would not prove zero real-world risk. Offline only: nothing here is
connected to the mouse.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nocursor
from nocursor.dynamic.dataset import load_session
from nocursor.dynamic.evaluate import load_systems
from nocursor.dynamic.features import FeatureConfig
from nocursor.dynamic.gate import GateConfig
from nocursor.dynamic.model_io import META_FILE
from nocursor.dynamic.review import ReviewError, file_sha256, load_marks
from nocursor.dynamic.schema import KIND_STREAM, TARGET_CLASSES, SequenceRecord
from nocursor.dynamic.stabilizer import StabilizerConfig
from nocursor.dynamic.stream_eval import (
    DecisionConfig,
    build_replay_cases,
    build_system_corpus,
    compute_windows,
    evaluate_config,
    grid_hash,
    percentile,
    rate_summary,
)

FREEZE_VERSION = 1
FREEZE_FILE = "freeze.json"
STARTED_FILE = "acceptance_started.json"
RUN_FILE = "acceptance_run.json"
ITEMS_FILE = "review_items.json"
KEY_FILE = "review_key.json"
MARKS_FILE = "review_marks.json"
VERDICT_FILE = "verdict.json"
MARKS = ("false", "genuine", "ambiguous")
REVIEW_MERGE_S = 1.0  # candidates this close in one recording, same class, are one review item
SOURCE_PACKAGES = ("dynamic",)
SOURCE_MODULES = ("scroll.py", "models.py")  # the pose measure the gate depends on


class FreezeError(RuntimeError):
    """The freeze is missing, would be unsafe to create, or no longer matches the code or model."""


class AcceptanceError(RuntimeError):
    """The acceptance run or its finalisation cannot proceed."""


class AcceptanceDataError(AcceptanceError):
    """The final data does not meet the predeclared requirements; the attempt is not consumed."""


class ReviewIncompleteError(AcceptanceError):
    """Not every candidate has a valid mark."""


@dataclass(frozen=True)
class AcceptanceCriteria:
    """Predeclared before any final recording exists; part of the freeze and its hash."""

    min_minutes: float = 40.0
    min_recordings: int = 4
    min_days: int = 2
    min_conditions: int = 2
    max_confirmed_false: int = 0  # ambiguous reviewed candidates count as false
    upper_bound_confidence: float = 0.95
    min_protocol_sessions: int = 2
    min_takes_per_gesture: int = 30
    min_pooled_recall: float = 0.85
    min_class_recall: float = 0.80
    max_cross_activations: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------------- hashing
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(package_root: Path | None = None) -> dict[str, str]:
    """SHA-256 of every source file the streaming decision depends on, by relative name."""
    root = package_root or Path(nocursor.__file__).resolve().parent
    files: list[Path] = []
    for package in SOURCE_PACKAGES:
        files.extend(sorted((root / package).glob("*.py")))
    files.extend(root / name for name in SOURCE_MODULES)
    return {p.relative_to(root).as_posix(): _sha256(p) for p in files if p.is_file()}


def _canonical_hash(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def streaming_dir(runs_dir: Path) -> Path:
    return runs_dir / "streaming"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _model_identity(run_dir: Path, system: str) -> dict[str, str]:
    if system == "gru":
        meta = run_dir / META_FILE
        if not meta.is_file():
            raise FreezeError(f"{run_dir.name} has no trained model")
        weights = json.loads(meta.read_text("utf-8"))["weights_sha256"]
        return {"kind": "gru", "weights_sha256": weights}
    baseline = run_dir / "baseline.json"
    if not baseline.is_file():
        raise FreezeError(f"{run_dir.name} has no baseline.json")
    return {"kind": "rule", "baseline_json_sha256": _sha256(baseline)}


# ----------------------------------------------------------------------------------- freeze
def create_freeze(
    runs_dir: Path, run_dir: Path, decisions: Sequence[DecisionConfig], primary: str,
    final_dir: Path, *, criteria: AcceptanceCriteria | None = None,
    dev_files: Sequence[Path] = (), tuning_report: Path | None = None,
    package_root: Path | None = None,
) -> Path:  # fmt: skip
    """Write ``freeze.json``. Refuses if final recordings already exist or the freeze exists."""
    if final_dir.exists() and any(final_dir.iterdir()):
        raise FreezeError(f"{final_dir} already holds files: the freeze must be made before any "
                          "final recording exists")  # fmt: skip
    if any(d.extras.active for d in decisions):
        raise FreezeError("the experimental settle and arming variants cannot be frozen: they are "
                          "not part of the acceptance run")
    names = [d.system for d in decisions]
    if primary not in names or len(set(names)) != len(names):
        raise FreezeError("decisions need one distinct setting per system, including the primary")
    criteria = criteria or AcceptanceCriteria()
    body = {
        "freeze_version": FREEZE_VERSION,
        "run": run_dir.name,
        "primary_system": primary,
        "decisions": {d.system: {**d.to_dict(), "config_hash": d.hash(),
                                 "model": _model_identity(run_dir, d.system)} for d in decisions},
        "criteria": criteria.to_dict(), "criteria_hash": criteria.hash(),
        "grid_hash": grid_hash(),
        "source_hashes": source_hashes(package_root),
        "dev_files": {p.name: _sha256(p) for p in dev_files},
        "tuning_report_sha256": _sha256(tuning_report) if tuning_report else None,
        "feature_config": json.loads((run_dir / "split.json").read_text("utf-8"))["feature_config"],
    }  # fmt: skip
    body["freeze_hash"] = _canonical_hash(body)
    freeze_id = f"freeze-{body['freeze_hash'][:10]}"
    out = streaming_dir(runs_dir) / freeze_id / FREEZE_FILE
    if out.exists():
        raise FreezeError(f"{freeze_id} already exists; a freeze is never overwritten")
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps({"freeze_id": freeze_id, "created_utc": _now(), **body},
                              indent=2, sort_keys=True) + "\n", encoding="utf-8")  # fmt: skip
    return out


def load_freeze(freeze_dir: Path) -> dict[str, Any]:
    path = freeze_dir / FREEZE_FILE
    try:
        freeze = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeError(f"cannot read {path}: {exc}") from exc
    body = {k: v for k, v in freeze.items() if k not in ("freeze_id", "created_utc", "freeze_hash")}
    if _canonical_hash(body) != freeze.get("freeze_hash"):
        raise FreezeError("freeze.json was edited: its hash no longer matches its content")
    return freeze


def verify_freeze(freeze: dict[str, Any], run_dir: Path, package_root: Path | None = None) -> None:
    """The code and model are exactly what was frozen."""
    problems = []
    current = source_hashes(package_root)
    for name, digest in freeze["source_hashes"].items():
        if current.get(name) != digest:
            problems.append(f"source file {name} changed since the freeze")
    for name in set(current) - set(freeze["source_hashes"]):
        problems.append(f"source file {name} was added since the freeze")
    for system, entry in freeze["decisions"].items():
        if _model_identity(run_dir, system) != entry["model"]:
            problems.append(f"the {system} model or baseline changed since the freeze")
    if problems:
        raise FreezeError("freeze verification failed: " + "; ".join(problems))


def decision_from_freeze(freeze: dict[str, Any], system: str) -> DecisionConfig:
    entry = freeze["decisions"][system]
    return DecisionConfig(system, GateConfig(**entry["gate"]),
                          StabilizerConfig(**entry["stabilizer"]))


# ------------------------------------------------------------------------------ final data
@dataclass
class FinalData:
    streams: list[tuple[SequenceRecord, str, str, str]]  # sequence, session id, conditions, created
    takes: list[SequenceRecord]
    protocol_sessions: int
    facts: dict[str, Any]


def load_final(final_dir: Path, freeze_created_utc: str) -> FinalData:
    """Read the sealed final recordings; only the acceptance run calls this."""
    if not final_dir.is_dir():
        raise AcceptanceDataError(f"no final data folder: {final_dir}")
    streams, takes = [], []
    protocol_sessions = 0
    early = []
    for file in sorted(final_dir.glob("*.jsonl")):
        session = load_session(file, strict=True)
        if session.header.created_utc <= freeze_created_utc:
            early.append(session.header.session_id)
        has_takes = False
        for seq in session.sequences:
            if seq.kind == KIND_STREAM:
                streams.append((seq, session.header.session_id, session.header.conditions,
                                session.header.created_utc))  # fmt: skip
            else:
                takes.append(seq)
                has_takes = True
        protocol_sessions += int(has_takes)
    if early:
        raise AcceptanceDataError("sessions created at or before the freeze cannot be final "
                                  f"data: {', '.join(early)}")  # fmt: skip
    per_gesture = {g: sum(1 for t in takes if t.label == g) for g in TARGET_CLASSES}
    facts = {
        "recordings": len(streams),
        "minutes": sum(s[0].duration_s for s in streams) / 60.0,
        "days": sorted({s[3][:10] for s in streams}),
        "conditions": sorted({s[2].strip().lower() for s in streams if s[2].strip()}),
        "protocol_sessions": protocol_sessions, "takes_per_gesture": per_gesture,
    }  # fmt: skip
    return FinalData(streams, takes, protocol_sessions, facts)


def check_data_requirements(criteria: AcceptanceCriteria, facts: dict[str, Any]) -> list[str]:
    problems = []
    if facts["recordings"] < criteria.min_recordings:
        problems.append(f"{facts['recordings']} streams (need {criteria.min_recordings})")
    if facts["minutes"] < criteria.min_minutes:
        problems.append(f"{facts['minutes']:.1f} minutes (need {criteria.min_minutes:g})")
    if len(facts["days"]) < criteria.min_days:
        problems.append(f"{len(facts['days'])} recording days (need {criteria.min_days})")
    if len(facts["conditions"]) < criteria.min_conditions:
        problems.append(f"{len(facts['conditions'])} distinct condition descriptions "
                        f"(need {criteria.min_conditions})")  # fmt: skip
    if facts["protocol_sessions"] < criteria.min_protocol_sessions:
        problems.append(f"{facts['protocol_sessions']} protocol sessions "
                        f"(need {criteria.min_protocol_sessions})")  # fmt: skip
    for gesture, count in facts["takes_per_gesture"].items():
        if count < criteria.min_takes_per_gesture:
            problems.append(f"{count} {gesture} takes (need {criteria.min_takes_per_gesture})")
    return problems


# ---------------------------------------------------------------------------------- review
def build_review(
    candidates: dict[str, list[dict[str, Any]]], window_s: float, seed: int
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:  # fmt: skip
    """Pool candidates of all systems into shuffled items that do not name the system.

    ``candidates[system]`` holds ``{"recording_id", "time_s", "label"}`` rows. Rows of one
    recording and class within ``REVIEW_MERGE_S`` of each other become one item (a person judges
    the movement once). Returns (items for the reviewer, item id -> systems, kept apart).
    """
    rows = sorted(
        ({**c, "system": s} for s, cs in candidates.items() for c in cs),
        key=lambda c: (c["recording_id"], c["label"], c["time_s"], c["system"]),
    )  # fmt: skip
    clusters: list[list[dict[str, Any]]] = []
    for row in rows:
        last = clusters[-1] if clusters else None
        if (last and last[0]["recording_id"] == row["recording_id"]
                and last[0]["label"] == row["label"]
                and row["time_s"] - last[-1]["time_s"] <= REVIEW_MERGE_S):  # fmt: skip
            last.append(row)
        else:
            clusters.append([row])
    order = list(range(len(clusters)))
    random.Random(seed).shuffle(order)
    items, key = [], {}
    for rank, index in enumerate(order, start=1):
        cluster = clusters[index]
        item_id = f"c{rank:03d}"
        end = max(c["time_s"] for c in cluster)
        items.append({"item_id": item_id, "recording_id": cluster[0]["recording_id"],
                      "predicted": cluster[0]["label"],
                      "from_s": round(max(0.0, min(c["time_s"] for c in cluster) - window_s), 2),
                      "to_s": round(end + 0.5, 2),
                      "candidate_times_s": sorted({round(c["time_s"], 2) for c in cluster})})
        key[item_id] = sorted({c["system"] for c in cluster})
    return items, key


def confirmed_false_counts(
    key: dict[str, list[str]], items: Sequence[dict[str, Any]], marks: dict[str, str], system: str
) -> dict[str, Any]:  # fmt: skip
    """Per-system counts from the marks. Ambiguous counts as false and is also listed apart."""
    missing = [i["item_id"] for i in items if marks.get(i["item_id"]) not in MARKS]
    if missing:
        raise ReviewIncompleteError(f"{len(missing)} candidate(s) have no valid mark: "
                                    f"{', '.join(missing[:5])}")  # fmt: skip
    per_recording: dict[str, dict[str, int]] = {}
    for item in items:
        if system not in key[item["item_id"]]:
            continue
        row = per_recording.setdefault(item["recording_id"],
                                       {"false": 0, "ambiguous": 0, "genuine": 0})  # fmt: skip
        row[marks[item["item_id"]]] += 1
    totals = {m: sum(r[m] for r in per_recording.values()) for m in MARKS}
    return {"per_recording": per_recording, **totals,
            "confirmed_false_for_acceptance": totals["false"] + totals["ambiguous"]}  # fmt: skip


# ----------------------------------------------------------------------------------- judge
def judge(
    criteria: AcceptanceCriteria, facts: dict[str, Any], confirmed_false: int,
    per_recording_false: dict[str, int], recall: dict[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """Apply the predeclared criteria. Every check must pass for the same frozen setting."""
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    problems = check_data_requirements(criteria, facts)
    check("data requirements met", not problems, "; ".join(problems) or "yes")
    check("confirmed false activations (ambiguous counted as false)",
          confirmed_false <= criteria.max_confirmed_false,
          f"{confirmed_false} (allowed {criteria.max_confirmed_false})")  # fmt: skip
    pooled = recall["pooled_recall"]
    check("pooled recall", pooled is not None and pooled >= criteria.min_pooled_recall,
          f"{pooled:.3f} (need {criteria.min_pooled_recall})" if pooled is not None else "none")
    for label in TARGET_CLASSES:
        value = recall["per_class"][label]["recall"]
        check(f"recall {label}", value is not None and value >= criteria.min_class_recall,
              f"{value:.3f} (need {criteria.min_class_recall})" if value is not None else "none")
    check("cross activations", recall["cross_activations"] <= criteria.max_cross_activations,
          f"{recall['cross_activations']} of {recall['gestures']} gesture takes "
          f"(allowed {criteria.max_cross_activations})")  # fmt: skip
    check("gesture takes available", recall["gestures"] >= 3 * criteria.min_takes_per_gesture,
          f"{recall['gestures']} replayed")  # fmt: skip
    return {"passed": all(c["passed"] for c in checks), "checks": checks,
            "per_recording_confirmed_false": per_recording_false}  # fmt: skip


# ------------------------------------------------------------------------------ the attempt
def accept_run(
    freeze_dir: Path, run_dir: Path, final_dir: Path, *, package_root: Path | None = None,
) -> Path:  # fmt: skip
    """The one attempt: score the sealed final data with the frozen settings. Returns the folder."""
    freeze = load_freeze(freeze_dir)
    verify_freeze(freeze, run_dir, package_root)
    criteria = AcceptanceCriteria(**freeze["criteria"])
    data = load_final(final_dir, freeze["created_utc"])
    problems = check_data_requirements(criteria, data.facts)
    if problems:
        raise AcceptanceDataError(
            "the final data does not meet the predeclared requirements (this attempt is not "
            "used up): " + "; ".join(problems)
        )

    out = freeze_dir / "acceptance"
    out.mkdir(exist_ok=True)
    started = out / STARTED_FILE
    try:
        with started.open("x", encoding="utf-8") as handle:
            marker = {"started_utc": _now(), "freeze_id": freeze["freeze_id"]}
            handle.write(json.dumps(marker) + "\n")
    except FileExistsError as exc:
        raise AcceptanceError(
            "acceptance has already been run for this freeze; a failed or repeated attempt "
            "needs a new final dataset after any change"
        ) from exc

    config = FeatureConfig(**freeze["feature_config"])
    systems = {s.name: s for s in load_systems(run_dir) if s.name in freeze["decisions"]}
    windows = [compute_windows(seq, config, recording_id=seq.sequence_id)
               for seq, *_ in data.streams]  # fmt: skip
    cases = build_replay_cases(data.takes, config)
    candidates: dict[str, list[dict[str, Any]]] = {}
    results: dict[str, Any] = {}
    for name, system in systems.items():
        corpus = build_system_corpus(name, system.classifier, windows, cases)
        result = evaluate_config(corpus, decision_from_freeze(freeze, name))
        candidates[name] = [
            {"recording_id": s["recording_id"], "time_s": t, "label": label}
            for s in result.streams for t, label in zip(s["times_s"], s["labels"], strict=True)
        ]  # fmt: skip
        latencies = [o["latency_s"] for o in result.outcomes if o.get("latency_s") is not None]
        results[name] = {
            "decision": decision_from_freeze(freeze, name).to_dict(),
            "stream_candidates": {s["recording_id"]: s["candidates"] for s in result.streams},
            "total_candidates": result.candidates, "minutes": result.minutes,
            "recall": result.recall,
            "latency_after_take_end_s": {"p50": percentile(latencies, 50),
                                         "p95": percentile(latencies, 95), "n": len(latencies)},
            "outcomes": result.outcomes,
        }  # fmt: skip
    seed = int(freeze["freeze_hash"][:8], 16)
    items, key = build_review(candidates, config.window_s, seed)
    (out / ITEMS_FILE).write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    (out / KEY_FILE).write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    (out / RUN_FILE).write_text(json.dumps({
        "freeze_id": freeze["freeze_id"], "ran_utc": _now(), "facts": data.facts,
        "primary_system": freeze["primary_system"], "systems": results,
        "review_items": len(items),
        "note": "Candidates are not yet reviewed: the verdict needs every item marked.",
    }, indent=2) + "\n", encoding="utf-8")  # fmt: skip
    return out


def accept_finalize(freeze_dir: Path) -> Path:
    """Read the review marks and write the verdict (once)."""
    freeze = load_freeze(freeze_dir)
    out = freeze_dir / "acceptance"
    if not (out / MARKS_FILE).is_file():
        raise AcceptanceError("missing or unreadable acceptance file: "
                              f"{MARKS_FILE} (review the candidates first)")
    try:
        run = json.loads((out / RUN_FILE).read_text("utf-8"))
        items = json.loads((out / ITEMS_FILE).read_text("utf-8"))
        key = json.loads((out / KEY_FILE).read_text("utf-8"))
        marks = load_marks(out / MARKS_FILE, items, file_sha256(out / ITEMS_FILE))
    except (OSError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"missing or unreadable acceptance file: {exc}") from exc
    except ReviewError as exc:
        raise ReviewIncompleteError(f"the review marks were rejected: {exc}") from exc
    verdict_path = out / VERDICT_FILE
    if verdict_path.exists():
        raise AcceptanceError("the verdict was already written; it is not recomputed")
    criteria = AcceptanceCriteria(**freeze["criteria"])
    verdicts: dict[str, Any] = {}
    for name, result in run["systems"].items():
        counts = confirmed_false_counts(key, items, marks, name)
        per_recording = {r: v["false"] + v["ambiguous"] for r, v in counts["per_recording"].items()}
        for recording_id in result["stream_candidates"]:
            per_recording.setdefault(recording_id, 0)
        judged = judge(criteria, run["facts"], counts["confirmed_false_for_acceptance"],
                       per_recording, result["recall"])  # fmt: skip
        verdicts[name] = {
            **judged, "review": counts,
            "false_activation_rate": rate_summary(counts["confirmed_false_for_acceptance"],
                                                  result["minutes"]),
            "caution": "A pass is evidence from one participant, one camera and "
                       f"{result['minutes']:.0f} minutes; it is not proof of zero real-world risk.",
        }  # fmt: skip
    primary = freeze["primary_system"]
    payload = {"freeze_id": freeze["freeze_id"], "decided_utc": _now(),
               "primary_system": primary, "accepted": verdicts[primary]["passed"],
               "systems": verdicts}  # fmt: skip
    verdict_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return verdict_path

