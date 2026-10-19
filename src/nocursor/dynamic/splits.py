"""Session-grouped, deterministic train / validation / test splitting, and the frozen manifest.

Why whole sessions: takes from one sitting share lighting, posture and habits, so splitting their
takes at random would let a model "recognise the session" and overstate how well it generalises.
Every session therefore lands wholly in one split.

Roles:

* **full** sessions (enough takes of every gesture) are shuffled with a seeded generator and dealt
  out to test, then validation, then train.
* **pilot** sessions (a few takes) may only add training examples.
* the continuous **stream** is reserved for the final false-activation evaluation. It is in no
  split and no loader opens it unless asked by name.

The result is written once, as ``split.json``, before any model is trained. Training and
evaluation re-verify it mechanically against the current files; if a dataset file changed, they
refuse to run rather than silently use different data. Takes that cannot become features are
excluded from model inputs and listed with their reason. Takes the label review flagged are never
removed or relabelled here; the manifest records how each is treated.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nocursor.dynamic.dataset import (
    Dataset,
    LoadedSession,
    content_hash,
    load_dataset,
    load_session,
)
from nocursor.dynamic.features import FeatureConfig, FeatureError, extract_features
from nocursor.dynamic.quality import review_label
from nocursor.dynamic.schema import CLASS_NAMES, KIND_STREAM, TARGET_CLASSES, SequenceRecord

SPLIT_VERSION = 1
SPLIT_FILE = "split.json"
ALGORITHM = (
    "full sessions sorted by id, shuffled with random.Random(seed); the first test_sessions go to "
    "test, the next validation_sessions to validation, the rest to train; pilot sessions train "
    "only; stream recordings are reserved"
)
FLAGGED_TREATMENT = (
    "kept (not relabelled, not removed) in every split of the primary run; removed from training "
    "and validation only in the separate sensitivity run; held-out metrics are reported with and "
    "without it"
)

SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST = "train", "validation", "test"
ROLE_TRAIN, ROLE_PILOT, ROLE_VALIDATION, ROLE_TEST, ROLE_STREAM = (
    "train", "pilot_train", "validation", "test", "reserved_stream"
)  # fmt: skip


class SplitError(ValueError):
    """The split is invalid, or no longer matches the data it was made from."""


class SplitCoverageError(SplitError):
    """There are not enough sessions or takes to make a meaningful split."""


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 0
    validation_sessions: int = 1
    test_sessions: int = 1
    min_full_takes: int = 10  # takes of each gesture for a session to count as "full"
    min_train_per_class: int = 10
    min_heldout_per_class: int = 5
    warn_heldout_per_class: int = 15

    def __post_init__(self) -> None:
        if self.validation_sessions < 1 or self.test_sessions < 1:
            raise ValueError("validation_sessions and test_sessions must be >= 1")
        floors = (self.min_full_takes, self.min_train_per_class, self.min_heldout_per_class)
        if min(floors) < 1:
            raise ValueError("minimum counts must be >= 1")


# ------------------------------------------------------------------------------ hashing
def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_hash(entries: Sequence[tuple[str, str]]) -> str:
    """Hash of every session file: changes if any recording is added, removed or edited."""
    text = "\n".join(f"{sid}:{digest}" for sid, digest in sorted(entries))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ------------------------------------------------------------------------------ creation
def _session_takes(session: LoadedSession) -> list[SequenceRecord]:
    return [s for s in session.sequences if s.kind != KIND_STREAM]


def _session_streams(session: LoadedSession) -> list[SequenceRecord]:
    return [s for s in session.sequences if s.kind == KIND_STREAM]


def create_split(
    dataset: Dataset,
    config: SplitConfig | None = None,
    feature_config: FeatureConfig | None = None,
    *,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Build the split manifest. Deterministic: the same data and seed give the same split."""
    config = config or SplitConfig()
    feature_config = feature_config or FeatureConfig()
    sessions = {s.header.session_id: s for s in dataset.sessions}
    if not sessions:
        raise SplitCoverageError("the dataset has no sessions")
    for sid, session in sessions.items():
        if _session_takes(session) and _session_streams(session):
            raise SplitError(f"session {sid} mixes takes and a continuous stream; record "
                             "streams in their own session")  # fmt: skip

    full, pilot, stream_only = [], [], []
    for sid in sorted(sessions):
        takes = _session_takes(sessions[sid])
        if not takes:
            stream_only.append(sid)
            continue
        per_class = Counter(t.label for t in takes)
        is_full = all(per_class[c] >= config.min_full_takes for c in TARGET_CLASSES)
        (full if is_full else pilot).append(sid)

    needed = config.validation_sessions + config.test_sessions + 1
    if len(full) < needed:
        raise SplitCoverageError(
            f"found {len(full)} full session(s) (at least {config.min_full_takes} takes of every "
            f"gesture); need {needed} to hold out {config.test_sessions} for test and "
            f"{config.validation_sessions} for validation and still train. Record more sessions."
        )

    order = list(full)
    random.Random(config.seed).shuffle(order)
    test_ids = sorted(order[: config.test_sessions])
    held = config.test_sessions + config.validation_sessions
    val_ids = sorted(order[config.test_sessions : held])
    train_ids = sorted(order[held:])
    roles: dict[str, str] = {sid: ROLE_TRAIN for sid in train_ids}
    roles.update({sid: ROLE_PILOT for sid in pilot})
    roles.update({sid: ROLE_VALIDATION for sid in val_ids})
    roles.update({sid: ROLE_TEST for sid in test_ids})
    roles.update({sid: ROLE_STREAM for sid in stream_only})
    split_of = {ROLE_TRAIN: SPLIT_TRAIN, ROLE_PILOT: SPLIT_TRAIN, ROLE_VALIDATION: SPLIT_VALIDATION,
                ROLE_TEST: SPLIT_TEST}  # fmt: skip

    excluded: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    split_names = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)
    kept: dict[str, Counter[str]] = {n: Counter() for n in split_names}
    raw: dict[str, Counter[str]] = {n: Counter() for n in kept}
    session_rows = []
    days: dict[str, set[str]] = {n: set() for n in kept}
    stream_rows = []
    digests = []
    for sid in sorted(sessions):
        session = sessions[sid]
        digest = file_sha256(session.path)
        digests.append((sid, digest))
        role = roles[sid]
        split = split_of.get(role)
        kept_here = 0
        for take in _session_takes(session):
            raw[split][take.label] += 1
            try:
                extract_features(take, feature_config)
            except FeatureError as exc:
                excluded.append({"sequence_id": take.sequence_id, "session_id": sid,
                                 "split": split, "label": take.label, "subtype": take.subtype,
                                 "reason": str(exc)})  # fmt: skip
                continue
            kept[split][take.label] += 1
            kept_here += 1
            note = review_label(take, feature_config)
            if note:
                flagged.append({"sequence_id": take.sequence_id, "session_id": sid, "split": split,
                                "label": take.label, "note": note,
                                "treatment": FLAGGED_TREATMENT})  # fmt: skip
        if split:
            days[split].add(session.header.created_utc[:10])
        for stream in _session_streams(session):
            stream_rows.append({"session_id": sid, "sequence_id": stream.sequence_id,
                                "duration_s": round(stream.duration_s, 1)})  # fmt: skip
        session_rows.append({
            "session_id": sid, "participant_id": session.header.participant_id,
            "date": session.header.created_utc[:10], "provenance": session.header.provenance,
            "file_sha256": digest, "role": role,
            "takes": len(_session_takes(session)), "feature_ready_takes": kept_here,
            "class_takes": dict(Counter(t.label for t in _session_takes(session))),
            "streams": len(_session_streams(session)),
        })  # fmt: skip

    warnings = _coverage_check(kept, config)
    assignments = {ROLE_TRAIN: train_ids, ROLE_PILOT: sorted(pilot), ROLE_VALIDATION: val_ids,
                   ROLE_TEST: test_ids, ROLE_STREAM: sorted(stream_only)}  # fmt: skip
    manifest = manifest_hash(digests)
    identity = {"version": SPLIT_VERSION, "seed": config.seed, "config": asdict(config),
                "assignments": assignments, "dataset_manifest_hash": manifest,
                "feature_schema_hash": feature_config.schema_hash()}  # fmt: skip
    split_id = "split-" + hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()[:10]
    return {
        "split_version": SPLIT_VERSION,
        "split_id": split_id,
        "created_utc": created_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": config.seed,
        "algorithm": ALGORITHM,
        "config": asdict(config),
        "feature_schema_hash": feature_config.schema_hash(),
        "feature_config": feature_config.to_dict(),
        "dataset_manifest_hash": manifest,
        "sessions": session_rows,
        "assignments": assignments,
        "counts": {n: {c: kept[n][c] for c in CLASS_NAMES} for n in kept},
        "counts_before_exclusion": {n: {c: raw[n][c] for c in CLASS_NAMES} for n in raw},
        "excluded": excluded,
        "flagged": flagged,
        "reserved_stream": stream_rows,
        "dates": {n: sorted(days[n]) for n in days},
        "warnings": warnings,
        "notes": [
            "Held-out evidence is limited: one validation session and one test session from a "
            "single participant, so results describe cross-session behaviour for that person only.",
            "The stream is reserved for the final false-activation evaluation and is never used "
            "for training, model selection or threshold tuning.",
        ],
    }


def _coverage_check(kept: dict[str, Counter[str]], config: SplitConfig) -> list[str]:
    warnings: list[str] = []
    for name, counts in kept.items():
        floor = config.min_train_per_class if name == SPLIT_TRAIN else config.min_heldout_per_class
        for label in CLASS_NAMES:
            if counts[label] == 0:
                raise SplitCoverageError(f"the {name} split has no {label} takes after exclusions")
            if counts[label] < floor:
                raise SplitCoverageError(
                    f"the {name} split has only {counts[label]} {label} takes after exclusions "
                    f"(need at least {floor})"
                )
            if name != SPLIT_TRAIN and counts[label] < config.warn_heldout_per_class:
                warnings.append(f"{name}: {label} has {counts[label]} takes, so its metrics are "
                                "indicative only")  # fmt: skip
    return warnings


# ------------------------------------------------------------------------- verification
def check_manifest(split: dict[str, Any]) -> list[str]:
    """Consistency of the manifest itself (no data files are opened)."""
    problems: list[str] = []
    if split.get("split_version") != SPLIT_VERSION:
        return [f"unsupported split_version {split.get('split_version')!r}"]
    assign = split["assignments"]
    seen: dict[str, str] = {}
    for role, ids in assign.items():
        for sid in ids:
            if sid in seen:
                problems.append(f"session {sid} is assigned to both {seen[sid]} and {role}")
            seen[sid] = role
    listed = {row["session_id"] for row in split["sessions"]}
    if listed != set(seen):
        problems.append("the session list and the assignments disagree")
    held_out = set(assign[ROLE_VALIDATION]) | set(assign[ROLE_TEST])
    if held_out & set(assign[ROLE_PILOT]):
        problems.append("a pilot session is in validation or test")
    minimum = split.get("config", {}).get("min_full_takes", 1)
    rows = {row["session_id"]: row for row in split["sessions"]}
    for sid in sorted(held_out):
        per_class = rows.get(sid, {}).get("class_takes")
        if per_class is not None and any(per_class.get(c, 0) < minimum for c in TARGET_CLASSES):
            problems.append(f"session {sid} is in validation or test but is too small to be a "
                            f"full session (needs {minimum} takes of every gesture)")
    if set(assign[ROLE_VALIDATION]) & set(assign[ROLE_TEST]):
        problems.append("a session is in both validation and test")
    if set(assign[ROLE_STREAM]) & (set(assign[ROLE_TRAIN]) | held_out | set(assign[ROLE_PILOT])):
        problems.append("the reserved stream session is also in a split")
    for name, counts in split["counts"].items():
        for label in CLASS_NAMES:
            if counts.get(label, 0) < 1:
                problems.append(f"the {name} split has no {label} takes")
    return problems


def check_files(split: dict[str, Any], dataset_dir: Path) -> list[str]:
    """Compare every session file's SHA-256 with the manifest. Bytes are hashed, never parsed."""
    problems = []
    digests = []
    for row in split["sessions"]:
        path = dataset_dir / f"{row['session_id']}.jsonl"
        if not path.is_file():
            problems.append(f"session file for {row['session_id']} is missing")
            continue
        digest = file_sha256(path)
        digests.append((row["session_id"], digest))
        if digest != row["file_sha256"]:
            problems.append(f"{path.name} changed since the split was made")
    if not problems and manifest_hash(digests) != split["dataset_manifest_hash"]:
        problems.append("the dataset manifest hash does not match")
    on_disk = {p.stem for p in dataset_dir.glob("*.jsonl")}
    extra = on_disk - {row["session_id"] for row in split["sessions"]}
    if extra:
        problems.append(f"session files not covered by the split: {', '.join(sorted(extra))}")
    return problems


def verify_manifest_and_files(split: dict[str, Any], dataset_dir: Path) -> None:
    """The check training uses: it confirms the frozen split still matches the files on disk
    without parsing any test or stream data."""
    problems = check_manifest(split) + check_files(split, dataset_dir)
    if problems:
        raise SplitError("split verification failed: " + "; ".join(problems))


def verify_split(split: dict[str, Any], dataset: Dataset) -> None:
    """Mechanically check the manifest against the data; raises ``SplitError`` listing problems."""
    problems: list[str] = check_manifest(split)  # roles, pilots, the stream, class counts
    sessions = {s.header.session_id: s for s in dataset.sessions}
    digests = {sid: file_sha256(s.path) for sid, s in sessions.items()}
    if manifest_hash(list(digests.items())) != split["dataset_manifest_hash"]:
        problems.append("the dataset files changed since this split was made (hash mismatch)")
    assign = split["assignments"]
    for sid in assign.get(ROLE_TRAIN, []) + assign.get(ROLE_PILOT, []) + assign.get(
            ROLE_VALIDATION, []) + assign.get(ROLE_TEST, []) + assign.get(ROLE_STREAM, []):
        if sid not in sessions:
            problems.append(f"session {sid} is in the split but not in the dataset")
    for sid in sessions:
        if not any(sid in ids for ids in assign.values()):
            problems.append(f"session {sid} is in the dataset but in no role")

    excluded = {e["sequence_id"] for e in split["excluded"]}
    split_sequences: dict[str, set[str]] = {SPLIT_TRAIN: set(), SPLIT_VALIDATION: set(),
                                            SPLIT_TEST: set()}  # fmt: skip
    split_content: dict[str, set[str]] = {n: set() for n in split_sequences}
    stream_ids: set[str] = set()
    stream_content: set[str] = set()
    role_to_split = {ROLE_TRAIN: SPLIT_TRAIN, ROLE_PILOT: SPLIT_TRAIN,
                     ROLE_VALIDATION: SPLIT_VALIDATION, ROLE_TEST: SPLIT_TEST}  # fmt: skip
    for role, ids in assign.items():
        for sid in ids:
            session = sessions.get(sid)
            if session is None:
                continue
            for seq in session.sequences:
                if seq.kind == KIND_STREAM:
                    stream_ids.add(seq.sequence_id)
                    stream_content.add(content_hash(seq))
                elif role in role_to_split and seq.sequence_id not in excluded:
                    split_sequences[role_to_split[role]].add(seq.sequence_id)
                    split_content[role_to_split[role]].add(content_hash(seq))
    names = list(split_sequences)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if split_sequences[a] & split_sequences[b]:
                problems.append(f"sequences are shared between {a} and {b}")
            if split_content[a] & split_content[b]:
                problems.append(f"identical landmark data appears in both {a} and {b}")
        if split_sequences[a] & stream_ids or split_content[a] & stream_content:
            problems.append(f"the reserved stream overlaps the {a} split")

    if problems:
        raise SplitError("split verification failed: " + "; ".join(problems))

    fresh = create_split(
        dataset, SplitConfig(**split["config"]), FeatureConfig(**split["feature_config"]),
        created_utc=split["created_utc"],
    )  # fmt: skip
    for key in ("split_id", "assignments", "counts", "excluded", "flagged"):
        if fresh[key] != split[key]:
            raise SplitError(f"split verification failed: {key} no longer matches a fresh "
                             "recomputation from the data")  # fmt: skip


# ------------------------------------------------------------------------------ storage
def split_path(runs_dir: Path, split_id: str) -> Path:
    return runs_dir / "splits" / split_id / SPLIT_FILE


def write_split(split: dict[str, Any], runs_dir: Path) -> tuple[Path, bool]:
    """Freeze the split on disk. Returns (path, newly_written). An existing identical split is
    left untouched; a different one with the same id cannot happen unless a file was edited."""
    path = split_path(runs_dir, split["split_id"])
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        same = (existing.get("split_id") == split["split_id"]
                and existing["assignments"] == split["assignments"])
        if not same:
            raise SplitError(f"{path.name} exists but differs; not overwriting a frozen split")
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, True


def read_split(reference: str | Path, runs_dir: Path) -> tuple[dict[str, Any], Path]:
    """Load a split from a file path or from a split id under ``runs_dir``."""
    candidate = Path(reference)
    path = candidate if candidate.suffix == ".json" else split_path(runs_dir, str(reference))
    if not path.is_file():
        raise SplitError(f"split not found: {reference} (looked for {path.name})")
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitError(f"cannot read split {path.name}: {exc}") from exc


# ------------------------------------------------------------------------------ loading
@dataclass
class SplitItem:
    sequence: SequenceRecord
    split: str
    session_id: str
    features: np.ndarray
    flagged: bool


def _sessions_with_role(split: dict[str, Any], roles: Sequence[str]) -> list[str]:
    return sorted(sid for role in roles for sid in split["assignments"][role])


def _load_verified(split: dict[str, Any], dataset_dir: Path, session_id: str) -> LoadedSession:
    path = dataset_dir / f"{session_id}.jsonl"
    expected = next(r["file_sha256"] for r in split["sessions"] if r["session_id"] == session_id)
    if not path.is_file():
        raise SplitError(f"session file for {session_id} is missing")
    if file_sha256(path) != expected:
        raise SplitError(f"{path.name} changed since the split was made (hash mismatch)")
    return load_session(path, strict=True)


def load_split_items(
    split: dict[str, Any],
    dataset_dir: Path,
    names: Sequence[str],
    *,
    allow_test: bool = False,
    exclude_flagged: bool = False,
) -> list[SplitItem]:
    """Feature-ready takes of the named splits, opening only the files those splits need.

    Reading the test split needs ``allow_test=True``, which only the evaluation code passes. The
    continuous stream is never returned here; see ``load_stream``.
    """
    if SPLIT_TEST in names and not allow_test:
        raise SplitError("the test split is reserved for evaluation and cannot be loaded here")
    role_map = {SPLIT_TRAIN: (ROLE_TRAIN, ROLE_PILOT), SPLIT_VALIDATION: (ROLE_VALIDATION,),
                SPLIT_TEST: (ROLE_TEST,)}  # fmt: skip
    config = FeatureConfig(**split["feature_config"])
    excluded = {e["sequence_id"] for e in split["excluded"]}
    flagged = {f["sequence_id"] for f in split["flagged"]}
    items: list[SplitItem] = []
    for name in names:
        for sid in _sessions_with_role(split, role_map[name]):
            for seq in _session_takes(_load_verified(split, dataset_dir, sid)):
                if seq.sequence_id in excluded:
                    continue
                if exclude_flagged and seq.sequence_id in flagged:
                    continue
                items.append(SplitItem(seq, name, sid, extract_features(seq, config),
                                       seq.sequence_id in flagged))  # fmt: skip
    return items


def load_stream(split: dict[str, Any], dataset_dir: Path) -> list[SequenceRecord]:
    """The reserved continuous recording(s). Called only by the final false-activation analysis."""
    streams: list[SequenceRecord] = []
    for sid in _sessions_with_role(split, (ROLE_STREAM,)):
        streams.extend(_session_streams(_load_verified(split, dataset_dir, sid)))
    return streams


def load_for_split(dataset_dir: Path) -> Dataset:
    return load_dataset(dataset_dir, strict=True)
