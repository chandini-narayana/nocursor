"""Reading, writing and validating dynamic-gesture dataset files.

One session is one JSON Lines file, ``<session_id>.jsonl``. Line 1 is the session header; every
further line is one sequence. Writing appends one validated line per accepted take, so a crash
loses at most the take in progress. The files hold landmark numbers only: no video, no images,
no names, hosts or paths.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nocursor.dynamic.schema import (
    ASPECT_RANGE,
    CLASS_NAMES,
    ID_PATTERN,
    KIND_STREAM,
    KIND_TAKE,
    KINDS,
    MAX_COORDINATE_MAGNITUDE,
    MAX_FRAME_INTERVAL_S,
    MIN_FRAMES,
    NEGATIVE_LABEL,
    SCHEMA_VERSION,
    STREAM_DURATION_RANGE_S,
    STREAM_SUBTYPE,
    TAKE_DURATION_RANGE_S,
    TAKE_NEGATIVE_SUBTYPES,
    TARGET_CLASSES,
    DatasetError,
    SequenceRecord,
    SessionHeader,
    header_problems,
)
from nocursor.models import NUM_LANDMARKS

SESSION_RECORD = "session"
SEQUENCE_RECORD = "sequence"
FILE_SUFFIX = ".jsonl"
_COORDINATE_DECIMALS = 6  # MediaPipe's own precision is about float32; this loses nothing useful


class RecordFormatError(ValueError):
    """One line of a dataset file has the wrong structure."""


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" (the sequence or file is unusable) or "warning"
    file: str  # file name only, so reports never expose directory names
    line: int | None
    message: str
    sequence_id: str | None = None

    def __str__(self) -> str:
        where = self.file if self.line is None else f"{self.file}:{self.line}"
        return f"{self.severity}: {where}: {self.message}"


class DatasetValidationError(DatasetError):
    """Strict loading found errors. ``issues`` lists everything that was found."""

    def __init__(self, issues: list[Issue]) -> None:
        errors = [i for i in issues if i.severity == "error"]
        shown = "; ".join(str(i) for i in errors[:3])
        more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
        super().__init__(f"{len(errors)} dataset error(s): {shown}{more}")
        self.issues = issues


# ----------------------------------------------------------------------------- validation
def validate_sequence(seq: SequenceRecord) -> list[str]:
    """Everything wrong with ``seq`` as a stored record; empty when it is acceptable."""
    problems: list[str] = []
    if not ID_PATTERN.match(seq.sequence_id):
        problems.append("sequence_id must be 3-64 letters, digits, '_' or '-'")
    if not ID_PATTERN.match(seq.session_id):
        problems.append("session_id is not a valid identifier")
    if seq.kind not in KINDS:
        problems.append(f"unknown kind {seq.kind!r}")
    if seq.label not in CLASS_NAMES:
        problems.append(f"unknown label {seq.label!r} (expected one of {', '.join(CLASS_NAMES)})")
    problems += _label_problems(seq)

    if seq.n_frames < MIN_FRAMES:
        problems.append(f"only {seq.n_frames} frames (need at least {MIN_FRAMES})")
    problems += _timing_problems(seq)
    if not (ASPECT_RANGE[0] <= seq.aspect_ratio <= ASPECT_RANGE[1]):
        problems.append(f"aspect_ratio {seq.aspect_ratio:g} is outside {ASPECT_RANGE}")
    problems += _value_problems(seq)
    return problems


def _label_problems(seq: SequenceRecord) -> list[str]:
    if seq.kind == KIND_STREAM:
        problems = []
        if seq.label != NEGATIVE_LABEL:
            problems.append(f"a stream must be labelled {NEGATIVE_LABEL}")
        if seq.subtype != STREAM_SUBTYPE:
            problems.append(f"a stream's subtype must be {STREAM_SUBTYPE}")
        return problems
    if seq.label in TARGET_CLASSES and seq.subtype is not None:
        return [f"{seq.label} takes must not have a subtype (got {seq.subtype!r})"]
    if seq.label == NEGATIVE_LABEL and seq.subtype not in (None, *TAKE_NEGATIVE_SUBTYPES):
        return [f"unknown no_gesture subtype {seq.subtype!r}"]
    return []


def _timing_problems(seq: SequenceRecord) -> list[str]:
    t = seq.t_ms
    if t.size == 0:
        return []
    if not np.all(np.isfinite(t)):
        return ["timestamps contain NaN or infinity"]
    problems = []
    if t[0] != 0:
        problems.append("t_ms must start at 0 (times are relative to the first frame)")
    steps = np.diff(t)
    if np.any(steps <= 0):
        problems.append("timestamps must strictly increase")
        return problems
    if steps.size and steps.max() / 1000.0 > MAX_FRAME_INTERVAL_S:
        problems.append(
            f"a gap of {steps.max() / 1000.0:.2f} s between two frames "
            f"(limit {MAX_FRAME_INTERVAL_S} s): the camera stalled"
        )
    low, high = STREAM_DURATION_RANGE_S if seq.kind == KIND_STREAM else TAKE_DURATION_RANGE_S
    if not (low <= seq.duration_s <= high):
        problems.append(f"duration {seq.duration_s:.2f} s is outside [{low}, {high}] s")
    return problems


def _value_problems(seq: SequenceRecord) -> list[str]:
    problems = []
    lm = seq.landmarks
    nan_any = np.isnan(lm).any(axis=(1, 2))
    nan_all = np.isnan(lm).all(axis=(1, 2))
    if np.any(nan_any & ~nan_all):
        problems.append(f"frame {int(np.flatnonzero(nan_any & ~nan_all)[0])} has partly missing "
                        "landmarks (a frame is either a full hand or empty)")  # fmt: skip
    if np.isinf(lm).any():
        problems.append("landmarks contain infinity")
    else:
        finite = lm[np.isfinite(lm)]
        if finite.size and np.abs(finite).max() > MAX_COORDINATE_MAGNITUDE:
            problems.append("a landmark value is implausibly large")
    conf = seq.confidence
    if not np.all(np.isfinite(conf)) or np.any(conf < 0.0) or np.any(conf > 1.0):
        problems.append("confidence values must be finite and within [0, 1]")
    if np.any(seq.usable & nan_all):
        problems.append("a frame is marked usable but has no landmarks")
    if int(seq.usable.sum()) < 2:
        problems.append("fewer than 2 usable frames")
    return problems


# ------------------------------------------------------------------------------ encoding
def header_to_json(header: SessionHeader) -> dict[str, Any]:
    return {
        "record": SESSION_RECORD,
        "schema_version": header.schema_version,
        "session_id": header.session_id,
        "participant_id": header.participant_id,
        "created_utc": header.created_utc,
        "hand": header.hand,
        "provenance": header.provenance,
        "camera": header.camera,
        "tool": header.tool,
        "protocol": header.protocol,
        "conditions": header.conditions,
    }


def header_from_json(obj: dict[str, Any]) -> SessionHeader:
    version = obj.get("schema_version")
    if version != SCHEMA_VERSION:
        raise DatasetError(f"unsupported schema_version {version!r} (this build reads "
                           f"{SCHEMA_VERSION})")  # fmt: skip
    try:
        return SessionHeader(
            session_id=str(obj["session_id"]),
            participant_id=str(obj["participant_id"]),
            created_utc=str(obj["created_utc"]),
            hand=str(obj.get("hand", "unspecified")),
            provenance=str(obj.get("provenance", "recorded")),
            camera=_mapping(obj.get("camera")),
            tool=_mapping(obj.get("tool")),
            protocol=_mapping(obj.get("protocol")),
            conditions=str(obj.get("conditions", "")),
            schema_version=int(version),
        )
    except KeyError as exc:
        raise DatasetError(f"session header is missing {exc.args[0]!r}") from exc


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise DatasetError("camera, tool and protocol must be JSON objects")
    return value


def sequence_to_json(seq: SequenceRecord) -> dict[str, Any]:
    rows = []
    for frame in seq.landmarks:
        empty = np.isnan(frame).all()
        rows.append(None if empty else np.round(frame, _COORDINATE_DECIMALS).tolist())
    return {
        "record": SEQUENCE_RECORD,
        "sequence_id": seq.sequence_id,
        "session_id": seq.session_id,
        "kind": seq.kind,
        "label": seq.label,
        "subtype": seq.subtype,
        "aspect_ratio": round(seq.aspect_ratio, 6),
        "t_ms": [int(round(v)) for v in seq.t_ms],
        "usable": [bool(v) for v in seq.usable],
        "confidence": [round(float(v), 4) for v in seq.confidence],
        "landmarks": rows,
    }


def sequence_from_json(obj: dict[str, Any]) -> SequenceRecord:
    """Decode one sequence line; raises ``RecordFormatError`` for structural problems."""
    try:
        t_ms = _number_list(obj["t_ms"], "t_ms")
        count = len(t_ms)
        confidence = _number_list(obj["confidence"], "confidence")
        usable = obj["usable"]
        raw = obj["landmarks"]
        if not isinstance(usable, list) or not all(isinstance(v, bool) for v in usable):
            raise RecordFormatError("usable must be a list of true/false")
        if not isinstance(raw, list):
            raise RecordFormatError("landmarks must be a list with one entry per frame")
        for name, length in (("confidence", len(confidence)), ("usable", len(usable)),
                             ("landmarks", len(raw))):  # fmt: skip
            if length != count:
                raise RecordFormatError(f"{name} has {length} entries but t_ms has {count}")
        landmarks = np.full((count, NUM_LANDMARKS, 3), np.nan)
        for i, row in enumerate(raw):
            if row is None:
                continue
            try:
                frame = np.asarray(row, dtype=float)
            except (TypeError, ValueError) as exc:
                raise RecordFormatError(f"frame {i}: landmarks are not numbers") from exc
            if frame.shape != (NUM_LANDMARKS, 3):
                raise RecordFormatError(
                    f"frame {i}: landmarks must be {NUM_LANDMARKS} x 3, got {frame.shape}"
                )
            landmarks[i] = frame
        subtype = obj.get("subtype")
        if subtype is not None and not isinstance(subtype, str):
            raise RecordFormatError("subtype must be a string or null")
        return SequenceRecord(
            sequence_id=_text(obj["sequence_id"], "sequence_id"),
            session_id=_text(obj["session_id"], "session_id"),
            label=_text(obj["label"], "label"),
            t_ms=np.asarray(t_ms),
            landmarks=landmarks,
            confidence=np.asarray(confidence),
            usable=np.asarray(usable, dtype=bool),
            aspect_ratio=float(obj["aspect_ratio"]),
            subtype=subtype,
            kind=_text(obj.get("kind", KIND_TAKE), "kind"),
        )
    except KeyError as exc:
        raise RecordFormatError(f"missing field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        if isinstance(exc, RecordFormatError):
            raise
        raise RecordFormatError(str(exc)) from exc


def _number_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or not all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in value
    ):
        raise RecordFormatError(f"{name} must be a list of numbers")
    return [float(v) for v in value]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise RecordFormatError(f"{name} must be a string")
    return value


def content_hash(seq: SequenceRecord) -> str:
    """Hash of the landmark values (to spot a take that was copied into a dataset twice)."""
    canonical = np.nan_to_num(np.round(seq.landmarks, _COORDINATE_DECIMALS), nan=-999.0)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


# ------------------------------------------------------------------------------- writing
class SessionWriter:
    """Creates ``<session_id>.jsonl`` and appends validated sequences to it.

    The file is created exclusively, so an existing recording is never overwritten. ``close()``
    removes a file that only ever got its header (a session where nothing was recorded).
    """

    def __init__(self, directory: Path, header: SessionHeader) -> None:
        self.header = header
        self.path = directory / f"{header.session_id}{FILE_SUFFIX}"
        directory.mkdir(parents=True, exist_ok=True)
        try:
            self._file = self.path.open("xb")
        except FileExistsError as exc:
            raise DatasetError(f"{self.path.name} exists; refusing to overwrite it") from exc
        self._offsets: list[int] = []
        self._ids: list[str] = []
        self._write_line(header_to_json(header))

    @property
    def count(self) -> int:
        return len(self._ids)

    def append(self, seq: SequenceRecord) -> None:
        if seq.session_id != self.header.session_id:
            raise DatasetError("sequence belongs to a different session")
        if seq.sequence_id in self._ids:
            raise DatasetError(f"duplicate sequence_id {seq.sequence_id}")
        problems = validate_sequence(seq)
        if problems:
            raise DatasetError("refusing to store an invalid sequence: " + "; ".join(problems))
        offset = self._file.tell()
        self._write_line(sequence_to_json(seq))
        self._offsets.append(offset)
        self._ids.append(seq.sequence_id)

    def delete_last(self) -> str | None:
        """Remove the most recently appended sequence from the file; returns its id."""
        if not self._ids:
            return None
        offset = self._offsets.pop()
        self._file.truncate(offset)
        self._file.seek(offset)
        self._file.flush()
        return self._ids.pop()

    def close(self) -> None:
        if self._file.closed:
            return
        self._file.close()
        if not self._ids:
            self.path.unlink(missing_ok=True)

    def _write_line(self, obj: dict[str, Any]) -> None:
        text = json.dumps(obj, separators=(",", ":"), allow_nan=False)
        self._file.write(text.encode("utf-8") + b"\n")
        self._file.flush()


# ------------------------------------------------------------------------------- reading
@dataclass
class LoadedSession:
    path: Path
    header: SessionHeader
    sequences: list[SequenceRecord]
    issues: list[Issue] = field(default_factory=list)


@dataclass
class Dataset:
    sessions: list[LoadedSession]
    issues: list[Issue] = field(default_factory=list)

    def sequences(self) -> Iterator[tuple[SessionHeader, SequenceRecord]]:
        for session in self.sessions:
            for seq in session.sequences:
                yield session.header, seq

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]


def load_session(path: Path, *, strict: bool = True) -> LoadedSession:
    """Read one session file. File-level problems always raise ``DatasetError``; problems in
    individual sequences raise only when ``strict`` and are otherwise reported and skipped."""
    name = path.name
    try:
        text = path.read_bytes().decode("utf-8")
    except OSError as exc:
        raise DatasetError(f"{name}: cannot be read ({exc.strerror or exc})") from exc
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{name}: is not valid UTF-8 text") from exc

    lines = text.split("\n")
    last_content = max((i for i, ln in enumerate(lines) if ln.strip()), default=-1)
    header: SessionHeader | None = None
    sequences: list[SequenceRecord] = []
    issues: list[Issue] = []
    seen: set[str] = set()

    def issue(severity: str, number: int, message: str, seq_id: str | None = None) -> None:
        issues.append(Issue(severity, name, number, message, seq_id))

    for index, raw in enumerate(lines):
        number = index + 1
        line = raw.strip()
        if not line:
            if index < last_content:
                issue("warning", number, "blank line ignored")
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            if header is None:
                raise DatasetError(f"{name}:{number}: the header is not valid JSON") from exc
            issue("error", number, f"invalid JSON ({exc.msg}); a truncated or damaged line")
            continue
        if not isinstance(obj, dict):
            if header is None:
                raise DatasetError(f"{name}:{number}: the session header must be a JSON object")
            issue("error", number, "line is not a JSON object")
            continue
        if header is None:
            if obj.get("record") != SESSION_RECORD:
                raise DatasetError(f"{name}:{number}: the first line must be the session header")
            try:
                header = header_from_json(obj)
            except DatasetError as exc:
                raise DatasetError(f"{name}:{number}: {exc}") from exc
            continue
        if obj.get("record") != SEQUENCE_RECORD:
            issue("error", number, f"unknown record type {obj.get('record')!r}")
            continue
        try:
            seq = sequence_from_json(obj)
        except RecordFormatError as exc:
            issue("error", number, str(exc), str(obj.get("sequence_id") or "") or None)
            continue
        problems = validate_sequence(seq)
        if seq.session_id != header.session_id:
            problems.append(f"belongs to session {seq.session_id}, not {header.session_id}")
        if seq.sequence_id in seen:
            problems.append("duplicate sequence_id in this file")
        for problem in problems:
            issue("error", number, problem, seq.sequence_id)
        if problems:
            continue
        seen.add(seq.sequence_id)
        sequences.append(seq)

    if header is None:
        raise DatasetError(f"{name}: empty file (no session header)")
    problems = header_problems(header)
    if problems:
        raise DatasetError(f"{name}: " + "; ".join(problems))
    if path.stem != header.session_id:
        issue("warning", 1, f"file name does not match session_id {header.session_id}")
    session = LoadedSession(path, header, sequences, issues)
    if strict and any(i.severity == "error" for i in issues):
        raise DatasetValidationError(issues)
    return session


def load_dataset(directory: Path, *, strict: bool = True) -> Dataset:
    """Read every ``*.jsonl`` session file in ``directory``.

    With ``strict=False`` unusable files and sequences are skipped and reported in ``issues``.
    """
    if not directory.is_dir():
        raise DatasetError(f"dataset directory not found: {directory}")
    sessions: list[LoadedSession] = []
    issues: list[Issue] = []
    seen_sessions: dict[str, str] = {}
    seen_sequences: dict[str, str] = {}
    for path in sorted(directory.glob(f"*{FILE_SUFFIX}")):
        try:
            session = load_session(path, strict=strict)
        except DatasetValidationError:
            raise
        except DatasetError as exc:
            if strict:
                raise
            issues.append(Issue("error", path.name, None, str(exc)))
            continue
        issues.extend(session.issues)
        sid = session.header.session_id
        if sid in seen_sessions:
            message = f"session_id {sid} is already used by {seen_sessions[sid]}"
            issues.append(Issue("error", path.name, None, message))
            if strict:
                raise DatasetValidationError(issues)
            continue
        seen_sessions[sid] = path.name
        for seq in session.sequences:
            if seq.sequence_id in seen_sequences:
                message = (f"sequence_id {seq.sequence_id} also appears in "
                           f"{seen_sequences[seq.sequence_id]}")
                issues.append(Issue("error", path.name, None, message, seq.sequence_id))
            seen_sequences[seq.sequence_id] = path.name
        sessions.append(session)
    dataset = Dataset(sessions, issues)
    issues.extend(_duplicate_content_warnings(dataset))
    if strict and dataset.errors:
        raise DatasetValidationError(issues)
    return dataset


def _duplicate_content_warnings(dataset: Dataset) -> list[Issue]:
    hashes: dict[str, list[tuple[str, str]]] = {}
    for session in dataset.sessions:
        for seq in session.sequences:
            hashes.setdefault(content_hash(seq), []).append((session.path.name, seq.sequence_id))
    warnings = []
    for owners in hashes.values():
        if len(owners) > 1:
            ids = ", ".join(seq_id for _, seq_id in owners)
            message = f"identical landmark data in sequences {ids}"
            warnings.append(Issue("warning", owners[0][0], None, message, owners[0][1]))
    return warnings


def count_by_label(sequences: list[SequenceRecord]) -> Counter[str]:
    return Counter(seq.label for seq in sequences)


__all__ = [
    "Dataset",
    "DatasetError",
    "DatasetValidationError",
    "Issue",
    "LoadedSession",
    "RecordFormatError",
    "SessionWriter",
    "content_hash",
    "count_by_label",
    "header_from_json",
    "header_to_json",
    "load_dataset",
    "load_session",
    "sequence_from_json",
    "sequence_to_json",
    "validate_sequence",
]
