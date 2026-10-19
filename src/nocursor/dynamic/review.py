"""Review marks for candidate activations: identity, persistence, validation and tallies.

A *candidate* is an activation the system produced on a recording of ordinary use. Whether it was
a false activation is a human judgement, made by watching the recorded landmark skeleton (see
``review_viewer.py``). This module keeps that judgement trustworthy:

* A mark is one of ``false`` (no gesture was intended), ``genuine`` (the person did perform the
  gesture) or ``ambiguous``. Nothing else is accepted.
* A mark is stored together with the identity of the candidate it belongs to (item id, recording,
  time range, the gesture the system reported) and a hash of the exact item list, so it can never
  silently attach to a different candidate or to a regenerated list.
* An unreviewed candidate has **no entry**. Nothing is ever inferred from a model prediction, and
  counts always keep "unreviewed" apart from "confirmed false".
* The file is rewritten atomically after every mark, so an interrupted review loses nothing.

Items never name the system that produced a candidate; that mapping is kept in a separate key file
and is only read when counting. Offline only: nothing here is connected to the mouse.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nocursor.dynamic.schema import TARGET_CLASSES

MARK_FORMAT = 2
MARKS = ("false", "genuine", "ambiguous")
IDENTITY_FIELDS = ("recording_id", "predicted", "from_s", "to_s")
TIME_TOLERANCE_S = 0.06  # candidate times are stored rounded to 0.01 s
TOOL_NAME = "nocursor.dynamic stream-review"


class ReviewError(ValueError):
    """The items or marks are malformed, inconsistent, or do not belong together."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------- items
def validate_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise ReviewError("the item list must be a JSON list")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ReviewError("every item must be a JSON object")
        missing = [k for k in ("item_id", *IDENTITY_FIELDS) if k not in item]
        if missing:
            raise ReviewError(f"item {item.get('item_id', '?')} is missing {', '.join(missing)}")
        if item["item_id"] in seen:
            raise ReviewError(f"duplicate item id {item['item_id']}")
        seen.add(item["item_id"])
        if item["predicted"] not in TARGET_CLASSES:
            raise ReviewError(f"item {item['item_id']} has an unknown gesture "
                              f"{item['predicted']!r}")
        if not (isinstance(item["from_s"], int | float) and isinstance(item["to_s"], int | float)
                and item["from_s"] >= 0 and item["to_s"] > item["from_s"]):  # fmt: skip
            raise ReviewError(f"item {item['item_id']} has an invalid time range")
        if "system" in item or "systems" in item:
            raise ReviewError(f"item {item['item_id']} names a system; items must be blind")
    return items


def load_items(path: Path) -> list[dict[str, Any]]:
    try:
        return validate_items(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"cannot read the item list {path}: {exc}") from exc


# ---------------------------------------------------------------------------------- marks
def _identity(item: dict[str, Any]) -> dict[str, Any]:
    return {k: item[k] for k in IDENTITY_FIELDS}


def _check_entries(
    raw: dict[str, Any], items: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:  # fmt: skip
    by_id = {item["item_id"]: item for item in items}
    entries: dict[str, dict[str, Any]] = {}
    for item_id, entry in raw.items():
        if item_id not in by_id:
            raise ReviewError(f"a mark refers to an unknown candidate {item_id}")
        if not isinstance(entry, dict) or entry.get("mark") not in MARKS:
            raise ReviewError(f"candidate {item_id} has an invalid mark; use one of "
                              f"{', '.join(MARKS)}")  # fmt: skip
        for field, value in _identity(by_id[item_id]).items():
            if entry.get(field) != value:
                raise ReviewError(f"the mark for {item_id} was made for a different candidate "
                                  f"({field}: {entry.get(field)!r} != {value!r})")  # fmt: skip
        entries[item_id] = entry
    return entries


def load_marks(
    path: Path, items: Sequence[dict[str, Any]], items_sha256: str | None = None
) -> dict[str, str]:  # fmt: skip
    """The marks as ``{item id: mark}``, validated against the items.

    Accepts the tool's own format and a hand-written ``{item id: mark}`` file. A candidate without
    a mark is simply absent from the result.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"cannot read the marks {path}: {exc}") from exc
    if isinstance(data, dict) and data.get("format") == MARK_FORMAT:
        if items_sha256 is not None and data.get("items_sha256") != items_sha256:
            raise ReviewError("these marks were made for a different candidate list "
                              "(item list hash mismatch)")  # fmt: skip
        raw = data.get("marks")
        if not isinstance(raw, dict):
            raise ReviewError("the marks file has no 'marks' object")
        return {k: v["mark"] for k, v in _check_entries(raw, items).items()}
    if not isinstance(data, dict):
        raise ReviewError("the marks file must be a JSON object")
    known = {item["item_id"] for item in items}
    for item_id, mark in data.items():
        if item_id not in known:
            raise ReviewError(f"a mark refers to an unknown candidate {item_id}")
        if mark not in MARKS:
            raise ReviewError(f"candidate {item_id} has an invalid mark {mark!r}; use one of "
                              f"{', '.join(MARKS)}")  # fmt: skip
    return dict(data)


class MarkStore:
    """The marks of one review, saved atomically after every change."""

    def __init__(self, path: Path, items: Sequence[dict[str, Any]], items_sha256: str) -> None:
        self.path = path
        self.items = list(items)
        self._by_id = {item["item_id"]: item for item in self.items}
        self.items_sha256 = items_sha256
        self.entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ReviewError(f"cannot read the marks {path}: {exc}") from exc
            if not (isinstance(data, dict) and data.get("format") == MARK_FORMAT):
                raise ReviewError(f"{path.name} is not a review-tool marks file; refusing to "
                                  "overwrite it")  # fmt: skip
            if data.get("items_sha256") != items_sha256:
                raise ReviewError(f"{path.name} belongs to a different candidate list; refusing "
                                  "to mix them")  # fmt: skip
            raw = data.get("marks")
            if not isinstance(raw, dict):
                raise ReviewError("the marks file has no 'marks' object")
            self.entries = _check_entries(raw, self.items)

    @classmethod
    def open(cls, items_path: Path, marks_path: Path | None = None) -> MarkStore:
        items = load_items(items_path)
        path = marks_path or items_path.parent / "review_marks.json"
        return cls(path, items, file_sha256(items_path))

    def mark_of(self, item_id: str) -> str | None:
        entry = self.entries.get(item_id)
        return None if entry is None else entry["mark"]

    def plain(self) -> dict[str, str]:
        return {k: v["mark"] for k, v in self.entries.items()}

    def progress(self) -> tuple[int, int]:
        return len(self.entries), len(self.items)

    def unreviewed(self) -> list[str]:
        return [i["item_id"] for i in self.items if i["item_id"] not in self.entries]

    def set(self, item_id: str, mark: str) -> None:
        if item_id not in self._by_id:
            raise ReviewError(f"unknown candidate {item_id}")
        if mark not in MARKS:
            raise ReviewError(f"invalid mark {mark!r}; use one of {', '.join(MARKS)}")
        previous = self.entries.get(item_id)
        self.entries[item_id] = {
            **_identity(self._by_id[item_id]), "mark": mark,
            "reviewed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "revisions": 1 if previous is None else previous.get("revisions", 1) + 1,
        }  # fmt: skip
        self._save()

    def clear(self, item_id: str) -> None:
        """Return a candidate to *unreviewed*."""
        if self.entries.pop(item_id, None) is not None:
            self._save()

    def _save(self) -> None:
        payload = {"format": MARK_FORMAT, "tool": TOOL_NAME, "items_sha256": self.items_sha256,
                   "marks": {k: self.entries[k] for k in sorted(self.entries)}}  # fmt: skip
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


# ------------------------------------------------------------------------------ provenance
COMPARED_FIELDS = (*IDENTITY_FIELDS, "candidate_times_s")


def describe_review_source(
    source: dict[str, Any], current_items: Sequence[dict[str, Any]], current_sha256: str
) -> dict[str, Any]:  # fmt: skip
    """Where the review marks used for a report came from, and how that item list relates to the
    list the report itself exports.

    ``source`` is what ``--review-dir`` loaded (folder, items, marks and their hashes). The tally
    matches candidates to items by content, not by file hash or item id, so a different hash is
    allowed; this records the difference instead of hiding it.
    """
    mine = {item["item_id"]: item for item in source["items"]}
    theirs = {item["item_id"]: item for item in current_items}
    differing = []
    for item_id in sorted(mine.keys() & theirs.keys()):
        fields = {field: {"source": mine[item_id].get(field), "current": theirs[item_id].get(field)}
                  for field in COMPARED_FIELDS
                  if mine[item_id].get(field) != theirs[item_id].get(field)}  # fmt: skip
        if fields:
            differing.append({"item_id": item_id, "fields": fields})
    return {
        "folder": source["folder"],
        "items_sha256": source["items_sha256"],
        "marks_file_present": source["marks_present"],
        "marks_sha256": source["marks_sha256"],
        "items_total": len(source["items"]),
        "items_marked": len(source["marks"]),
        "matches_current_export": source["items_sha256"] == current_sha256,
        "current_export_items_sha256": current_sha256,
        "differing_items": differing,
        "items_only_in_source": sorted(mine.keys() - theirs.keys()),
        "items_only_in_current": sorted(theirs.keys() - mine.keys()),
    }


# ---------------------------------------------------------------------------------- tally
def tally_candidates(
    candidates: Sequence[dict[str, Any]], variant: str, items: Sequence[dict[str, Any]],
    key: dict[str, list[str]], marks: dict[str, str],
) -> dict[str, int]:  # fmt: skip
    """Count one variant's candidates by review outcome.

    A candidate is matched to the review item with the same recording and gesture, found by that
    variant, whose recorded candidate times include its time. A candidate with no matching item, or
    whose item has no mark, is **unreviewed**; it is never counted as false.
    """
    counts = {"candidates": 0, "false": 0, "genuine": 0, "ambiguous": 0, "unreviewed": 0,
              "not_in_review_set": 0}  # fmt: skip
    for candidate in candidates:
        counts["candidates"] += 1
        match = next(
            (item for item in items
             if item["recording_id"] == candidate["recording_id"]
             and item["predicted"] == candidate["label"]
             and variant in key.get(item["item_id"], ())
             and any(abs(t - candidate["time_s"]) <= TIME_TOLERANCE_S
                     for t in item.get("candidate_times_s", ()))),  # fmt: skip
            None,
        )
        if match is None:
            counts["not_in_review_set"] += 1
            counts["unreviewed"] += 1
            continue
        mark = marks.get(match["item_id"])
        counts[mark if mark in MARKS else "unreviewed"] += 1
    return counts
