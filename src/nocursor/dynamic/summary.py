"""Dataset summary: what has been recorded, whether it is enough, and what looks suspicious.

The coverage notes compare the dataset against a data-sufficiency guide. They are not accuracy
estimates: a dataset that meets the guide can still produce a poor model, and one that does not
can still be useful for trying out the pipeline.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from nocursor.dynamic.dataset import Dataset
from nocursor.dynamic.features import (
    FeatureConfig,
    FeatureError,
    extract_features,
    failure_category,
)
from nocursor.dynamic.pose import open_palm_fraction
from nocursor.dynamic.quality import review_label
from nocursor.dynamic.schema import (
    CLASS_NAMES,
    KIND_STREAM,
    MIN_SESSIONS_TO_SPLIT,
    NEGATIVE_LABEL,
    RECOMMENDED_NEGATIVE_TAKES_PER_SESSION,
    RECOMMENDED_SESSIONS,
    RECOMMENDED_TARGET_TAKES_PER_SESSION,
    TARGET_CLASSES,
)


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def summarize(dataset: Dataset, config: FeatureConfig | None = None) -> dict[str, Any]:
    config = config or FeatureConfig()
    by_label: Counter[str] = Counter()
    by_subtype: Counter[str] = Counter()
    per_session: list[dict[str, Any]] = []
    unfeaturizable: Counter[str] = Counter()
    suspects: list[dict[str, str]] = []
    durations: list[float] = []
    usable: list[float] = []
    palm: list[float] = []
    streams = 0
    stream_seconds = 0.0
    takes = 0
    featurizable = 0

    for session in dataset.sessions:
        counts: Counter[str] = Counter()
        for seq in session.sequences:
            if seq.kind == KIND_STREAM:
                streams += 1
                stream_seconds += seq.duration_s
                continue
            takes += 1
            counts[seq.label] += 1
            by_label[seq.label] += 1
            if seq.subtype:
                by_subtype[seq.subtype] += 1
            durations.append(seq.duration_s)
            usable.append(seq.usable_fraction)
            if seq.label in TARGET_CLASSES:
                fraction = open_palm_fraction(seq)
                if fraction is not None:
                    palm.append(fraction)
            try:
                extract_features(seq, config)
                featurizable += 1
            except FeatureError as exc:
                unfeaturizable[failure_category(str(exc))] += 1
            note = review_label(seq, config)
            if note:
                suspects.append({"sequence_id": seq.sequence_id, "note": note})
        per_session.append({
            "session_id": session.header.session_id,
            "participant_id": session.header.participant_id,
            "hand": session.header.hand,
            "provenance": session.header.provenance,
            "counts": {label: counts[label] for label in CLASS_NAMES},
            "streams": sum(1 for s in session.sequences if s.kind == KIND_STREAM),
        })  # fmt: skip

    provenance = sorted({s.header.provenance for s in dataset.sessions})
    summary: dict[str, Any] = {
        "sessions": len(dataset.sessions),
        "participants": len({s.header.participant_id for s in dataset.sessions}),
        "takes": takes,
        "streams": streams,
        "stream_seconds": round(stream_seconds, 1),
        "by_label": {label: by_label[label] for label in CLASS_NAMES},
        "no_gesture_by_subtype": dict(sorted(by_subtype.items())),
        "per_session": per_session,
        "median_take_seconds": _median(durations),
        "median_usable_fraction": _median(usable),
        "median_open_palm_fraction_targets": _median(palm),
        "featurizable_takes": featurizable,
        "unfeaturizable_reasons": dict(unfeaturizable),
        "suspect_labels": suspects,
        "provenance": provenance,
        "issues": {"errors": len(dataset.errors), "warnings": len(dataset.warnings)},
    }
    summary["coverage_notes"] = coverage_notes(summary)
    return summary


def coverage_notes(summary: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    sessions = summary["sessions"]
    if "synthetic" in summary["provenance"]:
        notes.append("Contains SYNTHETIC sessions: use them for software tests only, never for "
                     "accuracy claims.")  # fmt: skip
    if sessions == 0:
        return notes + ["No sessions found."]
    if sessions < MIN_SESSIONS_TO_SPLIT:
        notes.append(
            f"{sessions} session(s): at least {MIN_SESSIONS_TO_SPLIT} are needed to make "
            "train/validation/test splits by session."
        )
    elif sessions < RECOMMENDED_SESSIONS:
        notes.append(
            f"{sessions} sessions: enough to try the pipeline, but {RECOMMENDED_SESSIONS}+ "
            "sessions are recommended before trusting any result."
        )
    if summary["participants"] < 2:
        notes.append("One participant: results can only show generalisation across sessions, "
                     "not across people.")  # fmt: skip
    for label in TARGET_CLASSES:
        want = RECOMMENDED_TARGET_TAKES_PER_SESSION * RECOMMENDED_SESSIONS
        have = summary["by_label"][label]
        if have < want:
            notes.append(f"{label}: {have} takes (guide: {want}).")
    want = RECOMMENDED_NEGATIVE_TAKES_PER_SESSION * RECOMMENDED_SESSIONS
    have = summary["by_label"][NEGATIVE_LABEL]
    if have < want:
        notes.append(f"{NEGATIVE_LABEL}: {have} takes (guide: {want}).")
    if summary["streams"] == 0:
        notes.append("No continuous-use recordings: false activations per minute cannot be "
                     "measured later without them.")  # fmt: skip
    return notes


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Dynamic gesture dataset",
        f"  sessions {summary['sessions']}, participants {summary['participants']}, "
        f"takes {summary['takes']}, continuous recordings {summary['streams']} "
        f"({summary['stream_seconds']:g} s)",
        "  takes per class:",
    ]
    lines += [f"    {label:<18}{summary['by_label'][label]:>5}" for label in CLASS_NAMES]
    if summary["no_gesture_by_subtype"]:
        kinds = summary["no_gesture_by_subtype"].items()
        lines.append("  no_gesture by type: " + ", ".join(f"{n} {c}" for n, c in kinds))
    if summary["per_session"]:
        lines.append("  per session (swipe_left/swipe_right/circle_clockwise/no_gesture, streams):")
        for row in summary["per_session"]:
            c = row["counts"]
            lines.append(
                f"    {row['session_id']}  {row['participant_id']}  hand {row['hand']}  "
                f"{c['swipe_left']}/{c['swipe_right']}/{c['circle_clockwise']}/{c['no_gesture']}"
                f"  streams {row['streams']}  [{row['provenance']}]"
            )
    if summary["takes"]:
        palm = summary["median_open_palm_fraction_targets"]
        lines.append(
            f"  median take {summary['median_take_seconds']:.2f} s, trusted frames "
            f"{summary['median_usable_fraction']:.0%}"
            + (f", open palm in gesture takes {palm:.0%}" if palm is not None else "")
        )
        lines.append(f"  takes that can be turned into features: {summary['featurizable_takes']}"
                     f" of {summary['takes']}")  # fmt: skip
        for reason, count in summary["unfeaturizable_reasons"].items():
            lines.append(f"    not featurizable ({count}): {reason}")
    if summary["suspect_labels"]:
        lines.append(f"  labels worth a second look ({len(summary['suspect_labels'])}; "
                     "warnings only, nothing was changed):")  # fmt: skip
        lines += [f"    {s['sequence_id']}: {s['note']}" for s in summary["suspect_labels"][:10]]
        if len(summary["suspect_labels"]) > 10:
            lines.append(f"    ... and {len(summary['suspect_labels']) - 10} more")
    issues = summary["issues"]
    lines.append(f"  validation: {issues['errors']} error(s), {issues['warnings']} warning(s)")
    if summary["coverage_notes"]:
        lines.append("  coverage vs the data-sufficiency guide (not accuracy estimates):")
        lines += [f"    - {note}" for note in summary["coverage_notes"]]
    return "\n".join(lines)

