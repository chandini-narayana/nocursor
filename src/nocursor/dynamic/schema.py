"""Constants, conventions and record types for the dynamic-gesture dataset (schema version 1).

Coordinate and direction conventions (the whole pipeline depends on these, and tests pin them):

* Raw landmarks are stored exactly as MediaPipe returned them: normalised to the *unmirrored*
  camera image, x to the right, y down. Nothing is flipped or rescaled on disk.
* Features are computed in *view* coordinates: x is mirrored (so the numbers match the mirrored
  preview the user watches) and multiplied by the aspect ratio, so one unit in x equals one unit
  in y (frame heights). In view coordinates y still points down.
* ``swipe_left`` / ``swipe_right`` name the direction the hand moves *in the mirrored preview*,
  i.e. the user's own left / right. ``circle_clockwise`` is clockwise as drawn on that preview
  (like clock hands as the user sees them), which is a positive shoelace area with y pointing down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nocursor.models import NUM_LANDMARKS

SCHEMA_VERSION = 1

KIND_TAKE = "take"  # one short, individually labelled recording of fixed length
KIND_STREAM = "stream"  # one long recording of ordinary use (negative class only)
KINDS = (KIND_TAKE, KIND_STREAM)

TARGET_CLASSES = ("swipe_left", "swipe_right", "circle_clockwise")
NEGATIVE_LABEL = "no_gesture"
CLASS_NAMES = (*TARGET_CLASSES, NEGATIVE_LABEL)

# Kinds of negative example. These are the hard negatives: movements that look similar to a
# target gesture, or that NoCursor already uses for something else.
STREAM_SUBTYPE = "continuous_use"
TAKE_NEGATIVE_SUBTYPES = (
    "idle",  # hand held still in view
    "pointing",  # index-finger pointing and moving the cursor around
    "pinch",  # pinch (click) without a sweep
    "drag",  # pinch-and-drag
    "scroll",  # two-finger scroll pose, moving up and down
    "open_palm_other",  # open palm doing something that is NOT a target gesture
    "ordinary_movement",  # reaching, waving, adjusting, fidgeting
    "entering_leaving",  # hand coming into or leaving the camera view
)
NEGATIVE_SUBTYPES = (*TAKE_NEGATIVE_SUBTYPES, STREAM_SUBTYPE)

HANDS = ("left", "right", "unspecified")
PROVENANCES = ("recorded", "synthetic")

# Length of one take, and of the window the features are computed over. Changing it changes
# the feature schema, so it is a constant, not a command-line option.
DEFAULT_WINDOW_S = 1.6

# Structural limits that a sequence must satisfy to be stored at all.
MIN_FRAMES = 8
TAKE_DURATION_RANGE_S = (0.4, 6.0)
STREAM_DURATION_RANGE_S = (5.0, 1800.0)
MAX_FRAME_INTERVAL_S = 0.5  # a longer stall between two frames makes timing meaningless
ASPECT_RANGE = (0.2, 10.0)
MAX_COORDINATE_MAGNITUDE = 5.0  # sanity bound on a stored landmark value

# Data-sufficiency guide (not accuracy claims): see docs/DYNAMIC_GESTURES.md.
MIN_SESSIONS_TO_SPLIT = 3
RECOMMENDED_SESSIONS = 6
RECOMMENDED_TARGET_TAKES_PER_SESSION = 15
RECOMMENDED_NEGATIVE_TAKES_PER_SESSION = 30

PARTICIPANT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$")
TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
MAX_CONDITIONS_CHARS = 200


class DatasetError(ValueError):
    """A dataset file or directory cannot be used; the message says why."""


@dataclass(frozen=True)
class SessionHeader:
    """First line of every dataset file: who and how, but never a name, host or path.

    ``participant_id`` is a pseudonym chosen by the person recording (for example ``p01``).
    Keep the mapping to a real person, if you need one at all, outside the dataset.
    """

    session_id: str
    participant_id: str
    created_utc: str
    hand: str = "unspecified"
    provenance: str = "recorded"
    camera: dict[str, Any] = field(default_factory=dict)
    tool: dict[str, Any] = field(default_factory=dict)
    protocol: dict[str, Any] = field(default_factory=dict)
    conditions: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        problems = header_problems(self)
        if problems:
            raise DatasetError("invalid session header: " + "; ".join(problems))


def header_problems(header: SessionHeader) -> list[str]:
    problems: list[str] = []
    if not ID_PATTERN.match(header.session_id):
        problems.append("session_id must be 3-64 letters, digits, '_' or '-'")
    if not PARTICIPANT_PATTERN.match(header.participant_id):
        problems.append(
            "participant_id must be a pseudonym of 2-32 lowercase letters, digits, '_' or '-' "
            "(for example p01); names and e-mail addresses are not accepted"
        )
    if not TIMESTAMP_PATTERN.match(header.created_utc):
        problems.append("created_utc must look like 2026-01-31T09:30:00Z")
    if header.hand not in HANDS:
        problems.append(f"hand must be one of {', '.join(HANDS)}")
    if header.provenance not in PROVENANCES:
        problems.append(f"provenance must be one of {', '.join(PROVENANCES)}")
    if len(header.conditions) > MAX_CONDITIONS_CHARS or "\n" in header.conditions:
        problems.append(f"conditions must be one line of at most {MAX_CONDITIONS_CHARS} characters")
    if header.schema_version != SCHEMA_VERSION:
        problems.append(
            f"unsupported schema_version {header.schema_version} "
            f"(this build reads {SCHEMA_VERSION})"
        )
    return problems


@dataclass(frozen=True, eq=False)
class SequenceRecord:
    """One recorded landmark sequence.

    ``landmarks`` is ``(T, 21, 3)`` in raw MediaPipe coordinates; frames where no hand was
    detected are all-NaN. ``usable`` marks frames whose hand passed NoCursor's per-frame trust
    checks (confidence, fully inside the frame, plausible size). Only landmarks are stored,
    never video frames.
    """

    sequence_id: str
    session_id: str
    label: str
    t_ms: np.ndarray
    landmarks: np.ndarray
    confidence: np.ndarray
    usable: np.ndarray
    aspect_ratio: float
    subtype: str | None = None
    kind: str = KIND_TAKE

    def __post_init__(self) -> None:
        for name, dtype in (("t_ms", float), ("landmarks", float), ("confidence", float),
                            ("usable", bool)):  # fmt: skip
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=dtype))
        object.__setattr__(self, "aspect_ratio", float(self.aspect_ratio))
        frames = self.landmarks.shape[0] if self.landmarks.ndim == 3 else -1
        if self.landmarks.shape != (frames, NUM_LANDMARKS, 3) or frames < 0:
            raise ValueError(
                f"landmarks must have shape (T, {NUM_LANDMARKS}, 3), got {self.landmarks.shape}"
            )
        for name in ("t_ms", "confidence", "usable"):
            if getattr(self, name).shape != (frames,):
                raise ValueError(
                    f"{name} must have one entry per frame ({frames}), "
                    f"got shape {getattr(self, name).shape}"
                )

    @property
    def n_frames(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.t_ms[-1] - self.t_ms[0]) / 1000.0 if self.n_frames else 0.0

    @property
    def usable_fraction(self) -> float:
        return float(self.usable.mean()) if self.n_frames else 0.0
