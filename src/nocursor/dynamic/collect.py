"""The landmark collection tool: what to record, in what order, and what the screen says.

``CollectionSession`` holds all the decisions and is pure (a fake clock and fake observations
drive it in the tests). ``run_collect`` wires it to the camera, the MediaPipe tracker and an
OpenCV window. Nothing here can send desktop input: the tool never imports the controller, and
the only thing it writes is landmark numbers.
"""

from __future__ import annotations

import hashlib
import random
import secrets
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from nocursor import __version__
from nocursor.dynamic.dataset import SessionWriter
from nocursor.dynamic.features import FeatureConfig
from nocursor.dynamic.quality import DEFAULT_MIN_PALM_FRACTION, review_label
from nocursor.dynamic.recorder import RecorderState, TakeOutcome, TakeRecorder
from nocursor.dynamic.schema import (
    CLASS_NAMES,
    DEFAULT_WINDOW_S,
    HANDS,
    KIND_STREAM,
    MAX_CONDITIONS_CHARS,
    NEGATIVE_LABEL,
    PARTICIPANT_PATTERN,
    SCHEMA_VERSION,
    STREAM_DURATION_RANGE_S,
    TAKE_NEGATIVE_SUBTYPES,
    TARGET_CLASSES,
    DatasetError,
    SequenceRecord,
    SessionHeader,
)
from nocursor.hud import Tone
from nocursor.models import HandObservation

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 64
EXIT_INTERRUPTED = 130

MAX_STREAM_SECONDS = 600.0
FOOTER = "Space record | Backspace delete/cancel | N skip | T type | 1-4 label (free) | Esc quit"
PRIVACY_LINE = "Landmarks only: no video or images are saved"
SHORT_LABELS = {"swipe_left": "left", "swipe_right": "right", "circle_clockwise": "circle",
                "no_gesture": "negative"}  # fmt: skip

POSE_LINE = "Open palm toward the camera, all four fingers extended."
INSTRUCTIONS: dict[str, tuple[str, ...]] = {
    "swipe_left": (
        POSE_LINE,
        "Sweep ONCE to YOUR LEFT (left on this mirrored screen), about two palm-lengths, "
        "then stop.",
        "Do not swipe back quickly. Lower or hold the hand after the sweep.",
    ),
    "swipe_right": (
        POSE_LINE,
        "Sweep ONCE to YOUR RIGHT (right on this mirrored screen), about two palm-lengths, "
        "then stop.",
        "Do not swipe back quickly. Lower or hold the hand after the sweep.",
    ),
    "circle_clockwise": (
        POSE_LINE,
        "Draw ONE full circle CLOCKWISE as it appears on this mirrored screen, about "
        "one to two palm-lengths across.",
        "Start at the top and finish near where you started.",
    ),
    "no_gesture/idle": ("NOT a gesture: hold your hand still in view, any pose.",),
    "no_gesture/pointing": (
        "NOT a gesture: point with the index finger and move it around as you would steer the "
        "cursor, including sweeps left and right and circles.",
    ),
    "no_gesture/pinch": (
        "NOT a gesture: pinch thumb and index together and release, as for a click.",
    ),
    "no_gesture/drag": (
        "NOT a gesture: pinch and move the hand sideways or in circles while pinched.",
    ),
    "no_gesture/scroll": (
        "NOT a gesture: two-finger scroll pose (index and middle up, ring and little folded), "
        "moving up and down.",
    ),
    "no_gesture/open_palm_other": (
        "NOT a target gesture: open palm doing something else - up/down sweeps, a wave, a slow "
        "drift, a half circle, a COUNTER-clockwise circle, a tiny wiggle.",
    ),
    "no_gesture/ordinary_movement": (
        "NOT a gesture: ordinary hand movement - reach, adjust, rub, gesture while talking, any "
        "pose. Avoid a clean open-palm swipe or clockwise circle.",
    ),
    "no_gesture/entering_leaving": (
        "NOT a gesture: bring your hand into view, or take it out of view, at natural speed.",
    ),
    "no_gesture/continuous_use": (
        "NOT a gesture: use your hands as you normally would at the computer for the whole "
        "recording. Keep a hand in view most of the time.",
    ),
}


class CollectCommand(Enum):
    START = "start"
    UNDO = "undo"
    SKIP = "skip"
    SUBTYPE = "subtype"
    LABEL_1 = "label1"
    LABEL_2 = "label2"
    LABEL_3 = "label3"
    LABEL_4 = "label4"


_LABEL_COMMANDS = {
    CollectCommand.LABEL_1: 0,
    CollectCommand.LABEL_2: 1,
    CollectCommand.LABEL_3: 2,
    CollectCommand.LABEL_4: 3,
}


@dataclass(frozen=True)
class PlanItem:
    label: str
    subtype: str | None = None

    @property
    def key(self) -> str:
        return self.label if self.subtype is None else f"{self.label}/{self.subtype}"


def build_protocol(per_class: int, negatives: int, seed: int) -> list[PlanItem]:
    """A shuffled-in-blocks recording plan, deterministic for a given ``seed``.

    Each block holds one take of every target gesture plus an even share of the negatives, in a
    random order, so no gesture is always recorded first or last and the session stays balanced
    even if you stop part-way. Negatives cycle through every hard-negative type.
    """
    if per_class < 1:
        raise ValueError("per_class must be >= 1")
    if negatives < 0:
        raise ValueError("negatives must be >= 0")
    rng = random.Random(seed)
    plan: list[PlanItem] = []
    subtype_counter = 0
    base, extra = divmod(negatives, per_class)
    for block in range(per_class):
        items = [PlanItem(label) for label in TARGET_CLASSES]
        for _ in range(base + (1 if block < extra else 0)):
            subtype = TAKE_NEGATIVE_SUBTYPES[subtype_counter % len(TAKE_NEGATIVE_SUBTYPES)]
            subtype_counter += 1
            items.append(PlanItem(NEGATIVE_LABEL, subtype))
        rng.shuffle(items)
        plan.extend(items)
    return plan


class TakeSink(Protocol):
    def append(self, seq: SequenceRecord) -> None: ...
    def delete_last(self) -> str | None: ...


@dataclass(frozen=True)
class CollectorHud:
    headline: str
    tone: Tone
    lines: tuple[tuple[str, Tone], ...]
    instructions: tuple[str, ...]
    footer: str = FOOTER
    privacy: str = PRIVACY_LINE
    big_text: str | None = None  # the countdown number
    progress: float | None = None  # 0..1 while recording
    recording: bool = False


@dataclass
class _Saved:
    label: str
    kind: str
    advanced: bool


class CollectionSession:
    """What is being recorded, what has been saved, and what the screen should say."""

    def __init__(
        self,
        sink: TakeSink,
        recorder: TakeRecorder,
        header: SessionHeader,
        plan: list[PlanItem] | None = None,
        *,
        feature_config: FeatureConfig | None = None,
        notice_seconds: float = 4.0,
    ) -> None:
        self._sink = sink
        self._recorder = recorder
        self._header = header
        self._plan = plan
        self._features = feature_config or FeatureConfig()
        self._notice_seconds = notice_seconds
        self._index = 0
        self._free_label = 0
        self._free_subtype = 0
        self._saved: list[_Saved] = []
        self.counts: Counter[str] = Counter()
        self.streams_saved = 0
        self.rejected = 0
        self._notice: tuple[str, Tone, float] | None = None
        self._stream_done = False

    # ------------------------------------------------------------------ state
    @property
    def protocol_mode(self) -> bool:
        return self._plan is not None

    @property
    def protocol_done(self) -> bool:
        return self._plan is not None and self._index >= len(self._plan)

    @property
    def stream_done(self) -> bool:
        return self._stream_done

    @property
    def saved_takes(self) -> int:
        return sum(self.counts.values())

    def current_item(self) -> PlanItem | None:
        if self._plan is not None:
            return self._plan[self._index] if self._index < len(self._plan) else None
        label = CLASS_NAMES[self._free_label]
        if label == NEGATIVE_LABEL:
            return PlanItem(label, TAKE_NEGATIVE_SUBTYPES[self._free_subtype])
        return PlanItem(label)

    # -------------------------------------------------------------- commands
    def start_take(self, now: float) -> None:
        if self._recorder.state is not RecorderState.IDLE:
            return
        item = self.current_item()
        if item is None:
            self._say("The protocol is complete. Press Esc to finish.", Tone.OK, now)
            return
        self._recorder.start(item.label, item.subtype, now)

    def start_stream(self, duration_s: float, now: float) -> None:
        self._recorder.start(NEGATIVE_LABEL, None, now, kind=KIND_STREAM, duration_s=duration_s)

    def undo(self, now: float) -> None:
        """Cancel the take in progress, or delete the last saved one."""
        if self._recorder.state is not RecorderState.IDLE:
            self._recorder.cancel()
            self._say("Take cancelled.", Tone.INFO, now)
            return
        removed = self._sink.delete_last()
        if removed is None:
            self._say("Nothing to delete.", Tone.INFO, now)
            return
        entry = self._saved.pop()
        if entry.kind == KIND_STREAM:
            self.streams_saved -= 1
        else:
            self.counts[entry.label] -= 1
        if entry.advanced and self._index > 0:
            self._index -= 1  # the protocol asks for that take again
        self._say(f"Deleted the last take ({entry.label}); record it again.", Tone.INFO, now)

    def skip(self, now: float) -> None:
        if self._recorder.state is not RecorderState.IDLE:
            return
        if self._plan is not None:
            if self._index < len(self._plan):
                self._index += 1
                self._say("Skipped.", Tone.INFO, now)
        else:
            self._free_label = (self._free_label + 1) % len(CLASS_NAMES)

    def select_label(self, index: int, now: float) -> None:
        if self._recorder.state is not RecorderState.IDLE:
            return
        if self._plan is not None:
            self._say("The protocol chooses the label. Use --free to pick labels yourself.",
                      Tone.INFO, now)  # fmt: skip
        elif 0 <= index < len(CLASS_NAMES):
            self._free_label = index

    def cycle_subtype(self, now: float) -> None:
        if self._recorder.state is not RecorderState.IDLE:
            return
        if self._plan is not None:
            self._say("The protocol chooses the negative type. Use --free to pick it.",
                      Tone.INFO, now)  # fmt: skip
        else:
            self._free_subtype = (self._free_subtype + 1) % len(TAKE_NEGATIVE_SUBTYPES)

    def handle(self, command: CollectCommand, now: float) -> None:
        if command is CollectCommand.START:
            self.start_take(now)
        elif command is CollectCommand.UNDO:
            self.undo(now)
        elif command is CollectCommand.SKIP:
            self.skip(now)
        elif command is CollectCommand.SUBTYPE:
            self.cycle_subtype(now)
        else:
            self.select_label(_LABEL_COMMANDS[command], now)

    # ----------------------------------------------------------------- frames
    def update(self, now: float, observation: HandObservation | None, aspect: float) -> None:
        outcome = self._recorder.feed(now, observation, aspect)
        if outcome is not None:
            self._on_outcome(outcome, now)

    def _on_outcome(self, outcome: TakeOutcome, now: float) -> None:
        seq = outcome.sequence
        if seq is None:
            self.rejected += 1
            self._say("Not saved: " + "; ".join(outcome.problems), Tone.WARN, now)
            return
        try:
            self._sink.append(seq)
        except DatasetError as exc:
            self.rejected += 1
            self._say(f"Could not save the take: {exc}", Tone.ERROR, now)
            return
        advanced = False
        if seq.kind == KIND_STREAM:
            self.streams_saved += 1
            self._stream_done = True
            self._say(f"Saved the {seq.duration_s:.0f} s continuous recording.", Tone.OK, now)
        else:
            self.counts[seq.label] += 1
            if self._plan is not None and self._index < len(self._plan):
                self._index += 1
                advanced = True
            warning = review_label(seq, self._features)
            if warning:
                self._say(f"Saved, but check it: {warning}. Backspace deletes it.", Tone.WARN, now)
            else:
                self._say(f"Saved take {self.saved_takes}: {seq.label}", Tone.OK, now)
        self._saved.append(_Saved(seq.label, seq.kind, advanced))

    # -------------------------------------------------------------------- HUD
    def _say(self, text: str, tone: Tone, now: float) -> None:
        self._notice = (text, tone, now + self._notice_seconds)

    def hud(self, now: float) -> CollectorHud:
        recorder = self._recorder
        item = self.current_item()
        stream = recorder.is_stream
        what = "continuous use" if stream else (item.key if item else "protocol complete")
        instructions = INSTRUCTIONS.get(
            "no_gesture/continuous_use" if stream else (item.key if item else ""), ()
        )

        big_text = None
        progress = None
        recording = False
        if recorder.state is RecorderState.COUNTDOWN:
            remaining = recorder.countdown_remaining(now)
            headline, tone = f"GET READY - {what}", Tone.PENDING
            big_text = str(int(remaining) + 1)
        elif recorder.state is RecorderState.RECORDING:
            headline, tone = f"RECORDING - {what}", Tone.ACTIVE
            progress = recorder.record_fraction(now)
            recording = True
        elif item is None:
            headline, tone = "PROTOCOL COMPLETE - press Esc to finish", Tone.OK
        elif not recorder.hand_usable:
            headline, tone = f"SHOW YOUR HAND - next: {what}", Tone.WARN
        else:
            headline, tone = f"READY - press Space to record: {what}", Tone.OK

        saved = " ".join(f"{SHORT_LABELS[label]} {self.counts[label]}" for label in CLASS_NAMES)
        lines: list[tuple[str, Tone]] = [
            (f"{self._header.participant_id} | hand {self._header.hand} | "
             f"{self._progress_text()}", Tone.INFO),
            (f"Saved: {saved} | streams {self.streams_saved} | rejected {self.rejected}",
             Tone.INFO),
        ]  # fmt: skip
        if self._notice and now < self._notice[2]:
            lines.append((self._notice[0], self._notice[1]))
        return CollectorHud(
            headline=headline, tone=tone, lines=tuple(lines), instructions=instructions,
            big_text=big_text, progress=progress, recording=recording,
        )

    def _progress_text(self) -> str:
        if self._recorder.is_stream:
            return "Continuous recording: press Backspace to cancel it"
        if self._plan is None:
            return "Free mode: 1-4 choose the label, T the negative type"
        return f"Protocol: take {min(self._index + 1, len(self._plan))} of {len(self._plan)}"


# --------------------------------------------------------------------------------- runner
class Camera(Protocol):
    def open(self) -> None: ...
    def read(self) -> np.ndarray | None: ...
    def release(self) -> None: ...


class Tracker(Protocol):
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandObservation | None: ...
    def close(self) -> None: ...


class CollectorView(Protocol):
    def show(
        self, frame_bgr: np.ndarray, observation: HandObservation | None, hud: CollectorHud
    ) -> bool: ...
    def take_commands(self) -> tuple[CollectCommand, ...]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class CollectOptions:
    dataset_dir: Path
    participant: str = "p01"
    hand: str = "unspecified"
    conditions: str = ""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    model_path: Path | None = None
    min_confidence: float = 0.6
    countdown_s: float = 2.0
    per_class: int = 15
    negatives: int = 30
    seed: int | None = None
    free: bool = False
    stream_seconds: float | None = None
    min_palm_fraction: float = DEFAULT_MIN_PALM_FRACTION

    def problems(self) -> list[str]:
        found = []
        if not PARTICIPANT_PATTERN.match(self.participant):
            found.append(
                "--participant must be a pseudonym of 2-32 lowercase letters, digits, '_' or '-' "
                "(for example p01); names and e-mail addresses are not accepted"
            )
        if len(self.conditions) > MAX_CONDITIONS_CHARS or "\n" in self.conditions:
            found.append(f"--conditions must be one line of at most {MAX_CONDITIONS_CHARS} chars")
        if self.hand not in HANDS:
            found.append(f"--hand must be one of {', '.join(HANDS)}")
        if self.camera_index < 0 or self.frame_width <= 0 or self.frame_height <= 0:
            found.append("camera index and frame size must be valid (index >= 0, size > 0)")
        if not (0.0 <= self.min_confidence <= 1.0):
            found.append("--min-confidence must be within [0, 1]")
        if not (0.0 <= self.countdown_s <= 10.0):
            found.append("--countdown must be within [0, 10] seconds")
        if not (1 <= self.per_class <= 500):
            found.append("--per-class must be within [1, 500]")
        if not (0 <= self.negatives <= 2000):
            found.append("--negatives must be within [0, 2000]")
        if not (0.0 < self.min_palm_fraction <= 1.0):
            found.append("--min-palm-fraction must be within (0, 1]")
        if self.stream_seconds is not None and not (
            STREAM_DURATION_RANGE_S[0] <= self.stream_seconds <= MAX_STREAM_SECONDS
        ):
            found.append(
                f"--stream-seconds must be within [{STREAM_DURATION_RANGE_S[0]:g}, "
                f"{MAX_STREAM_SECONDS:g}]"
            )
        return found


def new_session_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"s{stamp}-{secrets.token_hex(2)}"


def _mediapipe_version() -> str | None:
    from nocursor.environment import package_version

    return package_version("mediapipe")


def run_collect(
    options: CollectOptions,
    *,
    camera: Camera | None = None,
    tracker: Tracker | None = None,
    view: CollectorView | None = None,
    clock: Callable[[], float] = time.monotonic,
    out: Callable[[str], object] = print,
) -> int:
    """Run the collection loop until the window is closed or the protocol/stream finishes."""
    problems = options.problems()
    if problems:
        out("nocursor.dynamic: " + "; ".join(problems))
        return EXIT_USAGE

    model_sha: str | None = None
    created_tracker: Tracker | None = None
    try:
        if tracker is None:
            from nocursor.hand_tracker import HandTracker, resolve_model_path

            model_path = resolve_model_path(options.model_path)
            model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            tracker = created_tracker = HandTracker(model_path, options.min_confidence)
        if camera is None:
            from nocursor.camera import Camera as RealCamera

            camera = RealCamera(options.camera_index, options.frame_width, options.frame_height)
        if view is None:
            from nocursor.dynamic.view import OpenCVCollectorView

            view = OpenCVCollectorView()
    except Exception as exc:  # missing or unusable model, missing OpenCV
        out(f"nocursor.dynamic: could not start: {exc}")
        if created_tracker is not None:
            created_tracker.close()
        return EXIT_ERROR

    writer: SessionWriter | None = None
    code = EXIT_OK
    try:
        camera.open()
        first = None
        for _ in range(100):
            first = camera.read()
            if first is not None:
                break
        if first is None:
            out("nocursor.dynamic: the camera opened but delivered no frames")
            return EXIT_ERROR
        height, width = first.shape[:2]

        seed = options.seed if options.seed is not None else secrets.randbelow(2**31)
        plan = None if options.free or options.stream_seconds else build_protocol(
            options.per_class, options.negatives, seed
        )
        header = SessionHeader(
            session_id=new_session_id(),
            participant_id=options.participant,
            created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            hand=options.hand,
            provenance="recorded",
            camera={"index": options.camera_index, "frame_width": width, "frame_height": height},
            tool={
                "nocursor": __version__,
                "mediapipe": _mediapipe_version(),
                "hand_landmarker_sha256": model_sha,
                "schema_version": SCHEMA_VERSION,
            },
            protocol=_protocol_description(options, seed, plan),
            conditions=options.conditions,
        )
        writer = SessionWriter(options.dataset_dir, header)
        recorder = TakeRecorder(
            header.session_id, min_confidence=options.min_confidence,
            countdown_s=options.countdown_s, min_palm_fraction=options.min_palm_fraction,
        )  # fmt: skip
        session = CollectionSession(writer, recorder, header, plan)
        out(f"Recording session {header.session_id} (landmarks only, no video) -> {writer.path}")
        if options.stream_seconds:
            session.start_stream(options.stream_seconds, clock())

        aspect = width / height
        while True:
            frame = camera.read()
            now = clock()
            if frame is None:
                observation = None
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                aspect = frame.shape[1] / frame.shape[0]
                observation = tracker.detect(frame, int(now * 1000))
            session.update(now, observation, aspect)
            keep_going = view.show(frame, observation, session.hud(now))
            for command in view.take_commands():
                session.handle(command, now)
            if not keep_going or session.stream_done:
                break
        out(_final_report(session, writer))
    except KeyboardInterrupt:
        out("nocursor.dynamic: interrupted; recordings saved so far are kept")
        code = EXIT_INTERRUPTED
    except Exception as exc:
        out(f"nocursor.dynamic: {exc}")
        code = EXIT_ERROR
    finally:
        for closer in (view.close, tracker.close, camera.release,
                       writer.close if writer else (lambda: None)):  # fmt: skip
            try:
                closer()
            except Exception as exc:  # one failure must not skip the remaining cleanups
                out(f"nocursor.dynamic: error while closing: {exc}")
    return code


def _protocol_description(
    options: CollectOptions, seed: int, plan: list[PlanItem] | None
) -> dict[str, Any]:
    return {
        "mode": "stream" if options.stream_seconds else "free" if plan is None else "protocol",
        "seed": seed,
        "take_seconds": DEFAULT_WINDOW_S,
        "countdown_seconds": options.countdown_s,
        "planned_takes": None if plan is None else len(plan),
        "min_palm_fraction": options.min_palm_fraction,
        "min_confidence": options.min_confidence,
    }


def _final_report(session: CollectionSession, writer: SessionWriter) -> str:
    counts = ", ".join(f"{label} {session.counts[label]}" for label in CLASS_NAMES)
    if session.saved_takes == 0 and session.streams_saved == 0:
        return "Nothing was recorded, so no file was kept."
    return (
        f"Saved {session.saved_takes} takes ({counts}), {session.streams_saved} continuous "
        f"recording(s); {session.rejected} take(s) were not saved.\n"
        f"File: {writer.path}\n"
        "Next: python -m nocursor.dynamic summarize"
    )
