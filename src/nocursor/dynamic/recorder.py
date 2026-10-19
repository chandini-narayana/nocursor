"""Records one take at a time from a stream of hand observations. Pure logic: no camera, no UI.

Lifecycle: ``IDLE``, ``start()``, ``COUNTDOWN``, ``RECORDING``, a ``TakeOutcome``, ``IDLE``.
The caller feeds one observation per frame with its own clock reading, so tests drive it with a
fake clock. A take is accepted only if it is structurally valid and passes the quality gates in
``nocursor.dynamic.quality``; otherwise the reasons are returned and nothing is stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from nocursor.dynamic.dataset import validate_sequence
from nocursor.dynamic.features import FeatureConfig
from nocursor.dynamic.quality import DEFAULT_MIN_PALM_FRACTION, check_quality, rules_for
from nocursor.dynamic.schema import (
    CLASS_NAMES,
    KIND_STREAM,
    KIND_TAKE,
    NEGATIVE_LABEL,
    STREAM_SUBTYPE,
    SequenceRecord,
)
from nocursor.models import NUM_LANDMARKS, HandObservation
from nocursor.tracking import TrackingMonitor, TrackingState


class RecorderState(Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    RECORDING = "recording"


@dataclass(frozen=True)
class TakeOutcome:
    """The result of a finished take. ``sequence`` is set only when it was accepted."""

    sequence: SequenceRecord | None
    problems: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.sequence is not None


class TakeRecorder:
    def __init__(
        self,
        session_id: str,
        *,
        min_confidence: float = 0.6,
        min_hand_size: float = 0.05,
        countdown_s: float = 2.0,
        take_s: float | None = None,
        feature_config: FeatureConfig | None = None,
        min_palm_fraction: float = DEFAULT_MIN_PALM_FRACTION,
    ) -> None:
        if countdown_s < 0:
            raise ValueError("countdown_s must be >= 0")
        self._features = feature_config or FeatureConfig()
        self._take_s = self._features.window_s if take_s is None else take_s
        if self._take_s <= 0:
            raise ValueError("take_s must be > 0")
        self._session_id = session_id
        self._countdown_s = countdown_s
        self._min_palm_fraction = min_palm_fraction
        # One trusted frame is enough here: the per-frame trust test of the live app, without
        # its "several frames in a row" lock-on delay, which would hide the start of a gesture.
        self._monitor = TrackingMonitor(min_confidence, min_hand_size, stable_frames=1)
        self._state = RecorderState.IDLE
        self._counter = 0
        self._label = NEGATIVE_LABEL
        self._subtype: str | None = None
        self._kind = KIND_TAKE
        self._duration_s = self._take_s
        self._started = 0.0
        self._record_started = 0.0
        self.hand_usable = False  # whether the latest frame's hand passed the trust checks
        self._frames: list[tuple[float, np.ndarray | None, float, bool]] = []
        self._aspect = 4 / 3

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> RecorderState:
        return self._state

    @property
    def window_s(self) -> float:
        return self._take_s

    @property
    def is_stream(self) -> bool:
        """True while a continuous recording (not a short take) is counting down or recording."""
        return self._state is not RecorderState.IDLE and self._kind == KIND_STREAM

    def countdown_remaining(self, now: float) -> float:
        if self._state is not RecorderState.COUNTDOWN:
            return 0.0
        return max(0.0, self._countdown_s - (now - self._started))

    def record_fraction(self, now: float) -> float:
        """How far through the take, 0 to 1 (0 when not recording)."""
        if self._state is not RecorderState.RECORDING:
            return 0.0
        return min(1.0, (now - self._record_started) / self._duration_s)

    # --------------------------------------------------------------- control
    def start(
        self,
        label: str,
        subtype: str | None,
        now: float,
        *,
        kind: str = KIND_TAKE,
        duration_s: float | None = None,
    ) -> None:
        if self._state is not RecorderState.IDLE:
            raise RuntimeError("a take is already in progress")
        if label not in CLASS_NAMES:
            raise ValueError(f"unknown label {label!r}")
        if kind == KIND_STREAM:
            if duration_s is None or duration_s <= 0:
                raise ValueError("a stream needs a duration")
            label, subtype = NEGATIVE_LABEL, STREAM_SUBTYPE
        self._label, self._subtype, self._kind = label, subtype, kind
        self._duration_s = duration_s if kind == KIND_STREAM and duration_s else self._take_s
        self._started = now
        self._frames = []
        self._state = RecorderState.COUNTDOWN

    def cancel(self) -> None:
        self._state = RecorderState.IDLE
        self._frames = []

    # ------------------------------------------------------------------ frames
    def feed(
        self, now: float, observation: HandObservation | None, aspect_ratio: float
    ) -> TakeOutcome | None:
        """Process one camera frame; returns the outcome on the frame that ends a take."""
        usable = self._monitor.update(observation) is TrackingState.TRACKING
        self.hand_usable = usable
        if self._state is RecorderState.COUNTDOWN and now - self._started >= self._countdown_s:
            self._state = RecorderState.RECORDING
            self._record_started = now
            self._aspect = aspect_ratio
        if self._state is not RecorderState.RECORDING:
            return None
        landmarks = None if observation is None else observation.landmarks.copy()
        confidence = 0.0 if observation is None else float(observation.confidence)
        self._frames.append((now, landmarks, confidence, usable))
        if now - self._record_started >= self._duration_s:
            return self._finish()
        return None

    # --------------------------------------------------------------- finishing
    def _finish(self) -> TakeOutcome:
        frames, self._frames = self._frames, []
        self._state = RecorderState.IDLE
        count = len(frames)
        t0 = frames[0][0]
        t_ms = np.zeros(count)
        for i, (t, *_rest) in enumerate(frames):
            t_ms[i] = max(round((t - t0) * 1000.0), t_ms[i - 1] + 1 if i else 0.0)
        landmarks = np.full((count, NUM_LANDMARKS, 3), np.nan)
        for i, (_, lm, _, _) in enumerate(frames):
            if lm is not None:
                landmarks[i] = lm
        self._counter += 1
        seq = SequenceRecord(
            sequence_id=f"{self._session_id}-{self._counter:04d}",
            session_id=self._session_id,
            label=self._label,
            subtype=self._subtype,
            kind=self._kind,
            t_ms=t_ms,
            landmarks=landmarks,
            confidence=np.array([f[2] for f in frames]),
            usable=np.array([f[3] for f in frames], dtype=bool),
            aspect_ratio=self._aspect,
        )
        problems = validate_sequence(seq)
        if not problems:
            rules = rules_for(
                seq.label, seq.subtype, seq.kind, self._features, self._min_palm_fraction
            )
            problems = check_quality(seq, rules)
        if problems:
            return TakeOutcome(None, tuple(problems))
        return TakeOutcome(seq)
