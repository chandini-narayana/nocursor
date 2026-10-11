"""Turns hand observations into desktop intents. Pure logic: no OpenCV, no desktop I/O."""

from __future__ import annotations

from dataclasses import dataclass

from nocursor.config import Config
from nocursor.gestures import PinchDetector, PinchState, pinch_ratio
from nocursor.mapping import CursorMapper
from nocursor.models import HandObservation
from nocursor.smoothing import ExponentialSmoother
from nocursor.tracking import TrackingMonitor, TrackingState


@dataclass(frozen=True)
class GestureFrame:
    """What the desktop *should* do for one frame. Executing it is someone else's job."""

    tracking: TrackingState
    pinch: PinchState | None = None
    pinch_ratio: float | None = None
    cursor: tuple[int, int] | None = None  # target pixel, or None to leave the cursor alone
    click: bool = False


class GestureProcessor:
    """Composes tracking checks, cursor mapping, smoothing and pinch detection.

    Nothing is emitted (no cursor target, no click) unless tracking is
    ``TRACKING``. Any other state also clears smoothing and pinch state so
    stale data cannot leak into the next acquisition.
    """

    def __init__(self, config: Config, screen_size: tuple[int, int]) -> None:
        self._monitor = TrackingMonitor(
            config.min_confidence, config.min_hand_size, config.stable_frames
        )
        self._mapper = CursorMapper(screen_size[0], screen_size[1], margin=config.active_margin)
        self._smoother = ExponentialSmoother(config.smoothing)
        self._pinch = PinchDetector(
            threshold=config.pinch_threshold,
            release_factor=config.pinch_release_factor,
            confirm_frames=config.pinch_confirm_frames,
            cooldown_s=config.click_cooldown_s,
        )

    def process(self, observation: HandObservation | None, now: float) -> GestureFrame:
        tracking = self._monitor.update(observation)
        if tracking is not TrackingState.TRACKING or observation is None:
            self._smoother.reset()
            self._pinch.reset()
            return GestureFrame(tracking=tracking)

        target = self._mapper.map(*observation.index_tip)
        smoothed = self._smoother.update(target)

        ratio = pinch_ratio(observation)
        pinch = self._pinch.update(ratio, now)

        cursor = None if pinch.freeze_cursor else (round(smoothed[0]), round(smoothed[1]))
        return GestureFrame(
            tracking=tracking,
            pinch=pinch.state,
            pinch_ratio=ratio,
            cursor=cursor,
            click=pinch.clicked,
        )
