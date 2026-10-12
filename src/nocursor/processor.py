"""Turns hand observations into desktop intents. Pure logic: no OpenCV, no desktop I/O."""

from __future__ import annotations

from dataclasses import dataclass

from nocursor.config import Config
from nocursor.diagnostics import PinchDiagnostics, explain
from nocursor.gestures import (
    PinchDetector,
    PinchResult,
    PinchState,
    pinch_ratio,
    pinch_ratio_2d,
)
from nocursor.interaction import (
    Action,
    InteractionMachine,
    InteractionState,
    MachineInput,
    MouseDown,
    MoveTo,
)
from nocursor.mapping import CursorMapper, ViewRegion
from nocursor.models import HandObservation
from nocursor.scroll import ScrollTracker, classify_hand, finger_ratios, scroll_signal
from nocursor.smoothing import ExponentialSmoother
from nocursor.tracking import TrackingMonitor, TrackingState

_HAND_SIZE_SMOOTHING = 0.8  # slow average, so scroll scale is not driven by landmark noise


@dataclass(frozen=True)
class GestureFrame:
    """What the desktop *should* do for one frame. Executing it is someone else's job."""

    tracking: TrackingState
    state: InteractionState = InteractionState.SEARCHING
    pinch: PinchState | None = None
    pinch_ratio: float | None = None
    actions: tuple[Action, ...] = ()
    scroll_pose: bool = False
    # Pinch candidate progress (frames so far, frames needed) while a pinch is still being
    # confirmed; None otherwise. A candidate is not a click: nothing is pressed until it ends.
    pinch_progress: tuple[int, int] | None = None
    # Plain-words detail for tracking states: why a hand was rejected, or "n/m" while locking on.
    tracking_note: str | None = None

    @property
    def cursor(self) -> tuple[int, int] | None:
        """Target of this frame's last pointer move, or None to leave the cursor alone."""
        for action in reversed(self.actions):
            if isinstance(action, MoveTo):
                return action.x, action.y
        return None

    @property
    def click(self) -> bool:
        """True on the frame a pinch is confirmed (the button press). Phase 1 compatibility:
        the desktop is driven by ``actions``; a press followed by a release is a click."""
        return any(isinstance(action, MouseDown) for action in self.actions)


class GestureProcessor:
    """Composes tracking checks, cursor mapping, smoothing, pinch and scroll detection.

    Nothing is emitted (no cursor target, no press, no scroll) unless tracking is
    ``TRACKING`` and the app is not paused. Any other situation clears smoothing and
    pinch state so stale data cannot leak into the next acquisition, and the
    interaction machine releases any held button.
    """

    def __init__(
        self,
        config: Config,
        screen_size: tuple[int, int],
        region: ViewRegion | None = None,
    ) -> None:
        self._config = config
        self._screen_size = screen_size
        self._monitor = TrackingMonitor(
            config.min_confidence, config.min_hand_size, config.stable_frames
        )
        self._region = region
        self._mapper = self._make_mapper(region)
        self._smoother = ExponentialSmoother(config.smoothing)
        self._scroll_smoother = ExponentialSmoother(config.smoothing)
        self._hand_size = ExponentialSmoother(_HAND_SIZE_SMOOTHING)
        self._pinch = PinchDetector(
            threshold=config.pinch_threshold,
            release_factor=config.pinch_release_factor,
            confirm_frames=config.pinch_confirm_frames,
            cooldown_s=config.click_cooldown_s,
            release_frames=config.pinch_release_frames,
        )
        self._release_level = config.pinch_threshold * config.pinch_release_factor
        self._use_depth = config.pinch_use_depth
        self._diagnostics_enabled = config.diagnostics
        self.diagnostics: PinchDiagnostics | None = None  # latest record when enabled
        self._machine = InteractionMachine(
            ScrollTracker(
                dead_zone=config.scroll_dead_zone,
                speed=config.scroll_speed,
                interval_s=config.scroll_interval_s,
                invert=config.scroll_invert,
            ),
            drag_start_px=config.drag_start_px,
            scroll_stable_frames=config.scroll_stable_frames,
        )
        self._was_inactive = False

    # ------------------------------------------------------------ properties
    @property
    def region(self) -> ViewRegion | None:
        """The calibrated region in use, or None for the default margin-based mapping."""
        return self._region

    @property
    def state(self) -> InteractionState:
        return self._machine.state

    @property
    def depth_in_use(self) -> bool:
        """True when the experimental 3D pinch ratio is active (False: the 2D default)."""
        return self._use_depth

    @property
    def pinch_threshold(self) -> float:
        return self._pinch.snapshot(0.0).engage

    @property
    def holding_button(self) -> bool:
        """True while the state machine believes a mouse press is outstanding."""
        return self._machine.holding

    # --------------------------------------------------------------- control
    def apply_calibration(self, region: ViewRegion | None) -> None:
        """Switch to a calibrated region (or back to the default mapping with None)."""
        self._region = region
        self._mapper = self._make_mapper(region)
        self._smoother.reset()

    def halt(self, reason: str) -> tuple[Action, ...]:
        """Terminal stop: returns the release for a held press. No later frame acts."""
        return self._machine.halt(reason)

    def abort_interaction(self) -> None:
        """The desktop refused a press; drop the interaction (never retried)."""
        self._machine.abort_interaction()

    def idle(self, observation: HandObservation | None, now: float) -> GestureFrame:
        """Track the hand but never act (e.g. while calibrating)."""
        return self.process(observation, now, paused=True)

    # --------------------------------------------------------------- process
    def process(
        self, observation: HandObservation | None, now: float, *, paused: bool = False
    ) -> GestureFrame:
        if self._was_inactive and not paused:
            self._monitor.reset()  # resuming: a fresh run of stable frames is required
        self._was_inactive = paused
        tracking = self._monitor.update(observation)

        if paused or tracking is not TrackingState.TRACKING or observation is None:
            self._smoother.reset()
            self._scroll_smoother.reset()
            self._hand_size.reset()
            self._pinch.reset()
            actions = self._machine.step(MachineInput(now, tracking, paused=paused))
            if self._diagnostics_enabled:
                self.diagnostics = self._diagnose(
                    now, tracking, observation, paused=paused, ratio=None, scroll_pose=False,
                    clicked=False,
                )
            return GestureFrame(
                tracking=tracking, state=self._machine.state, actions=actions,
                tracking_note=self._tracking_note(tracking),
            )

        target = self._smoother.update(self._mapper.map(*observation.index_tip))
        ratio = pinch_ratio(observation, self._use_depth)
        hand_size = self._hand_size.update((observation.hand_size, 0.0))[0]

        # A scroll pose is only recognised with the thumb and index finger clearly apart and
        # no button held.
        scroll_pose = (
            classify_hand(observation).is_scroll_pose
            and ratio >= self._release_level
            and not self._machine.holding
        )
        scroll_y: float | None = None
        if scroll_pose:
            scroll_y = self._scroll_smoother.update((scroll_signal(observation), 0.0))[0]
        else:
            self._scroll_smoother.reset()

        # Once scroll mode is *active* the pinch detector is held disarmed, so a thumb
        # brushing a finger cannot click and the hand must be seen open again afterwards.
        # Merely *looking* like the scroll pose (the candidate frames before scroll mode
        # starts) must not touch the detector: those open-hand frames are what re-arm it,
        # and resetting it on every candidate frame left it permanently disarmed.
        if scroll_pose and self._machine.state is InteractionState.SCROLL:
            self._pinch.reset()
            pinch = PinchResult(PinchState.REOPEN, clicked=False, freeze_cursor=True)
        else:
            pinch = self._pinch.update(ratio, now)

        actions = self._machine.step(
            MachineInput(
                now, tracking, target=target, pinch=pinch, scroll_pose=scroll_pose,
                scroll_y=scroll_y, scroll_scale=hand_size,
            )
        )  # fmt: skip
        if self._machine.state is InteractionState.SCROLL:
            self._pinch.reset()  # includes the frame on which scroll mode just started
        if self._diagnostics_enabled:
            self.diagnostics = self._diagnose(
                now, tracking, observation, paused=False, ratio=ratio, scroll_pose=scroll_pose,
                clicked=pinch.clicked,
            )
        snapshot = self._pinch.snapshot(now)  # read-only
        progress = None
        if snapshot.candidate_frames > 0 and not snapshot.pinched:
            progress = (snapshot.candidate_frames, snapshot.confirm_frames)
        return GestureFrame(
            tracking=tracking,
            state=self._machine.state,
            pinch=pinch.state,
            pinch_ratio=ratio,
            actions=actions,
            scroll_pose=scroll_pose,
            pinch_progress=progress,
        )

    # --------------------------------------------------------------- helpers
    def _tracking_note(self, tracking: TrackingState) -> str | None:
        if tracking is TrackingState.UNCERTAIN:
            return self._monitor.last_reason
        if tracking is TrackingState.ACQUIRING:
            return f"{self._monitor.good_streak}/{self._monitor.stable_frames}"
        return None

    def _diagnose(
        self,
        now: float,
        tracking: TrackingState,
        observation: HandObservation | None,
        *,
        paused: bool,
        ratio: float | None,
        scroll_pose: bool,
        clicked: bool,
    ) -> PinchDiagnostics:
        """Build the read-only diagnostics record. Never alters any gesture state."""
        usable = observation is not None and tracking is TrackingState.TRACKING and not paused
        ratio_2d = pinch_ratio_2d(observation) if usable and observation else None
        ratio_3d = pinch_ratio(observation, True) if usable and observation else None
        snapshot = self._pinch.snapshot(now)
        monitor = self._monitor
        return PinchDiagnostics(
            tracking=tracking,
            state=self._machine.state,
            reason=explain(
                tracking=tracking,
                tracking_reason=monitor.last_reason,
                acquiring=f"{monitor.good_streak}/{monitor.stable_frames}",
                paused=paused,
                state=self._machine.state,
                ratio=ratio,
                ratio_2d=ratio_2d,
                depth_in_use=self._use_depth,
                scroll_candidate=scroll_pose,
                snapshot=snapshot if usable else None,
                clicked=clicked,
            ),
            ratio=ratio,
            ratio_2d=ratio_2d,
            ratio_3d=ratio_3d,
            depth_in_use=self._use_depth,
            engage=snapshot.engage,
            release=snapshot.release,
            armed=snapshot.armed if usable else None,
            candidate_frames=snapshot.candidate_frames,
            confirm_frames=snapshot.confirm_frames,
            scroll_candidate=scroll_pose,
            fingers=finger_ratios(observation) if usable and observation else (),
        )

    def _make_mapper(self, region: ViewRegion | None) -> CursorMapper:
        return CursorMapper(
            self._screen_size[0],
            self._screen_size[1],
            margin=self._config.active_margin,
            bounds=region,
        )
