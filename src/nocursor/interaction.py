"""The authoritative gesture state machine.

One machine owns the whole interaction lifecycle. It consumes one ``MachineInput``
per frame and returns the desktop ``Action`` s to perform; it never touches the
desktop itself (see ``SafeController`` for that).

Priority of mutually exclusive situations, highest first:

1. STOPPED (terminal) and PAUSED.
2. Unreliable tracking (lost / uncertain / still acquiring): SEARCHING. Any held
   button is released; nothing else happens.
3. A held pinch (PRESSED / DRAGGING) continues until the pinch is confirmed
   released. Nothing, including a scroll pose, can preempt it.
4. A stable two-finger scroll pose (the processor only reports it when the
   thumb and index finger are apart and no button is held), which also keeps
   the pinch detector disarmed.
5. A confirmed pinch-down starts PRESSED.
6. Pointing: the cursor follows the fingertip, except while the fingers are
   closing (cursor frozen) or while waiting for an open hand (REARMING).

Anything ambiguous falls through to the safest outcome: no scroll, no press.

Invariants: at most one press is outstanding (``holding``); every MouseDown has
exactly one later MouseUp; an interrupted interaction is never resumed (after
tracking loss, pause or reset the pinch detector must see an open hand again).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from nocursor.gestures import PinchResult, PinchState
from nocursor.scroll import ScrollTracker
from nocursor.tracking import TrackingState


class InteractionState(Enum):
    SEARCHING = "searching"  # tracking not reliable (lost / uncertain / acquiring)
    REARMING = "rearming"  # tracking OK, waiting for an open hand before pinches count
    POINTER = "pointer"  # armed and pointing
    PINCH_PENDING = "pinch pending"  # fingers closing / confirming, or pinch refused (cooldown)
    PRESSED = "pressed"  # button down, pointer frozen, no drag yet
    DRAGGING = "dragging"  # button down, pointer follows the fingertip
    SCROLL_PENDING = "scroll pending"  # two-finger pose seen, not yet stable
    SCROLL = "scroll"  # scrolling
    PAUSED = "paused"
    STOPPED = "stopped"  # terminal


@dataclass(frozen=True)
class MoveTo:
    x: int
    y: int


@dataclass(frozen=True)
class MouseDown:
    """Press and hold the left button."""


@dataclass(frozen=True)
class MouseUp:
    drag: bool  # False: nothing moved while pressed, so the press/release is a plain click
    reason: str = ""


@dataclass(frozen=True)
class Scroll:
    clicks: int  # positive = wheel up


@dataclass(frozen=True)
class RightClick:
    """One complete right-button click. The right button is never held, so it cannot be left
    pressed by tracking loss, a pause or a stop."""


Action = MoveTo | MouseDown | MouseUp | Scroll | RightClick


@dataclass(frozen=True)
class MachineInput:
    now: float
    tracking: TrackingState
    paused: bool = False
    target: tuple[float, float] | None = None  # smoothed screen position of the fingertip
    pinch: PinchResult | None = None
    scroll_pose: bool = False  # stable-pose candidate, already excluded when pinching
    scroll_y: float | None = None  # smoothed vertical fingertip position
    scroll_scale: float | None = None  # hand size for normalising scroll movement
    right: PinchResult | None = None  # right-click pinch (None: disabled or not applicable)


class InteractionMachine:
    def __init__(
        self,
        scroll: ScrollTracker,
        drag_start_px: float = 30.0,
        drag_start_frames: int = 3,
        scroll_stable_frames: int = 6,
    ) -> None:
        if drag_start_px <= 0:
            raise ValueError("drag_start_px must be > 0")
        if drag_start_frames < 1:
            raise ValueError("drag_start_frames must be >= 1")
        if scroll_stable_frames < 1:
            raise ValueError("scroll_stable_frames must be >= 1")
        self._scroll = scroll
        self._drag_start_px = drag_start_px
        self._drag_start_frames = drag_start_frames
        self._scroll_stable_frames = scroll_stable_frames
        self._state = InteractionState.SEARCHING
        self._holding = False
        self._dragging = False
        self._anchor: tuple[float, float] = (0.0, 0.0)
        self._drag_frames = 0
        self._scroll_frames = 0

    @property
    def state(self) -> InteractionState:
        return self._state

    @property
    def holding(self) -> bool:
        """True while this machine believes a mouse press is outstanding."""
        return self._holding

    # ------------------------------------------------------------- lifecycle
    def halt(self, reason: str) -> tuple[Action, ...]:
        """Enter the terminal STOPPED state; returns the release for a held press."""
        actions = self._release_actions(reason)
        self._clear()
        self._state = InteractionState.STOPPED
        return actions

    def reset(self, reason: str) -> tuple[Action, ...]:
        """Drop any interaction (e.g. calibration starts). STOPPED stays STOPPED."""
        if self._state is InteractionState.STOPPED:
            return ()
        actions = self._release_actions(reason)
        self._clear()
        self._state = InteractionState.SEARCHING
        return actions

    def abort_interaction(self) -> None:
        """The desktop refused the press: forget it, and do not retry until re-armed."""
        if self._holding:
            self._clear()
            self._state = InteractionState.REARMING

    # ------------------------------------------------------------------ step
    def step(self, inp: MachineInput) -> tuple[Action, ...]:
        if self._state is InteractionState.STOPPED:
            return ()
        if inp.paused:
            return self._abandon(InteractionState.PAUSED, "paused")
        if (
            inp.tracking is not TrackingState.TRACKING
            or inp.target is None
            or inp.pinch is None
        ):
            return self._abandon(InteractionState.SEARCHING, "tracking lost or unreliable")
        if self._holding:
            return self._step_held(inp.target, inp.pinch)
        return self._step_free(inp, inp.target, inp.pinch)

    def _step_held(self, target: tuple[float, float], pinch: PinchResult) -> tuple[Action, ...]:
        if not pinch.held or pinch.released:
            actions: list[Action] = [MouseUp(drag=self._dragging, reason="pinch released")]
            self._clear()
            if not pinch.freeze_cursor:  # pointer resumes on the release frame, after mouse-up
                actions.append(_move(target))
            self._state = self._free_state(pinch)
            return tuple(actions)

        if self._state is InteractionState.PRESSED:
            moved = math.hypot(target[0] - self._anchor[0], target[1] - self._anchor[1])
            self._drag_frames = self._drag_frames + 1 if moved > self._drag_start_px else 0
            if self._drag_frames >= self._drag_start_frames:
                self._state = InteractionState.DRAGGING
                self._dragging = True
                return (_move(target),)
            return ()
        return (_move(target),)  # DRAGGING: the pointer tracks the fingertip

    def _step_free(
        self, inp: MachineInput, target: tuple[float, float], pinch: PinchResult
    ) -> tuple[Action, ...]:
        if inp.scroll_pose and inp.scroll_y is not None and inp.scroll_scale is not None:
            return self._step_scroll(inp)

        self._end_scroll()
        if pinch.clicked:
            self._holding = True
            self._dragging = False
            self._anchor = target
            self._drag_frames = 0
            self._state = InteractionState.PRESSED
            return (MouseDown(),)

        # The left pinch has priority; the processor makes the two exclusive, and this keeps one
        # frame from ever carrying both.
        right = inp.right
        if right is not None and right.clicked:
            self._state = InteractionState.PINCH_PENDING
            return (RightClick(),)

        self._state = self._free_state(pinch)
        right_freeze = right is not None and right.freeze_cursor
        if right_freeze and self._state is InteractionState.POINTER:
            self._state = InteractionState.PINCH_PENDING
        return () if (pinch.freeze_cursor or right_freeze) else (_move(target),)

    def _step_scroll(self, inp: MachineInput) -> tuple[Action, ...]:
        assert inp.scroll_y is not None and inp.scroll_scale is not None
        self._scroll_frames += 1
        if self._state is InteractionState.SCROLL:
            clicks = self._scroll.update(inp.scroll_y, inp.now)
            return (Scroll(clicks),) if clicks else ()
        if self._scroll_frames >= self._scroll_stable_frames:
            self._state = InteractionState.SCROLL
            self._scroll.start(inp.scroll_y, inp.scroll_scale, inp.now)
        else:
            self._state = InteractionState.SCROLL_PENDING
        return ()

    # --------------------------------------------------------------- helpers
    def _abandon(self, state: InteractionState, reason: str) -> tuple[Action, ...]:
        actions = self._release_actions(reason)
        self._clear()
        self._state = state
        return actions

    def _release_actions(self, reason: str) -> tuple[Action, ...]:
        if not self._holding:
            return ()
        return (MouseUp(drag=self._dragging, reason=reason),)

    def _clear(self) -> None:
        self._holding = False
        self._dragging = False
        self._drag_frames = 0
        self._end_scroll()

    def _end_scroll(self) -> None:
        self._scroll_frames = 0
        self._scroll.reset()

    @staticmethod
    def _free_state(pinch: PinchResult) -> InteractionState:
        if pinch.state is PinchState.REOPEN:
            return InteractionState.REARMING
        if pinch.state in (PinchState.CLOSING, PinchState.PINCHED):
            return InteractionState.PINCH_PENDING
        return InteractionState.POINTER


def _move(target: tuple[float, float]) -> MoveTo:
    return MoveTo(round(target[0]), round(target[1]))
