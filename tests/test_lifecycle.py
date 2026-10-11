"""Click / drag lifecycle through the real processor and interaction state machine."""

import pytest

from helpers import make_hand
from nocursor.config import Config
from nocursor.gestures import PinchResult, PinchState
from nocursor.interaction import (
    InteractionMachine,
    InteractionState,
    MachineInput,
    MouseDown,
    MouseUp,
    MoveTo,
)
from nocursor.processor import GestureProcessor
from nocursor.scroll import ScrollTracker
from nocursor.tracking import TrackingState

DT = 1 / 30
SCREEN = (1920, 1080)
S = InteractionState


def pointing(x: float = 0.5):
    return make_hand(index_xy=(x, 0.5), pinch_ratio=1.0)


def pinched(x: float = 0.5):
    return make_hand(index_xy=(x, 0.5), pinch_ratio=0.10)


class Rig:
    """Feeds frames at 30 fps into a real GestureProcessor and records everything."""

    def __init__(self, **cfg):
        base = {"smoothing": 0.0, "stable_frames": 1, "active_margin": 0.0,
                "click_cooldown_s": 0.0}
        self.proc = GestureProcessor(Config(**{**base, **cfg}), SCREEN)
        self.t = 0.0
        self.frames = []

    def feed(self, hand, n: int = 1, paused: bool = False):
        out = []
        for _ in range(n):
            self.t += DT
            frame = self.proc.process(hand, self.t, paused=paused)
            self.frames.append(frame)
            out.append(frame)
        return out

    def feed_seq(self, hands):
        return [self.feed(h)[0] for h in hands]

    @property
    def actions(self):
        return [a for f in self.frames for a in f.actions]

    def count(self, kind, frames=None):
        pool = self.frames if frames is None else frames
        return sum(isinstance(a, kind) for f in pool for a in f.actions)

    def start_interaction(self):
        """Arm with an open hand, then pinch until the press is confirmed."""
        self.feed(pointing(), 3)
        return self.feed(pinched(), 5)


class TestClick:
    def test_short_deliberate_pinch_is_one_click(self):
        rig = Rig()
        rig.start_interaction()
        rig.feed(pinched(), 3)
        rig.feed(pointing(), 3)
        downs = [a for a in rig.actions if isinstance(a, MouseDown)]
        ups = [a for a in rig.actions if isinstance(a, MouseUp)]
        assert len(downs) == 1 and len(ups) == 1
        assert ups[0].drag is False  # nothing moved: a plain click
        order = [type(a) for a in rig.actions if isinstance(a, MouseDown | MouseUp)]
        assert order == [MouseDown, MouseUp]

    def test_press_happens_on_the_confirming_frame_not_earlier(self):
        rig = Rig()
        rig.feed(pointing(), 3)
        frames = rig.feed(pinched(), 5)
        assert [rig.count(MouseDown, [f]) for f in frames] == [0, 0, 0, 0, 1]
        assert frames[-1].state is S.PRESSED

    def test_pointer_is_frozen_while_pressed_without_movement(self):
        rig = Rig()
        rig.start_interaction()
        held = rig.feed(pinched(), 20)
        assert rig.count(MoveTo, held) == 0
        assert all(f.state is S.PRESSED for f in held)

    def test_holding_a_pinch_never_repeats_mouse_down(self):
        rig = Rig()
        rig.start_interaction()
        rig.feed(pinched(), 120)  # four seconds
        assert rig.count(MouseDown) == 1
        assert rig.count(MouseUp) == 0

    def test_open_hand_frames_never_repeat_mouse_up(self):
        rig = Rig()
        rig.start_interaction()
        rig.feed(pointing(), 150)
        assert rig.count(MouseDown) == 1 and rig.count(MouseUp) == 1

    def test_mouse_up_comes_before_the_pointer_resumes_on_the_release_frame(self):
        rig = Rig()
        rig.start_interaction()
        first_open, release = rig.feed(pointing(), 2)
        assert first_open.actions == ()  # release not confirmed yet: still held, still frozen
        assert [type(a) for a in release.actions] == [MouseUp, MoveTo]

    def test_one_glitch_frame_does_not_release_a_held_pinch(self):
        rig = Rig()
        rig.start_interaction()
        rig.feed(pinched(), 3)
        rig.feed(pointing(), 1)  # a single spurious "open" frame
        rig.feed(pinched(), 5)
        assert rig.count(MouseUp) == 0
        rig.feed(pointing(), 2)
        assert rig.count(MouseUp) == 1


class TestDrag:
    def test_held_pinch_with_movement_becomes_a_drag(self):
        rig = Rig()
        rig.start_interaction()
        moving = rig.feed(pinched(0.46), 6)  # ~77 px to the right of the press position
        assert moving[-1].state is S.DRAGGING
        rig.feed(pinched(0.46), 2)
        rig.feed(pointing(0.46), 2)
        ups = [a for a in rig.actions if isinstance(a, MouseUp)]
        assert rig.count(MouseDown) == 1 and len(ups) == 1
        assert ups[0].drag is True

    def test_pointer_follows_the_fingertip_during_a_drag(self):
        rig = Rig()
        rig.start_interaction()
        xs = []
        for i in range(12):
            frame = rig.feed(pinched(0.5 - 0.01 * i))[0]
            if frame.cursor:
                xs.append(frame.cursor[0])
        assert len(xs) >= 5
        assert xs == sorted(xs) and xs[-1] > xs[0]  # hand moves left in camera = right on screen

    def test_jitter_below_the_threshold_never_starts_a_drag(self):
        rig = Rig()
        rig.start_interaction()
        wobble = [pinched(0.5 + (0.004 if i % 2 else 0.0)) for i in range(60)]  # ~8 px
        frames = rig.feed_seq(wobble)
        assert all(f.state is S.PRESSED for f in frames)
        assert rig.count(MoveTo, frames) == 0
        rig.feed(pointing(), 2)
        assert [a.drag for a in rig.actions if isinstance(a, MouseUp)] == [False]

    def test_a_sustained_small_offset_never_starts_a_drag(self):
        rig = Rig()
        rig.start_interaction()
        frames = rig.feed(pinched(0.5 - 0.004), 90)  # steady ~8 px away from the press point
        assert all(f.state is S.PRESSED for f in frames)
        assert rig.count(MoveTo, frames) == 0

    def test_the_drag_threshold_sits_where_it_is_documented(self):
        just_under = Rig()  # 0.0150 camera units = 28.8 px, below the 30 px default
        just_under.start_interaction()
        assert just_under.feed(pinched(0.5 - 0.0150), 20)[-1].state is S.PRESSED
        just_over = Rig()  # 0.0170 camera units = 32.6 px, above it
        just_over.start_interaction()
        assert just_over.feed(pinched(0.5 - 0.0170), 20)[-1].state is S.DRAGGING

    def test_a_single_spike_frame_does_not_start_a_drag(self):
        rig = Rig()
        rig.start_interaction()
        frames = rig.feed_seq([pinched(), pinched(0.40), pinched(), pinched(0.40), pinched()])
        assert all(f.state is S.PRESSED for f in frames)

    def test_two_spike_frames_in_a_row_still_do_not_start_a_drag(self):
        rig = Rig()
        rig.start_interaction()
        frames = rig.feed_seq([pinched(0.40)] * 2 + [pinched()])
        assert all(f.state is S.PRESSED for f in frames)

    def test_movement_before_the_press_does_not_count_toward_a_drag(self):
        rig = Rig()
        rig.feed(pointing(0.5), 3)
        rig.feed(pointing(0.40), 3)  # big move while open
        frames = rig.feed(pinched(0.40), 8)
        assert frames[-1].state is S.PRESSED
        assert rig.count(MoveTo, frames) == 0

    def test_drag_threshold_is_configurable(self):
        rig = Rig(drag_start_px=5.0)
        rig.start_interaction()
        frames = rig.feed(pinched(0.495), 6)  # ~10 px
        assert frames[-1].state is S.DRAGGING

    def test_state_sequence_through_a_whole_drag(self):
        rig = Rig()
        seen = [f.state for f in rig.feed(pointing(), 3)]
        seen += [f.state for f in rig.feed(pinched(), 5)]
        seen += [f.state for f in rig.feed(pinched(0.45), 5)]
        seen += [f.state for f in rig.feed(pointing(0.45), 3)]
        ordered = []
        for s in seen:
            if not ordered or ordered[-1] is not s:
                ordered.append(s)
        assert ordered == [S.POINTER, S.PINCH_PENDING, S.PRESSED, S.DRAGGING, S.POINTER]


class TestInterruptions:
    def test_tracking_loss_during_a_drag_releases_immediately(self):
        rig = Rig()
        rig.start_interaction()
        rig.feed(pinched(0.45), 6)
        lost = rig.feed(None)[0]
        assert [type(a) for a in lost.actions] == [MouseUp]
        assert lost.actions[0].drag is True
        assert lost.state is S.SEARCHING

    def test_tracking_loss_while_pressed_releases_too(self):
        rig = Rig()
        rig.start_interaction()
        lost = rig.feed(None)[0]
        assert [type(a) for a in lost.actions] == [MouseUp]
        assert lost.actions[0].drag is False

    def test_low_confidence_during_a_press_releases(self):
        rig = Rig()
        rig.start_interaction()
        shaky = rig.feed(make_hand(pinch_ratio=0.1, confidence=0.2))[0]
        assert shaky.tracking is TrackingState.UNCERTAIN
        assert [type(a) for a in shaky.actions] == [MouseUp]

    def test_interrupted_drag_is_never_resumed_when_tracking_returns(self):
        rig = Rig()
        rig.start_interaction()
        rig.feed(pinched(0.45), 6)
        rig.feed(None, 3)
        before = len(rig.frames)
        rig.feed(pinched(0.40), 60)  # hand returns, still pinched and moving
        after = rig.frames[before:]
        assert rig.count(MouseDown, after) == 0
        assert rig.count(MoveTo, after) == 0
        assert not rig.proc.holding_button

    def test_a_new_interaction_needs_an_open_hand_after_recovery(self):
        rig = Rig()
        rig.start_interaction()
        rig.feed(None)
        rig.feed(pinched(), 30)
        assert rig.count(MouseDown) == 1  # still only the original press
        rig.feed(pointing(), 3)
        rig.feed(pinched(), 5)
        assert rig.count(MouseDown) == 2

    def test_pause_releases_and_blocks_all_actions(self):
        rig = Rig()
        rig.start_interaction()
        rig.feed(pinched(0.45), 6)
        paused = rig.feed(pinched(0.40), 1, paused=True)[0]
        assert [type(a) for a in paused.actions] == [MouseUp]
        assert paused.state is S.PAUSED
        later = rig.feed(pointing(), 30, paused=True)
        assert all(f.actions == () and f.state is S.PAUSED for f in later)

    def test_resume_requires_stable_tracking_and_an_open_hand(self):
        rig = Rig(stable_frames=3)
        rig.feed(pointing(), 4)
        rig.feed(pinched(), 5)
        rig.feed(pinched(), 1, paused=True)
        resumed = rig.feed(pinched(), 6)  # resumes while still pinching
        assert [f.tracking for f in resumed[:2]] == [TrackingState.ACQUIRING] * 2
        assert rig.count(MouseDown, resumed) == 0
        rig.feed(pointing(), 3)
        again = rig.feed(pinched(), 5)
        assert rig.count(MouseDown, again) == 1

    def test_halt_releases_and_nothing_acts_afterwards(self):
        rig = Rig()
        rig.start_interaction()
        released = rig.proc.halt("test stop")
        assert [type(a) for a in released] == [MouseUp]
        assert rig.proc.state is S.STOPPED
        later = rig.feed(pointing(), 10) + rig.feed(pinched(), 10)
        assert all(f.actions == () for f in later)
        assert rig.proc.halt("again") == ()

    def test_halt_without_a_press_releases_nothing(self):
        rig = Rig()
        rig.feed(pointing(), 5)
        assert rig.proc.halt("stop") == ()

    def test_abort_interaction_means_no_release_is_ever_emitted(self):
        rig = Rig()
        rig.start_interaction()
        rig.proc.abort_interaction()  # the desktop refused the press
        assert not rig.proc.holding_button
        rig.feed(pinched(), 5)
        rig.feed(pointing(), 5)
        assert rig.count(MouseUp) == 0 and rig.count(MouseDown) == 1


class TestCooldown:
    def test_pinch_inside_the_cooldown_is_consumed_without_a_press(self):
        rig = Rig(click_cooldown_s=0.5)
        rig.start_interaction()
        rig.feed(pointing(), 2)  # release
        refused = rig.feed(pinched(), 5)  # about 0.3 s after the first press
        assert rig.count(MouseDown, refused) == 0
        assert refused[-1].state is S.PINCH_PENDING

    def test_pinch_after_the_cooldown_works(self):
        rig = Rig(click_cooldown_s=0.2)
        rig.start_interaction()
        rig.feed(pointing(), 10)
        rig.feed(pinched(), 5)
        assert rig.count(MouseDown) == 2


class TestMachineInvariants:
    """Direct machine tests that bypass the pinch detector."""

    def machine(self):
        return InteractionMachine(ScrollTracker(), drag_start_px=30.0)

    def test_a_scroll_pose_cannot_preempt_a_held_pinch(self):
        m = self.machine()
        held = PinchResult(PinchState.PINCHED, False, True, held=True)
        press = PinchResult(PinchState.PINCHED, True, True, held=True)
        base = {"now": 0.0, "tracking": TrackingState.TRACKING, "target": (100.0, 100.0)}
        m.step(MachineInput(**base, pinch=press))
        assert m.holding
        for _ in range(20):
            actions = m.step(
                MachineInput(**base, pinch=held, scroll_pose=True, scroll_y=0.5, scroll_scale=0.2)
            )
            assert actions == ()
        assert m.state is S.PRESSED

    @pytest.mark.parametrize("state", [PinchState.OPEN, PinchState.REOPEN, PinchState.CLOSING])
    def test_a_pinch_that_is_no_longer_held_releases_even_without_a_release_event(self, state):
        m = self.machine()
        base = {"now": 0.0, "tracking": TrackingState.TRACKING, "target": (100.0, 100.0)}
        m.step(MachineInput(**base, pinch=PinchResult(PinchState.PINCHED, True, True, held=True)))
        assert m.holding
        # e.g. the detector was reset underneath us: not held, but no 'released' flag either
        out = m.step(MachineInput(**base, pinch=PinchResult(state, False, True)))
        assert [type(a) for a in out[:1]] == [MouseUp]
        assert not m.holding

    def test_missing_inputs_default_to_the_safe_state(self):
        m = self.machine()
        out = m.step(MachineInput(now=0.0, tracking=TrackingState.TRACKING))  # no target/pinch
        assert out == () and m.state is S.SEARCHING

    @pytest.mark.parametrize("bad", [0, -5.0])
    def test_invalid_drag_threshold_rejected(self, bad):
        with pytest.raises(ValueError):
            InteractionMachine(ScrollTracker(), drag_start_px=bad)
