"""Opt-in right click: a thumb-middle pinch made with the index, ring and little fingers extended.

Every hand here is SYNTHETIC geometry. These tests check the software's decisions, ordering and
safety invariants. They are not evidence that real hands, cameras or lighting behave this way:
the rule's thresholds come from hand proportions, and it has not been validated on a webcam.
"""

from __future__ import annotations

import itertools
import sys
import types

import numpy as np
import pytest

from helpers import make_hand, make_pose_hand
from nocursor.cli import build_parser, main
from nocursor.config import Config
from nocursor.controller import (
    FailSafeTriggered,
    InputBlocked,
    PyAutoGUIBackend,
    SafeController,
    ScreenSizeOnlyBackend,
)
from nocursor.diagnostics import DiagnosticsReporter
from nocursor.dispatch import apply_actions
from nocursor.emergency import EmergencyStop
from nocursor.gestures import (
    RIGHT_CLICK_INDEX_MARGIN,
    PinchResult,
    PinchState,
    pinch_ratio,
    pinch_ratio_2d,
)
from nocursor.hud import StatusInfo, build_hud, help_lines, toast_text
from nocursor.interaction import (
    InteractionMachine,
    InteractionState,
    MachineInput,
    MouseDown,
    MouseUp,
    MoveTo,
    RightClick,
    Scroll,
)
from nocursor.models import INDEX_TIP, MIDDLE_TIP, THUMB_TIP, WRIST, HandObservation
from nocursor.processor import GestureProcessor
from nocursor.scroll import FingerPose, FingerState, ScrollTracker
from nocursor.settings import describe_settings
from nocursor.tracking import TrackingState

DT = 1 / 30
S = InteractionState


# ------------------------------------------------------------------------------ hands
def right_hand(ratio: float = 0.10, **kwargs) -> HandObservation:
    """The right-click pose: index, ring, little extended; the thumb ``ratio`` hand sizes from
    the middle fingertip (0.10 = touching, 0.8 = well apart)."""
    return make_pose_hand(index="ext", middle="amb", ring="ext", pinky="ext", pinch_ratio=ratio,
                          thumb_to="middle", **kwargs)  # fmt: skip


OPEN = right_hand(0.8)  # the same pose with the thumb away: the "open" hand that arms a pinch
CLOSED = right_hand(0.10)
LEFT_PINCH_OPEN_HAND = make_pose_hand(index="ext", middle="ext", ring="ext", pinky="ext",
                                      pinch_ratio=0.10)  # thumb on the index tip  # fmt: skip


def thumb_at(hand: HandObservation, xy: tuple[float, float]) -> HandObservation:
    pts = hand.landmarks.copy()
    pts[THUMB_TIP, :2] = xy
    return HandObservation(pts, hand.confidence, hand.aspect_ratio)


def thumb_between_tips(fraction: float) -> HandObservation:
    """Thumb on the line from the index fingertip (0) to the middle fingertip (1) and beyond."""
    base = right_hand(0.8)
    a, b = base.landmarks[INDEX_TIP, :2], base.landmarks[MIDDLE_TIP, :2]
    return thumb_at(base, tuple(a + fraction * (b - a)))


def rotated(hand: HandObservation, degrees: float) -> HandObservation:
    """The whole hand rotated about the wrist (aspect 1, so distances are preserved)."""
    pts = hand.landmarks.copy()
    wrist = pts[WRIST, :2].copy()
    angle = np.radians(degrees)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    pts[:, :2] = (pts[:, :2] - wrist) @ matrix.T + wrist
    return HandObservation(pts, hand.confidence, hand.aspect_ratio)


class Rig:
    def __init__(self, right_click: bool = True, **cfg):
        base = {"smoothing": 0.0, "stable_frames": 1, "active_margin": 0.0,
                "click_cooldown_s": 0.0, "right_click": right_click}  # fmt: skip
        self.proc = GestureProcessor(Config(**{**base, **cfg}), (1920, 1080))
        self.t = 0.0
        self.frames = []

    def feed(self, hand, n=1, paused=False):
        out = []
        for _ in range(n):
            self.t += DT
            frame = self.proc.process(hand, self.t, paused=paused)
            self.frames.append(frame)
            out.append(frame)
        return out

    def count(self, kind, frames=None):
        pool = self.frames if frames is None else frames
        return sum(isinstance(a, kind) for f in pool for a in f.actions)

    def armed(self):
        """Open hand in the right-click pose for a few frames, so a pinch can count."""
        return self.feed(OPEN, 3)


# ------------------------------------------------------------------------------- pose
class TestRightClickPose:
    @staticmethod
    def pose(index, middle, ring, pinky):
        return FingerPose(index, middle, ring, pinky)

    E, F, A = FingerState.EXTENDED, FingerState.FOLDED, FingerState.AMBIGUOUS

    def test_index_ring_and_little_extended_is_the_pose_whatever_the_middle_finger_does(self):
        for middle in (self.E, self.F, self.A):
            assert self.pose(self.E, middle, self.E, self.E).is_right_click_pose

    @pytest.mark.parametrize("which", [0, 2, 3])
    def test_any_of_the_three_fingers_not_clearly_extended_breaks_it(self, which):
        for bad in (self.F, self.A):
            states = [self.E, self.A, self.E, self.E]
            states[which] = bad
            assert not self.pose(*states).is_right_click_pose

    def test_it_is_never_the_scroll_pose_or_the_pointing_pose(self):
        states = (self.E, self.F, self.A)
        for combo in itertools.product(states, repeat=4):
            pose = self.pose(*combo)
            assert not (pose.is_right_click_pose and pose.is_scroll_pose)
        pointing = self.pose(self.E, self.F, self.F, self.F)
        assert not pointing.is_right_click_pose  # the shape that rests the thumb on the middle

    def test_the_hand_builder_makes_the_pose_and_the_expected_ratios(self):
        from nocursor.scroll import classify_hand

        assert classify_hand(CLOSED).is_right_click_pose
        assert pinch_ratio(CLOSED, False, MIDDLE_TIP) == pytest.approx(0.10, abs=1e-6)
        assert pinch_ratio(CLOSED) - pinch_ratio(CLOSED, False, MIDDLE_TIP) > 0.3
        assert pinch_ratio(CLOSED) > RIGHT_CLICK_INDEX_MARGIN

    def test_the_default_pinch_ratio_is_still_the_thumb_index_one(self):
        hand = make_hand(pinch_ratio=0.25)
        assert pinch_ratio(hand) == pinch_ratio_2d(hand) == pytest.approx(0.25, abs=1e-6)
        assert pinch_ratio(hand, False, INDEX_TIP) == pinch_ratio(hand)


# ------------------------------------------------------------------- off by default
class TestOffByDefault:
    def test_the_option_is_off_unless_asked_for(self):
        assert Config().right_click is False
        assert GestureProcessor(Config(), (1920, 1080)).right_click_enabled is False

    def test_the_gesture_does_nothing_when_disabled(self):
        rig = Rig(right_click=False)
        rig.armed()
        frames = rig.feed(CLOSED, 60)
        assert rig.count(RightClick) == 0 and rig.count(MouseDown) == 0
        assert all(f.right_pinch is None and f.right_progress is None for f in rig.frames)
        assert rig.count(MoveTo, frames) > 0  # it is just pointing, as before

    def test_enabling_it_changes_nothing_while_the_right_click_pose_is_not_made(self):
        """The same scripted session, with and without the option, acts identically."""
        script = [
            (make_hand(), 3), (make_hand(pinch_ratio=0.10), 8), (make_hand(), 4),  # a click
            (make_hand(index_xy=(0.5, 0.5), pinch_ratio=0.10), 6),                 # press
            (make_hand(index_xy=(0.7, 0.5), pinch_ratio=0.10), 8),                 # drag
            (make_hand(index_xy=(0.7, 0.5)), 4), (None, 5),                         # drop, lose it
            (make_hand(), 3), (make_pose_hand(), 9),                                # scroll pose
            (make_pose_hand(shift=(0.0, -0.1)), 4), (make_pose_hand(middle="fold"), 6),
            (LEFT_PINCH_OPEN_HAND, 8), (make_pose_hand(ring="ext", pinky="ext"), 6),
        ]  # fmt: skip
        runs = []
        for enabled in (False, True):
            rig = Rig(right_click=enabled)
            for hand, n in script:
                rig.feed(hand, n)
            runs.append([(f.actions, f.state, f.pinch, f.scroll_pose) for f in rig.frames])
        assert runs[0] == runs[1]
        assert any(isinstance(a, MouseDown) for actions, *_ in runs[0] for a in actions)

    def test_the_command_line_flag_is_the_only_way_to_turn_it_on(self):
        parser = build_parser()
        assert parser.parse_args([]).right_click is False
        assert parser.parse_args(["--right-click"]).right_click is True
        text = parser.format_help()
        assert "--right-click" in text and "off by default" in text and "EXPERIMENTAL" in text

    def test_print_settings_shows_the_option_and_marks_it_when_changed(self, capsys):
        assert main(["--print-settings"]) == 0
        assert "right_click" in capsys.readouterr().out
        text = describe_settings(Config(right_click=True))
        line = next(row for row in text.splitlines() if "right_click" in row)
        assert line.strip().startswith("*") and "on" in line


# ----------------------------------------------------------------- the right click
class TestRightClickPinch:
    def test_a_confirmed_thumb_middle_pinch_gives_exactly_one_right_click(self):
        rig = Rig()
        rig.armed()
        frames = rig.feed(CLOSED, 5)
        assert [rig.count(RightClick, [f]) for f in frames] == [0, 0, 0, 0, 1]
        assert rig.count(MouseDown) == 0 and rig.count(MouseUp) == 0

    def test_an_unconfirmed_pinch_triggers_neither_button(self):
        rig = Rig()
        rig.armed()
        rig.feed(CLOSED, 4)  # one frame short of the five that confirm
        rig.feed(OPEN, 5)
        assert rig.count(RightClick) == 0 and rig.count(MouseDown) == 0

    def test_holding_the_pinch_never_clicks_again(self):
        rig = Rig()
        rig.armed()
        rig.feed(CLOSED, 300)
        assert rig.count(RightClick) == 1 and rig.count(MouseDown) == 0

    def test_releasing_and_pinching_again_clicks_again(self):
        rig = Rig()
        rig.armed()
        rig.feed(CLOSED, 8)
        rig.feed(OPEN, 3)
        rig.feed(CLOSED, 8)
        assert rig.count(RightClick) == 2

    def test_a_pinch_needs_the_hand_to_be_seen_open_first(self):
        rig = Rig()
        rig.feed(CLOSED, 30)  # never seen open
        assert rig.count(RightClick) == 0
        rig.feed(OPEN, 3)
        rig.feed(CLOSED, 6)
        assert rig.count(RightClick) == 1

    def test_the_click_cooldown_stops_a_second_click_too_soon(self):
        rig = Rig(click_cooldown_s=0.4)
        rig.armed()
        rig.feed(CLOSED, 6)
        rig.feed(OPEN, 2)
        rig.feed(CLOSED, 6)  # confirmed about 0.3 s after the first: refused
        assert rig.count(RightClick) == 1
        rig.feed(OPEN, 3)
        rig.feed(CLOSED, 6)  # now past the cooldown
        assert rig.count(RightClick) == 2

    def test_the_pointer_is_frozen_while_pinching_and_resumes_afterwards(self):
        rig = Rig()
        rig.armed()
        pinching = rig.feed(CLOSED, 12)
        assert rig.count(MoveTo, pinching) == 0
        assert all(f.state is S.PINCH_PENDING for f in pinching)
        after = rig.feed(OPEN, 3)
        assert rig.count(MoveTo, after) > 0

    def test_it_works_at_any_hand_rotation(self):
        for degrees in (-40, -20, 0, 20, 40):
            rig = Rig()
            rig.feed(rotated(OPEN, degrees), 3)
            rig.feed(rotated(CLOSED, degrees), 10)
            assert (rig.count(RightClick), rig.count(MouseDown)) == (1, 0), degrees

    def test_landmark_noise_around_the_threshold_does_not_make_repeated_clicks(self):
        rig = Rig()
        rig.armed()
        for i in range(120):  # flicker across the 0.20 threshold every frame
            rig.feed(right_hand(0.18 if i % 2 == 0 else 0.23))
        assert rig.count(RightClick) == 0
        rig.feed(OPEN, 3)
        for i in range(120):  # small jitter around a firm pinch
            rig.feed(right_hand(0.10 + 0.03 * ((-1) ** i)))
        assert rig.count(RightClick) == 1 and rig.count(MouseDown) == 0

    def test_a_confirmed_click_is_reported_in_the_frame_it_happens(self):
        rig = Rig()
        rig.armed()
        frames = rig.feed(CLOSED, 5)
        assert frames[-1].right_pinch is PinchState.PINCHED
        assert frames[-2].right_progress == (4, 5)


# ------------------------------------------------------------- left click preserved
class TestLeftClickPreserved:
    @pytest.mark.parametrize("hand", [make_hand(pinch_ratio=0.10), LEFT_PINCH_OPEN_HAND],
                             ids=["plain", "open hand with thumb on the index"])
    def test_a_thumb_index_pinch_is_exactly_one_left_click_and_no_right_click(self, hand):
        rig = Rig()
        rig.feed(make_hand(), 3)
        frames = rig.feed(hand, 5)
        assert rig.count(MouseDown, frames) == 1 and rig.count(RightClick) == 0
        rig.feed(hand, 100)
        rig.feed(make_hand(), 3)
        assert rig.count(MouseDown) == 1 and rig.count(MouseUp) == 1 and rig.count(RightClick) == 0

    def test_the_left_click_still_needs_the_normal_confirmation(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        rig.feed(make_hand(pinch_ratio=0.10), 4)
        rig.feed(make_hand(), 3)
        assert rig.count(MouseDown) == 0


# ------------------------------------------------------------------ never both
class TestNeverBothButtons:
    def sweep(self, hand_for, fractions, frames=14):
        results = {}
        for fraction in fractions:
            rig = Rig()
            rig.feed(OPEN, 3)
            rig.feed(hand_for(fraction), frames)
            rig.feed(OPEN, 3)
            results[fraction] = (rig.count(MouseDown), rig.count(RightClick))
        return results

    def test_a_thumb_anywhere_between_the_index_and_middle_tips_never_presses_both(self):
        fractions = [i / 20 for i in range(-4, 25)]  # a little beyond both tips as well
        for fraction, (left, right) in self.sweep(thumb_between_tips, fractions).items():
            assert left + right <= 1, (fraction, left, right)

    def test_the_overlap_is_decided_by_which_fingertip_is_nearer(self):
        results = self.sweep(thumb_between_tips, [0.0, 0.5, 1.0])
        assert results[0.0] == (1, 0)  # on the index tip: a left click, as always
        assert results[1.0] == (0, 1)  # on the middle tip: a right click
        left, right = results[0.5]  # exactly between: ambiguous, so the right click is refused
        assert right == 0 and left <= 1

    @staticmethod
    def close_tips_hand(gap: float, thumb_offset: float) -> HandObservation:
        """Index and middle fingertips ``gap`` hand sizes apart (fingers held together), the thumb
        ``thumb_offset`` to the right of the middle fingertip, so r_middle = ``thumb_offset``
        and r_index = ``gap + thumb_offset``."""
        # the middle finger fully out, so lowering the index tip to its height keeps the index
        # finger clearly extended (the pose does not judge the middle finger)
        base = make_pose_hand(index="ext", middle="ext", ring="ext", pinky="ext",
                              pinch_ratio=0.8, thumb_to="middle")  # fmt: skip
        pts = base.landmarks.copy()
        size = 0.2  # the helper's hand size, in frame heights
        pts[INDEX_TIP, :2] = pts[MIDDLE_TIP, :2] - np.array([gap * size, 0.0])
        pts[THUMB_TIP, :2] = pts[MIDDLE_TIP, :2] + np.array([thumb_offset * size, 0.0])
        return HandObservation(pts, base.confidence, base.aspect_ratio)

    def outcome(self, hand):
        rig = Rig()
        rig.feed(OPEN, 3)
        rig.feed(hand, 20)
        return rig.count(MouseDown), rig.count(RightClick)

    def test_the_hand_builder_reaches_the_overlap_zone(self):
        hand = self.close_tips_hand(gap=0.08, thumb_offset=0.05)
        assert pinch_ratio(hand, False, MIDDLE_TIP) == pytest.approx(0.05, abs=1e-6)
        assert pinch_ratio(hand) == pytest.approx(0.13, abs=1e-6)  # both pinches would engage
        from nocursor.scroll import classify_hand

        assert classify_hand(hand).is_right_click_pose  # and the index finger is still out

    @pytest.mark.parametrize("gap", [0.0, 0.03, 0.06, 0.09])
    def test_when_the_index_is_not_clearly_farther_there_is_no_right_click(self, gap):
        """Fingertips closer together than the margin: the thumb could be on either, so the right
        click is refused. The left click keeps its existing behaviour."""
        left, right = self.outcome(self.close_tips_hand(gap, 0.05))
        assert right == 0 and left <= 1

    @pytest.mark.parametrize("gap", [0.12, 0.15, 0.25])
    def test_when_the_index_is_clearly_farther_it_is_a_right_click_and_not_a_left_one(self, gap):
        assert self.outcome(self.close_tips_hand(gap, 0.05)) == (0, 1)

    def test_the_margin_is_what_separates_them(self):
        below = RIGHT_CLICK_INDEX_MARGIN - 0.02
        above = RIGHT_CLICK_INDEX_MARGIN + 0.02
        assert self.outcome(self.close_tips_hand(below, 0.05))[1] == 0
        assert self.outcome(self.close_tips_hand(above, 0.05)) == (0, 1)

    def test_one_frame_never_carries_both_a_press_and_a_right_click(self):
        rig = Rig()
        rig.feed(OPEN, 3)
        for fraction in [i / 10 for i in range(0, 11)] * 3:
            rig.feed(thumb_between_tips(fraction), 3)
            rig.feed(OPEN, 2)
        for frame in rig.frames:
            kinds = {type(a) for a in frame.actions}
            assert not ({MouseDown, RightClick} <= kinds)

    def test_a_left_pinch_that_slides_to_the_middle_finger_does_not_also_right_click(self):
        rig = Rig(click_cooldown_s=0.4)
        rig.feed(OPEN, 3)
        rig.feed(thumb_between_tips(0.0), 6)  # a left click
        rig.feed(thumb_between_tips(1.0), 6)  # the thumb slides across within the cooldown
        assert rig.count(MouseDown) == 1 and rig.count(RightClick) == 0

    def test_a_right_pinch_right_after_a_left_click_waits_for_the_cooldown(self):
        rig = Rig(click_cooldown_s=0.4)
        rig.feed(OPEN, 3)
        rig.feed(LEFT_PINCH_OPEN_HAND, 5)  # a left click
        rig.feed(OPEN, 4)  # released, and the right pinch is armed again
        soon = rig.feed(CLOSED, 6)  # confirmed well inside 0.4 s of the left click
        assert rig.count(MouseDown) == 1 and rig.count(RightClick, soon) == 0
        rig.feed(OPEN, 14)  # more than 0.4 s later
        rig.feed(CLOSED, 6)
        assert rig.count(RightClick) == 1

    def test_a_right_click_is_followed_by_no_left_press_within_the_cooldown(self):
        rig = Rig(click_cooldown_s=0.4)
        rig.armed()
        rig.feed(CLOSED, 6)  # a right click
        rig.feed(thumb_between_tips(0.0), 6)  # the thumb slides to the index tip
        assert rig.count(RightClick) == 1 and rig.count(MouseDown) == 0


# ------------------------------------------------------------ the wrong poses
class TestWrongPoses:
    """The resting shapes that put the thumb on the middle finger must never right-click."""

    def run(self, closed, opened=None):
        rig = Rig()
        rig.feed(opened or OPEN, 3)
        rig.feed(closed, 40)
        rig.feed(opened or OPEN, 3)
        rig.feed(closed, 40)
        return rig

    def test_pointing_with_the_thumb_resting_on_the_curled_middle_finger(self):
        closed = make_pose_hand(index="ext", middle="fold", ring="fold", pinky="fold",
                                pinch_ratio=0.10, thumb_to="middle")  # fmt: skip
        opened = make_pose_hand(index="ext", middle="fold", ring="fold", pinky="fold",
                                pinch_ratio=0.8, thumb_to="middle")  # fmt: skip
        rig = self.run(closed, opened)
        assert rig.count(RightClick) == 0 and rig.count(MouseDown) == 0

    @pytest.mark.parametrize("fingers", [
        {"index": "ext", "middle": "ext", "ring": "fold", "pinky": "fold"},  # the scroll pose
        {"index": "fold", "middle": "fold", "ring": "fold", "pinky": "fold"},  # a fist
        {"index": "ext", "middle": "amb", "ring": "ext", "pinky": "fold"},
        {"index": "ext", "middle": "amb", "ring": "amb", "pinky": "ext"},
        {"index": "amb", "middle": "amb", "ring": "ext", "pinky": "ext"},
    ])
    def test_other_hand_shapes_with_the_thumb_on_the_middle_fingertip(self, fingers):
        closed = make_pose_hand(**fingers, pinch_ratio=0.10, thumb_to="middle")
        opened = make_pose_hand(**fingers, pinch_ratio=0.8, thumb_to="middle")
        rig = self.run(closed, opened)
        assert rig.count(RightClick) == 0

    def test_the_two_finger_scroll_pose_still_scrolls_and_never_right_clicks(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        frames = rig.feed(make_pose_hand(), 6)
        assert frames[-1].state is S.SCROLL
        moving = [rig.feed(make_pose_hand(shift=(0.0, -0.01 * i)))[0] for i in range(1, 20)]
        assert sum(a.clicks for f in moving for a in f.actions if isinstance(a, Scroll)) > 0
        assert rig.count(RightClick) == 0

    def test_the_right_click_pose_never_starts_scrolling(self):
        rig = Rig()
        frames = rig.feed(CLOSED, 40) + rig.feed(OPEN, 40)
        assert all(f.state not in (S.SCROLL, S.SCROLL_PENDING) and not f.scroll_pose
                   for f in frames)  # fmt: skip


# ------------------------------------------------------- tracking loss, pause, hold
class TestSafetyInvariants:
    def test_tracking_loss_during_a_pinch_gives_no_click_and_no_held_button(self):
        rig = Rig()
        rig.armed()
        rig.feed(CLOSED, 3)  # candidate only
        lost = rig.feed(None, 30)
        assert all(f.actions == () and f.state is S.SEARCHING for f in lost)
        back = rig.feed(CLOSED, 10)  # tracking returns with the pinch already closed
        assert rig.count(RightClick) == 0 and rig.count(MouseDown) == 0
        assert not rig.proc.holding_button
        assert back[-1].state is not S.SEARCHING

    def test_after_tracking_loss_the_hand_must_be_seen_open_again(self):
        rig = Rig()
        rig.armed()
        rig.feed(None, 5)
        rig.feed(CLOSED, 20)
        assert rig.count(RightClick) == 0
        rig.feed(OPEN, 3)
        rig.feed(CLOSED, 6)
        assert rig.count(RightClick) == 1

    def test_tracking_loss_after_a_click_never_repeats_it(self):
        rig = Rig()
        rig.armed()
        rig.feed(CLOSED, 8)
        rig.feed(None, 10)
        rig.feed(CLOSED, 20)
        assert rig.count(RightClick) == 1

    def test_pausing_cancels_a_pending_pinch_and_needs_re_arming(self):
        rig = Rig()
        rig.armed()
        rig.feed(CLOSED, 3)
        paused = rig.feed(CLOSED, 20, paused=True)
        assert all(f.actions == () for f in paused)
        rig.feed(CLOSED, 20)  # resumed with the pinch already closed
        assert rig.count(RightClick) == 0
        rig.feed(OPEN, 3)
        rig.feed(CLOSED, 6)
        assert rig.count(RightClick) == 1

    def test_the_right_button_is_never_held_so_nothing_can_leave_it_pressed(self):
        rig = Rig()
        rig.armed()
        rig.feed(CLOSED, 8)
        assert not rig.proc.holding_button
        assert rig.proc.halt("test") == ()  # a stop has nothing to release

    def test_a_held_left_press_blocks_the_right_click_and_is_unchanged(self):
        rig = Rig()
        rig.feed(OPEN, 3)  # armed for the right click as well
        rig.feed(make_hand(index_xy=(0.5, 0.5), pinch_ratio=0.10), 6)  # a left press
        assert rig.proc.holding_button
        # the hand turns into the right-click pose; the press ends and the right pinch keeps
        # closing, but the right click may not follow without the hand being seen open again
        after = rig.feed(CLOSED, 25)
        assert rig.count(MouseDown) == 1 and rig.count(MouseUp) == 1
        assert rig.count(RightClick, after) == 0
        rig.feed(OPEN, 3)
        rig.feed(CLOSED, 6)
        assert rig.count(RightClick) == 1  # an ordinary, deliberate right click works afterwards

    def test_a_drag_is_unchanged_with_the_option_on(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        rig.feed(make_hand(index_xy=(0.5, 0.5), pinch_ratio=0.10), 6)
        for i in range(1, 12):
            rig.feed(make_hand(index_xy=(0.5 + 0.02 * i, 0.5), pinch_ratio=0.10))
        assert rig.proc.state is S.DRAGGING
        rig.feed(make_hand(index_xy=(0.72, 0.5)), 3)
        ups = [a for f in rig.frames for a in f.actions if isinstance(a, MouseUp)]
        assert [u.drag for u in ups] == [True] and rig.count(RightClick) == 0

    def test_a_right_click_pose_during_a_drag_never_clicks(self):
        rig = Rig(click_cooldown_s=0.4)
        rig.feed(make_hand(), 3)
        rig.feed(make_hand(index_xy=(0.5, 0.5), pinch_ratio=0.10), 6)
        for i in range(1, 12):
            rig.feed(make_hand(index_xy=(0.5 + 0.02 * i, 0.5), pinch_ratio=0.10))
        for hand in (OPEN, CLOSED, CLOSED, CLOSED, CLOSED, CLOSED, CLOSED):
            frames = rig.feed(hand)
            if rig.proc.holding_button:
                assert rig.count(RightClick, frames) == 0
        assert rig.count(MouseDown) == 1 and rig.count(MouseUp) == 1

    def test_calibration_style_idle_frames_never_act(self):
        rig = Rig()
        rig.armed()
        for _ in range(20):
            rig.t += DT
            frame = rig.proc.idle(CLOSED, rig.t)
            assert frame.actions == ()


# --------------------------------------------------------------- the state machine
class TestMachine:
    def machine(self):
        return InteractionMachine(ScrollTracker())

    @staticmethod
    def right(clicked=False, freeze=False, state=PinchState.OPEN):
        return PinchResult(state, clicked=clicked, freeze_cursor=freeze)

    @staticmethod
    def left(clicked=False, state=PinchState.OPEN):
        return PinchResult(state, clicked=clicked, freeze_cursor=False)

    def step(self, machine, **kwargs):
        return machine.step(MachineInput(0.0, TrackingState.TRACKING, target=(10.0, 20.0),
                                         pinch=kwargs.pop("pinch", self.left()), **kwargs))

    def test_a_confirmed_right_pinch_is_one_complete_click_and_no_held_button(self):
        machine = self.machine()
        actions = self.step(machine, right=self.right(clicked=True, freeze=True))
        assert actions == (RightClick(),)
        assert not machine.holding and machine.state is S.PINCH_PENDING

    def test_a_pending_right_pinch_freezes_the_pointer_without_acting(self):
        machine = self.machine()
        assert self.step(machine, right=self.right(freeze=True, state=PinchState.CLOSING)) == ()
        assert machine.state is S.PINCH_PENDING

    def test_without_a_right_pinch_the_pointer_follows(self):
        machine = self.machine()
        assert self.step(machine) == (MoveTo(10, 20),)

    def test_the_left_press_wins_if_both_ever_arrive_together(self):
        machine = self.machine()
        actions = self.step(machine, pinch=self.left(clicked=True, state=PinchState.PINCHED),
                            right=self.right(clicked=True))
        assert actions == (MouseDown(),) and machine.holding

    def test_a_right_pinch_is_ignored_while_the_button_is_held(self):
        machine = self.machine()
        self.step(machine, pinch=self.left(clicked=True, state=PinchState.PINCHED))
        held = PinchResult(PinchState.PINCHED, False, True, held=True)
        actions = self.step(machine, pinch=held, right=self.right(clicked=True))
        assert RightClick() not in actions and machine.holding

    def test_unreliable_tracking_or_pause_ignores_the_right_pinch(self):
        machine = self.machine()
        lost = machine.step(MachineInput(0.0, TrackingState.LOST, right=self.right(clicked=True)))
        assert lost == () and machine.state is S.SEARCHING
        paused = machine.step(MachineInput(0.0, TrackingState.TRACKING, paused=True,
                                           right=self.right(clicked=True)))  # fmt: skip
        assert paused == () and machine.state is S.PAUSED

    def test_a_right_pinch_does_not_interrupt_scrolling(self):
        machine = self.machine()
        kwargs = {"scroll_pose": True, "scroll_y": 0.5, "scroll_scale": 0.2}
        for _ in range(8):
            actions = self.step(machine, right=self.right(clicked=True), **kwargs)
        assert machine.state is S.SCROLL and RightClick() not in actions


# -------------------------------------------------------------------- the controller
class Backend:
    def __init__(self):
        self.events: list[str] = []
        self.fail: Exception | None = None

    def screen_size(self):
        return (1280, 720)

    def move_to(self, x, y):
        self.events.append("move")

    def click(self):
        self.events.append("click")

    def mouse_down(self):
        self.events.append("down")

    def mouse_up(self):
        self.events.append("up")

    def scroll(self, clicks):
        self.events.append("scroll")

    def right_click(self):
        if self.fail:
            raise self.fail
        self.events.append("right_click")


def controller(enabled=True):
    backend, stop = Backend(), EmergencyStop()
    return SafeController(backend, stop, enabled=enabled), backend, stop


class TestController:
    def test_a_right_click_reaches_the_backend_once_and_leaves_the_button_state_alone(self):
        c, b, _ = controller()
        assert c.right_click() and not c.button_down
        assert b.events == ["right_click"] and c.right_clicks_sent == 1

    def test_it_is_refused_while_a_left_press_is_outstanding(self):
        c, b, _ = controller()
        c.mouse_down()
        assert not c.right_click()
        assert b.events == ["down"] and c.refused_button_events == 1 and c.button_down
        assert c.mouse_up() and b.events == ["down", "up"]  # the release is untouched

    def test_a_simulated_press_in_dry_run_also_blocks_it(self):
        c, b, _ = controller(enabled=False)
        c.mouse_down()
        assert not c.right_click()
        assert b.events == [] and c.refused_button_events == 1

    def test_dry_run_records_the_intent_and_sends_nothing(self):
        c, b, _ = controller(enabled=False)
        before = c.events_accepted
        assert not c.right_click()
        assert b.events == [] and c.events_accepted == before + 1
        assert c.records[-1].kind == "right_click" and not c.records[-1].sent

    def test_after_the_emergency_stop_nothing_is_sent(self):
        c, b, stop = controller()
        stop.trigger("test")
        assert not c.right_click() and b.events == []
        assert c.actions_suppressed == 1

    def test_while_inhibited_by_pause_or_calibration_nothing_is_sent(self):
        c, b, _ = controller()
        c.inhibited = True
        assert not c.right_click() and b.events == []

    def test_a_backend_failure_trips_the_emergency_stop(self):
        c, b, stop = controller()
        b.fail = RuntimeError("boom")
        assert not c.right_click()
        assert stop.is_set() and c.right_clicks_sent == 0

    def test_the_screen_corner_failsafe_trips_the_emergency_stop(self):
        c, b, stop = controller()
        b.fail = FailSafeTriggered("corner")
        assert not c.right_click() and stop.is_set()

    def test_a_backend_that_may_only_report_the_screen_size_cannot_right_click(self):
        backend = ScreenSizeOnlyBackend(screen=(1920, 1080))
        with pytest.raises(InputBlocked):
            backend.right_click()
        stop = EmergencyStop()
        c = SafeController(backend, stop)
        assert not c.right_click() and stop.is_set()


class FakePyAutoGUI(types.ModuleType):
    class FailSafeException(Exception):
        pass

    def __init__(self):
        super().__init__("pyautogui")
        self.FAILSAFE = False
        self.PAUSE = 1.0
        self.corner = False
        self.calls: list[tuple] = []

    def rightClick(self):
        if self.FAILSAFE and self.corner:
            raise self.FailSafeException("corner")
        self.calls.append(("rightClick",))


class TestPyAutoGUIBackend:
    @pytest.fixture
    def pg(self, monkeypatch):
        fake = FakePyAutoGUI()
        monkeypatch.setitem(sys.modules, "pyautogui", fake)
        return fake

    def test_it_uses_one_complete_right_click(self, pg):
        PyAutoGUIBackend().right_click()
        assert pg.calls == [("rightClick",)]

    def test_the_corner_failsafe_becomes_a_fail_safe_trigger(self, pg):
        pg.corner = True
        with pytest.raises(FailSafeTriggered):
            PyAutoGUIBackend().right_click()


# ---------------------------------------------------------------------- dispatch/HUD
class TestDispatch:
    def run(self, actions, enabled=True, hold=False, stop_first=False):
        c, b, stop = controller(enabled=enabled)
        proc = GestureProcessor(Config(right_click=True), (1920, 1080))
        if hold:
            c.mouse_down()
        if stop_first:
            stop.trigger("test")
        return apply_actions(c, proc, actions), c, b

    def test_an_accepted_right_click_is_reported_as_an_event(self):
        events, _, b = self.run([RightClick()])
        assert events == ("right_click",) and b.events == ["right_click"]

    def test_in_dry_run_it_is_reported_as_simulated_not_sent(self):
        events, _, b = self.run([RightClick()], enabled=False)
        assert events == ("right_click",) and b.events == []

    def test_a_refused_right_click_reports_nothing(self):
        for kwargs in ({"hold": True}, {"stop_first": True}):
            events, _, _ = self.run([RightClick()], **kwargs)
            assert events == ()

    def test_a_refused_right_click_does_not_disturb_a_held_left_press(self):
        events, c, b = self.run([RightClick()], hold=True)
        assert c.button_down and b.events == ["down"] and events == ()


def hud_for(rig, **status):
    frame = rig.frames[-1]
    return build_hud(frame, StatusInfo("LIVE", 30.0, "ctrl+alt+q", pause_hotkey="ctrl+alt+p",
                                       **status))  # fmt: skip


class TestHud:
    def test_a_pending_right_click_says_nothing_has_happened_yet(self):
        rig = Rig()
        rig.armed()
        rig.feed(CLOSED, 3)
        text = hud_for(rig).headline.text
        assert text == "RIGHT-CLICK PENDING 3/5 - not clicked yet"
        assert "PRESSED" not in text and "DRAGGING" not in text

    def test_a_held_right_pinch_does_not_claim_a_click(self):
        rig = Rig()
        rig.armed()
        rig.feed(CLOSED, 8)
        text = hud_for(rig).headline.text
        assert text == "RIGHT PINCH HELD - release, then pinch again"

    def test_the_toast_names_the_event_and_marks_a_simulation(self):
        assert toast_text("right_click", simulated=False) == "Right-clicked"
        assert toast_text("right_click", simulated=True) == "Right-clicked (simulated)"

    def test_the_help_panel_mentions_it_only_when_enabled(self):
        assert not any("right-click" in line for line in help_lines("a", "b"))
        assert any("right-click" in line for line in help_lines("a", "b", right_click=True))

    def test_left_click_wording_is_unchanged(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        rig.feed(make_hand(pinch_ratio=0.10), 3)
        assert hud_for(rig).headline.text == "PINCH PENDING 3/5 - not clicked yet"


class TestDiagnostics:
    def lines(self, enabled):
        rig = Rig(right_click=enabled, diagnostics=True)
        rig.armed()
        rig.feed(CLOSED, 3)
        return DiagnosticsReporter().overlay_lines(rig.proc.diagnostics)

    def test_the_overlay_shows_the_right_click_line_only_when_enabled(self):
        on, off = self.lines(True), self.lines(False)
        assert any(line.startswith("right-click: thumb-middle") for line in on)
        assert not any("right-click" in line for line in off)
        assert len(on) == len(off) + 1

    def test_the_line_reports_eligibility(self):
        line = next(x for x in self.lines(True) if x.startswith("right-click"))
        assert "eligible: yes" in line


class TestConfig:
    def test_it_can_be_enabled_through_the_config_and_the_flag(self):
        assert Config(right_click=True).right_click is True
        assert Config.from_overrides(right_click=True).right_click is True

    def test_it_reuses_the_existing_pinch_thresholds(self):
        proc = GestureProcessor(Config(right_click=True, pinch_threshold=0.25), (1920, 1080))
        assert proc.pinch_threshold == 0.25 and proc.right_click_enabled
