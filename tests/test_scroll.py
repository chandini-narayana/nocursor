"""Two-finger scroll: pose classification, movement conversion, and gesture-state integration."""

import math

import pytest

from helpers import make_hand, make_pose_hand
from nocursor.config import Config
from nocursor.interaction import InteractionState, MouseDown, MoveTo, Scroll
from nocursor.processor import GestureProcessor
from nocursor.scroll import (
    FingerState,
    ScrollTracker,
    classify_finger,
    classify_hand,
    extension_ratio,
    scroll_signal,
)

DT = 1 / 30
S = InteractionState


class TestPoseClassification:
    def test_extension_ratios_match_the_built_geometry(self):
        hand = make_pose_hand(index="ext", middle="fold")
        assert extension_ratio(hand, 5, 8) == pytest.approx(1.95)
        assert extension_ratio(hand, 9, 12) == pytest.approx(0.9)

    @pytest.mark.parametrize("ratio,expected", [
        (2.0, FingerState.EXTENDED), (1.55, FingerState.EXTENDED),
        (1.54, FingerState.AMBIGUOUS), (1.35, FingerState.AMBIGUOUS), (1.21, FingerState.AMBIGUOUS),
        (1.2, FingerState.FOLDED), (0.8, FingerState.FOLDED),
        (math.nan, FingerState.AMBIGUOUS), (math.inf, FingerState.AMBIGUOUS),  # unreliable
    ])
    def test_finger_thresholds(self, ratio, expected):
        assert classify_finger(ratio) is expected

    def test_two_finger_pose_is_the_scroll_pose(self):
        assert classify_hand(make_pose_hand()).is_scroll_pose

    @pytest.mark.parametrize("kwargs", [
        {"middle": "fold"},  # normal pointing: index only
        {"ring": "ext", "pinky": "ext"},  # open hand
        {"index": "fold", "middle": "fold"},  # fist
        {"ring": "ext"},  # three fingers
        {"pinky": "ext"},
        {"middle": "amb"},  # half-curled middle finger
        {"index": "amb"},
        {"ring": "amb"},  # half-curled ring finger
        {"pinky": "amb"},
    ])
    def test_other_poses_are_not_the_scroll_pose(self, kwargs):
        assert not classify_hand(make_pose_hand(**kwargs)).is_scroll_pose

    def test_pose_does_not_depend_on_hand_distance(self):
        for size in (0.1, 0.2, 0.3):
            assert classify_hand(make_pose_hand(hand_size=size)).is_scroll_pose

    def test_phase1_style_hands_are_never_scroll_poses(self):
        assert not classify_hand(make_hand()).is_scroll_pose

    def test_scroll_signal_is_the_mean_fingertip_height(self):
        hand = make_pose_hand()
        assert scroll_signal(hand) == pytest.approx(
            (hand.landmarks[8, 1] + hand.landmarks[12, 1]) / 2
        )


class TestScrollTracker:
    def tracker(self, **kw):
        t = ScrollTracker(**{"dead_zone": 0.1, "speed": 10.0, "interval_s": 0.05, **kw})
        t.start(y=1.0, scale=0.2, now=0.0)
        return t

    def test_moving_up_scrolls_up(self):
        t = self.tracker()
        assert t.update(y=1.0 - 0.06, now=0.1) > 0  # 0.3 hand sizes up

    def test_moving_down_scrolls_down(self):
        t = self.tracker()
        assert t.update(y=1.0 + 0.06, now=0.1) < 0

    def test_invert_flips_the_direction(self):
        t = self.tracker(invert=True)
        assert t.update(y=1.0 - 0.06, now=0.1) < 0

    def test_movement_inside_the_dead_zone_is_ignored(self):
        t = self.tracker()
        # dead zone = 0.1 hand sizes = 0.02 frame heights
        assert all(t.update(1.0 + d, 0.1 + i * 0.1) == 0
                   for i, d in enumerate([0.01, -0.01, 0.015, -0.015, 0.0] * 20))

    def test_oscillation_inside_the_dead_zone_never_scrolls_even_at_high_speed(self):
        # +-0.015 frame heights = 0.075 hand sizes, under the 0.1 dead zone. At this speed
        # each wobble would be worth over two clicks if the dead zone were not applied.
        t = self.tracker(speed=30.0, interval_s=0.01)
        total = sum(abs(t.update(1.0 - 0.015 * (i % 2), 0.1 + 0.033 * i)) for i in range(200))
        assert total == 0

    def test_the_same_oscillation_just_outside_the_dead_zone_does_scroll(self):
        t = self.tracker(speed=30.0, interval_s=0.01)
        total = sum(abs(t.update(1.0 - 0.03 * (i % 2), 0.1 + 0.033 * i)) for i in range(200))
        assert total > 0

    def test_a_stationary_hand_never_scrolls(self):
        t = self.tracker()
        assert sum(abs(t.update(1.0, 0.1 + 0.033 * i)) for i in range(300)) == 0

    def test_scroll_amount_is_proportional_to_movement(self):
        small, large = self.tracker(max_clicks=50), self.tracker(max_clicks=50)
        a = small.update(1.0 - 0.04, 0.1)
        b = large.update(1.0 - 0.08, 0.1)
        assert 0 < a < b

    def test_rate_limit_allows_one_event_per_interval(self):
        t = self.tracker(interval_s=0.2)
        emitted = []
        for i in range(40):  # 1.2 s of continuous upward movement at 30 fps
            emitted.append(t.update(1.0 - 0.01 * (i + 1), 0.033 * (i + 1)))
        events = [e for e in emitted if e]
        assert len(events) <= 1.2 / 0.2 + 1
        assert all(0 < e <= 3 for e in events)

    def test_first_event_waits_one_interval(self):
        t = self.tracker(interval_s=0.2)
        assert t.update(1.0 - 0.1, now=0.05) == 0
        assert t.update(1.0 - 0.1, now=0.25) > 0

    def test_backlog_is_capped_so_a_flick_cannot_scroll_forever(self):
        t = self.tracker(max_clicks=3, interval_s=0.05)
        t.update(1.0 - 5.0, now=0.06)  # absurdly large flick
        total = sum(t.update(1.0 - 5.0, 0.06 + 0.05 * (i + 1)) for i in range(100))
        assert total <= 9  # capped backlog (3 x max_clicks), not thousands

    def test_slow_continuous_movement_still_scrolls(self):
        t = self.tracker(interval_s=0.05)
        total = sum(t.update(1.0 - 0.0015 * i, 0.1 + 0.033 * i) for i in range(200))
        assert total > 0

    def test_reset_stops_scrolling_until_started_again(self):
        t = self.tracker()
        t.reset()
        assert not t.active
        assert t.update(0.0, 1.0) == 0

    def test_invalid_start_values_leave_it_inactive(self):
        t = ScrollTracker()
        t.start(math.nan, 0.2, 0.0)
        assert not t.active
        t.start(1.0, 0.0, 0.0)
        assert not t.active

    @pytest.mark.parametrize("kwargs", [
        {"dead_zone": 0}, {"speed": 0}, {"interval_s": -1}, {"max_clicks": 0},
    ])
    def test_invalid_parameters_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ScrollTracker(**kwargs)


class Rig:
    def __init__(self, **cfg):
        base = {"smoothing": 0.0, "stable_frames": 1, "active_margin": 0.0,
                "click_cooldown_s": 0.0, "scroll_stable_frames": 6, "scroll_interval_s": 0.05}
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

    def scrolled(self, frames=None):
        pool = self.frames if frames is None else frames
        return sum(a.clicks for f in pool for a in f.actions if isinstance(a, Scroll))

    def enter_scroll(self, y=0.0):
        self.feed(make_hand(), 3)  # tracked and armed first
        frames = self.feed(make_pose_hand(shift=(0.0, y)), 6)
        assert frames[-1].state is S.SCROLL
        return frames


class TestScrollMode:
    def test_stable_pose_enters_scroll_mode_after_the_configured_frames(self):
        rig = Rig(scroll_stable_frames=6)
        rig.feed(make_hand(), 3)
        states = [f.state for f in rig.feed(make_pose_hand(), 6)]
        assert states == [S.SCROLL_PENDING] * 5 + [S.SCROLL]

    def test_a_brief_pose_never_enters_scroll_mode(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        frames = rig.feed(make_pose_hand(), 5) + rig.feed(make_hand(), 3) + rig.feed(
            make_pose_hand(), 5
        )
        assert all(f.state is not S.SCROLL for f in frames)

    def test_normal_pointing_pose_does_not_enter_scroll_mode(self):
        rig = Rig()
        frames = rig.feed(make_pose_hand(middle="fold"), 60)
        assert all(f.state in (S.POINTER, S.PINCH_PENDING) and not f.scroll_pose for f in frames)
        assert rig.scrolled() == 0

    def test_open_hand_and_fist_do_not_scroll(self):
        rig = Rig()
        frames = rig.feed(make_pose_hand(ring="ext", pinky="ext"), 30)
        frames += rig.feed(make_pose_hand(index="fold", middle="fold"), 30)
        assert all(f.state is not S.SCROLL for f in frames)

    def test_a_pinch_never_enters_scroll_mode(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        frames = rig.feed(make_pose_hand(pinch_ratio=0.10), 40)  # right fingers, thumb touching
        assert all(f.state is not S.SCROLL and not f.scroll_pose for f in frames)

    def test_thumb_touching_during_scroll_ends_it_without_clicking(self):
        rig = Rig()
        rig.enter_scroll()
        touching = rig.feed(make_pose_hand(pinch_ratio=0.10), 40)
        assert touching[0].state is not S.SCROLL
        assert rig.count(MouseDown) == 0  # disarmed: must open the hand first
        assert rig.scrolled(touching) == 0

    def test_moving_the_hand_up_scrolls_up_and_down_scrolls_down(self):
        up = Rig()
        up.enter_scroll()
        up_frames = [up.feed(make_pose_hand(shift=(0.0, -0.01 * i)))[0] for i in range(1, 25)]
        assert up.scrolled(up_frames) > 0

        down = Rig()
        down.enter_scroll()
        down_frames = [down.feed(make_pose_hand(shift=(0.0, 0.01 * i)))[0] for i in range(1, 25)]
        assert down.scrolled(down_frames) < 0

    def test_invert_option_reverses_the_direction(self):
        rig = Rig(scroll_invert=True)
        rig.enter_scroll()
        frames = [rig.feed(make_pose_hand(shift=(0.0, -0.01 * i)))[0] for i in range(1, 25)]
        assert rig.scrolled(frames) < 0

    def test_a_stationary_hand_produces_no_scroll_beyond_the_dead_zone(self):
        rig = Rig()
        rig.enter_scroll()
        jitter = [(0.0, 0.002 * ((-1) ** i)) for i in range(150)]  # tiny landmark wobble
        frames = [rig.feed(make_pose_hand(shift=s))[0] for s in jitter]
        assert rig.scrolled(frames) == 0

    def test_scrolling_stops_when_the_hand_stops(self):
        rig = Rig()
        rig.enter_scroll()
        for i in range(1, 11):
            rig.feed(make_pose_hand(shift=(0.0, -0.01 * i)))
        still = rig.feed(make_pose_hand(shift=(0.0, -0.10)), 60)
        assert abs(rig.scrolled(still)) <= 9  # only the capped backlog drains

    def test_scroll_events_are_rate_limited(self):
        rig = Rig(scroll_interval_s=0.2)
        rig.enter_scroll()
        frames = [rig.feed(make_pose_hand(shift=(0.0, -0.012 * i)))[0] for i in range(1, 31)]
        events = sum(1 for f in frames for a in f.actions if isinstance(a, Scroll))
        assert 0 < events <= 30 * DT / 0.2 + 1

    def test_scroll_never_moves_the_pointer(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        frames = rig.feed(make_pose_hand(), 6)
        frames += [rig.feed(make_pose_hand(shift=(0.02 * i, -0.01 * i)))[0] for i in range(1, 20)]
        assert rig.count(MoveTo, frames) == 0

    def test_leaving_the_pose_exits_scroll_mode_and_resumes_pointing(self):
        rig = Rig()
        rig.enter_scroll()
        after = rig.feed(make_pose_hand(middle="fold"), 3)
        assert after[0].state is not S.SCROLL
        assert after[-1].state is S.POINTER
        assert rig.count(MoveTo, after) > 0

    def test_pinch_after_leaving_scroll_needs_the_normal_confirmation(self):
        rig = Rig()
        rig.enter_scroll()
        rig.feed(make_hand(), 2)
        frames = rig.feed(make_hand(pinch_ratio=0.10), 5)
        assert rig.count(MouseDown, frames) == 1


class TestScrollSafety:
    def test_tracking_loss_stops_scrolling_immediately(self):
        rig = Rig()
        rig.enter_scroll()
        [rig.feed(make_pose_hand(shift=(0.0, -0.01 * i))) for i in range(1, 6)]
        lost = rig.feed(None, 30)
        assert all(f.actions == () and f.state is S.SEARCHING for f in lost)

    def test_scroll_needs_the_stable_period_again_after_tracking_returns(self):
        rig = Rig()
        rig.enter_scroll()
        rig.feed(None)
        back = rig.feed(make_pose_hand(), 6)
        assert back[0].state is not S.SCROLL
        assert back[2].state is not S.SCROLL
        assert rig.scrolled(back) == 0

    def test_low_confidence_pose_does_not_scroll(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        frames = rig.feed(make_pose_hand(confidence=0.2), 30)
        assert all(f.state is S.SEARCHING for f in frames)

    def test_pause_stops_scrolling_and_resume_needs_the_stable_period(self):
        rig = Rig()
        rig.enter_scroll()
        paused = rig.feed(make_pose_hand(shift=(0.0, -0.1)), 30, paused=True)
        assert all(f.actions == () and f.state is S.PAUSED for f in paused)
        resumed = rig.feed(make_pose_hand(shift=(0.0, -0.1)), 3)
        assert all(f.state is not S.SCROLL for f in resumed)

    def test_halt_stops_scrolling(self):
        rig = Rig()
        rig.enter_scroll()
        rig.proc.halt("stop")
        later = [rig.feed(make_pose_hand(shift=(0.0, -0.02 * i)))[0] for i in range(1, 15)]
        assert all(f.actions == () for f in later)
