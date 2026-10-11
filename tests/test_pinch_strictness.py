"""Regression tests for accidental pinch activations (stricter defaults, depth-aware ratio)."""

import math

import pytest

from helpers import make_hand
from nocursor.cli import build_parser, main
from nocursor.config import Config
from nocursor.gestures import PinchDetector, PinchState, pinch_ratio
from nocursor.processor import GestureProcessor
from test_app import build  # shared fakes for the loop-level tests

DT = 1 / 30
OPEN = 1.0
FIRM = 0.10  # clearly a pinch
RELAXED = 0.30  # relaxed hand, close to but above the activation threshold


def feed(detector: PinchDetector, ratios, start: float = 0.0):
    """Feed ratios at 30 fps; return the list of results."""
    return [detector.update(r, start + i * DT) for i, r in enumerate(ratios)]


def clicks(results) -> int:
    return sum(r.clicked for r in results)


def default_detector(**kw) -> PinchDetector:
    return PinchDetector(**kw)


class TestDefaults:
    def test_defaults_are_strict(self):
        cfg = Config()
        assert cfg.pinch_threshold == 0.20
        assert cfg.pinch_confirm_frames == 5
        assert cfg.pinch_release_factor == 1.5
        assert cfg.click_cooldown_s == 0.4

    def test_detector_defaults_match_config_defaults(self):
        d = PinchDetector()
        cfg = Config()
        # Probe the behaviour instead of private attributes: 4 frames never click, 5 do.
        d.update(OPEN, 0.0)
        assert clicks(feed(d, [cfg.pinch_threshold] * 4, 0.1)) == 0
        assert clicks(feed(d, [cfg.pinch_threshold], 0.3)) == 1

    def test_confirm_frames_is_exposed_on_the_command_line(self):
        args = build_parser().parse_args(
            ["--pinch-confirm-frames", "7", "--pinch-release-factor", "1.8"]
        )
        assert (args.pinch_confirm_frames, args.pinch_release_factor) == (7, 1.8)

    @pytest.mark.parametrize("flag,value", [
        ("--pinch-confirm-frames", "0"), ("--pinch-release-factor", "1.0"),
        ("--pinch-threshold", "0"),
    ])
    def test_bad_pinch_options_are_rejected_before_hardware_is_touched(self, flag, value, capsys):
        assert main([flag, value]) == 64
        assert "invalid configuration" in capsys.readouterr().err


class TestBriefFalsePinches:
    @pytest.mark.parametrize("dip_frames", [1, 2, 3, 4])
    def test_short_dips_below_threshold_never_click(self, dip_frames):
        d = default_detector()
        results = feed(d, [OPEN, 0.5] + [FIRM] * dip_frames + [0.5, OPEN, OPEN])
        assert clicks(results) == 0

    def test_dip_of_exactly_confirm_frames_does_click(self):
        d = default_detector()
        results = feed(d, [OPEN] + [FIRM] * 5)
        assert clicks(results) == 1
        assert results[-1].clicked  # on the fifth pinched frame, not earlier

    def test_streak_broken_by_one_frame_above_threshold_restarts_count(self):
        d = default_detector()
        results = feed(d, [OPEN] + [FIRM] * 4 + [0.21] + [FIRM] * 4)
        assert clicks(results) == 0

    def test_jitter_around_the_threshold_does_not_activate(self):
        d = default_detector()
        jitter = [0.19, 0.21] * 20  # straddles 0.20 every other frame
        assert clicks(feed(d, [OPEN] + jitter)) == 0


class TestNearThreshold:
    def test_ratio_exactly_at_threshold_counts_toward_a_pinch(self):
        d = default_detector(confirm_frames=1)
        d.update(OPEN, 0.0)
        assert d.update(0.20, 0.1).clicked

    @pytest.mark.parametrize("ratio", [0.2001, 0.21, 0.25, 0.29])
    def test_just_above_threshold_never_activates_even_when_held(self, ratio):
        d = default_detector()
        assert clicks(feed(d, [OPEN] + [ratio] * 90)) == 0  # 3 seconds

    def test_threshold_is_still_configurable(self):
        lenient = default_detector(threshold=0.25)
        strict = default_detector(threshold=0.15)
        for d in (lenient, strict):
            d.update(OPEN, 0.0)
        assert clicks(feed(lenient, [0.22] * 6, 0.1)) == 1
        assert clicks(feed(strict, [0.22] * 6, 0.1)) == 0

    def test_confirm_frames_is_still_configurable(self):
        quick = default_detector(confirm_frames=2)
        quick.update(OPEN, 0.0)
        assert clicks(feed(quick, [FIRM] * 2, 0.1)) == 1


class TestDeliberatePinch:
    def test_sustained_pinch_clicks_exactly_once(self):
        d = default_detector()
        results = feed(d, [OPEN] + [FIRM] * 90)  # held for 3 seconds
        assert clicks(results) == 1

    def test_gradual_close_then_hold_clicks_once(self):
        d = default_detector()
        closing = [OPEN, 0.8, 0.6, 0.45, 0.35, 0.28, 0.22]
        assert clicks(feed(d, closing + [FIRM] * 10)) == 1

    def test_click_happens_within_a_fraction_of_a_second(self):
        d = default_detector()
        results = feed(d, [OPEN] + [FIRM] * 10)
        first = next(i for i, r in enumerate(results) if r.clicked)
        assert first * DT <= 0.2  # 5 frames at 30 fps


class TestReleaseAndHysteresis:
    def test_pinch_is_held_between_activation_and_release_levels(self):
        d = default_detector()
        feed(d, [OPEN] + [FIRM] * 5)
        for ratio in (0.21, 0.25, 0.29):  # release level is 0.30
            r = d.update(ratio, 1.0)
            assert r.state is PinchState.PINCHED and not r.clicked and r.freeze_cursor

    def test_pinch_releases_once_ratio_passes_the_release_level(self):
        d = default_detector()
        feed(d, [OPEN] + [FIRM] * 5)
        r = d.update(0.31, 1.0)
        assert r.state is PinchState.OPEN and not r.clicked and not r.freeze_cursor

    def test_wobbling_inside_the_hysteresis_band_never_repeats_the_click(self):
        d = default_detector(cooldown_s=0.0)
        results = feed(d, [OPEN] + [FIRM] * 5 + [0.19, 0.26, 0.19, 0.28, 0.19, 0.22] * 10)
        assert clicks(results) == 1

    def test_second_deliberate_pinch_needs_release_and_full_confirmation(self):
        d = default_detector(cooldown_s=0.0)
        results = feed(d, [OPEN] + [FIRM] * 5 + [OPEN] * 3 + [FIRM] * 4 + [OPEN] + [FIRM] * 5)
        assert clicks(results) == 2  # the 4-frame attempt in the middle is ignored

    def test_cooldown_still_applies_after_release(self):
        d = default_detector(cooldown_s=0.4)
        first = feed(d, [OPEN] + [FIRM] * 5)
        assert clicks(first) == 1
        # Release and pinch again within 0.4 s of the first click: consumed, no click.
        t = 5 * DT
        again = feed(d, [OPEN] + [FIRM] * 5, t + DT)
        assert clicks(again) == 0


class TestRearmingAfterTrackingLoss:
    def test_reset_requires_open_hand_before_a_pinch_can_confirm(self):
        d = default_detector()
        feed(d, [OPEN] + [FIRM] * 5)
        d.reset()
        assert clicks(feed(d, [FIRM] * 30, 5.0)) == 0
        assert clicks(feed(d, [OPEN] + [FIRM] * 5, 6.0)) == 1

    def test_partial_streak_is_discarded_by_reset(self):
        d = default_detector(cooldown_s=0.0)
        d.update(OPEN, 0.0)
        feed(d, [FIRM] * 4, 0.1)  # one frame short of confirming
        d.reset()  # tracking lost
        results = feed(d, [OPEN] + [FIRM] * 4, 1.0)  # only 4 frames since re-arming
        assert clicks(results) == 0

    def test_processor_with_default_config_rearms_after_loss(self):
        proc = GestureProcessor(Config(smoothing=0.0, stable_frames=1), (1920, 1080))
        t = iter(i * DT for i in range(10_000))
        for _ in range(3):
            proc.process(make_hand(pinch_ratio=OPEN), next(t))
        proc.process(None, next(t))  # tracking lost
        held = [proc.process(make_hand(pinch_ratio=0.1), next(t)).click for _ in range(40)]
        assert not any(held)  # hand reappeared already pinched
        assert not proc.process(make_hand(pinch_ratio=0.1), next(t)).click
        proc.process(make_hand(pinch_ratio=OPEN), next(t))
        deliberate = [proc.process(make_hand(pinch_ratio=0.1), next(t)).click for _ in range(5)]
        assert deliberate == [False, False, False, False, True]


class TestDepthAwareRatio:
    """The 3D ratio is EXPERIMENTAL and opt-in (``use_depth=True`` / ``--pinch-depth``)."""

    def test_two_dimensional_ratio_is_the_default_and_ignores_depth(self):
        hand = make_hand(pinch_ratio=0.0, thumb_dz=0.1)
        assert pinch_ratio(hand) == pytest.approx(0.0)
        assert pinch_ratio(hand) == pinch_ratio(hand, use_depth=False)

    def test_matches_2d_ratio_when_there_is_no_depth_difference(self):
        hand = make_hand(pinch_ratio=0.4, aspect_ratio=16 / 9)
        assert pinch_ratio(hand) == pytest.approx(0.4)
        assert pinch_ratio(hand, use_depth=True) == pytest.approx(0.4)

    def test_depth_gap_increases_the_ratio_when_depth_is_enabled(self):
        flat = pinch_ratio(make_hand(pinch_ratio=0.0), use_depth=True)
        deep = pinch_ratio(make_hand(pinch_ratio=0.0, thumb_dz=0.1), use_depth=True)
        assert flat == pytest.approx(0.0)
        assert deep == pytest.approx(0.5)  # 0.1 / hand size 0.2

    def test_depth_is_aspect_corrected_like_x(self):
        hand = make_hand(pinch_ratio=0.0, thumb_dz=0.1, aspect_ratio=2.0)
        assert pinch_ratio(hand, use_depth=True) == pytest.approx(1.0)  # 0.1 * 2 / 0.2

    @pytest.mark.parametrize("use_depth", [False, True])
    def test_degenerate_hand_is_never_a_pinch(self, use_depth):
        hand = make_hand(hand_size=0.0)
        assert pinch_ratio(hand, use_depth=use_depth) == math.inf
        d = default_detector(confirm_frames=1)
        d.update(OPEN, 0.0)
        assert not d.update(pinch_ratio(hand, use_depth=use_depth), 0.1).clicked

    def test_with_depth_enabled_overlapping_tips_far_apart_in_depth_are_not_a_pinch(self):
        cfg = Config(smoothing=0.0, stable_frames=1, pinch_use_depth=True)
        proc = GestureProcessor(cfg, (1920, 1080))
        results = [proc.process(make_hand(pinch_ratio=OPEN), 0.0)]
        for i in range(1, 60):
            hand = make_hand(pinch_ratio=0.0, thumb_dz=0.08)  # 3D ratio ~0.4
            results.append(proc.process(hand, i * DT))
        assert not any(r.click for r in results)
        assert results[-1].cursor is not None  # not frozen either: 0.4 is above the release level

    def test_by_default_the_same_overlapping_tips_are_a_pinch(self):
        """The accepted trade-off of the 2D default: depth is not consulted."""
        proc = GestureProcessor(Config(smoothing=0.0, stable_frames=1), (1920, 1080))
        results = [proc.process(make_hand(pinch_ratio=OPEN), 0.0)]
        for i in range(1, 12):
            results.append(proc.process(make_hand(pinch_ratio=0.0, thumb_dz=0.08), i * DT))
        assert sum(r.click for r in results) == 1

    @pytest.mark.parametrize("use_depth", [False, True])
    def test_real_pinch_with_touching_tips_clicks_in_both_modes(self, use_depth):
        cfg = Config(smoothing=0.0, stable_frames=1, pinch_use_depth=use_depth)
        proc = GestureProcessor(cfg, (1920, 1080))
        proc.process(make_hand(pinch_ratio=OPEN), 0.0)
        results = [proc.process(make_hand(pinch_ratio=0.05, thumb_dz=0.005), (i + 1) * DT)
                   for i in range(8)]
        assert sum(r.click for r in results) == 1


class TestSafetyPreservedWithStricterDefaults:
    """Loop-level checks using the *default* pinch strictness (5 frames, 0.20)."""

    OPEN_HAND = make_hand(pinch_ratio=OPEN)
    PINCHED_HAND = make_hand(pinch_ratio=0.1)
    strict = {"pinch_confirm_frames": 5, "pinch_threshold": 0.20, "click_cooldown_s": 0.4}

    @pytest.mark.parametrize("pinched_frames,expected_clicks", [(4, 0), (5, 1)])
    def test_live_mode_clicks_only_after_five_pinched_frames(self, pinched_frames, expected_clicks):
        script = [self.OPEN_HAND] + [self.PINCHED_HAND] * pinched_frames + [self.OPEN_HAND]
        app, backend, *_ = build(script, **self.strict)
        app.run()
        assert backend.clicks == expected_clicks

    def test_dry_run_sends_nothing_even_for_a_deliberate_pinch(self):
        app, backend, *_ = build([self.OPEN_HAND] + [self.PINCHED_HAND] * 12 + [self.OPEN_HAND],
                                 dry_run=True, **self.strict)
        app.run()
        assert backend.moves == [] and backend.clicks == 0

    def test_no_click_when_tracking_is_lost_mid_pinch(self):
        script = [self.OPEN_HAND] + [self.PINCHED_HAND] * 4 + [None] + [self.PINCHED_HAND] * 4
        app, backend, *_ = build(script, **self.strict)
        app.run()
        assert backend.clicks == 0  # neither partial streak reaches 5, and loss re-arms

    def test_emergency_stop_still_blocks_a_confirmed_pinch(self):
        script = [self.OPEN_HAND] + [self.PINCHED_HAND] * 10
        app, backend, *_ = build(script, trigger_stop_at=3, **self.strict)
        app.run()
        assert backend.clicks == 0
