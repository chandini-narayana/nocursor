"""Regression: deliberate pinches were never detected on a real webcam after Phase 2.

Root cause covered here: the scroll-pose gate reset the pinch detector on every raw
candidate frame, so a hand that merely *looked like* the two-finger pose left the detector
disarmed (state ``reopen``) and no later pinch could register. Also covered: re-arming after
calibration/pause, realistic (non-flat) landmarks, the depth option, and the read-only
diagnostics that make such failures visible.
"""

import logging

import numpy as np
import pytest

from helpers import make_hand, make_pose_hand
from nocursor.cli import build_parser, main
from nocursor.config import Config
from nocursor.diagnostics import DiagnosticsReporter
from nocursor.gestures import PinchDetector, PinchState, pinch_ratio
from nocursor.interaction import InteractionState, MouseUp
from nocursor.preview import OpenCVPreview, PreviewCommand, StatusInfo
from nocursor.processor import GestureFrame, GestureProcessor
from nocursor.tracking import TrackingState
from test_app_phase2 import build
from test_preview_phase2 import STATUS, WindowlessCV2

DT = 1 / 30
S = InteractionState
REAL = {"aspect_ratio": 4 / 3, "depth": 0.1}  # what a 640x480 webcam + MediaPipe produce


def pointing(**kw):
    """Index finger only (middle folded): never the scroll pose."""
    return make_pose_hand(middle="fold", **kw)


def two_finger(**kw):
    """Index + middle extended, ring + little folded: the scroll pose when the thumb is apart."""
    return make_pose_hand(**kw)


class Rig:
    def __init__(self, **cfg):
        base = {"smoothing": 0.0, "stable_frames": 1, "active_margin": 0.0,
                "click_cooldown_s": 0.0, "diagnostics": True}
        self.proc = GestureProcessor(Config(**{**base, **cfg}), (1920, 1080))
        self.t = 0.0
        self.frames: list[GestureFrame] = []

    def feed(self, hand, n=1, paused=False):
        out = []
        for _ in range(n):
            self.t += DT
            frame = self.proc.process(hand, self.t, paused=paused)
            self.frames.append(frame)
            out.append(frame)
        return out

    def idle(self, hand, n=1):
        out = []
        for _ in range(n):
            self.t += DT
            out.append(self.proc.idle(hand, self.t))
        return out

    @staticmethod
    def presses(frames):
        return sum(f.click for f in frames)

    @staticmethod
    def releases(frames):
        return sum(isinstance(a, MouseUp) for f in frames for a in f.actions)

    @property
    def why(self):
        return self.proc.diagnostics.reason


class TestScrollLookAlikeDoesNotDisarmPinch:
    @pytest.mark.parametrize("dwell", [1, 2, 3, 5])
    def test_pinch_registers_after_briefly_looking_like_the_scroll_pose(self, dwell):
        rig = Rig()
        rig.feed(make_hand(), 3)  # tracked and armed
        rig.feed(two_finger(), dwell)  # scroll look-alike, fewer than the 6 stable frames
        pinch = rig.feed(two_finger(pinch_ratio=0.10), 6)
        assert rig.presses(pinch) == 1

    def test_the_detector_stays_armed_through_look_alike_frames(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        rig.feed(two_finger(), 5)
        assert rig.proc.diagnostics.armed is True
        assert rig.proc.diagnostics.scroll_candidate is True

    def test_flickering_between_look_alike_and_pointing_does_not_block_pinches(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        for _ in range(10):  # never six look-alike frames in a row
            rig.feed(two_finger(), 2)
            rig.feed(pointing(), 1)
        assert rig.presses(rig.feed(pointing(pinch_ratio=0.10), 6)) == 1

    def test_the_reproduction_from_the_bug_report(self):
        """An open hand in a two-finger-looking pose, then a pinch: used to stay 'reopen'."""
        rig = Rig()
        rig.feed(make_hand(), 3)
        rig.feed(two_finger(pinch_ratio=1.0), 3)
        frames = rig.feed(two_finger(pinch_ratio=0.10), 12)
        assert PinchState.REOPEN not in {f.pinch for f in frames}
        assert rig.presses(frames) == 1

    def test_active_scroll_mode_still_blocks_a_pinch_until_the_thumb_opens(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        entered = rig.feed(two_finger(), 8)
        assert entered[-1].state is S.SCROLL
        touching = rig.feed(two_finger(pinch_ratio=0.10), 20)
        assert rig.presses(touching) == 0  # designed: thumb brushing during scroll never clicks
        assert "NOT ARMED" in rig.why
        # Open the thumb with the scroll pose released: armed again, a deliberate pinch works.
        rig.feed(pointing(pinch_ratio=1.0), 2)
        assert rig.presses(rig.feed(pointing(pinch_ratio=0.10), 6)) == 1

    def test_entering_scroll_mode_resets_the_detector_on_the_entering_frame(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        frames = rig.feed(two_finger(), 6)
        assert frames[-1].state is S.SCROLL and rig.proc.diagnostics.armed is False


class TestRearmingAfterCalibrationAndPause:
    @pytest.mark.parametrize("stable", [1, 3])
    def test_a_hand_never_seen_open_after_calibration_cannot_press(self, stable):
        rig = Rig(stable_frames=stable)
        rig.feed(pointing(), 5)
        rig.idle(pointing(), 5)  # the calibration wizard is open
        frames = rig.feed(pointing(pinch_ratio=0.10), 12)  # closes straight away
        assert rig.presses(frames) == 0
        assert rig.why.startswith("NOT ARMED")
        assert frames[-1].pinch is PinchState.REOPEN

    @pytest.mark.parametrize("stable", [1, 3])
    def test_showing_an_open_hand_after_calibration_re_arms_and_pinch_works(self, stable):
        rig = Rig(stable_frames=stable)
        rig.feed(pointing(), 5)
        rig.idle(pointing(), 5)
        rig.feed(pointing(pinch_ratio=0.8), stable + 2)  # open hand, tracking stabilises
        assert rig.proc.diagnostics.armed is True
        pinch = rig.feed(pointing(pinch_ratio=0.10), 7)
        assert rig.presses(pinch) == 1

    def test_thumb_resting_closer_than_the_release_level_never_arms(self):
        rig = Rig()
        rig.feed(pointing(), 3)
        rig.idle(pointing(), 3)
        frames = rig.feed(pointing(pinch_ratio=0.25), 40)  # below 0.30: counts as "close"
        assert rig.presses(frames) == 0
        assert rig.proc.diagnostics.armed is False
        assert "must reach 0.30" in rig.why

    def test_pause_and_resume_follow_the_same_rule(self):
        rig = Rig()
        rig.feed(pointing(), 3)
        rig.feed(pointing(), 5, paused=True)
        assert rig.presses(rig.feed(pointing(pinch_ratio=0.10), 10)) == 0
        rig.feed(pointing(pinch_ratio=1.0), 3)
        assert rig.presses(rig.feed(pointing(pinch_ratio=0.10), 7)) == 1

    def test_calibration_to_dry_run_pinch_end_to_end(self, tmp_path):
        """The reported scenario: calibrate, then pinch in dry-run -> simulated press + release."""
        from nocursor.calibration import CalibrationStore, CalibrationWizard

        corners = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.7), (0.2, 0.7)]  # as seen in the preview

        def corner(view_xy):
            return make_hand(index_xy=(1 - view_xy[0], view_xy[1]), pinch_ratio=0.5)

        script, commands = [], {}
        for c in corners:
            script += [corner(c)] * 5
            commands[len(script)] = (PreviewCommand.CONFIRM,)
        open_hand = make_hand(index_xy=(0.8, 0.5), pinch_ratio=1.0)
        pinched = make_hand(index_xy=(0.8, 0.5), pinch_ratio=0.10)
        tail = [open_hand] * 4 + [pinched] * 8 + [open_hand] * 4
        r = build(script + tail, dry_run=True, wizard=CalibrationWizard(stable_frames=3),
                  store=CalibrationStore(tmp_path / "c.json"), start_in_calibration=True,
                  commands=commands)
        r.run()
        kinds = [rec.kind for rec in r.controller.records]
        assert kinds == ["mouse_down", "mouse_up", "click"]  # simulated press + release
        assert r.backend.events == []  # and nothing reached the real desktop
        assert r.processor.region is not None


class TestRealisticLandmarks:
    def test_a_sustained_pinch_with_realistic_geometry_presses_and_releases(self):
        rig = Rig()
        rig.feed(pointing(**REAL), 4)
        pinch = rig.feed(pointing(pinch_ratio=0.10, thumb_dz=0.005, **REAL), 8)
        release = rig.feed(pointing(pinch_ratio=1.0, **REAL), 3)
        assert pinch[-1].pinch_ratio < 0.20
        assert rig.presses(pinch) == 1 and rig.releases(release) == 1

    def test_open_hand_and_ordinary_movement_never_press(self):
        rng = np.random.default_rng(7)
        rig = Rig()
        frames = []
        for i in range(400):
            ratio = float(rng.uniform(0.45, 1.4))  # fingers moving around, never touching
            shift = (0.05 * np.sin(i / 9), 0.04 * np.cos(i / 13))
            frames += rig.feed(pointing(pinch_ratio=ratio, shift=shift, **REAL))
        assert rig.presses(frames) == 0

    def test_ordinary_finger_motion_through_the_pinch_zone_briefly_does_not_press(self):
        rig = Rig()
        rig.feed(pointing(**REAL), 3)
        for _ in range(10):
            rig.feed(pointing(pinch_ratio=0.10, **REAL), 4)  # four frames: under the five needed
            rig.feed(pointing(pinch_ratio=0.9, **REAL), 2)
        assert rig.presses(rig.frames) == 0

    def test_the_hardware_finding_default_2d_presses_when_tip_depths_disagree(self):
        """Reported on real hardware: tips touch in the image but MediaPipe's z estimates differ
        (here by 0.04 frame widths). The 2D default must press; experimental 3D does not."""
        deep = pointing(pinch_ratio=0.10, thumb_dz=0.04, **REAL)

        default = Rig()  # no depth option given: this is what a plain launch does
        default.feed(pointing(**REAL), 3)
        frames = default.feed(deep, 12)
        assert frames[-1].pinch_ratio < 0.20
        assert default.presses(frames) == 1
        assert default.proc.diagnostics.depth_in_use is False

        experimental = Rig(pinch_use_depth=True)
        experimental.feed(pointing(**REAL), 3)
        frames_3d = experimental.feed(deep, 12)
        assert frames_3d[-1].pinch_ratio > 0.20
        assert experimental.presses(frames_3d) == 0
        assert "run without --pinch-depth" in experimental.why
        assert experimental.proc.diagnostics.depth_in_use is True

    def test_two_dimensional_detection_is_the_default_everywhere(self):
        assert Config().pinch_use_depth is False
        assert GestureProcessor(Config(), (1920, 1080))._use_depth is False
        assert vars(build_parser().parse_args([])).get("pinch_use_depth") is None  # -> default

    @pytest.mark.parametrize("use_depth", [False, True])
    def test_tracking_loss_still_requires_rearming_in_both_modes(self, use_depth):
        rig = Rig(pinch_use_depth=use_depth)
        rig.feed(pointing(**REAL), 3)
        rig.feed(None)  # tracking lost
        assert rig.presses(rig.feed(pointing(pinch_ratio=0.10, **REAL), 12)) == 0  # must re-arm
        rig.feed(pointing(pinch_ratio=1.0, **REAL), 2)
        assert rig.presses(rig.feed(pointing(pinch_ratio=0.10, **REAL), 7)) == 1

    @pytest.mark.parametrize("use_depth", [False, True])
    def test_near_threshold_measurements_do_not_activate(self, use_depth):
        rig = Rig(pinch_use_depth=use_depth)
        rig.feed(pointing(**REAL), 3)
        for gap in (0.23, 0.26, 0.29):
            hand = pointing(pinch_ratio=gap, **REAL)
            # In 3D mode the hand-size reference grows with depth too, so check the value the
            # detector will actually see really is just above the 0.20 threshold.
            measured = pinch_ratio(hand, use_depth=use_depth)
            assert 0.20 < measured < 0.30
            frames = rig.feed(hand, 40)  # held for over a second
            assert rig.presses(frames) == 0
            rig.feed(pointing(pinch_ratio=1.0, **REAL), 2)

    @pytest.mark.parametrize("use_depth", [False, True])
    def test_exactly_five_confirmation_frames_are_needed(self, use_depth):
        for frames_held, expected in ((4, 0), (5, 1)):
            rig = Rig(pinch_use_depth=use_depth)
            rig.feed(pointing(**REAL), 3)
            held = rig.feed(pointing(pinch_ratio=0.10, **REAL), frames_held)
            rig.feed(pointing(pinch_ratio=1.0, **REAL), 3)
            assert rig.presses(held) == expected

    @pytest.mark.parametrize("use_depth", [False, True])
    def test_hysteresis_holds_a_pinch_between_the_threshold_and_release_level(self, use_depth):
        rig = Rig(pinch_use_depth=use_depth)
        rig.feed(pointing(**REAL), 3)
        rig.feed(pointing(pinch_ratio=0.10, **REAL), 5)
        wobble = []
        for ratio in (0.22, 0.27, 0.12, 0.28, 0.21):  # between 0.20 and the 0.30 release level
            wobble += rig.feed(pointing(pinch_ratio=ratio, **REAL), 3)
        assert rig.presses(wobble) == 0 and rig.releases(wobble) == 0  # still one held pinch
        assert rig.releases(rig.feed(pointing(pinch_ratio=0.9, **REAL), 3)) == 1

    def test_scroll_behaviour_is_unchanged_by_the_default(self):
        rig = Rig()
        rig.feed(make_hand(), 3)
        frames = rig.feed(two_finger(**REAL), 6)
        assert frames[-1].state is S.SCROLL
        moved = [rig.feed(make_pose_hand(shift=(0, -0.012 * i), **REAL))[0] for i in range(1, 20)]
        assert any(a.clicks > 0 for f in moved for a in f.actions if hasattr(a, "clicks"))


class TestDiagnosticsAreReadOnly:
    def script(self):
        hands = []
        hands += [make_hand()] * 3 + [pointing(pinch_ratio=1.0)] * 3
        hands += [pointing(pinch_ratio=0.10)] * 8 + [pointing(pinch_ratio=0.10, shift=(0.1, 0))] * 8
        hands += [pointing(pinch_ratio=1.0)] * 4 + [None] * 3
        hands += [two_finger()] * 10 + [make_pose_hand(shift=(0, -0.015 * i)) for i in range(12)]
        hands += [two_finger(pinch_ratio=0.10)] * 8 + [make_hand(confidence=0.2)] * 3
        hands += [pointing(pinch_ratio=1.0, **REAL)] * 3 + [pointing(pinch_ratio=0.1, **REAL)] * 8
        return hands

    def run(self, diagnostics):
        rig = Rig(diagnostics=diagnostics)
        out = []
        for i, hand in enumerate(self.script()):
            out += rig.feed(hand, paused=(40 <= i < 46))
        return out

    def test_every_frame_is_identical_with_and_without_diagnostics(self):
        with_diag, without = self.run(True), self.run(False)
        assert len(with_diag) == len(without) == len(self.script())
        assert with_diag == without
        assert any(f.click for f in with_diag) and any(f.actions for f in with_diag)

    def test_diagnostics_are_off_by_default_and_produce_nothing(self):
        proc = GestureProcessor(Config(), (1920, 1080))
        proc.process(make_hand(), 0.0)
        assert proc.diagnostics is None

    def test_reading_the_snapshot_never_changes_detector_behaviour(self):
        ratios = [1.0] * 3 + [0.1] * 8 + [1.0] * 3 + [0.1] * 8 + [1.0] * 3
        a, b = PinchDetector(release_frames=2), PinchDetector(release_frames=2)
        out_a, out_b = [], []
        for i, r in enumerate(ratios):
            out_a.append(a.update(r, i * DT))
            b.snapshot(i * DT)  # peek before and after every update
            out_b.append(b.update(r, i * DT))
            b.snapshot(i * DT)
        assert out_a == out_b

    def test_diagnostics_never_send_input_in_a_dry_run_session(self):
        script = [pointing(pinch_ratio=1.0)] * 3 + [pointing(pinch_ratio=0.1)] * 10
        r = build(script + [pointing(pinch_ratio=1.0)] * 4, dry_run=True, diagnostics=True,
                  reporter=DiagnosticsReporter())
        r.run()
        assert r.backend.events == []
        assert any(s.diagnostics_lines for s in r.preview.statuses)

    def test_a_live_session_acts_identically_with_and_without_diagnostics(self):
        script = (
            [pointing(pinch_ratio=1.0)] * 3 + [pointing(pinch_ratio=0.1)] * 8
            + [pointing(pinch_ratio=0.1, shift=(0.1 * i / 10, 0)) for i in range(10)]
            + [pointing(pinch_ratio=1.0)] * 4 + [two_finger()] * 6
            + [make_pose_hand(shift=(0, -0.015 * i)) for i in range(14)]
        )
        plain = build(list(script))
        plain.run()
        watched = build(list(script), diagnostics=True, reporter=DiagnosticsReporter())
        watched.run()
        assert plain.backend.events and plain.backend.events == watched.backend.events


class TestDiagnosticContent:
    def test_ratios_thresholds_and_counters_are_reported(self):
        rig = Rig()
        rig.feed(pointing(**REAL), 3)
        d = rig.feed(pointing(pinch_ratio=0.10, **REAL), 3) and rig.proc.diagnostics
        assert d.engage == pytest.approx(0.20) and d.release == pytest.approx(0.30)
        assert d.armed is True and d.candidate_frames == 3 and d.confirm_frames == 5
        assert d.ratio == pytest.approx(d.ratio_2d) and d.ratio_2d == pytest.approx(0.10, abs=0.02)
        assert d.depth_in_use is False and len(d.fingers) == 4
        assert d.state is S.PINCH_PENDING and d.tracking is TrackingState.TRACKING

    def test_confirmation_counter_counts_up_to_the_press(self):
        rig = Rig()
        rig.feed(pointing(), 3)
        seen = []
        for _ in range(5):
            rig.feed(pointing(pinch_ratio=0.10))
            seen.append((rig.proc.diagnostics.candidate_frames, rig.proc.diagnostics.reason))
        assert [c for c, _ in seen[:4]] == [1, 2, 3, 4]
        assert "confirming pinch 1/5" in seen[0][1] and "ACCEPTED" in seen[4][1]

    def test_configured_thresholds_are_what_is_reported(self):
        rig = Rig(pinch_threshold=0.15, pinch_release_factor=2.0)
        rig.feed(pointing(), 2)
        assert rig.proc.diagnostics.engage == pytest.approx(0.15)
        assert rig.proc.diagnostics.release == pytest.approx(0.30)

    def test_scroll_candidate_and_finger_ratios(self):
        rig = Rig()
        rig.feed(two_finger(), 2)
        d = rig.proc.diagnostics
        assert d.scroll_candidate and d.state is S.SCROLL_PENDING
        assert d.fingers[0] > 1.55 and d.fingers[1] > 1.55 and d.fingers[2] < 1.2

    @pytest.mark.parametrize("make,kwargs,expected", [
        (lambda: None, {}, "no hand detected"),
        (lambda: make_hand(confidence=0.2), {}, "confidence 0.20"),
        (lambda: make_hand(index_xy=(1.05, 0.5)), {}, "outside the camera frame"),
        (lambda: make_hand(hand_size=0.02), {}, "too small"),
        (lambda: make_hand(), {"stable_frames": 3}, "acquiring tracking (1/3"),
    ])
    def test_tracking_rejections_are_explained(self, make, kwargs, expected):
        rig = Rig(**kwargs)
        rig.feed(make())
        assert expected in rig.why

    def test_paused_is_explained(self):
        rig = Rig()
        rig.feed(pointing(), 2, paused=True)
        assert "paused" in rig.why

    def test_ratio_above_threshold_is_explained_with_both_numbers(self):
        rig = Rig()
        rig.feed(pointing(pinch_ratio=0.6), 3)
        assert "ratio 0.60 is above the pinch threshold 0.20" in rig.why

    def test_press_hold_and_drag_are_explained(self):
        rig = Rig()
        rig.feed(pointing(), 3)
        rig.feed(pointing(pinch_ratio=0.1), 5)
        assert "ACCEPTED" in rig.why
        rig.feed(pointing(pinch_ratio=0.1), 1)
        assert "button held" in rig.why
        rig.feed(pointing(pinch_ratio=0.1, shift=(-0.08, 0)), 5)
        assert "dragging" in rig.why

    def test_cooldown_refusal_is_explained(self):
        rig = Rig(click_cooldown_s=5.0)
        rig.feed(pointing(), 3)
        rig.feed(pointing(pinch_ratio=0.1), 5)
        rig.feed(pointing(pinch_ratio=1.0), 3)
        rig.feed(pointing(pinch_ratio=0.1), 6)
        assert "refused by cooldown" in rig.why

    def test_scroll_states_are_explained(self):
        rig = Rig()
        rig.feed(make_hand(), 2)
        rig.feed(two_finger(), 2)
        assert "scroll pending" in rig.why
        rig.feed(two_finger(), 6)
        assert "scroll mode" in rig.why


class TestDiagnosticsReporter:
    def diag(self, rig):
        return rig.proc.diagnostics

    def test_log_lines_are_rate_limited(self):
        rig = Rig()
        reporter = DiagnosticsReporter(interval_s=1.0)
        lines = 0
        for i in range(300):  # 10 seconds at 30 fps
            rig.feed(pointing(pinch_ratio=0.5))
            reporter.observe(self.diag(rig), i * DT)
            lines += reporter.log_line(self.diag(rig), i * DT) is not None
        assert 9 <= lines <= 11

    def test_first_line_is_immediate_then_silence_until_the_interval(self):
        rig = Rig()
        reporter = DiagnosticsReporter(interval_s=2.0)
        rig.feed(pointing(), 2)
        assert reporter.log_line(self.diag(rig), 0.0) is not None
        assert reporter.log_line(self.diag(rig), 1.9) is None
        assert reporter.log_line(self.diag(rig), 2.0) is not None

    def test_log_line_carries_the_numbers_needed_to_tune(self):
        rig = Rig()
        rig.feed(pointing(), 3)
        rig.feed(pointing(pinch_ratio=0.1), 2)
        reporter = DiagnosticsReporter()
        reporter.observe(self.diag(rig), 0.0)
        line = reporter.log_line(self.diag(rig), 0.0)
        for needle in ("ratio=0.10", "thr=0.20", "rel=0.30", "armed=yes", "confirm=2/5",
                       "scroll_candidate=no", "lowest3s=0.10", "confirming pinch 2/5"):
            assert needle in line

    def test_overlay_lines_show_ratio_thresholds_arming_and_reason(self):
        rig = Rig()
        rig.feed(pointing(pinch_ratio=0.6), 4)
        reporter = DiagnosticsReporter()
        reporter.observe(self.diag(rig), 0.0)
        lines = reporter.overlay_lines(self.diag(rig))
        text = "\n".join(lines)
        assert "mode: 2D pinch ratio (default)" in text and "ratio 0.60" in text
        assert "pinch <= 0.20" in text
        assert "release >= 0.30" in text and "armed: YES" in text and "why:" in text
        assert all(len(line) < 95 for line in lines)

    def test_lowest_recent_ratio_forgets_old_samples(self):
        rig = Rig()
        reporter = DiagnosticsReporter(window_s=3.0)
        rig.feed(pointing(pinch_ratio=0.1), 1)
        reporter.observe(self.diag(rig), 0.0)
        rig.feed(pointing(pinch_ratio=0.9), 1)
        reporter.observe(self.diag(rig), 1.0)
        assert reporter.min_recent_ratio() == pytest.approx(0.1)
        reporter.observe(self.diag(rig), 4.5)
        assert reporter.min_recent_ratio() == pytest.approx(0.9, abs=0.02)

    def test_no_hand_reports_dashes_not_numbers(self):
        rig = Rig()
        rig.feed(None)
        lines = DiagnosticsReporter().overlay_lines(self.diag(rig))
        assert "mode: 2D pinch ratio (default)" in lines[0]
        assert "no usable hand" in lines[1] and "armed: -" in lines[3]

    @pytest.mark.parametrize("kwargs", [{"interval_s": 0}, {"window_s": -1}])
    def test_invalid_arguments_rejected(self, kwargs):
        with pytest.raises(ValueError):
            DiagnosticsReporter(**kwargs)

    def test_app_logs_diagnostics_rate_limited_and_in_dry_run_only_logs(self, caplog):
        script = [pointing(pinch_ratio=0.6)] * 300
        r = build(script, dry_run=True, diagnostics=True, reporter=DiagnosticsReporter())
        with caplog.at_level(logging.DEBUG, logger="nocursor.diagnostics"):
            r.run()
        diag_lines = [m for m in caplog.messages if m.startswith("diag ")]
        # 300 frames, but at most one line per second of the app's clock
        assert len(diag_lines) < 300 / 5
        assert len(diag_lines) <= r.elapsed_s + 1  # never faster than one per second
        assert len(diag_lines) >= r.elapsed_s / 1.2  # and not starved either
        assert r.backend.events == []


class TestPreviewAndCli:
    def test_not_armed_is_spelled_out_in_the_preview(self):
        p = OpenCVPreview(0.15)
        p._cv2 = WindowlessCV2([-1])
        drawn: list[str] = []
        p._text = lambda view, text, origin, scale, color: drawn.append(text)  # type: ignore[method-assign]
        canvas = np.zeros((240, 640, 3), np.uint8)
        same_ratio = {"tracking": TrackingState.TRACKING, "pinch_ratio": 0.2}
        p._draw_status(canvas, GestureFrame(pinch=PinchState.REOPEN, **same_ratio), STATUS)
        assert any("NOT ARMED" in t and "spread thumb and index" in t for t in drawn)
        drawn.clear()
        p._draw_status(canvas, GestureFrame(pinch=PinchState.OPEN, **same_ratio), STATUS)
        assert not any("NOT ARMED" in t for t in drawn)

    def test_diagnostics_lines_are_drawn_only_when_provided(self):
        p = OpenCVPreview(0.15)
        p._cv2 = WindowlessCV2([-1])
        gesture = GestureFrame(tracking=TrackingState.TRACKING)
        plain, rich = np.zeros((300, 640, 3), np.uint8), np.zeros((300, 640, 3), np.uint8)
        p._draw_status(plain, gesture, STATUS)
        p._draw_status(rich, gesture, StatusInfo("LIVE", 30.0, "k", diagnostics_lines=("a", "b")))
        assert (plain != rich).any()

    def test_cli_flags(self):
        args = vars(build_parser().parse_args(["--diagnostics"]))
        assert args["diagnostics"] is True
        assert vars(build_parser().parse_args([]))["diagnostics"] is False
        assert Config().pinch_threshold == 0.20 and Config().pinch_confirm_frames == 5


def to_config(argv):
    args = vars(build_parser().parse_args(argv))
    args.pop("verbose")
    args.pop("reset_calibration", None)
    return Config.from_overrides(**args)


class TestPinchDepthFlags:
    def test_no_flag_means_2d(self):
        assert to_config([]).pinch_use_depth is False

    def test_no_pinch_depth_is_still_accepted_and_means_2d(self):
        assert to_config(["--no-pinch-depth"]).pinch_use_depth is False

    def test_pinch_depth_opts_in_to_the_experimental_3d_ratio(self):
        assert to_config(["--pinch-depth"]).pinch_use_depth is True

    @pytest.mark.parametrize("argv", [
        ["--pinch-depth", "--no-pinch-depth"], ["--no-pinch-depth", "--pinch-depth"],
    ])
    def test_giving_both_flags_is_an_error_not_a_silent_choice(self, argv, capsys):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(argv)
        assert exc.value.code == 2
        assert "not allowed with" in capsys.readouterr().err

    def test_main_rejects_conflicting_flags_before_touching_hardware(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--dry-run", "--pinch-depth", "--no-pinch-depth"])
        assert exc.value.code == 2

    def test_help_marks_the_default_and_the_experimental_option(self):
        text = " ".join(build_parser().format_help().split())
        assert "--pinch-depth" in text and "EXPERIMENTAL" in text
        assert "already the default" in text

    def test_existing_commands_keep_working(self):
        c = to_config(["--dry-run", "--diagnostics", "--no-pinch-depth", "--pinch-threshold",
                       "0.2", "--pinch-confirm-frames", "5"])
        assert (c.dry_run, c.diagnostics, c.pinch_use_depth) == (True, True, False)
        assert (c.pinch_threshold, c.pinch_confirm_frames) == (0.2, 5)


class TestTwoDimensionalFormula:
    """Default = image-plane thumb-index distance / the existing 2D wrist-to-knuckle size."""

    @pytest.mark.parametrize("gap", [0.05, 0.10, 0.20, 0.35, 0.80])
    def test_ratio_is_exactly_the_image_plane_gap_over_hand_size(self, gap):
        for z in ({}, {"depth": 0.1}, {"depth": 0.3, "thumb_dz": 0.05}):
            hand = make_pose_hand(pinch_ratio=gap, aspect_ratio=4 / 3, **z)
            expected = hand.distance(4, 8) / hand.hand_size  # both purely in the image plane
            assert pinch_ratio(hand) == pytest.approx(expected, abs=1e-12)
            assert pinch_ratio(hand) == pytest.approx(gap, abs=1e-9)

    def test_depth_on_the_hand_changes_only_the_experimental_ratio(self):
        flat = make_pose_hand(pinch_ratio=0.3, aspect_ratio=4 / 3)
        deep = make_pose_hand(pinch_ratio=0.3, aspect_ratio=4 / 3, depth=0.3, thumb_dz=0.05)
        assert pinch_ratio(flat) == pytest.approx(pinch_ratio(deep))
        assert pinch_ratio(flat, use_depth=True) != pytest.approx(
            pinch_ratio(deep, use_depth=True)
        )


class TestStartupBanner:
    def run_main(self, monkeypatch, tmp_path, capsys, argv):
        import nocursor.app
        import nocursor.controller
        import nocursor.emergency
        import nocursor.hand_tracker
        import nocursor.preview
        from nocursor.preview import NullPreview

        class FakeBackend:
            def screen_size(self):
                return (1920, 1080)

        class FakeTracker:
            def __init__(self, *_a, **_k):
                pass

            def close(self):
                pass

        monkeypatch.setattr(nocursor.controller, "PyAutoGUIBackend", FakeBackend)
        monkeypatch.setattr(nocursor.hand_tracker, "HandTracker", FakeTracker)
        monkeypatch.setattr(nocursor.hand_tracker, "resolve_model_path", lambda _p: tmp_path)
        monkeypatch.setattr(nocursor.preview, "OpenCVPreview", lambda _m: NullPreview())
        monkeypatch.setattr(nocursor.emergency, "system_key_state", lambda: None)
        monkeypatch.setattr(nocursor.app.NoCursorApp, "run", lambda self: 0)  # no camera loop
        code = main(["--dry-run", "--calibration-file", str(tmp_path / "c.json"), *argv])
        assert code == 0
        return capsys.readouterr().out

    def test_default_banner_says_2d_default(self, monkeypatch, tmp_path, capsys):
        out = self.run_main(monkeypatch, tmp_path, capsys, [])
        assert "Pinch measurement: 2D pinch ratio (default)." in out

    def test_opt_in_banner_says_3d_experimental(self, monkeypatch, tmp_path, capsys):
        out = self.run_main(monkeypatch, tmp_path, capsys, ["--pinch-depth"])
        assert "Pinch measurement: 3D pinch ratio (experimental)." in out

    def test_no_pinch_depth_banner_says_2d_default(self, monkeypatch, tmp_path, capsys):
        out = self.run_main(monkeypatch, tmp_path, capsys, ["--no-pinch-depth"])
        assert "Pinch measurement: 2D pinch ratio (default)." in out


class TestDiagnosticsStateWhichCalculationIsActive:
    def diag_for(self, **cfg):
        rig = Rig(**cfg)
        rig.feed(pointing(pinch_ratio=0.6, thumb_dz=0.02, **REAL), 3)
        reporter = DiagnosticsReporter()
        reporter.observe(rig.proc.diagnostics, 0.0)
        return reporter, rig.proc.diagnostics

    def test_default_is_labelled_2d_default(self):
        reporter, diag = self.diag_for()
        assert reporter.overlay_lines(diag)[0] == "mode: 2D pinch ratio (default)"
        assert "mode=[2D pinch ratio (default)]" in reporter.log_line(diag, 0.0)

    def test_depth_mode_is_labelled_3d_experimental(self):
        reporter, diag = self.diag_for(pinch_use_depth=True)
        assert reporter.overlay_lines(diag)[0] == "mode: 3D pinch ratio (experimental)"
        assert "mode=[3D pinch ratio (experimental)]" in reporter.log_line(diag, 0.0)

    def test_the_reported_ratio_is_the_one_of_the_active_mode(self):
        _, d2 = self.diag_for()
        _, d3 = self.diag_for(pinch_use_depth=True)
        assert d2.ratio == pytest.approx(d2.ratio_2d) and d3.ratio == pytest.approx(d3.ratio_3d)
        assert d2.ratio != pytest.approx(d3.ratio)  # the two calculations give different numbers

    def test_overlay_lines_fit_the_preview_and_thresholds_are_still_shown(self):
        reporter, diag = self.diag_for()
        lines = reporter.overlay_lines(diag)
        assert all(len(line) < 95 for line in lines)
        assert any("pinch <= 0.20" in line and "release >= 0.30" in line for line in lines)


class TestDryRunInBothModes:
    @pytest.mark.parametrize("use_depth,mode_in_log", [
        (False, "2D pinch ratio (default)"), (True, "3D pinch ratio (experimental)"),
    ])
    def test_a_valid_pinch_gives_a_simulated_press_and_release_and_no_real_input(
        self, use_depth, mode_in_log, caplog
    ):
        open_hand = pointing(pinch_ratio=1.0, **REAL)
        pinched = pointing(pinch_ratio=0.10, thumb_dz=0.005, **REAL)  # small, realistic z gap
        script = [open_hand] * 4 + [pinched] * 8 + [open_hand] * 4
        r = build(script, dry_run=True, diagnostics=True, pinch_use_depth=use_depth,
                  reporter=DiagnosticsReporter())
        with caplog.at_level(logging.DEBUG, logger="nocursor.diagnostics"):
            r.run()
        kinds = [rec.kind for rec in r.controller.records]
        assert kinds == ["mouse_down", "mouse_up", "click"]
        assert not any(rec.sent for rec in r.controller.records)
        assert r.backend.events == []
        assert any(mode_in_log in m for m in caplog.messages if m.startswith("diag "))

    def test_default_dry_run_with_a_depth_disagreement_still_simulates_the_pinch(self):
        script = ([pointing(pinch_ratio=1.0, **REAL)] * 4
                  + [pointing(pinch_ratio=0.10, thumb_dz=0.04, **REAL)] * 8
                  + [pointing(pinch_ratio=1.0, **REAL)] * 4)
        r = build(script, dry_run=True)
        r.run()
        assert [rec.kind for rec in r.controller.records] == ["mouse_down", "mouse_up", "click"]
        assert r.backend.events == []

    def test_live_default_session_presses_and_releases_through_the_controller(self):
        script = ([pointing(pinch_ratio=1.0, **REAL)] * 4
                  + [pointing(pinch_ratio=0.10, thumb_dz=0.04, **REAL)] * 8
                  + [pointing(pinch_ratio=1.0, **REAL)] * 4)
        r = build(script)
        r.run()
        names = r.backend.names()
        assert names.count("down") == 1 and names.count("up") == 1

    def test_emergency_stop_mid_pinch_still_releases_with_the_2d_default(self):
        holder = {}
        script = ([pointing(pinch_ratio=1.0, **REAL)] * 4
                  + [pointing(pinch_ratio=0.10, **REAL)] * 12)
        r = build(script, hooks={13: lambda: holder["r"].stop.trigger("test")})
        holder["r"] = r
        r.run()
        assert r.backend.names().count("down") == 1 and r.backend.names()[-1] == "up"
        assert not r.controller.button_down
