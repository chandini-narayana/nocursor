"""Open-palm pose, quality gates, label review and the take recorder (SYNTHETIC fixtures).

A fake clock drives the recorder at 30 fps; no camera is involved.
"""

import numpy as np
import pytest

from dynamic_helpers import circle_path, hand_at, make_sequence, still_path, swipe_path
from helpers import make_pose_hand
from nocursor.dynamic.features import FeatureConfig, extract_features, motion_summary
from nocursor.dynamic.pose import is_open_palm, open_palm_fraction
from nocursor.dynamic.quality import check_quality, review_label, rules_for
from nocursor.dynamic.recorder import RecorderState, TakeRecorder
from nocursor.dynamic.schema import DEFAULT_WINDOW_S, KIND_STREAM

FPS = 30.0
CONFIG = FeatureConfig()


class TestOpenPalmPose:
    def test_all_four_fingers_extended_is_an_open_palm(self):
        assert is_open_palm(make_pose_hand("ext", "ext", "ext", "ext"))

    @pytest.mark.parametrize("fingers", [
        ("ext", "fold", "fold", "fold"),  # pointing
        ("ext", "ext", "fold", "fold"),  # the two-finger scroll pose
        ("fold", "fold", "fold", "fold"),  # fist
        ("ext", "ext", "ext", "amb"),  # one finger half-curled: ambiguous is not open
        ("ext", "ext", "ext", "fold"),
    ])
    def test_other_poses_are_not_an_open_palm(self, fingers):
        assert not is_open_palm(make_pose_hand(*fingers))

    def test_the_thumb_is_not_judged(self):
        spread = make_pose_hand("ext", "ext", "ext", "ext", pinch_ratio=2.0)
        pinched = make_pose_hand("ext", "ext", "ext", "ext", pinch_ratio=0.1)
        assert is_open_palm(spread) and is_open_palm(pinched)

    def test_fraction_counts_only_trusted_frames(self):
        seq = make_sequence(still_path())
        assert open_palm_fraction(seq) == 1.0
        none_trusted = make_sequence(still_path(), usable=np.zeros(seq.n_frames, dtype=bool))
        assert open_palm_fraction(none_trusted) is None


class TestQualityRules:
    def test_gesture_takes_need_pose_tracking_and_no_long_gap(self):
        rules = rules_for("swipe_left", None, "take", CONFIG)
        assert rules.min_open_palm_fraction == 0.8 and rules.max_gap_s == CONFIG.max_gap_s

    def test_negatives_have_no_pose_requirement(self):
        rules = rules_for("no_gesture", "pointing", "take", CONFIG)
        assert rules.min_open_palm_fraction is None and rules.max_gap_s is None

    def test_entering_leaving_tolerates_a_hand_that_is_often_absent(self):
        assert rules_for("no_gesture", "entering_leaving", "take", CONFIG).min_usable_fraction < 0.5

    def test_streams_have_only_structural_checks(self):
        assert rules_for("no_gesture", "continuous_use", KIND_STREAM, CONFIG) is None

    def test_a_clean_swipe_passes(self):
        rules = rules_for("swipe_right", None, "take", CONFIG)
        assert check_quality(make_sequence(swipe_path(1)), rules) == []

    def test_a_scroll_pose_swipe_is_refused_for_the_pose(self):
        seq = make_sequence(swipe_path(1), ring="fold", pinky="fold")
        problems = check_quality(seq, rules_for("swipe_right", None, "take", CONFIG))
        assert any("open palm" in p for p in problems)

    def test_the_gate_judges_pose_and_tracking_not_the_motion(self):
        """A swipe labelled the wrong way still passes the gate: only review warns about it."""
        seq = make_sequence(swipe_path(1), label="swipe_left")
        assert check_quality(seq, rules_for("swipe_left", None, "take", CONFIG)) == []

    def test_low_usable_fraction_is_reported(self):
        count = make_sequence(still_path()).n_frames
        usable = np.arange(count) % 3 != 0  # a third of the frames lost, in single frames
        seq = make_sequence(still_path(), usable=usable)
        problems = check_quality(seq, rules_for("swipe_left", None, "take", CONFIG))
        assert any("reliably in only" in p for p in problems)

    def test_a_long_dropout_is_reported_with_a_hint_about_fast_swipes(self):
        count = make_sequence(still_path()).n_frames
        usable = np.ones(count, dtype=bool)
        usable[10:18] = False
        problems = check_quality(
            make_sequence(still_path(), usable=usable),
            rules_for("swipe_left", None, "take", CONFIG),
        )
        assert any("lost for" in p and "slower" in p for p in problems)


class TestLabelReview:
    def test_correct_labels_are_not_flagged(self):
        assert review_label(make_sequence(swipe_path(1), label="swipe_right")) is None
        assert review_label(make_sequence(swipe_path(-1), label="swipe_left")) is None
        assert review_label(make_sequence(circle_path(True), label="circle_clockwise")) is None

    def test_a_swipe_that_went_the_other_way_is_flagged(self):
        note = review_label(make_sequence(swipe_path(+1), label="swipe_left"))
        assert note and "moved right" in note
        note = review_label(make_sequence(swipe_path(-1), label="swipe_right"))
        assert note and "moved left" in note

    def test_a_tiny_sweep_is_flagged(self):
        note = review_label(make_sequence(swipe_path(1, amplitude=0.01), label="swipe_right"))
        assert note and "small" in note

    def test_a_counter_clockwise_circle_labelled_clockwise_is_flagged(self):
        note = review_label(make_sequence(circle_path(False), label="circle_clockwise"))
        assert note and "counter-clockwise" in note

    def test_a_tiny_loop_is_flagged(self):
        note = review_label(make_sequence(circle_path(True, radius=0.01), label="circle_clockwise"))
        assert note and "small" in note

    def test_negatives_streams_and_unfeaturizable_takes_are_not_reviewed(self):
        assert review_label(make_sequence(swipe_path(1), label="no_gesture")) is None
        assert review_label(make_sequence(swipe_path(1), duration=0.9, label="swipe_left")) is None


def drive(recorder, path=None, *, label="swipe_right", subtype=None, fps=FPS, seconds=3.0,
          hand=None, start_kwargs=None):  # fmt: skip
    """Start a take at t=0 and feed frames until it ends. ``hand(now, fraction)`` may override
    the observation (return None for no hand). Returns the outcome."""
    path = path or swipe_path(1)
    recorder.start(label, subtype, 0.0, **(start_kwargs or {}))
    for i in range(int(seconds * fps)):
        now = i / fps
        fraction = recorder.record_fraction(now)
        observation = hand(now, fraction) if hand else hand_at(path(fraction))
        outcome = recorder.feed(now, observation, 4 / 3)
        if outcome is not None:
            return outcome
    raise AssertionError("the take never finished")


def new_recorder(**kwargs):
    kwargs.setdefault("countdown_s", 0.2)
    return TakeRecorder("sess", **kwargs)


class TestRecorderLifecycle:
    def test_idle_countdown_recording_idle(self):
        recorder = new_recorder()
        assert recorder.state is RecorderState.IDLE
        recorder.start("swipe_right", None, 0.0)
        assert recorder.state is RecorderState.COUNTDOWN
        assert recorder.countdown_remaining(0.05) == pytest.approx(0.15)
        recorder.feed(0.05, hand_at((0, 0)), 4 / 3)
        assert recorder.state is RecorderState.COUNTDOWN
        recorder.feed(0.25, hand_at((0, 0)), 4 / 3)
        assert recorder.state is RecorderState.RECORDING
        assert recorder.countdown_remaining(0.3) == 0.0

    def test_nothing_is_recorded_during_the_countdown(self):
        recorder = new_recorder(countdown_s=1.0)
        recorder.start("swipe_right", None, 0.0)
        for i in range(20):
            assert recorder.feed(i / FPS, hand_at((0, 0)), 4 / 3) is None
        assert recorder.state is RecorderState.COUNTDOWN
        assert recorder._frames == []

    def test_a_good_swipe_is_accepted(self):
        recorder = new_recorder()
        outcome = drive(recorder)
        assert outcome.accepted and outcome.problems == ()
        seq = outcome.sequence
        assert recorder.state is RecorderState.IDLE
        assert (seq.label, seq.kind, seq.subtype) == ("swipe_right", "take", None)
        assert seq.session_id == "sess" and seq.sequence_id == "sess-0001"
        assert seq.t_ms[0] == 0 and np.all(np.diff(seq.t_ms) > 0)
        assert seq.duration_s == pytest.approx(DEFAULT_WINDOW_S, abs=0.05)
        assert seq.usable_fraction == 1.0
        assert motion_summary(extract_features(seq)).peak_dx > 1.0

    def test_raw_landmarks_are_stored_unmirrored_and_unscaled(self):
        recorder = new_recorder()
        outcome = drive(recorder, still_path())
        expected = hand_at((0, 0)).landmarks
        assert np.array_equal(outcome.sequence.landmarks[0], expected)

    def test_each_take_gets_the_next_sequence_number(self):
        recorder = new_recorder()
        first = drive(recorder).sequence
        second = drive(recorder).sequence
        assert (first.sequence_id, second.sequence_id) == ("sess-0001", "sess-0002")

    def test_cannot_start_while_busy(self):
        recorder = new_recorder()
        recorder.start("swipe_left", None, 0.0)
        with pytest.raises(RuntimeError):
            recorder.start("swipe_left", None, 0.1)

    def test_unknown_label_is_refused(self):
        with pytest.raises(ValueError):
            new_recorder().start("wave", None, 0.0)

    def test_cancel_discards_the_take_in_progress(self):
        recorder = new_recorder()
        recorder.start("swipe_left", None, 0.0)
        recorder.feed(0.25, hand_at((0, 0)), 4 / 3)
        recorder.cancel()
        assert recorder.state is RecorderState.IDLE
        assert recorder.feed(0.3, hand_at((0, 0)), 4 / 3) is None

    @pytest.mark.parametrize("kwargs", [{"countdown_s": -1.0}, {"take_s": 0.0}])
    def test_invalid_settings(self, kwargs):
        with pytest.raises(ValueError):
            TakeRecorder("sess", **kwargs)

    def test_hand_usable_follows_the_per_frame_trust_check(self):
        recorder = new_recorder()
        recorder.feed(0.0, hand_at((0, 0)), 4 / 3)
        assert recorder.hand_usable
        recorder.feed(0.03, hand_at((0, 0), ), 4 / 3)
        recorder.feed(0.06, None, 4 / 3)
        assert not recorder.hand_usable
        recorder.feed(0.09, hand_at((0, 0), confidence=0.2), 4 / 3)
        assert not recorder.hand_usable
        recorder.feed(0.12, hand_at((0, 0)), 4 / 3)
        assert recorder.hand_usable  # no multi-frame lock-on delay while recording


class TestRecorderQualityGates:
    def test_no_hand_at_all_is_rejected(self):
        outcome = drive(new_recorder(), hand=lambda now, fraction: None)
        assert not outcome.accepted
        assert any("reliably in only" in p or "usable" in p for p in outcome.problems)

    def test_untrusted_low_confidence_hand_is_rejected(self):
        outcome = drive(new_recorder(), hand=lambda n, f: hand_at(swipe_path(1)(f), confidence=0.3))
        assert not outcome.accepted

    def test_wrong_pose_is_rejected_with_advice(self):
        outcome = drive(new_recorder(), hand=lambda n, f: hand_at(swipe_path(1)(f), ring="fold"))
        assert not outcome.accepted
        assert any("keep all four fingers extended" in p for p in outcome.problems)

    def test_a_long_dropout_in_a_gesture_is_rejected(self):
        def hand(now, fraction):
            return None if 0.3 < fraction < 0.5 else hand_at(swipe_path(1)(fraction))

        outcome = drive(new_recorder(), hand=hand)
        assert not outcome.accepted

    def test_rejected_takes_return_no_sequence(self):
        outcome = drive(new_recorder(), hand=lambda n, f: None)
        assert outcome.sequence is None

    def test_a_negative_in_the_scroll_pose_is_accepted(self):
        outcome = drive(
            new_recorder(), label="no_gesture", subtype="scroll",
            hand=lambda n, f: hand_at(swipe_path(1)(f), ring="fold", pinky="fold"),
        )  # fmt: skip
        assert outcome.accepted
        assert (outcome.sequence.label, outcome.sequence.subtype) == ("no_gesture", "scroll")

    def test_a_negative_in_the_target_motion_is_kept_as_is(self):
        """A hard negative may look just like a gesture; the recorder must not filter it."""
        outcome = drive(new_recorder(), label="no_gesture", subtype="open_palm_other")
        assert outcome.accepted

    def test_entering_leaving_accepts_a_hand_that_appears_part_way(self):
        outcome = drive(
            new_recorder(), label="no_gesture", subtype="entering_leaving",
            hand=lambda n, f: hand_at((0, 0)) if f > 0.5 else None,
        )  # fmt: skip
        assert outcome.accepted

    def test_other_negatives_still_need_a_tracked_hand(self):
        outcome = drive(
            new_recorder(), label="no_gesture", subtype="idle",
            hand=lambda n, f: hand_at((0, 0)) if f > 0.5 else None,
        )  # fmt: skip
        assert not outcome.accepted

    def test_a_camera_too_slow_to_record_enough_frames(self):
        outcome = drive(new_recorder(), fps=4.0)
        assert not outcome.accepted
        assert any("frames per second" in p for p in outcome.problems)

    def test_a_modest_frame_rate_is_fine(self):
        assert drive(new_recorder(), fps=15.0).accepted


class TestContinuousRecording:
    def test_a_stream_is_a_negative_with_the_continuous_use_subtype(self):
        recorder = new_recorder()
        outcome = drive(
            recorder, start_kwargs={"kind": KIND_STREAM, "duration_s": 5.0}, seconds=8.0,
            label="swipe_left",  # ignored: a stream is always a negative
        )  # fmt: skip
        seq = outcome.sequence
        assert outcome.accepted
        assert (seq.kind, seq.label, seq.subtype) == ("stream", "no_gesture", "continuous_use")
        assert seq.duration_s == pytest.approx(5.0, abs=0.1)

    def test_a_stream_needs_a_duration(self):
        with pytest.raises(ValueError):
            new_recorder().start("no_gesture", None, 0.0, kind=KIND_STREAM)

    def test_is_stream_is_true_only_while_one_is_in_progress(self):
        recorder = new_recorder()
        assert not recorder.is_stream
        recorder.start("no_gesture", None, 0.0, kind=KIND_STREAM, duration_s=5.0)
        assert recorder.is_stream
        recorder.cancel()
        assert not recorder.is_stream
