"""Open-palm gate and temporal stabilizer: gating, state transitions, rejection.

Hands are SYNTHETIC geometry (see ``dynamic_helpers``); these tests check software behaviour only.
Window end times use exact binary fractions (multiples of 0.5 s) so that boundary comparisons are
exact and the tests do not depend on floating-point rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from dynamic_helpers import hand_at, make_sequence, still_path
from nocursor.dynamic.gate import (
    GateConfig,
    frame_palm_flags,
    gate_passes,
    window_palm_fraction,
)
from nocursor.dynamic.pose import is_open_palm, open_palm_fraction
from nocursor.dynamic.schema import SequenceRecord
from nocursor.dynamic.stabilizer import (
    Stabilizer,
    StabilizerConfig,
    WindowDecision,
    run_stabilizer,
)


# ------------------------------------------------------------------------------------ gate
class TestPoses:
    def test_a_flat_open_palm_passes(self):
        assert is_open_palm(hand_at((0, 0)))

    @pytest.mark.parametrize("finger", ["index", "middle", "ring", "pinky"])
    def test_a_folded_finger_fails(self, finger):
        assert not is_open_palm(hand_at((0, 0), **{finger: "fold"}))

    @pytest.mark.parametrize("finger", ["index", "middle", "ring", "pinky"])
    def test_an_ambiguous_finger_fails(self, finger):
        assert not is_open_palm(hand_at((0, 0), **{finger: "amb"}))

    def test_the_other_poses_the_application_uses_fail(self):
        assert not is_open_palm(hand_at((0, 0), middle="fold", ring="fold", pinky="fold"))  # point
        assert not is_open_palm(hand_at((0, 0), ring="fold", pinky="fold"))  # two-finger scroll
        assert not is_open_palm(hand_at((0, 0), index="fold", middle="fold", ring="fold",
                                        pinky="fold"))  # fist  # fmt: skip


def mixed(poses: list[dict[str, str]], fps: float = 30.0) -> SequenceRecord:
    """A still hand whose finger pose changes frame by frame."""
    hands = [hand_at((0, 0), **pose) for pose in poses]
    count = len(hands)
    return SequenceRecord(
        "s-0001", "s", "no_gesture", np.rint(np.arange(count) * 1000.0 / fps),
        np.stack([h.landmarks for h in hands]), np.full(count, 0.9), np.ones(count, bool),
        hands[0].aspect_ratio,
    )  # fmt: skip


class TestFrameFlags:
    def test_flags_follow_the_pose_per_frame(self):
        seq = mixed([{}] * 3 + [{"index": "fold"}] * 2 + [{}])
        assert frame_palm_flags(seq).tolist() == [True, True, True, False, False, True]

    def test_untrusted_frames_are_never_open(self):
        base = make_sequence(still_path(), duration=0.5)
        usable = np.ones(base.n_frames, bool)
        usable[2:5] = False
        seq = make_sequence(still_path(), duration=0.5, usable=usable)
        flags = frame_palm_flags(seq)
        assert not flags[2:5].any() and flags[~np.isin(np.arange(seq.n_frames), [2, 3, 4])].all()

    def test_window_fraction_matches_the_recorders_open_palm_fraction(self):
        poses = ([{}] * 7 + [{"index": "fold"}] * 3) * 6  # 60 frames, 70% open
        seq = mixed(poses)
        flags = frame_palm_flags(seq)
        assert window_palm_fraction(flags, seq.usable, 0, seq.n_frames) == pytest.approx(
            open_palm_fraction(seq))
        assert window_palm_fraction(flags, seq.usable, 0, 7) == 1.0
        assert window_palm_fraction(flags, seq.usable, 5, 12) == pytest.approx(
            4 / 7)  # frames 5-11: two open, three folded, two open

    def test_a_window_without_a_trusted_frame_has_no_fraction(self):
        flags = np.zeros(4, bool)
        assert window_palm_fraction(flags, np.zeros(4, bool), 0, 4) is None


class TestGate:
    def test_a_disabled_gate_passes_everything_even_without_a_measurement(self):
        assert gate_passes(None, GateConfig(0.0)) and gate_passes(0.0, GateConfig(0.0))

    def test_an_enabled_gate_fails_closed_without_a_measurement(self):
        assert not gate_passes(None, GateConfig(0.5))

    def test_the_fraction_boundary_is_inclusive(self):
        gate = GateConfig(0.8)
        assert gate_passes(0.8, gate) and gate_passes(0.81, gate) and not gate_passes(0.79, gate)

    def test_full_open_palm_gate(self):
        assert gate_passes(1.0, GateConfig(1.0)) and not gate_passes(0.99, GateConfig(1.0))

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_invalid_fractions_are_refused(self, value):
        with pytest.raises(ValueError):
            GateConfig(value)


# ------------------------------------------------------------------------------ stabilizer
def dec(i, label="swipe_right", conf=0.99, *, evaluated=True, valid=True, gate_ok=True):
    """Decision ``i`` of a stream whose windows end every half second (exact in binary)."""
    end = 2.0 + 0.5 * i
    return WindowDecision(end - 1.5, end, evaluated, label if evaluated else "", conf, valid,
                          gate_ok)  # fmt: skip


def fires(decisions, **config):
    floor = config.pop("floor", 0.0)
    return [a.time_s for a in run_stabilizer(decisions, StabilizerConfig(**config), floor)]


class TestVoting:
    def test_one_window_fires_when_one_of_one_is_enough(self):
        assert fires([dec(0)]) == [2.0]

    def test_a_single_spike_does_not_fire_a_three_of_five_vote(self):
        stream = [dec(0, "no_gesture"), dec(1), dec(2, "no_gesture"), dec(3, "no_gesture")]
        assert fires(stream, k=3, n=5) == []

    def test_it_fires_on_exactly_the_kth_agreeing_window(self):
        stream = [dec(i) for i in range(5)]
        assert fires(stream, k=3, n=5) == [3.0]  # windows 0, 1, 2 agree: decided at the third

    def test_agreement_may_be_interrupted_within_the_last_n(self):
        stream = [dec(0), dec(1, "no_gesture"), dec(2), dec(3, "no_gesture"), dec(4)]
        assert fires(stream, k=3, n=5) == [4.0]

    def test_votes_older_than_n_windows_are_forgotten(self):
        stream = [dec(0), dec(1)] + [dec(i, "no_gesture") for i in range(2, 6)] + [dec(6)]
        assert fires(stream, k=3, n=4) == []

    def test_alternating_left_and_right_never_reaches_a_majority_vote(self):
        labels = ["swipe_left", "swipe_right"] * 6
        stream = [dec(i, label) for i, label in enumerate(labels)]
        assert fires(stream, k=3, n=4, cooldown_s=0.0) == []
        # a looser 2-of-3 vote does admit the flip-flop: the stricter vote is what rejects it
        assert fires(stream, k=2, n=3, cooldown_s=0.0) != []

    def test_the_class_that_reaches_the_vote_is_the_one_that_fires(self):
        stream = [dec(0, "swipe_left"), dec(1, "circle_clockwise"), dec(2, "circle_clockwise")]
        activations = run_stabilizer(stream, StabilizerConfig(k=2, n=3))
        assert [a.label for a in activations] == ["circle_clockwise"]


class TestRejection:
    def test_confidence_below_tau_is_not_a_candidate(self):
        assert fires([dec(0, conf=0.89)], tau=0.9) == []
        assert fires([dec(0, conf=0.9)], tau=0.9) == [2.0]

    def test_the_models_own_rejection_floor_is_never_undercut(self):
        assert fires([dec(0, conf=0.7)], tau=0.5, floor=0.8) == []

    def test_a_perfect_tau_needs_certainty(self):
        assert fires([dec(0, conf=0.999)], tau=1.0) == []
        assert fires([dec(0, conf=1.0)], tau=1.0) == [2.0]

    def test_no_gesture_is_never_a_candidate(self):
        assert fires([dec(0, "no_gesture", conf=1.0)]) == []

    def test_a_rejected_prediction_is_never_a_candidate(self):
        assert fires([dec(0, "rejected", conf=1.0)]) == []

    def test_a_closed_gate_blocks_an_otherwise_perfect_window(self):
        assert fires([dec(0, gate_ok=False)]) == []

    def test_below_tau_windows_do_not_count_towards_the_vote(self):
        stream = [dec(0), dec(1, conf=0.6), dec(2)]
        assert fires(stream, k=3, n=3, tau=0.9) == []


class TestFailClosed:
    def test_a_tracking_loss_clears_the_vote(self):
        stream = [dec(0), dec(1), dec(2, evaluated=False), dec(3)]
        assert fires(stream, k=3, n=5) == []

    def test_the_vote_can_complete_after_a_tracking_loss(self):
        stream = [dec(0), dec(1), dec(2, evaluated=False), dec(3), dec(4), dec(5)]
        assert fires(stream, k=3, n=5) == [4.5]

    def test_an_invalid_prediction_clears_the_vote_and_is_not_counted(self):
        stream = [dec(0), dec(1), dec(2, valid=False), dec(3)]
        assert fires(stream, k=3, n=5) == []

    def test_an_unevaluated_window_never_fires(self):
        assert fires([dec(0, evaluated=False)]) == []

    def test_windows_must_arrive_in_increasing_time_order(self):
        machine = Stabilizer(StabilizerConfig())
        machine.step(dec(1))
        with pytest.raises(ValueError, match="increasing"):
            machine.step(dec(0))
        with pytest.raises(ValueError, match="increasing"):
            machine.step(dec(1))


class TestCooldown:
    def test_nothing_fires_inside_the_cooldown(self):
        stream = [dec(i) for i in range(4)]  # ends 2.0, 2.5, 3.0, 3.5
        assert fires(stream, cooldown_s=1.0) == [2.0, 3.0]

    def test_the_cooldown_boundary_is_inclusive(self):
        assert fires([dec(0), dec(1), dec(2)], cooldown_s=1.0) == [2.0, 3.0]
        assert fires([dec(0), dec(1), dec(2)], cooldown_s=1.5) == [2.0]

    def test_one_long_movement_fires_once_with_a_long_cooldown(self):
        stream = [dec(i) for i in range(16)]  # 8 s of continuous candidates
        assert fires(stream, cooldown_s=10.0) == [2.0]

    def test_a_zero_cooldown_fires_on_every_candidate(self):
        assert len(fires([dec(i) for i in range(4)], cooldown_s=0.0)) == 4

    def test_a_tracking_loss_does_not_end_a_cooldown(self):
        stream = [dec(0), dec(1, evaluated=False), dec(2)]
        assert fires(stream, cooldown_s=1.5) == [2.0]


class TestRearm:
    def test_continued_candidates_do_not_refire_until_a_quiet_stretch(self):
        stream = [dec(i) for i in range(10)]
        assert fires(stream, cooldown_s=1.0, rearm_quiet=2) == [2.0]

    def test_it_rearms_after_enough_quiet_windows(self):
        stream = ([dec(0)] + [dec(i, "no_gesture") for i in (1, 2, 3)] + [dec(4)])
        assert fires(stream, cooldown_s=1.0, rearm_quiet=2) == [2.0, 4.0]

    def test_too_few_quiet_windows_do_not_rearm(self):
        stream = [dec(0), dec(1, "no_gesture"), dec(2), dec(3), dec(4)]
        assert fires(stream, cooldown_s=1.0, rearm_quiet=2) == [2.0]

    def test_a_tracking_loss_counts_as_quiet(self):
        stream = [dec(0), dec(1, evaluated=False), dec(2, evaluated=False), dec(3)]
        assert fires(stream, cooldown_s=1.0, rearm_quiet=2) == [2.0, 3.5]


class TestReturnStroke:
    def test_an_opposite_swipe_is_held_back_by_the_guard(self):
        stream = [dec(0, "swipe_right"), dec(2, "swipe_left"), dec(3, "swipe_left"),
                  dec(4, "swipe_left")]  # fmt: skip
        # cooldown ends at 3.0; the guard extends the hold on the opposite swipe to 4.0
        assert fires(stream, cooldown_s=1.0, opposite_guard_s=1.0) == [2.0, 4.0]

    def test_the_same_direction_is_not_held_back(self):
        stream = [dec(0, "swipe_right"), dec(2, "swipe_right")]
        assert fires(stream, cooldown_s=1.0, opposite_guard_s=5.0) == [2.0, 3.0]

    def test_a_circle_is_not_an_opposite_of_anything(self):
        stream = [dec(0, "swipe_right"), dec(2, "circle_clockwise")]
        assert fires(stream, cooldown_s=1.0, opposite_guard_s=5.0) == [2.0, 3.0]

    def test_without_a_guard_the_return_stroke_fires_after_the_cooldown(self):
        stream = [dec(0, "swipe_right"), dec(2, "swipe_left")]
        assert fires(stream, cooldown_s=1.0) == [2.0, 3.0]


class TestStabilizerBasics:
    def test_reset_forgets_everything(self):
        machine = Stabilizer(StabilizerConfig(cooldown_s=100.0))
        assert machine.step(dec(0)) is not None and machine.step(dec(1)) is None
        machine.reset()
        assert machine.step(dec(2)) is not None

    def test_the_same_input_always_gives_the_same_output(self):
        stream = [dec(i, "swipe_left" if i % 7 < 4 else "no_gesture") for i in range(60)]
        a = fires(stream, k=2, n=3, cooldown_s=1.0)
        assert a == fires(stream, k=2, n=3, cooldown_s=1.0)

    @pytest.mark.parametrize("kwargs", [
        {"tau": 0.0}, {"tau": 1.1}, {"k": 0}, {"k": 3, "n": 2}, {"n": 0},
        {"cooldown_s": -1.0}, {"opposite_guard_s": -0.1}, {"rearm_quiet": -1},
    ])  # fmt: skip
    def test_invalid_settings_are_refused(self, kwargs):
        with pytest.raises(ValueError):
            StabilizerConfig(**kwargs)

    def test_the_activation_records_the_decision_time_and_confidence(self):
        (activation,) = run_stabilizer([dec(0, conf=0.95)], StabilizerConfig())
        assert (activation.time_s, activation.start_s, activation.confidence) == (2.0, 0.5, 0.95)
