"""Experimental settle gate and arming rule (offline variants), and the evaluator plumbing.

All data is SYNTHETIC geometry; these tests check software behaviour, not recognition accuracy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dynamic_helpers import hand_at, make_sequence, still_path, swipe_path
from nocursor.dynamic.arming import (
    ArmingConfig,
    armed_series,
    motion_onset_step,
    palm_speed_series,
)
from nocursor.dynamic.baseline import RuleBaseline, RuleParams
from nocursor.dynamic.features import FeatureConfig
from nocursor.dynamic.gate import GateConfig, frame_palm_flags
from nocursor.dynamic.schema import SequenceRecord
from nocursor.dynamic.stabilizer import StabilizerConfig, WindowDecision, run_stabilizer
from nocursor.dynamic.stream_accept import FreezeError, create_freeze
from nocursor.dynamic.stream_eval import (
    NO_EXTRAS,
    DecisionConfig,
    ExtraGates,
    build_replay_cases,
    build_system_corpus,
    compute_windows,
    decisions_for,
    evaluate_config,
    predict_windows,
    replay_outcomes,
    window_end_speed,
)
from test_dynamic_stream_eval import build_stream, one_swipe

CONFIG = FeatureConfig()
DEFAULTS = RuleBaseline(RuleParams(), "rule_defaults")
FPS = 30.0


def sequence(duration, x_of_s, pose_of_s=lambda s: {}, usable=None):
    return build_stream(duration, x_of_s, pose_of_s, usable=usable)


def hold_then_move(hold_s, total_s=8.0):
    """Still for ``hold_s`` seconds, then drifting right at 0.1 frame heights per second."""
    return sequence(total_s, lambda s: 0.0 if s < hold_s else 0.1 * (s - hold_s))


def armed_at(seq, config):
    return armed_series(seq, frame_palm_flags(seq), config)


# ------------------------------------------------------------------------------- speed
class TestPalmSpeed:
    def test_a_still_hand_has_zero_speed_after_the_first_frames(self):
        speed = palm_speed_series(sequence(2, lambda s: 0.0))
        assert np.isnan(speed[0]) and np.nanmax(speed) == pytest.approx(0.0, abs=1e-9)

    def test_a_steady_drift_has_the_expected_speed_in_hand_sizes_per_second(self):
        # 0.1 frame heights per second with a hand 0.2 frame heights across = 0.5 hand sizes/s
        speed = palm_speed_series(sequence(3, lambda s: 0.1 * s))
        assert np.nanmin(speed[10:]) == pytest.approx(0.5, rel=0.02)
        assert np.nanmax(speed[10:]) == pytest.approx(0.5, rel=0.02)

    def test_the_speed_does_not_depend_on_which_way_the_hand_moves(self):
        a = palm_speed_series(sequence(3, lambda s: 0.1 * s))
        b = palm_speed_series(sequence(3, lambda s: -0.1 * s))
        assert np.allclose(a[10:], b[10:], rtol=1e-6)

    def test_untrusted_frames_have_no_speed(self):
        count = int(3 * FPS) + 1
        usable = np.ones(count, bool)
        usable[20:30] = False
        speed = palm_speed_series(sequence(3, lambda s: 0.1 * s, usable=usable))
        assert np.isnan(speed[20:30]).all()

    def test_a_recording_with_one_trusted_frame_has_no_speed(self):
        count = int(1 * FPS) + 1
        usable = np.zeros(count, bool)
        usable[5] = True
        assert np.isnan(palm_speed_series(sequence(1, lambda s: 0.0, usable=usable))).all()


# ------------------------------------------------------------------------------ arming
class TestArming:
    ARM = ArmingConfig(hold_s=0.5, still_speed=0.4, max_age_s=2.0, drop_s=0.3)

    def test_it_arms_after_a_still_open_hold_and_not_before(self):
        seq = hold_then_move(2.0)
        armed = armed_at(seq, self.ARM)
        t = (seq.t_ms - seq.t_ms[0]) / 1000
        assert not armed[t < 0.55].any()  # the speed needs 0.1 s and the hold 0.5 s
        assert armed[(t > 0.7) & (t < 2.0)].all()

    def test_it_stays_armed_while_the_hand_moves_but_only_for_max_age(self):
        seq = sequence(8, lambda s: 0.0 if s < 1.0 else 0.4 * (s - 1.0))  # fast after 1 s
        armed = armed_at(seq, self.ARM)
        t = (seq.t_ms - seq.t_ms[0]) / 1000
        assert armed[(t > 1.2) & (t < 2.9)].all()  # moving, but within 2 s of the last still frame
        assert not armed[t > 3.2].any()

    def test_a_hand_that_never_rests_never_arms(self):
        seq = sequence(5, lambda s: 0.5 * s)
        assert not armed_at(seq, self.ARM).any()

    def test_a_shorter_hold_than_required_does_not_arm(self):
        seq = sequence(6, lambda s: 0.0 if s < 0.35 else 0.4 * (s - 0.35))
        assert not armed_at(seq, self.ARM).any()
        assert armed_at(seq, ArmingConfig(hold_s=0.2, still_speed=0.4, max_age_s=2.0)).any()

    def test_a_pointing_hand_never_arms(self):
        seq = sequence(4, lambda s: 0.0, lambda s: {"middle": "fold", "ring": "fold",
                                                    "pinky": "fold"})  # fmt: skip
        assert not armed_at(seq, self.ARM).any()

    def test_losing_the_open_palm_disarms_after_the_drop_time(self):
        def pose(s):
            return {"index": "fold"} if 1.5 <= s < 2.0 else {}  # a 0.5 s non-open stretch

        seq = sequence(5, lambda s: 0.0, pose)
        armed = armed_at(seq, ArmingConfig(hold_s=0.5, max_age_s=10.0, drop_s=0.3))
        t = (seq.t_ms - seq.t_ms[0]) / 1000
        assert armed[(t > 0.8) & (t < 1.7)].all()
        assert not armed[(t > 1.9) & (t < 2.0)].any()
        assert armed[t > 2.6].all()  # it has to be earned again by a new still hold

    def test_a_brief_tracking_dropout_does_not_disarm(self):
        count = int(5 * FPS) + 1
        usable = np.ones(count, bool)
        usable[60:68] = False  # 0.27 s, below drop_s
        seq = sequence(5, lambda s: 0.0, usable=usable)
        armed = armed_at(seq, ArmingConfig(hold_s=0.5, max_age_s=10.0, drop_s=0.3))
        assert armed[69:].all()

    def test_a_long_tracking_loss_disarms(self):
        count = int(5 * FPS) + 1
        usable = np.ones(count, bool)
        usable[60:90] = False  # 1 s
        seq = sequence(5, lambda s: 0.0, usable=usable)
        armed = armed_at(seq, ArmingConfig(hold_s=0.5, max_age_s=10.0, drop_s=0.3))
        assert armed[40] and not armed[85]

    @pytest.mark.parametrize("kwargs", [{"hold_s": 0.0}, {"hold_s": 0.5, "still_speed": 0.0},
                                        {"hold_s": 0.5, "max_age_s": -1.0},
                                        {"hold_s": 0.5, "drop_s": 0.0}])
    def test_invalid_settings_are_refused(self, kwargs):
        with pytest.raises(ValueError):
            ArmingConfig(**kwargs)


class TestArmedAtMotionOnset:
    ARM = ArmingConfig(hold_s=0.5, still_speed=0.5, max_age_s=3.0)

    def test_the_onset_is_the_first_step_that_leaves_the_still_band(self):
        features = np.zeros((32, 46))
        features[10:, 2] = 0.8
        assert motion_onset_step(features, 0.5) == 10
        assert motion_onset_step(features, 0.9) is None
        assert motion_onset_step(np.zeros((32, 46)), 0.5) is None

    def armed(self, seq):
        windows = compute_windows(seq, CONFIG, arming_configs=(self.ARM,))
        return windows, windows.armed[self.ARM]

    def test_a_window_with_no_motion_is_never_armed(self):
        _, armed = self.armed(sequence(6, lambda s: 0.0))
        assert not any(armed)

    def test_a_hold_before_the_motion_arms_the_windows_that_contain_the_motion(self):
        seq = sequence(8, lambda s: 0.0 if s < 2.0 else 0.4 * (s - 2.0))
        windows, armed = self.armed(seq)
        containing = [i for i, s in enumerate(windows.starts) if 1.4 <= s <= 2.0]
        assert containing and all(armed[i] for i in containing)

    def test_resting_only_after_the_motion_does_not_arm_it(self):
        # moves from the start, then rests for 4 s: windows that end in the rest still fail
        seq = sequence(8, lambda s: 0.4 * s if s < 1.5 else 0.6)
        windows, armed = self.armed(seq)
        late = [i for i, s in enumerate(windows.starts) if 0.4 <= s <= 1.4]
        assert late and not any(armed[i] for i in late)

    def test_motion_that_follows_a_hold_but_outlasts_max_age_is_not_armed(self):
        seq = sequence(10, lambda s: 0.0 if s < 1.0 else 0.4 * (s - 1.0))
        windows, armed = self.armed(seq)
        mid = [i for i, s in enumerate(windows.starts) if s > 5.0]
        assert mid and not any(armed[i] for i in mid)


# ------------------------------------------------------------------------------- settle
class TestSettleGate:
    def windows(self, seq):
        return compute_windows(seq, CONFIG, arming_configs=())

    def test_the_end_speed_of_a_still_window_is_zero_and_of_a_moving_one_is_not(self):
        still = self.windows(sequence(3, lambda s: 0.0))
        moving = self.windows(sequence(3, lambda s: 0.3 * s))
        assert window_end_speed(still.features[0]) == pytest.approx(0.0, abs=1e-6)
        assert window_end_speed(moving.features[0]) == pytest.approx(1.5, rel=0.05)

    def label_stream(self, label):
        windows = self.windows(sequence(3, lambda s: 0.3 * s))  # end speed about 1.5 hand sizes/s
        outputs = predict_windows("fake", _Always(label), windows)
        return windows, outputs

    def test_a_swipe_that_is_still_moving_at_the_end_of_the_window_is_not_settled(self):
        windows, outputs = self.label_stream("swipe_right")
        blocked = decisions_for(windows, outputs, GateConfig(0.0), ExtraGates(settle_speed=1.0))
        allowed = decisions_for(windows, outputs, GateConfig(0.0), ExtraGates(settle_speed=2.0))
        assert not blocked[0].settled and allowed[0].settled

    def test_circles_are_exempt_because_recorded_circles_end_in_motion(self):
        windows, outputs = self.label_stream("circle_clockwise")
        decisions = decisions_for(windows, outputs, GateConfig(0.0), ExtraGates(settle_speed=0.1))
        assert all(d.settled for d in decisions)

    def test_without_the_gate_everything_is_settled_and_armed(self):
        windows, outputs = self.label_stream("swipe_left")
        decisions = decisions_for(windows, outputs, GateConfig(0.0))
        assert all(d.settled and d.armed for d in decisions)

    def test_the_stabilizer_treats_unsettled_and_unarmed_windows_as_not_candidates(self):
        def window(**kwargs):
            return WindowDecision(0.5, 2.0, True, "swipe_left", 0.99, **kwargs)

        assert len(run_stabilizer([window()], StabilizerConfig())) == 1
        assert not run_stabilizer([window(settled=False)], StabilizerConfig())
        assert not run_stabilizer([window(armed=False)], StabilizerConfig())

    def test_a_swipe_is_decided_only_once_it_has_stopped(self):
        seq = sequence(20, one_swipe)
        windows = compute_windows(seq, CONFIG)
        corpus = build_system_corpus("rule_defaults", DEFAULTS, [windows], [])
        stab = StabilizerConfig(cooldown_s=2.0)
        plain = evaluate_config(corpus, DecisionConfig("rule_defaults", GateConfig(0.0), stab))
        settled = evaluate_config(
            corpus, DecisionConfig("rule_defaults", GateConfig(0.0), stab,
                                   ExtraGates(settle_speed=0.5)))  # fmt: skip
        assert plain.candidates == settled.candidates == 1
        assert settled.streams[0]["times_s"][0] >= plain.streams[0]["times_s"][0]


class _Always:
    """A fake model that always leans towards one class with high confidence."""

    threshold = 0.5

    def __init__(self, label):
        from nocursor.dynamic.classifier import Prediction
        from nocursor.dynamic.schema import CLASS_NAMES

        probs = [0.01] * len(CLASS_NAMES)
        probs[CLASS_NAMES.index(label)] = 0.97
        self.prediction = Prediction(label, tuple(probs))

    def predict_features(self, features):
        return self.prediction


# ------------------------------------------------------------------------ plumbing
class TestExtraGatesPlumbing:
    def test_the_original_pipeline_keeps_its_hash_and_dict(self):
        config = DecisionConfig("gru", GateConfig(0.8), StabilizerConfig(tau=0.9, k=3, n=5))
        assert "extras" not in config.to_dict() and config.extras is NO_EXTRAS
        canonical = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
        import hashlib

        assert config.hash() == hashlib.sha256(canonical.encode()).hexdigest()

    def test_extras_change_the_hash_and_are_recorded(self):
        base = DecisionConfig("gru", GateConfig(0.8), StabilizerConfig())
        settle = DecisionConfig("gru", GateConfig(0.8), StabilizerConfig(),
                                ExtraGates(settle_speed=1.0))  # fmt: skip
        arming = DecisionConfig("gru", GateConfig(0.8), StabilizerConfig(),
                                ExtraGates(arming=ArmingConfig(0.3)))  # fmt: skip
        assert len({base.hash(), settle.hash(), arming.hash()}) == 3
        assert settle.to_dict()["extras"]["settle_speed"] == 1.0
        assert arming.to_dict()["extras"]["arming"]["hold_s"] == 0.3

    def test_mechanism_names(self):
        assert ExtraGates().mechanism() == "base"
        assert ExtraGates(settle_speed=1.0).mechanism() == "settle"
        assert ExtraGates(arming=ArmingConfig(0.3)).mechanism() == "arming"
        assert ExtraGates(1.0, ArmingConfig(0.3)).mechanism() == "settle+arming"

    def test_windows_must_have_been_computed_for_the_arming_rule_that_is_asked_for(self):
        windows = compute_windows(sequence(3, lambda s: 0.0), CONFIG)
        outputs = predict_windows("fake", _Always("swipe_left"), windows)
        with pytest.raises(ValueError, match="not computed"):
            decisions_for(windows, outputs, GateConfig(0.0), ExtraGates(arming=ArmingConfig(0.3)))

    def test_the_decision_cache_does_not_change_any_result(self):
        take = make_sequence(swipe_path(+1), label="swipe_right")
        cases = build_replay_cases([take], CONFIG, arming_configs=(ArmingConfig(0.3),))
        corpus = build_system_corpus("rule_defaults", DEFAULTS, [], cases)
        config = DecisionConfig("rule_defaults", GateConfig(0.8), StabilizerConfig(),
                                ExtraGates(1.0, ArmingConfig(0.3)))  # fmt: skip
        assert evaluate_config(corpus, config).summary() == evaluate_config(
            corpus, config, cache={}).summary()  # fmt: skip

    def test_experimental_variants_cannot_be_frozen(self, tmp_path):
        decision = DecisionConfig("rule_defaults", GateConfig(0.0), StabilizerConfig(),
                                  ExtraGates(settle_speed=1.0))  # fmt: skip
        with pytest.raises(FreezeError, match="cannot be frozen"):
            create_freeze(tmp_path / "runs", tmp_path / "run", [decision], "rule_defaults",
                          tmp_path / "final")  # fmt: skip


# ----------------------------------------------------------------- replay integration
class TestReplayWithTheVariants:
    ARM = ArmingConfig(hold_s=0.5, still_speed=0.5, max_age_s=3.0)

    def corpus(self, take, **kwargs):
        cases = build_replay_cases([take], CONFIG, arming_configs=(self.ARM,), **kwargs)
        return build_system_corpus("rule_defaults", DEFAULTS, [], cases)

    def outcome(self, take, extras, **kwargs):
        (row,) = replay_outcomes(self.corpus(take, **kwargs), GateConfig(0.0), StabilizerConfig(),
                                 extras)  # fmt: skip
        return row

    def test_a_swipe_after_a_synthetic_rest_is_recalled_when_arming_is_required(self):
        take = make_sequence(swipe_path(+1), label="swipe_right")
        assert self.outcome(take, ExtraGates(arming=self.ARM))["outcome"] == "correct"

    def test_a_take_with_no_rest_of_its_own_is_lost_to_arming_when_nothing_is_padded(self):
        take = make_sequence(swipe_path(+1), label="swipe_right")
        row = self.outcome(take, ExtraGates(arming=self.ARM), lead_s=0.0)
        assert row["outcome"] == "missed" and row["reason"] == "arming"
        assert self.outcome(take, ExtraGates(), lead_s=0.0)["outcome"] == "correct"

    def test_a_swipe_still_moving_when_the_take_ends_is_lost_to_the_settle_gate(self):
        take = make_sequence(lambda t: (0.6 * t - 0.3, 0.0), label="swipe_right")
        assert self.outcome(take, ExtraGates(), tail_s=0.0)["outcome"] == "correct"
        row = self.outcome(take, ExtraGates(settle_speed=0.5), tail_s=0.0)
        assert row["outcome"] == "missed" and row["reason"] == "settle"

    def test_the_gate_is_blamed_before_arming_or_settling(self):
        take = make_sequence(swipe_path(+1), label="swipe_right", middle="fold", ring="fold",
                             pinky="fold")  # fmt: skip
        (row,) = replay_outcomes(self.corpus(take), GateConfig(0.8), StabilizerConfig(),
                                 ExtraGates(settle_speed=0.01, arming=self.ARM))  # fmt: skip
        assert row["reason"] == "gate"

    def test_a_still_hand_is_still_a_model_miss(self):
        take = make_sequence(still_path(), label="swipe_right")
        assert self.outcome(take, ExtraGates(arming=self.ARM))["reason"] == "model_or_confidence"


def test_the_stream_helpers_used_here_build_real_sequences():
    assert isinstance(hold_then_move(1.0), SequenceRecord)
    assert hand_at((0, 0)).landmarks.shape == (21, 3)
