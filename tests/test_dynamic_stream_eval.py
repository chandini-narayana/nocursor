"""Streaming evaluation: windows, replay recall, event counting, statistics, grid and selection.

All data is SYNTHETIC geometry; these tests check software behaviour, not recognition accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from dynamic_helpers import hand_at, make_sequence, still_path, swipe_path
from nocursor.dynamic.baseline import RuleBaseline, RuleParams
from nocursor.dynamic.classifier import Prediction
from nocursor.dynamic.dataset import SessionWriter
from nocursor.dynamic.evaluate import System, evaluate_stream, stream_windows
from nocursor.dynamic.features import FeatureConfig, extract_features
from nocursor.dynamic.gate import GateConfig
from nocursor.dynamic.pose import open_palm_fraction
from nocursor.dynamic.schema import SequenceRecord, SessionHeader
from nocursor.dynamic.stabilizer import Activation, StabilizerConfig
from nocursor.dynamic.stream_eval import (
    COOLDOWN_GRID,
    GATE_GRID,
    GUARD_GRID,
    QUIET_GRID,
    TAU_GRID,
    VOTE_GRID,
    ConfigResult,
    DecisionConfig,
    SealedDataError,
    assert_not_sealed,
    build_replay_cases,
    build_system_corpus,
    classify_replay,
    compute_windows,
    evaluate_config,
    grid_configs,
    grid_hash,
    leave_one_recording_out,
    legacy_counts,
    load_recordings,
    meets_dev_constraints,
    pad_take,
    pareto_front,
    pareto_summary,
    poisson_upper_one_sided,
    predict_windows,
    rate_summary,
    replay_outcomes,
    select_config,
    summarize_recall,
)

CONFIG = FeatureConfig()
DEFAULTS = RuleBaseline(RuleParams(), "rule_defaults")


def build_stream(duration, x_of_s, pose_of_s=lambda s: {}, *, fps=30.0, usable=None,
                 sequence_id="s-stream-0001", session_id="s-stream"):  # fmt: skip
    """A long synthetic recording whose hand position and finger pose follow functions of time."""
    count = int(round(duration * fps)) + 1
    hands = [hand_at((x_of_s(i / fps), 0.0), **pose_of_s(i / fps)) for i in range(count)]
    landmarks = np.stack([h.landmarks for h in hands])
    flags = np.ones(count, bool) if usable is None else np.asarray(usable, bool)
    landmarks[~flags] = np.nan
    return SequenceRecord(
        sequence_id, session_id, "no_gesture", np.rint(np.arange(count) * 1000.0 / fps),
        landmarks, np.where(flags, 0.9, 0.0), flags, hands[0].aspect_ratio,
        subtype="continuous_use", kind="stream",
    )  # fmt: skip


def one_swipe(s):
    """Rest at the left, swipe right over 0.8 s at t=3, then drift back slowly from t=12."""
    if s < 3:
        return -0.2
    if s < 3.8:
        u = (s - 3) / 0.8
        return -0.2 + 0.4 * (3 * u**2 - 2 * u**3)
    if s < 12:
        return 0.2
    return max(-0.2, 0.2 - 0.4 * (s - 12) / 4)


def pointing_after_8(s):
    return {} if s < 8 else {"middle": "fold", "ring": "fold", "pinky": "fold"}


# ----------------------------------------------------------------------------- windows
class TestWindows:
    def test_windows_match_the_earlier_evaluators_windows_and_features(self):
        seq = build_stream(12, one_swipe)
        mine = compute_windows(seq, CONFIG)
        theirs = stream_windows(seq, CONFIG)
        assert len(mine.starts) == len(theirs)
        for i, (start, window) in enumerate(theirs):
            assert mine.starts[i] == pytest.approx(start)
            assert (mine.features[i] is None) == (window is None)
            if window is not None:
                assert np.allclose(mine.features[i], extract_features(window, CONFIG))

    def test_skipped_windows_are_categorised_like_the_earlier_evaluator(self):
        count = int(round(12 * 30)) + 1
        usable = np.ones(count, bool)
        usable[150:170] = False  # a 0.67 s dropout
        seq = build_stream(12, one_swipe, usable=usable)
        windows = compute_windows(seq, CONFIG)
        reference = evaluate_stream(System("rule", DEFAULTS, {}), [seq], CONFIG)
        skipped: dict[str, int] = {}
        for reason in windows.skip:
            if reason:
                skipped[reason] = skipped.get(reason, 0) + 1
        assert skipped == reference["windows_skipped"] and skipped
        assert windows.evaluated == reference["windows_evaluated"]

    def test_palm_fraction_matches_the_recorders_measure_for_every_window(self):
        seq = build_stream(12, one_swipe, pointing_after_8)
        windows = compute_windows(seq, CONFIG)
        for i, (_, window) in enumerate(stream_windows(seq, CONFIG)):
            assert windows.palm_fraction[i] == pytest.approx(open_palm_fraction(window))
        fractions = [f for f in windows.palm_fraction if f is not None]
        assert min(fractions) == 0.0 and max(fractions) == 1.0

    def test_an_offset_shifts_the_window_grid(self):
        seq = build_stream(6, one_swipe)
        base = compute_windows(seq, CONFIG)
        shifted = compute_windows(seq, CONFIG, offset_s=0.05)
        assert shifted.starts[0] == pytest.approx(0.05) and base.starts[0] == 0.0
        assert len(shifted.starts) in (len(base.starts), len(base.starts) - 1)

    def test_minutes_come_from_the_recording_length(self):
        assert compute_windows(build_stream(12, one_swipe), CONFIG).minutes == pytest.approx(0.2)


# ---------------------------------------------------------------------------- predictions
class Probs:
    """A fake model: fixed class probabilities, with its own rejection threshold."""

    name = "fake-gru"

    def __init__(self, probabilities, threshold=0.5, valid=True):
        self.probabilities, self.threshold, self.valid = probabilities, threshold, valid

    def predict_features(self, features):
        if not self.valid:
            return Prediction("rejected", note="invalid", valid=False)
        return Prediction("swipe_left", tuple(self.probabilities))


class TestPredictWindows:
    def windows(self):
        return compute_windows(build_stream(4, lambda s: 0.0), CONFIG)

    def test_the_class_the_model_leans_towards_is_kept_before_any_threshold(self):
        windows = self.windows()
        out = predict_windows("g", Probs([0.1, 0.6, 0.2, 0.1], threshold=0.9), windows)
        assert set(out.labels[: windows.evaluated]) == {"swipe_right"}
        assert out.confidences[0] == pytest.approx(0.6) and out.floor == 0.9

    def test_rule_baselines_have_confidence_one_and_no_floor(self):
        windows = self.windows()
        out = predict_windows("rule", DEFAULTS, windows)
        assert out.floor == 0.0 and set(out.confidences[: windows.evaluated]) == {1.0}

    def test_unevaluated_windows_have_no_label(self):
        count = int(round(4 * 30)) + 1
        usable = np.ones(count, bool)
        usable[10:60] = False
        windows = compute_windows(build_stream(4, lambda s: 0.0, usable=usable), CONFIG)
        out = predict_windows("g", Probs([0.9, 0.05, 0.03, 0.02]), windows)
        assert "" in out.labels and all(
            (windows.features[i] is None) == (out.labels[i] == "") for i in range(len(out.labels)))

    def test_invalid_model_output_is_marked_and_rejected(self):
        out = predict_windows("g", Probs([1, 0, 0, 0], valid=False), self.windows())
        assert not any(out.valid[: self.windows().evaluated]) and "rejected" in out.labels


# ------------------------------------------------------------------------ legacy counting
class TestLegacyCounts:
    def test_they_reproduce_the_earlier_evaluators_event_counts(self):
        seq = build_stream(20, one_swipe)
        windows = compute_windows(seq, CONFIG)
        mine = legacy_counts(windows, predict_windows("rule", DEFAULTS, windows))
        reference = evaluate_stream(System("rule", DEFAULTS, {}), [seq], CONFIG)
        assert mine["flagged_windows"] == reference["flagged_windows"] > 0
        assert mine["events_any"] == reference["events_any"]["events"] >= 1
        assert mine["events_persistent"] == reference["events_persistent"]["events"]

    def test_the_floor_removes_low_confidence_windows_like_the_models_own_rejection(self):
        windows = compute_windows(build_stream(6, lambda s: 0.0), CONFIG)
        probs = [0.7, 0.1, 0.1, 0.1]
        high = legacy_counts(windows, predict_windows("g", Probs(probs, 0.6), windows))
        low = legacy_counts(windows, predict_windows("g", Probs(probs, 0.8), windows))
        assert high["flagged_windows"] > 0 and low["flagged_windows"] == 0


# --------------------------------------------------------------------------------- replay
class TestPadTake:
    def test_it_holds_the_first_and_last_frame_around_the_take(self):
        take = make_sequence(swipe_path(+1), label="swipe_right")
        padded, start, end = pad_take(take, lead_s=1.0, tail_s=1.5)
        assert start == pytest.approx(1.0, abs=0.04)
        assert end - start == pytest.approx(take.duration_s)
        assert padded.duration_s == pytest.approx(take.duration_s + 2.5, abs=0.08)
        assert padded.t_ms[0] == 0 and np.all(np.diff(padded.t_ms) > 0)
        assert np.array_equal(padded.landmarks[0], take.landmarks[0])
        assert np.array_equal(padded.landmarks[-1], take.landmarks[-1])
        assert padded.usable.all() and padded.label == "swipe_right"

    def test_leading_untrusted_frames_of_the_take_are_kept_not_hidden(self):
        base = make_sequence(swipe_path(+1))
        usable = np.ones(base.n_frames, bool)
        usable[:5] = False
        take = make_sequence(swipe_path(+1), usable=usable)
        padded, _, _ = pad_take(take, lead_s=1.0, tail_s=1.0)
        lead = int(round(1.0 * 1000 / 33.3))
        assert not padded.usable[lead : lead + 5].any()

    def test_a_take_without_a_trusted_frame_cannot_be_replayed(self):
        take = make_sequence(still_path(), usable=np.zeros(49, bool))
        with pytest.raises(ValueError):
            pad_take(take)


def act(t, label="swipe_right"):
    return Activation(t, t - 1.6, label, 0.99)


class TestClassifyReplay:
    # a take from 2.0 to 3.6 s; its window is 1.6 s, so activations count up to 5.2 s
    def result(self, activations, truth="swipe_right"):
        return classify_replay(activations, truth, 2.0, 3.6, 1.6)

    def test_the_correct_class_inside_the_window_is_a_hit(self):
        assert self.result([act(4.0)])["outcome"] == "correct"

    def test_no_activation_is_a_miss(self):
        assert self.result([])["outcome"] == "missed"

    def test_a_wrong_class_is_a_cross_activation_even_if_the_right_one_also_fires(self):
        assert self.result([act(4.0, "swipe_left")])["outcome"] == "cross"
        assert self.result([act(4.0), act(4.5, "swipe_left")])["outcome"] == "cross"

    def test_a_second_activation_of_the_right_class_is_a_duplicate(self):
        result = self.result([act(4.0), act(5.0)])
        assert result["outcome"] == "correct" and result["duplicates"] == 1

    def test_the_tolerance_boundaries(self):
        assert self.result([act(2.0)])["outcome"] == "missed"  # ends exactly when the take begins
        assert self.result([act(2.5)])["outcome"] == "correct"
        assert self.result([act(5.2)])["outcome"] == "correct"  # one window after the take ends
        late = self.result([act(5.3)])
        assert late["outcome"] == "missed" and late["stray"] == 1

    def test_activations_in_the_held_rest_are_stray_and_not_a_hit(self):
        result = self.result([act(1.0), act(4.0)])
        assert result["outcome"] == "correct" and result["stray"] == 1

    def test_any_activation_on_a_negative_take_is_a_false_activation(self):
        assert self.result([act(4.0)], "no_gesture")["outcome"] == "false_activation"
        assert self.result([act(9.0)], "no_gesture")["outcome"] == "false_activation"
        assert self.result([], "no_gesture")["outcome"] == "quiet"


def corpus_for(take):
    cases = build_replay_cases([take], CONFIG)
    return build_system_corpus("rule_defaults", DEFAULTS, [], cases)


class TestReplayEndToEnd:
    def test_a_clean_swipe_is_recalled_at_every_alignment(self):
        take = make_sequence(swipe_path(+1), label="swipe_right")
        (outcome,) = replay_outcomes(corpus_for(take), GateConfig(0.0), StabilizerConfig())
        assert outcome["outcome"] == "correct" and outcome["latency_s"] is not None

    def test_every_alignment_is_replayed(self):
        take = make_sequence(swipe_path(+1), label="swipe_right")
        cases = build_replay_cases([take], CONFIG, offsets=(0.0, 0.05))
        assert [c.offset_s for c in cases] == [0.0, 0.05]
        assert cases[0].windows.starts[0] == 0.0 and cases[1].windows.starts[0] == 0.05

    def test_a_still_hand_is_missed_because_of_the_model(self):
        take = make_sequence(still_path(), label="swipe_right")
        (outcome,) = replay_outcomes(corpus_for(take), GateConfig(0.0), StabilizerConfig())
        assert outcome["outcome"] == "missed" and outcome["reason"] == "model_or_confidence"

    def test_a_swipe_in_the_wrong_pose_is_lost_to_the_gate(self):
        take = make_sequence(swipe_path(+1), label="swipe_right", middle="fold", ring="fold",
                             pinky="fold")  # fmt: skip
        corpus = corpus_for(take)
        (open_gate,) = replay_outcomes(corpus, GateConfig(0.0), StabilizerConfig())
        (closed,) = replay_outcomes(corpus, GateConfig(0.8), StabilizerConfig())
        assert open_gate["outcome"] == "correct"
        assert closed["outcome"] == "missed" and closed["reason"] == "gate"

    def test_a_vote_longer_than_the_gesture_is_lost_to_the_stabilizer(self):
        take = make_sequence(swipe_path(+1), label="swipe_right")
        (outcome,) = replay_outcomes(corpus_for(take), GateConfig(0.0),
                                     StabilizerConfig(k=40, n=40))  # fmt: skip
        assert outcome["outcome"] == "missed" and outcome["reason"] == "stabilizer"

    def test_a_swipe_replayed_as_the_other_direction_is_a_cross_activation(self):
        take = make_sequence(swipe_path(+1), label="swipe_left")  # mislabelled on purpose
        (outcome,) = replay_outcomes(corpus_for(take), GateConfig(0.0), StabilizerConfig())
        assert outcome["outcome"] == "cross"

    def test_a_negative_take_that_the_system_fires_on_is_counted(self):
        take = make_sequence(swipe_path(+1), label="no_gesture", subtype="open_palm_other")
        (outcome,) = replay_outcomes(corpus_for(take), GateConfig(0.0), StabilizerConfig())
        assert outcome["outcome"] == "false_activation"

    def test_summary_counts_recall_cross_and_reasons(self):
        rows = [
            {"truth": "swipe_left", "outcome": "correct", "duplicates": 0, "stray": 0},
            {"truth": "swipe_left", "outcome": "missed", "duplicates": 0, "stray": 0,
             "reason": "gate"},
            {"truth": "swipe_right", "outcome": "cross", "duplicates": 0, "stray": 0},
            {"truth": "circle_clockwise", "outcome": "correct", "duplicates": 1, "stray": 1},
            {"truth": "no_gesture", "outcome": "false_activation", "duplicates": 0, "stray": 0},
            {"truth": "no_gesture", "outcome": "quiet", "duplicates": 0, "stray": 0},
        ]  # fmt: skip
        s = summarize_recall(rows)
        assert s["gestures"] == 4 and s["correct"] == 2 and s["pooled_recall"] == 0.5
        assert s["per_class"]["swipe_left"]["recall"] == 0.5
        assert s["per_class"]["swipe_right"]["recall"] == 0.0
        assert s["cross_activations"] == 1 and s["missed_by_reason"] == {"gate": 1}
        assert s["takes_with_duplicate_activations"] == 1 and s["takes_with_stray_activations"] == 1
        assert s["negative_takes"] == 2 and s["negative_takes_with_activation"] == 1


# ------------------------------------------------------------------------ event counting
class TestStreamActivationCounting:
    def corpus(self, seconds=20):
        seq = build_stream(seconds, one_swipe)
        windows = compute_windows(seq, CONFIG)
        return build_system_corpus("rule_defaults", DEFAULTS, [windows], []), windows

    def test_one_movement_is_one_activation_not_dozens_of_windows(self):
        corpus, windows = self.corpus()
        legacy = legacy_counts(*corpus.streams[0])
        result = evaluate_config(corpus, DecisionConfig("rule_defaults", GateConfig(0.0),
                                                        StabilizerConfig(cooldown_s=2.0)))
        assert legacy["flagged_windows"] > 5
        assert result.candidates == 1 and result.streams[0]["candidates"] == 1

    def test_the_rate_uses_the_whole_recording_as_its_denominator(self):
        corpus, windows = self.corpus()
        result = evaluate_config(corpus, DecisionConfig("rule_defaults", GateConfig(0.0),
                                                        StabilizerConfig(cooldown_s=2.0)))
        assert result.minutes == pytest.approx(windows.duration_s / 60)
        assert result.per_minute == pytest.approx(result.candidates / result.minutes)

    def test_a_closed_gate_removes_activations_made_in_the_wrong_pose(self):
        seq = build_stream(20, one_swipe, lambda s: {"middle": "fold", "ring": "fold",
                                                      "pinky": "fold"})  # fmt: skip
        corpus = build_system_corpus("rule_defaults", DEFAULTS, [compute_windows(seq, CONFIG)], [])
        loose = evaluate_config(corpus, DecisionConfig("rule_defaults", GateConfig(0.0),
                                                       StabilizerConfig()))  # fmt: skip
        gated = evaluate_config(corpus, DecisionConfig("rule_defaults", GateConfig(0.8),
                                                       StabilizerConfig()))  # fmt: skip
        assert loose.candidates >= 1 and gated.candidates == 0

    def test_the_recording_ids_are_kept_per_stream(self):
        a = compute_windows(build_stream(20, one_swipe, sequence_id="r1"), CONFIG)
        b = compute_windows(build_stream(20, one_swipe, sequence_id="r2"), CONFIG)
        corpus = build_system_corpus("rule_defaults", DEFAULTS, [a, b], [])
        result = evaluate_config(corpus, DecisionConfig("rule_defaults", GateConfig(0.0),
                                                        StabilizerConfig(cooldown_s=2.0)))
        assert [s["recording_id"] for s in result.streams] == ["r1", "r2"]
        assert result.candidates == sum(s["candidates"] for s in result.streams) == 2


# ------------------------------------------------------------------------------ statistics
class TestPoissonUpperBound:
    @pytest.mark.parametrize("events,expected", [(0, 2.9957), (1, 4.7439), (2, 6.2958),
                                                 (4, 9.1535), (9, 15.7052),
                                                 (10, 16.9622)])
    def test_known_one_sided_95_percent_values(self, events, expected):
        assert poisson_upper_one_sided(events, 1.0) == pytest.approx(expected, abs=2e-3)

    def test_it_scales_with_exposure(self):
        assert poisson_upper_one_sided(0, 40.0) == pytest.approx(2.9957 / 40, abs=1e-4)
        assert poisson_upper_one_sided(4, 40.0) == pytest.approx(9.1535 / 40, abs=1e-4)

    def test_it_is_below_the_two_sided_upper_bound(self):
        two_sided = rate_summary(3, 10.0)["ci95_two_sided_per_minute"][1]
        assert poisson_upper_one_sided(3, 10.0) < two_sided

    def test_more_events_never_lower_the_bound(self):
        bounds = [poisson_upper_one_sided(k, 20.0) for k in range(8)]
        assert bounds == sorted(bounds)

    @pytest.mark.parametrize("args", [(-1, 1.0), (0, 0.0), (0, -1.0)])
    def test_invalid_arguments(self, args):
        with pytest.raises(ValueError):
            poisson_upper_one_sided(*args)

    def test_the_rate_summary_reports_both_bounds(self):
        s = rate_summary(0, 40.0)
        assert s["per_minute"] == 0 and s["ci95_two_sided_per_minute"][0] == 0
        assert s["upper95_one_sided_per_minute"] == pytest.approx(0.0749, abs=1e-3)


# ------------------------------------------------------------------------------ the grid
class TestGrid:
    def test_the_grid_has_its_declared_size(self):
        expected = (len(GATE_GRID) * len(TAU_GRID) * len(VOTE_GRID) * len(COOLDOWN_GRID)
                    * len(QUIET_GRID) * len(GUARD_GRID))  # fmt: skip
        assert len(grid_configs("gru", True)) == expected == 1800
        assert len(grid_configs("rule_defaults", False)) == expected // len(TAU_GRID)

    def test_every_grid_setting_is_valid_and_distinct(self):
        configs = grid_configs("gru", True)
        assert len({c.hash() for c in configs}) == len(configs)

    def test_cooldown_is_never_shorter_than_one_window(self):
        assert min(COOLDOWN_GRID) >= CONFIG.window_s

    def test_the_grid_hash_is_stable_and_sensitive(self, monkeypatch):
        before = grid_hash()
        assert before == grid_hash()
        monkeypatch.setattr("nocursor.dynamic.stream_eval.TAU_GRID", (0.5, 0.6))
        assert grid_hash() != before

    def test_a_decision_config_hash_depends_on_every_field(self):
        base = DecisionConfig("gru", GateConfig(0.8), StabilizerConfig(tau=0.9, k=3, n=5))
        variants = [
            DecisionConfig("rule_tuned", GateConfig(0.8), StabilizerConfig(tau=0.9, k=3, n=5)),
            DecisionConfig("gru", GateConfig(0.9), StabilizerConfig(tau=0.9, k=3, n=5)),
            DecisionConfig("gru", GateConfig(0.8), StabilizerConfig(tau=0.95, k=3, n=5)),
            DecisionConfig("gru", GateConfig(0.8), StabilizerConfig(tau=0.9, k=3, n=5,
                                                                      cooldown_s=3.0)),
        ]  # fmt: skip
        assert len({base.hash(), *(v.hash() for v in variants)}) == 5


# ---------------------------------------------------------------------------- selection
def fake_result(gate, candidates, *, pooled=0.95, classes=(0.95, 0.95, 0.95), cross=0, dup=0,
                n=90, tau=0.9):  # fmt: skip
    per_class = {c: {"recall": r} for c, r in zip(
        ("swipe_left", "swipe_right", "circle_clockwise"), classes, strict=True)}  # fmt: skip
    recall = {"gestures": n, "pooled_recall": pooled, "per_class": per_class,
              "cross_activations": cross, "takes_with_duplicate_activations": dup}  # fmt: skip
    streams = [{"recording_id": r, "minutes": 2.0, "candidates": c}
               for r, c in candidates.items()]  # fmt: skip
    return ConfigResult(DecisionConfig("gru", GateConfig(gate), StabilizerConfig(tau=tau)),
                        streams, sum(candidates.values()), 2.0 * len(candidates), recall)


class TestSelection:
    def test_constraints_boundaries(self):
        assert meets_dev_constraints(fake_result(0.5, {"a": 0}, pooled=0.90).recall)
        assert not meets_dev_constraints(fake_result(0.5, {"a": 0}, pooled=0.89).recall)
        weak_class = fake_result(0.5, {"a": 0}, classes=(0.95, 0.84, 0.95))
        assert not meets_dev_constraints(weak_class.recall)
        # exploratory screening: at most 2 cross activations per 45 target takes (not acceptance)
        assert meets_dev_constraints(fake_result(0.5, {"a": 0}, cross=2, n=45).recall)
        assert not meets_dev_constraints(fake_result(0.5, {"a": 0}, cross=3, n=45).recall)
        assert meets_dev_constraints(fake_result(0.5, {"a": 0}, cross=4, n=90).recall)
        assert not meets_dev_constraints(fake_result(0.5, {"a": 0}, cross=5, n=90).recall)
        assert not meets_dev_constraints(fake_result(0.5, {"a": 0}, dup=5, n=90).recall)
        assert not meets_dev_constraints({"gestures": 0, "pooled_recall": None, "per_class": {},
                                          "cross_activations": 0,
                                          "takes_with_duplicate_activations": 0})  # fmt: skip

    def test_the_fewest_candidates_among_feasible_settings_wins(self):
        results = [fake_result(0.5, {"a": 5, "b": 5}), fake_result(0.8, {"a": 1, "b": 2}),
                   fake_result(1.0, {"a": 0, "b": 0}, pooled=0.5)]  # fmt: skip
        assert select_config(results).config.gate.min_palm_fraction == 0.8

    def test_ties_go_to_the_more_conservative_setting(self):
        results = [fake_result(0.5, {"a": 1}), fake_result(0.9, {"a": 1})]
        assert select_config(results).config.gate.min_palm_fraction == 0.9

    def test_a_better_recall_breaks_a_tie_in_candidates(self):
        results = [fake_result(0.5, {"a": 1}, pooled=0.92), fake_result(0.9, {"a": 1}, pooled=0.97)]
        assert select_config(results).recall["pooled_recall"] == 0.97

    def test_nothing_is_selected_when_no_setting_keeps_recall(self):
        assert select_config([fake_result(0.5, {"a": 0}, pooled=0.2)]) is None

    def test_the_frontier_keeps_only_non_dominated_settings(self):
        results = [fake_result(0.5, {"a": 10}, pooled=0.99),
                   fake_result(0.8, {"a": 5}, pooled=0.95),
                   fake_result(0.9, {"a": 6}, pooled=0.94), fake_result(1.0, {"a": 0}, pooled=0.5)]
        front = pareto_front(results)
        assert [r.candidates for r in front] == [0, 5, 10]


    def test_settings_with_identical_results_collapse_into_one_summary_row(self):
        results = [fake_result(0.5, {"a": 3}, pooled=0.95), fake_result(0.9, {"a": 3}, pooled=0.95),
                   fake_result(1.0, {"a": 1}, pooled=0.92)]
        rows = pareto_summary(results)
        assert [(r["candidates"], r["equivalent_settings"]) for r in rows] == [(1, 1), (3, 2)]
        assert rows[1]["gate"]["min_palm_fraction"] == 0.9  # the more conservative of the two


class TestLeaveOneRecordingOut:
    def test_each_fold_chooses_without_the_recording_it_scores(self):
        a = fake_result(0.8, {"r1": 0, "r2": 0, "r3": 9})  # great on r1, r2; bad on r3
        b = fake_result(0.9, {"r1": 2, "r2": 2, "r3": 2})
        report = leave_one_recording_out([a, b])
        by = {f["held_out"]: f for f in report["folds"]}
        # without r3, setting a looks best (0 candidates) and then fails on the held-out r3
        assert by["r3"]["chosen"]["gate"]["min_palm_fraction"] == 0.8
        assert by["r3"]["held_out_candidates"] == 9
        # without r1, a has 9 and b has 4, so b is chosen and scores 2 on r1
        assert by["r1"]["chosen"]["gate"]["min_palm_fraction"] == 0.9
        assert by["r1"]["held_out_candidates"] == 2
        assert report["pooled_held_out_candidates"] == by["r1"]["held_out_candidates"] + sum(
            by[r]["held_out_candidates"] for r in ("r2", "r3"))  # fmt: skip

    def test_it_needs_at_least_two_recordings(self):
        with pytest.raises(ValueError, match="at least two"):
            leave_one_recording_out([fake_result(0.5, {"only": 1})])

    def test_folds_without_a_feasible_setting_are_reported_not_hidden(self):
        report = leave_one_recording_out([fake_result(0.5, {"a": 0, "b": 0}, pooled=0.1)])
        assert report["folds_without_a_feasible_setting"] == 2
        assert report["pooled_held_out_per_minute"] is None


# -------------------------------------------------------------------- loading and sealing
def write_session(directory, session_id, sequences, *, created="2026-10-20T10:00:00Z",
                  conditions="desk lamp, 60 cm"):  # fmt: skip
    writer = SessionWriter(directory, SessionHeader(session_id, "p01", created, hand="right",
                                                    provenance="synthetic",
                                                    conditions=conditions))  # fmt: skip
    for seq in sequences:
        writer.append(seq)
    writer.close()
    return directory / f"{session_id}.jsonl"


class TestSealing:
    def test_the_final_folder_is_refused(self, tmp_path):
        with pytest.raises(SealedDataError):
            assert_not_sealed(tmp_path / "datasets" / "streaming" / "final")
        with pytest.raises(SealedDataError):
            assert_not_sealed(tmp_path / "datasets" / "Streaming" / "FINAL" / "s-x.jsonl")

    def test_the_dev_folder_and_unrelated_folders_are_allowed(self, tmp_path):
        assert_not_sealed(tmp_path / "datasets" / "streaming" / "dev")
        assert_not_sealed(tmp_path / "datasets")
        assert_not_sealed(tmp_path / "final" / "streaming")

    def test_loading_development_recordings_refuses_the_final_folder(self, tmp_path):
        final = tmp_path / "streaming" / "final"
        write_session(final, "s-final-1", [build_stream(6, one_swipe, session_id="s-final-1",
                                                        sequence_id="s-final-1-0001")])  # fmt: skip
        with pytest.raises(SealedDataError):
            load_recordings([final])
        with pytest.raises(SealedDataError):
            load_recordings([final / "s-final-1.jsonl"])


class TestLoadRecordings:
    def test_streams_and_takes_are_separated_and_hashed(self, tmp_path):
        dev = tmp_path / "streaming" / "dev"
        stream = build_stream(8, one_swipe, session_id="s-dev-1", sequence_id="s-dev-1-0001")
        path = write_session(dev, "s-dev-1", [stream], conditions="window light, 70 cm")
        take = make_sequence(swipe_path(+1), label="swipe_right", session_id="s-dev-2",
                             sequence_id="s-dev-2-0001")  # fmt: skip
        write_session(dev, "s-dev-2", [take])
        streams, takes = load_recordings([dev])
        assert [s.recording_id for s in streams] == ["s-dev-1-0001"]
        assert [t.sequence_id for t in takes] == ["s-dev-2-0001"]
        assert streams[0].conditions == "window light, 70 cm"
        assert streams[0].file_sha256 and streams[0].minutes == pytest.approx(8 / 60)
        assert load_recordings([path])[0][0].file == "s-dev-1.jsonl"

    def test_a_missing_path_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_recordings([tmp_path / "nope"])
