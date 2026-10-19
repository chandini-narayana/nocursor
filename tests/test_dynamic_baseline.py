"""The rule-based baseline: direction convention, rejection of non-gestures, tuning, storage.

Sequences are SYNTHETIC geometry (see dynamic_helpers); they test the rules, not real-world results.
"""

import json
import math

import numpy as np
import pytest

from dynamic_helpers import circle_path, make_sequence, still_path, swipe_path
from nocursor.dynamic.baseline import (
    GRIDS,
    RuleBaseline,
    RuleParams,
    classify_stats,
    load_baselines,
    save_baseline,
    trajectory_stats,
    tune,
)
from nocursor.dynamic.features import extract_features
from nocursor.dynamic.schema import CLASS_NAMES
from nocursor.dynamic.scoring import macro_f1_of


def features(path, **kwargs):
    return extract_features(make_sequence(path, **kwargs))


DEFAULTS = RuleParams()


def label(path, params=DEFAULTS, **kwargs):
    return classify_stats(trajectory_stats(features(path, **kwargs)), params)


class TestDirectionConvention:
    def test_a_sweep_to_the_users_right_is_swipe_right(self):
        assert label(swipe_path(+1)) == "swipe_right"

    def test_a_sweep_to_the_users_left_is_swipe_left(self):
        assert label(swipe_path(-1)) == "swipe_left"

    def test_the_signs_in_the_statistics(self):
        right = trajectory_stats(features(swipe_path(+1)))
        left = trajectory_stats(features(swipe_path(-1)))
        assert right.net_dx > 1.0 > -1.0 > left.net_dx
        assert right.net_dx == pytest.approx(-left.net_dx)

    def test_a_clockwise_circle_has_positive_turning_and_is_recognised(self):
        stats = trajectory_stats(features(circle_path(True)))
        assert stats.turning == pytest.approx(2 * math.pi, abs=0.8)
        assert label(circle_path(True)) == "circle_clockwise"

    def test_a_counter_clockwise_circle_is_not_a_clockwise_circle(self):
        stats = trajectory_stats(features(circle_path(False)))
        assert stats.turning < -4.0
        assert label(circle_path(False)) == "no_gesture"

    def test_the_decision_depends_only_on_the_features_not_on_pixel_conventions(self):
        """The same gesture in a wide frame must be classified the same way."""
        assert label(swipe_path(+1), aspect=16 / 9) == "swipe_right"
        assert label(circle_path(True), aspect=16 / 9) == "circle_clockwise"


class TestRejection:
    @pytest.mark.parametrize("name, path", [
        ("still hand", still_path()),
        ("vertical sweep", lambda t: (0.0, 0.2 * (2 * t - 1))),
        ("small sideways wiggle", lambda t: (0.012 * math.sin(6 * math.pi * t), 0.0)),
        ("slow drift", lambda t: (0.06 * t, 0.0)),
        ("sweep and quick return", lambda t: (0.2 * math.sin(math.pi * t) , 0.0)),
        ("diagonal sweep", lambda t: (0.2 * (2 * t - 1), 0.2 * (2 * t - 1))),
        ("tiny circle", circle_path(True, radius=0.02)),
    ])
    def test_things_that_are_not_gestures_are_no_gesture(self, name, path):
        assert label(path) == "no_gesture", name

    def test_a_half_circle_is_not_a_circle(self):
        half = lambda t: circle_path(True, 0.12)(t * 0.5)  # noqa: E731
        assert label(half) == "no_gesture"

    def test_the_baseline_never_returns_rejected_or_invalid(self):
        baseline = RuleBaseline(RuleParams(), "rule_defaults")
        for path in (swipe_path(1), swipe_path(-1), circle_path(True), still_path()):
            prediction = baseline.predict_features(features(path))
            assert prediction.valid and prediction.label != "rejected"
            assert prediction.probabilities is None

    def test_the_rules_are_deterministic(self):
        f = features(circle_path(True))
        baseline = RuleBaseline(RuleParams(), "rule_defaults")
        assert baseline.predict_features(f) == baseline.predict_features(f.copy())

    def test_each_criterion_can_veto(self):
        f = features(swipe_path(+1))
        stats = trajectory_stats(f)
        assert classify_stats(stats, RuleParams()) == "swipe_right"
        vetoes = {
            "swipe_min_dx": 99.0, "swipe_min_straightness": 1.01,
            "swipe_min_peak_speed": 999.0, "swipe_max_dy_ratio": -1.0,
        }  # fmt: skip
        for name, value in vetoes.items():
            params = RuleParams(**{name: value})
            assert classify_stats(stats, params) == "no_gesture", name


class TestTuning:
    @staticmethod
    def small_gesture_data():
        """Gestures smaller than the default thresholds expect, plus still negatives."""
        data = []
        for k in range(6):
            data += [
                (features(swipe_path(+1, 0.07 + 0.002 * k)), "swipe_right"),
                (features(swipe_path(-1, 0.07 + 0.002 * k)), "swipe_left"),
                (features(circle_path(True, 0.06 + 0.002 * k)), "circle_clockwise"),
                (features(still_path(), hand_size=0.2 - 0.002 * k), "no_gesture"),
                (features(lambda t: (0.01 * math.sin(6 * math.pi * t), 0.0)), "no_gesture"),
            ]
        return [f for f, _ in data], [lab for _, lab in data]

    def test_tuning_improves_the_objective_when_the_defaults_are_poor(self):
        feats, labels = self.small_gesture_data()
        result = tune(feats, labels)
        assert result.default_objective < 0.6
        assert result.objective > result.default_objective + 0.2
        assert result.params.swipe_min_dx < RuleParams().swipe_min_dx
        assert result.passes >= 1 and result.start in {"defaults", "loosest"}

    def test_tuning_never_makes_the_objective_worse(self):
        feats = [features(swipe_path(+1)), features(swipe_path(-1)), features(still_path())]
        result = tune(feats, ["swipe_right", "swipe_left", "no_gesture"])
        assert result.objective >= result.default_objective

    def test_tuning_is_deterministic(self):
        feats, labels = self.small_gesture_data()
        assert tune(feats, labels) == tune(feats, labels)

    def test_tuned_values_come_from_the_grids(self):
        feats, labels = self.small_gesture_data()
        params = tune(feats, labels).params
        defaults = RuleParams()
        for name, grid in GRIDS.items():
            assert getattr(params, name) in grid or getattr(params, name) == getattr(defaults, name)

    def test_the_objective_is_measured_on_exactly_the_data_given(self):
        """Tuning takes plain features and labels, so a held-out set cannot leak in by accident."""
        feats, labels = self.small_gesture_data()
        half = len(feats) // 2
        result = tune(feats[:half], labels[:half])
        stats = [trajectory_stats(f) for f in feats[:half]]
        matrix = np.zeros((4, 4), dtype=int)
        for truth, s_ in zip(labels[:half], stats, strict=True):
            guess = classify_stats(s_, result.params)
            matrix[CLASS_NAMES.index(truth), CLASS_NAMES.index(guess)] += 1
        assert result.objective == pytest.approx(macro_f1_of(matrix))

    def test_a_plateau_is_escaped_by_the_second_starting_point(self):
        """Loosening one criterion alone changes no prediction here; the loosest start copes."""
        feats, labels = self.small_gesture_data()
        result = tune(feats, labels)
        assert result.start == "loosest"
        assert result.objective > result.default_objective

    def test_an_explicit_start_is_used_alone(self):
        feats, labels = self.small_gesture_data()
        given = tune(feats, labels, start=RuleParams(swipe_min_dx=0.5))
        assert given.start == "given"
        assert given.default_objective == tune(feats, labels).default_objective

    def test_the_defaults_win_ties(self):
        feats = [features(swipe_path(+1)), features(swipe_path(-1)), features(still_path())]
        result = tune(feats, ["swipe_right", "swipe_left", "no_gesture"])
        assert result.start == "defaults" and result.objective == result.default_objective

    @pytest.mark.parametrize("features_, labels", [([], []), ([np.zeros((32, 46))], [])])
    def test_bad_inputs(self, features_, labels):
        with pytest.raises(ValueError):
            tune(features_, labels)


class TestStorage:
    def test_round_trip_and_the_record_of_what_it_was_tuned_on(self, tmp_path):
        feats, labels = TestTuning.small_gesture_data()
        result = tune(feats, labels)
        save_baseline(tmp_path, result, split_id="split-x",
                      tuning_counts={"swipe_left": 6}, exclude_flagged=False)  # fmt: skip
        defaults, tuned, payload = load_baselines(tmp_path)
        assert defaults.name == "rule_defaults" and tuned.name == "rule_tuned"
        assert defaults.params == RuleParams() and tuned.params == result.params
        assert payload["tuned_on"]["splits"] == ["train", "validation"]
        assert "never on the test split" in payload["note"]
        assert payload["split_id"] == "split-x"

    def test_a_damaged_file_is_reported(self, tmp_path):
        (tmp_path / "baseline.json").write_text("{nope", encoding="utf-8")
        with pytest.raises(ValueError, match="cannot read"):
            load_baselines(tmp_path)
        (tmp_path / "baseline.json").write_text(json.dumps({"rule_defaults": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="cannot read"):
            load_baselines(tmp_path)
