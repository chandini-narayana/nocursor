"""Metrics, intervals, the continuous-stream analysis, and end-to-end evaluation reports.

All data is SYNTHETIC (see ml_fixtures). The end-to-end runs use a random-weight model, so the
numbers they produce mean nothing; the tests check the machinery, the counts and the wording.
"""

import json
import random
import shutil

import numpy as np
import pytest

from dynamic_helpers import make_sequence, still_path
from ml_fixtures import FULL_SESSIONS, add_random_gru, build_study, freeze_split
from nocursor.dynamic.classifier import REJECTED, Prediction
from nocursor.dynamic.evaluate import (
    EVAL_LOG,
    EvaluationError,
    System,
    _events,
    evaluate_run,
    evaluate_stream,
    stream_windows,
    time_system,
)
from nocursor.dynamic.features import FeatureConfig
from nocursor.dynamic.model_io import META_FILE
from nocursor.dynamic.schema import SequenceRecord
from nocursor.dynamic.scoring import (
    ScoredSample,
    accuracy_of,
    bootstrap_interval,
    confusion_matrix,
    effective_matrix,
    macro_f1_of,
    poisson_rate_interval,
    precision_recall_f1,
    score_samples,
    wilson_interval,
)
from nocursor.dynamic.splits import load_split_items
from nocursor.dynamic.train import TrainConfig, run_fit_baseline


def sample(true, predicted, subtype=None, flagged=False, valid=True):
    return ScoredSample(true, Prediction(predicted, valid=valid), subtype, f"{true}-{predicted}",
                        flagged)  # fmt: skip


# 12 hand-checkable samples. After folding the rejections into no_gesture, the matrix is:
#                  swipe_left swipe_right circle no_gesture
#   swipe_left          2          1         0       1       (one cross activation, one rejected)
#   swipe_right         0          1         0       1
#   circle_clockwise    0          0         2       0
#   no_gesture          1          0         0       3       (one false activation, one rejected)
SAMPLES = [
    sample("swipe_left", "swipe_left"), sample("swipe_left", "swipe_left"),
    sample("swipe_left", "swipe_right"), sample("swipe_left", REJECTED),
    sample("swipe_right", "swipe_right"), sample("swipe_right", "no_gesture"),
    sample("circle_clockwise", "circle_clockwise"), sample("circle_clockwise", "circle_clockwise"),
    sample("no_gesture", "no_gesture", "idle"), sample("no_gesture", "no_gesture", "idle"),
    sample("no_gesture", "swipe_left", "pointing"), sample("no_gesture", REJECTED, "pinch"),
]  # fmt: skip


class TestMatrices:
    def test_the_raw_matrix_keeps_a_rejected_column(self):
        raw = confusion_matrix(SAMPLES)
        assert raw.shape == (4, 5)
        assert raw[0].tolist() == [2, 1, 0, 0, 1]  # swipe_left row; last column is "rejected"
        assert raw[3].tolist() == [1, 0, 0, 2, 1]

    def test_rejections_are_folded_into_no_gesture_for_the_headline(self):
        eff = effective_matrix(confusion_matrix(SAMPLES))
        assert eff.tolist() == [[2, 1, 0, 1], [0, 1, 0, 1], [0, 0, 2, 0], [1, 0, 0, 3]]

    def test_precision_recall_and_f1_against_hand_computed_values(self):
        stats = precision_recall_f1(effective_matrix(confusion_matrix(SAMPLES)))
        assert stats["swipe_left"]["precision"] == pytest.approx(2 / 3)
        assert stats["swipe_left"]["recall"] == pytest.approx(2 / 4)
        assert stats["swipe_left"]["f1"] == pytest.approx(0.571428, abs=1e-5)
        assert stats["swipe_right"]["precision"] == pytest.approx(1 / 2)
        assert stats["circle_clockwise"]["f1"] == pytest.approx(1.0)
        assert stats["no_gesture"]["precision"] == pytest.approx(3 / 5)
        assert stats["no_gesture"]["recall"] == pytest.approx(3 / 4)

    def test_macro_f1_and_accuracy(self):
        eff = effective_matrix(confusion_matrix(SAMPLES))
        assert macro_f1_of(eff) == pytest.approx((0.571428 + 0.5 + 1.0 + 0.666667) / 4, abs=1e-4)
        assert accuracy_of(eff) == pytest.approx(8 / 12)

    def test_a_class_never_predicted_has_no_precision_and_zero_f1(self):
        eff = np.array([[0, 0, 0, 2], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
        stats = precision_recall_f1(eff)
        assert stats["swipe_left"]["precision"] is None and stats["swipe_left"]["f1"] == 0.0


@pytest.fixture(scope="module")
def scored():
    return score_samples(SAMPLES)


class TestScoreSamples:
    def test_headline_numbers(self, scored):
        assert scored["n"] == 12
        assert scored["accuracy"]["value"] == pytest.approx(8 / 12)
        assert scored["class_support"] == {"swipe_left": 4, "swipe_right": 2,
                                           "circle_clockwise": 2, "no_gesture": 4}  # fmt: skip

    def test_false_activations_count_only_negatives_that_would_trigger(self, scored):
        fa = scored["false_activations"]
        assert (fa["count"], fa["negatives"]) == (1, 4)  # the rejected negative is not one
        assert fa["rate"] == 0.25
        assert fa["by_subtype"]["pointing"] == {"n": 1, "false_activations": 1}
        assert fa["by_subtype"]["pinch"]["false_activations"] == 0

    def test_cross_activations_are_counted_separately(self, scored):
        assert scored["cross_activations"]["count"] == 1  # the left swipe read as a right swipe
        assert scored["cross_activations"]["gestures"] == 8

    def test_rejections_and_true_gestures_lost(self, scored):
        rej = scored["rejection"]
        assert rej["rejected"] == 2 and rej["true_gestures_lost"] == 1
        assert rej["true_gestures"] == 8 and rej["invalid_model_outputs"] == 0
        assert rej["rate"] == pytest.approx(2 / 12)

    def test_invalid_outputs_are_counted(self):
        scored = score_samples([sample("swipe_left", REJECTED, valid=False),
                                sample("no_gesture", "no_gesture")])  # fmt: skip
        assert scored["rejection"]["invalid_model_outputs"] == 1

    def test_intervals_are_present_and_ordered(self, scored):
        for lo_hi in (scored["accuracy"]["ci95"], scored["macro_f1"]["ci95"],
                      scored["false_activations"]["rate_ci95"]):  # fmt: skip
            assert 0.0 <= lo_hi[0] <= lo_hi[1] <= 1.0
        left = scored["per_class"]["swipe_left"]
        assert left["recall_ci"][0] <= left["recall"] <= left["recall_ci"][1]

    def test_scoring_is_deterministic(self):
        assert score_samples(SAMPLES, seed=3) == score_samples(SAMPLES, seed=3)

    def test_flagged_samples_can_be_filtered_out_by_the_caller(self):
        flagged = [sample("swipe_right", "swipe_left", flagged=True), *SAMPLES]
        assert score_samples(flagged)["n"] == 13
        assert score_samples([s for s in flagged if not s.flagged])["n"] == 12

    def test_an_empty_set_does_not_crash(self):
        scored = score_samples([])
        assert scored["n"] == 0 and scored["false_activations"]["rate"] is None


class TestIntervals:
    def test_wilson_known_values(self):
        assert wilson_interval(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-3)
        low, high = wilson_interval(5, 10)
        assert (low, high) == pytest.approx((0.2366, 0.7634), abs=1e-3)
        assert wilson_interval(10, 10)[0] == pytest.approx(0.7225, abs=1e-3)
        assert wilson_interval(0, 0) is None

    def test_exact_poisson_known_values(self):
        low, high = poisson_rate_interval(0, 1.0)
        assert low == 0.0 and high == pytest.approx(3.689, abs=2e-3)
        low, high = poisson_rate_interval(5, 1.0)
        assert (low, high) == pytest.approx((1.623, 11.668), abs=5e-3)
        low, high = poisson_rate_interval(1, 1.0)
        assert (low, high) == pytest.approx((0.0253, 5.572), abs=5e-3)

    def test_zero_events_in_two_minutes_still_allows_a_rate_near_two_per_minute(self):
        """The point of reporting the interval: no events is not 'a rate of zero'."""
        _, high = poisson_rate_interval(0, 2.0)
        assert high == pytest.approx(1.844, abs=2e-3)

    def test_the_rate_interval_scales_with_exposure(self):
        low1, high1 = poisson_rate_interval(3, 1.0)
        low4, high4 = poisson_rate_interval(3, 4.0)
        assert (low4, high4) == pytest.approx((low1 / 4, high1 / 4))

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            poisson_rate_interval(-1, 1.0)
        with pytest.raises(ValueError):
            poisson_rate_interval(1, 0.0)

    def test_bootstrap_is_deterministic_and_sensible(self):
        true_idx = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3])
        pred_idx = np.array([0, 1, 1, 1, 2, 3, 3, 3, 3, 0])
        a = bootstrap_interval(true_idx, pred_idx, macro_f1_of, seed=1, resamples=300)
        b = bootstrap_interval(true_idx, pred_idx, macro_f1_of, seed=1, resamples=300)
        assert a == b and 0.0 <= a[0] <= a[1] <= 1.0
        assert bootstrap_interval(true_idx[:1], pred_idx[:1], macro_f1_of, seed=1) is None


# ------------------------------------------------------------------------ stream analysis
class Flagger:
    """A fake system that flags the windows whose (evaluated) index is in ``flagged``."""

    name = "fake"

    def __init__(self, flagged, label="swipe_left"):
        self.flagged, self.label, self.calls = set(flagged), label, 0

    def predict_features(self, features):
        index = self.calls
        self.calls += 1
        return Prediction(self.label if index in self.flagged else "no_gesture")


def stream_record(seconds=8.0, usable=None):
    base = make_sequence(still_path(), duration=seconds, sequence_id="s-stream-0001",
                         session_id="s-stream", label="no_gesture", hand_size=0.19,
                         usable=usable)  # fmt: skip
    return SequenceRecord("s-stream-0001", "s-stream", "no_gesture", base.t_ms, base.landmarks,
                          base.confidence, base.usable, base.aspect_ratio,
                          subtype="continuous_use", kind="stream")  # fmt: skip


CONFIG = FeatureConfig()


class TestStreamAnalysis:
    def test_windows_cover_the_stream_at_the_stride(self):
        windows = stream_windows(stream_record(8.0), CONFIG)
        assert len(windows) == 65 and windows[0][0] == 0.0
        assert windows[-1][0] == pytest.approx(6.4)
        assert all(w is not None and w.t_ms[0] == 0 for _, w in windows)

    def test_flagged_windows_are_merged_into_events(self):
        events = _events([(1.0, "swipe_left"), (1.1, "swipe_left"), (1.2, "swipe_left"),
                          (5.0, "swipe_left"),  # a new event: more than a second later
                          (5.1, "circle_clockwise"),  # a different class is a different event
                          (5.2, "circle_clockwise")])  # fmt: skip
        assert [(e["label"], e["windows"]) for e in events] == [
            ("swipe_left", 3), ("swipe_left", 1), ("circle_clockwise", 2)]

    def test_events_and_rates(self):
        system = System("fake", Flagger({10, 11, 12, 13, 40, 60, 61, 62}), {})
        result = evaluate_stream(system, [stream_record(8.0)], CONFIG)
        assert result["windows_total"] == result["windows_evaluated"] == 65
        assert result["flagged_windows"] == 8
        assert result["events_any"]["events"] == 3
        assert result["events_persistent"]["events"] == 2  # three or more windows in a row
        assert result["events_any"]["per_minute"] == pytest.approx(3 / (8 / 60))
        low, high = result["events_any"]["ci95_per_minute"]
        assert low < result["events_any"]["per_minute"] < high

    def test_no_events_still_reports_an_upper_bound(self):
        result = evaluate_stream(System("fake", Flagger(set()), {}), [stream_record(8.0)], CONFIG)
        assert result["events_any"]["events"] == 0
        assert result["events_any"]["ci95_per_minute"][0] == 0.0
        upper = result["events_any"]["ci95_per_minute"][1]
        assert upper == pytest.approx(3.689 / (8 / 60), abs=0.05)
        assert "limited evidence" in result["caution"]

    def test_windows_with_unusable_tracking_are_skipped_not_scored(self):
        frames = stream_record(8.0).n_frames
        usable = np.ones(frames, dtype=bool)
        usable[60:130] = False  # about 2.3 s with no trusted hand
        system = System("fake", Flagger(set(range(1000))), {})
        result = evaluate_stream(system, [stream_record(8.0, usable)], CONFIG)
        assert 0 < result["windows_evaluated"] < result["windows_total"]
        assert sum(result["windows_skipped"].values()) == (
            result["windows_total"] - result["windows_evaluated"])
        # reasons are grouped into a few named causes, not one key per percentage
        assert set(result["windows_skipped"]) <= {
            "tracking dropped for too long", "too few trusted frames", "shorter than the window",
            "hand too small", "too few frames", "other"}
        assert result["flagged_windows"] == result["windows_evaluated"]  # only evaluated ones

    def test_the_denominator_is_the_whole_stream(self):
        result = evaluate_stream(System("fake", Flagger({0}), {}), [stream_record(8.0)], CONFIG)
        assert result["stream_seconds"] == 8.0 and "minutes of stream" in result["denominator"]


class TestLatency:
    def test_too_few_samples_are_reported_as_insufficient(self):
        from nocursor.dynamic.splits import SplitItem

        seq = make_sequence(still_path())
        items = [SplitItem(seq, "test", "s", np.zeros((32, 46)), False) for _ in range(5)]
        result = time_system(System("fake", Flagger(set()), {}), items, CONFIG)
        assert result["total"] == "insufficient measurements"
        assert "not camera-to-action latency" in result["note"]

    def test_enough_samples_give_percentiles(self):
        from nocursor.dynamic.splits import SplitItem

        seq = make_sequence(still_path())
        items = [SplitItem(seq, "test", "s", np.zeros((32, 46)), False) for _ in range(40)]
        result = time_system(System("fake", Flagger(set()), {}), items, CONFIG)
        for part in ("feature_extraction", "inference", "total"):
            assert result[part]["n"] == 40 and result[part]["p95"] >= result[part]["p50"] >= 0


# ------------------------------------------------------------------- end to end (synthetic)
def seed_with_flagged_take_in_test():
    """A split seed that puts the session holding the flagged take (s-full-c) in the test split."""
    for seed in range(50):
        order = sorted(FULL_SESSIONS)
        random.Random(seed).shuffle(order)
        if order[0] == "s-full-c":
            return seed
    raise AssertionError("no seed found")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval")
    dataset_dir = build_study(root / "datasets")
    runs_dir = root / "ml_runs"
    split, split_file = freeze_split(dataset_dir, runs_dir, seed=seed_with_flagged_take_in_test())
    config = TrainConfig(allow_synthetic=True)
    primary = run_fit_baseline(split, split_file, dataset_dir, runs_dir, "primary", config,
                               log=lambda _: None)  # fmt: skip
    add_random_gru(primary, split, seed=1)
    sensitivity = run_fit_baseline(split, split_file, dataset_dir, runs_dir, "no-flagged",
                                   TrainConfig(allow_synthetic=True, exclude_flagged=True),
                                   log=lambda _: None)  # fmt: skip
    add_random_gru(sensitivity, split, seed=2)
    return {"root": root, "dataset": dataset_dir, "runs": runs_dir, "split": split,
            "primary": primary, "sensitivity": sensitivity, "split_file": split_file}  # fmt: skip


@pytest.fixture
def run_copy(prepared, tmp_path):
    """A private copy of the datasets and runs, for tests that evaluate or edit files."""
    shutil.copytree(prepared["dataset"], tmp_path / "datasets")
    shutil.copytree(prepared["runs"], tmp_path / "ml_runs")
    return {"dataset": tmp_path / "datasets", "runs": tmp_path / "ml_runs"}


def evaluate(copy, name="primary", **kwargs):
    return evaluate_run(copy["runs"] / name, copy["dataset"], log=lambda _: None, **kwargs)


@pytest.fixture(scope="module")
def result(prepared, tmp_path_factory):
    root = tmp_path_factory.mktemp("evalrun")
    shutil.copytree(prepared["dataset"], root / "datasets")
    shutil.copytree(prepared["runs"], root / "ml_runs")
    copy = {"dataset": root / "datasets", "runs": root / "ml_runs"}
    out = evaluate(copy, sensitivity_run=copy["runs"] / "no-flagged")
    return {"out": out, "copy": copy,
            "metrics": json.loads((out / "metrics.json").read_text(encoding="utf-8")),
            "report": (out / "report.md").read_text(encoding="utf-8")}  # fmt: skip


class TestEvaluateRun:
    def test_both_baselines_and_the_gru_are_scored_on_the_same_takes(self, result, prepared):
        metrics = result["metrics"]
        assert set(metrics["systems"]) == {"rule_defaults", "rule_tuned", "gru"}
        expected = sum(prepared["split"]["counts"]["test"].values())
        assert metrics["test_takes_evaluated"] == expected
        assert {s["all"]["n"] for s in metrics["systems"].values()} == {expected}

    def test_every_headline_metric_is_present(self, result):
        entry = result["metrics"]["systems"]["gru"]["all"]
        for key in ("accuracy", "macro_f1", "per_class", "confusion", "rejection",
                    "false_activations", "cross_activations"):  # fmt: skip
            assert key in entry
        assert entry["confusion"]["columns_predicted"][-1] == REJECTED

    def test_the_test_split_is_evaluated_and_logged(self, result):
        log_file = result["copy"]["runs"] / "primary" / EVAL_LOG
        log = json.loads(log_file.read_text(encoding="utf-8"))
        assert len(log) == 1 and result["metrics"]["evaluation_number_of_this_run"] == 1

    def test_the_stream_is_analysed_for_every_system(self, result):
        stream = result["metrics"]["stream"]
        assert set(stream) == {"rule_defaults", "rule_tuned", "gru"}
        assert all(r["stream_seconds"] == 8.0 and "caution" in r for r in stream.values())

    def test_the_flagged_take_in_the_test_session_is_reported_with_and_without(self, result):
        gru = result["metrics"]["systems"]["gru"]
        assert len(gru["flagged_takes"]) == 1
        assert gru["without_flagged"]["n"] == gru["all"]["n"] - 1

    def test_the_sensitivity_run_is_compared(self, result):
        sensitivity = result["metrics"]["sensitivity"]
        assert sensitivity["retrained_run"] == "no-flagged"
        assert set(sensitivity["systems"]) == {"rule_defaults", "rule_tuned", "gru"}
        assert "Retrained without the flagged takes" in result["report"]

    def test_the_seed_stability_table_is_absent_without_seed_models(self, result):
        assert result["metrics"]["seed_stability"] is None

    def test_provenance_is_stated_and_synthetic_data_is_labelled(self, result):
        assert "SYNTHETIC" in result["metrics"]["data_provenance"]
        assert "SYNTHETIC" in result["report"].splitlines()[0]

    def test_the_report_states_its_limits_and_makes_no_overclaim(self, result):
        report = result["report"]
        for phrase in ("Read this first", "one held-out session", "Limitations",
                       "limited evidence", "One participant", "not proof of reliable",
                       "Nothing here is connected to the mouse"):  # fmt: skip
            assert phrase in report, phrase
        for banned in ("state of the art", "outperforms", "significantly better", "proves"):
            assert banned not in report.lower()

    def test_the_report_names_exclusions_flags_and_assignments(self, result, prepared):
        report = result["report"]
        assert "Excluded from model inputs" in report and "Flagged takes" in report
        for session in prepared["split"]["assignments"]["test"]:
            assert session in report
        assert "reserved stream" in report

    def test_the_report_has_the_confusion_matrix_and_latency(self, result):
        report = result["report"]
        assert "true \\ predicted" in report and "rejected" in report
        assert "Latency (software timing" in report and "not camera-to-action" in result[
            "metrics"]["systems"]["gru"]["latency"]["note"]

    def test_the_machine_readable_metrics_carry_the_dataset_provenance(self, result, prepared):
        dataset = result["metrics"]["dataset"]
        assert dataset["manifest_hash"] == prepared["split"]["dataset_manifest_hash"]
        assert dataset["assignments"] == prepared["split"]["assignments"]
        assert len(dataset["excluded_takes"]) == 2 and len(dataset["flagged_takes"]) == 1
        assert result["metrics"]["limitations"]


class TestEvaluationGuards:
    def test_evaluations_are_counted(self, run_copy):
        first = evaluate(run_copy, include_stream=False)
        second = evaluate(run_copy, include_stream=False)
        assert first != second
        numbers = [json.loads((d / "metrics.json").read_text(encoding="utf-8"))
                   ["evaluation_number_of_this_run"] for d in sorted((first.parent).iterdir())]
        assert numbers == [1, 2]

    def test_a_stream_can_be_skipped(self, run_copy):
        out = evaluate(run_copy, include_stream=False)
        assert json.loads((out / "metrics.json").read_text(encoding="utf-8"))["stream"] is None
        assert "Not run." in (out / "report.md").read_text(encoding="utf-8")

    def test_a_changed_dataset_file_refuses_evaluation(self, run_copy):
        path = next(run_copy["dataset"].glob("s-full-a.jsonl"))
        path.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(EvaluationError, match="changed since"):
            evaluate(run_copy, include_stream=False)

    def test_a_model_trained_on_another_split_is_refused(self, run_copy):
        path = run_copy["runs"] / "primary" / META_FILE
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["split_id"] = "split-somethingelse"
        path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(EvaluationError, match="different split"):
            evaluate(run_copy, include_stream=False)

    def test_a_sensitivity_run_from_another_split_is_refused(self, run_copy):
        path = run_copy["runs"] / "no-flagged" / "split.json"
        split = json.loads(path.read_text(encoding="utf-8"))
        split["split_id"] = "split-other"
        path.write_text(json.dumps(split), encoding="utf-8")
        with pytest.raises(EvaluationError, match="different split"):
            evaluate(run_copy, sensitivity_run=run_copy["runs"] / "no-flagged",
                     include_stream=False)  # fmt: skip

    def test_a_run_without_a_model_still_scores_the_baselines(self, run_copy):
        for name in ("model.npz", META_FILE):
            (run_copy["runs"] / "primary" / name).unlink()
        out = evaluate(run_copy, include_stream=False)
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert set(metrics["systems"]) == {"rule_defaults", "rule_tuned"}

    def test_a_run_with_nothing_to_evaluate(self, run_copy):
        for name in ("model.npz", META_FILE, "baseline.json"):
            (run_copy["runs"] / "primary" / name).unlink()
        with pytest.raises(EvaluationError, match="neither baseline.json nor a model"):
            evaluate(run_copy, include_stream=False)

    def test_evaluation_does_not_alter_the_trained_artifacts(self, run_copy):
        run = run_copy["runs"] / "primary"
        names = ("model.npz", META_FILE, "baseline.json", "split.json")
        before = {n: (run / n).read_bytes() for n in names}
        evaluate(run_copy, include_stream=False)
        assert {n: (run / n).read_bytes() for n in names} == before

    def test_seed_models_are_scored_for_stability_only(self, run_copy, prepared):
        run = run_copy["runs"] / "primary"
        add_random_gru(run / "seeds" / "seed-1", prepared["split"], seed=11)
        add_random_gru(run / "seeds" / "seed-2", prepared["split"], seed=12)
        out = evaluate(run_copy, include_stream=False)
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        rows = metrics["seed_stability"]
        assert len(rows) == 3 and sum(r["is_final_model"] for r in rows) == 1
        report = (out / "report.md").read_text(encoding="utf-8")
        assert "none of this was used to choose the final model" in report


class TestTestSplitIsolationInTheRun:
    def test_fitting_the_baseline_never_parses_test_or_stream_data(self, prepared, tmp_path,
                                                                    monkeypatch):  # fmt: skip
        from nocursor.dynamic import splits as splits_module

        opened = []
        real = splits_module.load_session
        monkeypatch.setattr(splits_module, "load_session",
                            lambda path, **kw: opened.append(path.stem) or real(path, **kw))
        run_fit_baseline(prepared["split"], prepared["split_file"], prepared["dataset"],
                         tmp_path / "ml_runs", "isolated", TrainConfig(allow_synthetic=True),
                         log=lambda _: None)  # fmt: skip
        forbidden = {*prepared["split"]["assignments"]["test"], "s-stream-a"}
        assert opened and not forbidden & set(opened)

    def test_the_baseline_was_tuned_on_train_and_validation_counts_only(self, prepared):
        payload = json.loads((prepared["primary"] / "baseline.json").read_text(encoding="utf-8"))
        assert payload["tuned_on"]["splits"] == ["train", "validation"]
        split = prepared["split"]
        expected = {c: split["counts"]["train"][c] + split["counts"]["validation"][c]
                    for c in split["counts"]["train"]}  # fmt: skip
        assert payload["tuned_on"]["class_counts"] == expected

    def test_a_run_name_cannot_be_reused(self, prepared):
        from nocursor.dynamic.splits import SplitError

        with pytest.raises(SplitError, match="already exists"):
            run_fit_baseline(prepared["split"], prepared["split_file"], prepared["dataset"],
                             prepared["runs"], "primary", TrainConfig(allow_synthetic=True),
                             log=lambda _: None)  # fmt: skip

    def test_synthetic_sessions_are_refused_unless_explicitly_allowed(self, prepared, tmp_path):
        from nocursor.dynamic.train import TrainingDataError

        with pytest.raises(TrainingDataError, match="synthetic"):
            run_fit_baseline(prepared["split"], prepared["split_file"], prepared["dataset"],
                             tmp_path / "ml_runs", "x", TrainConfig(), log=lambda _: None)

    def test_the_flagged_option_is_recorded_when_that_take_is_held_out(self, prepared):
        """Here the flagged take is in the test session, so excluding flagged takes from training
        and validation removes nothing from the tuning data; the option is still recorded."""
        primary = json.loads((prepared["primary"] / "baseline.json").read_text(encoding="utf-8"))
        other = json.loads((prepared["sensitivity"] / "baseline.json").read_text(encoding="utf-8"))
        assert other["tuned_on"]["flagged_takes_excluded"] is True
        assert primary["tuned_on"]["flagged_takes_excluded"] is False
        assert prepared["split"]["flagged"][0]["split"] == "test"
        assert primary["tuned_on"]["class_counts"] == other["tuned_on"]["class_counts"]

    def test_loading_for_training_matches_what_the_manifest_counts(self, prepared):
        items = load_split_items(prepared["split"], prepared["dataset"], ["train", "validation"])
        counts = {}
        for item in items:
            counts[item.sequence.label] = counts.get(item.sequence.label, 0) + 1
        split = prepared["split"]
        assert counts == {c: split["counts"]["train"][c] + split["counts"]["validation"][c]
                          for c in counts}  # fmt: skip
