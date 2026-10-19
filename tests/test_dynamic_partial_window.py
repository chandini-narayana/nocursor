"""Partial-window training samples and the separately named candidate model.

All data is SYNTHETIC geometry; these tests check software behaviour, not recognition accuracy.
Training tests need PyTorch and are skipped where it is not installed (the ordinary environment).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from dynamic_helpers import circle_path, make_sequence, still_path, swipe_path
from ml_fixtures import add_random_gru, build_study, freeze_split
from nocursor.dynamic.cli import main
from nocursor.dynamic.features import FeatureConfig, motion_summary
from nocursor.dynamic.model_io import load_classifier
from nocursor.dynamic.partial_window import (
    ALIGNED_POSITIVE_SHIFTS_S,
    CANDIDATES_DIR,
    HOLD_S,
    KIND_NEGATIVE,
    KIND_POSITIVE,
    PARTIAL_NEGATIVE_SHIFTS_S,
    build_partial_samples,
    counts_by_kind,
    shifted_window,
)
from nocursor.dynamic.splits import SplitItem
from nocursor.dynamic.train import TrainConfig, run_fit_baseline

CONFIG = FeatureConfig()


def item(take, split="train"):
    return SplitItem(take, split, take.session_id, np.zeros((32, 46)), False)


def swipe(sid="s-a-0001", amplitude=0.2):
    return make_sequence(swipe_path(+1, amplitude), label="swipe_right", sequence_id=sid,
                         session_id="s-a")  # fmt: skip


class TestShiftedWindow:
    def test_an_unshifted_window_starts_where_the_take_starts(self):
        take = swipe()
        window = shifted_window(take, 0.0, CONFIG)
        assert np.allclose(window.landmarks[0], take.landmarks[0])
        assert window.t_ms[0] == 0 and window.duration_s == pytest.approx(1.6, abs=0.05)

    def test_a_late_window_has_lost_the_start_of_the_gesture(self):
        take = swipe()
        window = shifted_window(take, 0.8, CONFIG)
        middle = take.landmarks[int(round(0.8 * 30))]
        assert np.allclose(window.landmarks[0], middle, atol=2e-3)

    def test_an_early_window_starts_in_the_held_rest_before_the_gesture(self):
        take = swipe()
        window = shifted_window(take, -0.8, CONFIG)
        assert np.allclose(window.landmarks[0], take.landmarks[0])  # held first frame
        assert np.allclose(window.landmarks[5], take.landmarks[0])  # still resting

    def test_every_window_has_the_length_the_model_expects(self):
        for shift in (*PARTIAL_NEGATIVE_SHIFTS_S, *ALIGNED_POSITIVE_SHIFTS_S):
            assert shifted_window(swipe(), shift, CONFIG).duration_s == pytest.approx(
                1.6, abs=0.08)  # fmt: skip

    def test_the_hold_is_long_enough_for_the_largest_shift(self):
        assert max(abs(s) for s in PARTIAL_NEGATIVE_SHIFTS_S) <= HOLD_S


class TestBuildSamples:
    def build(self, takes):
        return build_partial_samples([item(t) for t in takes], CONFIG)

    def test_each_gesture_take_gives_its_partial_and_aligned_windows(self):
        samples = self.build([swipe()])
        assert len(samples) == len(PARTIAL_NEGATIVE_SHIFTS_S) + len(ALIGNED_POSITIVE_SHIFTS_S)
        assert samples.skipped == 0
        assert samples.pts.shape == (6, 32, 21, 2) and samples.features.shape == (6, 32, 46)

    def test_partial_windows_are_negatives_and_aligned_ones_keep_the_label(self):
        samples = self.build([swipe()])
        by_kind = {k: {lab for lab, kind in zip(samples.labels, samples.kinds, strict=True)
                       if kind == k} for k in (KIND_NEGATIVE, KIND_POSITIVE)}  # fmt: skip
        assert by_kind[KIND_NEGATIVE] == {"no_gesture"}
        assert by_kind[KIND_POSITIVE] == {"swipe_right"}
        assert counts_by_kind(samples) == {KIND_NEGATIVE: {"no_gesture": 4},
                                           KIND_POSITIVE: {"swipe_right": 2}}  # fmt: skip

    def test_only_gesture_takes_are_cut(self):
        idle = make_sequence(still_path(), label="no_gesture", subtype="idle",
                             sequence_id="s-a-0002", session_id="s-a")  # fmt: skip
        assert len(self.build([idle])) == 0
        assert len(self.build([swipe(), idle])) == 6

    def test_samples_remember_their_source_and_shift(self):
        samples = self.build([swipe("s-a-0007")])
        assert set(samples.sources) == {"s-a-0007"}
        assert sorted(samples.shifts) == sorted([*PARTIAL_NEGATIVE_SHIFTS_S,
                                                 *ALIGNED_POSITIVE_SHIFTS_S])  # fmt: skip

    def test_partial_windows_hold_less_of_the_gesture_than_aligned_ones(self):
        samples = self.build([swipe()])
        path = {(k, s): motion_summary(f).path_length for k, s, f in zip(
            samples.kinds, samples.shifts, samples.features, strict=True)}  # fmt: skip
        full = min(path[(KIND_POSITIVE, s)] for s in ALIGNED_POSITIVE_SHIFTS_S)
        for shift in PARTIAL_NEGATIVE_SHIFTS_S:
            assert path[(KIND_NEGATIVE, shift)] < 0.8 * full

    def test_circles_are_cut_too_and_stay_circles_when_aligned(self):
        circle = make_sequence(circle_path(True), label="circle_clockwise", sequence_id="s-a-0003",
                               session_id="s-a")  # fmt: skip
        samples = self.build([circle])
        assert counts_by_kind(samples)[KIND_POSITIVE] == {"circle_clockwise": 2}

    def test_the_result_is_deterministic(self):
        a, b = self.build([swipe()]), self.build([swipe()])
        assert np.array_equal(a.features, b.features) and a.labels == b.labels

    def test_an_empty_list_gives_empty_arrays(self):
        samples = self.build([])
        assert len(samples) == 0 and samples.pts.shape[0] == 0


# ---------------------------------------------------------------------------- training
HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="training the candidate needs PyTorch")


def tree_digest(directory: Path) -> dict[str, str]:
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}  # fmt: skip


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("cand")
    dataset_dir = build_study(root / "datasets", takes=10)
    runs_dir = root / "ml_runs"
    split, split_file = freeze_split(dataset_dir, runs_dir)
    primary = run_fit_baseline(split, split_file, dataset_dir, runs_dir, "primary",
                               TrainConfig(allow_synthetic=True), log=lambda _: None)  # fmt: skip
    add_random_gru(primary, split, seed=1)
    return {"root": root, "dataset": dataset_dir, "runs": runs_dir, "split": split,
            "split_file": split_file, "primary": primary}  # fmt: skip


TINY = {"hidden_sizes": (4,), "weight_decays": (0.0,), "seeds": (0,), "max_epochs": 3,
        "patience": 3, "allow_synthetic": True}  # fmt: skip


@pytest.fixture(scope="module")
def trained(prepared, tmp_path_factory):
    from nocursor.dynamic import splits
    from nocursor.dynamic.partial_window import run_partial_train

    root = tmp_path_factory.mktemp("trained")
    shutil.copytree(prepared["dataset"], root / "datasets")
    shutil.copytree(prepared["runs"], root / "ml_runs")
    before = tree_digest(root / "ml_runs" / "primary")
    split_before = tree_digest(root / "ml_runs" / "splits")
    requested = []
    original = splits.load_split_items
    mp = pytest.MonkeyPatch()
    mp.setattr(splits, "load_split_items",
               lambda split, d, names, **kw: (requested.append(list(names)),
                                              original(split, d, names, **kw))[1])  # fmt: skip
    split_file = next((root / "ml_runs" / "splits").rglob("split.json"))
    split = json.loads(split_file.read_text("utf-8"))
    target = run_partial_train(split, split_file, root / "datasets", root / "ml_runs",
                               "test-cand", TrainConfig(**TINY),
                               parent_run=root / "ml_runs" / "primary",
                               log=lambda _: None)
    mp.undo()
    return {"root": root, "target": target, "before": before, "requested": requested,
            "split_before": split_before}  # fmt: skip



@needs_torch
class TestCandidateTraining:
    def test_it_writes_a_loadable_model_under_the_candidates_folder(self, trained):
        target = trained["target"]
        assert target.parent == trained["root"] / "ml_runs" / Path(*CANDIDATES_DIR)
        classifier = load_classifier(target)
        assert classifier.meta["role_in_run"] == "candidate"
        assert (target / "split.json").is_file() and (target / "train_log.json").is_file()

    def test_it_records_where_it_came_from_and_what_it_cannot_claim(self, trained, prepared):
        meta = json.loads((trained["target"] / "model.json").read_text("utf-8"))
        info = meta["candidate"]
        parent = json.loads((prepared["primary"] / "model.json").read_text("utf-8"))
        assert info["name"] == "test-cand" and info["parent_run"] == "primary"
        assert info["parent_weights_sha256"] == parent["weights_sha256"]
        assert meta["weights_sha256"] != parent["weights_sha256"]
        text = " ".join(info["disclosures"])
        assert "validation takes" in text and "early stopping" in text
        assert "test session" in text and "not opened" in text

    def test_it_counts_the_partial_samples_it_used(self, trained):
        counts = json.loads((trained["target"] / "model.json").read_text("utf-8"))[
            "candidate"]["sample_counts"]  # fmt: skip
        assert counts["train_partial_samples"]["partial_negative"]["no_gesture"] > 0
        assert counts["training_samples_total"] > sum(counts["train_takes"].values())

    def test_only_training_and_validation_data_were_requested(self, trained):
        assert trained["requested"] and all(
            set(names) <= {"train", "validation"} for names in trained["requested"])  # fmt: skip

    def test_the_primary_run_and_the_frozen_split_are_untouched(self, trained):
        root = trained["root"]
        assert tree_digest(root / "ml_runs" / "primary") == trained["before"]
        assert tree_digest(root / "ml_runs" / "splits") == trained["split_before"]
        assert not (root / "ml_runs" / "primary" / "eval_log.json").exists()

    def test_a_candidate_is_never_overwritten(self, trained):
        from nocursor.dynamic.partial_window import run_partial_train
        from nocursor.dynamic.splits import SplitError

        root = trained["root"]
        split_file = next((root / "ml_runs" / "splits").rglob("split.json"))
        split = json.loads(split_file.read_text("utf-8"))
        with pytest.raises(SplitError, match="already exists"):
            run_partial_train(split, split_file, root / "datasets", root / "ml_runs", "test-cand",
                              TrainConfig(**TINY), log=lambda _: None)  # fmt: skip

    def test_the_cli_trains_a_second_candidate(self, trained):
        root = trained["root"]
        code = main(["stream-candidate", "--dataset-dir", str(root / "datasets"), "--runs-dir",
                     str(root / "ml_runs"), "--run", "primary", "--name", "cli-cand",
                     "--seeds", "1", "--max-epochs", "2", "--hidden-sizes", "4",
                     "--allow-synthetic"])  # fmt: skip
        assert code == 0
        assert (root / "ml_runs" / Path(*CANDIDATES_DIR) / "cli-cand" / "model.json").is_file()
