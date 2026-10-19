"""GRU training, early stopping, determinism and NumPy/PyTorch parity.

These tests need PyTorch, which is an optional, training-only dependency. In an environment
without it the whole module is skipped (and pytest reports the skip); run it in the separate ML
environment described in docs/DYNAMIC_GESTURES.md. The data is SYNTHETIC and the models are tiny:
the tests check the training machinery, not how well anything generalises.
"""

import json
import shutil

import numpy as np
import pytest

from ml_fixtures import build_study, freeze_split

torch = pytest.importorskip("torch")

from nocursor.dynamic import splits as splits_module  # noqa: E402
from nocursor.dynamic.cli import main  # noqa: E402
from nocursor.dynamic.features import FEATURE_DIM  # noqa: E402
from nocursor.dynamic.model_io import META_FILE, MODEL_FILE, load_classifier  # noqa: E402
from nocursor.dynamic.schema import CLASS_NAMES  # noqa: E402
from nocursor.dynamic.splits import load_split_items  # noqa: E402
from nocursor.dynamic.train import (  # noqa: E402
    TRAIN_LOG,
    TrainConfig,
    _build_model,
    run_train,
)

SMALL = {"hidden_sizes": (8,), "weight_decays": (0.0,), "seeds": (0, 1), "max_epochs": 6,
         "patience": 3, "allow_synthetic": True}  # fmt: skip


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("torch")
    dataset_dir = build_study(root / "datasets")
    split, split_file = freeze_split(dataset_dir, root / "ml_runs")
    return {"root": root, "dataset": dataset_dir, "runs": root / "ml_runs", "split": split,
            "split_file": split_file}  # fmt: skip


def train(study, name, **overrides):
    config = TrainConfig(**{**SMALL, **overrides})
    return run_train(study["split"], study["split_file"], study["dataset"], study["runs"], name,
                     config, log=lambda _: None)  # fmt: skip


@pytest.fixture(scope="module")
def trained(study):
    return train(study, "trained")


class TestArtifacts:
    def test_the_run_folder_layout(self, trained):
        for name in (MODEL_FILE, META_FILE, TRAIN_LOG, "baseline.json", "split.json"):
            assert (trained / name).is_file(), name
        assert len(list((trained / "seeds").glob("seed-*"))) == 1  # the non-final seed

    def test_the_model_loads_without_pytorch_machinery(self, trained):
        model = load_classifier(trained)
        prediction = model.predict_features(np.zeros((32, FEATURE_DIM)))
        assert prediction.valid

    def test_the_metadata_records_everything_needed_to_trust_and_reproduce_it(self, trained, study):
        meta = json.loads((trained / META_FILE).read_text(encoding="utf-8"))
        assert meta["class_names"] == list(CLASS_NAMES)
        assert len(meta["weights_sha256"]) == 64 and len(meta["feature_schema_hash"]) == 64
        assert meta["split_id"] == study["split"]["split_id"]
        assert meta["dataset_manifest_hash"] == study["split"]["dataset_manifest_hash"]
        assert meta["seed"] in (0, 1) and meta["split_seed"] == 0
        assert 0.0 < meta["rejection"]["threshold"] <= 1.0
        assert meta["libraries"]["torch"] == torch.__version__
        assert meta["libraries"]["numpy"] == np.__version__
        assert meta["training"]["selected_epoch"] >= 1
        assert "SYNTHETIC" in meta["data_provenance"]
        assert meta["training"]["class_counts"]["train"]["swipe_left"] > 0

    def test_the_run_keeps_a_copy_of_the_frozen_split(self, trained, study):
        copy = json.loads((trained / "split.json").read_text(encoding="utf-8"))
        assert copy == study["split"]

    def test_the_training_log_reports_the_whole_procedure(self, trained):
        log = json.loads((trained / TRAIN_LOG).read_text(encoding="utf-8"))
        assert log["class_counts"]["train"] and log["class_counts"]["validation"]
        assert len(log["excluded_takes"]) == 2 and len(log["flagged_takes"]) == 1
        assert log["flagged_takes_excluded_from_this_run"] is False
        assert log["selection"][0]["chosen"] is True and len(log["runs"]) == 2
        run = log["runs"][0]
        assert len(run["history"]) == run["epochs_run"]
        assert {"epoch", "train_loss", "val_loss", "val_macro_f1"} <= set(run["history"][0])
        assert 1 <= run["selected_epoch"] <= run["epochs_run"]
        assert run["history"][run["selected_epoch"] - 1]["val_loss"] == pytest.approx(
            run["best_val_loss"])  # the epoch that was restored is the one with the best loss
        assert log["selected"]["rejection"]["method"]
        assert all("_state" not in r and "_val_logits" not in r for r in log["runs"])


class TestParity:
    def test_the_numpy_model_matches_pytorch_on_the_validation_takes(self, trained):
        log = json.loads((trained / TRAIN_LOG).read_text(encoding="utf-8"))
        assert log["numpy_vs_torch_max_abs_logit_difference_validation"] < 1e-4

    @pytest.mark.parametrize("hidden", [1, 5, 16])
    def test_logits_agree_on_random_inputs_for_several_sizes(self, tmp_path, hidden):
        from ml_fixtures import model_metadata
        from nocursor.dynamic.model_io import save_artifact

        torch.manual_seed(hidden)
        net = _build_model(torch, hidden, 0.0).eval()
        state = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}
        mean = np.random.default_rng(0).normal(0, 0.3, FEATURE_DIM).astype(np.float32)
        std = np.random.default_rng(1).uniform(0.5, 2.0, FEATURE_DIM).astype(np.float32)
        arrays = {
            "gru.weight_ih": state["gru.weight_ih_l0"], "gru.weight_hh": state["gru.weight_hh_l0"],
            "gru.bias_ih": state["gru.bias_ih_l0"], "gru.bias_hh": state["gru.bias_hh_l0"],
            "fc.weight": state["fc.weight"], "fc.bias": state["fc.bias"],
            "norm.mean": mean, "norm.std": std,
        }  # fmt: skip
        save_artifact(tmp_path / "m", arrays, model_metadata(hidden=hidden))
        model = load_classifier(tmp_path / "m")
        rng = np.random.default_rng(5)
        for _ in range(20):
            features = rng.normal(0, 1.5, (32, FEATURE_DIM))
            standardised = torch.from_numpy(((features - mean) / std).astype(np.float32))[None]
            with torch.no_grad():
                expected = net(standardised)[0].numpy()
            assert np.allclose(model.logits(features), expected, atol=1e-4), hidden


class TestReproducibilityAndSelection:
    def test_the_same_seed_gives_identical_weights(self, study, trained):
        again = train(study, "trained-again")
        first = json.loads((trained / META_FILE).read_text(encoding="utf-8"))
        second = json.loads((again / META_FILE).read_text(encoding="utf-8"))
        assert first["weights_sha256"] == second["weights_sha256"]
        assert first["rejection"]["threshold"] == second["rejection"]["threshold"]
        log_a = json.loads((trained / TRAIN_LOG).read_text(encoding="utf-8"))
        log_b = json.loads((again / TRAIN_LOG).read_text(encoding="utf-8"))
        assert log_a["runs"] == log_b["runs"]  # every loss, in every epoch, of every seed

    def test_different_seeds_give_different_models(self, trained):
        log = json.loads((trained / TRAIN_LOG).read_text(encoding="utf-8"))
        a, b = log["runs"]
        assert a["history"][0]["train_loss"] != b["history"][0]["train_loss"]

    def test_early_stopping_stops_and_restores_the_best_epoch(self, study):
        run = train(study, "early", max_epochs=60, patience=1, min_delta=10.0, seeds=(0,))
        log = json.loads((run / TRAIN_LOG).read_text(encoding="utf-8"))
        only = log["runs"][0]
        assert only["epochs_run"] == 2 and only["selected_epoch"] == 1  # no epoch beats epoch 1

    def test_training_really_reduces_the_training_loss(self, study):
        run = train(study, "learns", max_epochs=25, patience=25, seeds=(0,), learning_rate=1e-2)
        history = json.loads((run / TRAIN_LOG).read_text(encoding="utf-8"))["runs"][0]["history"]
        assert history[-1]["train_loss"] < 0.8 * history[0]["train_loss"]

    def test_the_best_setting_is_chosen_by_validation_and_listed(self, study):
        run = train(study, "grid", hidden_sizes=(4, 8), seeds=(0, 1), max_epochs=4)
        log = json.loads((run / TRAIN_LOG).read_text(encoding="utf-8"))
        assert len(log["selection"]) == 2 and sum(row["chosen"] for row in log["selection"]) == 1
        best = max(log["selection"], key=lambda r: r["mean_val_macro_f1"])
        chosen = log["selection"][0]  # the table is sorted best first
        assert chosen["chosen"] and chosen["mean_val_macro_f1"] == best["mean_val_macro_f1"]

    def test_every_seed_of_the_chosen_setting_is_saved_for_the_stability_report(self, study):
        run = train(study, "seeds", seeds=(0, 1, 2), max_epochs=3)
        assert len(list((run / "seeds").glob("seed-*"))) == 2  # three seeds, one is the final model


class TestTestSplitStaysClosed:
    def test_training_never_parses_test_or_stream_sessions(self, study, monkeypatch):
        opened = []
        real = splits_module.load_session
        monkeypatch.setattr(splits_module, "load_session",
                            lambda path, **kw: opened.append(path.stem) or real(path, **kw))
        train(study, "closed", seeds=(0,), max_epochs=2)
        forbidden = {*study["split"]["assignments"]["test"], "s-stream-a"}
        assert opened and not forbidden & set(opened)

    def test_the_test_split_cannot_be_requested_by_the_training_code_path(self, study):
        from nocursor.dynamic.splits import SplitError

        with pytest.raises(SplitError, match="reserved for evaluation"):
            load_split_items(study["split"], study["dataset"], ["train", "test"])

    def test_the_standardisation_uses_training_takes_only(self, trained, study):
        meta_dir = trained
        with np.load(meta_dir / MODEL_FILE, allow_pickle=False) as data:
            mean = data["norm.mean"].astype(np.float64)
        train_items = load_split_items(study["split"], study["dataset"], ["train"])
        expected = np.stack([i.features for i in train_items]).reshape(-1, FEATURE_DIM).mean(axis=0)
        assert np.allclose(mean, expected, atol=1e-5)


class TestFlaggedTakes:
    def test_the_sensitivity_run_records_that_flagged_takes_were_left_out(self, study):
        run = train(study, "no-flagged", exclude_flagged=True, seeds=(0,), max_epochs=2)
        log = json.loads((run / TRAIN_LOG).read_text(encoding="utf-8"))
        meta = json.loads((run / META_FILE).read_text(encoding="utf-8"))
        assert log["flagged_takes_excluded_from_this_run"] is True
        assert meta["training"]["flagged_takes_excluded"] is True


class TestCommandLine:
    def test_train_and_evaluate_through_the_cli(self, study, tmp_path, capsys):
        shutil.copytree(study["dataset"], tmp_path / "datasets")
        args = ["--dataset-dir", str(tmp_path / "datasets"), "--runs-dir", str(tmp_path / "runs")]
        assert main(["split", *args]) == 0
        sid = next((tmp_path / "runs" / "splits").iterdir()).name
        assert main(["train", "--split", sid, "--run-name", "cli", "--seeds", "2", "--max-epochs",
                     "3", "--hidden-sizes", "4", "--allow-synthetic", *args]) == 0  # fmt: skip
        assert main(["evaluate", "--run", "cli", "--no-stream", *args]) == 0
        out = capsys.readouterr().out
        assert "Run saved" in out and "Report:" in out
        report = (next((tmp_path / "runs" / "cli" / "eval").iterdir()) / "report.md").read_text(
            encoding="utf-8")
        assert "SYNTHETIC" in report and "**gru**" in report

    def test_the_trained_run_can_be_evaluated_with_the_stream(self, trained, study, tmp_path):
        from nocursor.dynamic.evaluate import evaluate_run

        runs = tmp_path / "runs"
        shutil.copytree(study["runs"], runs)
        out = evaluate_run(runs / "trained", study["dataset"], log=lambda _: None)
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert set(metrics["systems"]) == {"rule_defaults", "rule_tuned", "gru"}
        assert metrics["seed_stability"] and len(metrics["seed_stability"]) == 2
        assert metrics["stream"]["gru"]["stream_seconds"] == 8.0
