"""The split / fit-baseline / evaluate commands end to end (SYNTHETIC data, no PyTorch needed)."""

import importlib.util
import json
import shutil

import pytest

from ml_fixtures import build_study
from nocursor.dynamic.cli import main

HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    return build_study(tmp_path_factory.mktemp("cli") / "datasets")


@pytest.fixture
def dirs(dataset_dir, tmp_path):
    shutil.copytree(dataset_dir, tmp_path / "datasets")
    flags = ["--dataset-dir", str(tmp_path / "datasets"), "--runs-dir", str(tmp_path / "runs")]
    return flags, tmp_path


def split_id(tmp_path):
    return next((tmp_path / "runs" / "splits").iterdir()).name


class TestSplitCommand:
    def test_it_freezes_and_reports_the_split(self, dirs, capsys):
        args, tmp = dirs
        assert main(["split", *args]) == 0
        out = capsys.readouterr().out
        assert "written and frozen" in out and "seed 0" in out
        assert "Verified:" in out and "all four classes in every split" in out
        for role in ("train", "pilot_train", "validation", "test", "reserved_stream"):
            assert role in out
        manifest = json.loads((tmp / "runs" / "splits" / split_id(tmp) / "split.json")
                              .read_text(encoding="utf-8"))
        assert manifest["seed"] == 0
        assert manifest["assignments"]["reserved_stream"] == ["s-stream-a"]

    def test_running_it_again_leaves_the_frozen_split_alone(self, dirs, capsys):
        args, tmp = dirs
        main(["split", *args])
        path = tmp / "runs" / "splits" / split_id(tmp) / "split.json"
        before = path.read_bytes()
        capsys.readouterr()
        assert main(["split", *args]) == 0
        assert "already frozen" in capsys.readouterr().out and path.read_bytes() == before

    def test_the_seed_is_recorded(self, dirs):
        args, tmp = dirs
        assert main(["split", "--seed", "3", *args]) == 0
        manifest = json.loads((tmp / "runs" / "splits" / split_id(tmp) / "split.json")
                              .read_text(encoding="utf-8"))
        assert manifest["seed"] == 3

    def test_too_few_sessions_is_a_clear_error(self, tmp_path, capsys):
        (tmp_path / "datasets").mkdir()
        code = main(["split", "--dataset-dir", str(tmp_path / "datasets"),
                     "--runs-dir", str(tmp_path / "runs")])  # fmt: skip
        assert code == 1 and "no sessions" in capsys.readouterr().err

    def test_a_missing_dataset_is_a_clear_error(self, tmp_path, capsys):
        code = main(["split", "--dataset-dir", str(tmp_path / "nope"),
                     "--runs-dir", str(tmp_path / "runs")])  # fmt: skip
        assert code == 1 and "not found" in capsys.readouterr().err


class TestFitAndEvaluate:
    def test_the_full_baseline_workflow(self, dirs, capsys):
        args, tmp = dirs
        main(["split", *args])
        sid = split_id(tmp)
        assert main(["fit-baseline", "--split", sid, "--run-name", "r1", "--allow-synthetic",
                     *args]) == 0  # fmt: skip
        assert (tmp / "runs" / "r1" / "baseline.json").is_file()
        assert (tmp / "runs" / "r1" / "split.json").is_file()
        capsys.readouterr()
        assert main(["evaluate", "--run", "r1", *args]) == 0
        report = (next((tmp / "runs" / "r1" / "eval").iterdir()) / "report.md")
        assert "Dynamic gesture evaluation: r1 **SYNTHETIC" in report.read_text(encoding="utf-8")

    def test_synthetic_data_needs_the_explicit_flag(self, dirs, capsys):
        args, tmp = dirs
        main(["split", *args])
        code = main(["fit-baseline", "--split", split_id(tmp), "--run-name", "r1", *args])
        assert code == 1 and "synthetic" in capsys.readouterr().err
        assert not (tmp / "runs" / "r1" / "baseline.json").exists()

    def test_the_split_may_be_given_as_a_file(self, dirs):
        args, tmp = dirs
        main(["split", *args])
        path = tmp / "runs" / "splits" / split_id(tmp) / "split.json"
        assert main(["fit-baseline", "--split", str(path), "--run-name", "r2", "--allow-synthetic",
                     *args]) == 0  # fmt: skip

    def test_an_unknown_split_is_a_clear_error(self, dirs, capsys):
        args, _ = dirs
        assert main(["fit-baseline", "--split", "split-nope", *args]) == 1
        assert "split not found" in capsys.readouterr().err

    def test_an_unknown_run_is_a_clear_error(self, dirs, capsys):
        args, _ = dirs
        assert main(["evaluate", "--run", "nope", *args]) == 1
        assert "run not found" in capsys.readouterr().err

    def test_the_sensitivity_flag_reaches_the_second_run(self, dirs):
        args, tmp = dirs
        main(["split", *args])
        sid = split_id(tmp)
        main(["fit-baseline", "--split", sid, "--run-name", "a", "--allow-synthetic", *args])
        main(["fit-baseline", "--split", sid, "--run-name", "b", "--allow-synthetic",
              "--exclude-flagged", *args])  # fmt: skip
        payload = json.loads((tmp / "runs" / "b" / "baseline.json").read_text(encoding="utf-8"))
        assert payload["tuned_on"]["flagged_takes_excluded"] is True
        assert main(["evaluate", "--run", "a", "--sensitivity-run", "b", "--no-stream", *args]) == 0

    def test_a_changed_dataset_stops_fitting(self, dirs, capsys):
        args, tmp = dirs
        main(["split", *args])
        victim = next((tmp / "datasets").glob("s-full-a.jsonl"))
        victim.write_bytes(victim.read_bytes() + b"\n")
        code = main(["fit-baseline", "--split", split_id(tmp), "--allow-synthetic", *args])
        assert code == 1 and "changed since" in capsys.readouterr().err


@pytest.mark.skipif(HAS_TORCH, reason="PyTorch is installed here; that path is unreachable")
class TestTrainWithoutPyTorch:
    def test_train_fails_with_instructions_and_creates_nothing(self, dirs, capsys):
        args, tmp = dirs
        main(["split", *args])
        code = main(["train", "--split", split_id(tmp), "--run-name", "t", "--allow-synthetic",
                     *args])  # fmt: skip
        err = capsys.readouterr().err
        assert code == 1 and "PyTorch is not installed" in err and "DYNAMIC_GESTURES.md" in err
        assert not (tmp / "runs" / "t").exists()


class TestArgumentChecks:
    @pytest.mark.parametrize("extra", [["--seeds", "0"], ["--max-epochs", "0"],
                                       ["--hidden-sizes", "0"]])  # fmt: skip
    def test_invalid_training_arguments_are_a_usage_error(self, dirs, extra, capsys):
        args, tmp = dirs
        main(["split", *args])
        code = main(["train", "--split", split_id(tmp), *extra, *args])
        assert code == 64 and "must be >= 1" in capsys.readouterr().err
