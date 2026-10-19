"""Summary, command line, ignore rules, documentation and isolation of the dynamic tools.

The datasets built here are SYNTHETIC and exist only to exercise the software.
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

from dynamic_helpers import circle_path, make_sequence, swipe_path
from nocursor.dynamic.cli import build_parser, main
from nocursor.dynamic.dataset import SessionWriter, load_dataset
from nocursor.dynamic.schema import SessionHeader
from nocursor.dynamic.summary import coverage_notes, format_summary, summarize
from nocursor.paths import default_dataset_dir

ROOT = Path(__file__).resolve().parents[1]
DYNAMIC = ROOT / "src" / "nocursor" / "dynamic"


def write_synthetic_session(directory, session_id, participant="p01", *, takes=2, mislabel=False):
    header = SessionHeader(session_id, participant, "2026-10-12T10:00:00Z",
                           hand="right", provenance="synthetic")  # fmt: skip
    writer = SessionWriter(directory, header)
    n = 0
    for k in range(takes):
        amplitude = 0.18 + 0.01 * k  # distinct takes, so no duplicate-content warning
        for label, path in (
            ("swipe_left", swipe_path(-1, amplitude)),
            ("swipe_right", swipe_path(+1, amplitude)),
            ("circle_clockwise", circle_path(True, 0.1 + 0.01 * k)),
        ):
            n += 1
            writer.append(make_sequence(path, label=label, sequence_id=f"{session_id}-{n:04d}",
                                        session_id=session_id))  # fmt: skip
        n += 1
        writer.append(make_sequence(swipe_path(+1, 0.05 + 0.01 * k), label="no_gesture",
                                    subtype="idle", sequence_id=f"{session_id}-{n:04d}",
                                    session_id=session_id))  # fmt: skip
    if mislabel:
        n += 1
        wrong = make_sequence(swipe_path(+1), label="swipe_left", session_id=session_id,
                              sequence_id=f"{session_id}-{n:04d}")  # moves right, labelled left
        writer.append(wrong)
    writer.close()


class TestSummary:
    def test_counts_by_label_session_and_participant(self, tmp_path):
        write_synthetic_session(tmp_path, "sess-a", "p01", takes=2)
        write_synthetic_session(tmp_path, "sess-b", "p02", takes=3)
        summary = summarize(load_dataset(tmp_path))
        assert summary["sessions"] == 2 and summary["participants"] == 2
        assert summary["takes"] == 20
        assert summary["by_label"] == {"swipe_left": 5, "swipe_right": 5, "circle_clockwise": 5,
                                       "no_gesture": 5}  # fmt: skip
        assert summary["no_gesture_by_subtype"] == {"idle": 5}
        assert [row["session_id"] for row in summary["per_session"]] == ["sess-a", "sess-b"]
        assert summary["featurizable_takes"] == 20
        assert summary["median_open_palm_fraction_targets"] == 1.0

    def test_synthetic_data_is_called_out(self, tmp_path):
        write_synthetic_session(tmp_path, "sess-a")
        notes = summarize(load_dataset(tmp_path))["coverage_notes"]
        assert any("SYNTHETIC" in n and "never for accuracy claims" in n for n in notes)

    def test_too_few_sessions_to_split_is_stated(self, tmp_path):
        write_synthetic_session(tmp_path, "sess-a")
        notes = summarize(load_dataset(tmp_path))["coverage_notes"]
        assert any("at least 3 are needed" in n for n in notes)
        assert any("One participant" in n and "not across people" in n for n in notes)
        assert any("No continuous-use recordings" in n for n in notes)

    def test_an_empty_dataset(self, tmp_path):
        summary = summarize(load_dataset(tmp_path))
        assert summary["sessions"] == 0 and "No sessions found." in summary["coverage_notes"]
        assert "sessions 0" in format_summary(summary)

    def test_a_mislabelled_take_is_listed_but_not_removed(self, tmp_path):
        write_synthetic_session(tmp_path, "sess-a", mislabel=True)
        summary = summarize(load_dataset(tmp_path))
        assert summary["by_label"]["swipe_left"] == 3  # still counted
        assert len(summary["suspect_labels"]) == 1
        assert "moved right" in summary["suspect_labels"][0]["note"]
        text = format_summary(summary)
        assert "second look" in text and "nothing was changed" in text

    def test_takes_that_cannot_become_features_are_counted_with_a_reason(self, tmp_path):
        header = SessionHeader("sess-a", "p01", "2026-10-12T10:00:00Z")
        writer = SessionWriter(tmp_path, header)
        writer.append(make_sequence(swipe_path(1), duration=1.0, label="no_gesture",
                                    sequence_id="sess-a-0001", session_id="sess-a"))  # fmt: skip
        writer.close()
        summary = summarize(load_dataset(tmp_path))
        assert summary["featurizable_takes"] == 0
        assert sum(summary["unfeaturizable_reasons"].values()) == 1

    def test_the_summary_is_json_serialisable(self, tmp_path):
        write_synthetic_session(tmp_path, "sess-a")
        json.dumps(summarize(load_dataset(tmp_path)))

    def test_coverage_never_claims_accuracy(self, tmp_path):
        write_synthetic_session(tmp_path, "sess-a")
        text = format_summary(summarize(load_dataset(tmp_path)))
        assert "not accuracy estimates" in text
        assert "accuracy of" not in text.lower()

    def test_enough_sessions_remove_the_split_warning(self):
        base = {"sessions": 6, "participants": 2, "provenance": ["recorded"], "streams": 1,
                "by_label": {"swipe_left": 90, "swipe_right": 90, "circle_clockwise": 90,
                             "no_gesture": 180}}  # fmt: skip
        assert coverage_notes(base) == []


class TestCommandLine:
    def test_no_command_is_a_usage_error(self):
        with pytest.raises(SystemExit) as info:
            main([])
        assert info.value.code == 2

    def test_the_modelling_commands_exist_and_need_their_arguments(self):
        for command in ("split", "fit-baseline", "train", "evaluate"):
            assert command in build_parser()._subparsers._group_actions[0].choices
        for command in ("fit-baseline", "train", "evaluate"):
            with pytest.raises(SystemExit) as info:  # required --split / --run is missing
                main([command])
            assert info.value.code == 2

    def test_collect_defaults(self):
        args = build_parser().parse_args(["collect"])
        assert args.participant == "p01" and args.hand == "unspecified"
        assert (args.per_class, args.negatives, args.countdown_s) == (15, 30, 2.0)
        assert not args.free and args.stream_seconds is None and args.seed is None

    def test_a_real_name_as_participant_is_refused_before_the_camera_is_touched(self, capsys):
        code = main(["collect", "--participant", "Jane Doe"])
        assert code == 64
        assert "pseudonym" in capsys.readouterr().err

    def test_summarize_prints_and_succeeds(self, tmp_path, capsys):
        write_synthetic_session(tmp_path, "sess-a")
        assert main(["summarize", "--dataset-dir", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "Dynamic gesture dataset" in out and "swipe_left" in out

    def test_summarize_json(self, tmp_path, capsys):
        write_synthetic_session(tmp_path, "sess-a")
        main(["summarize", "--dataset-dir", str(tmp_path), "--json"])
        assert json.loads(capsys.readouterr().out)["sessions"] == 1

    def test_summarize_reports_damaged_files_but_still_shows_the_rest(self, tmp_path, capsys):
        write_synthetic_session(tmp_path, "sess-a")
        (tmp_path / "broken.jsonl").write_text("garbage\n", encoding="utf-8")
        assert main(["summarize", "--dataset-dir", str(tmp_path)]) == 0
        captured = capsys.readouterr()
        assert "sessions 1" in captured.out and "broken.jsonl" in captured.err

    def test_validate_succeeds_on_a_clean_dataset(self, tmp_path, capsys):
        write_synthetic_session(tmp_path, "sess-a")
        assert main(["validate", "--dataset-dir", str(tmp_path)]) == 0
        assert "0 error(s)" in capsys.readouterr().out

    def test_validate_fails_on_a_damaged_file(self, tmp_path, capsys):
        write_synthetic_session(tmp_path, "sess-a")
        path = tmp_path / "sess-a.jsonl"
        lines = path.read_text(encoding="utf-8").split("\n")
        lines[2] = lines[2][:100]
        path.write_text("\n".join(lines), encoding="utf-8")
        assert main(["validate", "--dataset-dir", str(tmp_path)]) == 1
        out = capsys.readouterr().out
        assert "sess-a.jsonl:3" in out and "1 error(s)" in out

    def test_a_missing_dataset_directory(self, tmp_path, capsys):
        assert main(["validate", "--dataset-dir", str(tmp_path / "nope")]) == 1
        assert "not found" in capsys.readouterr().err

    def test_the_default_dataset_directory_is_the_ignored_one(self):
        assert default_dataset_dir().name == "datasets"


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class TestIsolation:
    FORBIDDEN = {"pyautogui", "nocursor.controller", "nocursor.dispatch", "nocursor.app",
                 "nocursor.runtime", "nocursor.emergency"}  # fmt: skip

    @pytest.mark.parametrize("path", sorted(DYNAMIC.glob("*.py")), ids=lambda p: p.name)
    def test_the_dynamic_tools_cannot_send_desktop_input(self, path):
        assert not imported_modules(path) & self.FORBIDDEN

    def test_pytorch_is_imported_only_inside_the_training_functions(self):
        """The only module that may name torch is train.py, and never at module level."""
        heavy = {"tensorflow", "sklearn", "scipy", "onnxruntime", "pandas"}
        for path in DYNAMIC.glob("*.py"):
            modules = {m.split(".")[0] for m in imported_modules(path)}
            assert not modules & heavy, path.name
            if path.name != "train.py":
                assert "torch" not in modules, f"{path.name} imports torch"
        tree = ast.parse((DYNAMIC / "train.py").read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "torch" for a in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "torch"

    def test_the_existing_application_does_not_depend_on_the_new_package(self):
        for path in (ROOT / "src" / "nocursor").glob("*.py"):
            assert not any(m.startswith("nocursor.dynamic") for m in imported_modules(path)), (
                f"{path.name} imports the dynamic-gesture package"
            )

    def test_the_console_script_and_runtime_dependencies_are_unchanged(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        assert project["scripts"] == {"nocursor": "nocursor.cli:main"}
        assert len(project["dependencies"]) == 4


class TestIgnoreRulesAndDocs:
    GUIDE = ROOT / "docs" / "DYNAMIC_GESTURES.md"

    @pytest.mark.parametrize("pattern", ["datasets/", "ml_runs/"])
    def test_personal_data_and_derived_artifacts_are_ignored(self, pattern):
        lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert pattern in lines

    def test_every_option_of_every_command_is_documented(self):
        text = self.GUIDE.read_text(encoding="utf-8")
        choices = build_parser()._subparsers._group_actions[0].choices
        assert {"collect", "summarize", "validate", "split", "fit-baseline", "train",
                "evaluate"} <= set(choices)  # fmt: skip
        for command, parser in choices.items():
            flags = {s for a in parser._actions for s in a.option_strings
                     if s.startswith("--")} - {"--help"}
            missing = sorted(f for f in flags if f not in text)
            assert not missing, f"{command}: undocumented options {missing}"

    def test_every_command_is_documented_with_the_module_entry_point(self):
        text = self.GUIDE.read_text(encoding="utf-8")
        for command in ("collect", "summarize", "validate", "split", "fit-baseline", "train",
                        "evaluate"):  # fmt: skip
            assert f"python -m nocursor.dynamic {command}" in text

    def test_the_guide_states_the_conventions_and_limits_it_depends_on(self):
        text = " ".join(self.GUIDE.read_text(encoding="utf-8").split()).lower()  # unwrap lines
        for phrase in ("mirrored preview", "open palm", "pseudonym", "no video", "synthetic",
                       "limited evidence", "cannot send desktop input", "reserved stream",
                       "frozen", "never needs pytorch", ".venv-ml"):  # fmt: skip
            assert phrase in text, phrase

    def test_the_guide_does_not_report_results_that_do_not_exist(self):
        text = self.GUIDE.read_text(encoding="utf-8")
        assert not re.search(r"\b\d{2,3}(\.\d+)?\s?%\s+(accuracy|recall|precision)", text)
        assert "F1 of" not in text and "accuracy of" not in text.lower()

    def test_the_readme_and_architecture_point_at_the_guide(self):
        for name in ("README.md", "docs/ARCHITECTURE.md"):
            assert "DYNAMIC_GESTURES.md" in (ROOT / name).read_text(encoding="utf-8"), name


IMPORT_EVERYTHING = """
import importlib, pkgutil, sys
import nocursor.dynamic
for info in pkgutil.iter_modules(nocursor.dynamic.__path__):
    if info.name != "__main__":
        importlib.import_module("nocursor.dynamic." + info.name)
for name in ("nocursor.cli", "nocursor.app", "nocursor.processor", "nocursor.controller",
             "nocursor.interaction", "nocursor.runtime", "nocursor.hud", "nocursor.dispatch"):
    importlib.import_module(name)
"""


def run_python(code, *, block_torch=False, cwd=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    prefix = "import sys\nsys.modules['torch'] = None\n" if block_torch else ""
    return subprocess.run([sys.executable, "-c", prefix + code], capture_output=True, text=True,
                          env=env, cwd=cwd or ROOT, timeout=120)


class TestTorchIsolation:
    """Checked in fresh interpreters, so nothing imported by the test run itself can interfere."""

    def test_everything_imports_when_pytorch_is_unavailable(self):
        """`sys.modules['torch'] = None` makes `import torch` fail, as without PyTorch installed."""
        result = run_python(IMPORT_EVERYTHING + "print('ok')", block_torch=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"

    def test_training_explains_itself_when_pytorch_is_unavailable(self):
        code = IMPORT_EVERYTHING + """
from nocursor.dynamic.train import TorchUnavailableError, _import_torch
try:
    _import_torch()
except TorchUnavailableError as exc:
    print("raised:", "PyTorch is not installed" in str(exc))
"""
        result = run_python(code, block_torch=True)
        assert result.returncode == 0, result.stderr
        assert "raised: True" in result.stdout

    def test_importing_the_runtime_and_every_dynamic_module_does_not_load_pytorch(self):
        result = run_python(IMPORT_EVERYTHING + "print('torch' in sys.modules)")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"

    def test_the_command_line_help_does_not_load_pytorch(self):
        code = """
import sys
from nocursor.dynamic.cli import main
try:
    main(["train", "--help"])
except SystemExit:
    pass
print('torch' in sys.modules)
"""
        result = run_python(code)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().endswith("False")

    def test_loading_and_running_a_saved_model_does_not_load_pytorch(self, tmp_path):
        from ml_fixtures import model_metadata, random_arrays
        from nocursor.dynamic.model_io import save_artifact

        save_artifact(tmp_path / "model", random_arrays(), model_metadata())
        code = f"""
import sys
import numpy as np
from nocursor.dynamic.model_io import load_classifier
model = load_classifier(__import__("pathlib").Path({str(tmp_path / "model")!r}))
prediction = model.predict_features(np.zeros((32, 46)))
print(prediction.valid, sys.modules.get('torch') is None)
"""
        result = run_python(code, block_torch=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True True"  # valid; torch is still the sentinel


class TestPackagingForTheMlExtra:
    PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    def test_pytorch_is_only_in_the_optional_ml_extra(self):
        project = self.PYPROJECT["project"]
        extras = project["optional-dependencies"]
        assert [d.split(">")[0] for d in extras["ml"]] == ["torch"]
        assert not any("torch" in d for d in project["dependencies"])
        assert not any("torch" in d for d in extras["dev"])
        assert len(project["dependencies"]) == 4

    def test_the_ml_environment_is_ignored_by_git(self):
        lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert ".venv-ml/" in lines and ".venv/" in lines
