"""Packaging metadata, entry points, and documentation consistency."""

import importlib
import re
from pathlib import Path

import pytest
import tomllib

from nocursor import __version__
from nocursor.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
DOC_FILES = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("*.md"))]


def requirement_names(lines):
    names = set()
    for line in lines:
        line = line.split("#")[0].strip()
        if not line:
            continue
        names.add(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


class TestMetadata:
    def test_runtime_dependencies_are_the_documented_four(self):
        names = requirement_names(PYPROJECT["project"]["dependencies"])
        assert names == {"opencv-contrib-python", "mediapipe", "numpy", "pyautogui"}

    def test_exactly_one_opencv_distribution_is_required(self):
        names = requirement_names(PYPROJECT["project"]["dependencies"])
        opencv = {n for n in names if n.startswith("opencv")}
        assert opencv == {"opencv-contrib-python"}  # what MediaPipe itself requires; has GUI

    def test_requirements_txt_matches_pyproject_plus_dev_tools(self):
        lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        assert requirement_names(lines) == (
            requirement_names(PYPROJECT["project"]["dependencies"]) | {"pytest", "ruff"}
        )

    def test_lock_file_pins_every_runtime_dependency_exactly(self):
        lock = ROOT / "requirements-lock.txt"
        if not lock.is_file():
            pytest.skip("requirements-lock.txt not generated")
        pinned = {}
        for line in lock.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "==" in line:
                name, version = line.split("==", 1)
                pinned[name.lower().replace("_", "-")] = version
        for name in requirement_names(PYPROJECT["project"]["dependencies"]):
            assert name in pinned, f"{name} is not pinned in requirements-lock.txt"
        assert "opencv-python" not in pinned and "opencv-python-headless" not in pinned

    def test_dev_extra_has_the_test_and_lint_tools(self):
        dev = requirement_names(PYPROJECT["project"]["optional-dependencies"]["dev"])
        assert dev == {"pytest", "ruff"}

    def test_version_matches_the_package(self):
        assert PYPROJECT["project"]["version"] == __version__

    def test_python_requirement_is_declared(self):
        assert PYPROJECT["project"]["requires-python"].startswith(">=3.")

    def test_readme_is_the_long_description_source(self):
        assert PYPROJECT["project"]["readme"] == "README.md"

    def test_no_license_was_chosen_for_the_owner(self):
        project = PYPROJECT["project"]
        assert "license" not in project and "license-files" not in project
        assert not any(c.startswith("License ::") for c in project.get("classifiers", []))
        assert not list(ROOT.glob("LICENSE*"))

    def test_the_model_binary_is_not_part_of_the_package(self):
        find = PYPROJECT["tool"]["setuptools"]["packages"]["find"]
        assert find["where"] == ["src"]
        assert not list((ROOT / "src").rglob("*.task"))


class TestEntryPoints:
    def test_console_script_points_at_cli_main(self):
        assert PYPROJECT["project"]["scripts"] == {"nocursor": "nocursor.cli:main"}

    def test_the_entry_point_target_exists_and_is_callable(self):
        module, attribute = PYPROJECT["project"]["scripts"]["nocursor"].split(":")
        assert callable(getattr(importlib.import_module(module), attribute))

    def test_python_dash_m_entry_point_exists(self):
        assert (ROOT / "src" / "nocursor" / "__main__.py").is_file()

    def test_installed_metadata_exposes_the_command_when_installed(self):
        import importlib.metadata as md

        try:
            eps = md.entry_points(group="console_scripts")
        except Exception:  # pragma: no cover
            pytest.skip("entry points unavailable")
        ours = [e for e in eps if e.name == "nocursor"]
        if not ours:
            pytest.skip("package not installed in this environment")
        assert ours[0].value == "nocursor.cli:main"


class TestRuntimeFilesStayOutOfSourceControl:
    @pytest.mark.parametrize("pattern", [
        ".venv/", "calibration.local.json", "logs/", "benchmarks/", "models/*.task",
        "models/*.part", "build/", "dist/", "*.egg-info/", "__pycache__/",
    ])
    def test_gitignore_covers_it(self, pattern):
        lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert pattern in lines


def markdown_links(path):
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)  # ignore code blocks
    for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)\)", text):
        if not target.startswith(("http://", "https://", "mailto:", "#")):
            yield target.split("#")[0]


class TestDocumentation:
    @pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.name)
    def test_relative_links_resolve(self, doc):
        for target in markdown_links(doc):
            assert (doc.parent / target).resolve().exists(), f"{doc.name} links to missing {target}"

    def test_the_renamed_manual_is_referenced_everywhere_by_its_new_name(self):
        assert (ROOT / "docs" / "MANUAL_TESTING.md").is_file()
        assert not (ROOT / "docs" / "PHASE1_MANUAL_TESTING.md").exists()
        for doc in DOC_FILES:  # a historical note may name the old file; no link may point to it
            assert not any("PHASE1_MANUAL_TESTING" in target for target in markdown_links(doc))
        assert "PHASE1_MANUAL_TESTING" not in (ROOT / "README.md").read_text(encoding="utf-8")

    def test_every_command_line_option_is_documented_somewhere(self):
        corpus = "\n".join(p.read_text(encoding="utf-8") for p in DOC_FILES)
        flags = {s for a in build_parser()._actions for s in a.option_strings if s.startswith("--")}
        flags -= {"--help"}
        missing = sorted(f for f in flags if f not in corpus)
        assert not missing, f"undocumented options: {missing}"

    def test_the_readme_has_the_required_sections(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for heading in ("## Features", "## Gestures", "## Architecture", "## Technology stack",
                        "## Install", "## Model setup", "## Run", "## Dry-run", "## Calibration",
                        "## Safety and the emergency stop", "## Benchmarking", "## Testing",
                        "## Known limitations", "## Troubleshooting", "## Privacy",
                        "## Roadmap and contributing"):
            assert heading in readme, heading

    def test_diagrams_are_well_formed(self):
        for doc in DOC_FILES:
            text = doc.read_text(encoding="utf-8")
            blocks = re.findall(r"```mermaid\n(.*?)```", text, flags=re.DOTALL)
            assert text.count("```") % 2 == 0, f"unbalanced code fences in {doc.name}"
            for block in blocks:
                assert block.lstrip().startswith(("flowchart", "stateDiagram-v2")), doc.name
                assert len(re.findall(r"^\s*subgraph\b", block, re.M)) == len(
                    re.findall(r"^\s*end\s*$", block, re.M)
                )

    def test_architecture_diagram_uses_every_real_state_name(self):
        from nocursor.interaction import InteractionState

        text = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        for state in InteractionState:
            assert state.name in text, state.name

    def test_documented_defaults_match_the_code(self):
        from nocursor.config import Config

        c = Config()
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
        assert (c.pinch_threshold, c.pinch_confirm_frames, c.pinch_release_factor) == (0.2, 5, 1.5)
        assert "`0.20`" in readme and "`5`" in readme
        assert "`0.20`" in arch and "`5` consecutive frames" in arch
        assert f"`{c.drag_start_px:g}`" in arch and "`0.4 s`" in arch and c.click_cooldown_s == 0.4

    def test_no_placeholder_text_was_left_in_the_docs(self):
        for doc in DOC_FILES:
            text = doc.read_text(encoding="utf-8")
            for marker in ("RESULTS_PLACEHOLDER", "TODO", "FIXME", "XXX", "lorem ipsum"):
                assert marker not in text, f"{marker} found in {doc.name}"

    def test_docs_do_not_claim_artifacts_that_do_not_exist(self):
        demo = (ROOT / "docs" / "DEMO_PLAN.md").read_text(encoding="utf-8")
        assert "No video, screenshot or GIF of NoCursor has been produced" in demo
        for ext in (".mp4", ".gif", ".png", ".jpg", ".webm"):
            assert not list(ROOT.glob(f"docs/*{ext}")) or ext not in demo

    def test_benchmark_docs_never_equate_software_timing_with_display_latency(self):
        text = (ROOT / "docs" / "BENCHMARKING.md").read_text(encoding="utf-8")
        assert "not camera-to-display latency" in text.lower()
        for bad in ("end-to-end latency of", "camera-to-display latency is", "glass-to-glass"):
            assert bad not in text.lower()
