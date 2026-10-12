"""CLI: grouped help, documented ranges, informational commands, error reporting, logging."""

import logging
from dataclasses import fields
from pathlib import Path

import pytest

from nocursor import __version__
from nocursor.checks import FAIL, OK, WARN, checks_exit_code, format_checks, run_checks
from nocursor.cli import RANGES, build_parser, main
from nocursor.config import Config, ConfigError
from nocursor.settings import SETTING_GROUPS, describe_settings, ungrouped_fields

NOT_ON_THE_COMMAND_LINE = {"max_read_failures", "min_hand_size", "stable_frames"}


@pytest.fixture(autouse=True)
def clean_logging():
    yield
    logging.getLogger("nocursor").handlers.clear()
    for h in list(logging.getLogger().handlers):
        if getattr(h, "_nocursor_handler", False):
            logging.getLogger().removeHandler(h)


def help_text():
    return " ".join(build_parser().format_help().split())


class TestHelp:
    def test_options_are_grouped_by_purpose(self):
        text = build_parser().format_help()
        for title in ("getting started:", "camera and model:", "pointer:", "pinch: click and drag:",
                      "two-finger scrolling:", "calibration:", "safety:",
                      "diagnostics, logging and performance:", "benchmarks (always dry-run"):
            assert title in text

    def test_every_config_option_has_a_command_line_flag_except_the_documented_few(self):
        dests = {a.dest for a in build_parser()._actions}
        missing = {f.name for f in fields(Config)} - dests
        assert missing == NOT_ON_THE_COMMAND_LINE

    def test_every_ranged_option_documents_its_default_and_its_range(self):
        text = help_text()
        defaults = Config()
        for field, (range_text, _valid, _invalid) in RANGES.items():
            if field.startswith("benchmark_"):
                assert f"valid: {range_text}" in text
                continue
            default = getattr(defaults, field)
            shown = f"{default:g}" if isinstance(default, float) else str(default)
            assert f"default {shown}" in text, field
            assert f"valid: {range_text}" in text, field

    @pytest.mark.parametrize("field", sorted(RANGES))
    def test_documented_ranges_match_the_real_validation(self, field):
        _text, valid, invalid = RANGES[field]
        extra = {}
        if field == "benchmark_frames":
            extra = {"benchmark": "synthetic"}
        elif field == "benchmark_seconds":
            extra = {"benchmark": "live"}
        Config(**{field: valid, **extra})  # the documented edge is accepted ...
        with pytest.raises(ConfigError):
            Config(**{field: invalid, **extra})  # ... and just beyond it is rejected

    def test_epilog_has_examples_and_keys(self):
        text = help_text()
        for needle in ("examples:", "nocursor --check", "--dry-run --diagnostics",
                       "--benchmark synthetic", "keys in the preview window",
                       "Ctrl+Alt+Q stops from any window"):
            assert needle in text

    def test_the_depth_flags_are_documented_as_default_and_experimental(self):
        text = help_text()
        assert "already the default" in text and "EXPERIMENTAL" in text

    def test_benchmark_options_state_that_they_cannot_send_input(self):
        text = help_text()
        assert "always dry-run: they cannot send desktop input" in text
        assert "Results from the three are never mixed" in text

    def test_default_pinch_is_still_2d_and_unchanged_thresholds(self):
        c = Config()
        assert (c.pinch_use_depth, c.pinch_threshold, c.pinch_confirm_frames) == (False, 0.2, 5)

    def test_no_log_file_is_written_by_default(self):
        assert vars(build_parser().parse_args([]))["log_file"] is None

    def test_log_file_flag_with_and_without_a_path(self, tmp_path):
        bare = vars(build_parser().parse_args(["--log-file"]))["log_file"]
        assert bare.name == "nocursor.log" and bare.parent.name == "logs"
        given = vars(build_parser().parse_args(["--log-file", str(tmp_path / "x.log")]))
        assert given["log_file"] == tmp_path / "x.log"


class TestInformationalCommands:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0 and f"nocursor {__version__}" in capsys.readouterr().out

    def test_version_matches_the_package_metadata(self):
        import importlib.metadata as md

        try:
            installed = md.version("nocursor")
        except md.PackageNotFoundError:
            pytest.skip("package metadata not available")
        assert installed == __version__

    def test_print_settings_groups_and_marks_changes(self, capsys):
        assert main(["--print-settings", "--smoothing", "0.8", "--pinch-depth"]) == 0
        out = capsys.readouterr().out
        assert "Pinch measurement: 3D pinch ratio (experimental)" in out
        assert "* smoothing" in out and "0.8" in out
        assert "  pinch_threshold" in out  # unchanged: no marker
        assert "Benchmark (always dry-run):" in out

    def test_print_settings_rejects_an_invalid_value_first(self, capsys):
        assert main(["--print-settings", "--smoothing", "5"]) == 64

    def test_every_config_field_is_in_a_settings_group(self):
        assert ungrouped_fields() == []
        assert sum(len(v) for v in SETTING_GROUPS.values()) == len(fields(Config))

    def test_settings_show_unset_and_boolean_values_readably(self):
        text = describe_settings(Config(dry_run=True))
        assert "(not set)" in text and "dry_run" in text and "on" in text

    def test_download_model_runs_the_downloader_and_exits(self, monkeypatch):
        import nocursor.model_setup as ms

        calls = []
        def fake_download(force=False, dest=None):
            calls.append(force)
            return 0

        monkeypatch.setattr(ms, "run_download", fake_download)
        assert main(["--download-model"]) == 0
        assert main(["--redownload-model"]) == 0
        assert calls == [False, True]


class FakeBackend:
    def screen_size(self):
        return (1920, 1080)

    def _no(self, *a):
        raise AssertionError("--check must not send input")

    move_to = click = mouse_down = mouse_up = scroll = _no


class TestCheck:
    @pytest.fixture
    def healthy(self, monkeypatch, tmp_path):
        import nocursor.checks as checks
        import nocursor.controller as controller
        import nocursor.hand_tracker as tracker

        model = tmp_path / "hand_landmarker.task"
        model.write_bytes(b"x" * 2_000_000)
        monkeypatch.setattr(tracker, "resolve_model_path", lambda _p: model)
        monkeypatch.setattr(controller, "PyAutoGUIBackend", FakeBackend)
        monkeypatch.setattr(checks, "installed_opencv", lambda: {"opencv-contrib-python": "5.0"})
        monkeypatch.setattr(checks, "opencv_gui_support", lambda: True)
        return tmp_path

    def config(self, tmp_path, **kw):
        return Config(calibration_path=tmp_path / "c.json", **kw)

    def test_a_healthy_installation_is_ready(self, healthy):
        items = run_checks(self.config(healthy))
        assert not [i for i in items if i.status == FAIL]
        assert "Ready to run." in format_checks(items) and checks_exit_code(items) == 0

    def test_it_never_opens_the_camera(self, healthy, monkeypatch):
        import cv2

        def forbidden(*a, **k):
            raise AssertionError("--check must not open a camera")

        monkeypatch.setattr(cv2, "VideoCapture", forbidden)
        items = run_checks(self.config(healthy))
        camera = next(i for i in items if i.name == "Camera")
        assert "not opened" in camera.detail

    def test_missing_model_fails_with_the_download_command(self, healthy, monkeypatch):
        import nocursor.hand_tracker as tracker

        def missing(_p):
            raise tracker.ModelNotFoundError("hand_landmarker.task not found. Download it ...")

        monkeypatch.setattr(tracker, "resolve_model_path", missing)
        items = run_checks(self.config(healthy))
        model = next(i for i in items if i.name == "Hand-landmark model")
        assert model.status == FAIL and checks_exit_code(items) == 1
        assert "problem(s) must be fixed" in format_checks(items)

    def test_headless_opencv_is_a_failure_with_the_fix(self, healthy, monkeypatch):
        import nocursor.checks as checks

        monkeypatch.setattr(checks, "opencv_gui_support", lambda: False)
        items = run_checks(self.config(healthy))
        gui = next(i for i in items if i.name == "OpenCV window support")
        assert gui.status == FAIL and "headless" in gui.detail
        assert "opencv-contrib-python" in gui.detail

    def test_duplicate_opencv_packages_warn(self, healthy, monkeypatch):
        import nocursor.checks as checks

        monkeypatch.setattr(checks, "installed_opencv",
                            lambda: {"opencv-contrib-python": "5.0", "opencv-python": "5.0"})
        items = run_checks(self.config(healthy))
        dup = next(i for i in items if i.name == "OpenCV packages")
        assert dup.status == WARN and "overwrite" in dup.detail
        assert checks_exit_code(items) == 0  # a warning, not a failure

    def test_a_missing_package_fails_with_the_install_command(self, healthy, monkeypatch):
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda name, *a: None if name == "mediapipe" else real(name, *a))
        items = run_checks(self.config(healthy))
        mp = next(i for i in items if i.name == "mediapipe")
        assert mp.status == FAIL and "pip install -r requirements.txt" in mp.detail

    def test_a_corrupt_calibration_is_a_warning(self, healthy):
        (healthy / "c.json").write_text("not json")
        items = run_checks(self.config(healthy))
        cal = next(i for i in items if i.name == "Calibration")
        assert cal.status == WARN and "default mapping" in cal.detail

    def test_a_valid_calibration_is_reported(self, healthy):
        from nocursor.calibration import CalibrationStore
        from nocursor.mapping import ViewRegion

        CalibrationStore(healthy / "c.json").save(ViewRegion(0.2, 0.2, 0.8, 0.8), 0, 640 / 480)
        cal = next(i for i in run_checks(self.config(healthy)) if i.name == "Calibration")
        assert cal.status == OK

    def test_no_calibration_flag_is_reported(self, healthy):
        cal = next(i for i in run_checks(self.config(healthy, use_calibration=False))
                   if i.name == "Calibration")
        assert "--no-calibration" in cal.detail

    def test_screen_read_failure_is_a_warning(self, healthy, monkeypatch):
        import nocursor.controller as controller

        class Broken:
            def __init__(self):
                raise RuntimeError("no display")

        monkeypatch.setattr(controller, "PyAutoGUIBackend", Broken)
        screen = next(i for i in run_checks(self.config(healthy)) if i.name == "Screen size")
        assert screen.status == WARN and "no display" in screen.detail

    def test_log_file_location_is_checked_only_when_requested(self, healthy):
        names = [i.name for i in run_checks(self.config(healthy))]
        assert "Log file" not in names
        items = run_checks(self.config(healthy, log_file=healthy / "logs" / "n.log"))
        assert next(i for i in items if i.name == "Log file").status == OK
        assert not list((healthy / "logs").glob(".nocursor_write_test"))

    def test_unwritable_log_location_warns(self, healthy):
        blocker = healthy / "blocked"
        blocker.write_text("file")
        items = run_checks(self.config(healthy, log_file=blocker / "x" / "n.log"))
        assert next(i for i in items if i.name == "Log file").status == WARN

    def test_pinch_mode_is_shown(self, healthy):
        mode = next(i for i in run_checks(self.config(healthy)) if i.name == "Pinch measurement")
        assert "2D pinch ratio (default)" in mode.detail
        mode3 = next(i for i in run_checks(self.config(healthy, pinch_use_depth=True))
                     if i.name == "Pinch measurement")
        assert "experimental" in mode3.detail

    def test_cli_check_exit_codes(self, healthy, capsys):
        args = ["--check", "--calibration-file", str(healthy / "c.json")]
        assert main(args) == 0
        assert "[ ok ]" in capsys.readouterr().out


class TestErrorReporting:
    def test_missing_packages_give_one_clear_message_not_a_traceback(self, monkeypatch, capsys):
        import nocursor.runtime as runtime

        monkeypatch.setattr(runtime, "missing_dependencies",
                            lambda: [("mediapipe", "hand-landmark detection")])
        assert main(["--dry-run"]) == 1
        err = capsys.readouterr().err
        assert "missing required package(s): mediapipe" in err
        assert "pip install -r requirements.txt" in err and "Traceback" not in err

    def test_missing_model_names_the_download_command(self, monkeypatch, capsys, tmp_path):
        import nocursor.hand_tracker as tracker

        def missing(_p):
            raise tracker.ModelNotFoundError(
                "hand_landmarker.task not found. Download it with `nocursor --download-model` "
                "or pass --model PATH.")

        monkeypatch.setattr(tracker, "resolve_model_path", missing)
        assert main(["--dry-run", "--calibration-file", str(tmp_path / "c.json")]) == 1
        assert "nocursor --download-model" in capsys.readouterr().err

    def test_a_corrupt_model_suggests_redownloading(self, monkeypatch, capsys, tmp_path):
        import nocursor.controller
        import nocursor.hand_tracker as tracker

        class Backend:
            def screen_size(self):
                return (1920, 1080)

        class Bad:
            def __init__(self, *a, **k):
                raise RuntimeError("Unable to open zip archive.")

        monkeypatch.setattr(nocursor.controller, "PyAutoGUIBackend", Backend)
        monkeypatch.setattr(tracker, "resolve_model_path", lambda _p: tmp_path)
        monkeypatch.setattr(tracker, "HandTracker", Bad)
        assert main(["--dry-run", "--calibration-file", str(tmp_path / "c.json")]) == 1
        err = capsys.readouterr().err
        assert "Unable to open zip archive" in err and "--redownload-model" in err

    def test_invalid_arguments_name_the_flag(self, capsys):
        assert main(["--scroll-speed", "500"]) == 64
        err = capsys.readouterr().err
        assert "--scroll-speed" in err and "(0, 100]" in err

    @pytest.mark.parametrize("argv,needle", [
        (["--benchmark", "video"], "--benchmark-video"),
        (["--benchmark-frames", "10"], "only apply together with --benchmark"),
        (["--benchmark", "live", "--benchmark-frames", "10"], "--benchmark-seconds"),
        (["--benchmark", "synthetic", "--calibrate"], "--calibrate cannot be combined"),
    ])
    def test_benchmark_misuse_is_a_usage_error(self, argv, needle, capsys):
        assert main(argv) == 64
        assert needle in capsys.readouterr().err

    def test_ctrl_c_during_startup_exits_cleanly(self, monkeypatch, capsys):
        import nocursor.cli as cli

        def interrupted(config):
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "run", interrupted)
        assert main(["--dry-run"]) == 130
        assert "interrupted" in capsys.readouterr().err


class TestBenchmarkCommand:
    def test_runs_and_prints_the_summary(self, monkeypatch, capsys):
        import nocursor.benchmark as bm

        seen = {}

        def fake(config):
            seen["config"] = config
            return bm.BenchmarkResult({}, "SUMMARY TEXT", None)

        monkeypatch.setattr(bm, "run_benchmark", fake)
        assert main(["--benchmark", "synthetic", "--benchmark-frames", "50"]) == 0
        out = capsys.readouterr().out
        assert "dry-run: no desktop input can be sent" in out and "SUMMARY TEXT" in out
        assert seen["config"].benchmark == "synthetic" and seen["config"].benchmark_frames == 50

    def test_failure_is_reported_with_exit_one(self, monkeypatch, capsys):
        import nocursor.benchmark as bm

        def fake(config):
            raise bm.BenchmarkError("could not open camera 0")

        monkeypatch.setattr(bm, "run_benchmark", fake)
        assert main(["--benchmark", "live"]) == 1
        assert "benchmark failed: could not open camera 0" in capsys.readouterr().err

    def test_live_run_tells_the_user_what_to_do_and_how_to_stop(self, monkeypatch, capsys):
        import nocursor.benchmark as bm

        monkeypatch.setattr(bm, "run_benchmark", lambda c: bm.BenchmarkResult({}, "S", None))
        main(["--benchmark", "live", "--benchmark-seconds", "12"])
        out = capsys.readouterr().out
        assert "12 seconds" in out and "Esc/Q" in out and "ctrl+alt+q" in out

    def test_synthetic_run_says_it_is_not_webcam_performance(self, monkeypatch, capsys):
        import nocursor.benchmark as bm

        monkeypatch.setattr(bm, "run_benchmark", lambda c: bm.BenchmarkResult({}, "S", None))
        main(["--benchmark", "synthetic"])
        assert "not your webcam" in capsys.readouterr().out


class TestLoggingFromTheCommandLine:
    def test_log_file_is_created_only_when_asked_and_gets_startup_records(
        self, monkeypatch, tmp_path
    ):
        import nocursor.cli as cli

        def fake_run(config):
            logging.getLogger("nocursor.app").info("control loop starting: DRY-RUN",
                                                   extra={"file_only": True})
            return 0

        monkeypatch.setattr(cli, "run", fake_run)
        target = tmp_path / "logs" / "n.log"
        assert main(["--dry-run", "--log-file", str(target)]) == 0
        assert "control loop starting" in target.read_text(encoding="utf-8")

    def test_no_file_without_the_flag(self, monkeypatch, tmp_path):
        import nocursor.cli as cli

        monkeypatch.setattr(cli, "run", lambda c: 0)
        monkeypatch.chdir(tmp_path)
        main(["--dry-run"])
        assert list(tmp_path.iterdir()) == []

    def test_handlers_are_removed_when_main_returns(self, monkeypatch):
        import nocursor.cli as cli

        monkeypatch.setattr(cli, "run", lambda c: 0)
        main(["--dry-run"])
        ours = [h for h in logging.getLogger("nocursor").handlers
                if getattr(h, "_nocursor_handler", False)]
        assert ours == []

    def test_diagnostics_flag_reaches_the_console_at_debug(self, monkeypatch, capsys):
        import nocursor.cli as cli

        def fake_run(config):
            logging.getLogger("nocursor.diagnostics").debug("diag ratio=0.2")
            logging.getLogger("nocursor.app").debug("plain debug")
            return 0

        monkeypatch.setattr(cli, "run", fake_run)
        main(["--dry-run", "--diagnostics"])
        err = capsys.readouterr().err
        assert "diag ratio=0.2" in err and "plain debug" not in err

    def test_verbose_shows_debug(self, monkeypatch, capsys):
        import nocursor.cli as cli

        def fake_run(config):
            logging.getLogger("nocursor.x").debug("deep")
            return 0

        monkeypatch.setattr(cli, "run", fake_run)
        main(["--dry-run", "-v"])
        assert "deep" in capsys.readouterr().err

    def test_the_startup_banner_mentions_the_log_file(self, monkeypatch, tmp_path, capsys):
        import nocursor.app
        import nocursor.controller
        import nocursor.emergency
        import nocursor.hand_tracker
        import nocursor.preview

        class B:
            def screen_size(self):
                return (1920, 1080)

        monkeypatch.setattr(nocursor.controller, "PyAutoGUIBackend", B)
        monkeypatch.setattr(nocursor.hand_tracker, "HandTracker", lambda *a, **k: object())
        monkeypatch.setattr(nocursor.hand_tracker, "resolve_model_path", lambda _p: tmp_path)
        monkeypatch.setattr(nocursor.preview, "OpenCVPreview", lambda _m: object())
        monkeypatch.setattr(nocursor.emergency, "system_key_state", lambda: None)
        monkeypatch.setattr(nocursor.app.NoCursorApp, "run", lambda self: 0)
        log = tmp_path / "n.log"
        main(["--dry-run", "--calibration-file", str(tmp_path / "c.json"), "--log-file", str(log)])
        assert f"Logging to: {log}" in capsys.readouterr().out


def test_path_type_is_used_for_path_options():
    args = vars(build_parser().parse_args(["--model", "m.task", "--benchmark-output", "r.json"]))
    assert isinstance(args["model_path"], Path) and isinstance(args["benchmark_output"], Path)
