"""Phase 2 configuration and command line: defaults, validation, backward compatibility."""

import pytest

from nocursor.cli import build_parser, main
from nocursor.config import Config, ConfigError

PHASE1_FLAGS = [
    "--dry-run", "--camera", "--width", "--height", "--model", "--smoothing", "--margin",
    "--pinch-threshold", "--pinch-confirm-frames", "--pinch-release-factor", "--click-cooldown",
    "--min-confidence", "--stop-key", "--verbose",
]
PHASE2_FLAGS = [
    "--pinch-release-frames", "--drag-start", "--scroll-stable-frames", "--scroll-dead-zone",
    "--scroll-speed", "--scroll-interval", "--scroll-invert", "--pause-key", "--calibrate",
    "--no-calibration", "--calibration-file", "--reset-calibration",
]


def to_config(argv):
    args = vars(build_parser().parse_args(argv))
    args.pop("verbose")
    args.pop("reset_calibration", None)
    return Config.from_overrides(**args)


class TestDefaults:
    def test_phase1_defaults_are_unchanged(self):
        c = Config()
        assert (c.pinch_threshold, c.pinch_confirm_frames, c.pinch_release_factor) == (0.20, 5, 1.5)
        assert (c.click_cooldown_s, c.smoothing, c.active_margin) == (0.4, 0.5, 0.15)
        assert c.emergency_hotkey == "ctrl+alt+q" and c.dry_run is False

    def test_phase2_defaults(self):
        c = Config()
        assert c.drag_start_px == 30.0
        assert c.pinch_release_frames == 2
        assert (c.scroll_stable_frames, c.scroll_dead_zone) == (6, 0.10)
        assert (c.scroll_speed, c.scroll_interval_s, c.scroll_invert) == (8.0, 0.05, False)
        assert c.pause_hotkey == "ctrl+alt+p"
        assert c.use_calibration is True and c.start_calibration is False
        assert c.calibration_path is None


class TestValidation:
    @pytest.mark.parametrize("field,value,flag", [
        ("pinch_release_frames", 0, "--pinch-release-frames"),
        ("drag_start_px", 0, "--drag-start"),
        ("drag_start_px", -3.0, "--drag-start"),
        ("drag_start_px", 501.0, "--drag-start"),
        ("scroll_stable_frames", 0, "--scroll-stable-frames"),
        ("scroll_dead_zone", 0, "--scroll-dead-zone"),
        ("scroll_dead_zone", 1.5, "--scroll-dead-zone"),
        ("scroll_speed", 0, "--scroll-speed"),
        ("scroll_speed", 101, "--scroll-speed"),
        ("scroll_interval_s", 0.0, "--scroll-interval"),
        ("scroll_interval_s", 5.0, "--scroll-interval"),
    ])
    def test_out_of_range_values_are_rejected_with_the_flag_to_fix(self, field, value, flag):
        with pytest.raises(ConfigError, match=flag):
            Config(**{field: value})

    @pytest.mark.parametrize("value", ["", "ctrl+alt", "nonsense+p"])
    def test_bad_pause_hotkey_rejected(self, value):
        with pytest.raises(ConfigError, match="pause_hotkey"):
            Config(pause_hotkey=value)

    @pytest.mark.parametrize("pause", ["ctrl+alt+q", "alt+ctrl+q", "CTRL+ALT+Q"])
    def test_pause_hotkey_must_differ_from_the_emergency_stop(self, pause):
        with pytest.raises(ConfigError, match="differ"):
            Config(pause_hotkey=pause)

    def test_changing_the_stop_key_to_the_default_pause_key_is_rejected(self):
        with pytest.raises(ConfigError, match="differ"):
            Config(emergency_hotkey="ctrl+alt+p")

    def test_existing_validation_still_applies(self):
        for field, value in [("smoothing", 1.0), ("active_margin", 0.5), ("pinch_threshold", 0)]:
            with pytest.raises(ConfigError):
                Config(**{field: value})


class TestCommandLine:
    def test_every_phase1_option_still_exists(self):
        help_text = build_parser().format_help()
        for flag in PHASE1_FLAGS:
            assert flag in help_text

    def test_every_phase2_option_is_documented_in_help(self):
        help_text = build_parser().format_help()
        for flag in PHASE2_FLAGS:
            assert flag in help_text

    def test_flags_map_to_config_fields(self, tmp_path):
        c = to_config([
            "--drag-start", "45", "--pinch-release-frames", "3", "--scroll-stable-frames", "9",
            "--scroll-dead-zone", "0.2", "--scroll-speed", "4", "--scroll-interval", "0.1",
            "--scroll-invert", "--pause-key", "ctrl+alt+o", "--calibrate", "--no-calibration",
            "--calibration-file", str(tmp_path / "c.json"),
        ])
        assert (c.drag_start_px, c.pinch_release_frames, c.scroll_stable_frames) == (45, 3, 9)
        assert (c.scroll_dead_zone, c.scroll_speed, c.scroll_interval_s) == (0.2, 4, 0.1)
        assert c.scroll_invert and c.start_calibration and not c.use_calibration
        assert c.pause_hotkey == "ctrl+alt+o"
        assert c.calibration_path == tmp_path / "c.json"

    def test_no_flags_gives_the_defaults(self):
        assert to_config([]) == Config()

    @pytest.mark.parametrize("argv,mention", [
        (["--drag-start", "0"], "--drag-start"),
        (["--scroll-speed", "-1"], "--scroll-speed"),
        (["--scroll-dead-zone", "9"], "--scroll-dead-zone"),
        (["--pinch-release-frames", "0"], "--pinch-release-frames"),
        (["--pause-key", "ctrl+alt+q"], "differ"),
        (["--pause-key", "banana"], "pause_hotkey"),
    ])
    def test_invalid_values_exit_with_an_actionable_message(self, argv, mention, capsys):
        assert main(argv) == 64
        err = capsys.readouterr().err
        assert "invalid configuration" in err and mention in err

    def test_reset_calibration_deletes_the_file_and_exits_without_starting_anything(
        self, tmp_path, capsys
    ):
        target = tmp_path / "c.json"
        target.write_text("{}")
        assert main(["--reset-calibration", "--calibration-file", str(target)]) == 0
        assert not target.exists()
        assert "deleted" in capsys.readouterr().out

    def test_reset_calibration_with_no_file_is_harmless(self, tmp_path, capsys):
        assert main(["--reset-calibration", "--calibration-file", str(tmp_path / "x.json")]) == 0
        assert "not found" in capsys.readouterr().out

    def test_reset_calibration_does_not_run_when_the_config_is_invalid(self, tmp_path):
        target = tmp_path / "c.json"
        target.write_text("{}")
        assert main(["--reset-calibration", "--calibration-file", str(target),
                     "--drag-start", "0"]) == 64
        assert target.exists()
