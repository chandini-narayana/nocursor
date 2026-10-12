from pathlib import Path

import numpy as np
import pytest

from nocursor.camera import Camera, CameraError
from nocursor.cli import build_parser, main
from nocursor.config import Config, ConfigError
from nocursor.hand_tracker import ModelNotFoundError, resolve_model_path


class FakeCapture:
    def __init__(self, opened=True, frames=None):
        self._opened = opened
        self.frames = list(frames or [])
        self.released = 0
        self.props: dict[int, float] = {}

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def read(self):
        if self.frames:
            frame = self.frames.pop(0)
            return (frame is not None), frame
        return False, None

    def release(self):
        self.released += 1


FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


class TestCamera:
    def test_open_failure_raises_clear_error_and_releases(self):
        cap = FakeCapture(opened=False)
        with pytest.raises(CameraError, match="could not open camera 2"):
            Camera(index=2, capture_factory=lambda _i: cap).open()
        assert cap.released == 1

    def test_factory_exception_becomes_camera_error(self):
        def factory(_i):
            raise RuntimeError("driver exploded")

        with pytest.raises(CameraError, match="driver exploded"):
            Camera(capture_factory=factory).open()

    def test_read_returns_frames(self):
        cam = Camera(capture_factory=lambda _i: FakeCapture(frames=[FRAME]))
        cam.open()
        assert cam.read() is FRAME

    def test_read_before_open_raises(self):
        with pytest.raises(CameraError, match="not open"):
            Camera(capture_factory=lambda _i: FakeCapture()).read()

    def test_transient_failures_return_none_then_recover(self):
        cap = FakeCapture(frames=[None, None, FRAME])
        cam = Camera(max_read_failures=5, capture_factory=lambda _i: cap)
        cam.open()
        assert cam.read() is None and cam.read() is None
        assert cam.read() is FRAME

    def test_persistent_failure_raises(self):
        cam = Camera(max_read_failures=3, capture_factory=lambda _i: FakeCapture())
        cam.open()
        assert cam.read() is None and cam.read() is None
        with pytest.raises(CameraError, match="stopped delivering"):
            cam.read()

    def test_failure_counter_resets_after_success(self):
        cap = FakeCapture(frames=[None, None, FRAME, None, None, FRAME])
        cam = Camera(max_read_failures=3, capture_factory=lambda _i: cap)
        cam.open()
        results = [cam.read() for _ in range(6)]
        assert results[2] is FRAME and results[5] is FRAME

    def test_release_is_idempotent(self):
        cap = FakeCapture()
        cam = Camera(capture_factory=lambda _i: cap)
        cam.open()
        cam.release()
        cam.release()
        assert cap.released == 1

    def test_context_manager_releases_on_exception(self):
        cap = FakeCapture()
        with pytest.raises(ValueError), Camera(capture_factory=lambda _i: cap):
            raise ValueError("boom")
        assert cap.released == 1


class TestConfig:
    def test_defaults_are_valid(self):
        Config()

    @pytest.mark.parametrize("field,value", [
        ("camera_index", -1), ("frame_width", 0), ("frame_height", -5),
        ("max_read_failures", 0), ("min_confidence", 1.5), ("min_hand_size", -0.1),
        ("stable_frames", 0), ("active_margin", 0.5), ("active_margin", -0.01),
        ("smoothing", 1.0), ("smoothing", -0.1), ("pinch_threshold", 0),
        ("pinch_release_factor", 1.0), ("pinch_confirm_frames", 0),
        ("click_cooldown_s", -1), ("emergency_hotkey", ""), ("emergency_hotkey", "ctrl+alt"),
    ])
    def test_invalid_values_rejected(self, field, value):
        with pytest.raises(ConfigError):
            Config(**{field: value})

    def test_from_overrides_ignores_none_and_rejects_unknown(self):
        assert Config.from_overrides(smoothing=None, pinch_threshold=0.3).pinch_threshold == 0.3
        assert Config.from_overrides(smoothing=None).smoothing == Config().smoothing
        with pytest.raises(ConfigError):
            Config.from_overrides(bogus=1)


class TestCli:
    def test_parser_maps_flags_to_config_fields(self):
        args = build_parser().parse_args(
            ["--dry-run", "--smoothing", "0.7", "--pinch-threshold", "0.3", "--camera", "1"]
        )
        config = Config.from_overrides(**{k: v for k, v in vars(args).items() if k != "verbose"})
        assert (config.dry_run, config.smoothing, config.pinch_threshold, config.camera_index) == (
            True, 0.7, 0.3, 1)

    def test_invalid_value_exits_with_usage_error_before_touching_hardware(self, capsys):
        assert main(["--smoothing", "5"]) == 64
        assert "invalid configuration" in capsys.readouterr().err


class TestResolveModelPath:
    def test_explicit_existing_path_is_used(self, tmp_path):
        model = tmp_path / "m.task"
        model.write_bytes(b"x")
        assert resolve_model_path(model) == model

    def test_explicit_missing_path_raises(self, tmp_path):
        with pytest.raises(ModelNotFoundError):
            resolve_model_path(tmp_path / "missing.task")

    def test_missing_default_explains_how_to_download(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        try:
            found = resolve_model_path(None)
        except ModelNotFoundError as exc:
            assert "nocursor --download-model" in str(exc)
        else:  # a real model in the project's models/ folder is also acceptable
            assert isinstance(found, Path) and found.is_file()
