"""Benchmarks: modes, dry-run guarantee, measurement boundaries, reports. No webcam, no input."""

import getpass
import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from nocursor.benchmark import (
    COMMON_NOT_MEASURED,
    MODE_CAVEATS,
    MODE_TITLES,
    SCHEMA_VERSION,
    SYNTHETIC_DEFAULT_FRAMES,
    WARMUP_FRAMES,
    BenchmarkError,
    run_benchmark,
)
from nocursor.config import Config, ConfigError
from nocursor.controller import InputBlocked, SafeController, ScreenSizeOnlyBackend
from nocursor.emergency import EmergencyStop
from nocursor.profiling import FrameProfiler
from nocursor.resources import MemoryInfo, ResourceSampler
from nocursor.runtime import StartupError

MS = 1_000_000


class TickNs:
    def __init__(self, step_ms=1):
        self.t = 0
        self.step = int(step_ms * MS)

    def __call__(self):
        self.t += self.step
        return self.t


class FakeTracker:
    """Returns no hand, checks timestamps rise, remembers how it was used."""

    def __init__(self, hand_every=None):
        self.calls = 0
        self.closed = 0
        self.last_ts = -1
        self.hand_every = hand_every
        self.frame_shapes = set()

    def detect(self, frame, timestamp_ms):
        assert timestamp_ms > self.last_ts, "timestamps must increase"
        self.last_ts = timestamp_ms
        self.calls += 1
        self.frame_shapes.add(frame.shape)
        if self.hand_every and self.calls % self.hand_every == 0:
            from nocursor.synthetic import build_hand

            return build_hand()
        return None

    def close(self):
        self.closed += 1


def fake_resources():
    wall = {"t": 0.0}

    def tick():
        wall["t"] += 0.25
        return wall["t"]

    return ResourceSampler(wall_clock=tick, cpu_clock=lambda: wall["t"] * 0.5,
                           memory=lambda: MemoryInfo(50_000_000, 80_000_000), logical_cpus=4)


def synthetic(config_kwargs=None, frames=200, step_ms=1, tracker=None, **kw):
    tracker = tracker or FakeTracker()
    config = Config(benchmark="synthetic", benchmark_frames=frames, **(config_kwargs or {}))
    result = run_benchmark(
        config, tracker_factory=lambda _c: tracker, write=kw.pop("write", False),
        profiler=FrameProfiler(clock_ns=TickNs(step_ms), keep_session=True),
        resources=fake_resources(), **kw,
    )
    return result, tracker


class TestSyntheticMode:
    def test_runs_the_requested_frames_after_discarding_warmup(self):
        result, tracker = synthetic(frames=120)
        r = result.report["results"]
        assert r["frames"] == 120 and result.report["mode"] == "synthetic"
        assert tracker.calls == 120 + WARMUP_FRAMES  # warm-up inferences ran but were not timed
        assert r["stages_ms"]["inference"]["n"] == 120
        assert tracker.closed == 1

    def test_default_frame_count(self):
        result, _ = synthetic(frames=None)
        assert result.report["results"]["frames"] == SYNTHETIC_DEFAULT_FRAMES

    def test_inference_uses_a_frame_of_the_configured_size(self):
        _, tracker = synthetic(config_kwargs={"frame_width": 320, "frame_height": 240}, frames=5)
        assert tracker.frame_shapes == {(240, 320, 3)}

    def test_the_scripted_session_exercises_presses_scrolls_and_tracking_loss(self):
        result, _ = synthetic(frames=400)
        r = result.report["results"]
        assert r["end_to_end_action_frames_ms"]["n"] > 10  # presses, releases and scrolls
        assert r["tracking"]["loss_events"] >= 2
        assert r["tracking"]["frames_by_state"]["tracking"] > 100

    def test_omits_stages_that_do_not_exist_without_a_camera_or_preview(self):
        stages = synthetic(frames=30)[0].report["results"]["stages_ms"]
        assert "frame_wait" not in stages and "preview" not in stages
        assert {"inference", "gesture", "dispatch"} <= set(stages)

    def test_a_hand_free_tracker_fills_only_the_no_hand_series(self):
        stages = synthetic(frames=40)[0].report["results"]["stages_ms"]
        assert stages["inference_hand"] is None
        assert stages["inference_no_hand"]["n"] == 40

    def test_inference_series_are_split_by_the_tracker_result(self):
        stages = synthetic(frames=40, tracker=FakeTracker(hand_every=2))[0] \
            .report["results"]["stages_ms"]
        assert stages["inference_hand"]["n"] + stages["inference_no_hand"]["n"] == 40
        assert stages["inference_hand"]["n"] == 20  # calls 11..50: every second one finds a hand

    def test_it_is_repeatable_for_the_same_clock(self):
        a = synthetic(frames=300)[0].report
        b = synthetic(frames=300)[0].report
        assert a["results"] == b["results"] and a["conditions"] == b["conditions"]

    def test_failure_during_the_run_still_closes_the_tracker(self):
        class Exploding(FakeTracker):
            def detect(self, frame, ts):
                if self.calls > WARMUP_FRAMES + 3:
                    raise RuntimeError("inference crashed")
                return super().detect(frame, ts)

        tracker = Exploding()
        with pytest.raises(RuntimeError, match="inference crashed"):
            synthetic(frames=50, tracker=tracker)
        assert tracker.closed == 1


class TestMeasurementIsMeasuredNotHardcoded:
    def test_reported_latency_follows_the_clock(self):
        slow = synthetic(frames=60, step_ms=2)[0].report["results"]
        fast = synthetic(frames=60, step_ms=1)[0].report["results"]
        # the loop reads the clock 7 times between arrival and end of dispatch
        assert fast["end_to_end_software_ms"]["mean"] == pytest.approx(7.0)
        assert slow["end_to_end_software_ms"]["mean"] == pytest.approx(14.0)
        assert slow["stages_ms"]["inference"]["mean"] == pytest.approx(2.0)
        assert slow["frame_rate"]["frames_per_second"] != fast["frame_rate"]["frames_per_second"]

    def test_stage_boundaries_in_the_synthetic_loop(self):
        r = synthetic(frames=40)[0].report["results"]
        for stage in ("inference", "gesture", "dispatch"):
            assert r["stages_ms"][stage]["mean"] == pytest.approx(1.0)
        assert r["end_to_end_software_ms"]["p50"] == pytest.approx(7.0)

    def test_percentiles_appear_only_with_enough_samples(self):
        few = synthetic(frames=15)[0].report["results"]["end_to_end_software_ms"]
        some = synthetic(frames=50)[0].report["results"]["end_to_end_software_ms"]
        many = synthetic(frames=150)[0].report["results"]["end_to_end_software_ms"]
        assert few["p95"] is None and few["p99"] is None
        assert some["p95"] is not None and some["p99"] is None
        assert many["p95"] is not None and many["p99"] is not None

    def test_cpu_figures_come_from_the_sampler_with_documented_normalisation(self):
        r = synthetic(frames=40)[0].report
        usage = r["results"]["cpu_and_memory"]
        assert usage["cpu_percent_of_one_core"] == pytest.approx(50.0)
        assert usage["cpu_percent_of_machine"] == pytest.approx(12.5)  # 50% / 4 logical CPUs
        assert usage["peak_rss_bytes"] == 80_000_000
        assert "delta" in r["cpu_normalization"] and "logical_cpus" in r["cpu_normalization"]

    def test_the_method_text_denies_camera_to_display_latency(self):
        method = synthetic(frames=10)[0].report["measurement_method"]
        assert "not camera-to-display" in method and "software" in method


class TestDryRunByConstruction:
    def test_scripted_input_never_reaches_a_real_backend(self, monkeypatch):
        import nocursor.controller as controller

        def forbidden(*_a, **_k):
            raise AssertionError("PyAutoGUI backend must not be created by the synthetic benchmark")

        monkeypatch.setattr(controller, "PyAutoGUIBackend", forbidden)
        result, _ = synthetic(frames=250)
        cond = result.report["conditions"]
        assert cond["input_sent"] is False and cond["input_events_sent"] == 0
        assert cond["input_events_simulated"] > 10  # the dry-run path really did run
        assert cond["controller_enabled"] is False
        assert result.report["configuration"]["dry_run"] is True

    def test_dry_run_is_forced_even_if_the_config_says_otherwise(self):
        result, _ = synthetic(config_kwargs={"dry_run": False}, frames=20)
        assert result.report["configuration"]["dry_run"] is True

    def test_backend_blocks_every_input_method(self):
        backend = ScreenSizeOnlyBackend(screen=(800, 600))
        assert backend.screen_size() == (800, 600)
        for call in (lambda: backend.move_to(1, 1), backend.click, backend.mouse_down,
                     backend.mouse_up, lambda: backend.scroll(1)):
            with pytest.raises(InputBlocked):
                call()

    def test_even_an_enabled_controller_cannot_send_through_it(self):
        stop = EmergencyStop()
        controller = SafeController(ScreenSizeOnlyBackend(screen=(800, 600)), stop, enabled=True)
        assert controller.mouse_down() is False and controller.move_to(5, 5) is False
        assert stop.is_set()  # the blocked call tripped the emergency stop
        assert controller.moves_sent == 0 and controller.mouse_downs_sent == 0

    def test_screen_size_can_be_read_from_a_source_backend(self):
        class Source:
            def screen_size(self):
                return (1366, 768)

        assert ScreenSizeOnlyBackend(source=Source()).screen_size() == (1366, 768)

    def test_a_backend_needs_a_size_or_a_source(self):
        with pytest.raises(ValueError):
            ScreenSizeOnlyBackend()

    def test_the_real_runtime_builder_blocks_input_in_benchmark_mode(self, monkeypatch, tmp_path):
        import nocursor.controller
        import nocursor.emergency
        import nocursor.hand_tracker
        import nocursor.preview
        from nocursor.runtime import build_runtime

        class Real:
            def screen_size(self):
                return (1920, 1080)

            def mouse_down(self):
                raise AssertionError("the real backend must never be used for input")

            move_to = click = mouse_up = scroll = mouse_down

        class FakeTrackerCls:
            def __init__(self, *_a, **_k):
                pass

        monkeypatch.setattr(nocursor.controller, "PyAutoGUIBackend", Real)
        monkeypatch.setattr(nocursor.hand_tracker, "HandTracker", FakeTrackerCls)
        monkeypatch.setattr(nocursor.hand_tracker, "resolve_model_path", lambda _p: tmp_path)
        monkeypatch.setattr(nocursor.preview, "OpenCVPreview", lambda _m: object())
        monkeypatch.setattr(nocursor.emergency, "system_key_state", lambda: None)
        runtime = build_runtime(Config(calibration_path=tmp_path / "c.json"), input_blocked=True)
        assert runtime.controller.enabled is False
        assert runtime.controller.mouse_down() is False  # simulated at most, never sent
        with pytest.raises(InputBlocked):
            runtime.controller._backend.mouse_down()


class TestVideoMode:
    class FakeCapture:
        def __init__(self, frames=30, opened=True):
            self.remaining = frames
            self.opened = opened
            self.released = 0

        def isOpened(self):
            return self.opened

        def read(self):
            if self.remaining <= 0:
                return False, None
            self.remaining -= 1
            return True, np.zeros((48, 64, 3), np.uint8)

        def get(self, prop):
            import cv2

            return {cv2.CAP_PROP_FPS: 25.0, cv2.CAP_PROP_FRAME_COUNT: 30.0}.get(prop, 0.0)

        def release(self):
            self.released += 1

    def run_video(self, tmp_path, capture=None, frames=None, tracker=None, **kw):
        clip = tmp_path / "my_private_clip.mp4"
        clip.write_bytes(b"not really a video")
        capture = capture or self.FakeCapture()
        config = Config(benchmark="video", benchmark_video=clip, benchmark_frames=frames)
        result = run_benchmark(
            config, tracker_factory=lambda _c: tracker or FakeTracker(hand_every=3),
            video_opener=lambda _p: capture, write=False,
            profiler=FrameProfiler(clock_ns=TickNs(), keep_session=True),
            resources=fake_resources(), **kw,
        )
        return result, capture

    def test_processes_every_frame_until_the_clip_ends(self, tmp_path):
        result, capture = self.run_video(tmp_path)
        r = result.report["results"]
        assert r["frames"] == 30 and capture.released == 1
        assert result.report["mode"] == "video"
        assert result.report["conditions"]["source_fps"] == 25.0
        assert result.report["conditions"]["frame_size"] == "64x48"

    def test_frame_limit_is_respected(self, tmp_path):
        result, _ = self.run_video(tmp_path, frames=12)
        assert result.report["results"]["frames"] == 12

    def test_hand_frames_are_counted_and_inference_is_split(self, tmp_path):
        result, _ = self.run_video(tmp_path)
        stages = result.report["results"]["stages_ms"]
        assert result.report["conditions"]["hand_present_frames"] == 10
        assert stages["inference_hand"]["n"] == 10 and stages["inference_no_hand"]["n"] == 20

    def test_only_the_file_name_is_recorded_not_its_directory(self, tmp_path):
        result, _ = self.run_video(tmp_path)
        assert result.report["conditions"]["video_file"] == "my_private_clip.mp4"
        text = json.dumps(result.report)
        # JSON escapes backslashes, so compare the distinctive folder name, not the full path
        stored = result.report["conditions"]["video_file"]
        assert tmp_path.name not in text and "/" not in stored and chr(92) not in stored

    def test_video_report_has_decode_wait_but_no_preview(self, tmp_path):
        stages = self.run_video(tmp_path)[0].report["results"]["stages_ms"]
        assert "frame_wait" in stages and "preview" not in stages

    def test_a_missing_file_is_an_actionable_error(self, tmp_path):
        config = Config(benchmark="video", benchmark_video=tmp_path / "nope.mp4")
        with pytest.raises(BenchmarkError, match="video file not found: nope.mp4"):
            run_benchmark(config, tracker_factory=lambda _c: FakeTracker(), write=False)

    def test_an_unreadable_clip_is_an_actionable_error_and_is_released(self, tmp_path):
        capture = self.FakeCapture(opened=False)
        with pytest.raises(BenchmarkError, match="could not open the video"):
            self.run_video(tmp_path, capture=capture)
        assert capture.released == 1

    def test_an_empty_clip_reports_nothing_rather_than_zeros(self, tmp_path):
        with pytest.raises(BenchmarkError, match="no frames were processed"):
            self.run_video(tmp_path, capture=self.FakeCapture(frames=0))

    def test_the_tracker_is_closed(self, tmp_path):
        tracker = FakeTracker()
        self.run_video(tmp_path, tracker=tracker)
        assert tracker.closed == 1

    def test_video_replay_never_creates_a_real_input_backend(self, tmp_path, monkeypatch):
        import nocursor.controller as controller

        monkeypatch.setattr(controller, "PyAutoGUIBackend",
                            lambda: (_ for _ in ()).throw(AssertionError("no real backend")))
        result, _ = self.run_video(tmp_path)
        cond = result.report["conditions"]
        assert cond["input_sent"] is False and cond["input_events_sent"] == 0
        assert cond["controller_enabled"] is False


class TestLiveMode:
    class FakeRuntime:
        def __init__(self, profiler, ticks, frames=40, reason=None, controller=None):
            self.stop = EmergencyStop()
            if reason:
                self.stop.trigger(reason)
            self.profiler, self.ticks, self.frames = profiler, ticks, frames
            self.controller = controller or SafeController(
                ScreenSizeOnlyBackend(screen=(800, 600)), self.stop, enabled=False
            )
            self.app = self

        frame_size = (640, 480)

        def run(self):
            for _ in range(self.frames):
                self.profiler.begin_wait()
                self.profiler.frame_arrived()
                self.profiler.note_tracking("tracking")
                self.profiler.end_dispatch(False)
            self.controller.mouse_down()  # a dry-run press and release, like a real session
            self.controller.mouse_up()
            return 2 if self.stop.is_set() else 0

    def run_live(self, frames=40, reason=None, controller=None, **config_kwargs):
        seen = {}

        def builder(config, **kwargs):
            seen["config"], seen["kwargs"] = config, kwargs
            return self.FakeRuntime(kwargs["profiler"], None, frames, reason, controller)

        config = Config(benchmark="live", benchmark_seconds=12.5, **config_kwargs)
        result = run_benchmark(
            config, runtime_builder=builder, write=False,
            profiler=FrameProfiler(clock_ns=TickNs(), keep_session=True),
            resources=fake_resources(),
        )
        return result, seen

    def test_uses_the_real_app_with_input_blocked_and_the_requested_time_limit(self):
        result, seen = self.run_live()
        assert seen["kwargs"]["input_blocked"] is True
        assert seen["kwargs"]["time_limit_s"] == 12.5
        assert seen["config"].dry_run is True
        cond = result.report["conditions"]
        assert cond["input_sent"] is False and cond["input_events_sent"] == 0
        assert cond["input_events_simulated"] == 2  # the press and the release, simulated

    def test_input_sent_is_derived_from_the_controller_not_assumed(self):
        class Recorder:
            def screen_size(self):
                return (800, 600)

            def mouse_down(self):
                pass

            mouse_up = move_to = click = scroll = mouse_down

        stop = EmergencyStop()
        enabled = SafeController(Recorder(), stop, enabled=True)  # deliberately NOT dry-run
        result, _ = self.run_live(controller=enabled)
        cond = result.report["conditions"]
        assert cond["input_sent"] is True and cond["input_events_sent"] == 2
        assert cond["controller_enabled"] is True  # the report would expose such a mistake

    def test_reports_how_it_ended(self):
        assert self.run_live()[0].report["conditions"]["ended_by"] == "time limit"
        ended = self.run_live(reason="emergency hotkey pressed")[0]
        assert ended.report["conditions"]["ended_by"] == "emergency hotkey pressed"

    def test_records_camera_pacing_and_preview_as_part_of_the_conditions(self):
        cond = self.run_live()[0].report["conditions"]
        assert cond["paced_by_camera"] is True and cond["preview"] is True
        assert cond["frame_size"] == "640x480" and cond["requested_resolution"] == "640x480"

    def test_startup_failures_become_benchmark_errors(self):
        def builder(config, **kwargs):
            raise StartupError("could not open camera 0")

        config = Config(benchmark="live")
        with pytest.raises(BenchmarkError, match="could not open camera 0"):
            run_benchmark(config, runtime_builder=builder, write=False)

    def test_a_live_run_that_failed_to_start_says_so(self):
        class Failed(self.FakeRuntime):
            def run(self):
                return 1  # e.g. the camera could not be opened: exit code 1, no frames

        def builder(config, **kwargs):
            return Failed(kwargs["profiler"], None, 0)

        with pytest.raises(BenchmarkError, match="failed before any frame was processed"):
            run_benchmark(Config(benchmark="live"), runtime_builder=builder, write=False,
                          profiler=FrameProfiler(clock_ns=TickNs(), keep_session=True))

    def test_zero_frames_is_an_error_not_an_empty_report(self):
        with pytest.raises(BenchmarkError, match="no frames"):
            self.run_live(frames=0)


class TestBenchmarkDoesNotPerturbItsOwnTiming:
    def test_the_benchmark_controller_narrates_at_debug_not_info(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="nocursor.controller"):
            synthetic(frames=250)
        simulated = [r for r in caplog.records if "dry-run" in r.getMessage()]
        assert simulated and all(r.levelno == logging.DEBUG for r in simulated)

    def test_a_normal_dry_run_controller_still_narrates_at_info(self, caplog):
        controller = SafeController(ScreenSizeOnlyBackend(screen=(800, 600)), EmergencyStop(),
                                    enabled=False)
        with caplog.at_level(logging.DEBUG, logger="nocursor.controller"):
            controller.mouse_down()
            controller.mouse_up()
            controller.scroll(2)
        levels = {r.levelno for r in caplog.records if "dry-run" in r.getMessage()}
        assert levels == {logging.INFO}

    def test_no_console_lines_are_written_during_a_synthetic_run(self, capsys):
        from nocursor.logging_setup import configure_logging

        handle = configure_logging()
        try:
            synthetic(frames=250)
        finally:
            handle.close()
        assert "dry-run" not in capsys.readouterr().err

    def test_quiet_runtime_controller_in_benchmark_mode(self, monkeypatch, tmp_path):
        import nocursor.controller
        import nocursor.emergency
        import nocursor.hand_tracker
        import nocursor.preview
        from nocursor.runtime import build_runtime

        class Real:
            def screen_size(self):
                return (1920, 1080)

        monkeypatch.setattr(nocursor.controller, "PyAutoGUIBackend", Real)
        monkeypatch.setattr(nocursor.hand_tracker, "HandTracker", lambda *a, **k: object())
        monkeypatch.setattr(nocursor.hand_tracker, "resolve_model_path", lambda _p: tmp_path)
        monkeypatch.setattr(nocursor.preview, "OpenCVPreview", lambda _m: object())
        monkeypatch.setattr(nocursor.emergency, "system_key_state", lambda: None)
        cfg = Config(calibration_path=tmp_path / "c.json")
        assert build_runtime(cfg, input_blocked=True).controller._quiet is True
        assert build_runtime(cfg).controller._quiet is False


class TestModesStaySeparate:
    def test_each_mode_has_its_own_title_and_caveats(self):
        assert len(set(MODE_TITLES.values())) == 3
        assert len({tuple(c) for c in MODE_CAVEATS.values()}) == 3
        assert "not webcam performance" in MODE_TITLES["synthetic"]
        assert "not webcam performance" in MODE_TITLES["video"]
        assert "LIVE WEBCAM" in MODE_TITLES["live"]

    def test_the_report_names_its_mode_and_never_borrows_another_modes_caveats(self):
        syn = synthetic(frames=20)[0].report
        assert syn["mode"] == "synthetic" and syn["caveats"] == list(MODE_CAVEATS["synthetic"])
        assert not any("camera to deliver" in c for c in syn["caveats"])

    def test_synthetic_and_live_do_not_claim_the_same_frame_rate_meaning(self):
        syn = synthetic(frames=20)[0].report["results"]["frame_rate"]["meaning"]
        live = TestLiveMode().run_live()[0].report["results"]["frame_rate"]["meaning"]
        assert "no camera pacing" in syn and "camera" in live and syn != live

    def test_synthetic_caveat_does_not_claim_a_direction_for_real_hands(self):
        text = " ".join(MODE_CAVEATS["synthetic"]).lower()
        assert "typically slower" not in text and "no direction is claimed" in text

    def test_files_are_named_by_mode(self, tmp_path, monkeypatch):
        import nocursor.benchmark as bm

        monkeypatch.setattr(bm, "default_benchmark_dir", lambda: tmp_path)
        synthetic(frames=10, write=True)
        names = [p.name for p in tmp_path.iterdir()]
        assert len(names) == 1 and names[0].startswith("synthetic-") and names[0].endswith(".json")


class TestReports:
    def test_schema_and_required_sections(self):
        report = synthetic(frames=30)[0].report
        for key in ("schema_version", "tool", "mode", "generated_utc", "measurement_method",
                    "conditions", "configuration", "environment", "results", "caveats",
                    "not_measured", "cpu_normalization"):
            assert key in report
        assert report["schema_version"] == SCHEMA_VERSION
        assert report["not_measured"] == list(COMMON_NOT_MEASURED)

    def test_configuration_snapshot_records_the_pinch_mode_and_thresholds(self):
        cfg = synthetic(frames=10)[0].report["configuration"]
        assert cfg["pinch_mode"] == "2D (default)" and cfg["pinch_threshold"] == 0.20
        cfg3 = synthetic(config_kwargs={"pinch_use_depth": True}, frames=10)[0] \
            .report["configuration"]
        assert cfg3["pinch_mode"] == "3D (experimental)"

    def test_environment_describes_the_machine_without_identifying_the_user(self):
        report = synthetic(frames=10)[0].report
        env = report["environment"]
        assert env["python"] and env["logical_cpus"] and "numpy" in env["packages"]
        text = json.dumps(report)
        assert str(Path.home()) not in text
        for private in {getpass.getuser(), os.environ.get("COMPUTERNAME", "")} - {""}:
            assert private.lower() not in text.lower() or len(private) < 4

    def test_written_report_is_valid_json_and_complete(self, tmp_path):
        out = tmp_path / "deep" / "dir" / "r.json"
        config = Config(benchmark="synthetic", benchmark_frames=25, benchmark_output=out)
        result = run_benchmark(
            config, tracker_factory=lambda _c: FakeTracker(),
            profiler=FrameProfiler(clock_ns=TickNs(), keep_session=True),
            resources=fake_resources(),
        )
        assert result.path == out and out.is_file()
        assert json.loads(out.read_text(encoding="utf-8")) == json.loads(json.dumps(result.report))
        assert not list(out.parent.glob("*.tmp"))
        assert str(out) in result.summary

    def test_no_file_when_writing_is_disabled(self, tmp_path):
        out = tmp_path / "x.json"
        config = Config(benchmark="synthetic", benchmark_frames=10, benchmark_output=out)
        result = run_benchmark(config, tracker_factory=lambda _c: FakeTracker(), write=False)
        assert result.path is None and not out.exists()

    def test_unwritable_output_is_an_actionable_error(self, tmp_path):
        blocker = tmp_path / "blocked"
        blocker.write_text("a file")
        config = Config(benchmark="synthetic", benchmark_frames=10,
                        benchmark_output=blocker / "r.json")
        with pytest.raises(BenchmarkError, match="could not write the benchmark report"):
            run_benchmark(config, tracker_factory=lambda _c: FakeTracker())

    def test_summary_text_is_labelled_and_lists_caveats(self):
        summary = synthetic(frames=60)[0].summary
        assert "SYNTHETIC (scripted input; not webcam performance)" in summary
        assert "END-TO-END (software)" in summary and "Caveats:" in summary
        assert "Not measured: camera exposure" in summary and "display" in summary
        assert "n/a" in summary  # p99 is not available with only 60 samples
        assert "of one core" in summary and "logical CPUs" in summary

    def test_missing_dependencies_are_reported_not_traceback(self, monkeypatch):
        import nocursor.benchmark as bm

        monkeypatch.setattr(bm, "missing_dependencies", lambda: [("mediapipe", "inference")])
        with pytest.raises(BenchmarkError, match="missing required package"):
            run_benchmark(Config(benchmark="synthetic"), write=False)

    def test_missing_model_is_an_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import nocursor.hand_tracker as ht

        monkeypatch.setattr(ht, "resolve_model_path", lambda _p: (_ for _ in ()).throw(
            ht.ModelNotFoundError("hand_landmarker.task not found. Download it ...")))
        with pytest.raises(BenchmarkError, match="not found"):
            run_benchmark(Config(benchmark="synthetic"), write=False)

    def test_no_mode_is_an_error(self):
        with pytest.raises(BenchmarkError, match="no benchmark mode"):
            run_benchmark(Config())

    def test_logging_never_receives_frames_or_landmarks(self, caplog):
        with caplog.at_level(logging.DEBUG):
            synthetic(frames=30)
        assert "ndarray" not in caplog.text and "landmarks" not in caplog.text.lower()


class TestBenchmarkConfig:
    @pytest.mark.parametrize("kwargs,message", [
        ({"benchmark": "turbo"}, "must be one of"),
        ({"benchmark": "video"}, "--benchmark-video"),
        ({"benchmark": "synthetic", "benchmark_video": Path("x.mp4")}, "only applies to"),
        ({"benchmark": "live", "benchmark_frames": 10}, "uses --benchmark-seconds"),
        ({"benchmark": "synthetic", "benchmark_seconds": 5.0}, "applies to live runs"),
        ({"benchmark": "synthetic", "benchmark_frames": 0}, "benchmark_frames"),
        ({"benchmark": "synthetic", "benchmark_frames": 2_000_000}, "benchmark_frames"),
        ({"benchmark": "live", "benchmark_seconds": 0}, "benchmark_seconds"),
        ({"benchmark": "live", "benchmark_seconds": 7200}, "benchmark_seconds"),
        ({"benchmark": "synthetic", "start_calibration": True}, "--calibrate"),
        ({"benchmark_frames": 10}, "only apply together with --benchmark"),
        ({"benchmark_output": Path("r.json")}, "only apply together with --benchmark"),
    ])
    def test_invalid_combinations_are_rejected_with_the_flag_to_fix(self, kwargs, message):
        with pytest.raises(ConfigError, match=message):
            Config(**kwargs)

    def test_valid_combinations(self):
        Config(benchmark="synthetic", benchmark_frames=500)
        Config(benchmark="live", benchmark_seconds=30.0)
        Config(benchmark="video", benchmark_video=Path("clip.mp4"), benchmark_frames=100)
        Config()

    def test_benchmark_is_off_by_default(self):
        assert Config().benchmark is None and Config().show_perf is False
        assert Config().log_file is None

