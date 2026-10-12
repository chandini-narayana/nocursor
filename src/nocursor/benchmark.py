"""Software benchmarks: synthetic, video replay and live webcam. All dry-run.

Three modes, deliberately kept apart because they measure different things:

``synthetic``  Fixed, hand-free image + scripted landmarks. Real MediaPipe inference runs on the
               hand-free image (the "no hand" path); the real gesture and controller code run on
               scripted landmarks. Repeatable, but it is NOT webcam performance, and its
               end-to-end figure is a composite of two independent measurements.
``video``      A clip you supply, decoded and processed as fast as possible (unpaced). Real
               inference on real hands, repeatable for the same clip. Decode time stands in for
               the camera, so ``frame_wait`` is decoding, not a webcam.
``live``       The real application loop with your webcam and preview, for a fixed time. Shows
               what you actually get, including camera pacing, but varies with what you do.

Every mode is dry-run by construction: the controller's backend can only report a screen size
(``ScreenSizeOnlyBackend``), so no desktop input can be sent. The timings are software timings
taken with ``time.perf_counter_ns`` inside this process. They exclude camera exposure/USB/driver
delay, the display, and operating-system input handling, so none of them is camera-to-display
latency, and no such claim is made.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nocursor import __version__
from nocursor.config import Config
from nocursor.controller import SafeController, ScreenSizeOnlyBackend
from nocursor.dispatch import apply_actions
from nocursor.emergency import EmergencyStop
from nocursor.environment import collect_environment, missing_dependencies
from nocursor.interaction import MouseDown, MouseUp, Scroll
from nocursor.metrics import P95_MIN_SAMPLES, P99_MIN_SAMPLES, format_ms
from nocursor.paths import default_benchmark_dir
from nocursor.processor import GestureProcessor
from nocursor.profiling import FrameProfiler
from nocursor.resources import ResourceSampler
from nocursor.synthetic import SYNTHETIC_FPS, scripted_hands, synthetic_frame

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SYNTHETIC_DEFAULT_FRAMES = 1500
LIVE_DEFAULT_SECONDS = 30.0
WARMUP_FRAMES = 10  # synthetic only: inferences run and discarded before timing starts
SYNTHETIC_SCREEN = (1920, 1080)  # fixed so coordinates (not timings) are reproducible

STAGE_ORDER = ("frame_wait", "inference", "inference_hand", "inference_no_hand", "gesture",
               "dispatch", "preview")  # fmt: skip

MODE_TITLES = {
    "synthetic": "SYNTHETIC (scripted input; not webcam performance)",
    "video": "VIDEO REPLAY (your clip, decoded unpaced; not webcam performance)",
    "live": "LIVE WEBCAM (dry-run; includes camera pacing and the preview)",
}

COMMON_NOT_MEASURED = (
    "camera exposure, USB transfer and driver delay",
    "display / monitor latency",
    "operating-system input handling after the controller call",
)

MODE_CAVEATS = {
    "synthetic": (
        "MediaPipe inference was timed on a fixed image containing no hand, i.e. the path where "
        "its palm detector runs on every frame. With a tracked hand MediaPipe works differently "
        "(it can reuse the previous landmarks), so hand-present frames may be faster or slower: "
        "use --benchmark video or live for those. No direction is claimed here.",
        "Gesture processing and dispatch were timed on scripted landmarks, independently of the "
        "inference; the end-to-end figure is the sum of those separate measurements per frame.",
        "No camera pacing: the frame rate is the processing throughput this machine could "
        "sustain, not what a 30 fps webcam delivers.",
        f"The first {WARMUP_FRAMES} inferences are discarded as warm-up.",
        "The controller ran in dry-run against a fixed 1920x1080 virtual screen.",
    ),
    "video": (
        "The clip is read and processed as fast as possible with no real-time pacing; "
        "frame_wait is video decoding, not a camera.",
        "Results depend on the clip's content, resolution and codec, and on the first frames "
        "including model warm-up (visible in p99 / max).",
        "Gesture timing uses a virtual clock at the clip's frame rate; no preview is drawn.",
        "The controller ran in dry-run against a fixed 1920x1080 virtual screen.",
    ),
    "live": (
        "frame_wait includes waiting for the camera to deliver the next frame, so the frame "
        "rate is limited by the camera and its exposure settings.",
        "What the numbers show depends on what your hand did during the run (tracking-loss "
        "counts especially); repeat runs will differ.",
        "Preview drawing is timed separately and is not part of the end-to-end latency.",
        "Controller dispatch is dry-run: it measures the controller's own bookkeeping, not an "
        "operating-system call.",
    ),
}


class BenchmarkError(Exception):
    """The benchmark could not run; the message says why and what to do."""


@dataclass
class BenchmarkResult:
    report: dict[str, Any]
    summary: str
    path: Path | None


# ------------------------------------------------------------------- public entry
def run_benchmark(
    config: Config,
    *,
    tracker_factory: Callable[[Config], Any] | None = None,
    video_opener: Callable[[str], Any] | None = None,
    runtime_builder: Callable[..., Any] | None = None,
    write: bool = True,
    profiler: FrameProfiler | None = None,
    resources: ResourceSampler | None = None,
) -> BenchmarkResult:
    """Run the benchmark selected by ``config.benchmark``. Always dry-run; never sends input.

    ``tracker_factory``, ``video_opener``, ``runtime_builder``, ``profiler`` and ``resources``
    exist so tests can substitute fakes; normal use passes none of them.
    """
    if config.benchmark is None:
        raise BenchmarkError("no benchmark mode selected (--benchmark synthetic|video|live)")
    config = replace(config, dry_run=True)  # belt and braces: the backend also blocks input
    profiler = profiler if profiler is not None else FrameProfiler(keep_session=True)
    resources = resources if resources is not None else ResourceSampler()
    started_utc = datetime.now(UTC)

    if config.benchmark == "synthetic":
        conditions = _run_synthetic(config, profiler, resources, tracker_factory)
    elif config.benchmark == "video":
        conditions = _run_video(config, profiler, resources, tracker_factory, video_opener)
    else:
        conditions = _run_live(config, profiler, resources, runtime_builder)

    if profiler.frames == 0:
        raise BenchmarkError("no frames were processed, so there is nothing to report")
    report = build_report(config, conditions, profiler, resources, started_utc)
    path = _write_report(report, config) if write else None
    return BenchmarkResult(report, format_summary(report, path), path)


# --------------------------------------------------------------------- the modes
def _need(modules: tuple[str, ...]) -> None:
    missing = [d for d, _ in missing_dependencies() if d.split("-")[0] in modules or d in modules]
    if missing:
        raise BenchmarkError(
            f"missing required package(s): {', '.join(missing)}. "
            "Run: python -m pip install -r requirements.txt"
        )


def _default_tracker(config: Config) -> Any:
    from nocursor.hand_tracker import HandTracker, ModelNotFoundError, resolve_model_path

    try:
        model = resolve_model_path(config.model_path)
    except ModelNotFoundError as exc:
        raise BenchmarkError(str(exc)) from exc
    try:
        return HandTracker(model, config.min_confidence)
    except Exception as exc:
        raise BenchmarkError(
            f"could not load the hand-landmark model ({exc}); "
            "download it again with: nocursor --redownload-model"
        ) from exc


def _dry_run_controller() -> tuple[SafeController, EmergencyStop]:
    stop = EmergencyStop()
    backend = ScreenSizeOnlyBackend(screen=SYNTHETIC_SCREEN)  # cannot send input at all
    return SafeController(backend, stop, enabled=False, quiet=True), stop


def _input_evidence(controller: SafeController) -> dict[str, Any]:
    """What the controller actually did: derived from its counters, never assumed.

    ``input_events_sent`` counts every move / press / release / click / scroll that reached a
    backend (it must be 0 in a benchmark); ``input_events_simulated`` counts the press,
    release and scroll events it accepted in dry-run, which shows the code path really ran.
    """
    sent = (controller.moves_sent + controller.clicks_sent + controller.mouse_downs_sent
            + controller.mouse_ups_sent + controller.scrolls_sent)  # fmt: skip
    return {
        "input_sent": sent > 0,
        "input_events_sent": sent,
        "input_events_simulated": controller.events_accepted,
        "controller_enabled": controller.enabled,
    }


def _discrete(gesture: Any) -> bool:
    return any(isinstance(a, MouseDown | MouseUp | Scroll) for a in gesture.actions)


def _run_synthetic(
    config: Config,
    profiler: FrameProfiler,
    resources: ResourceSampler,
    tracker_factory: Callable[[Config], Any] | None,
) -> dict[str, Any]:
    _need(("mediapipe", "numpy"))
    frames = config.benchmark_frames or SYNTHETIC_DEFAULT_FRAMES
    tracker = (tracker_factory or _default_tracker)(config)
    controller, _stop = _dry_run_controller()
    processor = GestureProcessor(config, SYNTHETIC_SCREEN, None)
    image = synthetic_frame(config.frame_width, config.frame_height)
    aspect = config.frame_width / config.frame_height
    step_ms = 1000 // SYNTHETIC_FPS
    try:
        for i in range(WARMUP_FRAMES):  # discarded: first inferences initialise the model
            tracker.detect(image, i * step_ms)
        resources.start()
        for i, hand in enumerate(scripted_hands(frames, aspect)):
            profiler.begin_wait()
            profiler.frame_arrived()
            t = profiler.mark()
            found = tracker.detect(image, (WARMUP_FRAMES + i) * step_ms)
            profiler.record_since("inference", t, "hand" if found is not None else "no_hand")
            t = profiler.mark()
            gesture = processor.process(hand, i / SYNTHETIC_FPS)
            profiler.record_since("gesture", t)
            profiler.note_tracking(gesture.tracking.value)
            t = profiler.mark()
            apply_actions(controller, processor, gesture.actions)
            profiler.record_since("dispatch", t)
            profiler.end_dispatch(_discrete(gesture))
            resources.maybe_sample()
    finally:
        tracker.close()
        processor.halt("benchmark finished")
    return {
        "mode": "synthetic",
        "frames_requested": frames,
        "frame_size": f"{config.frame_width}x{config.frame_height}",
        "image": "fixed synthetic gradient with no hand",
        "landmarks": "scripted cycle: pointing, click, drag, two-finger scroll, tracking loss",
        "gesture_clock": f"virtual, {SYNTHETIC_FPS} fps (timings themselves use perf_counter)",
        "warmup_inferences_discarded": WARMUP_FRAMES,
        "screen": f"{SYNTHETIC_SCREEN[0]}x{SYNTHETIC_SCREEN[1]} (fixed, virtual)",
        "camera": False, "preview": False, "paced_by_camera": False,
        "frame_rate_meaning": "processing throughput (no camera pacing)",
        **_input_evidence(controller),
    }  # fmt: skip


def _run_video(
    config: Config,
    profiler: FrameProfiler,
    resources: ResourceSampler,
    tracker_factory: Callable[[Config], Any] | None,
    video_opener: Callable[[str], Any] | None,
) -> dict[str, Any]:
    _need(("cv2", "opencv", "mediapipe", "numpy"))
    path = config.benchmark_video
    assert path is not None
    if not path.is_file():
        raise BenchmarkError(f"video file not found: {path.name}")
    if video_opener is None:
        import cv2

        video_opener = cv2.VideoCapture
    capture = video_opener(str(path))
    if not capture.isOpened():
        capture.release()
        raise BenchmarkError(f"OpenCV could not open the video: {path.name}")
    try:
        import cv2

        source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(SYNTHETIC_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    except Exception:  # a custom opener without OpenCV properties
        source_fps, total = float(SYNTHETIC_FPS), None
    limit = config.benchmark_frames
    tracker = (tracker_factory or _default_tracker)(config)
    controller, _stop = _dry_run_controller()
    processor = GestureProcessor(config, SYNTHETIC_SCREEN, None)
    size = None
    hand_frames = 0
    try:
        resources.start()
        i = 0
        while limit is None or i < limit:
            profiler.begin_wait()
            ok, frame = capture.read()
            if not ok or frame is None:
                break  # end of the clip
            profiler.frame_arrived()
            if size is None:
                size = f"{frame.shape[1]}x{frame.shape[0]}"
            t = profiler.mark()
            observation = tracker.detect(frame, int(i * 1000 / source_fps))
            profiler.record_since(
                "inference", t, "hand" if observation is not None else "no_hand"
            )
            hand_frames += observation is not None
            t = profiler.mark()
            gesture = processor.process(observation, i / source_fps)
            profiler.record_since("gesture", t)
            profiler.note_tracking(gesture.tracking.value)
            t = profiler.mark()
            apply_actions(controller, processor, gesture.actions)
            profiler.record_since("dispatch", t)
            profiler.end_dispatch(_discrete(gesture))
            resources.maybe_sample()
            i += 1
    finally:
        capture.release()
        tracker.close()
        processor.halt("benchmark finished")
    return {
        "mode": "video",
        "video_file": path.name,  # name only: no directory (and so no user name) is recorded
        "video_frames_in_file": total,
        "frames_requested": limit,
        "frame_size": size,
        "source_fps": round(source_fps, 3),
        "gesture_clock": f"virtual, at the clip's {source_fps:.2f} fps",
        "screen": f"{SYNTHETIC_SCREEN[0]}x{SYNTHETIC_SCREEN[1]} (fixed, virtual)",
        "camera": False, "preview": False, "paced_by_camera": False,
        "frame_rate_meaning": "decode + processing throughput (unpaced)",
        "hand_present_frames": hand_frames,
        **_input_evidence(controller),
    }  # fmt: skip


def _run_live(
    config: Config,
    profiler: FrameProfiler,
    resources: ResourceSampler,
    runtime_builder: Callable[..., Any] | None,
) -> dict[str, Any]:
    from nocursor.runtime import StartupError, build_runtime

    seconds = config.benchmark_seconds or LIVE_DEFAULT_SECONDS
    builder = runtime_builder or build_runtime
    try:
        runtime = builder(
            config, input_blocked=True, profiler=profiler, resources=resources,
            time_limit_s=seconds,
        )  # fmt: skip
    except StartupError as exc:
        raise BenchmarkError(str(exc)) from exc
    exit_code = runtime.app.run()
    if exit_code == 1 and profiler.frames == 0:
        raise BenchmarkError(
            "the live run failed before any frame was processed (see the error above; usually "
            "the camera could not be opened). Try --camera N, or close other apps using it."
        )
    ended_by = runtime.stop.reason if runtime.stop.is_set() else "time limit"
    size = getattr(runtime.app, "frame_size", None)
    return {
        "mode": "live",
        "seconds_requested": seconds,
        "ended_by": ended_by,
        "exit_code": exit_code,
        "camera_index": config.camera_index,
        "requested_resolution": f"{config.frame_width}x{config.frame_height}",
        "frame_size": f"{size[0]}x{size[1]}" if size else None,
        "camera": True, "preview": True, "paced_by_camera": True,
        "frame_rate_meaning": "frames delivered by the camera and processed",
        "controller": "dry-run; its backend can only report the screen size",
        **_input_evidence(runtime.controller),
    }  # fmt: skip


# ----------------------------------------------------------------------- reports
def _stat(profiler: FrameProfiler, name: str) -> dict[str, Any] | None:
    summary = profiler.session_summary(name)
    return None if summary is None else summary.as_dict()


def build_report(
    config: Config,
    conditions: dict[str, Any],
    profiler: FrameProfiler,
    resources: ResourceSampler,
    started_utc: datetime,
) -> dict[str, Any]:
    mode = conditions["mode"]
    stages = {name: _stat(profiler, name) for name in STAGE_ORDER}
    if mode == "synthetic":
        stages.pop("frame_wait")  # there is no camera to wait for
        stages.pop("preview")
    if mode == "video":
        stages.pop("preview")
    tracking = profiler.tracking
    resource_summary = resources.summary()
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "nocursor", "version": __version__},
        "mode": mode,
        "mode_title": MODE_TITLES[mode],
        "generated_utc": started_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "measurement_method": (
            "Software timings from time.perf_counter_ns inside this process. end_to_end runs "
            "from the moment a frame arrives (camera.read / video decode returned) to the end "
            "of controller dispatch. It is software processing latency, not camera-to-display "
            "latency. Percentiles are nearest-rank; p95 needs "
            f">= {P95_MIN_SAMPLES} samples and p99 >= {P99_MIN_SAMPLES}, otherwise null."
        ),
        "conditions": conditions,
        "configuration": _config_snapshot(config),
        "environment": collect_environment(),
        "results": {
            "frames": profiler.frames,
            "dropped_frames": profiler.dropped_frames,
            "duration_s": round(profiler.elapsed_s, 3),
            "frame_rate": {
                "frames_per_second": _round(profiler.session_fps()),
                "meaning": conditions["frame_rate_meaning"],
            },
            "stages_ms": stages,
            "end_to_end_software_ms": _stat(profiler, "end_to_end"),
            "end_to_end_action_frames_ms": _stat(profiler, "end_to_end_action"),
            "tracking": {
                "frames_by_state": dict(tracking.frames_by_state),
                "non_tracking_fraction": round(tracking.non_tracking_fraction, 4),
                "loss_events": tracking.loss_events,
                "loss_events_per_minute": _round(profiler.loss_events_per_minute()),
            },
            "cpu_and_memory": resource_summary.as_dict() if resource_summary else None,
        },
        "cpu_normalization": (
            "cpu_percent_of_one_core = 100 * delta(process CPU time) / delta(wall time) and can "
            "exceed 100 with several threads; cpu_percent_of_machine divides that by "
            "logical_cpus. cpu_time_s is absolute CPU seconds, not a rate."
        ),
        "caveats": list(MODE_CAVEATS[mode]),
        "not_measured": list(COMMON_NOT_MEASURED),
    }


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def _config_snapshot(config: Config) -> dict[str, Any]:
    return {
        "pinch_mode": "3D (experimental)" if config.pinch_use_depth else "2D (default)",
        "pinch_threshold": config.pinch_threshold,
        "pinch_confirm_frames": config.pinch_confirm_frames,
        "pinch_release_factor": config.pinch_release_factor,
        "pinch_release_frames": config.pinch_release_frames,
        "click_cooldown_s": config.click_cooldown_s,
        "drag_start_px": config.drag_start_px,
        "smoothing": config.smoothing,
        "active_margin": config.active_margin,
        "stable_frames": config.stable_frames,
        "min_confidence": config.min_confidence,
        "frame_width": config.frame_width,
        "frame_height": config.frame_height,
        "scroll_stable_frames": config.scroll_stable_frames,
        "scroll_dead_zone": config.scroll_dead_zone,
        "scroll_speed": config.scroll_speed,
        "scroll_interval_s": config.scroll_interval_s,
        "dry_run": True,
    }


def _write_report(report: dict[str, Any], config: Config) -> Path:
    path = config.benchmark_output
    if path is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        path = default_benchmark_dir() / f"{report['mode']}-{stamp}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    except OSError as exc:
        raise BenchmarkError(f"could not write the benchmark report to {path}: {exc}") from exc
    return path


def format_summary(report: dict[str, Any], path: Path | None = None) -> str:
    """Human-readable summary of a report (what the terminal shows)."""
    r = report["results"]
    cond = report["conditions"]
    lines = [
        f"NoCursor benchmark - {report['mode_title']}",
        f"Frames: {r['frames']} in {r['duration_s']} s"
        + (f" ({r['dropped_frames']} dropped)" if r["dropped_frames"] else ""),
        f"Frame rate: {_fmt(r['frame_rate']['frames_per_second'])} fps  "
        f"[{r['frame_rate']['meaning']}]",
        "",
        f"{'Stage (milliseconds)':<26}{'n':>7}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}",
    ]

    def row(label: str, stat: dict[str, Any] | None) -> str:
        if stat is None:
            return f"{label:<26}{'-':>7}  (no samples)"
        return (f"{label:<26}{stat['n']:>7}{format_ms(stat['mean']):>9}{format_ms(stat['p50']):>9}"
                f"{format_ms(stat['p95']):>9}{format_ms(stat['p99']):>9}{format_ms(stat['max']):>9}")

    for name, stat in r["stages_ms"].items():
        lines.append(row(name, stat))
    lines.append(row("END-TO-END (software)", r["end_to_end_software_ms"]))
    lines.append(row("  frames with an action", r["end_to_end_action_frames_ms"]))
    t = r["tracking"]
    lines += [
        "",
        f"Tracking: {t['loss_events']} loss event(s)"
        + (f" ({_fmt(t['loss_events_per_minute'])}/min)" if t["loss_events_per_minute"] else "")
        + f"; {t['non_tracking_fraction']:.1%} of frames not tracking",
    ]
    usage = r["cpu_and_memory"]
    if usage:
        lines.append(
            f"CPU: {_fmt(usage['cpu_percent_of_one_core'])}% of one core "
            f"({_fmt(usage['cpu_percent_of_machine'])}% of the machine's "
            f"{usage['logical_cpus']} logical CPUs), {usage['cpu_time_s']} s CPU time"
        )
        lines.append(
            f"Memory: {_megabytes(usage['rss_bytes'])} resident, "
            f"{_megabytes(usage['peak_rss_bytes'])} peak"
        )
    if "hand_present_frames" in cond and cond["hand_present_frames"] is not None:
        lines.append(f"Frames with a hand found: {cond['hand_present_frames']}")
    lines += ["", "Caveats:"] + [f"  - {c}" for c in report["caveats"]]
    lines += ["Not measured: " + "; ".join(report["not_measured"]) + "."]
    if path is not None:
        lines.append(f"Report written to: {path}")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g}"


def _megabytes(value: int | None) -> str:
    return "n/a" if value is None else f"{value / 1e6:.0f} MB"
