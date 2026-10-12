"""Command-line entry point.

Options are grouped by purpose in ``--help``. Every numeric option shows its default and its
valid range; the ranges come from ``RANGES`` below, which a test checks against the real
validation so the help text cannot drift from the behaviour.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from nocursor import __version__
from nocursor.config import Config, ConfigError
from nocursor.paths import default_log_path

logger = logging.getLogger("nocursor")

EXIT_USAGE = 64
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

# field -> (range as shown in --help, a valid example, an invalid example)
RANGES: dict[str, tuple[str, object, object]] = {
    "camera_index": (">= 0", 0, -1),
    "frame_width": ("> 0", 640, 0),
    "frame_height": ("> 0", 480, 0),
    "min_confidence": ("0 to 1", 1.0, 1.01),
    "smoothing": ("0 to below 1", 0.0, 1.0),
    "active_margin": ("0 to below 0.5", 0.0, 0.5),
    "pinch_threshold": ("> 0", 0.01, 0.0),
    "pinch_confirm_frames": (">= 1", 1, 0),
    "pinch_release_factor": ("> 1", 1.01, 1.0),
    "pinch_release_frames": (">= 1", 1, 0),
    "click_cooldown_s": (">= 0", 0.0, -0.1),
    "drag_start_px": ("above 0 up to 500", 500.0, 501.0),
    "scroll_stable_frames": (">= 1", 1, 0),
    "scroll_dead_zone": ("above 0 up to 1", 1.0, 1.01),
    "scroll_speed": ("above 0 up to 100", 100.0, 100.1),
    "scroll_interval_s": ("0.01 to 2", 2.0, 2.01),
    "benchmark_frames": ("1 to 1000000", 1_000_000, 1_000_001),
    "benchmark_seconds": ("above 0 up to 3600", 3600.0, 3601.0),
}

ACTION_FLAGS = ("verbose", "reset_calibration", "check", "print_settings", "download_model",
                "redownload_model")  # fmt: skip

EPILOG = """\
examples:
  nocursor --check                          check the installation (never opens the camera)
  nocursor --download-model                 one-time download of the hand-landmark model
  nocursor --dry-run --diagnostics          try everything; nothing is sent to the desktop
  nocursor                                  live control (mouse input is sent)
  nocursor --calibrate                      mark the 4 corners of your hand area first
  nocursor --benchmark synthetic            repeatable software benchmark (no camera)
  nocursor --benchmark live --benchmark-seconds 30    measure with your webcam (dry-run)

keys in the preview window:
  Esc / Q  stop (always)      P  pause / resume      C  calibrate      H  help panel
  Space/Enter  capture a calibration corner      Backspace/X  cancel      R  reset calibration
  Ctrl+Alt+Q stops from any window (Windows); Ctrl+Alt+P pauses/resumes.
"""


def _help(defaults: Config, field: str, text: str, shown_default: object = None) -> str:
    """Help text with the default and, for numeric options, the valid range."""
    value = getattr(defaults, field) if shown_default is None else shown_default
    if isinstance(value, float):
        value = f"{value:g}"
    parts = [f"default {value}"] if value not in (None, "") else []
    if field in RANGES:
        parts.append(f"valid: {RANGES[field][0]}")
    return f"{text} ({'; '.join(parts)})" if parts else text


def build_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(
        prog="nocursor",
        description="Control your desktop cursor with hand gestures from a webcam. "
        "Everything runs locally; video is never recorded or sent anywhere.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    start = p.add_argument_group("getting started")
    start.add_argument("--version", action="version", version=f"nocursor {__version__}")
    start.add_argument("--check", action="store_true", default=argparse.SUPPRESS,
                       help="check packages, model, window support and calibration, then exit "
                            "(opens no camera, sends no input)")
    start.add_argument("--print-settings", action="store_true", default=argparse.SUPPRESS,
                       help="print the effective settings, then exit")
    start.add_argument("--download-model", action="store_true", default=argparse.SUPPRESS,
                       help="download the MediaPipe hand-landmark model (the only network "
                            "access), then exit")
    start.add_argument("--redownload-model", action="store_true", default=argparse.SUPPRESS,
                       help="download the model again even if one exists (use if it is corrupt)")

    cam = p.add_argument_group("camera and model")
    cam.add_argument("--camera", type=int, dest="camera_index",
                     help=_help(d, "camera_index", "camera index"))
    cam.add_argument("--width", type=int, dest="frame_width",
                     help=_help(d, "frame_width", "capture width in pixels"))
    cam.add_argument("--height", type=int, dest="frame_height",
                     help=_help(d, "frame_height", "capture height in pixels"))
    cam.add_argument("--model", type=Path, dest="model_path", metavar="PATH",
                     help="path to hand_landmarker.task (default: models/ in the project or in "
                          "the working directory)")
    cam.add_argument("--min-confidence", type=float, dest="min_confidence",
                     help=_help(d, "min_confidence", "minimum hand-detection confidence"))

    pointer = p.add_argument_group("pointer")
    pointer.add_argument("--smoothing", type=float,
                         help=_help(d, "smoothing", "cursor smoothing; 0 = none, higher = steadier "
                                                    "but laggier"))
    pointer.add_argument("--margin", type=float, dest="active_margin",
                         help=_help(d, "active_margin", "fraction of each camera edge ignored by "
                                                        "the default mapping"))

    pinch = p.add_argument_group("pinch: click and drag")
    pinch.add_argument("--pinch-threshold", type=float, dest="pinch_threshold",
                       help=_help(d, "pinch_threshold", "thumb-index distance / hand size that "
                                                        "counts as a pinch; lower = stricter"))
    pinch.add_argument("--pinch-confirm-frames", type=int, dest="pinch_confirm_frames",
                       help=_help(d, "pinch_confirm_frames", "consecutive frames at or below the "
                                                             "threshold before a pinch counts"))
    pinch.add_argument("--pinch-release-factor", type=float, dest="pinch_release_factor",
                       help=_help(d, "pinch_release_factor", "the pinch ends at threshold x this"))
    pinch.add_argument("--pinch-release-frames", type=int, dest="pinch_release_frames",
                       help=_help(d, "pinch_release_frames", "open frames needed to confirm a "
                                                             "release, so one glitch cannot drop "
                                                             "a drag"))
    pinch.add_argument("--click-cooldown", type=float, dest="click_cooldown_s",
                       help=_help(d, "click_cooldown_s",
                                  "minimum seconds between accepted pinches"))
    pinch.add_argument("--drag-start", type=float, dest="drag_start_px",
                       help=_help(d, "drag_start_px", "screen pixels the fingertip must move "
                                                      "while pinched before a click becomes a "
                                                      "drag"))
    depth = pinch.add_mutually_exclusive_group()  # giving both flags is an error, never silent
    depth.add_argument("--no-pinch-depth", action="store_false", dest="pinch_use_depth",
                       default=None,
                       help="measure the pinch in the image plane (2D). This is already the "
                            "default; the flag is kept so existing commands keep working")
    depth.add_argument("--pinch-depth", action="store_true", dest="pinch_use_depth",
                       default=None,
                       help="EXPERIMENTAL: also use MediaPipe's coarse depth estimate (3D pinch "
                            "ratio). It stopped pinches registering on at least one webcam")

    scroll = p.add_argument_group("two-finger scrolling")
    scroll.add_argument("--scroll-stable-frames", type=int, dest="scroll_stable_frames",
                        help=_help(d, "scroll_stable_frames", "frames the pose must hold before "
                                                              "scrolling starts"))
    scroll.add_argument("--scroll-dead-zone", type=float, dest="scroll_dead_zone",
                        help=_help(d, "scroll_dead_zone", "vertical movement, in hand sizes, "
                                                          "ignored as jitter"))
    scroll.add_argument("--scroll-speed", type=float, dest="scroll_speed",
                        help=_help(d, "scroll_speed", "wheel notches per hand size of movement"))
    scroll.add_argument("--scroll-interval", type=float, dest="scroll_interval_s",
                        help=_help(d, "scroll_interval_s", "minimum seconds between scroll events"))
    scroll.add_argument("--scroll-invert", action="store_true", dest="scroll_invert",
                        help="hand up scrolls down instead of up")

    cal = p.add_argument_group("calibration")
    cal.add_argument("--calibrate", action="store_true", dest="start_calibration",
                     help="start the 4-corner calibration wizard (no mouse input is sent while "
                          "it runs)")
    cal.add_argument("--no-calibration", action="store_false", dest="use_calibration",
                     help="ignore any saved calibration and use the default margin mapping")
    cal.add_argument("--calibration-file", type=Path, dest="calibration_path", metavar="PATH",
                     help="where calibration is stored (default: calibration.local.json in the "
                          "project)")
    cal.add_argument("--reset-calibration", action="store_true", default=argparse.SUPPRESS,
                     help="delete the saved calibration, then exit")

    safety = p.add_argument_group("safety")
    safety.add_argument("--dry-run", action="store_true",
                        help="track and preview, but never send mouse input")
    safety.add_argument("--stop-key", dest="emergency_hotkey", metavar="KEYS",
                        help=_help(d, "emergency_hotkey", "global emergency-stop hotkey"))
    safety.add_argument("--pause-key", dest="pause_hotkey", metavar="KEYS",
                        help=_help(d, "pause_hotkey", "global pause / resume hotkey"))

    diag = p.add_argument_group("diagnostics, logging and performance")
    diag.add_argument("--diagnostics", action="store_true", dest="diagnostics",
                      help="show live pinch internals in the preview and log them at DEBUG, at "
                           "most once per second (read-only; changes no behaviour)")
    diag.add_argument("--show-perf", action="store_true", dest="show_perf",
                      help="show rolling software timings (fps, inference, end-to-end) in the "
                           "preview")
    diag.add_argument("--log-file", nargs="?", const=default_log_path(), type=Path,
                      dest="log_file", metavar="PATH",
                      help="also write a rotating log file (about 4 MB at most); without PATH "
                           f"it goes to {default_log_path().parent.name}/{default_log_path().name}."
                           " No log file is written unless you ask")
    diag.add_argument("-v", "--verbose", action="store_true",
                      help="debug-level logging, including routine MediaPipe start-up output")

    bench = p.add_argument_group(
        "benchmarks (always dry-run: they cannot send desktop input)")
    bench.add_argument("--benchmark", choices=("synthetic", "video", "live"),
                       help="synthetic: scripted input, no camera, repeatable. video: your own "
                            "clip, unpaced. live: your webcam for a fixed time. Results from the "
                            "three are never mixed")
    bench.add_argument("--benchmark-frames", type=int, dest="benchmark_frames",
                       help=_help(d, "benchmark_frames", "frames to process for synthetic "
                                                         "(default 1500) or video (default: whole "
                                                         "clip)", shown_default=""))
    bench.add_argument("--benchmark-seconds", type=float, dest="benchmark_seconds",
                       help=_help(d, "benchmark_seconds", "duration of a live benchmark "
                                                          "(default 30)", shown_default=""))
    bench.add_argument("--benchmark-video", type=Path, dest="benchmark_video", metavar="PATH",
                       help="video clip for --benchmark video (read locally, never copied)")
    bench.add_argument("--benchmark-output", type=Path, dest="benchmark_output", metavar="PATH",
                       help="where to write the JSON report "
                            "(default: benchmarks/<mode>-<time>.json)")
    return p  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("\nnocursor: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


def _main(argv: Sequence[str] | None) -> int:
    args = build_parser().parse_args(argv)
    options = vars(args)
    overrides = {k: v for k, v in options.items() if k not in ACTION_FLAGS}
    overrides["dry_run"] = bool(overrides["dry_run"])
    try:
        config = Config.from_overrides(**overrides)
    except ConfigError as exc:
        print(f"nocursor: invalid configuration: {exc}", file=sys.stderr)
        return EXIT_USAGE

    from nocursor.logging_setup import configure_logging

    handle = configure_logging(
        verbose=bool(options.get("verbose")), diagnostics=config.diagnostics,
        log_file=config.log_file,
    )
    try:
        if options.get("download_model") or options.get("redownload_model"):
            from nocursor.model_setup import run_download

            return run_download(force=bool(options.get("redownload_model")))
        if options.get("print_settings"):
            from nocursor.settings import describe_settings

            print(describe_settings(config))
            return 0
        if options.get("check"):
            return run_check(config)
        if options.get("reset_calibration"):
            return reset_saved_calibration(config)
        if config.benchmark is not None:
            return run_benchmark_command(config)
        return run(config)
    finally:
        handle.close()


def run_check(config: Config) -> int:
    from nocursor.checks import checks_exit_code, format_checks, run_checks

    items = run_checks(config)
    print(format_checks(items))
    return checks_exit_code(items)


def reset_saved_calibration(config: Config) -> int:
    from nocursor.calibration import CalibrationStore, default_calibration_path

    store = CalibrationStore(config.calibration_path or default_calibration_path())
    try:
        removed = store.delete()
    except OSError as exc:
        print(f"nocursor: could not delete {store.path}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Calibration {'deleted' if removed else 'not found (nothing to delete)'}: {store.path}")
    return 0


def run_benchmark_command(config: Config) -> int:
    from nocursor.benchmark import LIVE_DEFAULT_SECONDS, BenchmarkError, run_benchmark

    print(f"NoCursor benchmark: {config.benchmark}. It is dry-run: no desktop input can be sent.")
    if config.benchmark == "live":
        seconds = config.benchmark_seconds or LIVE_DEFAULT_SECONDS
        print(f"Keep one hand in view and use your gestures normally for {seconds:g} seconds. "
              f"Stop early with Esc/Q or {config.emergency_hotkey}.")  # fmt: skip
    elif config.benchmark == "synthetic":
        print("Scripted input, no camera; this measures the software, not your webcam.")
    try:
        result = run_benchmark(config)
    except BenchmarkError as exc:
        print(f"nocursor: benchmark failed: {exc}", file=sys.stderr)
        logger.error("benchmark failed: %s", exc)
        return EXIT_ERROR
    print()
    print(result.summary)
    logger.info("benchmark finished: mode=%s report=%s", config.benchmark, result.path,
                extra={"file_only": True})  # fmt: skip
    return 0


def run(config: Config) -> int:
    # Heavy imports live in build_runtime, so `--help` and configuration errors stay fast.
    from nocursor.diagnostics import mode_label
    from nocursor.runtime import StartupError, build_runtime

    try:
        runtime = build_runtime(config)
    except StartupError as exc:
        print(f"nocursor: {exc}", file=sys.stderr)
        logger.error("startup failed: %s", exc)
        return EXIT_ERROR

    mode = "DRY-RUN (no mouse input will be sent)" if config.dry_run else "LIVE"
    print(f"NoCursor running in {mode} mode.")
    print(f"Emergency stop: {config.emergency_hotkey} (any window), Esc/Q in the preview, "
          "or Ctrl+C in this terminal.")  # fmt: skip
    print(f"Pinch measurement: {mode_label(config.pinch_use_depth)}.")
    print(f"Pause/resume: {config.pause_hotkey} or P in the preview. "
          "Calibrate: C. Reset calibration: R. Help: H.")  # fmt: skip
    if runtime.calibration_in_use:
        print(f"Using saved calibration: {runtime.calibration_path}")
    if config.log_file is not None:
        print(f"Logging to: {config.log_file}")
    code = runtime.app.run()
    stop = runtime.stop
    print("NoCursor stopped." + (f" Reason: {stop.reason}" if stop.is_set() else ""))
    if runtime.controller.fault_message:
        print(f"WARNING: {runtime.controller.fault_message}", file=sys.stderr)
    return code
