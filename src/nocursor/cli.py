"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from nocursor.config import Config, ConfigError

logger = logging.getLogger("nocursor")

EXIT_USAGE = 64


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    p = argparse.ArgumentParser(
        prog="nocursor", description="Control your desktop cursor with hand gestures."
    )
    p.add_argument("--dry-run", action="store_true",
                   help="track and preview, but never send mouse input")
    p.add_argument("--camera", type=int, dest="camera_index",
                   help=f"camera index (default {defaults.camera_index})")
    p.add_argument("--width", type=int, dest="frame_width",
                   help=f"capture width (default {defaults.frame_width})")
    p.add_argument("--height", type=int, dest="frame_height",
                   help=f"capture height (default {defaults.frame_height})")
    p.add_argument("--model", type=Path, dest="model_path",
                   help="path to hand_landmarker.task (default: ./models or project models/)")
    p.add_argument("--smoothing", type=float,
                   help=f"cursor smoothing in [0,1): 0 = none (default {defaults.smoothing})")
    p.add_argument("--margin", type=float, dest="active_margin",
                   help=f"camera edge fraction ignored, [0,0.5) (default {defaults.active_margin})")
    p.add_argument("--pinch-threshold", type=float, dest="pinch_threshold",
                   help="thumb-index distance / hand size that counts as a pinch "
                        f"(default {defaults.pinch_threshold})")
    p.add_argument("--pinch-confirm-frames", type=int, dest="pinch_confirm_frames",
                   help="consecutive frames at/below the threshold needed to accept a pinch "
                        f"(default {defaults.pinch_confirm_frames})")
    p.add_argument("--pinch-release-factor", type=float, dest="pinch_release_factor",
                   help="pinch ends at threshold x this factor, must be > 1 "
                        f"(default {defaults.pinch_release_factor})")
    p.add_argument("--pinch-release-frames", type=int, dest="pinch_release_frames",
                   help="consecutive open frames needed to confirm a release, so one glitch "
                        f"cannot drop a drag (default {defaults.pinch_release_frames})")
    depth = p.add_mutually_exclusive_group()  # giving both flags is an error, never silent
    depth.add_argument("--no-pinch-depth", action="store_false", dest="pinch_use_depth",
                       default=None,
                       help="measure the pinch in the image plane (2D). This is already the "
                            "default; the flag is kept so existing commands keep working")
    depth.add_argument("--pinch-depth", action="store_true", dest="pinch_use_depth",
                       default=None,
                       help="EXPERIMENTAL: also use MediaPipe's coarse depth estimate (3D pinch "
                            "ratio). It stopped pinches registering on at least one webcam")
    p.add_argument("--diagnostics", action="store_true", dest="diagnostics",
                   help="show live pinch internals in the preview and log one line per second "
                        "(read-only; changes no behaviour, sends no input)")
    p.add_argument("--click-cooldown", type=float, dest="click_cooldown_s",
                   help="minimum seconds between accepted pinches "
                        f"(default {defaults.click_cooldown_s})")
    p.add_argument("--drag-start", type=float, dest="drag_start_px",
                   help="screen pixels the fingertip must move while pinched before a click "
                        f"becomes a drag (default {defaults.drag_start_px:g})")
    p.add_argument("--scroll-stable-frames", type=int, dest="scroll_stable_frames",
                   help="frames the two-finger pose must hold before scrolling starts "
                        f"(default {defaults.scroll_stable_frames})")
    p.add_argument("--scroll-dead-zone", type=float, dest="scroll_dead_zone",
                   help="vertical movement, in hand sizes, ignored as jitter "
                        f"(default {defaults.scroll_dead_zone})")
    p.add_argument("--scroll-speed", type=float, dest="scroll_speed",
                   help="wheel clicks per hand size of movement "
                        f"(default {defaults.scroll_speed:g})")
    p.add_argument("--scroll-interval", type=float, dest="scroll_interval_s",
                   help="minimum seconds between scroll events "
                        f"(default {defaults.scroll_interval_s})")
    p.add_argument("--scroll-invert", action="store_true", dest="scroll_invert",
                   help="hand up scrolls down instead of up")
    p.add_argument("--min-confidence", type=float, dest="min_confidence",
                   help=f"minimum hand confidence (default {defaults.min_confidence})")
    p.add_argument("--stop-key", dest="emergency_hotkey",
                   help=f"global emergency-stop hotkey (default {defaults.emergency_hotkey})")
    p.add_argument("--pause-key", dest="pause_hotkey",
                   help=f"global pause/resume hotkey (default {defaults.pause_hotkey})")
    p.add_argument("--calibrate", action="store_true", dest="start_calibration",
                   help="start the 4-corner calibration wizard at launch (no mouse input is sent "
                        "while it runs)")
    p.add_argument("--no-calibration", action="store_false", dest="use_calibration",
                   help="ignore any saved calibration and use the default margin mapping")
    p.add_argument("--calibration-file", type=Path, dest="calibration_path",
                   help="where calibration is stored (default: calibration.local.json in the "
                        "project)")
    p.add_argument("--reset-calibration", action="store_true", default=argparse.SUPPRESS,
                   help="delete the saved calibration and exit")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    options = vars(args)
    reset_calibration = bool(options.get("reset_calibration", False))
    overrides = {k: v for k, v in options.items() if k not in ("verbose", "reset_calibration")}
    overrides["dry_run"] = bool(overrides["dry_run"])
    try:
        config = Config.from_overrides(**overrides)
    except ConfigError as exc:
        print(f"nocursor: invalid configuration: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if reset_calibration:
        return reset_saved_calibration(config)
    return run(config)


def reset_saved_calibration(config: Config) -> int:
    from nocursor.calibration import CalibrationStore, default_calibration_path

    store = CalibrationStore(config.calibration_path or default_calibration_path())
    try:
        removed = store.delete()
    except OSError as exc:
        print(f"nocursor: could not delete {store.path}: {exc}", file=sys.stderr)
        return 1
    print(f"Calibration {'deleted' if removed else 'not found (nothing to delete)'}: {store.path}")
    return 0


def run(config: Config) -> int:
    # Heavy imports are deferred so `--help` and config errors stay fast.
    from nocursor.app import EXIT_ERROR, NoCursorApp
    from nocursor.calibration import CalibrationStore, CalibrationWizard, default_calibration_path
    from nocursor.camera import Camera
    from nocursor.controller import PyAutoGUIBackend, SafeController
    from nocursor.diagnostics import DiagnosticsReporter, mode_label
    from nocursor.emergency import (
        EmergencyStop,
        HotkeyAction,
        HotkeyWatcher,
        PauseControl,
        parse_hotkey,
        system_key_state,
    )
    from nocursor.hand_tracker import HandTracker, ModelNotFoundError, resolve_model_path
    from nocursor.preview import OpenCVPreview
    from nocursor.processor import GestureProcessor

    try:
        model_path = resolve_model_path(config.model_path)
    except ModelNotFoundError as exc:
        print(f"nocursor: {exc}", file=sys.stderr)
        return EXIT_ERROR

    stop = EmergencyStop()
    pause = PauseControl()
    controller = SafeController(PyAutoGUIBackend(), stop, enabled=not config.dry_run)

    store = CalibrationStore(config.calibration_path or default_calibration_path())
    region = None
    if config.use_calibration:
        loaded = store.load(config.camera_index, config.frame_width / config.frame_height)
        region = loaded.region
        if loaded.message:
            print(f"nocursor: {loaded.message}", file=sys.stderr)
        elif region is not None:
            print(f"Using saved calibration: {store.path}")
    processor = GestureProcessor(config, controller.screen_size(), region)

    is_key_down = system_key_state()
    watcher = None
    extra_watchers = []
    if is_key_down is None:
        logger.warning("global hotkeys are unsupported on this platform; "
                       "use the preview keys (Esc/Q, P) or Ctrl+C")
    else:
        watcher = HotkeyWatcher(stop, parse_hotkey(config.emergency_hotkey), is_key_down)
        extra_watchers.append(
            HotkeyAction(parse_hotkey(config.pause_hotkey), is_key_down, pause.toggle)
        )

    tracker = HandTracker(model_path, config.min_confidence)
    camera = Camera(config.camera_index, config.frame_width, config.frame_height,
                    config.max_read_failures)
    app = NoCursorApp(
        camera, tracker, processor, controller, OpenCVPreview(config.active_margin),
        stop, config.emergency_hotkey, watcher,
        pause=pause, wizard=CalibrationWizard(), store=store, extra_watchers=extra_watchers,
        camera_index=config.camera_index, pause_hotkey_label=config.pause_hotkey,
        start_in_calibration=config.start_calibration,
        diagnostics=DiagnosticsReporter() if config.diagnostics else None,
    )  # fmt: skip

    mode = "DRY-RUN (no mouse input will be sent)" if config.dry_run else "LIVE"
    print(f"NoCursor running in {mode} mode.")
    print(f"Emergency stop: {config.emergency_hotkey} (any window), Esc/Q in the preview, "
          "or Ctrl+C in this terminal.")  # fmt: skip
    print(f"Pinch measurement: {mode_label(config.pinch_use_depth)}.")
    print(f"Pause/resume: {config.pause_hotkey} or P in the preview. "
          "Calibrate: C. Reset calibration: R.")  # fmt: skip
    code = app.run()
    print("NoCursor stopped." + (f" Reason: {stop.reason}" if stop.is_set() else ""))
    if controller.fault_message:
        print(f"WARNING: {controller.fault_message}", file=sys.stderr)
    return code
