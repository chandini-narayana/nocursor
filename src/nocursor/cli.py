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
    p.add_argument("--click-cooldown", type=float, dest="click_cooldown_s",
                   help=f"minimum seconds between clicks (default {defaults.click_cooldown_s})")
    p.add_argument("--min-confidence", type=float, dest="min_confidence",
                   help=f"minimum hand confidence (default {defaults.min_confidence})")
    p.add_argument("--stop-key", dest="emergency_hotkey",
                   help=f"global emergency-stop hotkey (default {defaults.emergency_hotkey})")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    overrides = {k: v for k, v in vars(args).items() if k != "verbose"}
    overrides["dry_run"] = bool(overrides["dry_run"])
    try:
        config = Config.from_overrides(**overrides)
    except ConfigError as exc:
        print(f"nocursor: invalid configuration: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return run(config)


def run(config: Config) -> int:
    # Heavy imports are deferred so `--help` and config errors stay fast.
    from nocursor.app import EXIT_ERROR, NoCursorApp
    from nocursor.camera import Camera
    from nocursor.controller import PyAutoGUIBackend, SafeController
    from nocursor.emergency import EmergencyStop, HotkeyWatcher, parse_hotkey, system_key_state
    from nocursor.hand_tracker import HandTracker, ModelNotFoundError, resolve_model_path
    from nocursor.preview import OpenCVPreview
    from nocursor.processor import GestureProcessor

    try:
        model_path = resolve_model_path(config.model_path)
    except ModelNotFoundError as exc:
        print(f"nocursor: {exc}", file=sys.stderr)
        return EXIT_ERROR

    stop = EmergencyStop()
    controller = SafeController(PyAutoGUIBackend(), stop, enabled=not config.dry_run)
    processor = GestureProcessor(config, controller.screen_size())

    is_key_down = system_key_state()
    watcher = None
    if is_key_down is None:
        logger.warning("global emergency hotkey is unsupported on this platform; "
                       "use Esc/Q in the preview window or Ctrl+C")
    else:
        watcher = HotkeyWatcher(stop, parse_hotkey(config.emergency_hotkey), is_key_down)

    tracker = HandTracker(model_path, config.min_confidence)
    camera = Camera(config.camera_index, config.frame_width, config.frame_height,
                    config.max_read_failures)
    app = NoCursorApp(camera, tracker, processor, controller, OpenCVPreview(config.active_margin),
                      stop, config.emergency_hotkey, watcher)  # fmt: skip

    mode = "DRY-RUN (no mouse input will be sent)" if config.dry_run else "LIVE"
    print(f"NoCursor running in {mode} mode.")
    print(f"Emergency stop: {config.emergency_hotkey} (any window), Esc/Q in the preview, "
          "or Ctrl+C in this terminal.")  # fmt: skip
    code = app.run()
    print("NoCursor stopped." + (f" Reason: {stop.reason}" if stop.is_set() else ""))
    return code
