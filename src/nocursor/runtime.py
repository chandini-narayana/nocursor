"""Assembles the real application from a ``Config``.

Used by the CLI for normal runs and by the live benchmark, so a benchmark exercises exactly the
same camera, tracker, gesture, controller and preview code that you use day to day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from nocursor.config import Config
from nocursor.environment import missing_dependencies, missing_dependencies_message

logger = logging.getLogger(__name__)


class StartupError(Exception):
    """Something prevented NoCursor from starting; the message says what to do about it."""


@dataclass
class Runtime:
    app: Any  # NoCursorApp
    stop: Any  # EmergencyStop
    controller: Any  # SafeController
    processor: Any  # GestureProcessor
    camera: Any  # Camera
    profiler: Any
    resources: Any | None
    calibration_message: str | None = None
    calibration_path: Any | None = None
    calibration_in_use: bool = False


def ensure_dependencies() -> None:
    missing = missing_dependencies()
    if missing:
        raise StartupError(missing_dependencies_message(missing))


def build_runtime(
    config: Config,
    *,
    input_blocked: bool = False,
    profiler: Any | None = None,
    resources: Any | None = None,
    time_limit_s: float | None = None,
    preview: Any | None = None,
    install_hotkeys: bool = True,
) -> Runtime:
    """Build the app. With ``input_blocked`` the controller's backend cannot send any input.

    Imports happen here, not at module load, so ``--help`` and configuration errors stay fast
    and a missing package produces a clear message instead of a traceback.
    """
    ensure_dependencies()
    try:
        from nocursor.app import NoCursorApp
        from nocursor.calibration import (
            CalibrationStore,
            CalibrationWizard,
            default_calibration_path,
        )
        from nocursor.camera import Camera
        from nocursor.controller import PyAutoGUIBackend, SafeController, ScreenSizeOnlyBackend
        from nocursor.diagnostics import DiagnosticsReporter
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
        from nocursor.profiling import FrameProfiler
    except ImportError as exc:
        raise StartupError(f"a required package failed to import ({exc}). "
                           "Reinstall with: python -m pip install -r requirements.txt") from exc

    try:
        model_path = resolve_model_path(config.model_path)
    except ModelNotFoundError as exc:
        raise StartupError(str(exc)) from exc

    stop = EmergencyStop()
    pause = PauseControl()
    real_backend = PyAutoGUIBackend()
    backend = ScreenSizeOnlyBackend(source=real_backend) if input_blocked else real_backend
    controller = SafeController(
        backend, stop, enabled=not config.dry_run and not input_blocked, quiet=input_blocked
    )

    store = CalibrationStore(config.calibration_path or default_calibration_path())
    region = None
    runtime_message = None
    if config.use_calibration:
        loaded = store.load(config.camera_index, config.frame_width / config.frame_height)
        region = loaded.region
        runtime_message = loaded.message
        if loaded.message:
            logger.warning("%s", loaded.message)
    processor = GestureProcessor(config, controller.screen_size(), region)

    watcher = None
    extra_watchers: list[Any] = []
    is_key_down = system_key_state() if install_hotkeys else None
    if is_key_down is None:
        if install_hotkeys:
            logger.warning(
                "global hotkeys are unsupported on this platform; "
                "use the preview keys (Esc/Q, P) or Ctrl+C"
            )
    else:
        watcher = HotkeyWatcher(stop, parse_hotkey(config.emergency_hotkey), is_key_down)
        extra_watchers.append(
            HotkeyAction(parse_hotkey(config.pause_hotkey), is_key_down, pause.toggle)
        )

    try:
        tracker = HandTracker(model_path, config.min_confidence)
    except Exception as exc:  # MediaPipe raises plain RuntimeError for bad model files
        raise StartupError(
            f"could not load the hand-landmark model ({exc}). The file may be corrupt or "
            "incompatible; download it again with: nocursor --redownload-model"
        ) from exc

    camera = Camera(config.camera_index, config.frame_width, config.frame_height,
                    config.max_read_failures)  # fmt: skip
    if profiler is None and config.show_perf:
        profiler = FrameProfiler()
    app = NoCursorApp(
        camera, tracker, processor, controller,
        preview if preview is not None else OpenCVPreview(config.active_margin),
        stop, config.emergency_hotkey, watcher,
        pause=pause, wizard=CalibrationWizard(), store=store, extra_watchers=extra_watchers,
        camera_index=config.camera_index, pause_hotkey_label=config.pause_hotkey,
        start_in_calibration=config.start_calibration,
        diagnostics=DiagnosticsReporter() if config.diagnostics else None,
        profiler=profiler, resources=resources, show_perf=config.show_perf,
        time_limit_s=time_limit_s,
    )  # fmt: skip
    return Runtime(
        app=app, stop=stop, controller=controller, processor=processor, camera=camera,
        profiler=profiler, resources=resources, calibration_message=runtime_message,
        calibration_path=store.path, calibration_in_use=region is not None,
    )
