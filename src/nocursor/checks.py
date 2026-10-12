"""``nocursor --check``: a first-run health check that never opens the camera or sends input."""

from __future__ import annotations

from dataclasses import dataclass

from nocursor.config import Config
from nocursor.environment import (
    installed_opencv,
    opencv_gui_support,
    package_version,
    python_check,
)

OK, WARN, FAIL, INFO = "ok", "warn", "FAIL", "info"
_MARK = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]", INFO: "[info]"}


@dataclass(frozen=True)
class CheckItem:
    status: str
    name: str
    detail: str


def run_checks(config: Config) -> list[CheckItem]:
    """Check the installation. Reads versions and files only; opens no camera, sends no input."""
    items: list[CheckItem] = []
    py = python_check()
    items.append(CheckItem(OK if py.ok else FAIL, "Python", f"{py.version} (3.10 or newer needed)"))

    import importlib.util

    from nocursor.environment import REQUIRED_IMPORTS

    for module, (dist, purpose) in REQUIRED_IMPORTS.items():
        if importlib.util.find_spec(module) is None:
            items.append(CheckItem(
                FAIL, dist,
                f"not installed ({purpose}). Run: python -m pip install -r requirements.txt",
            ))
        else:
            version = package_version(dist) or "version unknown"
            items.append(CheckItem(OK, dist, f"{version} ({purpose})"))

    opencv = installed_opencv()
    if len(opencv) > 1:
        names = ", ".join(f"{n} {v}" for n, v in opencv.items())
        items.append(CheckItem(
            WARN, "OpenCV packages",
            f"several are installed ({names}); they overwrite each other's 'cv2'. "
            "Keep only opencv-contrib-python (see README, Troubleshooting).",
        ))
    gui = opencv_gui_support()
    if gui is False:
        items.append(CheckItem(
            FAIL, "OpenCV window support",
            "this OpenCV build is headless, so the preview window cannot open. Install "
            "opencv-contrib-python (and remove any '-headless' OpenCV package).",
        ))
    elif gui is None:
        items.append(CheckItem(WARN, "OpenCV window support", "could not be determined"))
    else:
        items.append(CheckItem(OK, "OpenCV window support", "available"))

    items.append(_model_check(config))
    items.append(_screen_check())
    items.append(_hotkey_check())
    items.append(_calibration_check(config))
    items.append(CheckItem(
        INFO, "Camera",
        f"not opened by --check. Camera {config.camera_index} is used when you run NoCursor "
        "(choose another with --camera N).",
    ))
    items.append(CheckItem(
        INFO, "Pinch measurement",
        "3D pinch ratio (experimental)" if config.pinch_use_depth else "2D pinch ratio (default)",
    ))
    if config.log_file is not None:
        items.append(_log_check(config))
    return items


def _model_check(config: Config) -> CheckItem:
    from nocursor.hand_tracker import ModelNotFoundError, resolve_model_path

    try:
        path = resolve_model_path(config.model_path)
    except ModelNotFoundError as exc:
        return CheckItem(FAIL, "Hand-landmark model", str(exc))
    return CheckItem(OK, "Hand-landmark model", f"{path} ({path.stat().st_size / 1e6:.1f} MB)")


def _screen_check() -> CheckItem:
    try:
        from nocursor.controller import PyAutoGUIBackend

        width, height = PyAutoGUIBackend().screen_size()  # read-only
    except Exception as exc:
        return CheckItem(WARN, "Screen size", f"could not be read ({exc})")
    return CheckItem(OK, "Screen size", f"{width}x{height} (primary monitor only)")


def _hotkey_check() -> CheckItem:
    from nocursor.emergency import system_key_state

    if system_key_state() is None:
        return CheckItem(
            WARN, "Global hotkeys",
            "unsupported on this platform; use Esc/Q in the preview window or Ctrl+C to stop.",
        )
    return CheckItem(
        OK, "Global hotkeys", "available (emergency stop and pause work from any window)"
    )


def _calibration_check(config: Config) -> CheckItem:
    from nocursor.calibration import CalibrationStore, default_calibration_path

    if not config.use_calibration:
        return CheckItem(INFO, "Calibration", "ignored (--no-calibration)")
    store = CalibrationStore(config.calibration_path or default_calibration_path())
    loaded = store.load(config.camera_index, config.frame_width / config.frame_height)
    if loaded.message:
        return CheckItem(WARN, "Calibration", loaded.message)
    if loaded.region is None:
        return CheckItem(
            INFO, "Calibration", "none saved; the default mapping is used (run with --calibrate)"
        )
    return CheckItem(OK, "Calibration", f"a valid calibration is saved ({store.path.name})")


def _log_check(config: Config) -> CheckItem:
    path = config.log_file
    assert path is not None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        probe = path.parent / ".nocursor_write_test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return CheckItem(WARN, "Log file", f"{path} is not writable ({exc})")
    return CheckItem(OK, "Log file", f"{path} (rotating, at most about 4 MB)")


def format_checks(items: list[CheckItem]) -> str:
    width = max(len(i.name) for i in items)
    lines = ["NoCursor installation check (the camera is not opened, no input is sent)", ""]
    lines += [f"{_MARK[i.status]} {i.name:<{width}}  {i.detail}" for i in items]
    failed = [i for i in items if i.status == FAIL]
    lines += ["", f"{len(failed)} problem(s) must be fixed." if failed else "Ready to run."]
    return "\n".join(lines)


def checks_exit_code(items: list[CheckItem]) -> int:
    return 1 if any(i.status == FAIL for i in items) else 0
