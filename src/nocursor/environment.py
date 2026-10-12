"""Facts about the machine and installation, for benchmark reports and ``--check``.

Nothing here identifies the user: no host name, user name, or paths are collected, only
the Python / OS / CPU description and installed package versions.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import sys
from dataclasses import dataclass

MIN_PYTHON = (3, 10)

# import name -> (pip distribution to install, why NoCursor needs it)
REQUIRED_IMPORTS = {
    "cv2": ("opencv-contrib-python", "camera capture and the preview window"),
    "mediapipe": ("mediapipe", "hand-landmark detection"),
    "numpy": ("numpy", "coordinate maths"),
    "pyautogui": ("pyautogui", "mouse control"),
}
OPENCV_DISTRIBUTIONS = (
    "opencv-contrib-python", "opencv-python", "opencv-contrib-python-headless",
    "opencv-python-headless",
)  # fmt: skip


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_opencv() -> dict[str, str]:
    """Every OpenCV distribution installed (more than one can conflict)."""
    found = {}
    for name in OPENCV_DISTRIBUTIONS:
        version = package_version(name)
        if version is not None:
            found[name] = version
    return found


def missing_dependencies() -> list[tuple[str, str]]:
    """(distribution, purpose) for each required package that cannot be imported."""
    return [
        (dist, purpose)
        for module, (dist, purpose) in REQUIRED_IMPORTS.items()
        if importlib.util.find_spec(module) is None
    ]


def missing_dependencies_message(missing: list[tuple[str, str]]) -> str:
    names = ", ".join(dist for dist, _ in missing)
    detail = "; ".join(f"{dist} ({purpose})" for dist, purpose in missing)
    return (
        f"missing required package(s): {names}. NoCursor needs {detail}. "
        "Activate your virtual environment and run: python -m pip install -r requirements.txt"
    )


def opencv_gui_support() -> bool | None:
    """True if this OpenCV build can open windows, False if headless, None if unknown."""
    try:
        import cv2

        info = cv2.getBuildInformation()
    except Exception:
        return None
    for line in info.splitlines():
        stripped = line.strip()
        if stripped.startswith("GUI:"):
            return stripped.removeprefix("GUI:").strip().upper() not in ("", "NONE")
    return None


@dataclass(frozen=True)
class PythonCheck:
    ok: bool
    version: str


def python_check() -> PythonCheck:
    version = ".".join(str(p) for p in sys.version_info[:3])
    return PythonCheck(sys.version_info[:2] >= MIN_PYTHON, version)


def collect_environment() -> dict[str, object]:
    """Describe the platform and package versions (no names, hosts or paths)."""
    packages: dict[str, str | None] = {
        "nocursor": package_version("nocursor"),
        "numpy": package_version("numpy"),
        "mediapipe": package_version("mediapipe"),
        "pyautogui": package_version("pyautogui"),
    }
    for name, version in installed_opencv().items():
        packages[name] = version
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "packages": packages,
    }
