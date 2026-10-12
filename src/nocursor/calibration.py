"""Four-corner calibration: validation, persistence, and the interactive wizard.

Calibration stores one axis-aligned rectangle of the camera view that should map
onto the whole screen. Coordinates are in the *mirrored preview* space the user
sees (x grows to the user's right), the same space ``CursorMapper`` uses after
mirroring. Nothing but the rectangle, the camera index and the frame aspect ratio
is stored: no images and no landmark history.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from nocursor.mapping import ViewRegion
from nocursor.paths import runtime_root

logger = logging.getLogger(__name__)

CALIBRATION_VERSION = 1
COORDINATE_SPACE = "mirrored_view"
DEFAULT_FILENAME = "calibration.local.json"
MAX_FILE_BYTES = 8192

MIN_EXTENT = 0.20  # the region must span at least 20% of the view in each direction
MIN_CORNER_SEPARATION = 0.05  # distinct corners must be at least 5% of the view apart
MAX_CORNER_MISALIGNMENT = 0.35  # a corner may stray at most 35% of the extent from its edge
ASPECT_TOLERANCE = 0.02

CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


class CalibrationError(ValueError):
    """Calibration data is missing, malformed, or geometrically unusable."""


# ---------------------------------------------------------------- validation
def validate_region(region: ViewRegion) -> ViewRegion:
    """Return ``region`` if it is big enough to use, else raise CalibrationError."""
    if region.width < MIN_EXTENT or region.height < MIN_EXTENT:
        raise CalibrationError(
            f"calibrated area is too small ({region.width:.0%} x {region.height:.0%} of the "
            f"view; need at least {MIN_EXTENT:.0%} in each direction)"
        )
    return region


def region_from_corners(points: Sequence[tuple[float, float]]) -> ViewRegion:
    """Build a region from corners ordered top-left, top-right, bottom-right, bottom-left."""
    if len(points) != 4:
        raise CalibrationError(f"need exactly 4 corner points, got {len(points)}")
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise CalibrationError("a corner point is outside the camera view or not a number")
    for i in range(4):
        for j in range(i + 1, 4):
            if math.dist(points[i], points[j]) < MIN_CORNER_SEPARATION:
                raise CalibrationError(
                    f"the {CORNER_NAMES[i]} and {CORNER_NAMES[j]} corners are too close together"
                )
    (tl, tr, br, bl) = points
    left, right = (tl[0] + bl[0]) / 2, (tr[0] + br[0]) / 2
    top, bottom = (tl[1] + tr[1]) / 2, (bl[1] + br[1]) / 2
    if not (left < right and top < bottom):
        raise CalibrationError(
            "corners are in the wrong order or mirrored; expected top-left, top-right, "
            "bottom-right, bottom-left as seen in the preview"
        )
    width, height = right - left, bottom - top
    if (
        abs(tl[0] - bl[0]) > MAX_CORNER_MISALIGNMENT * width
        or abs(tr[0] - br[0]) > MAX_CORNER_MISALIGNMENT * width
        or abs(tl[1] - tr[1]) > MAX_CORNER_MISALIGNMENT * height
        or abs(bl[1] - br[1]) > MAX_CORNER_MISALIGNMENT * height
    ):
        raise CalibrationError("the four corners do not form a roughly rectangular area")
    return validate_region(ViewRegion(left, top, right, bottom))


# --------------------------------------------------------------- persistence
def default_calibration_path() -> Path:
    """Local runtime file inside the project (or the working directory if not a checkout)."""
    return runtime_root() / DEFAULT_FILENAME


@dataclass(frozen=True)
class LoadResult:
    region: ViewRegion | None
    message: str | None  # None when there was simply no calibration file


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise CalibrationError(f"{name} must be a finite number")
    return float(value)


def _parse(data: object) -> tuple[ViewRegion, int, float]:
    if not isinstance(data, dict):
        raise CalibrationError("top level must be an object")
    version = data.get("version")
    if isinstance(version, bool) or version != CALIBRATION_VERSION:
        raise CalibrationError(f"unsupported calibration version {version!r}")
    if data.get("coordinate_space") != COORDINATE_SPACE:
        raise CalibrationError("unsupported coordinate space")
    region_data, camera = data.get("region"), data.get("camera")
    if not isinstance(region_data, dict) or not isinstance(camera, dict):
        raise CalibrationError("region and camera sections are required")
    try:
        region = ViewRegion(**{k: _number(region_data.get(k), k) for k in
                               ("left", "top", "right", "bottom")})
    except ValueError as exc:  # includes CalibrationError
        raise CalibrationError(str(exc)) from exc
    validate_region(region)
    index = camera.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise CalibrationError("camera.index must be a non-negative integer")
    aspect = _number(camera.get("aspect_ratio"), "camera.aspect_ratio")
    if aspect <= 0:
        raise CalibrationError("camera.aspect_ratio must be positive")
    return region, index, aspect


class CalibrationStore:
    """Reads and writes the calibration file. ``load`` never raises."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load(self, camera_index: int, aspect_ratio: float) -> LoadResult:
        if not self._path.is_file():
            return LoadResult(None, None)
        fallback = f"using the default mapping; recalibrate with --calibrate ({self._path})"
        try:
            if self._path.stat().st_size > MAX_FILE_BYTES:
                raise CalibrationError("file is unexpectedly large")
            region, index, aspect = _parse(json.loads(self._path.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            return LoadResult(None, f"calibration file is unusable ({exc}); {fallback}")
        if index != camera_index:
            return LoadResult(
                None, f"calibration was made for camera {index}, not {camera_index}; {fallback}"
            )
        if abs(aspect - aspect_ratio) > ASPECT_TOLERANCE * aspect_ratio:
            return LoadResult(
                None, f"calibration was made at a different frame shape; {fallback}"
            )
        return LoadResult(region, None)

    def save(self, region: ViewRegion, camera_index: int, aspect_ratio: float) -> None:
        """Validate and write atomically. Raises CalibrationError or OSError."""
        validate_region(region)
        if camera_index < 0 or not (math.isfinite(aspect_ratio) and aspect_ratio > 0):
            raise CalibrationError("invalid camera information")
        payload = {
            "version": CALIBRATION_VERSION,
            "coordinate_space": COORDINATE_SPACE,
            "region": {
                "left": region.left, "top": region.top,
                "right": region.right, "bottom": region.bottom,
            },
            "camera": {"index": camera_index, "aspect_ratio": aspect_ratio},
        }  # fmt: skip
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp = self._path.with_name(self._path.name + ".tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, self._path)

    def delete(self) -> bool:
        """Remove the file. True if one existed."""
        try:
            self._path.unlink()
        except FileNotFoundError:
            return False
        return True


# -------------------------------------------------------------------- wizard
class WizardState(Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class WizardView:
    """What the preview needs to draw the wizard."""

    state: WizardState
    title: str
    instruction: str
    step: int  # 1-based corner being captured
    total: int
    captured: tuple[tuple[float, float], ...]  # mirrored view coordinates
    stability: float  # 0..1: how close the fingertip is to a capturable, steady reading
    notice: str | None


class CalibrationWizard:
    """Collects four corner points, each captured only on an explicit confirmation.

    Feed it the fingertip position (mirrored view coordinates) every frame, or
    ``None`` whenever tracking is not reliable. ``confirm()`` captures the corner
    only if the last ``stable_frames`` readings are all reliable and within
    ``tolerance`` of each other; it records the median of those readings.
    The wizard performs no desktop input; it is pure logic.
    """

    TITLE = (
        "CALIBRATION: mark the 4 corners of the area your hand will use "
        "(as seen in this mirrored preview)"
    )

    def __init__(self, stable_frames: int = 10, tolerance: float = 0.02) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0")
        self._stable_frames = stable_frames
        self._tolerance = tolerance
        self._buffer: deque[tuple[float, float]] = deque(maxlen=stable_frames)
        self._captured: list[tuple[float, float]] = []
        self._state = WizardState.IDLE
        self._region: ViewRegion | None = None
        self._notice: str | None = None

    @property
    def state(self) -> WizardState:
        return self._state

    @property
    def active(self) -> bool:
        return self._state is WizardState.COLLECTING

    @property
    def region(self) -> ViewRegion | None:
        """The result once SUCCEEDED."""
        return self._region

    @property
    def notice(self) -> str | None:
        return self._notice

    def start(self) -> None:
        self._buffer.clear()
        self._captured.clear()
        self._region = None
        self._notice = None
        self._state = WizardState.COLLECTING

    def cancel(self) -> None:
        if self._state is WizardState.COLLECTING:
            self._buffer.clear()
            self._captured.clear()
            self._notice = "Calibration cancelled; nothing was changed."
            self._state = WizardState.CANCELLED

    def acknowledge(self) -> None:
        """Return to IDLE after the outcome (success / failure / cancel) has been reported."""
        if self._state is not WizardState.COLLECTING:
            self._state = WizardState.IDLE

    def feed(self, tip: tuple[float, float] | None) -> None:
        if self._state is not WizardState.COLLECTING:
            return
        if tip is None or not all(math.isfinite(v) for v in tip):
            self._buffer.clear()
            return
        self._buffer.append(tip)

    @property
    def stability(self) -> float:
        if len(self._buffer) < self._stable_frames:
            return len(self._buffer) / self._stable_frames * 0.5
        return 1.0 if self._is_steady() else 0.5

    def confirm(self) -> bool:
        """Capture the current corner if the fingertip has been steady. True if captured."""
        if self._state is not WizardState.COLLECTING:
            return False
        if len(self._buffer) < self._stable_frames:
            self._notice = "Waiting for a steady, reliably tracked fingertip..."
            return False
        if not self._is_steady():
            self._notice = "Too much movement: hold your fingertip still, then press SPACE."
            return False
        xs = sorted(p[0] for p in self._buffer)
        ys = sorted(p[1] for p in self._buffer)
        point = (xs[len(xs) // 2], ys[len(ys) // 2])
        for earlier in self._captured:
            if math.dist(point, earlier) < MIN_CORNER_SEPARATION:
                self._notice = "That is too close to a corner you already marked."
                return False
        self._captured.append(point)
        self._buffer.clear()
        self._notice = None
        if len(self._captured) == 4:
            self._finish()
        return True

    def view(self) -> WizardView:
        step = min(len(self._captured) + 1, 4)
        if self._state is WizardState.COLLECTING:
            corner = CORNER_NAMES[len(self._captured)].upper()
            instruction = (
                f"Step {step}/4: hold your index fingertip at the {corner} corner, "
                "keep still, press SPACE. Backspace/X cancels."
            )
        else:
            instruction = self._notice or ""
        return WizardView(
            self._state, self.TITLE, instruction, step, 4,
            tuple(self._captured), self.stability, self._notice,
        )  # fmt: skip

    def _is_steady(self) -> bool:
        xs = [p[0] for p in self._buffer]
        ys = [p[1] for p in self._buffer]
        return max(xs) - min(xs) <= self._tolerance and max(ys) - min(ys) <= self._tolerance

    def _finish(self) -> None:
        try:
            self._region = region_from_corners(self._captured)
        except CalibrationError as exc:
            self._region = None
            self._notice = f"Calibration rejected: {exc}. Press C to try again."
            self._state = WizardState.FAILED
            return
        self._notice = "Calibration complete."
        self._state = WizardState.SUCCEEDED
