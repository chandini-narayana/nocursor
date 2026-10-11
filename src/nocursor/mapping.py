"""Map normalized camera coordinates to desktop pixel coordinates."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ViewRegion:
    """The camera area that maps onto the whole screen, in *mirrored preview* coordinates.

    ``left``/``right``/``top``/``bottom`` are fractions of the preview image the user
    sees (x grows to the user's right). ``left < right`` and ``top < bottom``.
    """

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
            raise ValueError("region edges must be finite and within [0, 1]")
        if not (self.left < self.right and self.top < self.bottom):
            raise ValueError("region must have left < right and top < bottom")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @classmethod
    def from_margin(cls, margin: float) -> ViewRegion:
        return cls(margin, margin, 1.0 - margin, 1.0 - margin)


class CursorMapper:
    """Maps a normalized camera point to screen pixels.

    By default only the central "active region" of the camera frame is used
    (``margin`` is the fraction trimmed from each side) so the screen edges are
    reachable without moving the hand to the very edge of the camera view. A
    calibrated ``bounds`` region replaces the margin-derived one. The x axis is
    mirrored by default so moving the hand to the user's right moves the cursor
    right. Output is clamped ``edge_inset`` pixels inside the screen so hand
    tracking cannot land exactly on a corner (which would trip PyAutoGUI's
    corner fail-safe). Points outside the region clamp to its edge.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        margin: float = 0.15,
        mirror: bool = True,
        edge_inset: int = 1,
        bounds: ViewRegion | None = None,
    ) -> None:
        if screen_width <= 0 or screen_height <= 0:
            raise ValueError("screen dimensions must be positive")
        if not 0.0 <= margin < 0.5:
            raise ValueError("margin must be within [0, 0.5)")
        if edge_inset < 0 or 2 * edge_inset >= min(screen_width, screen_height):
            raise ValueError("edge_inset is out of range for this screen")
        self._width = screen_width
        self._height = screen_height
        self._region = bounds if bounds is not None else ViewRegion.from_margin(margin)
        self._mirror = mirror
        self._inset = edge_inset

    @property
    def screen_size(self) -> tuple[int, int]:
        return self._width, self._height

    @property
    def region(self) -> ViewRegion:
        return self._region

    def map(self, x: float, y: float) -> tuple[float, float]:
        """Return (screen_x, screen_y) in pixels for normalized camera (x, y)."""
        if self._mirror:
            x = 1.0 - x
        r = self._region
        nx = _clamp((x - r.left) / r.width, 0.0, 1.0)
        ny = _clamp((y - r.top) / r.height, 0.0, 1.0)
        lo_x, hi_x = self._inset, self._width - 1 - self._inset
        lo_y, hi_y = self._inset, self._height - 1 - self._inset
        return lo_x + nx * (hi_x - lo_x), lo_y + ny * (hi_y - lo_y)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
