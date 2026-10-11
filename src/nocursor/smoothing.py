"""Cursor smoothing."""

from __future__ import annotations

import numpy as np


class ExponentialSmoother:
    """Exponential moving average over a 2D point.

    ``smoothing`` is in [0, 1): 0 passes values through unchanged, values closer
    to 1 respond more slowly (smoother but laggier).
    """

    def __init__(self, smoothing: float) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be within [0, 1)")
        self._alpha = 1.0 - smoothing
        self._state: np.ndarray | None = None

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        value = np.asarray(point, dtype=float)
        if self._state is None:
            self._state = value
        else:
            self._state = self._alpha * value + (1.0 - self._alpha) * self._state
        return float(self._state[0]), float(self._state[1])

    def reset(self) -> None:
        """Forget history so the next sample is taken as-is (e.g. after tracking loss)."""
        self._state = None
