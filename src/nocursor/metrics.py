"""Latency statistics: nearest-rank percentiles and a fixed-size rolling window.

Percentiles use the nearest-rank method on the sorted samples: the p-th percentile of n
samples is the value at rank ceil(p/100 * n). A percentile is only reported when there are
enough samples for it to mean something (p95 needs 20, p99 needs 100); otherwise it is
``None`` and shown as "n/a" instead of a misleading number.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

P95_MIN_SAMPLES = 20
P99_MIN_SAMPLES = 100


def nearest_rank(sorted_values: Sequence[float], percent: int) -> float:
    """The ``percent``-th percentile (integer 1-100) of already sorted values."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("no samples")
    if not 1 <= percent <= 100:
        raise ValueError("percent must be within 1..100")
    rank = max(1, (n * percent + 99) // 100)  # integer ceil(n * percent / 100)
    return sorted_values[min(rank, n) - 1]


@dataclass(frozen=True)
class StatSummary:
    """Summary of a set of durations in milliseconds."""

    n: int
    mean: float
    minimum: float
    p50: float
    p95: float | None  # None when n < P95_MIN_SAMPLES
    p99: float | None  # None when n < P99_MIN_SAMPLES
    maximum: float

    def as_dict(self, digits: int = 3) -> dict[str, float | int | None]:
        def r(value: float | None) -> float | None:
            return None if value is None else round(value, digits)

        return {
            "n": self.n, "mean": r(self.mean), "min": r(self.minimum), "p50": r(self.p50),
            "p95": r(self.p95), "p99": r(self.p99), "max": r(self.maximum),
        }  # fmt: skip


def summarize(samples: Iterable[float]) -> StatSummary | None:
    """Summarise durations (ms). Returns None when there are no samples."""
    values = sorted(samples)
    n = len(values)
    if n == 0:
        return None
    return StatSummary(
        n=n,
        mean=sum(values) / n,
        minimum=values[0],
        p50=nearest_rank(values, 50),
        p95=nearest_rank(values, 95) if n >= P95_MIN_SAMPLES else None,
        p99=nearest_rank(values, 99) if n >= P99_MIN_SAMPLES else None,
        maximum=values[-1],
    )


class RollingWindow:
    """The most recent ``size`` samples, so one slow frame cannot dominate the result."""

    def __init__(self, size: int = 300) -> None:
        if size < 1:
            raise ValueError("size must be >= 1")
        self._values: deque[float] = deque(maxlen=size)

    def add(self, value: float) -> None:
        self._values.append(value)

    def clear(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)

    def summary(self) -> StatSummary | None:
        return summarize(self._values)


def format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"
