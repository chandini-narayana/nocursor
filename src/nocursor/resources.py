"""Process CPU and memory usage from the standard library only.

CPU
    ``time.process_time()`` is the CPU time (user + kernel, summed over every thread) this
    process has used. Utilisation is a ratio of two *deltas* over the same wall-clock interval:

        cpu_percent_of_one_core = 100 * delta_cpu_time / delta_wall_time
        cpu_percent_of_machine  = cpu_percent_of_one_core / logical_cpu_count

    The first can exceed 100 when several threads run at once (MediaPipe uses threads); the
    second is normalised to the whole machine and cannot. Total CPU time in seconds is
    reported separately, because it is an absolute amount, not a rate. On Windows the CPU
    clock ticks about every 15.6 ms, so very short sampling intervals are noisy: samples are
    taken at least ``min_interval_s`` (default 1 s) apart.

Memory
    Windows: working set (resident memory) and its peak via ``GetProcessMemoryInfo``.
    Linux: resident memory from ``/proc/self/statm`` plus the peak from ``getrusage``.
    Other POSIX: peak only. Anything unavailable is ``None``; nothing is guessed.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryInfo:
    rss_bytes: int | None  # current resident memory
    peak_rss_bytes: int | None


def process_memory() -> MemoryInfo:
    """Current and peak resident memory of this process, where the platform provides it."""
    try:
        if sys.platform == "win32":
            return _windows_memory()
        return _posix_memory()
    except Exception:  # never let a diagnostic break the app
        return MemoryInfo(None, None)


def _windows_memory() -> MemoryInfo:
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]  # fmt: skip

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    psapi = ctypes.WinDLL("psapi", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        return MemoryInfo(None, None)
    return MemoryInfo(int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize))


def _posix_memory() -> MemoryInfo:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak) if sys.platform == "darwin" else int(peak) * 1024  # KiB on Linux
    current = None
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            current = int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    return MemoryInfo(current, peak_bytes)


@dataclass(frozen=True)
class ResourceSummary:
    """Resource use over the sampled interval. Rates come from deltas, not absolute totals."""

    wall_s: float
    cpu_time_s: float  # absolute CPU seconds used by the process over the interval
    cpu_percent_of_one_core: float | None  # may exceed 100 (multi-threaded)
    cpu_percent_of_machine: float | None  # normalised by the logical CPU count
    cpu_percent_of_one_core_peak: float | None  # highest single sampling interval
    logical_cpus: int
    samples: int
    rss_bytes: int | None
    peak_rss_bytes: int | None

    def as_dict(self) -> dict[str, float | int | None]:
        def r(value: float | None, digits: int = 2) -> float | None:
            return None if value is None else round(value, digits)

        return {
            "wall_s": r(self.wall_s), "cpu_time_s": r(self.cpu_time_s, 3),
            "cpu_percent_of_one_core": r(self.cpu_percent_of_one_core),
            "cpu_percent_of_machine": r(self.cpu_percent_of_machine),
            "cpu_percent_of_one_core_peak": r(self.cpu_percent_of_one_core_peak),
            "logical_cpus": self.logical_cpus, "sampling_intervals": self.samples,
            "rss_bytes": self.rss_bytes, "peak_rss_bytes": self.peak_rss_bytes,
        }  # fmt: skip


class ResourceSampler:
    """Samples CPU time and memory at most once per ``min_interval_s`` from the frame loop."""

    def __init__(
        self,
        min_interval_s: float = 1.0,
        wall_clock: Callable[[], float] = time.perf_counter,
        cpu_clock: Callable[[], float] = time.process_time,
        memory: Callable[[], MemoryInfo] = process_memory,
        logical_cpus: int | None = None,
    ) -> None:
        self._min_interval_s = min_interval_s
        self._wall = wall_clock
        self._cpu = cpu_clock
        self._memory = memory
        self._cpus = max(1, logical_cpus or os.cpu_count() or 1)
        self._start_wall: float | None = None
        self._start_cpu = 0.0
        self._last_wall = 0.0
        self._last_cpu = 0.0
        self._peak_percent: float | None = None
        self._samples = 0
        self._last_memory = MemoryInfo(None, None)

    def start(self) -> None:
        self._start_wall = self._last_wall = self._wall()
        self._start_cpu = self._last_cpu = self._cpu()
        self._peak_percent = None
        self._samples = 0

    def maybe_sample(self) -> None:
        """Take a sample if at least ``min_interval_s`` has passed since the last one."""
        if self._start_wall is None:
            self.start()
            return
        now = self._wall()
        if now - self._last_wall >= self._min_interval_s:
            self._sample(now)

    def _sample(self, now: float, closing: bool = False) -> None:
        cpu = self._cpu()
        interval = now - self._last_wall
        # A closing partial interval shorter than the sampling interval is too noisy (the CPU
        # clock is coarse) to count toward the peak; it still counts toward the totals.
        if interval > 0 and (not closing or interval >= self._min_interval_s):
            percent = 100.0 * (cpu - self._last_cpu) / interval
            if self._peak_percent is None or percent > self._peak_percent:
                self._peak_percent = percent
        self._last_wall, self._last_cpu = now, cpu
        self._samples += 1
        self._last_memory = self._memory()

    def summary(self) -> ResourceSummary | None:
        """Overall usage from start to the final sample. None if never started."""
        if self._start_wall is None:
            return None
        self._sample(self._wall(), closing=True)  # close the interval so totals are up to date
        wall = self._last_wall - self._start_wall
        cpu = self._last_cpu - self._start_cpu
        one_core = 100.0 * cpu / wall if wall > 0 else None
        return ResourceSummary(
            wall_s=wall,
            cpu_time_s=cpu,
            cpu_percent_of_one_core=one_core,
            cpu_percent_of_machine=None if one_core is None else one_core / self._cpus,
            cpu_percent_of_one_core_peak=self._peak_percent,
            logical_cpus=self._cpus,
            samples=self._samples,
            rss_bytes=self._last_memory.rss_bytes,
            peak_rss_bytes=self._last_memory.peak_rss_bytes,
        )
