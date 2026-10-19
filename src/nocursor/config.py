"""Validated runtime configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

from nocursor.emergency import parse_hotkey

BENCHMARK_MODES = ("synthetic", "video", "live")


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


@dataclass
class Config:
    # Camera
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    max_read_failures: int = 30

    # Hand tracking
    model_path: Path | None = None
    min_confidence: float = 0.6
    min_hand_size: float = 0.05
    stable_frames: int = 3

    # Cursor mapping / smoothing
    active_margin: float = 0.15
    smoothing: float = 0.5

    # Pinch-to-click
    pinch_threshold: float = 0.20
    pinch_release_factor: float = 1.5
    pinch_confirm_frames: int = 5
    click_cooldown_s: float = 0.4
    pinch_release_frames: int = 2  # frames the fingers must stay open before the button is released
    # False (default): thumb-index distance in the image plane (2D). True (experimental): also
    # use MediaPipe's coarse z estimate, which keeps touching fingertips above the threshold
    # on some webcams.
    pinch_use_depth: bool = False
    # Opt-in right click: a thumb-middle pinch made with the index, ring and little fingers
    # extended. Off unless --right-click is given; it reuses the pinch thresholds above.
    right_click: bool = False
    diagnostics: bool = False  # show/log live pinch internals (read-only; changes no behaviour)

    # Drag: how far (screen pixels, smoothed) the fingertip must move after the press
    # before the pinch becomes a drag. ~1.5% of a 1920 px wide screen.
    drag_start_px: float = 30.0

    # Two-finger scroll. Distances are in hand sizes (wrist to middle knuckle).
    scroll_stable_frames: int = 6  # frames the pose must hold before scrolling starts
    scroll_dead_zone: float = 0.10  # movement below this is ignored
    scroll_speed: float = 8.0  # wheel clicks per hand size of movement
    scroll_interval_s: float = 0.05  # minimum time between scroll events
    scroll_invert: bool = False  # False: hand up scrolls up

    # Calibration
    calibration_path: Path | None = None  # None: calibration.local.json in the project
    use_calibration: bool = True  # False ignores any saved calibration
    start_calibration: bool = False  # open the calibration wizard at launch

    # Safety
    emergency_hotkey: str = "ctrl+alt+q"
    pause_hotkey: str = "ctrl+alt+p"
    dry_run: bool = False

    # Performance and logging
    show_perf: bool = False  # show rolling software timings in the preview
    log_file: Path | None = None  # opt-in rotating log file (None: no file is written)

    # Benchmarks. All modes are dry-run: they can never send desktop input.
    benchmark: str | None = None  # "synthetic", "video" or "live"
    benchmark_frames: int | None = None  # synthetic / video: frames to process
    benchmark_seconds: float | None = None  # live: duration
    benchmark_video: Path | None = None  # video: a clip you supply
    benchmark_output: Path | None = None  # where to write the JSON report

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        def require(condition: bool, message: str) -> None:
            if not condition:
                raise ConfigError(message)

        require(self.camera_index >= 0, "camera_index must be >= 0")
        require(self.frame_width > 0 and self.frame_height > 0, "frame size must be positive")
        require(self.max_read_failures >= 1, "max_read_failures must be >= 1")
        require(0.0 <= self.min_confidence <= 1.0, "min_confidence must be within [0, 1]")
        require(0.0 <= self.min_hand_size < 1.0, "min_hand_size must be within [0, 1)")
        require(self.stable_frames >= 1, "stable_frames must be >= 1")
        require(0.0 <= self.active_margin < 0.5, "active_margin must be within [0, 0.5)")
        require(0.0 <= self.smoothing < 1.0, "smoothing must be within [0, 1)")
        require(self.pinch_threshold > 0.0, "pinch_threshold must be > 0")
        require(self.pinch_release_factor > 1.0, "pinch_release_factor must be > 1")
        require(self.pinch_confirm_frames >= 1, "pinch_confirm_frames must be >= 1")
        require(self.click_cooldown_s >= 0.0, "click_cooldown_s must be >= 0")
        require(
            self.pinch_release_frames >= 1,
            "pinch_release_frames must be >= 1 (--pinch-release-frames)",
        )
        require(
            0.0 < self.drag_start_px <= 500.0,
            "drag_start_px must be within (0, 500] screen pixels (--drag-start)",
        )
        require(
            self.scroll_stable_frames >= 1,
            "scroll_stable_frames must be >= 1 (--scroll-stable-frames)",
        )
        require(
            0.0 < self.scroll_dead_zone <= 1.0,
            "scroll_dead_zone must be within (0, 1] hand sizes (--scroll-dead-zone)",
        )
        require(
            0.0 < self.scroll_speed <= 100.0,
            "scroll_speed must be within (0, 100] clicks per hand size (--scroll-speed)",
        )
        require(
            0.01 <= self.scroll_interval_s <= 2.0,
            "scroll_interval_s must be within [0.01, 2] seconds (--scroll-interval)",
        )
        keys = {}
        for name in ("emergency_hotkey", "pause_hotkey"):
            try:
                keys[name] = frozenset(parse_hotkey(getattr(self, name)))
            except ValueError as exc:
                raise ConfigError(f"invalid {name}: {exc}") from exc
        require(
            keys["emergency_hotkey"] != keys["pause_hotkey"],
            "pause_hotkey must differ from emergency_hotkey (--pause-key / --stop-key)",
        )
        self._validate_benchmark(require)

    def _validate_benchmark(self, require: Callable[[bool, str], None]) -> None:
        given = {
            "--benchmark-frames": self.benchmark_frames is not None,
            "--benchmark-seconds": self.benchmark_seconds is not None,
            "--benchmark-video": self.benchmark_video is not None,
            "--benchmark-output": self.benchmark_output is not None,
        }
        if self.benchmark is None:
            stray = [flag for flag, present in given.items() if present]
            require(not stray, f"{', '.join(stray)} only apply together with --benchmark")
            return
        require(
            self.benchmark in BENCHMARK_MODES,
            f"benchmark must be one of {', '.join(BENCHMARK_MODES)} (--benchmark)",
        )
        require(
            self.benchmark_frames is None or 1 <= self.benchmark_frames <= 1_000_000,
            "benchmark_frames must be within [1, 1000000] (--benchmark-frames)",
        )
        require(
            self.benchmark_seconds is None or 0.0 < self.benchmark_seconds <= 3600.0,
            "benchmark_seconds must be within (0, 3600] seconds (--benchmark-seconds)",
        )
        require(
            not (self.benchmark == "live" and given["--benchmark-frames"]),
            "--benchmark-frames applies to synthetic and video runs; live uses --benchmark-seconds",
        )
        require(
            not (self.benchmark != "live" and given["--benchmark-seconds"]),
            "--benchmark-seconds applies to live runs; synthetic and video use --benchmark-frames",
        )
        require(
            self.benchmark != "video" or self.benchmark_video is not None,
            "--benchmark video needs a clip: --benchmark-video PATH",
        )
        require(
            self.benchmark == "video" or self.benchmark_video is None,
            "--benchmark-video only applies to --benchmark video",
        )
        require(
            not self.start_calibration,
            "--calibrate cannot be combined with --benchmark (benchmarks send no input)",
        )

    @classmethod
    def from_overrides(cls, **overrides: object) -> Config:
        """Build a config from defaults, applying only non-None overrides."""
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in overrides.items() if v is not None})  # type: ignore[arg-type]
