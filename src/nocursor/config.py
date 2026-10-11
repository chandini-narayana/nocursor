"""Validated runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from nocursor.emergency import parse_hotkey


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

    # Safety
    emergency_hotkey: str = "ctrl+alt+q"
    dry_run: bool = False

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
        try:
            parse_hotkey(self.emergency_hotkey)
        except ValueError as exc:
            raise ConfigError(f"invalid emergency_hotkey: {exc}") from exc

    @classmethod
    def from_overrides(cls, **overrides: object) -> Config:
        """Build a config from defaults, applying only non-None overrides."""
        known = {f.name for f in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in overrides.items() if v is not None})  # type: ignore[arg-type]
