"""``nocursor --print-settings``: the effective configuration, grouped, with changes marked."""

from __future__ import annotations

from dataclasses import fields

from nocursor.config import Config
from nocursor.diagnostics import mode_label

SETTING_GROUPS: dict[str, tuple[str, ...]] = {
    "Camera and model": (
        "camera_index", "frame_width", "frame_height", "max_read_failures", "model_path",
        "min_confidence", "min_hand_size", "stable_frames",
    ),
    "Pointer": ("active_margin", "smoothing"),
    "Pinch (click and drag)": (
        "pinch_use_depth", "pinch_threshold", "pinch_confirm_frames", "pinch_release_factor",
        "pinch_release_frames", "click_cooldown_s", "drag_start_px",
    ),
    "Scrolling": (
        "scroll_stable_frames", "scroll_dead_zone", "scroll_speed", "scroll_interval_s",
        "scroll_invert",
    ),
    "Calibration": ("use_calibration", "start_calibration", "calibration_path"),
    "Safety": ("dry_run", "emergency_hotkey", "pause_hotkey"),
    "Diagnostics, performance and logging": ("diagnostics", "show_perf", "log_file"),
    "Benchmark (always dry-run)": (
        "benchmark", "benchmark_frames", "benchmark_seconds", "benchmark_video",
        "benchmark_output",
    ),
}  # fmt: skip


def ungrouped_fields() -> list[str]:
    """Config fields missing from every group (a test keeps this empty)."""
    grouped = {name for names in SETTING_GROUPS.values() for name in names}
    return [f.name for f in fields(Config) if f.name not in grouped]


def describe_settings(config: Config) -> str:
    defaults = Config()
    lines = [
        "NoCursor effective settings ('*' marks values that differ from the defaults)",
        "",
        f"Pinch measurement: {mode_label(config.pinch_use_depth)}",
    ]
    for group, names in SETTING_GROUPS.items():
        lines += ["", f"{group}:"]
        for name in names:
            value, default = getattr(config, name), getattr(defaults, name)
            marker = "*" if value != default else " "
            lines.append(f"  {marker} {name:<22} {_show(value)}")
    return "\n".join(lines)


def _show(value: object) -> str:
    if value is None:
        return "(not set)"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)
