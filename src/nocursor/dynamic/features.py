"""Turning a landmark sequence into model-ready features (feature schema version 1).

Pure NumPy; deterministic; no randomness. The steps are:

1. Convert raw landmarks to *view* coordinates (x mirrored and aspect-corrected, y down), so one
   unit means the same distance in x and y and the numbers match the mirrored preview.
2. Resample onto a fixed time grid (``steps`` points over ``window_s`` seconds), so the frame rate
   of the camera does not matter. Short tracking dropouts are bridged by linear interpolation;
   anything longer is an error rather than a guess.
3. Normalise by hand size (wrist to middle knuckle) so distance from the camera and the camera's
   frame dimensions do not matter.

What is deliberately *kept*: the palm's trajectory relative to where it started, with its sign.
A swipe to the left therefore has a negative x trajectory and a swipe to the right a positive one,
and a clockwise circle has a positive signed area. Depth (MediaPipe's z) is never used.

Per-step feature vector (``FEATURE_DIM`` = 46 numbers):

* ``traj_x, traj_y``: palm-centroid position relative to the first step, in hand sizes.
* ``vel_x, vel_y``: its velocity, in hand sizes per second.
* ``shape_<i>_x, shape_<i>_y`` for the 21 landmarks: position relative to the palm centroid, in
  hand sizes of that step (the hand's shape, independent of where it is).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from nocursor.dynamic.schema import DEFAULT_WINDOW_S, SequenceRecord
from nocursor.models import MIDDLE_MCP, NUM_LANDMARKS, WRIST

FEATURE_SCHEMA_VERSION = 1
PALM_POINTS = (0, 5, 9, 13, 17)  # wrist and the four finger knuckles
CONVENTION = "view: x mirrored and aspect-corrected, y down; units: hand sizes; z unused"

FEATURE_NAMES: tuple[str, ...] = (
    "traj_x", "traj_y", "vel_x", "vel_y",
    *(f"shape_{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in "xy"),
)  # fmt: skip
FEATURE_DIM = len(FEATURE_NAMES)


class FeatureError(ValueError):
    """A sequence cannot be turned into features; the message says why."""


@dataclass(frozen=True)
class FeatureConfig:
    steps: int = 32  # points on the resampled time grid
    window_s: float = DEFAULT_WINDOW_S  # seconds covered by the grid
    duration_tolerance_s: float = 0.1  # how much shorter than window_s a sequence may be
    max_gap_s: float = 0.2  # longest tracking dropout that is bridged by interpolation
    min_usable_fraction: float = 0.8  # share of frames that must be trusted
    min_hand_size: float = 0.02  # frame heights; a smaller hand is too small to normalise by

    def __post_init__(self) -> None:
        if self.steps < 4:
            raise ValueError("steps must be >= 4")
        if self.window_s <= 0:
            raise ValueError("window_s must be > 0")
        if not (0.0 <= self.duration_tolerance_s < self.window_s):
            raise ValueError("duration_tolerance_s must be within [0, window_s)")
        if self.max_gap_s <= 0:
            raise ValueError("max_gap_s must be > 0")
        if not (0.0 < self.min_usable_fraction <= 1.0):
            raise ValueError("min_usable_fraction must be within (0, 1]")
        if self.min_hand_size <= 0:
            raise ValueError("min_hand_size must be > 0")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def schema_hash(self) -> str:
        """Identifies the exact feature definition; a trained model records it, and inference
        refuses a model whose hash differs from the one it would compute."""
        payload = {
            "version": FEATURE_SCHEMA_VERSION,
            "names": FEATURE_NAMES,
            "palm_points": PALM_POINTS,
            "convention": CONVENTION,
            "config": self.to_dict(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def failure_category(message: str) -> str:
    """A stable name for a ``FeatureError`` message, so counts group instead of splintering on the
    numbers inside the text (for example "hand trusted in only 76.2% of frames")."""
    if message.startswith("tracking dropped"):
        return "tracking dropped for too long"
    if message.startswith("hand trusted in only"):
        return "too few trusted frames"
    if message.startswith("sequence covers") or message.startswith("fewer than 2 frames"):
        return "shorter than the window"
    if message.startswith("hand too small"):
        return "hand too small"
    return "other"


def to_view_coordinates(landmarks: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """Raw ``(..., 21, 3)`` landmarks to view ``(..., 21, 2)``: x mirrored and aspect-corrected.

    One unit is one frame height in both axes. Moving the hand to the user's right (to the right
    in the mirrored preview) increases x.
    """
    view = np.empty(landmarks.shape[:-1] + (2,))
    view[..., 0] = (1.0 - landmarks[..., 0]) * aspect_ratio
    view[..., 1] = landmarks[..., 1]
    return view


def longest_tracking_gap_s(t_s: np.ndarray, usable: np.ndarray) -> float:
    """Longest stretch without a trusted frame, counting the start and end of the sequence."""
    trusted = t_s[usable]
    if trusted.size == 0:
        return float("inf")
    gaps = [trusted[0] - t_s[0], t_s[-1] - trusted[-1]]
    if trusted.size > 1:
        gaps.append(float(np.diff(trusted).max()))
    return float(max(gaps))


def resample_view(seq: SequenceRecord, config: FeatureConfig) -> np.ndarray:
    """View-coordinate landmarks on the fixed time grid, shape ``(steps, 21, 2)``."""
    t_s = (seq.t_ms - seq.t_ms[0]) / 1000.0
    keep = t_s <= config.window_s + 1e-9
    if keep.sum() < 2:
        raise FeatureError("fewer than 2 frames inside the window")
    t_s, usable = t_s[keep], seq.usable[keep]
    if t_s[-1] < config.window_s - config.duration_tolerance_s:
        raise FeatureError(
            f"sequence covers {t_s[-1]:.2f} s but the window is {config.window_s:.2f} s"
        )
    if float(usable.mean()) < config.min_usable_fraction:
        raise FeatureError(
            f"hand trusted in only {usable.mean():.1%} of frames "
            f"(need {config.min_usable_fraction:.1%})"
        )
    gap = longest_tracking_gap_s(t_s, usable)
    if gap > config.max_gap_s:
        raise FeatureError(f"tracking dropped for {gap:.2f} s (limit {config.max_gap_s:.2f} s)")

    view = to_view_coordinates(seq.landmarks[keep][usable], seq.aspect_ratio)
    trusted_t = t_s[usable]
    grid = np.linspace(0.0, config.window_s, config.steps)
    flat = view.reshape(view.shape[0], -1)
    resampled = np.column_stack(
        [np.interp(grid, trusted_t, flat[:, k]) for k in range(flat.shape[1])]
    )
    return resampled.reshape(config.steps, NUM_LANDMARKS, 2)


def extract_features(seq: SequenceRecord, config: FeatureConfig | None = None) -> np.ndarray:
    """Feature matrix ``(steps, FEATURE_DIM)`` for one sequence; raises ``FeatureError``."""
    config = config or FeatureConfig()
    return features_from_resampled(resample_view(seq, config), config)


def features_from_resampled(pts: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Feature matrix from landmarks already on the time grid, ``(steps, 21, 2)`` view coords.

    Split out of ``extract_features`` so training-time augmentation can perturb the resampled
    landmarks and then compute features with exactly the code live inference uses.
    """
    if pts.shape != (config.steps, NUM_LANDMARKS, 2):
        raise FeatureError(f"expected landmarks of shape {(config.steps, NUM_LANDMARKS, 2)}")
    sizes = np.linalg.norm(pts[:, WRIST, :] - pts[:, MIDDLE_MCP, :], axis=1)
    reference = float(np.median(sizes))
    if not np.isfinite(reference) or reference < config.min_hand_size:
        raise FeatureError(f"hand too small to normalise (size {reference:.3f} frame heights)")

    centroid = pts[:, PALM_POINTS, :].mean(axis=1)
    trajectory = (centroid - centroid[0]) / reference
    velocity = np.gradient(trajectory, config.window_s / (config.steps - 1), axis=0)
    step_size = np.maximum(sizes, config.min_hand_size)[:, None, None]
    shape = (pts - centroid[:, None, :]) / step_size
    features = np.concatenate([trajectory, velocity, shape.reshape(config.steps, -1)], axis=1)
    if not np.all(np.isfinite(features)):
        raise FeatureError("features are not finite")
    return features


@dataclass(frozen=True)
class MotionSummary:
    """Plain geometry of the palm trajectory, in hand sizes, for review and for the baseline."""

    peak_dx: float  # signed horizontal excursion of largest size; negative = user's left
    peak_dy: float  # signed vertical excursion of largest size; positive = down
    path_length: float
    signed_area: float  # positive = clockwise as seen on the mirrored preview (y down)


def motion_summary(features: np.ndarray) -> MotionSummary:
    trajectory = features[:, 0:2]
    x, y = trajectory[:, 0], trajectory[:, 1]
    path = float(np.linalg.norm(np.diff(trajectory, axis=0), axis=1).sum())
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    return MotionSummary(
        peak_dx=float(x[np.argmax(np.abs(x))]),
        peak_dy=float(y[np.argmax(np.abs(y))]),
        path_length=path,
        signed_area=area,
    )
