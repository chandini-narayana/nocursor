"""Training-only augmentation of resampled landmark sequences (view coordinates).

Applied to the *training* split only, with a seeded generator, before features are computed by the
same function inference uses. Each perturbation models a variation that a real person produces
between sessions, and none can change a gesture's direction:

* **speed**: the same gesture performed up to 15% faster (end held) or 10% slower. A slower
  gesture inside a fixed window is a partly finished one, so the slow end is kept small;
* **amplitude**: the same gesture drawn 20% smaller or larger, with the hand's shape unchanged
  (scaling the whole hand would do nothing, because features are divided by hand size);
* **rotation**: the hand and path tilted by a few degrees, as when the camera or wrist is angled;
* **jitter**: a little landmark noise, comparable to the tracker's own.

There is deliberately **no horizontal flip**: it would have to swap left and right, and a mirrored
clockwise circle is counter-clockwise, which is not a class.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from nocursor.dynamic.features import PALM_POINTS


@dataclass(frozen=True)
class AugmentConfig:
    speed_range: tuple[float, float] = (0.9, 1.15)
    amplitude_range: tuple[float, float] = (0.8, 1.25)
    max_rotation_deg: float = 8.0
    jitter_sigma: float = 0.003  # frame heights; MediaPipe landmark jitter is of this order

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def augment_resampled(
    pts: np.ndarray, rng: np.random.Generator, config: AugmentConfig
) -> np.ndarray:
    """A perturbed copy of ``(steps, 21, 2)`` landmarks. The input is never modified."""
    steps = pts.shape[0]
    out = pts.copy()

    speed = rng.uniform(*config.speed_range)
    positions = np.clip(np.arange(steps) * speed, 0, steps - 1)
    flat = out.reshape(steps, -1)
    out = np.column_stack(
        [np.interp(positions, np.arange(steps), flat[:, k]) for k in range(flat.shape[1])]
    ).reshape(pts.shape)

    centroid = out[:, PALM_POINTS, :].mean(axis=1)
    amplitude = rng.uniform(*config.amplitude_range)
    moved = centroid[0] + amplitude * (centroid - centroid[0])
    out = out - centroid[:, None, :] + moved[:, None, :]

    angle = math.radians(rng.uniform(-config.max_rotation_deg, config.max_rotation_deg))
    cos, sin = math.cos(angle), math.sin(angle)
    origin = out[0, PALM_POINTS, :].mean(axis=0)
    rel = out - origin
    out = np.stack([rel[..., 0] * cos - rel[..., 1] * sin,
                    rel[..., 0] * sin + rel[..., 1] * cos], axis=-1) + origin  # fmt: skip

    return out + rng.normal(0.0, config.jitter_sigma, out.shape)
