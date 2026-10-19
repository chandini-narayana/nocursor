"""Experimental arming rule: a gesture only counts after a deliberate still open-palm hold.

Ordinary open-palm movement is the main source of false activations, and it is indistinguishable
from a gesture by shape alone. An arming rule asks the person to *prepare*: hold an open palm still
for ``hold_s`` seconds first. The system is then armed for at most ``max_age_s`` seconds, which is
long enough for one gesture; it disarms when the open palm is lost (not an open palm, or not
tracked) for ``drop_s`` seconds.

Stillness is the palm centre's speed in hand sizes per second over a 0.1 s baseline, so it does not
depend on camera distance. The pose is the existing ``is_open_palm`` (through ``frame_palm_flags``).

A window passes the rule only if the system was armed at the moment the motion *in that window
begins* (the first step whose palm speed leaves the "still" band). Checking the state at the end of
the window instead would accept a window that merely ends in a rest after the gesture, which is a
"stop afterwards" rule, not "prepare first". A window with no motion has no onset and fails.

This changes how the gesture must be performed, so it can only be judged on recordings made with
that instruction. The existing takes were recorded without it: most start moving almost at once
(see docs/DYNAMIC_GESTURES.md), so an evaluation on them cannot show whether people could use it.

An experiment only: offline, and not connected to the mouse or to any desktop-control logic. A fire
does not consume the arming; the stabilizer's cooldown and re-arm handle repeats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nocursor.dynamic.features import PALM_POINTS, to_view_coordinates
from nocursor.dynamic.schema import SequenceRecord
from nocursor.models import MIDDLE_MCP, WRIST

SPEED_BASELINE_S = 0.1


@dataclass(frozen=True)
class ArmingConfig:
    hold_s: float
    still_speed: float = 0.5  # hand sizes per second
    max_age_s: float = 3.0
    drop_s: float = 0.3

    def __post_init__(self) -> None:
        if self.hold_s <= 0 or self.still_speed <= 0 or self.max_age_s <= 0 or self.drop_s <= 0:
            raise ValueError("hold_s, still_speed, max_age_s and drop_s must be > 0")

    def label(self) -> str:
        return f"hold {self.hold_s:g}s, still < {self.still_speed:g}/s, valid {self.max_age_s:g}s"


def motion_onset_step(features: np.ndarray, still_speed: float) -> int | None:
    """Index of the first feature step whose palm speed reaches ``still_speed``; None if none."""
    speed = np.linalg.norm(features[:, 2:4], axis=1)
    fast = np.flatnonzero(speed >= still_speed)
    return int(fast[0]) if fast.size else None


def palm_speed_series(seq: SequenceRecord) -> np.ndarray:
    """Palm-centre speed per frame in hand sizes per second; NaN where it cannot be measured."""
    speed = np.full(seq.n_frames, np.nan)
    trusted = np.flatnonzero(seq.usable)
    if trusted.size < 2:
        return speed
    view = to_view_coordinates(seq.landmarks[trusted], seq.aspect_ratio)
    centre = view[:, list(PALM_POINTS), :].mean(axis=1)
    size = np.linalg.norm(view[:, WRIST] - view[:, MIDDLE_MCP], axis=1)
    t = (seq.t_ms[trusted] - seq.t_ms[0]) / 1000.0
    back = np.searchsorted(t, t - SPEED_BASELINE_S, side="right") - 1
    ok = (back >= 0) & (back < np.arange(trusted.size)) & (size > 0)
    k = np.flatnonzero(ok)
    dt = t[k] - t[back[k]]
    good = dt > 0
    k, dt = k[good], dt[good]
    speed[trusted[k]] = np.linalg.norm(centre[k] - centre[back[k]], axis=1) / size[k] / dt
    return speed


def armed_series(seq: SequenceRecord, palm_flags: np.ndarray, config: ArmingConfig) -> np.ndarray:
    """Per frame: is the system armed at this moment?"""
    speed = palm_speed_series(seq)
    t = (seq.t_ms - seq.t_ms[0]) / 1000.0
    armed = np.zeros(seq.n_frames, dtype=bool)
    run_start: float | None = None
    lost_since: float | None = None
    is_armed = False
    last_arm = -np.inf
    for i in range(seq.n_frames):
        now = float(t[i])
        open_palm = bool(seq.usable[i] and palm_flags[i])
        still = open_palm and speed[i] < config.still_speed  # NaN compares false
        if still:
            if run_start is None:
                run_start = now
            if now - run_start >= config.hold_s - 1e-9:
                is_armed, last_arm = True, now
        else:
            run_start = None
        if open_palm:
            lost_since = None
        else:
            lost_since = now if lost_since is None else lost_since
            if now - lost_since >= config.drop_s - 1e-9:
                is_armed = False
        armed[i] = is_armed and (now - last_arm) <= config.max_age_s + 1e-9
    return armed
