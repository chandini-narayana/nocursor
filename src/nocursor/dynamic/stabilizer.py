"""Temporal stabilisation: turning overlapping per-window predictions into discrete activations.

A 1.6 s window slides over a recording every 0.1 s, so one movement is seen by about sixteen
windows. Counting flagged windows therefore says little about how often a user would be disturbed.
The ``Stabilizer`` is a small deterministic state machine that fires at most once per movement:

* A window is a *candidate* only if it was evaluated, its prediction is valid, it passed the
  open-palm gate (and, in the experimental variants, the settle and arming gates), it predicts a
  target gesture and its confidence reaches ``tau`` (and the model's own rejection floor).
* **Vote:** it fires when at least ``k`` of the last ``n`` evaluated windows agree on the same class
  and the current window is one of them. A left/right flip-flop therefore never fires.
* **Cooldown:** after a fire nothing fires for ``cooldown_s``. An opposite-direction swipe is
  held back for a further ``opposite_guard_s``, because a fast return stroke looks like a swipe the
  other way.
* **Re-arm:** after a fire it needs ``rearm_quiet`` consecutive windows without a candidate (a
  window with unusable tracking counts as quiet) before it can fire again.
* **Fail closed:** a window with unusable tracking or an invalid prediction clears the vote, so a
  gesture is never assembled across a tracking loss.

Decision time is the *end* of the window (when all of its frames exist), so the machine is causal.
Offline only: nothing here is connected to the mouse.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from nocursor.dynamic.schema import TARGET_CLASSES

_OPPOSITE = {"swipe_left": "swipe_right", "swipe_right": "swipe_left"}


@dataclass(frozen=True)
class StabilizerConfig:
    tau: float = 0.5
    k: int = 1
    n: int = 1
    cooldown_s: float = 1.6
    rearm_quiet: int = 0
    opposite_guard_s: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be within (0, 1]")
        if self.n < 1 or not (1 <= self.k <= self.n):
            raise ValueError("need 1 <= k <= n")
        if self.cooldown_s < 0.0 or self.opposite_guard_s < 0.0:
            raise ValueError("cooldown_s and opposite_guard_s must be >= 0")
        if self.rearm_quiet < 0:
            raise ValueError("rearm_quiet must be >= 0")


@dataclass(frozen=True)
class WindowDecision:
    """What the classifier said about one window, after the gate."""

    start_s: float
    end_s: float
    evaluated: bool  # False: tracking too poor to compute features
    label: str = ""
    confidence: float = 0.0
    valid: bool = True  # False: the model produced unusable output
    gate_ok: bool = True
    settled: bool = True  # experimental: the hand was still at the end of the window
    armed: bool = True  # experimental: a still open-palm hold preceded the window


@dataclass(frozen=True)
class Activation:
    time_s: float  # decision time: the end of the window that completed the vote
    start_s: float
    label: str
    confidence: float


class Stabilizer:
    def __init__(self, config: StabilizerConfig, floor: float = 0.0) -> None:
        self.config = config
        self.floor = floor  # the model's own rejection threshold; never undercut
        self.reset()

    def reset(self) -> None:
        self._history: deque[str | None] = deque(maxlen=self.config.n)
        self._last_end = float("-inf")
        self._cooldown_until = float("-inf")
        self._guard_until = float("-inf")
        self._last_label: str | None = None
        self._armed = True
        self._quiet = 0

    def is_candidate(self, d: WindowDecision) -> bool:
        return bool(
            d.evaluated and d.valid and d.gate_ok and d.settled and d.armed
            and d.label in TARGET_CLASSES and d.confidence >= max(self.config.tau, self.floor)
        )  # fmt: skip

    def step(self, d: WindowDecision) -> Activation | None:
        if d.end_s <= self._last_end:
            raise ValueError("windows must arrive in strictly increasing time order")
        self._last_end = d.end_s
        cfg = self.config

        if not d.evaluated or not d.valid:
            self._history.clear()  # fail closed: no vote across a tracking loss or bad output
        candidate = self.is_candidate(d)
        if d.evaluated and d.valid:
            self._history.append(d.label if candidate else None)
        if candidate:
            self._quiet = 0
        else:
            self._quiet += 1
        if not self._armed and self._quiet >= cfg.rearm_quiet:
            self._armed = True

        if not candidate or not self._armed or d.end_s < self._cooldown_until:
            return None
        if (self._last_label is not None and _OPPOSITE.get(self._last_label) == d.label
                and d.end_s < self._guard_until):  # fmt: skip
            return None
        if sum(1 for entry in self._history if entry == d.label) < cfg.k:
            return None

        self._cooldown_until = d.end_s + cfg.cooldown_s
        self._guard_until = self._cooldown_until + cfg.opposite_guard_s
        self._last_label = d.label
        self._history.clear()
        self._armed = cfg.rearm_quiet == 0
        self._quiet = 0
        return Activation(d.end_s, d.start_s, d.label, d.confidence)


def run_stabilizer(
    decisions: list[WindowDecision], config: StabilizerConfig, floor: float = 0.0
) -> list[Activation]:  # fmt: skip
    machine = Stabilizer(config, floor)
    out = []
    for decision in decisions:
        activation = machine.step(decision)
        if activation is not None:
            out.append(activation)
    return out
