# Contributing to NoCursor

Thanks for your interest. NoCursor controls a real mouse, so the most valuable contributions keep
it **safe, predictable and well tested**. Please read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
first; it explains the state machine and the button-safety invariants.

> **License:** no license has been chosen for this project yet; that is the owner's decision.
> Until one is added, please treat the code as all rights reserved and ask before reusing it.

## Development setup (Windows, Python 3.10+)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e . --no-deps
.\.venv\Scripts\python -m nocursor --download-model     # one-time, ~8 MB
.\.venv\Scripts\python -m nocursor --check
```

## Before you submit a change

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\ruff check .
```

Both must pass. The test suite is hermetic: during tests, outbound network access, opening a
camera and sending real mouse input all raise (`tests/conftest.py`). Never work around that guard;
write a fake instead.

## Ground rules

- **Desktop input goes through `SafeController` only.** Do not call PyAutoGUI from anywhere else.
- **Gesture logic stays pure.** `processor`, `interaction`, `gestures`, `scroll`, `mapping`,
  `calibration` and `hud` do no I/O and import no window or camera code.
- **Keep the invariants.** A change that touches the press / release path needs tests for: stop,
  tracking loss, pause, controller failure and shutdown while a button is held.
- **Do not weaken thresholds or safety gates to make something look smoother.** Changing a default
  (pinch threshold, confirmation frames, hysteresis) needs measurements from real hardware.
- **Do not log or store frames or landmark data.** The log formatter already summarises arrays;
  keep it that way.
- **Be honest in docs.** Benchmarks must come from an actual run and say what they measure.
  Software timings are not camera-to-display latency.
- **Dry-run and benchmarks must never be able to send input.** Keep `ScreenSizeOnlyBackend` in
  that path.

## Adding or changing a gesture

1. Decide where it sits in the priority list in the architecture document.
2. Add the state(s) and transition(s) to `InteractionMachine`; emit actions, never call the
   controller.
3. Add its wording to `hud.py` and a test that a *candidate* is never described as an action.
4. Add tests with synthetic landmarks (`tests/helpers.py` has pose builders) covering the normal
   path, jitter just under every threshold, tracking loss, pause and emergency stop.
5. Update the README gesture table and the manual checklist.

## Reporting a problem

Please include the output of `nocursor --check` and, for gesture problems, a few lines from
`nocursor --dry-run --diagnostics --log-file`. The log never contains video or landmark data and
replaces your home folder with `~`, but look it over before sharing. Say which camera and
resolution you use.

## Roadmap ideas (not promised)

- Left/right-hand and multi-monitor support; non-Windows global hotkeys.
- A recorded-landmark replay format so hand-present benchmarks need no video file.
- Adaptive per-user thresholds learned during calibration.
- Perspective-correct calibration (currently an axis-aligned rectangle).
- Further gestures (right click) once the existing ones are validated by more users.
