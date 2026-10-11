# NoCursor

Control your desktop cursor with your hand and a webcam. Phase 1 (core): index-finger
cursor movement, pinch-to-click, live preview, emergency stop.

All webcam processing is local. Frames are never saved or transmitted.

## Setup (Windows, Python 3.10+)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e . --no-deps
.\.venv\Scripts\python scripts\download_model.py   # one-time download of the Google hand model
```

## Run

```powershell
.\.venv\Scripts\python -m nocursor --dry-run   # tracking + preview, NO mouse input (start here)
.\.venv\Scripts\python -m nocursor             # live control
```

**Emergency stop** (any one of these): `Ctrl+Alt+Q` from any window, `Esc`/`Q` in the preview
window, closing the preview window, `Ctrl+C` in the terminal, or pushing the *physical* mouse into
a screen corner (PyAutoGUI fail-safe).

Useful options: `--smoothing 0.5` (0 = none, <1), `--margin 0.15`, `--camera 0`,
`--stop-key ctrl+alt+q`, and the pinch controls below. See `--help`.

| Pinch option | Default | Meaning |
|---|---|---|
| `--pinch-threshold` | 0.20 | Activation level: thumb-index distance / hand size (3D). Lower = stricter. |
| `--pinch-confirm-frames` | 5 | Consecutive frames at/below the threshold before a pinch counts (~0.17 s at 30 fps). |
| `--pinch-release-factor` | 1.5 | Pinch ends at threshold x factor (0.30 by default). Must be > 1. |
| `--click-cooldown` | 0.4 | Minimum seconds between clicks. |

## Develop

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\ruff check .
```

Manual verification steps: [docs/PHASE1_MANUAL_TESTING.md](docs/PHASE1_MANUAL_TESTING.md).

## Layout

`src/nocursor/`: `camera`, `hand_tracker` (MediaPipe), `tracking` (reliability), `mapping`,
`smoothing`, `gestures` (pinch), `processor` (pure gesture logic), `controller` (safety-gated
desktop input), `emergency`, `preview`, `app` (loop), `cli`.
