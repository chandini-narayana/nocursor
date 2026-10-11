# NoCursor

Control your desktop with your hand and a webcam.

- **Phase 1:** index-finger pointer, pinch-to-click, live preview, emergency stop.
- **Phase 2:** pinch-and-drag, two-finger scrolling, 4-corner calibration, explicit gesture
  states, pause/resume, and a safer button-release path.

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
.\.venv\Scripts\python -m nocursor --calibrate # open the calibration wizard first
```

## Gestures

| Gesture | Result |
|---|---|
| Index fingertip | Moves the pointer (frozen while thumb and index close in) |
| Pinch (thumb + index), hold still, release | Click: a mouse press when the pinch is confirmed, a release when it ends |
| Pinch, then move more than 30 px, release at the drop point | Drag |
| Index + middle extended, ring + little folded; then move up/down | Scroll (hand up = wheel up) |

A pinch only counts after the hand has been seen open, meaning thumb and index tips clearly
apart (ratio at least 0.30). That is required after start-up, tracking loss, calibration,
pause/resume, and when scroll mode ends. Until then the preview says **NOT ARMED - spread
thumb and index apart**. Merely holding a two-finger-looking pose does not disarm the pinch;
only genuinely active scroll mode (a stable pose for 6 frames) does. The pointer does not
move while scrolling.

If pinches are not registering, run with `--diagnostics` (below) and read the `why:` line.

### Pinch measurement: 2D by default
The pinch ratio is the distance between the thumb tip and index tip **in the camera image**,
divided by the wrist-to-middle-knuckle distance (also in the image). This **2D pinch ratio is the
default**. The terminal banner and `--diagnostics` always state which calculation is active:
`2D pinch ratio (default)` or `3D pinch ratio (experimental)`.

Why not 3D? A 3D ratio also uses MediaPipe's depth (z) estimate. On real webcams the two
fingertips' depth estimates can disagree enough that touching fingers never reach the 0.20
threshold; this was confirmed on hardware (a pinch registered only with the 2D ratio). The
experimental `--pinch-depth` mode can reject fingertips that merely overlap in the image, at the
cost of that risk. The trade-off of the 2D default: a thumb passing in front of the index finger
can look like a touch, which the 5-frame confirmation, 0.20/0.30 hysteresis, cooldown and re-arming
rules are there to absorb. Everything else (threshold, confirmation, release, cooldown, re-arming,
drag and scroll behaviour) is identical in both modes.

## Keys (preview window)

| Key | Action |
|---|---|
| `Esc` / `Q` | **Stop** (always, in every mode) |
| `P` | Pause / resume (global hotkey: `Ctrl+Alt+P`, `--pause-key`) |
| `C` | Start calibration |
| `Space` / `Enter` | Capture the current calibration corner |
| `Backspace` / `X` | Cancel calibration |
| `R` | Reset calibration to the default mapping |

**Emergency stop** (any one of these): `Ctrl+Alt+Q` from any window, `Esc`/`Q` in the preview
window, closing the preview window, `Ctrl+C` in the terminal, or pushing the *physical* mouse into
a screen corner (PyAutoGUI fail-safe). Every stop releases a held mouse button first.

## Options

See `--help`. The ones you are most likely to tune:

| Option | Default | Meaning |
|---|---|---|
| `--smoothing` | 0.5 | Cursor smoothing in [0, 1); 0 = none |
| `--margin` | 0.15 | Camera edge fraction ignored by the default mapping |
| `--pinch-threshold` | 0.20 | Thumb-index distance / hand size (2D, image plane) that counts as a pinch. Lower = stricter |
| `--pinch-confirm-frames` | 5 | Consecutive frames before a pinch is accepted (~0.17 s at 30 fps) |
| `--pinch-release-frames` | 2 | Consecutive open frames before the release is confirmed (so one glitch cannot drop a drag) |
| `--pinch-release-factor` | 1.5 | Pinch ends at threshold x factor (0.30 by default) |
| `--click-cooldown` | 0.4 | Minimum seconds between accepted pinches |
| `--drag-start` | 30 | Screen pixels the fingertip must move while pinched before a click becomes a drag |
| `--scroll-stable-frames` | 6 | Frames the two-finger pose must hold before scrolling starts |
| `--scroll-dead-zone` | 0.10 | Vertical movement (in hand sizes) ignored as jitter |
| `--scroll-speed` | 8 | Wheel notches per hand size of movement |
| `--scroll-interval` | 0.05 | Minimum seconds between scroll events (rate limit) |
| `--scroll-invert` | off | Hand up scrolls down |
| `--diagnostics` | off | Show live pinch internals in the preview and log one line per second (read-only: changes no behaviour) |
| `--pinch-depth` | off | **Experimental.** Also use MediaPipe's depth estimate (3D pinch ratio). See "Pinch measurement" |
| `--no-pinch-depth` | (already the default) | Kept so existing commands work: the 2D ratio. Combining it with `--pinch-depth` is an error |
| `--pause-key` / `--stop-key` | `ctrl+alt+p` / `ctrl+alt+q` | Global hotkeys (Windows) |
| `--calibrate` | off | Open the wizard at launch |
| `--no-calibration` | | Ignore the saved calibration |
| `--calibration-file` | `calibration.local.json` | Where calibration is stored |
| `--reset-calibration` | | Delete the saved calibration and exit |

## Calibration

The wizard asks for the top-left, top-right, bottom-right and bottom-left corners of the area
your hand will use (as shown in the mirrored preview). Each corner is captured only when you press
`Space` while your fingertip is steady. The result is an axis-aligned rectangle that maps to the
whole screen; fingertips outside it clamp to the screen edge. It is saved to
`calibration.local.json` (a versioned JSON file containing only the rectangle, camera index and
frame aspect ratio; it is git-ignored and never contains images). It is ignored with a message if
the camera index or frame shape differs, or if the file is corrupt or invalid. No desktop input
is sent while the wizard is open.

## Design

`GestureProcessor` (pure logic) turns each frame into typed actions through one state machine,
`InteractionMachine`. `SafeController` is the only code that sends input.

| State | Meaning |
|---|---|
| `searching` | Tracking not reliable (lost, uncertain or still acquiring); no input, any held button released |
| `rearming` | Tracking fine, but an open hand must be seen before a pinch counts |
| `pointer` | Armed and pointing; the pointer follows the fingertip |
| `pinch pending` | Fingers closing / confirming (or a pinch refused by the cooldown); pointer frozen |
| `pressed` | Button down, pointer frozen, no drag yet |
| `dragging` | Button down, pointer follows the fingertip |
| `scroll pending` / `scroll` | Two-finger pose seen / scrolling |
| `paused` / `stopped` | No input; `stopped` is terminal |

Priority, highest first: stop/pause, unreliable tracking, a held pinch (nothing can preempt it),
a stable scroll pose, a confirmed pinch-down, pointing. Anything ambiguous (half-curled fingers,
low confidence, partly visible hand) falls back to the safe outcome: no scroll, no press.

Button safety: at most one press is outstanding (enforced in the state machine **and** the
controller). A release is attempted on pinch release, tracking loss, pause, calibration start,
emergency stop (from any thread, immediately), controller error, and shutdown. Releasing is never
blocked by the stop flag and bypasses PyAutoGUI's corner fail-safe. If a release fails, the app
stops, prints a warning that the button may still be held, and shows it in the preview.

## Develop

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\ruff check .
```

Manual hardware checks: [docs/PHASE1_MANUAL_TESTING.md](docs/PHASE1_MANUAL_TESTING.md)
(Phase 1 and Phase 2 sections).

## Layout

`src/nocursor/`: `camera`, `hand_tracker` (MediaPipe), `tracking` (reliability), `mapping`,
`smoothing`, `gestures` (pinch detection), `scroll` (pose + wheel conversion), `interaction`
(state machine), `calibration`, `processor` (pure gesture logic), `controller` (safety-gated
desktop input), `emergency` (stop, pause, hotkeys), `preview`, `app` (loop), `cli`.

## Known limitations

See the Phase 2 report; in short: Windows-only global hotkeys, primary monitor only, one hand,
finger-pose thresholds and drop accuracy not yet validated on real hardware.
