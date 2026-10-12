# Manual Hardware Testing Checklist

> Sections 0-5 are the original core checks (updated where later phases changed behaviour),
> 6-13 cover dragging, scrolling, calibration, pause and pinch diagnostics, and 14-17 cover the
> interface, logging, benchmarks and installation. This file was formerly
> `PHASE1_MANUAL_TESTING.md`.

None of this has been performed by the assistant. It needs a real webcam, hand and desktop.
Tick each item yourself. Run commands from the project folder.

**Before you start:** save your work and close anything important. In live mode the pointer
really moves, the mouse button really presses and the wheel really scrolls. Know your
emergency stop: **Ctrl+Alt+Q** (works from any window).

**Safe targets for live tests:** an empty Notepad window, a browser page you can scroll, and
a window's title bar for dragging. Do **not** practise dragging on files you care about.

## Controls at a glance

| Action | How |
|---|---|
| Move pointer | Index fingertip (cursor freezes while thumb and index close in) |
| Click | Pinch thumb + index, hold still ~0.2 s, release |
| Drag | Pinch, then move your hand more than the drag threshold (30 px) while still pinched; release the pinch to drop |
| Scroll | Peace-sign pose: index + middle extended, ring + little folded. Hold ~0.2 s, then move the hand up/down |
| Pause / resume | `P` in the preview, or `Ctrl+Alt+P` anywhere (`--pause-key`) |
| Calibrate | `C` in the preview, or launch with `--calibrate` |
| Confirm calibration corner | `Space` or `Enter` |
| Cancel calibration | `Backspace` or `X` |
| Reset calibration | `R` in the preview, or `--reset-calibration` |
| Controls help panel | `H` (or `?`) in the preview |
| **Emergency stop** | `Ctrl+Alt+Q` anywhere, `Esc` or `Q` in the preview (always, in every mode), close the preview window, `Ctrl+C`, or push the physical mouse into a screen corner |

## 0. Setup
- [ ] `.\.venv\Scripts\python -m pytest` passes and `.\.venv\Scripts\ruff check .` is clean.
- [ ] `.\.venv\Scripts\python -m nocursor --check` ends with `Ready to run.`
- [ ] `models\hand_landmarker.task` exists (otherwise `.\.venv\Scripts\python -m nocursor --download-model`).

## 1. Webcam + tracking (dry-run: no mouse input is ever sent)
Run `.\.venv\Scripts\python -m nocursor --dry-run`.
- [ ] A preview window opens within a few seconds showing a **mirrored** image of you.
- [ ] Status shows `CONTROL: DRY-RUN`, `TRACKING: lost` with no hand in view.
- [ ] Raise one open hand: a skeleton with 21 points appears; tracking goes `acquiring` then `tracking`, and `STATE` becomes `pointer`.
- [ ] Move the hand: the skeleton follows it, FPS is shown and looks plausible (note the value).
- [ ] Hold your hand half outside the frame, or cover it partly: tracking drops to `uncertain`/`lost` (skeleton turns orange, `STATE: searching`).
- [ ] Pinch thumb and index: the two fingertip circles turn red and `PINCH: pinched` appears. Open: `open`.
- [ ] Your real cursor does **not** move or click during the whole dry-run.

### 1b. False-activation check (dry-run)
In dry-run the terminal logs `dry-run: mouse_down (simulated, not sent)` when a pinch is accepted and
`dry-run: mouse_up (simulated) -> would be a click` (or `drag`) when it ends.
Run `.\.venv\Scripts\python -m nocursor --dry-run --pinch-threshold 0.20 --pinch-confirm-frames 5`
and keep a tally of those messages. Spend about a minute on each:
- [ ] Wave and move your open hand around the whole frame at normal speed: **0** would-be presses.
- [ ] Point with your index finger, tuck the other fingers, turn the hand left/right and tilt it toward and away from the camera: **0**.
- [ ] Slowly open and close the hand into a loose fist and back: **0**.
- [ ] Rest the hand in a relaxed, slightly curled pose for 10 s: **0**.
- [ ] Move your hand farther from the camera (small hand in the frame) and repeat the relaxed pose: **0**.
- [ ] Now pinch thumb and index tip firmly for about half a second, 5 times, releasing fully between: **5** `mouse_down` and **5** `would be a click` lines.
- [ ] Watch the `ratio` in the overlay: relaxed hand should read well above 0.30, a firm pinch below 0.20. Note the values you see.
- [ ] Stop with `Esc` in the preview window: window closes, terminal prints "NoCursor stopped".

## 2. Cursor movement (live)
Run `.\.venv\Scripts\python -m nocursor`. Terminal shows `LIVE`.
- [ ] Move your hand right/left/up/down: the cursor moves right/left/up/down (not inverted).
- [ ] The cursor reaches all four edges of the screen without your hand leaving the camera view (the grey rectangle in the preview is the active region).
- [ ] Cursor is smooth. Try `--smoothing 0` (jittery/fast) vs `--smoothing 0.8` (smooth/laggy) and confirm the difference.
- [ ] Remove the hand: the cursor stops where it was and does not drift.
- [ ] Multi-monitor users: only the primary display is expected to be reachable.

## 3. Clicking
Open a harmless target (e.g. an empty Notepad window, or click inside it to place the text cursor).
A click is now a button **press** when the pinch is confirmed and a **release** when it ends.
- [ ] Hover the cursor on the target, pinch thumb+index briefly and release: exactly one left click.
- [ ] The cursor does not slide off the target as you pinch.
- [ ] Hold the pinch for 3+ seconds without moving: the button stays pressed (a long press) and no repeated clicks happen; releasing ends it.
- [ ] Release and pinch again within ~0.4 s of the previous press: the second pinch is ignored; after about half a second it works.
- [ ] Hand in a loose relaxed pose (thumb not touching index): no clicks.
- [ ] Pinch while your hand is moving into view from off-frame: no press until you have shown an open hand first.
- [ ] Tune if needed: `--pinch-threshold` (larger = easier pinch), `--pinch-confirm-frames` (smaller = faster response, more false clicks), `--click-cooldown`.
- [ ] A very light touch (fingertips just meeting) may not register at 0.20. If real pinches get missed, raise the threshold in small steps (0.22, 0.25) before reducing the frame count.

## 4. Emergency stop (test every path; restart the app between them)
- [ ] **Ctrl+Alt+Q** while another window (e.g. Notepad) is focused: the app stops within a moment, preview closes, terminal prints reason "emergency hotkey pressed".
- [ ] **Esc** or **Q** with the preview window focused.
- [ ] Close the preview window with its X button (use the mouse).
- [ ] **Ctrl+C** in the terminal.
- [ ] Push the physical mouse into a screen corner: app stops ("fail-safe"). Also confirm that moving your *hand* to the extreme corner of the active region does **not** trigger this.
- [ ] After any stop, the cursor no longer moves and nothing clicks.

## 5. Shutdown and camera release
- [ ] After each stop above, the webcam indicator light turns off.
- [ ] Open the Windows Camera app right afterwards: it can use the camera (not "in use").
- [ ] Task Manager shows no leftover `python.exe` from NoCursor.
- [ ] Unplug the webcam (or cover it with another app using it) while running: the app reports a camera error, exits, and never clicks.
- [ ] Start with a wrong camera: `--camera 5` prints a clear "could not open camera" error and exits.
- [ ] Start without the model (temporarily rename it): clear message pointing to `nocursor --download-model`.

---

# Phase 2

## 6. Dry-run of drag and scroll (no input is sent)
Run `.\.venv\Scripts\python -m nocursor --dry-run`.
- [ ] Pinch and hold still, then release: log shows `mouse_down` then `would be a click`.
- [ ] Pinch, move your hand clearly to one side (more than ~30 px of pointer travel) while pinched, then release: `STATE` shows `dragging`; log shows `mouse_down` then `would be a drag`.
- [ ] Pinch and hold still for 5 s with natural hand tremor: it never becomes a drag (`STATE` stays `pressed`).
- [ ] Make the peace sign and hold: `STATE` goes `scroll pending` then `scroll`, with a cyan **SCROLL MODE** banner. Move up/down: `dry-run: scroll +N suppressed` / `-N` lines appear. Your real page does **not** scroll.
- [ ] Your real mouse never moves, presses or scrolls during any of this.

## 7. Click and drag (live)
Open Notepad (empty) and have its title bar visible. Run `.\.venv\Scripts\python -m nocursor`.
- [ ] **Click:** put the pointer in Notepad's text area, pinch briefly and release: the text cursor is placed there. Nothing is dragged.
- [ ] **Drag a window:** put the pointer on the title bar, pinch (wait ~0.2 s), move your hand sideways, release the pinch: the window moves with the pointer and is dropped where you release.
- [ ] **Select text:** type a few words in Notepad, then pinch at the start of a line, move across it, release: the text is selected.
- [ ] **Still pinch = no drag:** pinch on the title bar and hold still for 3 s: the window does not move.
- [ ] **Pointer is frozen while closing the pinch:** the pointer stays on the target as you pinch.
- [ ] **One glitch doesn't drop it:** mid-drag, briefly relax your fingers for a split second (< 2 frames) if you can: the drag continues. A deliberate open hand drops it.
- [ ] **Drop accuracy:** note how far the window lands from where you intended (opening your fingers moves the fingertip slightly).

### 7b. Interruptions during a drag (the mouse button must never stay stuck)
For each, start a drag on Notepad's title bar, then trigger the event. Afterwards move your **physical** mouse: the window must **not** follow it, and a normal click must work.
- [ ] Take your hand out of the camera view (tracking lost). The window is dropped; when your hand returns, still pinching, nothing is grabbed again.
- [ ] Cover the camera with your hand or turn off the room light briefly.
- [ ] **Ctrl+Alt+Q** while dragging, with the preview not focused.
- [ ] **Esc** / **Q** in the preview while dragging.
- [ ] Close the preview window with the mouse while dragging.
- [ ] **Ctrl+C** in the terminal while dragging.
- [ ] Push the physical mouse into a screen corner while dragging.
- [ ] Press **P** (or Ctrl+Alt+P) while dragging: the window is dropped and the preview shows PAUSED.
- [ ] Unplug the webcam while dragging (if practical): error reported, button released.
- [ ] After tracking returns following a loss, an interrupted drag is **not** resumed; you must show an open hand and pinch again.
- [ ] If the terminal ever prints `WARNING: Could not release the left mouse button ...`, click the physical mouse button once, and tell me exactly what you were doing.

## 8. Two-finger scrolling (live)
Open a long web page or document. Place the pointer over it first (scrolling happens at the pointer).
- [ ] Peace sign (index + middle straight up, ring + little folded, thumb anywhere not touching the index) held ~0.2 s: SCROLL MODE banner appears and the skeleton turns cyan.
- [ ] Move your hand **up**: the page scrolls **up** (like a mouse wheel forward). Move **down**: it scrolls down. With `--scroll-invert` this is reversed.
- [ ] Hold the hand still in scroll mode: no scrolling at all.
- [ ] Small natural tremor while holding still: no scrolling.
- [ ] A fast flick scrolls a bit and then stops when your hand stops (no runaway scrolling).
- [ ] The pointer does **not** move while you scroll. (When you leave scroll mode, the pointer jumps back to your fingertip.)
- [ ] Normal pointing (index only), open hand, and fist never enter scroll mode.
- [ ] Touch your thumb to your index finger while in scroll mode: scrolling ends and **no click** happens. You must open your hand before a pinch counts again.
- [ ] Start a pinch while in the peace sign: it is treated as a pinch attempt, not as scrolling.
- [ ] Take your hand out of view mid-scroll: scrolling stops immediately; it needs the stable pose again afterwards.
- [ ] Press P mid-scroll: scrolling stops and does not resume until you resume and re-establish the pose.
- [ ] Tune: `--scroll-speed` (clicks per hand-size of movement, default 8), `--scroll-dead-zone` (default 0.10), `--scroll-stable-frames` (default 6), `--scroll-interval` (default 0.05 s).
- [ ] **Please report:** whether the peace sign was recognised reliably, and whether ring/little-finger folding or an ambiguous half-curl ever blocked or falsely triggered it. The finger-extension limits (extended >= 1.55, folded <= 1.2 of wrist-to-knuckle distance) are derived from hand proportions, not measured on your camera.

## 9. Calibration
Calibration maps the camera area **you** choose to the full screen. No desktop input is sent while it runs.
- [ ] Run `.\.venv\Scripts\python -m nocursor --calibrate`. The preview shows `CONTROL: CALIBRATING` and instructions.
- [ ] Move your hand and pinch while the wizard is open: the real mouse does **not** move or click.
- [ ] Put your index fingertip at the **top-left** corner of the area you want to use, hold still, press **Space**. The progress bar (top right) fills when your fingertip is steady; a captured corner shows as a green dot.
- [ ] Repeat for **top-right**, **bottom-right**, **bottom-left** (as seen in the mirrored preview).
- [ ] After the 4th corner: message "Calibration saved...", a coloured rectangle replaces the grey one, and `calibration.local.json` appears in the project folder.
- [ ] Show an open hand until tracking is stable: the pointer resumes. With your fingertip at each calibrated corner, the pointer reaches the matching screen corner area (left stays left, up stays up).
- [ ] Press **Space** before holding still: "Too much movement" / "Waiting for a steady fingertip" notice, nothing captured.
- [ ] Mark the same corner twice: "too close to a corner you already marked".
- [ ] Mark the corners in the wrong order (e.g. right-to-left): "Calibration rejected ...", default mapping stays in place and no file is written.
- [ ] Press **Backspace** (or X) partway through: "Calibration cancelled; nothing was changed."
- [ ] Press **C** during normal use to recalibrate. Press **R** to reset to the default margin mapping (the file is deleted).
- [ ] Quit and relaunch normally: "Using saved calibration" is printed and the mapping is kept.
- [ ] `--no-calibration` ignores the saved file; `--reset-calibration` deletes it and exits.
- [ ] Relaunch with `--camera 1` (or a different `--width/--height` aspect): a message says the calibration doesn't match and the default mapping is used.
- [ ] Edit `calibration.local.json` to garbage and relaunch: a clear warning, default mapping, no crash.
- [ ] `git status` is yours to run: `calibration.local.json` should be ignored (I added it to `.gitignore`).

## 10. Pause and resume
- [ ] Press **P**: banner "PAUSED"; move your hand, pinch, make the peace sign: nothing happens on the desktop.
- [ ] Press **Ctrl+Alt+P** from another window: toggles pause the same way (holding the keys does not toggle repeatedly).
- [ ] Press P again: tracking must become stable again (`acquiring` -> `tracking`) and you must show an open hand. If you resume **while pinching**, no press happens until you open and re-pinch.
- [ ] Pausing is not an emergency stop: after resuming, everything works without restarting.
- [ ] Pause during calibration is ignored with a notice; Esc/Q still quit.

## 11. Dry-run guarantee
- [ ] With `--dry-run`, perform clicks, a drag, scrolling and a full calibration. The real desktop never receives any of it. (Calibration needs no desktop input anyway; the terminal log is the only record of intent.)

## 12. Phase 1 safety regression (all must still hold)
- [ ] Every stop path in section 4 still works, including while scrolling, while calibrating and while paused.
- [ ] Hand partly out of frame, low confidence or a very small/far hand: no actions.
- [ ] After tracking loss, a pinch only counts after an open hand is seen.
- [ ] Camera is released on every exit path (indicator light off, Camera app can open it).

## 13. Pinch detection: 2D default, diagnostics, optional 3D (dry-run)
**Verify the 2D default first.** Run exactly this (no mouse input is ever sent in `--dry-run`):

```powershell
.\.venv\Scripts\python -m nocursor --dry-run --diagnostics
```

- The terminal banner must say `Pinch measurement: 2D pinch ratio (default).`
- The preview's yellow panel starts with `mode: 2D pinch ratio (default)`, and the terminal gets
  at most one `diag mode=[2D pinch ratio (default)] ...` line per second.
- This is the same calculation as your earlier `--no-pinch-depth` run, so `--no-pinch-depth` still
  works and gives an identical result. The experimental opt-in is `--pinch-depth`; giving both
  flags is an error.

| `why:` says | Meaning and what to do |
|---|---|
| `NOT ARMED: open thumb and index apart (ratio X must reach 0.30)` | The detector has not seen your thumb and index clearly apart since it was last reset (start-up, tracking loss, calibration, pause, or scroll mode). Spread them wide for a moment; `armed:` turns YES. |
| `ratio X is above the pinch threshold 0.20` | The tips are not close enough in the image. Watch `lowest in 3s` while you pinch firmly: if it bottoms out at, say, 0.26, try `--pinch-threshold 0.30`. |
| `... (the 2D ratio Y would pass: depth is inflating it; run without --pinch-depth)` | Only appears in the experimental 3D mode: its depth term is what keeps the ratio high. Drop `--pinch-depth`. |
| `confirming pinch n/5` | Working: hold the pinch for all 5 frames. |
| `two-finger pose seen: scroll pending` / `scroll mode` | Your hand looks like the scroll pose (index + middle extended, ring + little folded). Fold or relax your middle finger to pinch. |
| `hand rejected: ...` | Tracking safety gate (low confidence, hand partly out of frame, hand too small). Move into view. |
| `pinch refused by cooldown` | A second pinch came less than 0.4 s after the previous one. |

Checks (default 2D mode):
- [ ] Banner and panel both state `2D pinch ratio (default)`.
- [ ] Relaxed open hand: `armed: YES`, ratio well above 0.30.
- [ ] Touch thumb and index tips and hold ~0.5 s: `confirm` counts 1/5 ... 5/5, then `why:` says ACCEPTED, and the terminal shows `dry-run: mouse_down (simulated, not sent)`. Release: `would be a click`. (This is the case that failed with the old 3D default.)
- [ ] Do that 5 times with a full release between: 5 `mouse_down` and 5 `would be a click`.
- [ ] False-activation sweep, about a minute each, expecting **0** `mouse_down` lines: wave the open hand around the frame; point and turn/tilt the hand; slowly open and close a loose fist; rest the hand relaxed; move farther from the camera.
- [ ] A thumb passing in front of the index finger without touching: note whether it ever reaches 5/5. (This is the accepted trade-off of 2D; tell me if it happens in normal use.)
- [ ] Hold the pinch just above the threshold (ratio about 0.22-0.28) for several seconds: no press.
- [ ] Tracking loss (hand out of view), then back while still pinching: no press until you spread thumb and index, then pinch again.
- [ ] After a calibration, spread thumb and index once, then pinch: it registers.
- [ ] Hold a two-finger pose for 2 s, then pinch (fold the middle finger first if SCROLL MODE appears): registers.
- [ ] `Ctrl+Alt+Q` still stops everything. The real mouse never moves or clicks.

Optional - experimental 3D mode, for comparison only:
```powershell
.\.venv\Scripts\python -m nocursor --dry-run --diagnostics --pinch-depth
```
- [ ] Banner and panel say `3D pinch ratio (experimental)`.
- [ ] Note the `lowest in 3s` value for a firm pinch in 3D versus the 2D default.

Please send me: the `diag` lines from any pinch attempt that fails, your camera and resolution, and the `lowest in 3s` values.

## 14. Interface (HUD) feedback
Run `.\.venv\Scripts\python -m nocursor --dry-run` and watch the top-left status panel. It should
always say what is happening in plain words, and it must never say something happened that did not.
- [ ] No hand: `NO HAND - show one hand to the camera`. Partly out of frame or far away: `HAND NOT USABLE - <reason>`.
- [ ] Hand appears: `LOCKING ON (n/3)`, then `POINTING` in green.
- [ ] Close thumb and index slowly: `PINCH PENDING 1/5 - not clicked yet` ... `4/5`, with orange fingertip rings. **No click is claimed during this.**
- [ ] After 5 frames: `PRESSED - button held (simulated)` with red rings, and the toast `Button pressed (simulated)`.
- [ ] Release: toast `Released: click (simulated)`. Drag: `DRAGGING - button held (simulated)`, then `Released: drag finished (simulated)`.
- [ ] Second pinch right after the first: `PINCH IGNORED - release, then pinch again` (cooldown).
- [ ] After tracking loss or `P`: `NOT ARMED - spread thumb and index apart`.
- [ ] Peace sign: `SCROLL POSE - hold still to start scrolling` (a candidate), then `SCROLLING - move your hand up or down` (cyan skeleton).
- [ ] `P`: `PAUSED - no input is sent. Press P to resume`. `C`: `CALIBRATING - corner n/4`. Stop key: `STOPPED - <reason>`.
- [ ] The `Control:` line says `DRY-RUN (no input is sent)` here and `LIVE (mouse input is sent)` without `--dry-run`.
- [ ] The pinch line names the active mode, e.g. `2D pinch ratio (default): 0.24  (pinch at <= 0.20)`; with `--pinch-depth` it says `3D pinch ratio (experimental)`.
- [ ] `H` shows the controls panel; `H` again hides it; `Esc`/`Q` still stop while it is open.
- [ ] `--show-perf` adds a `PERF ... fps | inference p50 ... | software e2e p50 ... / p95 ...` line after a few seconds, labelled as software timings.
- [ ] Text stays readable over a bright and a dark background (the panel is darkened behind it).
- [ ] In a live run (no `--dry-run`), the press and release toasts appear only when the click really lands. Block it deliberately if you can (run NoCursor inside an elevated window's focus) and confirm the HUD says `PRESS NOT SENT`, not `PRESSED`.

## 15. Logging and error messages
- [ ] A normal start prints a short banner and no raw MediaPipe lines (`INFO: Created TensorFlow Lite XNNPACK delegate`, `W0000 ...`).
- [ ] `-v` shows them (as DEBUG lines starting `MediaPipe model load:`), plus logger names.
- [ ] `--log-file` creates `logs
ocursor.log`; without the flag no `logs` folder appears. The file contains start/end lines, pause/resume, stop reasons, and no video or landmark data. Your user name does not appear (your home folder shows as `~`).
- [ ] `--diagnostics` shows `diag ...` lines about once per second in the console (and in the log file with `--log-file`); without it, none.
- [ ] Rename the model file: the error tells you to run `nocursor --download-model`.
- [ ] Unplug the camera / use `--camera 5`: a clear camera error, exit code 1, no traceback.
- [ ] Replace the model with a junk file (keep a backup): the error says it may be corrupt and to run `nocursor --redownload-model`.
- [ ] An invalid option (for example `--scroll-speed 500`) names the flag and the valid range, exit code 64.
- [ ] `nocursor --check` reports problems in plain words (try it with the model renamed).

## 16. Benchmarks (all dry-run: they cannot send input)
Results from the three modes are never comparable with each other; each report says which it is.
- [ ] **Synthetic** `nocursor --benchmark synthetic` finishes, prints a table, writes `benchmarks\synthetic-<time>.json`. The title says `not webcam performance`. Run it twice: the *shape* of the results repeats (timings vary a little).
- [ ] **Video replay** with a short clip of your own hand: `nocursor --benchmark video --benchmark-video path	o\clip.mp4`. The report shows only the file name, `hand_present_frames`, and separate `inference_hand` / `inference_no_hand` rows.
- [ ] **Live** `nocursor --benchmark live --benchmark-seconds 20`: a preview opens, you keep your hand in view and use gestures for 20 s.
  - [ ] **Your real mouse never moves, presses or scrolls** during the run.
  - [ ] `Esc`/`Q` ends it early; the report then says why in `ended_by`.
  - [ ] The report lists `input_events_sent: 0`, `input_sent: false`, and some `input_events_simulated` if you pinched.
  - [ ] Note fps, inference p50/p95, end-to-end p50/p95, tracking-loss events per minute, CPU and memory.
- [ ] Every report states the measurement method and that it is not camera-to-display latency.
- [ ] Do not publish any of these numbers without the mode, your camera, resolution and machine.

## 17. Installation in a clean environment
See the README, "Install into a clean environment".
- [ ] `python -m venv`, `pip install -r requirements.txt`, `pip install -e . --no-deps` succeed.
- [ ] `pip list` shows `opencv-contrib-python` and **not** `opencv-python`.
- [ ] `nocursor --version`, `nocursor --check`, `nocursor --download-model` work; `nocursor --dry-run` opens the preview window (OpenCV window support is available).

## Record your findings
Note: camera model, resolution, FPS seen, any misclicks, accidental drags or scrolls, stuck-button incidents, calibration issues, and which `--smoothing`, `--pinch-threshold`, `--drag-start`, `--scroll-speed` values felt right. Report them back to the project owner.
