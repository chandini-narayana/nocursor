# Phase 1 Manual Testing Checklist

None of this has been performed by the assistant. It needs a real webcam, hand and desktop.
Tick each item yourself. Run commands from the project folder.

**Before you start:** save your work and close anything important. In live mode the cursor
really moves and really clicks. Know your emergency stop: **Ctrl+Alt+Q**.

## 0. Setup
- [ ] `.\.venv\Scripts\python -m pytest` passes and `.\.venv\Scripts\ruff check .` is clean.
- [ ] `models\hand_landmarker.task` exists (otherwise `python scripts\download_model.py`).

## 1. Webcam + tracking (dry-run: no mouse input is ever sent)
Run `.\.venv\Scripts\python -m nocursor --dry-run`.
- [ ] A preview window opens within a few seconds showing a **mirrored** image of you.
- [ ] Status shows `CONTROL: DRY-RUN`, `TRACKING: lost` with no hand in view.
- [ ] Raise one open hand: a skeleton with 21 points appears; tracking goes `acquiring` then `tracking`.
- [ ] Move the hand: the skeleton follows it, FPS is shown and looks plausible (note the value).
- [ ] Hold your hand half outside the frame, or cover it partly: tracking drops to `uncertain`/`lost` (skeleton turns orange).
- [ ] Pinch thumb and index: the two fingertip circles turn red and `PINCH: pinched` appears. Open: `open`.
- [ ] Your real cursor does **not** move or click during the whole dry-run.

### 1b. False-activation check (dry-run; the terminal logs `dry-run: click suppressed` for every click that *would* have happened)
Run `.\.venv\Scripts\python -m nocursor --dry-run --pinch-threshold 0.20 --pinch-confirm-frames 5`
and keep a tally of the terminal messages. Spend about a minute on each:
- [ ] Wave and move your open hand around the whole frame at normal speed: **0** would-be clicks.
- [ ] Point with your index finger, tuck the other fingers, turn the hand left/right and tilt it toward and away from the camera: **0**.
- [ ] Slowly open and close the hand into a loose fist and back: **0**.
- [ ] Rest the hand in a relaxed, slightly curled pose for 10 s: **0**.
- [ ] Move your hand farther from the camera (small hand in the frame) and repeat the relaxed pose: **0**.
- [ ] Now pinch thumb and index tip firmly for about half a second, 5 times, releasing fully between: **5** would-be clicks, each one logged once.
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
Open a harmless target (e.g. an empty Notepad window or a desktop folder you select, not delete).
- [ ] Hover the cursor on the target, pinch thumb+index briefly: exactly one left click.
- [ ] The cursor does not slide off the target as you pinch.
- [ ] Hold the pinch for 3+ seconds: still only one click.
- [ ] Release and pinch again within ~0.4 s: second click is ignored; after about half a second it works.
- [ ] Hand in a loose relaxed pose (thumb not touching index): no clicks.
- [ ] Pinch while your hand is moving into view from off-frame: no click until you have shown an open hand first.
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
- [ ] Start without the model (temporarily rename it): clear message pointing to `download_model.py`.

## Record your findings
Note: camera model, resolution, FPS seen, any misclicks or lost-tracking issues, and which
`--smoothing` / `--pinch-threshold` values felt right. Report them back before Phase 2.
