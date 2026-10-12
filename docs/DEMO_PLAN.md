# Demonstration recording plan

> **Status:** this is a plan only. No video, screenshot or GIF of NoCursor has been produced, and
> nothing in this repository claims one exists. Record the demo yourself following the steps below.

The goal is a 3 to 4 minute screen recording that shows NoCursor working *and* shows its safety
features, using content that is harmless to touch.

## Before you record

**Privacy.** The recording will show your screen and, if you include the preview, your face and
hands. Record only content you are happy to publish. Close chat apps, mail and anything with
notifications; turn on Do Not Disturb. Make sure no window shows your user name or private paths.

**Safe desktop content.** Use only things you can freely move, select and scroll:
- an empty Notepad window (to click in and select text you type),
- a folder named `demo` on the desktop containing three empty text files you created for this,
- a long, public web page or a long document to scroll,
- never real files, never anything with a confirm/delete dialog in reach.

**Setup checklist**
1. Fresh virtual environment, `nocursor --check` shows `Ready to run.`
2. Good, even lighting, plain background, camera at eye level.
3. Resolution: record at 1080p, preview window at its default size beside the demo content.
4. Run once in `--dry-run --diagnostics` to rehearse; then `nocursor --calibrate` once so the
   pointer reaches the whole screen comfortably.
5. Know your emergency stop (`Ctrl+Alt+Q`) before you start.
6. Screen recorder of your choice (for example the Windows Game Bar, `Win+G`). Capture the screen
   *including* the NoCursor preview window; the on-screen status line is what proves what is
   happening.

## Shot list

| # | Scene | What to do | What must be visible | ~Time |
|---|---|---|---|---|
| 1 | Intro | Terminal: `nocursor --check`, then `nocursor --dry-run` | The check result; the banner saying DRY-RUN and `2D pinch ratio (default)` | 0:20 |
| 2 | Tracking and the HUD | Show a hand; move it; hide it | `NO HAND`, `LOCKING ON`, `POINTING`; skeleton overlay | 0:20 |
| 3 | Candidate vs. action (dry-run) | Pinch slowly | `PINCH PENDING n/5 - not clicked yet`, then `PRESSED - button held (simulated)`, toast `Button pressed (simulated)` | 0:25 |
| 4 | Cursor movement (live) | Restart without `--dry-run`; move the pointer around the screen; hover the `demo` folder | Pointer follows the fingertip | 0:20 |
| 5 | Click | Click in Notepad; click a file icon to select it | `PRESSED`, toast `Released: click` | 0:20 |
| 6 | Drag | Drag a file from `demo` to the desktop and back; drag the Notepad window by its title bar | `DRAGGING - button held`, toast `Released: drag finished` | 0:30 |
| 7 | Scroll | Peace sign over the long page; move up and down | `SCROLL POSE`, then `SCROLLING`; the page scrolls | 0:30 |
| 8 | Calibration | Press `C`; mark four corners with `Space`; show the saved message | `CALIBRATING - corner n/4`, the "hold still" bar, then `Calibration saved` | 0:35 |
| 9 | Pause / resume | Press `P`, wave and pinch (nothing happens), press `P`, show `NOT ARMED`, spread your hand, carry on | `PAUSED - no input is sent`, then `NOT ARMED - spread thumb and index apart` | 0:25 |
| 10 | Tracking loss during a drag | Start a drag, take your hand out of view | The drag is dropped, `NO HAND`; the file stays where it was dropped | 0:20 |
| 11 | Emergency stop | Start a drag, press `Ctrl+Alt+Q` from the keyboard | `STOPPED - emergency hotkey pressed`, the button released, the window closes | 0:20 |
| 12 | Close | Terminal shows `NoCursor stopped. Reason: ...` | The exit message | 0:10 |

## Narration points (keep claims modest)

- "Everything runs locally; video is never recorded or sent anywhere."
- "A pinch is only a candidate until it is confirmed; the display says so."
- "Every stop path releases the mouse button first."
- Do **not** quote latency or frame-rate numbers in the narration unless you ran
  `nocursor --benchmark live` on that machine and say exactly what it measured. Never describe
  software timings as camera-to-display latency.
- Do not claim accessibility or production readiness; NoCursor is a hobby-grade prototype with
  documented limitations (see the README).

## After recording

1. Watch it through once: no private paths, notifications, or names visible.
2. Trim dead time; keep the status line readable (do not crop the preview window).
3. Add captions for the on-screen status text if the video is small.
4. Add alt text / a caption describing what the clip shows when you publish it.
5. Where it lives is your decision (for example a release asset or a link in the README); large
   video files are best not committed into the source tree.
6. Re-check the README so it describes only what the recording actually shows.
