# Benchmarking NoCursor

NoCursor can measure its own software performance on your machine. This document says exactly what
is measured, how, what is **not** measured, and how to read the output.

> **Software timings are not camera-to-display latency.** Every figure here is time spent inside
> the NoCursor process, taken with `time.perf_counter_ns`. None of them include the camera's
> exposure and USB/driver delay, the monitor, or the operating system's handling of the mouse
> event after NoCursor hands it over. The real delay you feel is larger than any number here, and
> NoCursor has no way to measure it. Nothing in the project claims otherwise.

## What a benchmark can never do

All benchmark modes are **dry-run by construction**. The controller they use has a backend that
can report a screen size and nothing else (`ScreenSizeOnlyBackend`); every input method raises.
The report states how many input events were actually sent (always `0`) and how many were merely
simulated, taken from the controller's own counters rather than assumed.

## The three modes

Results from different modes are never mixed or compared; each report and each table says which
mode produced it.

| | `synthetic` | `video` | `live` |
|---|---|---|---|
| Command | `nocursor --benchmark synthetic` | `nocursor --benchmark video --benchmark-video clip.mp4` | `nocursor --benchmark live --benchmark-seconds 30` |
| Frames come from | one fixed generated image, **no hand in it** | a clip you supply | your webcam |
| Hand landmarks come from | a **scripted** sequence (pointing, click, drag, scroll, tracking loss) | real MediaPipe detection on the clip | real MediaPipe detection on your hand |
| MediaPipe inference | real, on the hand-free image | real, on real frames | real, on real frames |
| Gesture + dispatch code | real, on the scripted landmarks | real | real (the actual app loop) |
| Pacing | none: runs as fast as it can | none: as fast as it decodes | paced by the camera |
| Preview window | no | no | yes (its cost is reported separately) |
| Repeatable | yes (same machine, similar load) | yes (same clip) | no: depends on what you do |
| What it tells you | the software cost of one frame on this machine | throughput on real hands, repeatably | what you actually get, including tracking losses |
| What it does **not** tell you | webcam performance; hand-present inference cost | camera pacing or preview cost | anything about other machines or lighting |

Important consequences:

- **Synthetic inference is the "no hand" path.** MediaPipe's palm detector runs on every frame
  when no hand is tracked; with a tracked hand it works differently and may be faster or slower.
  The synthetic figure therefore says nothing certain about frames that contain a hand. Use
  `video` or `live` for those. No direction is claimed.
- **Synthetic end-to-end is a composite.** Inference (on the empty image) and gesture/dispatch
  (on scripted landmarks) are measured separately within a frame and their sum is reported.
- **Synthetic frame rate is throughput.** It is how fast the software path can go with no camera
  to wait for, not what a 30 fps webcam delivers.
- **Video replay is unpaced.** `frame_wait` is video decoding, not a camera, and the first frames
  include model warm-up.
- **Live frame rate is camera-limited.** `frame_wait` includes waiting for the next frame.

## What is timed

All in milliseconds, per frame.

| Series | Boundaries |
|---|---|
| `frame_wait` | inside `camera.read()` (live) or video decode (video). Not in synthetic. |
| `inference` | `tracker.detect()`; also split into `inference_hand` and `inference_no_hand` by whether a hand was found |
| `gesture` | `processor.process()`: tracking checks, mapping, smoothing, pinch/scroll detection, state machine |
| `dispatch` | applying the frame's actions through `SafeController` (dry-run bookkeeping, no OS call) |
| `preview` | `preview.show()` (live only; **not** part of latency) |
| **`end_to_end` (software)** | from the frame arriving to the end of dispatch |
| `end_to_end_action` | the same, only for frames that carried a press, release or scroll |

Also reported: frames processed and dropped, the frame rate, **tracking loss events** (stable
tracking becoming lost or unreliable) and their rate per minute, the share of frames not tracking,
CPU use and memory.

### Percentiles and windows

- Percentiles are **nearest-rank** on the sorted samples. `p95` needs at least 20 samples and
  `p99` at least 100; with fewer, the report shows `n/a` rather than a misleading number.
- The live `--show-perf` overlay uses a **rolling window** of the last 300 frames, so one slow
  frame cannot dominate it. The benchmark reports use the whole session.
- The first `10` inferences of a synthetic run are discarded as warm-up. Video and live runs keep
  every frame, so their `max` and `p99` include model warm-up.

### CPU and memory

CPU utilisation is a ratio of **deltas**:

```
cpu_percent_of_one_core = 100 x (process CPU time used in the interval) / (wall-clock time of the interval)
cpu_percent_of_machine  = cpu_percent_of_one_core / logical_cpu_count
```

`cpu_time_s` is the absolute CPU time the process used (a total, not a rate). The first figure
can exceed 100 because MediaPipe uses several threads; the second is normalised to the whole
machine. On Windows the process-CPU clock ticks about every 15.6 ms, so very short intervals are
noisy; the sampler works over intervals of at least one second. Memory is the process working set
(resident memory) and its peak, from the operating system; nothing is estimated.

## Commands

```powershell
# repeatable software benchmark (no camera, no preview, no desktop input)
nocursor --benchmark synthetic                       # 1500 scripted frames
nocursor --benchmark synthetic --benchmark-frames 3000

# your own clip (read locally, never copied; only its file name is recorded)
nocursor --benchmark video --benchmark-video path\to\clip.mp4 --benchmark-frames 600

# your webcam, for 30 seconds (a preview opens; the mouse is never touched)
nocursor --benchmark live --benchmark-seconds 30
nocursor --benchmark live --benchmark-seconds 60 --show-perf   # also show the rolling overlay

# choose where the JSON report goes (default: benchmarks\<mode>-<UTC time>.json)
nocursor --benchmark synthetic --benchmark-output my-run.json
```

During a live benchmark, `Esc`/`Q` or `Ctrl+Alt+Q` end it early; the report then records why.

## The report

A terminal summary is printed and a JSON report is written (the `benchmarks/` folder is
git-ignored). The JSON contains:

- `mode`, `mode_title`, `generated_utc`, `schema_version`, the NoCursor version;
- `measurement_method`: the text above, in short;
- `conditions`: everything that shapes the numbers for that mode (frame size, whether a camera
  and preview were involved, the virtual clock, warm-up, source frame rate, hand-present frame
  count, how a live run ended, and the input evidence: `input_events_sent`,
  `input_events_simulated`, `controller_enabled`);
- `configuration`: the settings that matter (pinch mode and thresholds, smoothing, resolution);
- `environment`: Python, OS, CPU description and logical CPU count, package versions. No host
  name, user name or path is recorded;
- `results`: frames, durations, the frame rate and what it means, per-stage statistics,
  end-to-end statistics, tracking statistics, CPU and memory;
- `caveats` and `not_measured`: in the report itself, so a number is never separated from its
  limitations.

## Getting trustworthy numbers

1. Close other heavy programs; plug in a laptop; use the same power plan each time.
2. Run each benchmark at least three times and compare the *spread*, not one run.
3. Report the mode, the camera, the resolution and the machine with every number.
4. Compare like with like: synthetic with synthetic on the same machine, never synthetic with live.
5. Re-run after changing anything you want to evaluate; "faster" needs two measured runs.

## Results

*(Filled in below from actual runs. No result appears here that was not produced by the commands
above, and each states its mode and machine.)*

Only the **synthetic** mode has been run so far, because it needs no webcam or clip. **No video
or live-webcam result is published here**, since producing one needs your hand and your camera;
run those commands and keep the numbers with their mode, camera and machine.

### Synthetic mode (scripted input; not webcam performance)

Command: `nocursor --benchmark synthetic` (1500 scripted frames), run three times in a row on an
otherwise idle machine, from this repository's development environment (an editable install),
on 2026-10-11 (UTC). The reports were generated by the tool; every figure below was copied from
them by a script, not typed.

**Run 1 in full**

- **Mode:** SYNTHETIC (scripted input; not webcam performance)
- **Run:** 2026-10-11T19:33:59Z, NoCursor 0.1.0
- **Machine:** Windows 11 (build 10.0.26200), AMD64, 12 logical CPUs, CPU: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel
- **Software:** Python 3.12.0 (CPython), MediaPipe 1.1.0, NumPy 2.5.4, OpenCV 5.0.0.93
- **Settings:** 2D (default) pinch, 640x480, smoothing 0.5
- **Conditions:** 1500 scripted frames, 10 warm-up inferences discarded, gesture clock virtual, 30 fps (timings themselves use perf_counter); image: fixed synthetic gradient with no hand
- **Input:** 0 events sent to the desktop; 123 press/release/scroll events simulated; controller enabled: False
- **Frames:** 1500 in 82.441 s -> 18.2 frames/s (processing throughput (no camera pacing))

| Stage (ms) | n | mean | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| inference | 1500 | 54.3 | 56.0 | 67.4 | 77.0 | 115.1 |
| inference_no_hand | 1500 | 54.3 | 56.0 | 67.4 | 77.0 | 115.1 |
| gesture | 1500 | 0.4 | 0.4 | 0.5 | 0.7 | 1.1 |
| dispatch | 1500 | 0.0 | 0.0 | 0.1 | 0.2 | 0.3 |
| **end-to-end (software)** | 1500 | 54.8 | 56.5 | 67.8 | 77.5 | 115.6 |
| end-to-end, frames with an action | 123 | 57.7 | 57.3 | 74.6 | 80.8 | 85.5 |

- **Tracking (scripted):** 8 loss events; 7.6% of frames not tracking
- **CPU:** 97.2% of one core = 8.1% of the machine (12 logical CPUs); 80.266 s of CPU time over 82.55 s
- **Memory:** 111 MB resident, 169 MB peak


**The three runs side by side** (same machine, same settings; the spread is the point):

| Measure | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Frames | 1500 | 1500 | 1500 |
| Duration (s) | 82.4 | 83.2 | 82.5 |
| Throughput (frames/s) | 18.2 | 18.0 | 18.2 |
| Inference mean / p50 (ms) | 54.3 / 56.0 | 54.8 / 56.4 | 54.4 / 56.6 |
| Inference p95 / p99 / max (ms) | 67.4 / 77.0 / 115.1 | 70.3 / 83.5 / 127.9 | 67.3 / 75.6 / 123.0 |
| Gesture mean / p99 (ms) | 0.37 / 0.70 | 0.37 / 0.76 | 0.37 / 0.66 |
| Dispatch mean / p99 (ms) | 0.019 / 0.175 | 0.020 / 0.186 | 0.019 / 0.178 |
| End-to-end (software) p50 (ms) | 56.5 | 56.9 | 57.0 |
| End-to-end p95 (ms) | 67.8 | 70.7 | 67.7 |
| End-to-end p99 (ms) | 77.5 | 83.9 | 76.1 |
| End-to-end max (ms) | 115.6 | 128.2 | 124.2 |
| CPU, % of one core | 97.2 | 97.5 | 97.9 |
| CPU, % of whole machine | 8.1 | 8.1 | 8.2 |
| Memory resident / peak (MB) | 111 / 169 | 111 / 169 | 111 / 169 |
| Input events sent to the desktop | 0 | 0 | 0 |
| Press/release/scroll events simulated | 123 | 123 | 123 |

**What this does and does not show**

- On this machine, the software path for one frame **with no hand in view** costs about
  55 ms, almost all of it MediaPipe inference (gesture processing is about 0.4 ms and
  dry-run dispatch about 0.02 ms). That is a throughput of about 18 frames per second when nothing
  waits for a camera.
- The loop kept one core about 97% busy, so on this path it is compute-bound rather than waiting.
- The median is stable across runs (56.5 to 57.0 ms end to end); the tail varies more (p99 76 to
  84 ms, max 116 to 128 ms). That is consistent with scheduling noise on a general-purpose
  operating system, but its cause was not investigated.
- **It does not say what a webcam session feels like.** The image contains no hand; with a
  tracked hand MediaPipe works differently, so hand-present frames may be faster or slower. A
  webcam also limits the frame rate itself. Use `--benchmark live` (or `video` with a clip of
  your hand) for those.
- It says nothing about camera exposure, USB transfer, the monitor, or the time Windows takes to
  deliver a mouse event; none of that is in these figures. They are **not** camera-to-display
  latency.
- An earlier informal run during development (not in the tables) showed a higher median inference
  time, around 68 ms: figures move with machine load, so treat them as one machine's measurements,
  not guarantees.
- Dispatch here is dry-run bookkeeping (the controller counted 123 simulated press, release and
  scroll events and sent 0 to the desktop); a live run also pays for the operating-system call.

### Video replay and live webcam

*No results yet.* Run `nocursor --benchmark video --benchmark-video your_clip.mp4` and
`nocursor --benchmark live --benchmark-seconds 30` and keep each report with its mode.

## Known limits of this benchmark

- It measures this process only; other processes, the GPU driver and Windows scheduling are not
  separated out, which is why repeated runs differ.
- Python's `perf_counter_ns` resolution is fine, but the OS can pause the process at any moment;
  `max` values include such pauses.
- MediaPipe's own internal stages (palm detection vs landmark model) are not timed separately.
- There is no accuracy benchmark: nothing here says how *correct* the landmarks are.
- Only the primary-monitor screen size is used (a fixed 1920x1080 virtual screen in synthetic and
  video modes); it affects pointer coordinates, not timings.
