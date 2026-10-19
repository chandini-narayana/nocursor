# Dynamic gesture research extension

This is a research extension to NoCursor: can a small temporal classifier recognise three
open-palm hand gestures from sequences of MediaPipe hand landmarks, and does it do so better than
a hand-written rule-based baseline on recordings it has never seen?

**Current status: offline experiments only.** Recording, validation, feature extraction, a
session-grouped split, a rule-based baseline, GRU training, NumPy-only inference and an offline
evaluation are implemented and can be run today. Nothing in this extension is connected to the
mouse or to the existing pointer, click, drag and scroll controls, and it cannot send desktop
input. Whether a trained model is any good is a question for the evaluation report that
`evaluate` writes for your own recordings; this guide states no results.

| Part | Status |
|---|---|
| Recording tool, dataset format, validation, summary | available (`collect`, `validate`, `summarize`) |
| Feature extraction (resampling, normalisation) | available |
| Session-grouped split, frozen before training | available (`split`) |
| Rule-based baseline, untuned and tuned | available (`fit-baseline`, also run by `train`) |
| GRU training (optional PyTorch, separate environment) | available (`train`) |
| NumPy-only model loading and inference | available; the application never needs PyTorch |
| Offline evaluation, reports, continuous-stream analysis | available (`evaluate`) |
| Live use of a trained model | not built; needs your review of the offline results first |

## Research question and design

> Using landmark sequences recorded in several separate sessions, can a small GRU classifier
> distinguish `swipe_left`, `swipe_right`, `circle_clockwise` and `no_gesture`, and does it beat
> a rule-based baseline on a held-out *session*, including how often it fires during ordinary use?

The design: whole sessions are assigned to train, validation and test with a seeded shuffle and
the assignment is frozen before anything is trained; a rule-based baseline (untuned and tuned) and
a GRU are fitted on the training and validation data only; both are then scored on the same
held-out test takes, and on a reserved continuous recording for false activations per minute.
Everything is local, and the data is one person's recordings, so the evaluation is a small pilot.

## The gestures

All three target gestures are performed in the **open palm** pose and are described as seen on
the **mirrored preview** (the window behaves like a mirror: move your hand to your right and the
hand on screen moves right).

| Label | What you do |
|---|---|
| `swipe_left` | One sweep to **your left**, about two palm-lengths, then stop |
| `swipe_right` | One sweep to **your right**, about two palm-lengths, then stop |
| `circle_clockwise` | One full circle, **clockwise as it looks on the mirrored preview** (like clock hands as you see them), one to two palm-lengths across |
| `no_gesture` | Everything else, including things that look similar (see below) |

A palm-length is the distance from your wrist to the knuckle of your middle finger; it is also the
unit all distances are measured in, so camera distance does not matter.

**Open palm** means the index, middle, ring and little fingers are all clearly extended. The thumb
is not judged. "Extended" uses the same finger measure as the existing two-finger scroll pose
(wrist-to-fingertip over wrist-to-knuckle distance in the image plane), so the three poses stay
distinct: pointing is index only, scrolling is index and middle with the other two folded, and
open palm is all four. Those thresholds come from hand proportions and have not been fitted to
real hardware. Requiring a pose is what keeps the new gestures apart from ordinary cursor
movement, which is itself a hand moving sideways.

After a swipe, stop or lower the hand rather than snapping it back: a fast return stroke looks
like a swipe in the other direction.

### Hard negatives

`no_gesture` takes are not just an idle hand. Each carries one of these types, because each is a
movement the classifier must learn *not* to react to:

| Type | What to record |
|---|---|
| `idle` | Hand held still in view, any pose |
| `pointing` | Pointing with the index finger and steering the cursor around, including sideways sweeps and circles |
| `pinch` | Pinching thumb and index together and releasing, as for a click |
| `drag` | Pinching and moving the hand while pinched |
| `scroll` | The two-finger scroll pose, moving up and down |
| `open_palm_other` | Open palm doing something that is **not** a target: up/down sweeps, waving, slow drift, half circles, **counter-clockwise** circles, a small wiggle |
| `ordinary_movement` | Reaching, adjusting, rubbing, gesturing while talking, any pose |
| `entering_leaving` | Bringing the hand into view or taking it out |

Separately, a **continuous-use recording** (`--stream-seconds`) captures minutes of ordinary
computer use as one negative recording. It is the only honest way to measure, later, how often a
classifier would fire by accident per minute.

## Recording protocol

Install and check NoCursor first (`nocursor --check`; the hand-landmark model comes from
`nocursor --download-model`). Then, from the project folder with the environment active:

```
python -m nocursor.dynamic collect --participant p01 --hand right --conditions "daylight, 60 cm"
```

A session with the default protocol is 75 takes: 15 of each of the three gestures and 30
`no_gesture` takes spread evenly over the eight hard-negative types, in a shuffled order that is
balanced block by block, so stopping early still leaves a balanced session. Budget roughly ten
minutes including pauses (this has not been timed). For each take the window shows what to do,
then a short countdown, then a red recording indicator for 1.6 seconds.

Keys: **Space** records the next take, **Backspace** cancels the take in progress or deletes the
last saved one (the protocol asks for it again), **N** skips an item (in `--free` mode it moves to the next label), **T** cycles the
negative type and **1-4** pick the label (those two only in `--free` mode), **Esc** or **Q** quits.

### How much data, and how to spread it

| | Bare minimum to attempt a split | Recommended |
|---|---|---|
| Sessions | 3 (so that train, validation and test can each be a whole session) | 6 or more |
| Takes per target gesture per session | a handful | 15 |
| `no_gesture` takes per session | a handful of each type | 30, covering every type |
| Continuous-use recordings | 0 | 1 per session, 2 to 5 minutes (`--stream-seconds 120`) |

Only the session count is checked today (`summarize` warns below 3 and below 6). Per-class minimums
will be enforced, with clear errors, when the training stage exists.

Totals at the recommended level are 90 takes of each gesture, 180 negatives and six continuous
recordings. These are a data-sufficiency guide, not an accuracy promise: more sessions matter more
than more takes within a session.

Because the later split is **by session**, sessions should differ from each other. Record on
different days or times, with different lighting, and at somewhat different distances and camera
angles. Do not record all sessions back to back in one sitting; they would be near-copies of each
other and the held-out test would be too easy to be informative. Record each hand you want the
system to support as separate sessions, and say which in `--hand`.

Other tips:

- Keep the whole hand in view and well lit. A take is rejected, with a reason on screen, if the
  hand was tracked reliably in too few frames, if tracking dropped for a long stretch (very fast
  swipes can blur the hand), or if a gesture take was not an open palm.
- The recorder never rejects a take because the *motion* looks wrong. That would keep only the
  easy examples. `summarize` instead lists takes whose motion contradicts the label (for example a
  swipe labelled left that went right) so you can look again. Nothing is deleted or relabelled
  automatically.
- A rejected take is not saved and the same item is offered again.
- `--free` skips the protocol so you can add extra examples of any type you are short of.

### Options

| Option | Meaning |
|---|---|
| `--dataset-dir DIR` | where session files go (default `datasets/` in the project) |
| `--participant ID` | pseudonymous id such as `p01` (default `p01`); never a name or e-mail address |
| `--hand {left,right,unspecified}` | the hand you record with |
| `--conditions TEXT` | one short line about the setup, with no personal information |
| `--camera N`, `--width W`, `--height H` | camera index and capture size (defaults 0, 640, 480) |
| `--model PATH` | path to `hand_landmarker.task` |
| `--min-confidence X` | minimum hand-detection confidence (default 0.6) |
| `--countdown S` | seconds of countdown before each take (default 2) |
| `--per-class N` | takes of each gesture in the protocol (default 15) |
| `--negatives N` | `no_gesture` takes in the protocol (default 30) |
| `--seed N` | seed for the protocol order (default random, recorded in the file) |
| `--free` | no protocol; choose each take's label yourself |
| `--stream-seconds S` | record one continuous-use recording of S seconds (5 to 600) instead of takes |
| `--min-palm-fraction X` | share of tracked frames that must be an open palm in a gesture take (default 0.8) |

### Inspecting what you have

```
python -m nocursor.dynamic summarize
python -m nocursor.dynamic validate
```

`summarize` prints counts per class, per session and per negative type, how many takes can be
turned into features, takes whose motion contradicts their label, and notes comparing the dataset
with the table above. `summarize --json` prints the same as JSON. `validate` checks every file
line by line, prints each problem with its file and line number, and exits with 1 if any file or
sequence is unusable. Both accept `--dataset-dir`.

## Dataset format (schema version 1)

One session is one UTF-8 JSON Lines file, `datasets/<session_id>.jsonl`. The first line is the
session header; every later line is one sequence. A file is created fresh for each run and an
existing one is never overwritten. Each accepted take is appended and flushed immediately, so a
crash loses at most the take in progress.

Header (abridged):

```json
{"record":"session","schema_version":1,"session_id":"s20261012T100000Z-3fa9",
 "participant_id":"p01","created_utc":"2026-10-12T10:00:00Z","hand":"right",
 "provenance":"recorded","camera":{"index":0,"frame_width":640,"frame_height":480},
 "tool":{"nocursor":"0.1.0","mediapipe":"1.1.0","hand_landmarker_sha256":"..."},
 "protocol":{"mode":"protocol","seed":123,"take_seconds":1.6,"planned_takes":75},
 "conditions":"daylight, 60 cm"}
```

Sequence (abridged):

```json
{"record":"sequence","sequence_id":"s20261012T100000Z-3fa9-0001","session_id":"s20261012T100000Z-3fa9",
 "kind":"take","label":"swipe_left","subtype":null,"aspect_ratio":1.333333,
 "t_ms":[0,33,67],"usable":[true,true,false],"confidence":[0.93,0.95,0.0],
 "landmarks":[[[0.61,0.74,-0.0],"... 21 landmarks ..."],[[0.6,0.74,-0.0],"..."],null]}
```

| Field | Meaning |
|---|---|
| `kind` | `take` (a 1.6 s labelled recording) or `stream` (a long continuous-use recording, always `no_gesture`) |
| `label` | `swipe_left`, `swipe_right`, `circle_clockwise` or `no_gesture` |
| `subtype` | for `no_gesture` takes, one of the hard-negative types; `continuous_use` for streams; otherwise null |
| `t_ms` | frame times in milliseconds, relative to the first frame (so it starts at 0) |
| `landmarks` | per frame, 21 landmarks as `[x, y, z]` exactly as MediaPipe returned them (unmirrored image, normalised, y down), or `null` when no hand was detected |
| `usable` | per frame, whether the hand passed NoCursor's trust checks (confidence, fully inside the frame, plausible size) |
| `confidence` | per frame detection confidence (0 where there was no hand) |
| `aspect_ratio` | frame width over height, needed to compare x and y distances |
| `participant_id` (header) | a pseudonym; see Privacy |
| `provenance` (header) | `recorded`, or `synthetic` for software-test fixtures, which are never to be used for accuracy claims |

Validation rejects: unknown labels or subtypes, a subtype on a gesture take, fewer than 8 frames,
durations outside 0.4 to 6 s (5 to 1800 s for streams), timestamps that are not strictly
increasing, start at a non-zero time or contain a stall of more than 0.5 s, landmark arrays of
the wrong shape, partly missing frames, non-finite or absurd values, confidence outside 0 to 1,
`usable` frames without landmarks, fewer than two usable frames, duplicate sequence ids, and
sequences that belong to another session. Damaged lines are reported with their line number.
Identical landmark data appearing twice is a warning. A reader refuses any `schema_version` it
does not know.

## Features

Training will use features computed from these recordings by one deterministic function, so the
same preprocessing can be applied live. In summary:

1. Landmarks are converted to *view* coordinates: x is mirrored (so the numbers match the
   mirrored preview) and multiplied by the aspect ratio, so one unit means the same distance in x
   and y. y still points down.
2. The sequence is resampled onto a fixed grid of 32 time steps over 1.6 s, so the camera's frame
   rate does not matter. Dropouts of up to 0.2 s are bridged by interpolation; a longer dropout,
   or fewer than 80% trusted frames, means the take cannot be used, rather than being guessed.
3. Distances are divided by hand size (wrist to middle-finger knuckle), so distance from the camera
   and the frame's dimensions do not matter.
4. Each time step becomes 46 numbers: the palm's position relative to where it started (2), its
   velocity (2), and the 21 landmarks relative to the palm centre (42).

The palm trajectory keeps its sign, so a swipe to the left has a negative x trajectory and a swipe
to the right a positive one, and a clockwise circle has a positive signed area (y down). That is
tested directly, including from raw, unmirrored camera coordinates. Depth (MediaPipe's z) is
stored but never used, because it is coarse and the existing 2D pinch default was chosen after
depth caused real false rejections. A hash of the feature definition will be stored with any
trained model so that a mismatch is detected rather than silently ignored.

## Privacy

- **Only landmark numbers are saved. No video or image is ever written** by this tool, and nothing
  is uploaded. The window says so while you record.
- Landmark sequences still describe the shape and movement of your hand, so treat them as personal
  data. They stay on your machine in `datasets/`, which is excluded from version control, as is
  `ml_runs/` (where trained models and reports will go). Anything derived from your recordings
  should be reviewed by you before it is added to Git or shared.
- Participants are identified only by a **pseudonym** you choose (`p01`). The tool rejects ids
  that look like names or e-mail addresses (only lowercase letters, digits, `_` and `-` are
  accepted), but it cannot stop you choosing your own name as a pseudonym: please don't. If you
  need to remember who is who, keep that mapping outside the dataset.
- The header holds no user name, host name or file path. It does hold a timestamp, the camera
  index and frame size, package versions, and the hash of the hand-landmark model file. The
  `--conditions` text is free-form: keep personal information out of it.
- Deleting a recording means deleting its `.jsonl` file.

### Adding other participants later

The format already carries `participant_id`, so recordings from other people can be added as
their own sessions. Only record someone with their informed consent, give them their own
pseudonym, and let them delete their sessions on request. Until sessions from more than one person
exist and have been evaluated with participants held out, nothing can be claimed about
generalisation across people. `summarize` says so whenever there is only one participant.

## Limitations and known sources of bias

- **The tests use synthetic data.** Synthetic hands check software behaviour. They say nothing
  about how real hands, cameras or lighting behave, and any report produced from them is stamped
  SYNTHETIC. Numbers that mean anything come only from your own recordings.
- **One person, one camera so far.** Even a good result would show generalisation across sessions
  only, not across people, hand sizes, skin tones, cameras or rooms.
- **One held-out test session and one validation session.** The test evidence is limited however
  many takes the session holds, the intervals in the report understate the uncertainty, and the
  validation session chooses the model settings, the stopping point and the rejection threshold
  with only a few dozen negative takes. A different seed would hold out different sessions.
- **The continuous stream is a single short recording.** It is limited evidence about false
  activations per minute and not proof of reliable operation, whatever the count.
- **Excluded takes were never seen by the classifiers.** Takes with unreliable tracking, mostly a
  hand entering or leaving view, are excluded from model inputs and from the held-out scores.
- **The baseline is tuned on training and validation together**, and the GRU's settings are chosen
  on validation: neither comparison is free of the small size of those sets.
- **Performed, not spontaneous.** Takes are deliberate and cued by a countdown. A person doing a
  gesture on purpose behaves differently from one making a similar movement by chance.
- **Labels come from the performer's intent**, not from an independent annotator. The review
  warnings catch contradictions, not subtle mislabelling.
- **Negatives are never exhaustive.** The hard-negative list is a deliberate attempt, not a
  guarantee of coverage.
- **Fixed 1.6 s windows.** Takes are cut to a fixed length, but live use would need a sliding
  window, so behaviour on the unsegmented continuous-use recordings matters and cannot be assumed
  from per-take results.
- **Fast gestures can lose tracking.** The motion blur of a quick swipe can make MediaPipe drop
  the hand; such takes are rejected, which biases the dataset toward gestures the tracker can
  follow. The number of rejected takes is shown while recording.
- **Open-palm thresholds are estimates** from hand proportions, not fitted to a camera.
- **The window and frame rate depend on the camera.** Auto-exposure and low frame rates in dim
  light change what the tracker sees.

## Environment for training

NoCursor's own environment is unchanged and never needs PyTorch. Training uses a **separate,
isolated environment** in the project folder, `.venv-ml/` (excluded from Git), with PyTorch,
NumPy and pytest only (no OpenCV, MediaPipe or PyAutoGUI). From the project folder, in PowerShell:

```
py -3.12 -m venv .venv-ml
.\.venv-ml\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ml\Scripts\python.exe -m pip install numpy pytest torch
.\.venv-ml\Scripts\python.exe -m pip install -e . --no-deps
```

On Windows the PyPI `torch` wheel is the CPU build. `pip install -e ".[ml]"` (the `[ml]` extra is
just `torch`) is the equivalent for a single environment, but a separate one keeps NoCursor's
dependencies exactly as they were. The versions actually used are recorded in every model's
`model.json`. Run the tests that need PyTorch (and the rest) there with
`.\.venv-ml\Scripts\python.exe -m pytest tests -q`; in the ordinary environment they are skipped
and pytest says so. Everything except `train` works in the ordinary environment.

## Splitting the data

Takes from one session share lighting, posture and habits, so a random split of takes would let a
model recognise the *session* and overstate how well it generalises. `split` therefore keeps whole
sessions together:

- A session with at least 10 takes of every gesture is **full**; one with fewer is a **pilot** and
  may only add training examples; a recording with only continuous use is the **reserved stream**.
- Full sessions are sorted by id and shuffled with `random.Random(seed)`; the first
  `--test-sessions` go to test, the next `--validation-sessions` to validation and the rest to
  train. The default seed is 0, chosen before any model was trained and not to be changed after
  seeing results.
- The continuous stream is in no split. Nothing opens it except the final false-activation
  analysis: it is not used for training, validation, model selection or any threshold.
- Takes that cannot become features (mostly a hand entering or leaving view, where tracking
  dropped) are **excluded from model inputs** and listed with their reason. Takes the label review
  flagged are **never removed or relabelled**; the manifest records how each is treated.
- Checks run mechanically when the split is made and again before use: sessions and sequences are
  disjoint, identical landmark data is not shared between splits, pilots are not in validation or
  test, the stream is in no split, and all four classes are present in every split. It fails with
  a clear message if there are too few full sessions or too few takes per class.
- The manifest, `ml_runs/splits/<split_id>/split.json`, records the seed, the algorithm, every
  session's role and file checksum, class counts before and after exclusions, every excluded and
  flagged take, and a hash of the whole dataset. It is written once and frozen. Training and
  evaluation refuse to run if a dataset file no longer matches it. A run keeps a copy.

One held-out full session is limited evidence even when it holds many takes: they are not
independent trials, and the confidence intervals in the report, which treat them as if they were,
understate the real uncertainty.

## The rule-based baseline

The baseline uses the same features as the GRU: signed horizontal displacement, path length and
straightness, peak speed, and signed turning angle with closure for the clockwise circle. Left and
right follow the mirrored-preview convention. It returns `no_gesture` unless every criterion of a
gesture holds, and has no confidence, so it never "rejects".

- **rule_defaults**: thresholds fixed in the code from the geometry of the gestures as described
  to the person recording them, before any data was scored.
- **rule_tuned**: a deterministic coordinate search that maximises macro-F1 on the pooled
  *training and validation* takes only, run from the defaults and from the loosest setting (the
  better result is kept, because a gesture must satisfy several criteria at once and loosening one
  alone can leave every prediction unchanged). It never sees the test split or the stream. Because
  it reads training and validation together, it has slightly more data for choosing its few
  numbers than the GRU has for gradient updates, which favours the baseline.

## The GRU

A one-layer GRU over the 32 x 46 feature sequence, then dropout and a linear layer to the four
classes. Inputs are standardised with a mean and standard deviation from the training takes only
(the deviation floored at 1% of a hand size, so tracker noise is not amplified).

- **Training data only** drives gradient updates. **Validation** drives early stopping, the choice
  between the few hyper-parameter settings (hidden size, weight decay) and the rejection threshold.
- **Augmentation** (training only, seeded): gesture speed, gesture amplitude, a few degrees of
  rotation and landmark jitter. There is no horizontal flip, because a mirrored circle is not a
  class. Each perturbation is tested to leave the direction of a swipe and of a circle unchanged.
- **Selection:** every setting is trained with several seeds; the setting with the best mean
  validation macro-F1 wins, and its best seed is the final model. The other seeds are kept so the
  report can show how much the result depends on the seed.
- **Rejection threshold:** the smallest value on a fixed grid at which no validation `no_gesture`
  take triggers a gesture. A predicted gesture whose probability is below it becomes `rejected`:
  it triggers nothing. Validation has few negative takes, so the threshold is a coarse estimate.
- **Flagged takes** are kept in the primary run. A separate run with `--exclude-flagged` leaves
  them out of training and validation, for the sensitivity comparison.

### The model artifact

Each run is a folder under `ml_runs/` (excluded from Git; review anything before sharing it):

```
ml_runs/splits/<split_id>/split.json    the frozen split
ml_runs/<run>/split.json                a copy, so the run is self-contained
ml_runs/<run>/baseline.json             both rule baselines and how the tuned one was found
ml_runs/<run>/model.npz                 weights and standardisation arrays, no pickle
ml_runs/<run>/model.json                everything needed to use and to trust them
ml_runs/<run>/train_log.json            losses per epoch, selection table, threshold, warnings
ml_runs/<run>/seeds/seed-N/             the other seeds of the chosen setting, for stability
ml_runs/<run>/eval/eval-NNN-<time>/     metrics.json, report.md, confusion-matrix images
```

`model.json` holds the class names, the artifact and feature schema versions, the feature
definition and its hash, the architecture, the rejection threshold and how it was chosen, a
checksum of the weights, the seed, the training configuration and selected epoch, library
versions, and the dataset and split identifiers. Loading validates all of it, uses
`np.load(..., allow_pickle=False)`, and raises a clear error for anything missing, corrupt,
incompatible (class names, feature hash, tensor names, shapes, dtypes) or not finite, including a
weights file that was edited. The forward pass is plain NumPy; at the end of training the saved
model is loaded and compared with PyTorch on the validation takes, and the run fails if they
disagree. A model that produces unusable output is treated as a rejection, never as a guess.

## Commands

Make the split once, from the ordinary environment:

```
python -m nocursor.dynamic split
```

Train in the ML environment (`<id>` is the split id that `split` printed). The second run is the
sensitivity run without the flagged takes:

```
.\.venv-ml\Scripts\python.exe -m nocursor.dynamic train --split <id> --run-name primary
.\.venv-ml\Scripts\python.exe -m nocursor.dynamic train --split <id> --run-name sensitivity --exclude-flagged
```

Then evaluate. This is the only step that opens the test split and the reserved stream:

```
.\.venv-ml\Scripts\python.exe -m nocursor.dynamic evaluate --run primary --sensitivity-run sensitivity
```

`fit-baseline` does just the rule baselines and needs no PyTorch. `evaluate` also works in the
ordinary environment (it never needs PyTorch). In the ordinary environment the commands read
`python -m nocursor.dynamic fit-baseline --split <id>`, `python -m nocursor.dynamic evaluate
--run <name>`, and (only where PyTorch is installed) `python -m nocursor.dynamic train
--split <id>`.

| Option | Meaning |
|---|---|
| `split --seed N` | shuffle seed (default 0) |
| `split --validation-sessions N`, `--test-sessions N` | full sessions held out (default 1 each) |
| `--dataset-dir DIR` | where the recorded sessions are |
| `--runs-dir DIR` | where splits, models and reports go (default `ml_runs/`) |
| `--split ID_OR_FILE` | the split to use: its id, or the path of its `split.json` |
| `--run-name NAME` | name of the run folder (default `run-<UTC time>`; never overwritten) |
| `--exclude-flagged` | sensitivity run: leave flagged takes out of training and validation |
| `--allow-synthetic` | software tests only: permit synthetic sessions; outputs are stamped SYNTHETIC |
| `train --seeds N` | seeds per setting, 0 to N-1 (default 5) |
| `train --max-epochs N` | epoch limit per model (default 150) |
| `train --hidden-sizes A B` | GRU sizes to choose between (default 32 64) |
| `evaluate --run DIR_OR_NAME` | the run to score |
| `evaluate --sensitivity-run DIR_OR_NAME` | a run trained with `--exclude-flagged` on the same split |
| `evaluate --no-stream` | skip the continuous-stream analysis |

## Evaluation

`evaluate` re-verifies the frozen split, then scores `rule_defaults`, `rule_tuned` and the GRU on
the same test takes. Every scoring of a run's test split is counted in its `eval_log.json` and the
report says which evaluation number it is.

- **Conventions.** A rejected take counts as `no_gesture` in accuracy and F1, because it triggers
  nothing; the confusion matrix keeps a separate `rejected` column, and the number of true
  gestures lost to rejection is reported. A *false activation* is a `no_gesture` take the system
  would act on. A *cross activation* is a true gesture that triggered the wrong gesture.
- **Metrics:** per-class precision and recall, macro-F1, accuracy, confusion matrices, rejection
  rate, false activations (also by kind of negative), and software latency (p50 and p95, feature
  extraction timed separately, "insufficient measurements" below 30 samples). Proportions have
  Wilson intervals and macro-F1 a bootstrap interval; they treat takes as independent.
- **The continuous stream.** A 1.6 s window slides over the reserved recording every 0.1 s;
  windows with unusable tracking are skipped (and counted), as the live gate would. Flagged
  windows of one class within a second of each other form one event. The report gives events,
  events per minute with an exact Poisson interval, and the same for events lasting three or more
  windows. One short recording is limited evidence: zero events in two minutes is still
  compatible with a true rate near 1.8 per minute, and overlapping windows are not independent.
- **Sensitivity to the flagged takes.** The held-out metrics are shown with and without the
  flagged takes, and, with `--sensitivity-run`, for the model retrained without them.
- **Seed stability.** If the run kept its other seeds, each is scored on the test takes for
  transparency; none of it is used to choose a model.

Outputs are `metrics.json` (machine-readable) and `report.md`, plus a confusion-matrix image when
matplotlib is installed (the text matrices in the report are always complete). Both carry the
split, dataset provenance, model configuration and a limitations section. They derive from your
recordings, so review them before adding anything to Git.

### Reading the results

Read the confusion matrices before the headline numbers: a high score can hide a gesture that
is never recognised or one that fires on ordinary movement, and a cross activation (a left swipe
read as a right swipe) is as unwelcome as a false activation. Compare systems on the same takes,
look at the intervals, and remember that the false-activation rate on the stream is the figure
that matters most for any live use and is the least certain.

## Reproducibility

Everything random is seeded and recorded: the split seed, the training seeds, the augmentation
generator and the bootstrap. The split manifest and the artifacts carry hashes of the data,
the feature definition and the weights, and the library versions used. Training the same data with
the same configuration in the same environment gives identical weights (tested); bit-identical
results on a different machine or PyTorch build are not promised. Reports record the evaluation
number and the checksums, so a result can be traced to exactly the data and model that made it.

## Streaming robustness checkpoint

The first evaluation counted flagged sliding windows on one 120 s recording and found far too many
false events per minute for any live use. This checkpoint adds, **offline only**, an explicit
open-palm gate and a temporal stabilizer, measures what they cost in genuine-gesture recall, and
defines a one-shot acceptance test. Nothing here is connected to the mouse. The earlier split,
runs and `eval-001` report are historical evidence and are never modified; everything new goes
under `ml_runs/streaming/`, and new recordings go in subfolders of `datasets/` so that the frozen
split keeps verifying.

### The pipeline

`classifier -> open-palm gate -> stabilizer -> discrete activation`

- **Gate** (`gate.py`): the share of the window's trusted frames that are an open palm (the
  existing `is_open_palm`: index, middle, ring and little finger clearly extended, thumb ignored)
  must reach a minimum fraction. Takes were recorded with an open-palm requirement of 0.8, so a gate
  at or below 0.8 costs *takes* no recall by construction; the real loss is underestimated.
- **Stabilizer** (`stabilizer.py`): a window is a candidate if it was evaluated, its prediction is
  valid, it passed the gate, it predicts a gesture and its confidence reaches `tau` and the model's
  own rejection threshold. It fires when `k` of the last `n` windows agree, then waits for a
  cooldown (and a further guard for the opposite swipe, because a fast return stroke looks like a
  swipe the other way) and, optionally, for a quiet stretch before it can fire again. A tracking
  loss or invalid output clears the vote.
- **Activation**: the unit that is counted. One movement is one activation, not sixteen flagged
  windows.

### Data folders and roles

| Folder | Role | May be used for |
|---|---|---|
| `datasets/` (top level), `ml_runs/primary`, `ml_runs/splits/...` | Historical, read only | Comparison. The 120 s stream is now **development** evidence |
| `datasets/streaming/dev/` | Development | Choosing the gate, `tau`, vote, cooldown; leave-one-recording-out |
| `datasets/streaming/final/` | **Sealed** | Opened only by `stream-accept`, once |

Record with the existing tool, pointing `--dataset-dir` at the folder:

```
python -m nocursor.dynamic collect --dataset-dir datasets/streaming/dev --stream-seconds 600 --conditions "window light, 70 cm, browsing"
python -m nocursor.dynamic collect --dataset-dir datasets/streaming/dev --participant p01 --hand right --conditions "..."   # protocol session
```

Development streams: 30 to 40 minutes in total, several recordings, at least four different
combinations of day, lighting, distance and task (browsing, typing and mouse work with the hand
returning, a call with natural gesturing, a hand resting in view, reaching for things). Do not
deliberately perform the three target gestures. Describe the conditions truthfully in
`--conditions`. Final streams are recorded only after the freeze (below).

### Commands

```
python -m nocursor.dynamic stream-eval --run primary        # preliminary, development only, selects nothing
python -m nocursor.dynamic stream-tune --run primary        # needs >= 2 dev recordings; leave-one-recording-out
python -m nocursor.dynamic stream-freeze --run primary --tuning-report <metrics.json>
python -m nocursor.dynamic stream-accept --run primary --freeze <freeze-id>             # the one attempt
python -m nocursor.dynamic stream-accept --run primary --freeze <freeze-id> --finalize  # after review
```

| Option | Meaning |
|---|---|
| `--run DIR_OR_NAME` | the trained run whose systems are scored (all four commands) |
| `--dataset-dir DIR`, `--runs-dir DIR` | where recordings and runs are (defaults as above) |
| `--dev-path PATH` | extra development session files or folders; repeatable; the sealed final folder is refused |
| `--no-legacy-stream` | leave out the original 120 s stream |
| `--no-validation-takes` | leave out the run's validation-session takes (no recall is then measured) |
| `stream-tune --system NAME` | `gru` (default), `rule_defaults` or `rule_tuned` |
| `stream-freeze --tuning-report FILE` | the `metrics.json` of a `stream-tune` run; its chosen setting is frozen |
| `stream-freeze --primary NAME` | the system the acceptance is judged on (must match the tuned one) |
| `--final-dir DIR` | where the sealed final recordings go (default `datasets/streaming/final`) |
| `stream-accept --freeze ID_OR_DIR` | the freeze to accept against |
| `stream-accept --finalize` | after review: read `review_marks.json` and write the verdict |
| `stream-review --items FILE` | the `review_items.json` to review |
| `stream-review --marks FILE` | where marks are saved (default: next to the items) |
| `stream-review --recordings PATH` | session files or folders holding the recordings; repeatable |
| `stream-candidate --name NAME` | name of the partial-window candidate model (never overwritten) |
| `stream-compare --candidate NAME_OR_DIR` | a candidate to include in the comparison; repeatable |
| `stream-compare --review-dir DIR` | a review folder from an earlier comparison, to count reviewed candidates apart |

`stream-eval` writes `ml_runs/streaming/dev-eval-NNN-<time>/` with a fixed, untuned ladder (one
mechanism added per row) and a trade-off frontier over the declared grid. Because the dev data is
small, the frontier is in-sample and optimistic. `stream-tune` refuses to run with fewer than two
development recordings.

### Recall under streaming

Recall is measured by *replaying* isolated labelled takes through the same pipeline: the take's
first and last frames are held still for a few seconds each side and the windows slide over it at
four alignments. A take counts as recalled only if it is recalled at **every** alignment, and as a
cross activation if it triggers a wrong gesture at **any** alignment. An activation belongs to a
take if it is decided after the take began and no later than one window after it ended. Missed
takes are attributed to the model or confidence, the gate, or the stabilizer. This is cleaner than
live use, where the hand does not rest perfectly still before and after a gesture.

### Freeze and the one-shot acceptance

1. `stream-freeze` writes `freeze.json`: the model or baseline hash, the decision settings, the
   acceptance criteria, the declared grid hash and the SHA-256 of every relevant source file. It
   refuses if any file already exists in `datasets/streaming/final/`, and it is never overwritten.
2. Only then are the final recordings made (at least four streams totalling 40 minutes on at least
   two days and two described conditions, plus two full protocol sessions).
3. `stream-accept` verifies the code and model against the freeze, rejects any session created
   before the freeze, checks the data requirements *before* using up the attempt, writes a marker,
   and scores the sealed data once. A second run is refused; a failed or repeated attempt needs a
   new final dataset after any change.
4. Candidate activations of all systems are merged into review items that do not name the system,
   shuffled with a seed. Each is marked `false`, `genuine` or `ambiguous` in `review_marks.json`.
   Items give the recording and time range; mark them with the review viewer (below).
5. `--finalize` writes the verdict. **Ambiguous candidates count as false for acceptance** and are
   also reported separately.

Predeclared criteria (part of the freeze): zero confirmed false activations in at least 40 minutes
of final ordinary use, reported with the exact one-sided 95% Poisson upper bound and per-recording
counts; pooled recall of at least 0.85, at least 0.80 for each gesture, and no more than two cross
activations over at least 90 gesture takes; all for the same frozen setting. Zero events in 40
minutes is still compatible with a true rate up to about 0.075 per minute, and a pass is evidence
from one participant and one camera. It is not proof of zero real-world risk.

This is tamper-evident, not tamper-proof: the code logs and refuses, but a person with access to the
files can still edit them.

### Reviewing candidates

Whether a candidate activation was a false activation is a human judgement. The viewer replays the
**recorded landmark sequence only**: a hand skeleton drawn from numbers, in the same mirrored view
as the preview, with the palm's path as a faint trail. There is no webcam, no video and no desktop
input anywhere in it.

```
python -m nocursor.dynamic stream-review --items ml_runs/streaming/<folder>/review/review_items.json --run primary
```

`--run` also loads the original 120 s stream; use `--recordings` for dev folders. For the
candidates of an acceptance run (items inside an `acceptance` folder) the sealed final recordings
are opened for review only.

What you see: an anonymous candidate id, the gesture the system reported and what that gesture
means, the recording and time range, and your current mark. You never see which system produced the
candidate or how confident it was. The question is "was the person actually performing this
gesture?":

| Key | Meaning |
|---|---|
| `F` | **false**: no gesture was intended |
| `G` | **genuine**: the person did perform it |
| `A` | **ambiguous**: cannot tell (counts as false for acceptance, reported apart) |
| `U` | un-mark: back to unreviewed |
| Left / `P`, Right / `S` | previous, next candidate |
| `N` | next unreviewed candidate |
| Space, `R` | pause, restart the clip |
| `Q`, Esc | quit (everything marked so far is already saved) |

Only those keys mark anything. Moving between candidates, looping, pausing and quitting leave
unreviewed candidates unreviewed, and nothing is ever inferred from a model's prediction. A mark is
saved after every keypress to `review_marks.json` together with the candidate's identity (id,
recording, time range, reported gesture), the time, and a hash of the exact item list, so a mark can
never attach to a different candidate or to a regenerated list; a mismatch is refused, not merged.
Reports count *confirmed false*, *genuine*, *ambiguous* and *unreviewed* candidates separately.

### Offline fallbacks (exploratory)

`stream-compare` scores every variant on the **same** development evidence and protocol:
the GRU, any candidate, `rule_defaults` and `rule_tuned`, each with and without the two mechanisms
below, at a fixed untuned reference setting and across a small declared grid. It selects and
freezes nothing, and its screening threshold (at most 2 cross activations per 45 target takes, raw
count always shown) is for exploration only; the acceptance criteria are unchanged. The grid, the
reference settings and the screening thresholds are hashed in the report.

- **Partial-window handling.** In continuous use most windows hold only part of a movement, which
  a model trained on whole takes never saw: a half circle looks like a swipe. `stream-candidate`
  trains a separately named model (`ml_runs/streaming/candidates/<name>/`) on whole takes plus
  windows shifted so a gesture is cut off at one end (labelled `no_gesture`) and lightly shifted
  full gestures (keeping their label). Only training and validation takes are used. The `primary`
  run, its weights and every historical artifact are left alone; the validation takes also drive
  the candidate's early stopping and rejection threshold, so results on them are not independent
  evidence. Nothing is mined from the 120 s stream: with one development recording it could not
  then be evaluated honestly.
- **Settle gate.** A swipe counts only if the palm has stopped by the end of the window (the swipe
  is defined as "then stop"). Circles are exempt because recorded circles end in motion.
- **Arming rule.** A gesture counts only if an open palm was held still for a short time first,
  which asks the person to prepare. It changes how a gesture must be performed, so it can only be
  judged on recordings made with that instruction. The existing takes mostly start moving almost at
  once, so two replay protocols are reported: `rest_padded` (the take's first and last frame held
  still, which is kind to arming) and `own_lead_only` (no synthetic rest at all).

None of this is adopted automatically, none of it can be frozen, and none of it is connected to the
mouse.

## What is not built

Live use of a trained model is not built, and is not enabled by any of this. A later, separate
step would need your review of the offline results first, and would have to be opt-in, dry-run
first, routed through the existing safety-gated controller, blocked during pause, calibration,
unreliable tracking, an active drag and the emergency stop, and would have to fail closed when the
model is missing, corrupt or incompatible.
