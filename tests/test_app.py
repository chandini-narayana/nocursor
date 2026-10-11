"""Tests of the real-time loop using fakes for camera, tracker, preview and desktop."""

import numpy as np

from helpers import make_hand
from nocursor.app import EXIT_EMERGENCY, EXIT_ERROR, EXIT_OK, NoCursorApp
from nocursor.camera import Camera
from nocursor.config import Config
from nocursor.controller import SafeController
from nocursor.emergency import EmergencyStop
from nocursor.preview import NullPreview
from nocursor.processor import GestureProcessor

FRAME = np.zeros((48, 64, 3), dtype=np.uint8)
OPEN = make_hand(pinch_ratio=1.0)
PINCHED = make_hand(pinch_ratio=0.1)


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = 0

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        if self.frames:
            f = self.frames.pop(0)
            return f is not None, f
        return False, None

    def release(self):
        self.released += 1


class FakeTracker:
    """Returns scripted observations, one per detect() call."""

    def __init__(self, script):
        self.script = list(script)
        self.closed = 0

    def detect(self, frame, timestamp_ms):
        return self.script.pop(0) if self.script else None

    def close(self):
        self.closed += 1


class FakeBackend:
    def __init__(self):
        self.moves, self.clicks = [], 0

    def screen_size(self):
        return (1920, 1080)

    def move_to(self, x, y):
        self.moves.append((x, y))

    def click(self):
        self.clicks += 1


class QuitAfter(NullPreview):
    def __init__(self, frames, stop=None, trigger_stop_at=None):
        self.frames, self.shown, self.closed = frames, 0, 0
        self.stop, self.trigger_stop_at = stop, trigger_stop_at

    def show(self, *args):
        self.shown += 1
        if self.stop is not None and self.shown == self.trigger_stop_at:
            self.stop.trigger("test hotkey")
        return self.shown < self.frames

    def close(self):
        self.closed += 1


def build(script, *, dry_run=False, frames=None, preview_frames=100, trigger_stop_at=None,
          capture=None, **cfg):
    config = Config(**{"smoothing": 0.0, "stable_frames": 1, "pinch_confirm_frames": 1,
                       "click_cooldown_s": 0.0, "active_margin": 0.0, "max_read_failures": 3,
                       **cfg})
    stop, backend = EmergencyStop(), FakeBackend()
    controller = SafeController(backend, stop, enabled=not dry_run)
    capture = capture or FakeCapture(frames if frames is not None else [FRAME] * len(script))
    camera = Camera(capture_factory=lambda _i: capture, max_read_failures=3)
    tracker = FakeTracker(script)
    preview = QuitAfter(preview_frames, stop, trigger_stop_at)
    app = NoCursorApp(camera, tracker, GestureProcessor(config, controller.screen_size()),
                      controller, preview, stop, "ctrl+alt+q")
    return app, backend, capture, tracker, preview, stop


def test_hand_moves_cursor_and_pinch_clicks_once():
    script = [OPEN, OPEN, PINCHED, PINCHED, PINCHED, OPEN]
    app, backend, capture, tracker, preview, _ = build(script)
    assert app.run() == EXIT_ERROR  # scripted frames run out -> camera "stops delivering"
    assert backend.clicks == 1
    assert len(backend.moves) == 3  # OPEN, OPEN, then the final OPEN; pinched frames freeze it


def test_dry_run_sends_no_input():
    script = [OPEN, OPEN, PINCHED, PINCHED, OPEN]
    app, backend, *_ = build(script, dry_run=True)
    app.run()
    assert backend.moves == [] and backend.clicks == 0


def test_no_clicks_or_moves_when_tracking_lost():
    script = [None] * 5
    app, backend, *_ = build(script)
    app.run()
    assert backend.moves == [] and backend.clicks == 0


def test_dropped_frames_cause_no_actions():
    app, backend, *_ = build([], frames=[None, None])
    app.run()
    assert backend.moves == [] and backend.clicks == 0


def test_resources_released_when_camera_dies():
    app, backend, capture, tracker, preview, _ = build([OPEN, OPEN])
    assert app.run() == EXIT_ERROR  # frames run out -> persistent read failure
    assert capture.released == 1 and tracker.closed == 1 and preview.closed == 1


def test_camera_open_failure_returns_error_and_cleans_up():
    class Unopenable(FakeCapture):
        def isOpened(self):
            return False

    app, backend, capture, tracker, preview, _ = build([], capture=Unopenable([]))
    assert app.run() == EXIT_ERROR
    assert tracker.closed == 1 and preview.closed == 1
    assert backend.moves == [] and backend.clicks == 0


def test_preview_quit_stops_loop_and_releases_camera():
    script = [OPEN] * 50
    app, backend, capture, tracker, preview, stop = build(script, preview_frames=3)
    code = app.run()
    assert code == EXIT_EMERGENCY and stop.is_set()
    assert preview.shown == 3
    assert capture.released == 1 and tracker.closed == 1 and preview.closed == 1


def test_emergency_stop_halts_all_later_actions():
    script = [OPEN, OPEN, OPEN, PINCHED, PINCHED]
    app, backend, capture, tracker, preview, stop = build(script, trigger_stop_at=2)
    assert app.run() == EXIT_EMERGENCY
    assert len(backend.moves) == 2  # frames 1-2 ran; the stop fired while showing frame 2
    assert backend.clicks == 0  # the pinched frames were never processed
    assert capture.released == 1


def test_stop_already_set_does_nothing_and_still_cleans_up():
    app, backend, capture, tracker, preview, stop = build([OPEN] * 3)
    stop.trigger("pre-set")
    assert app.run() == EXIT_EMERGENCY
    assert backend.moves == [] and backend.clicks == 0
    assert capture.released == 1


def test_keyboard_interrupt_releases_camera():
    app, backend, capture, tracker, preview, stop = build([OPEN] * 3)

    def interrupt(*_):
        raise KeyboardInterrupt

    tracker.detect = interrupt
    assert app.run() == EXIT_OK
    assert capture.released == 1 and tracker.closed == 1


def test_unexpected_error_stops_safely_and_cleans_up():
    app, backend, capture, tracker, preview, stop = build([OPEN] * 3)

    def explode(*_):
        raise RuntimeError("inference crashed")

    tracker.detect = explode
    assert app.run() == EXIT_ERROR
    assert stop.is_set() and capture.released == 1 and tracker.closed == 1
