"""Loop-level Phase 2 tests with fake camera, tracker, preview, desktop and timing."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from helpers import make_hand, make_pose_hand
from nocursor.app import EXIT_EMERGENCY, EXIT_ERROR, EXIT_OK, NoCursorApp
from nocursor.calibration import CalibrationStore, CalibrationWizard, WizardState
from nocursor.camera import Camera
from nocursor.config import Config
from nocursor.controller import SafeController
from nocursor.emergency import EmergencyStop, PauseControl
from nocursor.interaction import InteractionState, Scroll
from nocursor.preview import NullPreview, PreviewCommand
from nocursor.processor import GestureProcessor
from test_app import FRAME, FakeCapture
from test_lifecycle import pinched, pointing

S = InteractionState


class OrderedBackend:
    """Records every desktop call in order; individual calls can be made to fail."""

    def __init__(self):
        self.events: list[tuple] = []
        self.fail: dict[str, Exception] = {}

    def _do(self, name, *args):
        if name in self.fail:
            raise self.fail[name]
        self.events.append((name, *args))

    def screen_size(self):
        return (1920, 1080)

    def move_to(self, x, y):
        self._do("move", x, y)

    def click(self):
        self._do("click")

    def mouse_down(self):
        self._do("down")

    def mouse_up(self):
        self._do("up")

    def scroll(self, clicks):
        self._do("scroll", clicks)

    def names(self, start=0):
        return [e[0] for e in self.events[start:]]


class HookTracker:
    """Scripted observations; ``hooks[n]`` runs just before the n-th detect() returns."""

    def __init__(self, script, hooks=None):
        self.script = list(script)
        self.hooks = hooks or {}
        self.calls = 0
        self.closed = 0

    def detect(self, frame, timestamp_ms):
        self.calls += 1
        hook = self.hooks.get(self.calls)
        if hook:
            hook()
        return self.script.pop(0) if self.script else None

    def close(self):
        self.closed += 1


class ScriptedPreview(NullPreview):
    """Quits after ``max_frames``; ``commands[n]`` are returned after the n-th show."""

    def __init__(self, max_frames, commands=None):
        self.max_frames = max_frames
        self.commands = commands or {}
        self.shown = 0
        self.closed = 0
        self.statuses = []
        self.gestures = []

    def show(self, frame, observation, gesture, status):
        self.shown += 1
        self.statuses.append(status)
        self.gestures.append(gesture)
        return self.shown < self.max_frames

    def take_commands(self):
        return tuple(self.commands.get(self.shown, ()))

    def close(self):
        self.closed += 1


class FakeClock:
    """Deterministic time: every reading advances 1/30 s, so rate limits behave like 30 fps."""

    def __init__(self, step=1 / 30):
        self.now = 1000.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


class FakeWatcher:
    def __init__(self):
        self.started = self.closed = 0

    def start(self):
        self.started += 1

    def close(self):
        self.closed += 1


@dataclass
class Run:
    app: NoCursorApp
    backend: OrderedBackend
    controller: SafeController
    processor: GestureProcessor
    preview: ScriptedPreview
    capture: FakeCapture
    tracker: HookTracker
    stop: EmergencyStop
    pause: PauseControl
    wizard: CalibrationWizard | None
    store: CalibrationStore | None
    watchers: list[FakeWatcher] = field(default_factory=list)

    def run(self) -> int:
        return self.app.run()


def build(
    script,
    *,
    dry_run=False,
    hooks=None,
    commands=None,
    wizard=None,
    store=None,
    start_in_calibration=False,
    frames=None,
    max_frames=None,
    reporter=None,
    **cfg: Any,
) -> Run:
    config = Config(**{"smoothing": 0.0, "stable_frames": 1, "active_margin": 0.0,
                       "click_cooldown_s": 0.0, "max_read_failures": 3, **cfg})
    stop, backend, pause = EmergencyStop(), OrderedBackend(), PauseControl()
    controller = SafeController(backend, stop, enabled=not dry_run)
    processor = GestureProcessor(config, controller.screen_size())
    capture = FakeCapture(frames if frames is not None else [FRAME] * len(script))
    camera = Camera(capture_factory=lambda _i: capture, max_read_failures=3)
    tracker = HookTracker(script, hooks)
    preview = ScriptedPreview(max_frames or len(script), commands)
    watchers = [FakeWatcher(), FakeWatcher()]
    app = NoCursorApp(
        camera, tracker, processor, controller, preview, stop, "ctrl+alt+q",
        watchers[0], pause=pause, wizard=wizard, store=store, extra_watchers=[watchers[1]],
        clock=FakeClock(), start_in_calibration=start_in_calibration, message_seconds=1000.0,
        diagnostics=reporter,
    )  # fmt: skip
    return Run(app, backend, controller, processor, preview, capture, tracker, stop, pause,
               wizard, store, watchers)


ARM = [pointing()] * 3
PRESS = [pinched()] * 5
RELEASE = [pointing()] * 2


class TestClickAndDragThroughTheController:
    def test_a_pinch_is_one_mouse_down_and_one_mouse_up_not_a_direct_click(self):
        r = build(ARM + PRESS + [pinched()] * 3 + RELEASE + [pointing()] * 3)
        r.run()
        assert r.backend.names().count("down") == 1
        assert r.backend.names().count("up") == 1
        assert "click" not in r.backend.names()
        names = r.backend.names()
        assert names.index("down") < names.index("up")
        between = names[names.index("down") : names.index("up")]
        assert "move" not in between  # pointer frozen: a plain click

    def test_a_drag_moves_the_pointer_between_down_and_up(self):
        script = ARM + PRESS + [pinched(0.5 - 0.01 * i) for i in range(1, 15)] + RELEASE
        r = build(script + [pointing()] * 3)
        r.run()
        names = r.backend.names()
        down, up = names.index("down"), names.index("up")
        assert names[down:up].count("move") >= 8
        assert names.count("down") == 1 and names.count("up") == 1

    def test_a_normal_release_is_not_treated_as_a_safety_release(self):
        r = build(ARM + PRESS + [pinched()] * 3 + RELEASE + [pointing()] * 3)
        r.run()
        kinds = [rec.kind for rec in r.controller.records]
        assert "release" not in kinds  # the safety path is for abnormal endings only
        assert kinds == ["mouse_down", "mouse_up", "click"]
        assert r.controller.refused_button_events == 0

    def test_holding_a_pinch_for_ages_sends_one_down_and_nothing_else(self):
        r = build(ARM + PRESS + [pinched()] * 300, max_frames=308)
        r.run()
        assert r.backend.names().count("down") == 1

    def test_open_hand_for_ages_sends_one_up(self):
        r = build(ARM + PRESS + RELEASE + [pointing()] * 300, max_frames=310)
        r.run()
        assert r.backend.names().count("up") == 1


class TestInterruptionsEndToEnd:
    def test_tracking_loss_during_a_drag_releases_the_button_at_once(self):
        script = ARM + PRESS + [pinched(0.5 - 0.01 * i) for i in range(1, 10)] + [None] * 5
        r = build(script + [pinched(0.3)] * 20)
        r.run()
        names = r.backend.names()
        last_up = max(i for i, n in enumerate(names) if n == "up")
        assert names.count("up") == 1
        assert "move" not in names[last_up + 1 :]  # nothing else after the release
        assert names.count("down") == 1  # and the hand coming back pinched never re-presses

    def test_emergency_stop_during_a_drag_releases_immediately_and_blocks_the_rest(self):
        snapshot = {}
        r_holder = {}

        def hotkey():  # as if the hotkey thread fired in the middle of inference
            r_holder["r"].stop.trigger("hotkey")
            snapshot["names"] = r_holder["r"].backend.names()

        script = ARM + PRESS + [pinched(0.5 - 0.01 * i) for i in range(1, 12)]
        r = build(script + [pointing()] * 5, hooks={14: hotkey})
        r_holder["r"] = r
        code = r.run()
        assert code == EXIT_EMERGENCY
        assert snapshot["names"][-1] == "up"  # released inside trigger(), before the loop resumed
        assert r.backend.names() == snapshot["names"]  # and nothing at all afterwards
        assert not r.controller.button_down

    def test_controller_failure_during_a_drag_stops_everything_and_tries_to_release(self):
        holder = {}

        def break_moves():
            holder["r"].backend.fail["move"] = RuntimeError("input blocked")

        script = ARM + PRESS + [pinched(0.5 - 0.01 * i) for i in range(1, 15)]
        r = build(script + [pointing()] * 3, hooks={12: break_moves})
        holder["r"] = r
        assert r.run() == EXIT_EMERGENCY
        assert r.stop.is_set() and "error" in r.stop.reason
        names = r.backend.names()
        assert names.count("up") == 1 and names[-1] == "up"
        assert not r.controller.button_down
        assert r.controller.fault_message is None  # the release itself succeeded

    def test_failed_release_is_recorded_blocks_input_and_reaches_the_preview(self):
        holder = {}

        def break_up():
            holder["r"].backend.fail["up"] = OSError("input desktop locked")

        script = ARM + PRESS + [pinched()] * 3
        r = build(script, hooks={10: break_up}, max_frames=len(script))
        holder["r"] = r
        assert r.run() == EXIT_EMERGENCY
        assert r.controller.fault_message and "locked" in r.controller.fault_message
        assert "may still be held" in r.controller.fault_message
        assert r.stop.is_set()
        assert r.backend.names().count("down") == 1 and "up" not in r.backend.names()
        # resources still released
        assert r.capture.released == 1 and r.tracker.closed == 1 and r.preview.closed == 1

    def test_failed_release_message_is_shown_in_the_status(self):
        holder = {}

        def break_up():
            holder["r"].backend.fail["up"] = OSError("locked")

        script = ARM + PRESS + [pinched()] * 10 + RELEASE + [pointing()] * 5
        r = build(script, hooks={9: break_up})
        holder["r"] = r
        r.run()
        assert any(s.fault for s in r.preview.statuses)

    def test_press_refused_by_the_backend_is_not_retried_or_left_pending(self):
        r = build(ARM + [pinched()] * 40)
        r.backend.fail["down"] = RuntimeError("denied")
        assert r.run() == EXIT_EMERGENCY
        assert "down" not in r.backend.names()
        assert not r.controller.button_down and not r.processor.holding_button

    def test_a_press_refused_without_a_stop_is_dropped_by_the_state_machine(self):
        holding_after = []
        holder = {}

        def probe():
            holding_after.append(holder["r"].processor.holding_button)

        r = build(ARM + [pinched()] * 5 + [pinched()] * 6 + RELEASE + [pointing()] * 3,
                  hooks={10: probe}, max_frames=1000)
        holder["r"] = r
        r.controller.mouse_down = lambda: False  # refused, but nothing tripped the stop
        r.run()
        assert holding_after == [False]  # the machine does not believe a press exists
        assert r.controller.refused_button_events == 0  # so it never tries to release one
        assert r.backend.events.count(("up",)) == 0

    def test_camera_dying_mid_drag_still_releases(self):
        script = ARM + PRESS + [pinched(0.5 - 0.01 * i) for i in range(1, 8)]
        r = build(script, max_frames=1000)  # frames run out -> camera failure
        assert r.run() == EXIT_ERROR
        assert r.backend.names().count("up") == 1
        assert r.capture.released == 1 and r.tracker.closed == 1 and r.preview.closed == 1

    def test_ctrl_c_mid_drag_releases(self):
        def interrupt():
            raise KeyboardInterrupt

        script = ARM + PRESS + [pinched(0.5 - 0.01 * i) for i in range(1, 10)]
        r = build(script, hooks={16: interrupt}, max_frames=1000)
        assert r.run() == EXIT_OK
        assert r.backend.names().count("up") == 1
        assert r.capture.released == 1

    def test_unexpected_error_mid_drag_releases(self):
        def explode():
            raise RuntimeError("model crashed")

        script = ARM + PRESS + [pinched(0.5 - 0.01 * i) for i in range(1, 10)]
        r = build(script, hooks={16: explode}, max_frames=1000)
        assert r.run() == EXIT_ERROR
        assert r.backend.names().count("up") == 1

    def test_state_machine_and_controller_disagreement_is_corrected(self):
        r = build([pointing()] * 5)
        r.controller.mouse_down()  # something pressed the button behind the machine's back
        r.run()
        names = r.backend.names()
        assert names[:2] == ["down", "up"]  # corrected before the first frame could act on it
        assert names.count("down") == 1 and names.count("up") == 1
        assert not r.controller.button_down

    def test_all_watchers_started_and_closed(self):
        r = build(ARM + [pointing()] * 3)
        r.run()
        assert [(w.started, w.closed) for w in r.watchers] == [(1, 1), (1, 1)]


class TestPause:
    def test_pausing_mid_drag_releases_and_stops_all_input(self):
        holder = {}
        script = ARM + PRESS + [pinched(0.5 - 0.01 * i) for i in range(1, 10)]
        r = build(script + [pinched(0.2)] * 30, hooks={16: lambda: holder["r"].pause.toggle()})
        holder["r"] = r
        r.run()
        names = r.backend.names()
        assert names.count("up") == 1
        assert names[-1] == "up"  # nothing after the release, though the hand keeps moving
        assert r.preview.statuses[-1].control == "PAUSED"
        assert r.preview.gestures[-1].state is S.PAUSED

    def test_status_shows_paused(self):
        r = build([pointing()] * 8)
        r.pause.set_paused(True)
        r.run()
        assert [s.control for s in r.preview.statuses] == ["PAUSED"] * 8
        assert r.backend.events == []

    def test_p_key_toggles_pause_and_resume_needs_a_fresh_open_hand(self):
        # frames 1-3 pointing | 4-9 pinched | 10-12 pointing | 13-18 pinched
        script = ARM + [pinched()] * 6 + [pointing()] * 3 + [pinched()] * 6
        r = build(
            script,
            commands={3: (PreviewCommand.PAUSE_TOGGLE,), 8: (PreviewCommand.PAUSE_TOGGLE,)},
        )
        r.run()
        controls = [s.control for s in r.preview.statuses]
        assert controls == ["LIVE"] * 3 + ["PAUSED"] * 5 + ["LIVE"] * 10  # frames 4-8 paused
        presses = [i for i, g in enumerate(r.preview.gestures) if g.click]
        # The pinch already in progress at resume (frame 9) must not press; only the fresh
        # pinch after an open hand does (5th pinched frame = frame 17).
        assert presses == [16]
        assert r.backend.names().count("down") == 1
        assert r.pause.paused is False

    def test_pause_key_is_ignored_while_calibrating(self):
        wizard = CalibrationWizard()
        r = build([pointing()] * 6, wizard=wizard, start_in_calibration=True,
                  commands={2: (PreviewCommand.PAUSE_TOGGLE,)})
        r.run()
        assert r.pause.paused is False
        assert all(s.control in ("CALIBRATING", "STOPPED") for s in r.preview.statuses)


class TestDryRun:
    def test_a_full_session_sends_nothing_to_the_desktop(self):
        peace = [make_pose_hand(shift=(0.0, -0.01 * i)) for i in range(0, 25)]
        script = (
            ARM + PRESS + [pinched()] * 3 + RELEASE          # click
            + [pointing()] * 4 + PRESS                       # drag
            + [pinched(0.5 - 0.01 * i) for i in range(1, 12)] + RELEASE
            + [make_hand()] * 3 + peace                      # scroll
        )
        r = build(script, dry_run=True)
        r.run()
        assert r.backend.events == []  # no moves, no down/up, no scroll, no click
        kinds = [rec.kind for rec in r.controller.records]
        assert kinds.count("mouse_down") == 2 and kinds.count("mouse_up") == 2
        assert "click" in kinds and "drag" in kinds and "scroll" in kinds
        assert not any(rec.sent for rec in r.controller.records)
        assert r.preview.statuses[0].control == "DRY-RUN"

    def test_dry_run_emergency_stop_clears_the_simulated_button_without_touching_the_desktop(self):
        holder = {}
        r = build(ARM + PRESS + [pinched()] * 10, dry_run=True,
                  hooks={12: lambda: holder["r"].stop.trigger("test")})
        holder["r"] = r
        r.run()
        assert r.backend.events == [] and not r.controller.button_down


class TestScrollEndToEnd:
    def test_two_finger_scroll_reaches_the_controller_with_the_right_sign(self):
        up = [make_pose_hand(shift=(0.0, -0.012 * i)) for i in range(0, 25)]
        r = build([make_hand()] * 3 + up)
        r.run()
        scrolls = [e[1] for e in r.backend.events if e[0] == "scroll"]
        assert scrolls and all(c > 0 for c in scrolls)
        assert "move" not in r.backend.names()[r.backend.names().index("scroll") :]

    def test_scrolling_down(self):
        down = [make_pose_hand(shift=(0.0, 0.012 * i)) for i in range(0, 25)]
        r = build([make_hand()] * 3 + down)
        r.run()
        scrolls = [e[1] for e in r.backend.events if e[0] == "scroll"]
        assert scrolls and all(c < 0 for c in scrolls)

    def test_tracking_loss_ends_scrolling_at_once_and_it_does_not_resume_by_itself(self):
        # 3 setup frames, 14 scrolling frames (hand keeps moving up), 3 lost, 10 pose frames back.
        up = [make_pose_hand(shift=(0.0, -0.012 * i)) for i in range(0, 14)]
        back = [make_pose_hand(shift=(0.0, -0.2 - 0.012 * i)) for i in range(10)]
        r = build([make_hand()] * 3 + up + [None] * 3 + back)
        r.run()
        gestures = r.preview.gestures
        assert any(isinstance(a, Scroll) for g in gestures[:17] for a in g.actions)  # it did scroll
        assert [g.state for g in gestures[17:20]] == [S.SEARCHING] * 3
        assert all(g.actions == () for g in gestures[17:20])
        # The pose is back and the hand is still moving, but scrolling needs the stable period
        # again: the first five frames back produce nothing and are not in scroll mode.
        assert all(g.state is not S.SCROLL and g.actions == () for g in gestures[20:25])

    def test_stop_prevents_further_scrolling(self):
        holder = {}
        snapshot = {}

        def stop_now():
            holder["r"].stop.trigger("test")
            snapshot["events"] = list(holder["r"].backend.events)

        up = [make_pose_hand(shift=(0.0, -0.012 * i)) for i in range(0, 40)]
        r = build([make_hand()] * 3 + up, hooks={22: stop_now})
        holder["r"] = r
        assert r.run() == EXIT_EMERGENCY
        assert any(e[0] == "scroll" for e in snapshot["events"])  # it was scrolling before
        assert r.backend.events == snapshot["events"]  # and not a single event afterwards
        assert r.preview.shown <= 22


class TestCalibrationEndToEnd:
    # Corners as the user sees them (mirrored preview); raw camera x = 1 - view x.
    CORNERS = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.7), (0.2, 0.7)]

    def corner_hand(self, view_xy, pinch=0.5):
        return make_hand(index_xy=(1 - view_xy[0], view_xy[1]), pinch_ratio=pinch)

    def script(self):
        script, commands = [], {}
        for i, corner in enumerate(self.CORNERS):
            ratio = 0.1 if i == 0 else 0.5  # first corner is held while pinching
            script += [self.corner_hand(corner, ratio)] * 5
            commands[len(script)] = (PreviewCommand.CONFIRM,)
        return script, commands

    def test_wizard_blocks_all_desktop_input_even_for_pinches_and_scroll_poses(self, tmp_path):
        wizard = CalibrationWizard(stable_frames=3)
        script = [pointing()] * 3 + [pinched()] * 12 + [make_pose_hand()] * 10 + [pointing()] * 5
        r = build(script, wizard=wizard, start_in_calibration=True)
        r.run()
        assert r.backend.events == []
        assert all(s.control in ("CALIBRATING", "STOPPED") for s in r.preview.statuses)

    def test_four_confirmed_corners_calibrate_save_and_apply(self, tmp_path):
        wizard = CalibrationWizard(stable_frames=3)
        store = CalibrationStore(tmp_path / "calibration.local.json")
        script, commands = self.script()
        # Two lost-tracking frames afterwards let the "saved" notice reach the preview
        # without any hand that could move the pointer.
        r = build(script + [None] * 2, wizard=wizard, store=store, start_in_calibration=True,
                  commands=commands)
        r.run()
        assert r.backend.events == []  # nothing was sent while calibrating
        region = r.processor.region
        assert region is not None
        assert (region.left, region.top, region.right, region.bottom) == pytest.approx(
            (0.2, 0.2, 0.8, 0.7)
        )
        assert store.path.is_file()
        assert store.load(0, 4 / 3).region == region  # what was saved reloads identically
        assert wizard.state is WizardState.IDLE
        assert any("saved" in (s.message or "") for s in r.preview.statuses)

    def test_after_calibration_control_resumes_with_the_new_mapping(self, tmp_path):
        wizard = CalibrationWizard(stable_frames=3)
        store = CalibrationStore(tmp_path / "c.json")
        script, commands = self.script()
        tail = [self.corner_hand((0.2, 0.2), 1.0)] * 3 + [self.corner_hand((0.2, 0.2), 0.1)] * 5
        r = build(script + tail + [pointing()] * 3, wizard=wizard, store=store,
                  start_in_calibration=True, commands=commands)
        r.run()
        moves = [e for e in r.backend.events if e[0] == "move"]
        assert moves, "pointer should resume after calibration"
        assert moves[0][1] <= 3 and moves[0][2] <= 3  # calibrated top-left -> screen top-left
        assert r.backend.names().count("down") == 1  # a deliberate pinch works again

    def test_cancel_changes_nothing_and_leaves_no_file(self, tmp_path):
        wizard = CalibrationWizard(stable_frames=3)
        store = CalibrationStore(tmp_path / "c.json")
        script = [self.corner_hand(self.CORNERS[0])] * 6 + [pointing()] * 6
        r = build(script, wizard=wizard, store=store, start_in_calibration=True,
                  commands={4: (PreviewCommand.CONFIRM,), 6: (PreviewCommand.CANCEL,)})
        r.run()
        assert r.processor.region is None
        assert not store.path.exists()
        assert wizard.state is WizardState.IDLE
        controls = [s.control for s in r.preview.statuses]
        assert controls[0] == "CALIBRATING" and "LIVE" in controls
        assert any("cancelled" in (s.message or "") for s in r.preview.statuses)

    def test_failed_calibration_reports_and_keeps_the_default_mapping(self, tmp_path):
        wizard = CalibrationWizard(stable_frames=3)
        store = CalibrationStore(tmp_path / "c.json")
        wrong_order = [self.CORNERS[1], self.CORNERS[0], self.CORNERS[3], self.CORNERS[2]]
        script, commands = [], {}
        for corner in wrong_order:
            script += [self.corner_hand(corner)] * 5
            commands[len(script)] = (PreviewCommand.CONFIRM,)
        r = build(script + [pointing()] * 4, wizard=wizard, store=store,
                  start_in_calibration=True, commands=commands)
        r.run()
        assert r.processor.region is None and not store.path.exists()
        assert any("rejected" in (s.message or "") for s in r.preview.statuses)

    def test_c_key_starts_calibration_during_normal_use(self, tmp_path):
        wizard = CalibrationWizard(stable_frames=3)
        r = build([pointing()] * 10, wizard=wizard,
                  commands={4: (PreviewCommand.CALIBRATE,)})
        r.run()
        controls = [s.control for s in r.preview.statuses]
        assert controls == ["LIVE"] * 4 + ["CALIBRATING"] * 6

    def test_starting_calibration_mid_drag_releases_the_button_first(self, tmp_path):
        wizard = CalibrationWizard(stable_frames=3)
        script = ARM + PRESS + [pinched(0.5 - 0.01 * i) for i in range(1, 8)] + [pinched(0.3)] * 10
        r = build(script, wizard=wizard, commands={14: (PreviewCommand.CALIBRATE,)})
        r.run()
        names = r.backend.names()
        assert names.count("down") == 1 and names.count("up") == 1
        assert names[-1] == "up"

    def test_reset_key_restores_the_default_mapping_and_deletes_the_file(self, tmp_path):
        wizard = CalibrationWizard(stable_frames=3)
        store = CalibrationStore(tmp_path / "c.json")
        script, commands = self.script()
        commands[len(script) + 3] = (PreviewCommand.RESET_CALIBRATION,)
        r = build(script + [pointing()] * 6, wizard=wizard, store=store,
                  start_in_calibration=True, commands=commands)
        r.run()
        assert r.processor.region is None and not store.path.exists()

    def test_unsaveable_calibration_is_applied_for_the_session_with_a_warning(self, tmp_path):
        wizard = CalibrationWizard(stable_frames=3)
        blocker = tmp_path / "blocked"
        blocker.write_text("a file, so a directory cannot be created below it")
        store = CalibrationStore(blocker / "c.json")
        script, commands = self.script()
        r = build(script + [pointing()] * 3, wizard=wizard, store=store,
                  start_in_calibration=True, commands=commands)
        r.run()
        assert r.processor.region is not None
        assert any("NOT saved" in (s.message or "") for s in r.preview.statuses)

    def test_commands_are_ignored_when_calibration_is_unavailable(self):
        commands = {2: (PreviewCommand.CALIBRATE, PreviewCommand.CONFIRM)}
        r = build([pointing()] * 5, commands=commands)  # no wizard injected
        assert r.run() == EXIT_EMERGENCY  # ends via the scripted preview quit
        assert r.backend.names().count("move") == 5  # normal control carried on throughout
        assert {s.control for s in r.preview.statuses} == {"LIVE"}


def test_dropped_frames_are_treated_as_tracking_lost_and_take_no_action():
    r = build([pointing()] * 3, frames=[FRAME, None, FRAME, None, FRAME], max_frames=5)
    r.run()
    states = [g.state for g in r.preview.gestures]
    assert states[1] is S.SEARCHING and states[3] is S.SEARCHING
    assert r.backend.names().count("move") == 3  # only the three real frames moved the pointer


def test_phase1_exit_codes_are_unchanged():
    assert (EXIT_OK, EXIT_ERROR, EXIT_EMERGENCY) == (0, 1, 2)
