"""Phase 3: HUD wording, preview keys, profiler wiring, time limit, toasts, logging."""

import logging

import numpy as np
import pytest

from helpers import make_hand, make_pose_hand
from nocursor.app import EXIT_EMERGENCY, EXIT_OK
from nocursor.calibration import CalibrationWizard
from nocursor.dispatch import apply_actions
from nocursor.emergency import EmergencyStop, PauseControl
from nocursor.gestures import PinchState
from nocursor.hud import (
    CONTROL_TEXT,
    StatusInfo,
    Tone,
    build_hud,
    help_lines,
    toast_text,
)
from nocursor.interaction import InteractionState, MouseDown, MouseUp, MoveTo, Scroll
from nocursor.preview import OpenCVPreview, PreviewCommand
from nocursor.processor import GestureFrame
from nocursor.profiling import FrameProfiler
from nocursor.resources import MemoryInfo, ResourceSampler
from nocursor.tracking import TrackingState
from test_app import FRAME
from test_app_phase2 import ARM, PRESS, RELEASE, ScriptedPreview, build
from test_lifecycle import pinched, pointing
from test_preview_phase2 import WindowlessCV2

S = InteractionState
T = TrackingState
MS = 1_000_000


def frame(state=S.POINTER, tracking=T.TRACKING, **kw):
    return GestureFrame(tracking=tracking, state=state, **kw)


def status(control="LIVE", **kw):
    return StatusInfo(control, 30.0, "ctrl+alt+q", pause_hotkey="ctrl+alt+p", **kw)


def headline(gesture, st=None):
    return build_hud(gesture, st or status()).headline


class TestHeadline:
    def test_tracking_problems_are_explained_in_plain_words(self):
        assert "NO HAND" in headline(frame(S.SEARCHING, T.LOST)).text
        h = headline(frame(S.SEARCHING, T.UNCERTAIN, tracking_note="confidence 0.20 < 0.60"))
        assert h.text == "HAND NOT USABLE - confidence 0.20 < 0.60" and h.tone is Tone.WARN
        locking = headline(frame(S.SEARCHING, T.ACQUIRING, tracking_note="1/3"))
        assert "LOCKING ON (1/3)" in locking.text

    def test_unreliable_tracking_overrides_a_stale_gesture_state(self):
        for state in (S.PRESSED, S.DRAGGING, S.SCROLL, S.POINTER):
            text = headline(frame(state, T.LOST)).text
            assert "NO HAND" in text and "PRESSED" not in text and "DRAGGING" not in text

    def test_pointing(self):
        h = headline(frame(S.POINTER))
        assert h.text == "POINTING" and h.tone is Tone.OK

    def test_a_pending_pinch_shows_progress_and_says_it_is_not_a_click(self):
        h = headline(frame(S.PINCH_PENDING, pinch=PinchState.CLOSING, pinch_progress=(3, 5)))
        assert h.text == "PINCH PENDING 3/5 - not clicked yet" and h.tone is Tone.PENDING

    def test_candidates_are_never_described_as_actions(self):
        candidates = [
            frame(S.PINCH_PENDING, pinch=PinchState.CLOSING, pinch_progress=(4, 5)),
            frame(S.PINCH_PENDING, pinch=PinchState.CLOSING),
            frame(S.SCROLL_PENDING, scroll_pose=True),
        ]
        for g in candidates:
            text = headline(g).text.upper()
            label = text.split(" - ")[0]  # the state name, before the explanation
            assert not label.startswith(("PRESSED", "DRAGGING", "SCROLLING", "CLICK"))
            assert "NOT CLICKED YET" in text or "HOLD STILL" in text  # says nothing happened yet

    def test_refused_pinch_is_called_ignored_not_pending(self):
        g = frame(S.PINCH_PENDING, pinch=PinchState.PINCHED)
        assert "PINCH IGNORED" in headline(g).text

    @pytest.mark.parametrize("state,word", [(S.PRESSED, "PRESSED"), (S.DRAGGING, "DRAGGING")])
    def test_pressed_and_dragging_need_a_button_actually_held(self, state, word):
        held = headline(frame(state), status("LIVE", button_held=True))
        assert held.text == f"{word} - button held" and held.tone is Tone.ACTIVE
        blocked = headline(frame(state), status("LIVE", button_held=False))
        assert "PRESS NOT SENT" in blocked.text and word not in blocked.text
        assert blocked.tone is Tone.ERROR

    @pytest.mark.parametrize("state", [S.PRESSED, S.DRAGGING])
    def test_dry_run_labels_the_press_as_simulated(self, state):
        text = headline(frame(state), status("DRY-RUN", button_held=True)).text
        assert text.endswith("(simulated)")

    def test_scroll_states(self):
        assert "SCROLL POSE - hold still" in headline(frame(S.SCROLL_PENDING)).text
        h = headline(frame(S.SCROLL))
        assert "SCROLLING" in h.text and h.tone is Tone.SCROLL

    def test_not_armed(self):
        for state in (S.REARMING, S.SEARCHING):
            g = frame(state, pinch=PinchState.REOPEN)
            assert headline(g).text == "NOT ARMED - spread thumb and index apart"

    def test_control_modes_override_gesture_state(self):
        lost = frame(S.PRESSED, T.LOST)
        assert "PAUSED" in headline(lost, status("PAUSED")).text
        assert "STOPPED - emergency hotkey pressed" in headline(
            lost, status("STOPPED", stop_reason="emergency hotkey pressed")
        ).text
        wizard = CalibrationWizard()
        wizard.start()
        text = headline(lost, status("CALIBRATING", calibration=wizard.view())).text
        assert text == "CALIBRATING - corner 1/4"

    def test_stopped_without_a_reason(self):
        assert headline(frame(), status("STOPPED")).text == "STOPPED"

    def test_every_state_and_tracking_combination_has_a_headline(self):
        for state in InteractionState:
            for tracking in TrackingState:
                for control in CONTROL_TEXT:
                    assert build_hud(frame(state, tracking), status(control)).headline.text


class TestDetailsNoticesAndHints:
    def test_control_line_says_whether_input_is_sent(self):
        assert "mouse input is sent" in build_hud(frame(), status("LIVE")).details[0].text
        assert "no input is sent" in build_hud(frame(), status("DRY-RUN")).details[0].text
        assert "no input is sent" in build_hud(frame(), status("PAUSED")).details[0].text

    def test_tracking_line(self):
        assert build_hud(frame(), status()).details[1].text == "Tracking: stable"
        assert "no hand" in build_hud(frame(S.SEARCHING, T.LOST), status()).details[1].text
        text = build_hud(frame(S.SEARCHING, T.UNCERTAIN, tracking_note="hand too small"),
                         status()).details[1].text
        assert "unreliable" in text and "hand too small" in text

    def test_pinch_line_names_the_active_mode_ratio_and_threshold(self):
        st = status(pinch_mode="2D pinch ratio (default)", pinch_threshold=0.2)
        line = build_hud(frame(pinch_ratio=0.243), st).details[2].text
        assert line == "2D pinch ratio (default): 0.24  (pinch at <= 0.20)"
        st3 = status(pinch_mode="3D pinch ratio (experimental)", pinch_threshold=0.2)
        line3 = build_hud(frame(pinch_ratio=0.1), st3).details[2].text
        assert "3D pinch ratio (experimental)" in line3

    def test_pinch_line_without_a_hand(self):
        lost = build_hud(frame(S.SEARCHING, T.LOST), status(pinch_mode="m"))
        assert lost.details[2].text.startswith("m: -")

    def test_fps_line_is_replaced_by_the_perf_line_when_given(self):
        assert build_hud(frame(), status()).details[3].text == "FPS 30"
        perf = "PERF 29 fps | inference p50 9.0 ms"
        assert build_hud(frame(), status(perf_line=perf)).details[3].text == perf

    def test_notices_order_and_absence(self):
        assert build_hud(frame(), status()).notices == ()
        st = status(toast="Button pressed", fault="held", message="Saved")
        n = build_hud(frame(), st).notices
        assert [x.text for x in n] == ["Button pressed", "FAULT: held", "Saved"]
        assert n[1].tone is Tone.ERROR

    def test_hint_lists_the_essential_controls(self):
        hint = build_hud(frame(), status()).hint
        for needle in ("Esc/Q", "ctrl+alt+q", "P / ctrl+alt+p", "C", "H"):
            assert needle in hint

    def test_toast_wording(self):
        assert toast_text("press", False) == "Button pressed"
        assert toast_text("press", True) == "Button pressed (simulated)"
        assert toast_text("click", False) == "Released: click"
        assert toast_text("drag_end", True) == "Released: drag finished (simulated)"
        assert toast_text("scroll +3", False) == "Scrolled +3"

    def test_help_panel_lists_keys_and_gestures(self):
        text = "\n".join(help_lines("ctrl+alt+q", "ctrl+alt+p"))
        for needle in ("Esc / Q", "ctrl+alt+q", "ctrl+alt+p", "calibrate", "Space", "Backspace",
                       "reset calibration", "click", "drag", "scroll"):
            assert needle in text


def preview_with(keys):
    p = OpenCVPreview(0.15)
    p._cv2 = WindowlessCV2(keys)
    return p


class TestPreviewKeys:
    def test_h_toggles_help_and_is_not_a_command(self):
        p = preview_with([ord("h"), -1, ord("H")])
        drawn = []
        p._draw_help = lambda view, st: drawn.append(1)  # type: ignore[method-assign]
        args = (np.zeros((240, 320, 3), np.uint8), None, frame(), status())
        p.show(*args)  # 'h' pressed during this frame: help turns on
        assert drawn == []
        p.show(*args)
        assert drawn == [1]  # help drawn on the next frame
        p.show(*args)  # 'H' pressed: help turns off again
        p.show(*args)
        assert drawn == [1, 1][:1] + [1] or len(drawn) == 2
        assert p.take_commands() == ()

    def test_question_mark_also_toggles_help(self):
        p = preview_with([ord("?"), -1])
        drawn = []
        p._draw_help = lambda view, st: drawn.append(1)  # type: ignore[method-assign]
        args = (np.zeros((240, 320, 3), np.uint8), None, frame(), status())
        p.show(*args)
        p.show(*args)
        assert drawn == [1]

    @pytest.mark.parametrize("key", [27, ord("q"), ord("Q")])
    def test_quit_keys_work_while_help_is_open(self, key):
        p = preview_with([ord("h"), key])
        args = (np.zeros((240, 320, 3), np.uint8), None, frame(), status())
        assert p.show(*args) is True
        assert p.show(*args) is False

    def test_existing_command_keys_are_unchanged(self):
        p = preview_with([ord("p"), ord("c"), 32, 8, ord("r")])
        args = (np.zeros((240, 320, 3), np.uint8), None, frame(), status())
        for _ in range(5):
            p.show(*args)
        assert p.take_commands() == (
            PreviewCommand.PAUSE_TOGGLE, PreviewCommand.CALIBRATE, PreviewCommand.CONFIRM,
            PreviewCommand.CANCEL, PreviewCommand.RESET_CALIBRATION,
        )

    def test_help_panel_actually_draws(self):
        p = preview_with([-1])
        plain, helped = np.zeros((300, 640, 3), np.uint8), np.zeros((300, 640, 3), np.uint8)
        p._draw_status(plain, frame(), status())
        p._draw_status(helped, frame(), status())
        p._draw_help(helped, status())
        assert (plain != helped).any()

    def test_status_drawing_has_a_backing_panel_for_readability(self):
        p = preview_with([-1])
        bright = np.full((300, 640, 3), 255, np.uint8)
        p._draw_status(bright, frame(), status())
        assert bright[8:12, 8:12].mean() < 200  # the top-left block was darkened

    def test_fingertip_rings_distinguish_candidate_from_confirmed(self):
        class RecordingCV2(WindowlessCV2):
            def __init__(self):
                super().__init__([-1])
                self.rings = []

            def circle(self, img, center, radius, color, thickness):
                if radius == 9:
                    self.rings.append(color)

        cases = {
            "idle": (frame(), (255, 200, 0)),
            "candidate": (frame(S.PINCH_PENDING, pinch=PinchState.CLOSING, pinch_progress=(2, 5)),
                          (0, 165, 255)),
            "confirmed": (frame(S.PRESSED, pinch=PinchState.PINCHED), (60, 60, 255)),
        }
        for gesture, color in cases.values():
            p = OpenCVPreview(0.15)
            p._cv2 = RecordingCV2()
            p._draw_hand(np.zeros((240, 320, 3), np.uint8), make_hand(), gesture)
            assert set(p._cv2.rings) == {color}


class TickNs:
    """A clock that advances ``step_ms`` every time it is read, so durations are exact."""

    def __init__(self, step_ms=1):
        self.t = 0
        self.step = int(step_ms * MS)

    def __call__(self):
        self.t += self.step
        return self.t

    def jump_ms(self, ms):
        self.t += int(ms * MS)


class SlowPreview(ScriptedPreview):
    def __init__(self, *args, ticks, cost_ms, **kwargs):
        super().__init__(*args, **kwargs)
        self.ticks, self.cost_ms = ticks, cost_ms

    def show(self, *args):
        self.ticks.jump_ms(self.cost_ms)
        return super().show(*args)


class TestProfilerInTheLoop:
    def run_profiled(self, script, **kw):
        ticks = TickNs(1)
        profiler = FrameProfiler(clock_ns=ticks, keep_session=True)
        r = build(script, app_kwargs={"profiler": profiler}, **kw)
        r.run()
        return r, profiler, ticks

    def test_every_stage_is_recorded_once_per_frame(self):
        r, p, _ = self.run_profiled([pointing()] * 20)
        for name in ("frame_wait", "inference", "gesture", "dispatch", "preview", "end_to_end"):
            assert p.session_summary(name).n == 20, name
        assert p.frames == 20 and p.dropped_frames == 0

    def test_stage_boundaries_in_the_real_loop(self):
        _, p, _ = self.run_profiled([pointing()] * 5)
        # With a clock that ticks 1 ms per read: each stage spans one tick pair ...
        for name in ("inference", "gesture", "dispatch"):
            assert p.session_summary(name).mean == pytest.approx(1.0), name
        # ... and end-to-end spans arrival -> end of dispatch (7 reads), nothing before or after.
        assert p.session_summary("end_to_end").mean == pytest.approx(7.0)

    def test_preview_cost_never_enters_end_to_end_latency(self):
        ticks = TickNs(1)
        profiler = FrameProfiler(clock_ns=ticks, keep_session=True)
        script = [pointing()] * 6
        r = build(script, app_kwargs={"profiler": profiler})
        r.app._preview = SlowPreview(len(script), ticks=ticks, cost_ms=500)
        r.run()
        assert profiler.session_summary("preview").mean > 500
        assert profiler.session_summary("end_to_end").mean == pytest.approx(7.0)

    def test_dropped_frames_have_no_inference_and_count_as_lost_tracking(self):
        r, p, _ = self.run_profiled(
            [pointing()] * 3, frames=[FRAME, None, FRAME, None, FRAME], max_frames=5
        )
        assert p.frames == 5 and p.dropped_frames == 2
        assert p.session_summary("inference").n == 3  # nothing was inferred on the dropped ones
        # stable_frames=1 here, so each drop interrupts stable tracking: two loss events
        assert p.tracking.frames_by_state == {"tracking": 3, "lost": 2}
        assert p.tracking.loss_events == 2

    def test_tracking_loss_is_counted(self):
        script = [pointing()] * 6 + [None] * 3 + [pointing()] * 6 + [None] * 2
        r, p, _ = self.run_profiled(script, stable_frames=1)
        assert p.tracking.loss_events == 2
        assert p.tracking.frames_by_state["lost"] == 5

    def test_action_frames_are_the_discrete_ones(self):
        script = ARM + PRESS + [pinched()] * 3 + RELEASE + [pointing()] * 3
        r, p, _ = self.run_profiled(script)
        # one press frame and one release frame; the many pointer-move frames are excluded
        assert p.session_summary("end_to_end_action").n == 2
        assert p.session_summary("end_to_end").n == len(script)

    def test_inference_is_tagged_by_hand_presence(self):
        script = [pointing()] * 4 + [None] * 3
        r, p, _ = self.run_profiled(script)
        assert p.session_summary("inference_hand").n == 4
        assert p.session_summary("inference_no_hand").n == 3

    def test_profiling_changes_no_behaviour(self):
        script = ARM + PRESS + [pinched(0.5 - 0.01 * i) for i in range(1, 12)] + RELEASE
        plain = build(list(script))
        plain.run()
        timed = build(list(script), app_kwargs={"profiler": FrameProfiler(clock_ns=TickNs())})
        timed.run()
        assert plain.backend.events == timed.backend.events and plain.backend.events

    def test_default_loop_uses_the_null_profiler(self):
        r = build([pointing()] * 3)
        assert r.app._profiler.enabled is False
        r.run()


class TestPerfOverlayAndResources:
    def test_perf_line_appears_after_data_and_is_labelled_as_software(self):
        profiler = FrameProfiler(clock_ns=TickNs())
        r = build([pointing()] * 40, app_kwargs={"profiler": profiler, "show_perf": True})
        r.run()
        lines = [s.perf_line for s in r.preview.statuses if s.perf_line]
        assert lines and "software e2e" in lines[-1] and "fps" in lines[-1]

    def test_no_perf_line_unless_requested(self):
        profiler = FrameProfiler(clock_ns=TickNs())
        r = build([pointing()] * 40, app_kwargs={"profiler": profiler})
        r.run()
        assert all(s.perf_line is None for s in r.preview.statuses)

    def test_resources_are_sampled_during_the_run(self):
        wall = {"t": 0.0}

        def advance():
            wall["t"] += 0.5
            return wall["t"]

        sampler = ResourceSampler(wall_clock=advance, cpu_clock=lambda: wall["t"] * 0.25,
                                  memory=lambda: MemoryInfo(5, 9), logical_cpus=4)
        r = build([pointing()] * 20, app_kwargs={"resources": sampler})
        r.run()
        summary = sampler.summary()
        assert summary.samples >= 2 and summary.cpu_percent_of_one_core == pytest.approx(25.0)
        assert summary.peak_rss_bytes == 9


class TestTimeLimit:
    def test_loop_ends_cleanly_when_the_time_limit_is_reached(self):
        r = build([pointing()] * 400, max_frames=10_000, app_kwargs={"time_limit_s": 2.0})
        code = r.run()
        assert code == EXIT_OK and not r.stop.is_set()
        assert 0 < r.preview.shown < 400
        assert r.capture.released == 1 and r.tracker.closed == 1 and r.preview.closed == 1
        assert r.elapsed_s < 4.0  # stopped soon after the limit, not at the end of the script

    def test_a_held_press_is_released_when_the_limit_ends_the_loop(self):
        script = ARM + PRESS + [pinched()] * 400
        r = build(script, max_frames=10_000, app_kwargs={"time_limit_s": 1.0})
        r.run()
        assert r.backend.names().count("down") == r.backend.names().count("up")
        assert not r.controller.button_down

    def test_no_limit_by_default(self):
        r = build([pointing()] * 30)
        assert r.run() == EXIT_EMERGENCY  # ended by the scripted preview quit, not a limit


class TestToastsAndPressConfirmation:
    def toasts(self, r):
        return [s.toast for s in r.preview.statuses if s.toast]

    def test_dry_run_toasts_are_marked_simulated(self):
        r = build(ARM + PRESS + [pinched()] * 3 + RELEASE + [pointing()] * 4, dry_run=True)
        r.run()
        toasts = self.toasts(r)
        assert "Button pressed (simulated)" in toasts and "Released: click (simulated)" in toasts
        assert r.backend.events == []

    def test_live_toasts_are_not_marked_simulated(self):
        r = build(ARM + PRESS + [pinched()] * 3 + RELEASE + [pointing()] * 4)
        r.run()
        toasts = self.toasts(r)
        assert "Button pressed" in toasts and "Released: click" in toasts
        assert not any("simulated" in t for t in toasts)

    def test_drag_and_scroll_toasts(self):
        drag = [pinched(0.5 - 0.01 * i) for i in range(1, 12)]
        peace = [make_pose_hand(shift=(0, -0.012 * i)) for i in range(25)]
        r = build(ARM + PRESS + drag + RELEASE + [make_hand()] * 3 + peace + [make_hand()] * 3,
                  dry_run=True)
        r.run()
        toasts = self.toasts(r)
        assert "Released: drag finished (simulated)" in toasts
        assert any(t.startswith("Scrolled +") for t in toasts)

    def test_no_press_toast_when_the_controller_refused_the_press(self):
        r = build(ARM + PRESS + [pinched()] * 3)
        r.controller.mouse_down = lambda: False  # refused; button stays up
        r.run()
        assert not any("pressed" in t for t in self.toasts(r))

    def test_hud_says_press_not_sent_when_the_gesture_state_outran_the_controller(self):
        r = build(ARM + PRESS + [pinched()] * 3)
        r.controller.mouse_down = lambda: False
        r.run()
        pairs = [(g, s) for g, s in zip(r.preview.gestures, r.preview.statuses, strict=True)
                 if g.state is S.PRESSED]
        assert pairs and all(not s.button_held for _, s in pairs)
        assert all("PRESS NOT SENT" in build_hud(g, s).headline.text for g, s in pairs)

    def test_status_carries_mode_threshold_and_button_state(self):
        r = build(ARM + PRESS + [pinched()] * 2, dry_run=True)
        r.run()
        last = r.preview.statuses[-1]
        assert last.pinch_mode == "2D pinch ratio (default)"
        assert last.pinch_threshold == pytest.approx(0.20)
        assert last.button_held is True  # simulated press in dry-run
        assert r.preview.statuses[0].button_held is False

    def test_experimental_mode_is_reported_in_the_status(self):
        r = build(ARM, pinch_use_depth=True)
        r.run()
        assert r.preview.statuses[-1].pinch_mode == "3D pinch ratio (experimental)"

    def test_hud_walks_pending_then_pressed_then_released_for_a_dry_run_click(self):
        r = build(ARM + PRESS + [pinched()] * 2 + RELEASE + [pointing()] * 2, dry_run=True)
        r.run()
        headlines = []
        for g, s in zip(r.preview.gestures, r.preview.statuses, strict=True):
            text = build_hud(g, s).headline.text
            if not headlines or headlines[-1] != text:
                headlines.append(text)
        assert headlines[0] == "POINTING"
        assert "PINCH PENDING 1/5 - not clicked yet" in headlines
        assert "PRESSED - button held (simulated)" in headlines
        assert headlines.index("PRESSED - button held (simulated)") > headlines.index(
            "PINCH PENDING 4/5 - not clicked yet"
        )
        assert headlines[-1] == "POINTING"


class TestDispatch:
    def run_actions(self, actions, *, enabled=True, stop_first=False):
        r = build([pointing()] * 3, dry_run=not enabled)
        if stop_first:
            r.stop.trigger("test")
        events = apply_actions(r.controller, r.processor, actions)
        return r, events

    def test_events_are_reported_for_accepted_actions(self):
        r, events = self.run_actions([MoveTo(1, 1), MouseDown(), MouseUp(drag=False)])
        assert events == ("press", "click")
        r2, events2 = self.run_actions([MouseDown(), MouseUp(drag=True), Scroll(3), Scroll(-2)])
        assert events2 == ("press", "drag_end", "scroll +3", "scroll -2")

    def test_dry_run_events_are_reported_but_nothing_is_sent(self):
        r, events = self.run_actions([MouseDown(), MouseUp(drag=False)], enabled=False)
        assert events == ("press", "click") and r.backend.events == []

    def test_blocked_actions_report_nothing(self):
        r, events = self.run_actions([MouseDown(), Scroll(2), MoveTo(5, 5)], stop_first=True)
        assert events == () and r.backend.events == []

    def test_a_refused_press_is_aborted_not_reported(self):
        r = build([pointing()] * 3)
        r.controller.inhibited = True
        events = apply_actions(r.controller, r.processor, [MouseDown()])
        assert events == () and not r.controller.button_down

    def test_a_duplicate_press_is_refused_and_not_reported_as_a_new_press(self):
        r = build([pointing()] * 3)
        assert r.controller.mouse_down() is True  # a press is already outstanding
        events = apply_actions(r.controller, r.processor, [MouseDown()])
        assert events == ()  # refused by the controller: nothing new happened
        assert r.controller.button_down  # and the original press is untouched
        assert r.backend.names().count("down") == 1

    def test_release_without_press_reports_nothing(self):
        r, events = self.run_actions([MouseUp(drag=False)])
        assert events == ()


class TestOperationalLogging:
    def test_start_and_end_are_logged_for_the_log_file(self, caplog):
        r = build([pointing()] * 3, dry_run=True)
        with caplog.at_level(logging.INFO, logger="nocursor.app"):
            r.run()
        starts = [x for x in caplog.records if "control loop starting" in x.getMessage()]
        ends = [x for x in caplog.records if "control loop ended" in x.getMessage()]
        assert len(starts) == len(ends) == 1
        assert "DRY-RUN" in starts[0].getMessage()
        assert "2D pinch ratio (default)" in starts[0].getMessage()
        assert starts[0].file_only is True  # not repeated on the console
        assert "exit code 2" in ends[0].getMessage()

    def test_pause_and_resume_are_logged_whoever_triggers_them(self, caplog):
        pause = PauseControl()
        with caplog.at_level(logging.INFO, logger="nocursor.emergency"):
            pause.toggle()
            pause.toggle()
            pause.set_paused(True)
            pause.set_paused(True)  # unchanged: not logged again
        messages = [x.getMessage() for x in caplog.records]
        assert messages == ["paused: no input will be sent", "resumed",
                            "paused: no input will be sent"]

    def test_emergency_stop_is_logged_with_its_reason(self, caplog):
        with caplog.at_level(logging.WARNING, logger="nocursor.emergency"):
            EmergencyStop().trigger("hotkey pressed")
        assert any("EMERGENCY STOP: hotkey pressed" in x.getMessage() for x in caplog.records)

    def test_controller_failure_is_logged_with_a_traceback_and_stops(self, caplog):
        script = ARM + PRESS + [pinched(0.5 - 0.01 * i) for i in range(1, 12)]
        r = build(script, hooks={12: lambda: r.backend.fail.update(move=RuntimeError("blocked"))})
        with caplog.at_level(logging.INFO):
            r.run()
        errors = [x for x in caplog.records if x.levelno >= logging.ERROR]
        assert any("desktop input failed" in x.getMessage() and x.exc_info for x in errors)
        assert r.stop.is_set() and "desktop input error" in r.stop.reason

    def test_failed_release_is_logged_as_an_error_naming_the_cause(self, caplog):
        script = ARM + PRESS + [pinched()] * 3
        r = build(script, hooks={10: lambda: r.backend.fail.update(up=OSError("desktop locked"))})
        with caplog.at_level(logging.INFO):
            r.run()
        text = "\n".join(x.getMessage() for x in caplog.records if x.levelno >= logging.ERROR)
        assert "Could not release the left mouse button" in text and "desktop locked" in text
        assert "may still be held" in text

    def test_calibration_start_and_save_are_logged(self, caplog, tmp_path):
        from nocursor.calibration import CalibrationStore

        corners = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.7), (0.2, 0.7)]
        script, commands = [], {}
        for c in corners:
            script += [make_hand(index_xy=(1 - c[0], c[1]), pinch_ratio=0.5)] * 5
            commands[len(script)] = (PreviewCommand.CONFIRM,)
        r = build(script + [None] * 2, wizard=CalibrationWizard(stable_frames=3),
                  store=CalibrationStore(tmp_path / "c.json"), start_in_calibration=True,
                  commands=commands)
        with caplog.at_level(logging.INFO, logger="nocursor.app"):
            r.run()
        messages = " | ".join(x.getMessage() for x in caplog.records)
        assert "calibration started" in messages and "Calibration saved" in messages
