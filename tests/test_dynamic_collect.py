"""The collection tool: protocol, session logic, HUD wording and a headless run of the loop.

Everything is driven by fake observations and a fake clock. No camera is opened, no window is
shown, and the "hands" are SYNTHETIC; nothing here measures real-world behaviour.
"""

from collections import Counter

import numpy as np
import pytest

from dynamic_helpers import hand_at, swipe_path
from nocursor.dynamic.collect import (
    EXIT_ERROR,
    EXIT_OK,
    EXIT_USAGE,
    INSTRUCTIONS,
    CollectCommand,
    CollectionSession,
    CollectOptions,
    PlanItem,
    build_protocol,
    new_session_id,
    run_collect,
)
from nocursor.dynamic.dataset import SessionWriter, load_dataset
from nocursor.dynamic.recorder import RecorderState, TakeRecorder
from nocursor.dynamic.schema import (
    CLASS_NAMES,
    DEFAULT_WINDOW_S,
    ID_PATTERN,
    TAKE_NEGATIVE_SUBTYPES,
    TARGET_CLASSES,
    SessionHeader,
)
from nocursor.dynamic.view import draw_overlay, wrap_text
from nocursor.hud import Tone

FPS = 30.0


class TestProtocol:
    def test_deterministic_for_a_seed_and_different_for_another(self):
        assert build_protocol(5, 10, seed=1) == build_protocol(5, 10, seed=1)
        assert build_protocol(5, 10, seed=1) != build_protocol(5, 10, seed=2)

    def test_counts_are_exactly_what_was_asked(self):
        plan = build_protocol(15, 30, seed=7)
        counts = Counter(item.label for item in plan)
        assert counts == {"swipe_left": 15, "swipe_right": 15, "circle_clockwise": 15,
                          "no_gesture": 30}  # fmt: skip
        assert len(plan) == 75

    def test_every_hard_negative_type_is_covered(self):
        plan = build_protocol(15, 30, seed=3)
        subtypes = Counter(i.subtype for i in plan if i.label == "no_gesture")
        assert set(subtypes) == set(TAKE_NEGATIVE_SUBTYPES)
        assert max(subtypes.values()) - min(subtypes.values()) <= 1  # evenly spread

    def test_each_block_is_balanced_so_stopping_early_stays_balanced(self):
        plan = build_protocol(6, 12, seed=4)
        per_block = len(plan) // 6
        for start in range(0, len(plan), per_block):
            block = Counter(i.label for i in plan[start:start + per_block])
            assert [block[label] for label in TARGET_CLASSES] == [1, 1, 1]

    def test_negatives_that_do_not_divide_evenly_are_still_all_planned(self):
        plan = build_protocol(4, 7, seed=0)
        assert sum(1 for i in plan if i.label == "no_gesture") == 7

    def test_only_valid_items_are_planned(self):
        for item in build_protocol(3, 9, seed=0):
            assert item.label in CLASS_NAMES
            assert (item.subtype is None) == (item.label != "no_gesture")

    @pytest.mark.parametrize("args", [(0, 5), (3, -1)])
    def test_invalid_sizes(self, args):
        with pytest.raises(ValueError):
            build_protocol(*args, seed=0)

    def test_every_plan_item_has_instructions(self):
        for item in build_protocol(2, 16, seed=0):
            assert INSTRUCTIONS[item.key]
        assert INSTRUCTIONS["no_gesture/continuous_use"]


def make_session(tmp_path, plan=None, **recorder_kwargs):
    header = SessionHeader("sess", "p01", "2026-10-12T10:00:00Z", hand="right")
    writer = SessionWriter(tmp_path, header)
    recorder = TakeRecorder("sess", countdown_s=0.2, **recorder_kwargs)
    return CollectionSession(writer, recorder, header, plan), writer, recorder


def perform(session, now, hand=None):
    """Run one take to completion with a rightward open-palm swipe; returns the new time."""
    session.start_take(now)
    for _ in range(int(3 * FPS)):
        fraction = session._recorder.record_fraction(now)
        observation = hand(fraction) if hand else hand_at(swipe_path(1)(fraction))
        session.update(now, observation, 4 / 3)
        now += 1 / FPS
        if session._recorder.state is RecorderState.IDLE:
            return now
    raise AssertionError("take did not finish")


PLAN = [PlanItem("swipe_right"), PlanItem("no_gesture", "idle"), PlanItem("swipe_left")]


class TestSessionProtocolMode:
    def test_an_accepted_take_is_saved_and_the_protocol_advances(self, tmp_path):
        session, writer, _ = make_session(tmp_path, list(PLAN))
        assert session.current_item() == PLAN[0]
        perform(session, 0.0)
        writer.close()
        assert session.counts["swipe_right"] == 1 and session.current_item() == PLAN[1]
        assert len(load_dataset(tmp_path).sessions[0].sequences) == 1

    def test_a_rejected_take_is_not_saved_and_is_repeated(self, tmp_path):
        session, writer, _ = make_session(tmp_path, list(PLAN))
        now = perform(session, 0.0, hand=lambda fraction: None)
        assert session.rejected == 1 and session.saved_takes == 0
        assert session.current_item() == PLAN[0]  # same item again
        assert writer.count == 0
        hud = session.hud(now)
        assert any("Not saved" in text for text, _ in hud.lines)
        writer.close()

    def test_the_protocol_completes(self, tmp_path):
        session, writer, _ = make_session(tmp_path, [PlanItem("swipe_right")])
        now = perform(session, 0.0)
        assert session.protocol_done and session.current_item() is None
        session.start_take(now)  # nothing left to record
        assert session._recorder.state is RecorderState.IDLE
        assert "PROTOCOL COMPLETE" in session.hud(now).headline
        writer.close()

    def test_undo_deletes_the_last_take_and_asks_for_it_again(self, tmp_path):
        session, writer, _ = make_session(tmp_path, list(PLAN))
        now = perform(session, 0.0)
        session.undo(now)
        assert session.saved_takes == 0 and writer.count == 0
        assert session.current_item() == PLAN[0]
        writer.close()
        assert not writer.path.exists()  # nothing left, so no file

    def test_undo_during_a_take_cancels_it(self, tmp_path):
        session, writer, recorder = make_session(tmp_path, list(PLAN))
        session.start_take(0.0)
        session.undo(0.1)
        assert recorder.state is RecorderState.IDLE and writer.count == 0
        writer.close()

    def test_undo_with_nothing_saved(self, tmp_path):
        session, writer, _ = make_session(tmp_path, list(PLAN))
        session.undo(0.0)
        assert "Nothing to delete" in session.hud(0.0).lines[-1][0]
        writer.close()

    def test_skip_moves_on_without_saving(self, tmp_path):
        session, writer, _ = make_session(tmp_path, list(PLAN))
        session.skip(0.0)
        assert session.current_item() == PLAN[1] and session.saved_takes == 0
        writer.close()

    def test_labels_cannot_be_chosen_by_hand_in_protocol_mode(self, tmp_path):
        session, writer, _ = make_session(tmp_path, list(PLAN))
        session.select_label(1, 0.0)
        session.cycle_subtype(0.0)
        assert session.current_item() == PLAN[0]
        assert "--free" in session.hud(0.0).lines[-1][0]
        writer.close()

    def test_commands_are_ignored_mid_take_except_undo(self, tmp_path):
        session, writer, recorder = make_session(tmp_path, list(PLAN))
        session.start_take(0.0)
        session.skip(0.0)
        session.start_take(0.0)  # a second Space must not restart or crash
        assert session.current_item() == PLAN[0] and recorder.state is RecorderState.COUNTDOWN
        writer.close()

    def test_a_save_that_looks_wrong_warns_but_keeps_the_take(self, tmp_path):
        session, writer, _ = make_session(tmp_path, [PlanItem("swipe_left")])
        now = perform(session, 0.0)  # the synthetic hand swipes RIGHT while the label says left
        assert session.counts["swipe_left"] == 1  # kept: review never deletes
        notice, tone = session.hud(now).lines[-1]
        assert tone is Tone.WARN and "moved right" in notice and "Backspace" in notice
        writer.close()


class TestSessionFreeMode:
    def test_labels_are_chosen_with_keys_and_subtypes_cycle(self, tmp_path):
        session, writer, _ = make_session(tmp_path, None)
        assert session.current_item() == PlanItem("swipe_left")
        session.handle(CollectCommand.LABEL_2, 0.0)
        assert session.current_item() == PlanItem("swipe_right")
        session.handle(CollectCommand.LABEL_4, 0.0)
        assert session.current_item() == PlanItem("no_gesture", TAKE_NEGATIVE_SUBTYPES[0])
        session.handle(CollectCommand.SUBTYPE, 0.0)
        assert session.current_item() == PlanItem("no_gesture", TAKE_NEGATIVE_SUBTYPES[1])
        session.handle(CollectCommand.SKIP, 0.0)
        assert session.current_item().label == "swipe_left"  # wraps around
        writer.close()

    def test_free_mode_never_finishes_and_saves_whatever_is_selected(self, tmp_path):
        session, writer, _ = make_session(tmp_path, None)
        session.handle(CollectCommand.LABEL_2, 0.0)
        perform(session, 0.0)
        assert session.counts["swipe_right"] == 1 and not session.protocol_done
        writer.close()


class TestHud:
    def test_asks_for_a_hand_before_it_is_ready(self, tmp_path):
        session, writer, _ = make_session(tmp_path, list(PLAN))
        session.update(0.0, None, 4 / 3)
        hud = session.hud(0.0)
        assert hud.headline.startswith("SHOW YOUR HAND") and hud.tone is Tone.WARN
        session.update(0.03, hand_at((0, 0)), 4 / 3)
        hud = session.hud(0.03)
        assert hud.headline.startswith("READY") and "Space" in hud.headline
        writer.close()

    def test_countdown_then_recording_states(self, tmp_path):
        session, writer, _ = make_session(tmp_path, list(PLAN))
        session.start_take(0.0)
        session.update(0.0, hand_at((0, 0)), 4 / 3)
        hud = session.hud(0.0)
        assert hud.headline.startswith("GET READY") and hud.big_text == "1" and not hud.recording
        session.update(0.25, hand_at((0, 0)), 4 / 3)
        hud = session.hud(0.3)
        assert hud.headline.startswith("RECORDING") and hud.recording
        assert hud.progress is not None and 0.0 <= hud.progress <= 1.0
        writer.close()

    def test_every_screen_says_no_video_is_saved(self, tmp_path):
        session, writer, _ = make_session(tmp_path, list(PLAN))
        assert "no video" in session.hud(0.0).privacy.lower()
        writer.close()

    def test_direction_instructions_use_the_users_point_of_view(self):
        left = " ".join(INSTRUCTIONS["swipe_left"])
        right = " ".join(INSTRUCTIONS["swipe_right"])
        circle = " ".join(INSTRUCTIONS["circle_clockwise"])
        assert "YOUR LEFT" in left and "mirrored" in left
        assert "YOUR RIGHT" in right and "mirrored" in right
        assert "CLOCKWISE" in circle and "mirrored" in circle
        assert all("Open palm" in " ".join(INSTRUCTIONS[k]) for k in TARGET_CLASSES)

    def test_negative_instructions_say_they_are_not_gestures(self):
        for key, text in INSTRUCTIONS.items():
            if key.startswith("no_gesture/"):
                assert "NOT a" in " ".join(text), key

    def test_the_overlay_draws_without_error(self, tmp_path):
        pytest.importorskip("cv2")  # the lean ML environment has no OpenCV
        session, writer, _ = make_session(tmp_path, list(PLAN))
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_overlay(canvas, hand_at((0, 0)), session.hud(0.0))
        assert canvas.any()
        session.start_take(0.0)
        session.update(0.0, hand_at((0, 0)), 4 / 3)
        draw_overlay(np.zeros((480, 640, 3), dtype=np.uint8), None, session.hud(0.1))
        writer.close()

    def test_text_wrapping(self):
        assert wrap_text("one two three four", 9) == ["one two", "three", "four"]
        assert wrap_text("", 10) == []
        assert wrap_text("unbreakableword", 5) == ["unbreakableword"]


class FakeCamera:
    def __init__(self, frames=None, fail_open=False):
        self.opened = self.released = False
        self.fail_open = fail_open
        self._frames = frames  # None: endless frames

    def open(self):
        if self.fail_open:
            raise RuntimeError("no camera")
        self.opened = True

    def read(self):
        if self._frames is not None:
            if not self._frames:
                return None
            return self._frames.pop(0)
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeTracker:
    def __init__(self, hand=None):
        self.closed = False
        self.timestamps = []
        self._hand = hand or (lambda call: hand_at(swipe_path(1)((call % 49) / 48)))
        self.calls = 0

    def detect(self, frame, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        self.calls += 1
        return self._hand(self.calls)

    def close(self):
        self.closed = True


class FakeView:
    def __init__(self, script, quit_after):
        self.script = script  # frame index -> commands
        self.quit_after = quit_after
        self.huds = []
        self.closed = False

    def show(self, frame, observation, hud):
        self.huds.append(hud)
        return len(self.huds) < self.quit_after

    def take_commands(self):
        return tuple(self.script.get(len(self.huds) - 1, ()))

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.t = -1 / FPS

    def __call__(self):
        self.t += 1 / FPS
        return self.t


def options(tmp_path, **kwargs):
    defaults = {"dataset_dir": tmp_path, "countdown_s": 0.2, "per_class": 1, "negatives": 0,
                "seed": 1}  # fmt: skip
    return CollectOptions(**{**defaults, **kwargs})


def run(tmp_path, *, view, camera=None, tracker=None, **option_kwargs):
    messages = []
    code = run_collect(
        options(tmp_path, **option_kwargs), camera=camera or FakeCamera(),
        tracker=tracker or FakeTracker(), view=view, clock=Clock(), out=messages.append,
    )  # fmt: skip
    return code, messages


class TestRunCollect:
    def test_a_headless_session_records_and_cleans_up(self, tmp_path):
        camera, tracker = FakeCamera(), FakeTracker()
        view = FakeView({0: [CollectCommand.START]}, quit_after=90)
        code, messages = run(tmp_path, view=view, camera=camera, tracker=tracker)
        assert code == EXIT_OK
        assert camera.opened and camera.released and tracker.closed and view.closed
        dataset = load_dataset(tmp_path)
        (session,) = dataset.sessions
        assert len(session.sequences) == 1
        seq = session.sequences[0]
        assert seq.label in TARGET_CLASSES
        assert seq.duration_s == pytest.approx(DEFAULT_WINDOW_S, abs=0.1)
        assert session.header.participant_id == "p01"
        assert session.header.camera["frame_width"] == 640
        protocol = session.header.protocol
        assert protocol["mode"] == "protocol" and protocol["seed"] == 1
        assert ID_PATTERN.match(session.header.session_id)
        assert any("Saved 1 takes" in m for m in messages)
        assert any("no video" in m for m in messages)

    def test_timestamps_given_to_the_tracker_increase(self, tmp_path):
        tracker = FakeTracker()
        run(tmp_path, view=FakeView({}, quit_after=40), tracker=tracker)
        assert tracker.timestamps == sorted(set(tracker.timestamps))

    def test_nothing_recorded_means_no_file(self, tmp_path):
        code, messages = run(tmp_path, view=FakeView({}, quit_after=10))
        assert code == EXIT_OK
        assert list(tmp_path.glob("*.jsonl")) == []
        assert any("Nothing was recorded" in m for m in messages)

    def test_a_continuous_recording_ends_the_session_by_itself(self, tmp_path):
        view = FakeView({}, quit_after=100000)
        code, _ = run(tmp_path, view=view, stream_seconds=5.0)
        assert code == EXIT_OK
        (session,) = load_dataset(tmp_path).sessions
        (stream,) = session.sequences
        assert (stream.kind, stream.label, stream.subtype) == ("stream", "no_gesture",
                                                                "continuous_use")  # fmt: skip
        assert session.header.protocol["mode"] == "stream"
        assert len(view.huds) < 300  # it stopped on its own after about 5.2 s

    def test_cancelling_a_continuous_recording_keeps_nothing(self, tmp_path):
        view = FakeView({20: [CollectCommand.UNDO]}, quit_after=40)
        run(tmp_path, view=view, stream_seconds=5.0)
        assert list(tmp_path.glob("*.jsonl")) == []

    def test_free_mode_is_recorded_in_the_header(self, tmp_path):
        view = FakeView({0: [CollectCommand.START]}, quit_after=90)
        run(tmp_path, view=view, free=True)
        assert load_dataset(tmp_path).sessions[0].header.protocol["mode"] == "free"

    def test_a_camera_that_never_delivers_frames(self, tmp_path):
        camera = FakeCamera(frames=[])
        code, messages = run(tmp_path, view=FakeView({}, 5), camera=camera)
        assert code == EXIT_ERROR and camera.released
        assert any("no frames" in m for m in messages)
        assert list(tmp_path.glob("*.jsonl")) == []

    def test_a_camera_that_cannot_open(self, tmp_path):
        camera, tracker = FakeCamera(fail_open=True), FakeTracker()
        view = FakeView({}, 5)
        code, messages = run(tmp_path, view=view, camera=camera, tracker=tracker)
        assert code == EXIT_ERROR and tracker.closed and view.closed
        assert any("no camera" in m for m in messages)

    def test_dropped_frames_are_survivable(self, tmp_path):
        frames = [None, np.zeros((480, 640, 3), dtype=np.uint8)] * 30
        camera = FakeCamera(frames=frames)
        code, _ = run(tmp_path, view=FakeView({}, quit_after=40), camera=camera)
        assert code == EXIT_OK

    @pytest.mark.parametrize("kwargs, text", [
        ({"participant": "Jane Doe"}, "pseudonym"),
        ({"participant": "jane@example.com"}, "pseudonym"),
        ({"hand": "both"}, "--hand"),
        ({"per_class": 0}, "--per-class"),
        ({"negatives": -1}, "--negatives"),
        ({"countdown_s": 30.0}, "--countdown"),
        ({"stream_seconds": 1.0}, "--stream-seconds"),
        ({"min_palm_fraction": 0.0}, "--min-palm-fraction"),
        ({"conditions": "a" * 500}, "--conditions"),
        ({"min_confidence": 2.0}, "--min-confidence"),
    ])
    def test_bad_options_are_rejected_before_anything_starts(self, tmp_path, kwargs, text):
        camera = FakeCamera()
        code, messages = run(tmp_path, view=FakeView({}, 5), camera=camera, **kwargs)
        assert code == EXIT_USAGE and any(text in m for m in messages)
        assert not camera.opened and list(tmp_path.glob("*.jsonl")) == []

    def test_a_missing_model_is_a_clean_error(self, tmp_path):
        messages = []
        code = run_collect(
            options(tmp_path, model_path=tmp_path / "missing.task"),
            camera=FakeCamera(), view=FakeView({}, 5), out=messages.append,
        )  # fmt: skip
        assert code == EXIT_ERROR
        assert any("could not start" in m and "not found" in m for m in messages)

    def test_the_session_files_are_new_each_run(self, tmp_path):
        for _ in range(2):
            run(tmp_path, view=FakeView({0: [CollectCommand.START]}, quit_after=90))
        assert len(list(tmp_path.glob("*.jsonl"))) == 2


def test_session_ids_are_valid_and_unique():
    ids = {new_session_id() for _ in range(50)}
    assert len(ids) == 50 and all(ID_PATTERN.match(i) for i in ids)
