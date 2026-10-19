"""Review marks and the skeleton replay viewer.

All data is SYNTHETIC geometry; these tests check software behaviour, not recognition accuracy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from nocursor.dynamic.cli import main
from nocursor.dynamic.review import (
    MARK_FORMAT,
    MARKS,
    MarkStore,
    ReviewError,
    describe_review_source,
    file_sha256,
    load_items,
    load_marks,
    tally_candidates,
    validate_items,
)
from nocursor.dynamic.review_viewer import (
    ACTIONS,
    ReviewSession,
    key_to_action,
    render_frame,
    run_viewer,
)
from test_dynamic_stream_eval import build_stream, one_swipe, write_session


def make_items():
    return [
        {"item_id": "c001", "recording_id": "rec-a", "predicted": "swipe_left", "from_s": 1.0,
         "to_s": 4.0, "candidate_times_s": [2.5, 2.6]},
        {"item_id": "c002", "recording_id": "rec-a", "predicted": "circle_clockwise",
         "from_s": 6.0, "to_s": 9.0, "candidate_times_s": [8.0]},
        {"item_id": "c003", "recording_id": "rec-b", "predicted": "swipe_right", "from_s": 2.0,
         "to_s": 5.0, "candidate_times_s": [3.5]},
    ]  # fmt: skip


@pytest.fixture
def items_file(tmp_path):
    path = tmp_path / "review_items.json"
    path.write_text(json.dumps(make_items()), encoding="utf-8")
    return path


@pytest.fixture
def store(items_file):
    return MarkStore.open(items_file)


@pytest.fixture
def sequences():
    return {
        "rec-a": build_stream(12, one_swipe, sequence_id="rec-a"),
        "rec-b": build_stream(12, one_swipe, sequence_id="rec-b"),
    }


# ----------------------------------------------------------------------------- items
class TestItems:
    def test_a_valid_list_passes(self):
        assert validate_items(make_items()) == make_items()

    def test_duplicate_ids_are_refused(self):
        items = make_items()
        items[1]["item_id"] = "c001"
        with pytest.raises(ReviewError, match="duplicate"):
            validate_items(items)

    @pytest.mark.parametrize("field", ["item_id", "recording_id", "predicted", "from_s", "to_s"])
    def test_a_missing_identity_field_is_refused(self, field):
        items = make_items()
        del items[0][field]
        with pytest.raises(ReviewError, match="missing"):
            validate_items(items)

    def test_an_unknown_gesture_or_bad_time_range_is_refused(self):
        items = make_items()
        items[0]["predicted"] = "wave"
        with pytest.raises(ReviewError, match="unknown gesture"):
            validate_items(items)
        items = make_items()
        items[0]["to_s"] = items[0]["from_s"]
        with pytest.raises(ReviewError, match="time range"):
            validate_items(items)

    def test_items_must_be_blind(self):
        items = make_items()
        items[0]["system"] = "gru"
        with pytest.raises(ReviewError, match="blind"):
            validate_items(items)

    def test_the_file_must_be_readable_json(self, tmp_path):
        with pytest.raises(ReviewError):
            load_items(tmp_path / "missing.json")
        broken = tmp_path / "broken.json"
        broken.write_text("{", encoding="utf-8")
        with pytest.raises(ReviewError):
            load_items(broken)


# ------------------------------------------------------------------------------ marks
class TestMarkStore:
    def test_nothing_is_marked_at_first(self, store):
        assert store.progress() == (0, 3) and store.plain() == {}
        assert store.unreviewed() == ["c001", "c002", "c003"]
        assert not store.path.exists()  # opening alone writes nothing

    def test_a_mark_is_saved_with_the_identity_of_its_candidate(self, store, items_file):
        store.set("c002", "false")
        data = json.loads(store.path.read_text("utf-8"))
        assert data["format"] == MARK_FORMAT and data["items_sha256"] == file_sha256(items_file)
        entry = data["marks"]["c002"]
        assert entry["mark"] == "false" and entry["revisions"] == 1
        assert (entry["recording_id"], entry["predicted"]) == ("rec-a", "circle_clockwise")
        assert (entry["from_s"], entry["to_s"]) == (6.0, 9.0) and entry["reviewed_utc"]
        assert set(data["marks"]) == {"c002"}  # the others have no entry at all

    def test_marks_survive_reopening(self, store, items_file):
        store.set("c001", "genuine")
        store.set("c003", "ambiguous")
        again = MarkStore.open(items_file)
        assert again.plain() == {"c001": "genuine", "c003": "ambiguous"}
        assert again.unreviewed() == ["c002"] and again.progress() == (2, 3)

    def test_every_allowed_label_and_only_those(self, store):
        for mark in MARKS:
            store.set("c001", mark)
            assert store.mark_of("c001") == mark
        assert MARKS == ("false", "genuine", "ambiguous")
        for bad in ("maybe", "", "FALSE", "true", None):
            with pytest.raises(ReviewError, match="invalid mark"):
                store.set("c001", bad)
        assert store.mark_of("c001") == "ambiguous"  # the failed attempts changed nothing

    def test_an_unknown_candidate_is_refused_and_nothing_is_written(self, store):
        with pytest.raises(ReviewError, match="unknown candidate"):
            store.set("c999", "false")
        assert not store.path.exists()

    def test_changing_a_mark_counts_a_revision(self, store):
        store.set("c001", "false")
        store.set("c001", "genuine")
        assert store.entries["c001"]["revisions"] == 2 and store.mark_of("c001") == "genuine"

    def test_clearing_returns_a_candidate_to_unreviewed(self, store, items_file):
        store.set("c001", "false")
        store.clear("c001")
        assert store.mark_of("c001") is None and "c001" in store.unreviewed()
        assert "c001" not in json.loads(store.path.read_text("utf-8"))["marks"]
        store.clear("c002")  # clearing an unmarked candidate is harmless
        assert MarkStore.open(items_file).plain() == {}

    def test_saving_is_atomic_and_leaves_no_temporary_file(self, store):
        store.set("c001", "false")
        assert not list(store.path.parent.glob("*.tmp"))

    def test_marks_for_a_different_candidate_list_are_refused(self, store, tmp_path):
        store.set("c001", "false")
        other = make_items()
        other[0]["to_s"] = 5.0
        other_path = tmp_path / "other_items.json"
        other_path.write_text(json.dumps(other), encoding="utf-8")
        with pytest.raises(ReviewError, match="different candidate list"):
            MarkStore.open(other_path, store.path)

    def test_a_marks_file_that_is_not_the_tools_own_is_never_overwritten(self, items_file):
        path = items_file.parent / "review_marks.json"
        path.write_text(json.dumps({"c001": "false"}), encoding="utf-8")
        with pytest.raises(ReviewError, match="refusing to overwrite"):
            MarkStore.open(items_file)
        assert json.loads(path.read_text("utf-8")) == {"c001": "false"}

    def test_a_mark_edited_to_point_at_another_recording_is_detected(self, store, items_file):
        store.set("c001", "false")
        data = json.loads(store.path.read_text("utf-8"))
        data["marks"]["c001"]["recording_id"] = "rec-b"
        store.path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ReviewError, match="different candidate"):
            MarkStore.open(items_file)

    def test_an_unknown_candidate_or_a_bad_label_in_the_file_is_detected(self, store, items_file):
        store.set("c001", "false")
        data = json.loads(store.path.read_text("utf-8"))
        data["marks"]["c001"]["mark"] = "yes"
        store.path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ReviewError, match="invalid mark"):
            MarkStore.open(items_file)
        data["marks"] = {"c404": {"mark": "false"}}
        store.path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ReviewError, match="unknown candidate"):
            MarkStore.open(items_file)


class TestLoadMarks:
    def test_the_tools_file_loads_as_a_plain_mapping(self, store, items_file):
        store.set("c001", "false")
        store.set("c003", "genuine")
        marks = load_marks(store.path, make_items(), file_sha256(items_file))
        assert marks == {"c001": "false", "c003": "genuine"}

    def test_unmarked_candidates_stay_absent_and_are_never_inferred(self, store, items_file):
        store.set("c001", "false")
        marks = load_marks(store.path, make_items(), file_sha256(items_file))
        assert "c002" not in marks and "c003" not in marks

    def test_a_hand_written_mapping_is_accepted_but_validated(self, tmp_path):
        path = tmp_path / "marks.json"
        path.write_text(json.dumps({"c001": "false", "c002": "ambiguous"}), encoding="utf-8")
        assert load_marks(path, make_items()) == {"c001": "false", "c002": "ambiguous"}
        path.write_text(json.dumps({"c001": "nope"}), encoding="utf-8")
        with pytest.raises(ReviewError, match="invalid mark"):
            load_marks(path, make_items())
        path.write_text(json.dumps({"c777": "false"}), encoding="utf-8")
        with pytest.raises(ReviewError, match="unknown candidate"):
            load_marks(path, make_items())

    def test_marks_made_for_another_list_are_refused(self, store, items_file):
        store.set("c001", "false")
        with pytest.raises(ReviewError, match="different candidate list"):
            load_marks(store.path, make_items(), "0" * 64)

    def test_unreadable_files_are_refused(self, tmp_path):
        with pytest.raises(ReviewError):
            load_marks(tmp_path / "nope.json", make_items())
        bad = tmp_path / "bad.json"
        bad.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ReviewError):
            load_marks(bad, make_items())


class TestTally:
    KEY = {"c001": ["gru", "rule_defaults"], "c002": ["gru"], "c003": ["rule_defaults"]}

    def candidate(self, recording, time, label):
        return {"recording_id": recording, "time_s": time, "label": label}

    def test_reviewed_and_unreviewed_candidates_are_counted_apart(self):
        marks = {"c001": "false", "c002": "genuine"}
        gru = [self.candidate("rec-a", 2.5, "swipe_left"),
               self.candidate("rec-a", 8.0, "circle_clockwise")]  # fmt: skip
        counts = tally_candidates(gru, "gru", make_items(), self.KEY, marks)
        assert counts == {"candidates": 2, "false": 1, "genuine": 1, "ambiguous": 0,
                          "unreviewed": 0, "not_in_review_set": 0}  # fmt: skip

    def test_a_candidate_whose_item_has_no_mark_is_unreviewed_not_false(self):
        counts = tally_candidates([self.candidate("rec-a", 2.6, "swipe_left")], "gru",
                                  make_items(), self.KEY, {})  # fmt: skip
        assert counts["unreviewed"] == 1 and counts["false"] == 0

    def test_a_candidate_with_no_review_item_is_unreviewed_and_flagged(self):
        counts = tally_candidates([self.candidate("rec-a", 11.0, "swipe_left")], "gru",
                                  make_items(), self.KEY, {"c001": "false"})  # fmt: skip
        assert counts["unreviewed"] == 1 and counts["not_in_review_set"] == 1
        assert counts["false"] == 0

    def test_only_items_found_by_that_variant_count(self):
        marks = {"c003": "false"}
        counts = tally_candidates([self.candidate("rec-b", 3.5, "swipe_right")], "gru",
                                  make_items(), self.KEY, marks)  # fmt: skip
        assert counts["false"] == 0 and counts["not_in_review_set"] == 1
        other = tally_candidates([self.candidate("rec-b", 3.5, "swipe_right")], "rule_defaults",
                                 make_items(), self.KEY, marks)  # fmt: skip
        assert other["false"] == 1

    def test_the_gesture_and_the_time_must_match(self):
        marks = {"c001": "false"}
        wrong_gesture = tally_candidates([self.candidate("rec-a", 2.5, "swipe_right")], "gru",
                                         make_items(), self.KEY, marks)  # fmt: skip
        too_far = tally_candidates([self.candidate("rec-a", 2.9, "swipe_left")], "gru",
                                   make_items(), self.KEY, marks)  # fmt: skip
        assert wrong_gesture["false"] == 0 and too_far["false"] == 0


class TestDescribeReviewSource:
    def source(self, items=None, marks=None, present=True):
        return {"folder": "some/folder", "items": items or make_items(),
                "items_sha256": "a" * 64, "marks": marks or {}, "marks_present": present,
                "marks_sha256": "b" * 64 if present else None}  # fmt: skip

    def test_an_identical_list_is_reported_as_matching(self):
        info = describe_review_source(self.source(marks={"c001": "false"}), make_items(), "a" * 64)
        assert info["matches_current_export"] is True
        assert info["differing_items"] == [] and info["items_only_in_source"] == []
        assert info["items_only_in_current"] == []
        assert (info["items_total"], info["items_marked"]) == (3, 1)
        assert info["folder"] == "some/folder" and info["marks_sha256"] == "b" * 64
        assert info["items_sha256"] == info["current_export_items_sha256"] == "a" * 64

    def test_a_different_hash_is_reported_with_the_items_that_differ(self):
        current = make_items()
        current[0]["candidate_times_s"] = [2.5]  # the source has [2.5, 2.6]
        info = describe_review_source(self.source(), current, "c" * 64)
        assert info["matches_current_export"] is False
        assert info["differing_items"] == [{"item_id": "c001", "fields": {
            "candidate_times_s": {"source": [2.5, 2.6], "current": [2.5]}}}]  # fmt: skip

    def test_a_changed_identity_field_is_reported_too(self):
        current = make_items()
        current[1]["to_s"] = 9.5
        info = describe_review_source(self.source(), current, "c" * 64)
        assert info["differing_items"][0]["fields"]["to_s"] == {"source": 9.0, "current": 9.5}

    def test_items_present_on_only_one_side_are_listed(self):
        current = make_items()[:2] + [{**make_items()[2], "item_id": "c004"}]
        info = describe_review_source(self.source(), current, "c" * 64)
        assert info["items_only_in_source"] == ["c003"]
        assert info["items_only_in_current"] == ["c004"]

    def test_a_missing_marks_file_is_recorded_as_absent(self):
        info = describe_review_source(self.source(present=False), make_items(), "a" * 64)
        assert info["marks_file_present"] is False and info["marks_sha256"] is None
        assert info["items_marked"] == 0


# ----------------------------------------------------------------------------- session
class TestSession:
    def test_it_starts_at_the_first_unreviewed_candidate(self, store, sequences):
        store.set("c001", "false")
        assert ReviewSession(store.items, store, sequences).item["item_id"] == "c002"

    def test_marking_saves_and_moves_to_the_next_unreviewed_candidate(self, store, sequences):
        session = ReviewSession(store.items, store, sequences)
        session.apply("false")
        assert session.item["item_id"] == "c002" and store.mark_of("c001") == "false"
        session.apply("ambiguous")
        session.apply("genuine")
        assert store.plain() == {"c001": "false", "c002": "ambiguous", "c003": "genuine"}
        assert session.item["item_id"] == "c003"  # nothing left to review: it stays put

    def test_marks_are_attached_to_the_candidate_that_was_on_screen(self, store, sequences):
        session = ReviewSession(store.items, store, sequences)
        session.apply("next")
        session.apply("next")
        session.apply("genuine")
        assert store.plain() == {"c003": "genuine"}
        assert store.entries["c003"]["recording_id"] == "rec-b"

    def test_nothing_but_a_mark_action_ever_marks_anything(self, store, sequences):
        session = ReviewSession(store.items, store, sequences)
        for action in ("next", "prev", "pause", "pause", "restart", "next_unreviewed", "clear",
                       "next", "prev", "quit"):  # fmt: skip
            session.apply(action)
        for _ in range(500):
            session.tick(0.5)  # many loops of the clip
        assert store.plain() == {} and not store.path.exists()
        assert session.finished

    def test_quitting_midway_keeps_the_marks_made_and_leaves_the_rest_unreviewed(
        self, store, sequences, items_file
    ):
        session = ReviewSession(store.items, store, sequences)
        session.apply("false")
        session.apply("quit")
        again = MarkStore.open(items_file)
        assert again.plain() == {"c001": "false"} and again.unreviewed() == ["c002", "c003"]

    def test_clearing_returns_the_candidate_to_unreviewed(self, store, sequences):
        session = ReviewSession(store.items, store, sequences)
        session.apply("false")
        session.apply("prev")
        session.apply("clear")
        assert store.plain() == {}

    def test_navigation_stops_at_the_ends(self, store, sequences):
        session = ReviewSession(store.items, store, sequences)
        session.apply("prev")
        assert session.index == 0
        for _ in range(5):
            session.apply("next")
        assert session.index == 2

    def test_next_unreviewed_wraps_and_skips_reviewed_candidates(self, store, sequences):
        store.set("c002", "false")
        session = ReviewSession(store.items, store, sequences)
        session.apply("next_unreviewed")
        assert session.item["item_id"] == "c003"
        session.apply("next_unreviewed")
        assert session.item["item_id"] == "c001"

    def test_an_unknown_action_is_refused(self, store, sequences):
        with pytest.raises(ReviewError, match="unknown action"):
            ReviewSession(store.items, store, sequences).apply("delete_everything")

    def test_a_missing_recording_or_an_empty_list_is_refused(self, store, sequences):
        with pytest.raises(ReviewError, match="rec-b"):
            ReviewSession(store.items, store, {"rec-a": sequences["rec-a"]})
        with pytest.raises(ReviewError, match="no candidates"):
            ReviewSession([], store, sequences)

    def test_playback_loops_inside_the_candidates_time_range(self, store, sequences):
        session = ReviewSession(store.items, store, sequences)
        start, end = session.item["from_s"], session.item["to_s"]
        for _ in range(200):
            session.tick(0.37)
            assert start <= session.t <= end

    def test_pausing_stops_playback_and_restart_rewinds(self, store, sequences):
        session = ReviewSession(store.items, store, sequences)
        session.tick(1.0)
        moved = session.t
        session.apply("pause")
        session.tick(1.0)
        assert session.t == moved and session.paused
        session.apply("restart")
        assert session.t == session.item["from_s"]

    def test_the_skeleton_is_in_the_mirrored_view_and_follows_the_recording(self, store, sequences):
        session = ReviewSession(store.items, store, sequences)
        # one_swipe: rests at the left until 3 s, then sweeps to the user's right
        before = session.hand_at(2.0)
        after = session.hand_at(3.9)
        assert before.shape == (21, 2) and after[:, 0].mean() > before[:, 0].mean()

    def test_an_untracked_moment_has_no_hand(self, store):
        count = int(round(12 * 30)) + 1
        usable = np.ones(count, bool)
        usable[60:90] = False  # 2.0 to 3.0 s
        sequences = {"rec-a": build_stream(12, one_swipe, usable=usable, sequence_id="rec-a"),
                     "rec-b": build_stream(12, one_swipe, sequence_id="rec-b")}  # fmt: skip
        session = ReviewSession(store.items, store, sequences)
        assert session.hand_at(2.5) is None and session.hand_at(1.5) is not None
        assert session.hand_at(-1.0) is None

    def test_the_trail_covers_the_clip(self, store, sequences):
        session = ReviewSession(store.items, store, sequences)
        trail = session.trail()
        assert trail.ndim == 2 and trail.shape[1] == 2 and len(trail) > 10


class TestBlindPresentation:
    def test_the_screen_text_has_the_gesture_and_the_range_but_no_system_or_confidence(
        self, store, sequences
    ):
        text = "\n".join(ReviewSession(store.items, store, sequences).hud_lines())
        assert "swipe_left" in text and "rec-a" in text and "c001" in text
        for forbidden in ("gru", "rule", "confidence", "probab", "model", "tau", "threshold"):
            assert forbidden not in text.lower()

    def test_the_text_says_when_a_candidate_has_not_been_reviewed(self, store, sequences):
        session = ReviewSession(store.items, store, sequences)
        assert "NOT REVIEWED" in "\n".join(session.hud_lines())
        session.apply("genuine")
        session.apply("prev")
        assert "genuine" in "\n".join(session.hud_lines())

    def test_progress_is_shown(self, store, sequences):
        store.set("c002", "false")
        assert "Reviewed 1 of 3" in "\n".join(
            ReviewSession(store.items, store, sequences).hud_lines())  # fmt: skip


class TestKeys:
    @pytest.mark.parametrize("char,action", [("f", "false"), ("F", "false"), ("g", "genuine"),
                                             ("G", "genuine"), ("a", "ambiguous"),
                                             ("A", "ambiguous"), ("u", "clear"),
                                             ("n", "next_unreviewed"), ("p", "prev"),
                                             ("s", "next"), ("r", "restart"), ("q", "quit")])
    def test_letter_keys(self, char, action):
        assert key_to_action(ord(char)) == action

    def test_space_escape_and_arrows(self):
        assert key_to_action(32) == "pause" and key_to_action(27) == "quit"
        assert key_to_action(2555904) == "next" and key_to_action(2424832) == "prev"

    def test_other_keys_do_nothing(self):
        assert key_to_action(ord("x")) is None and key_to_action(-1) is None
        assert key_to_action(13) is None  # Enter must not mark anything

    def test_every_action_the_keys_can_trigger_exists(self):
        assert {key_to_action(c) for c in range(0, 3_000_000, 1)
                if key_to_action(c)} <= set(ACTIONS)  # fmt: skip


# -------------------------------------------------------------------------------- drawing
class TestDrawing:
    def test_a_frame_is_a_canvas_image_that_changes_with_time(self, store, sequences):
        pytest.importorskip("cv2")
        session = ReviewSession(store.items, store, sequences)
        first = render_frame(session)
        assert first.shape == (480, 640, 3) and first.dtype == np.uint8
        session.t = 3.9
        assert not np.array_equal(first, render_frame(session))
        again = ReviewSession(store.items, store, sequences)
        again.t = 3.9
        assert np.array_equal(render_frame(session), render_frame(again))  # deterministic

    def test_an_untracked_moment_still_renders(self, store):
        pytest.importorskip("cv2")
        count = int(round(12 * 30)) + 1
        usable = np.zeros(count, bool)
        sequences = {r: build_stream(12, one_swipe, usable=usable, sequence_id=r)
                     for r in ("rec-a", "rec-b")}  # fmt: skip
        assert render_frame(ReviewSession(store.items, store, sequences)).shape == (480, 640, 3)


class FakeCv2:
    """Stands in for OpenCV's window functions so the loop can run headless."""

    WINDOW_AUTOSIZE = 1
    WND_PROP_VISIBLE = 4

    def __init__(self, keys, visible_after=None):
        self.keys = list(keys)
        self.shown = 0
        self.closed = False
        self.visible_after = visible_after

    def namedWindow(self, *args):
        pass

    def imshow(self, name, image):
        self.shown += 1

    def waitKeyEx(self, delay):
        return self.keys.pop(0) if self.keys else ord("q")

    def getWindowProperty(self, *args):
        if self.visible_after is not None and self.shown > self.visible_after:
            return 0
        return 1

    def destroyAllWindows(self):
        self.closed = True


class TestViewerLoop:
    def run(self, store, sequences, keys, monkeypatch, **kwargs):
        import nocursor.dynamic.review_viewer as viewer

        fake = FakeCv2(keys, **kwargs)
        monkeypatch.setitem(sys.modules, "cv2", fake)
        monkeypatch.setattr(viewer, "render_frame", lambda session, size=None: None)
        session = ReviewSession(store.items, store, sequences)
        run_viewer(session, clock=iter(np.arange(0, 1000, 0.05)).__next__)
        return session, fake

    def test_keys_mark_candidates_and_q_ends_the_loop(self, store, sequences, monkeypatch):
        keys = [ord("f"), -1, ord("g"), ord("x"), ord("a"), ord("q")]
        session, fake = self.run(store, sequences, keys, monkeypatch)
        assert store.plain() == {"c001": "false", "c002": "genuine", "c003": "ambiguous"}
        assert session.finished and fake.closed

    def test_no_key_means_no_mark(self, store, sequences, monkeypatch):
        session, _ = self.run(store, sequences, [-1] * 20 + [ord("q")], monkeypatch)
        assert store.plain() == {}

    def test_closing_the_window_ends_the_loop_without_marking(self, store, sequences,
                                                              monkeypatch):  # fmt: skip
        session, fake = self.run(store, sequences, [-1] * 100, monkeypatch, visible_after=3)
        assert store.plain() == {} and fake.closed and not session.finished


# ------------------------------------------------------------------------------------ CLI
class TestCli:
    def setup(self, tmp_path):
        dev = tmp_path / "datasets" / "streaming" / "dev"
        for rid in ("rec-a", "rec-b"):
            write_session(dev, f"s-{rid}", [build_stream(12, one_swipe, session_id=f"s-{rid}",
                                                        sequence_id=rid)])  # fmt: skip
        items = tmp_path / "review" / "review_items.json"
        items.parent.mkdir()
        items.write_text(json.dumps(make_items()), encoding="utf-8")
        return dev, items

    def patch_viewer(self, monkeypatch, actions):
        import nocursor.dynamic.review_viewer as viewer

        def fake_run(session, **kwargs):
            for action in actions:
                session.apply(action)

        monkeypatch.setattr(viewer, "run_viewer", fake_run)

    def test_marks_are_saved_next_to_the_items_and_the_rest_stay_unreviewed(
        self, tmp_path, monkeypatch, capsys
    ):
        dev, items = self.setup(tmp_path)
        self.patch_viewer(monkeypatch, ["false", "genuine"])
        code = main(["stream-review", "--items", str(items), "--recordings", str(dev)])
        assert code == 0
        marks = load_marks(items.parent / "review_marks.json", make_items(), file_sha256(items))
        assert marks == {"c001": "false", "c002": "genuine"}
        assert "Reviewed 2 of 3" in capsys.readouterr().out

    def test_a_recording_that_cannot_be_found_is_an_error(self, tmp_path, monkeypatch, capsys):
        dev, items = self.setup(tmp_path)
        (dev / "s-rec-b.jsonl").unlink()
        self.patch_viewer(monkeypatch, [])
        assert main(["stream-review", "--items", str(items), "--recordings", str(dev)]) == 1
        assert "rec-b" in capsys.readouterr().err

    def test_the_sealed_final_folder_is_refused_for_ordinary_candidates(
        self, tmp_path, monkeypatch, capsys
    ):
        dev, items = self.setup(tmp_path)
        final = tmp_path / "datasets" / "streaming" / "final"
        final.mkdir(parents=True)
        self.patch_viewer(monkeypatch, [])
        assert main(["stream-review", "--items", str(items), "--recordings", str(final)]) == 1
        assert "sealed" in capsys.readouterr().err

    def test_the_candidates_of_an_acceptance_run_may_be_reviewed_against_the_final_data(
        self, tmp_path, monkeypatch
    ):
        dev, _ = self.setup(tmp_path)
        final = tmp_path / "datasets" / "streaming" / "final"
        for rid in ("rec-a", "rec-b"):
            write_session(final, f"s-final-{rid}", [build_stream(
                12, one_swipe, session_id=f"s-final-{rid}", sequence_id=rid)])  # fmt: skip
        acceptance = tmp_path / "ml_runs" / "streaming" / "freeze-x" / "acceptance"
        acceptance.mkdir(parents=True)
        items = acceptance / "review_items.json"
        items.write_text(json.dumps(make_items()), encoding="utf-8")
        self.patch_viewer(monkeypatch, ["ambiguous"])
        code = main(["stream-review", "--items", str(items), "--dataset-dir",
                     str(tmp_path / "datasets")])  # fmt: skip
        assert code == 0
        assert MarkStore.open(items).plain() == {"c001": "ambiguous"}

    def test_a_marks_file_for_another_list_is_refused(self, tmp_path, monkeypatch, capsys):
        dev, items = self.setup(tmp_path)
        MarkStore.open(items).set("c001", "false")
        other = make_items()
        other[2]["to_s"] = 6.0
        items.write_text(json.dumps(other), encoding="utf-8")  # the list changed afterwards
        self.patch_viewer(monkeypatch, [])
        assert main(["stream-review", "--items", str(items), "--recordings", str(dev)]) == 1
        assert "different candidate list" in capsys.readouterr().err


def test_the_viewer_does_not_import_a_camera_or_the_desktop_controls():
    source = (Path(__file__).resolve().parents[1] / "src" / "nocursor" / "dynamic"
              / "review_viewer.py").read_text("utf-8")  # fmt: skip
    for forbidden in ("VideoCapture", "pyautogui", "nocursor.controller", "nocursor.camera",
                      "hand_tracker", "VideoWriter", "imwrite"):  # fmt: skip
        assert forbidden not in source


def test_validate_and_store_agree_on_what_a_valid_list_is(tmp_path):
    path = tmp_path / "items.json"
    path.write_text(json.dumps(make_items()), encoding="utf-8")
    assert MarkStore.open(path).items == validate_items(make_items())
