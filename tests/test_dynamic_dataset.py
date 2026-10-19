"""Dataset serialisation, validation and malformed-file handling (SYNTHETIC fixtures)."""

import json

import numpy as np
import pytest

from dynamic_helpers import circle_path, make_sequence, swipe_path
from nocursor.dynamic.dataset import (
    DatasetValidationError,
    SessionWriter,
    load_dataset,
    load_session,
    sequence_from_json,
    sequence_to_json,
    validate_sequence,
)
from nocursor.dynamic.schema import (
    SCHEMA_VERSION,
    DatasetError,
    SequenceRecord,
    SessionHeader,
)

SESSION = "sess"


def header(session_id=SESSION, participant="p01", **kwargs):
    return SessionHeader(
        session_id=session_id, participant_id=participant,
        created_utc="2026-10-12T10:00:00Z", **kwargs,
    )  # fmt: skip


def seq(n=1, label="swipe_right", session_id=SESSION, path=None, **kwargs):
    path = path or swipe_path(1 if label == "swipe_right" else -1)
    return make_sequence(
        path, label=label, sequence_id=f"{session_id}-{n:04d}", session_id=session_id, **kwargs
    )


def write_session(directory, sequences, session_id=SESSION, **header_kwargs):
    writer = SessionWriter(directory, header(session_id, **header_kwargs))
    for item in sequences:
        writer.append(item)
    writer.close()
    return writer.path


def edit_lines(path, edit):
    lines = path.read_text(encoding="utf-8").split("\n")
    path.write_text("\n".join(edit(lines)), encoding="utf-8")


class TestHeader:
    @pytest.mark.parametrize("participant", [
        "", "p", "P01", "jane doe", "jane@example.com", "x" * 33, "../escape", "name!",
    ])
    def test_participant_must_be_a_pseudonym_not_a_name_or_address(self, participant):
        with pytest.raises(DatasetError, match="participant_id"):
            header(participant=participant)

    @pytest.mark.parametrize("participant", ["p01", "participant-2", "a1", "user_7"])
    def test_pseudonyms_are_accepted(self, participant):
        assert header(participant=participant).participant_id == participant

    def test_other_fields_are_validated(self):
        with pytest.raises(DatasetError, match="hand"):
            header(hand="both")
        with pytest.raises(DatasetError, match="provenance"):
            header(provenance="invented")
        with pytest.raises(DatasetError, match="created_utc"):
            SessionHeader("sess", "p01", "yesterday")
        with pytest.raises(DatasetError, match="conditions"):
            header(conditions="line one\nline two")
        with pytest.raises(DatasetError, match="session_id"):
            header(session_id="a b")


class TestRoundTrip:
    def test_a_written_session_reads_back_the_same(self, tmp_path):
        original = [seq(1, "swipe_right"), seq(2, "swipe_left"),
                    seq(3, "circle_clockwise", path=circle_path())]  # fmt: skip
        write_session(tmp_path, original, hand="right", conditions="synthetic test")
        loaded = load_session(tmp_path / f"{SESSION}.jsonl")
        assert loaded.header.hand == "right" and loaded.header.participant_id == "p01"
        assert [s.sequence_id for s in loaded.sequences] == [s.sequence_id for s in original]
        for before, after in zip(original, loaded.sequences, strict=True):
            assert after.label == before.label and after.kind == before.kind
            assert np.allclose(after.landmarks, before.landmarks, atol=1e-6, equal_nan=True)
            assert np.array_equal(after.t_ms, before.t_ms)
            assert np.array_equal(after.usable, before.usable)
            assert after.aspect_ratio == pytest.approx(before.aspect_ratio)

    def test_missing_frames_survive_as_nan(self, tmp_path):
        count = seq().n_frames
        usable = np.ones(count, dtype=bool)
        usable[10:12] = False
        write_session(tmp_path, [seq(usable=usable)])
        loaded = load_session(tmp_path / f"{SESSION}.jsonl").sequences[0]
        assert np.isnan(loaded.landmarks[10]).all() and not loaded.usable[10]
        assert np.isfinite(loaded.landmarks[9]).all()

    def test_only_landmark_numbers_are_stored(self):
        """The record's fields are fixed: nothing that could hold video or an image."""
        obj = sequence_to_json(seq())
        assert set(obj) == {
            "record", "sequence_id", "session_id", "kind", "label", "subtype", "aspect_ratio",
            "t_ms", "usable", "confidence", "landmarks",
        }  # fmt: skip
        text = json.dumps(obj)
        assert "frame" not in text and "image" not in text and "video" not in text

    def test_the_file_has_one_header_line_then_one_line_per_sequence(self, tmp_path):
        path = write_session(tmp_path, [seq(1), seq(2)])
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["record"] == "session"
        assert json.loads(lines[0])["schema_version"] == SCHEMA_VERSION
        assert all(json.loads(line)["record"] == "sequence" for line in lines[1:])

    def test_the_header_holds_no_path_or_name(self, tmp_path):
        path = write_session(tmp_path, [seq()])
        first = path.read_text(encoding="utf-8").split("\n")[0]
        assert str(tmp_path) not in first


class TestWriter:
    def test_an_existing_file_is_never_overwritten(self, tmp_path):
        write_session(tmp_path, [seq()])
        with pytest.raises(DatasetError, match="refusing to overwrite"):
            SessionWriter(tmp_path, header())

    def test_an_invalid_sequence_is_refused_and_nothing_is_written(self, tmp_path):
        writer = SessionWriter(tmp_path, header())
        good, bad = seq(1), seq(2)
        bad_times = bad.t_ms.copy()
        bad_times[5] = bad_times[4]  # not strictly increasing
        broken = SequenceRecord(
            sequence_id=bad.sequence_id, session_id=SESSION, label=bad.label, t_ms=bad_times,
            landmarks=bad.landmarks, confidence=bad.confidence, usable=bad.usable,
            aspect_ratio=bad.aspect_ratio,
        )  # fmt: skip
        writer.append(good)
        with pytest.raises(DatasetError, match="strictly increase"):
            writer.append(broken)
        writer.close()
        assert len(load_session(writer.path).sequences) == 1

    def test_a_sequence_from_another_session_is_refused(self, tmp_path):
        writer = SessionWriter(tmp_path, header())
        with pytest.raises(DatasetError, match="different session"):
            writer.append(seq(session_id="other"))
        writer.close()

    def test_duplicate_sequence_ids_are_refused(self, tmp_path):
        writer = SessionWriter(tmp_path, header())
        writer.append(seq(1))
        with pytest.raises(DatasetError, match="duplicate"):
            writer.append(seq(1))
        writer.close()

    def test_delete_last_removes_exactly_the_latest_sequence(self, tmp_path):
        writer = SessionWriter(tmp_path, header())
        for n in (1, 2, 3):
            writer.append(seq(n))
        assert writer.delete_last() == f"{SESSION}-0003"
        writer.append(seq(4))  # the file is still appendable after a truncate
        writer.close()
        ids = [s.sequence_id for s in load_session(writer.path).sequences]
        assert ids == [f"{SESSION}-0001", f"{SESSION}-0002", f"{SESSION}-0004"]

    def test_delete_last_really_shortens_the_file(self, tmp_path):
        """With nothing appended afterwards, the deleted bytes must be gone, not just forgotten."""
        writer = SessionWriter(tmp_path, header())
        writer.append(seq(1))
        writer.append(seq(2))
        before = writer.path.stat().st_size
        writer.delete_last()
        writer.close()
        assert writer.path.stat().st_size < before
        assert [s.sequence_id for s in load_session(writer.path).sequences] == [f"{SESSION}-0001"]

    def test_delete_last_with_nothing_recorded(self, tmp_path):
        writer = SessionWriter(tmp_path, header())
        assert writer.delete_last() is None
        writer.close()

    def test_a_session_with_no_takes_leaves_no_file(self, tmp_path):
        writer = SessionWriter(tmp_path, header())
        writer.close()
        assert not writer.path.exists()

    def test_a_session_whose_takes_were_all_deleted_leaves_no_file(self, tmp_path):
        writer = SessionWriter(tmp_path, header())
        writer.append(seq(1))
        writer.delete_last()
        writer.close()
        assert not writer.path.exists()


class TestSequenceValidation:
    def test_a_good_sequence_has_no_problems(self):
        assert validate_sequence(seq()) == []

    @staticmethod
    def altered(base, **changes):
        fields = {
            "sequence_id": base.sequence_id, "session_id": base.session_id,
            "label": base.label, "t_ms": base.t_ms, "landmarks": base.landmarks,
            "confidence": base.confidence, "usable": base.usable,
            "aspect_ratio": base.aspect_ratio, "subtype": base.subtype, "kind": base.kind,
        }  # fmt: skip
        fields.update(changes)
        return SequenceRecord(**fields)

    @classmethod
    def problems_with(cls, base, **changes):
        return validate_sequence(cls.altered(base, **changes))

    def test_unknown_label(self):
        assert any("unknown label" in p for p in self.problems_with(seq(), label="wave"))

    def test_target_gestures_take_no_subtype(self):
        problems = validate_sequence(self.altered(seq(), subtype="idle"))
        assert any("must not have a subtype" in p for p in problems)

    def test_unknown_negative_subtype(self):
        problems = validate_sequence(self.altered(seq(), label="no_gesture", subtype="dancing"))
        assert any("subtype" in p for p in problems)

    def test_known_negative_subtype_is_fine(self):
        assert validate_sequence(self.altered(seq(), label="no_gesture", subtype="pinch")) == []

    def test_a_stream_must_be_a_continuous_use_negative(self):
        problems = validate_sequence(self.altered(seq(), kind="stream"))
        assert any("labelled no_gesture" in p for p in problems)

    def test_too_few_frames(self):
        short = make_sequence(swipe_path(1), duration=0.2, fps=30.0)
        assert any("frames" in p for p in validate_sequence(short))

    def test_timestamps_must_start_at_zero(self):
        base = seq()
        problems = validate_sequence(self.altered(base, t_ms=base.t_ms + 5))
        assert any("start at 0" in p for p in problems)

    def test_a_camera_stall_is_flagged(self):
        base = seq()
        t = base.t_ms.copy()
        t[20:] += 900.0
        assert any("stalled" in p for p in validate_sequence(self.altered(base, t_ms=t)))

    def test_non_finite_timestamps(self):
        base = seq()
        t = base.t_ms.copy()
        t[3] = np.nan
        assert any("NaN" in p for p in validate_sequence(self.altered(base, t_ms=t)))

    def test_partly_missing_landmarks(self):
        base = seq()
        lm = base.landmarks.copy()
        lm[7, 4, 0] = np.nan
        assert any("partly missing" in p for p in self.problems_with(base, landmarks=lm))

    def test_infinite_and_absurd_landmark_values(self):
        base = seq()
        lm = base.landmarks.copy()
        lm[2, 0, 0] = np.inf
        assert any("infinity" in p for p in validate_sequence(self.altered(base, landmarks=lm)))
        lm = base.landmarks.copy()
        lm[2, 0, 0] = 99.0
        assert any("implausibly" in p for p in validate_sequence(self.altered(base, landmarks=lm)))

    def test_usable_frame_without_landmarks(self):
        base = seq()
        lm = base.landmarks.copy()
        lm[4] = np.nan
        assert any("marked usable" in p for p in self.problems_with(base, landmarks=lm))

    def test_confidence_outside_zero_one(self):
        base = seq()
        conf = base.confidence.copy()
        conf[0] = 1.5
        assert any("confidence" in p for p in self.problems_with(base, confidence=conf))

    def test_aspect_ratio_range(self):
        assert any("aspect_ratio" in p for p in self.problems_with(seq(), aspect_ratio=0.0))

    def test_fewer_than_two_usable_frames(self):
        base = seq()
        none = np.zeros(base.n_frames, dtype=bool)
        lm = np.full_like(base.landmarks, np.nan)
        problems = validate_sequence(self.altered(base, usable=none, landmarks=lm))
        assert any("usable" in p for p in problems)

    def test_array_shapes_must_agree(self):
        base = seq()
        with pytest.raises(ValueError, match="one entry per frame"):
            self.altered(base, confidence=base.confidence[:-1])
        with pytest.raises(ValueError, match="shape"):
            self.altered(base, landmarks=base.landmarks[:, :20])


class TestMalformedFiles:
    def test_a_truncated_last_line_is_reported_with_its_line_number(self, tmp_path):
        path = write_session(tmp_path, [seq(1), seq(2)])
        edit_lines(path, lambda lines: [*lines[:-2], lines[-2][:200]])
        with pytest.raises(DatasetValidationError, match=r"sess\.jsonl:3.*invalid JSON"):
            load_session(path)

    def test_lenient_loading_skips_the_bad_line_and_keeps_the_rest(self, tmp_path):
        path = write_session(tmp_path, [seq(1), seq(2), seq(3)])
        edit_lines(path, lambda lines: [lines[0], lines[1], "{not json", *lines[3:]])
        loaded = load_session(path, strict=False)
        assert [s.sequence_id for s in loaded.sequences] == [f"{SESSION}-0001", f"{SESSION}-0003"]
        assert [(i.line, i.severity) for i in loaded.issues] == [(3, "error")]

    def test_wrong_landmark_shape(self, tmp_path):
        obj = sequence_to_json(seq())
        obj["landmarks"][5] = [[0.1, 0.2, 0.3]] * 20
        with pytest.raises(Exception, match="21 x 3"):
            sequence_from_json(obj)

    @pytest.mark.parametrize("mutate, message", [
        (lambda o: o.pop("label"), "missing field 'label'"),
        (lambda o: o.update(t_ms=o["t_ms"][:-1]), "entries"),
        (lambda o: o.update(usable=[1] * len(o["usable"])), "true/false"),
        (lambda o: o.update(t_ms="soon"), "list of numbers"),
        (lambda o: o.update(landmarks="none"), "one entry per frame"),
        (lambda o: o.update(subtype=7), "subtype"),
    ])
    def test_structural_errors_name_the_problem(self, mutate, message):
        obj = sequence_to_json(seq())
        mutate(obj)
        with pytest.raises(Exception, match=message):
            sequence_from_json(obj)

    def test_a_file_without_a_header(self, tmp_path):
        path = write_session(tmp_path, [seq()])
        edit_lines(path, lambda lines: lines[1:])
        with pytest.raises(DatasetError, match="first line must be the session header"):
            load_session(path)

    def test_an_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(DatasetError, match="empty file"):
            load_session(path)

    def test_an_unsupported_schema_version(self, tmp_path):
        path = write_session(tmp_path, [seq()])
        edit_lines(path, lambda lines: [
            lines[0].replace(f'"schema_version":{SCHEMA_VERSION}', '"schema_version":99'),
            *lines[1:],
        ])
        with pytest.raises(DatasetError, match="unsupported schema_version"):
            load_session(path)

    def test_a_file_that_is_not_text(self, tmp_path):
        path = tmp_path / "binary.jsonl"
        path.write_bytes(b"\xff\xfe\x00\x01")
        with pytest.raises(DatasetError, match="UTF-8"):
            load_session(path)

    def test_a_sequence_from_a_different_session_in_the_file(self, tmp_path):
        path = write_session(tmp_path, [seq(1)])
        foreign = json.dumps(sequence_to_json(seq(2, session_id="other")), separators=(",", ":"))
        path.write_text(path.read_text(encoding="utf-8") + foreign + "\n", encoding="utf-8")
        loaded = load_session(path, strict=False)
        assert len(loaded.sequences) == 1
        assert any("belongs to session" in i.message for i in loaded.issues)

    def test_a_duplicate_sequence_id_in_one_file(self, tmp_path):
        path = write_session(tmp_path, [seq(1)])
        line = path.read_text(encoding="utf-8").split("\n")[1]
        path.write_text(path.read_text(encoding="utf-8") + line + "\n", encoding="utf-8")
        loaded = load_session(path, strict=False)
        assert len(loaded.sequences) == 1
        assert any("duplicate sequence_id" in i.message for i in loaded.issues)

    def test_blank_lines_in_the_middle_are_warned_about_not_fatal(self, tmp_path):
        path = write_session(tmp_path, [seq(1), seq(2)])
        edit_lines(path, lambda lines: [lines[0], "", *lines[1:]])
        loaded = load_session(path)
        assert len(loaded.sequences) == 2
        assert [i.severity for i in loaded.issues] == ["warning"]

    def test_windows_line_endings_are_accepted(self, tmp_path):
        path = write_session(tmp_path, [seq(1)])
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        assert len(load_session(path).sequences) == 1


class TestDirectory:
    def test_missing_directory(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path / "nope")

    def test_empty_directory_is_an_empty_dataset(self, tmp_path):
        assert load_dataset(tmp_path).sessions == []

    def test_loads_every_session_in_name_order(self, tmp_path):
        write_session(tmp_path, [seq(1, session_id="sess-b")], session_id="sess-b")
        write_session(tmp_path, [seq(1, session_id="sess-a")], session_id="sess-a")
        dataset = load_dataset(tmp_path)
        assert [s.header.session_id for s in dataset.sessions] == ["sess-a", "sess-b"]
        assert len(list(dataset.sequences())) == 2

    def test_strict_loading_raises_but_lenient_loading_reports(self, tmp_path):
        write_session(tmp_path, [seq(1, session_id="sess-a")], session_id="sess-a")
        (tmp_path / "sess-b.jsonl").write_text("garbage\n", encoding="utf-8")
        with pytest.raises(DatasetError):
            load_dataset(tmp_path)
        dataset = load_dataset(tmp_path, strict=False)
        assert [s.header.session_id for s in dataset.sessions] == ["sess-a"]
        assert any(i.file == "sess-b.jsonl" and i.severity == "error" for i in dataset.issues)

    def test_the_same_session_id_in_two_files(self, tmp_path):
        path = write_session(tmp_path, [seq(1)])
        (tmp_path / "copy.jsonl").write_bytes(path.read_bytes())
        dataset = load_dataset(tmp_path, strict=False)
        assert len(dataset.sessions) == 1
        assert any("already used" in i.message for i in dataset.errors)

    def test_copied_takes_are_flagged_as_identical_content(self, tmp_path):
        write_session(tmp_path, [seq(1), seq(2)])  # same synthetic motion twice
        dataset = load_dataset(tmp_path)
        assert any("identical landmark data" in w.message for w in dataset.warnings)

    def test_issue_text_names_the_file_not_the_directory(self, tmp_path):
        (tmp_path / "bad.jsonl").write_text("garbage\n", encoding="utf-8")
        issue = load_dataset(tmp_path, strict=False).issues[0]
        assert str(tmp_path) not in str(issue)
