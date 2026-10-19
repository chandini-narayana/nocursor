"""Session-grouped splitting: determinism, leakage prevention, coverage, freezing and isolation.

All data is SYNTHETIC (see ml_fixtures); these tests check the software, not any gesture result.
"""

import copy
import json
import shutil

import pytest

from ml_fixtures import (
    FULL_SESSIONS,
    PILOT,
    STREAM,
    build_study,
    write_session,
    write_stream_session,
)
from nocursor.dynamic import splits as splits_module
from nocursor.dynamic.dataset import Dataset, load_dataset
from nocursor.dynamic.features import FeatureConfig
from nocursor.dynamic.schema import CLASS_NAMES, SequenceRecord
from nocursor.dynamic.splits import (
    SplitConfig,
    SplitCoverageError,
    SplitError,
    check_files,
    check_manifest,
    create_split,
    load_split_items,
    load_stream,
    read_split,
    verify_manifest_and_files,
    verify_split,
    write_split,
)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """Built once and shared read-only: generating hundreds of synthetic sequences is slow."""
    return build_study(tmp_path_factory.mktemp("shared") / "datasets")


@pytest.fixture
def study_copy(study, tmp_path):
    """A private copy, for tests that edit dataset files."""
    target = tmp_path / "datasets"
    shutil.copytree(study, target)
    return target


def make(study, seed=0, **config):
    dataset = load_dataset(study)
    return dataset, create_split(dataset, SplitConfig(seed=seed, **config), FeatureConfig(),
                                 created_utc="2026-10-13T00:00:00Z")  # fmt: skip


@pytest.fixture(scope="module")
def made(study):
    """The default split of the shared study (tests must not mutate it; copy first)."""
    return make(study)


class TestAssignment:
    def test_the_same_data_and_seed_always_give_the_same_split(self, study):
        _, first = make(study)
        _, second = make(study)
        assert first == second

    def test_the_seed_changes_the_assignment(self, study):
        _, base = make(study, seed=0)
        others = [make(study, seed=s)[1] for s in range(1, 4)]
        assert any(o["assignments"] != base["assignments"] for o in others)
        assert all(o["split_id"] != base["split_id"] for o in others
                   if o["assignments"] != base["assignments"])  # fmt: skip

    def test_input_order_does_not_matter(self, study, made):
        dataset, base = made
        reversed_dataset = Dataset(list(reversed(dataset.sessions)), dataset.issues)
        again = create_split(reversed_dataset, SplitConfig(), FeatureConfig(),
                             created_utc="2026-10-13T00:00:00Z")  # fmt: skip
        assert again == base

    def test_roles_follow_the_plan(self, study, made):
        _, split = made
        a = split["assignments"]
        assert len(a["test"]) == 1 and len(a["validation"]) == 1 and len(a["train"]) == 4
        assert a["pilot_train"] == [PILOT] and a["reserved_stream"] == [STREAM]
        assert sorted(a["train"] + a["validation"] + a["test"]) == sorted(FULL_SESSIONS)

    def test_every_session_is_in_exactly_one_role(self, study, made):
        _, split = made
        everything = [sid for ids in split["assignments"].values() for sid in ids]
        assert sorted(everything) == sorted([*FULL_SESSIONS, PILOT, STREAM])
        assert len(everything) == len(set(everything))

    def test_splits_are_made_of_whole_sessions(self, study, made):
        """Every take of a session lands in the same split, and no split shares a take."""
        _, split = made
        runs = study.parent / "runs"
        write_split(split, runs)
        by_split = {n: load_split_items(split, study, [n], allow_test=True)
                    for n in ("train", "validation", "test")}  # fmt: skip
        seen = {}
        for name, items in by_split.items():
            for item in items:
                assert seen.setdefault(item.session_id, name) == name
        ids = [i.sequence.sequence_id for items in by_split.values() for i in items]
        assert len(ids) == len(set(ids))

    def test_pilots_only_train_and_the_stream_is_in_no_split(self, study, made):
        _, split = made
        train = load_split_items(split, study, ["train"])
        assert PILOT in {i.session_id for i in train}
        assert STREAM not in {i.session_id for i in train}
        held = load_split_items(split, study, ["validation", "test"], allow_test=True)
        assert PILOT not in {i.session_id for i in held}
        assert all(i.sequence.kind == "take" for i in train + held)

    def test_all_four_classes_are_in_every_split(self, study, made):
        _, split = made
        for counts in split["counts"].values():
            assert all(counts[c] >= 1 for c in CLASS_NAMES)

    def test_the_split_id_reflects_the_data(self, tmp_path):
        study_a = build_study(tmp_path / "a", takes=10)
        study_b = build_study(tmp_path / "b", takes=11)
        assert make(study_a)[1]["split_id"] != make(study_b)[1]["split_id"]

    def test_the_manifest_records_the_methodology(self, study, made):
        _, split = made
        assert split["seed"] == 0 and split["algorithm"]
        assert len(split["dataset_manifest_hash"]) == 64
        assert split["feature_schema_hash"] == FeatureConfig().schema_hash()
        assert {r["session_id"] for r in split["sessions"]} == {*FULL_SESSIONS, PILOT, STREAM}
        assert split["reserved_stream"][0]["session_id"] == STREAM
        assert all(r["provenance"] == "synthetic" for r in split["sessions"])
        assert any("single participant" in n for n in split["notes"])


class TestExclusionsAndFlags:
    def test_takes_that_fail_feature_extraction_are_excluded_with_a_reason(self, study, made):
        _, split = made
        assert len(split["excluded"]) == 2
        assert all(e["reason"].startswith("hand trusted in only") for e in split["excluded"])
        assert all(e["label"] == "no_gesture" and e["subtype"] == "entering_leaving"
                   for e in split["excluded"])  # fmt: skip
        raw = sum(sum(c.values()) for c in split["counts_before_exclusion"].values())
        kept = sum(sum(c.values()) for c in split["counts"].values())
        assert raw - kept == 2

    def test_excluded_takes_are_never_loaded(self, study, made):
        _, split = made
        excluded = {e["sequence_id"] for e in split["excluded"]}
        items = load_split_items(split, study, ["train", "validation"])
        loaded_all = {i.sequence.sequence_id for i in items}
        loaded_all |= {i.sequence.sequence_id
                       for i in load_split_items(split, study, ["test"], allow_test=True)}
        assert not excluded & loaded_all

    def test_flagged_takes_are_recorded_with_their_treatment_and_kept(self, study, made):
        _, split = made
        assert len(split["flagged"]) == 1
        flagged = split["flagged"][0]
        assert flagged["note"].startswith("sweep is small")
        assert "not relabelled" in flagged["treatment"] and "removed" in flagged["treatment"]
        items = load_split_items(split, study, ["train", "validation"])
        items += load_split_items(split, study, ["test"], allow_test=True)
        kept = [i for i in items if i.flagged]
        assert [i.sequence.sequence_id for i in kept] == [flagged["sequence_id"]]
        assert kept[0].sequence.label == "swipe_right"  # the label was not touched

    def test_the_sensitivity_option_removes_only_the_flagged_take(self, study, made):
        _, split = made
        flagged = split["flagged"][0]["sequence_id"]
        names = ["train", "validation", "test"]
        everything = load_split_items(split, study, names, allow_test=True)
        without = load_split_items(split, study, names, allow_test=True, exclude_flagged=True)
        assert {i.sequence.sequence_id for i in everything} - {
            i.sequence.sequence_id for i in without} == {flagged}  # fmt: skip


class TestCoverage:
    def test_too_few_full_sessions(self, tmp_path):
        for i in range(2):
            write_session(tmp_path, f"s-full-{i}", i)
        with pytest.raises(SplitCoverageError, match="need 3"):
            create_split(load_dataset(tmp_path))

    def test_pilots_do_not_count_as_full_sessions(self, tmp_path):
        for i in range(2):
            write_session(tmp_path, f"s-full-{i}", i)
        write_session(tmp_path, "s-pilot-x", 7, takes=3, negatives=3)
        with pytest.raises(SplitCoverageError, match="found 2 full"):
            create_split(load_dataset(tmp_path))

    def test_a_dataset_with_no_sessions(self, tmp_path):
        with pytest.raises(SplitCoverageError, match="no sessions"):
            create_split(load_dataset(tmp_path))

    def test_a_class_that_vanishes_after_exclusions_is_an_error(self, tmp_path):
        for i in range(3):
            write_session(tmp_path, f"s-full-{i}", i, negatives=0, bad_negatives=12)
        with pytest.raises(SplitCoverageError, match="no no_gesture takes after exclusions"):
            create_split(load_dataset(tmp_path), SplitConfig(min_heldout_per_class=1))

    def test_too_few_takes_in_a_held_out_split(self, tmp_path):
        for i in range(3):
            write_session(tmp_path, f"s-full-{i}", i, takes=10, negatives=3)
        with pytest.raises(SplitCoverageError, match="only 3 no_gesture takes"):
            create_split(load_dataset(tmp_path))

    def test_small_held_out_splits_get_a_warning(self, tmp_path):
        for i in range(3):
            write_session(tmp_path, f"s-full-{i}", i, takes=10, negatives=12)
        warnings = create_split(load_dataset(tmp_path))["warnings"]
        assert any("indicative only" in w for w in warnings)

    def test_a_session_mixing_takes_and_a_stream_is_refused(self, tmp_path):
        for i in range(3):
            write_session(tmp_path, f"s-full-{i}", i)
        dataset = load_dataset(tmp_path)
        session = dataset.sessions[0]
        take = session.sequences[0]
        session.sequences.append(SequenceRecord(
            f"{session.header.session_id}-9999", session.header.session_id, "no_gesture",
            take.t_ms, take.landmarks, take.confidence, take.usable, take.aspect_ratio,
            subtype="continuous_use", kind="stream"))  # fmt: skip
        with pytest.raises(SplitError, match="mixes takes and a continuous stream"):
            create_split(dataset)

    @pytest.mark.parametrize("kwargs", [{"validation_sessions": 0}, {"test_sessions": 0},
                                        {"min_full_takes": 0}])  # fmt: skip
    def test_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            SplitConfig(**kwargs)


class TestVerification:
    def test_a_fresh_split_verifies(self, study, made):
        dataset, split = made
        verify_split(split, dataset)
        verify_manifest_and_files(split, study)
        assert check_manifest(split) == [] and check_files(split, study) == []

    def tampered(self, split, edit):
        clone = copy.deepcopy(split)
        edit(clone)
        return clone

    def test_a_session_in_two_roles_is_caught(self, study, made):
        dataset, split = made
        bad = self.tampered(split, lambda s: s["assignments"]["validation"].append(
            s["assignments"]["test"][0]))  # fmt: skip
        with pytest.raises(SplitError, match="assigned to both"):
            verify_split(bad, dataset)

    def test_a_pilot_in_validation_is_caught(self, study, made):
        dataset, split = made

        def edit(s):
            s["assignments"]["pilot_train"].remove(PILOT)
            s["assignments"]["validation"].append(PILOT)

        with pytest.raises(SplitError):
            verify_split(self.tampered(split, edit), dataset)

    def test_the_stream_in_a_split_is_caught(self, study, made):
        dataset, split = made

        def edit(s):
            s["assignments"]["train"].append(STREAM)

        with pytest.raises(SplitError, match="assigned to both|also in a split"):
            verify_split(self.tampered(split, edit), dataset)

    def test_a_missing_class_is_caught(self, study, made):
        dataset, split = made
        bad = self.tampered(split, lambda s: s["counts"]["test"].update(swipe_left=0))
        with pytest.raises(SplitError, match="no swipe_left"):
            verify_split(bad, dataset)

    def test_a_changed_counts_table_no_longer_matches_a_fresh_computation(self, study, made):
        dataset, split = made
        bad = self.tampered(split, lambda s: s["counts"]["train"].update(swipe_left=999))
        with pytest.raises(SplitError, match="counts no longer matches"):
            verify_split(bad, dataset)

    def test_a_dataset_file_that_changed_is_caught(self, study_copy):
        dataset, split = make(study_copy)
        path = study_copy / f"{FULL_SESSIONS[0]}.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises(SplitError, match="changed since"):
            verify_manifest_and_files(split, study_copy)
        with pytest.raises(SplitError, match="changed"):
            load_split_items(split, study_copy, ["train", "validation", "test"], allow_test=True)

    def test_a_new_session_file_is_caught(self, study_copy):
        _, split = make(study_copy)
        write_session(study_copy, "s-extra", 11)
        with pytest.raises(SplitError, match="not covered by the split"):
            verify_manifest_and_files(split, study_copy)

    def test_a_missing_session_file_is_caught(self, study_copy):
        _, split = make(study_copy)
        (study_copy / f"{FULL_SESSIONS[0]}.jsonl").unlink()
        with pytest.raises(SplitError, match="missing"):
            verify_manifest_and_files(split, study_copy)

    def test_identical_landmark_data_across_splits_is_caught(self, study):
        """A take copied from a training session into the test session would leak."""
        dataset, split = make(study)
        train_session = next(s for s in dataset.sessions
                             if s.header.session_id == split["assignments"]["train"][0])
        test_session = next(s for s in dataset.sessions
                            if s.header.session_id == split["assignments"]["test"][0])
        original = train_session.sequences[0]
        test_session.sequences.append(SequenceRecord(
            "s-copied-0001", test_session.header.session_id, original.label, original.t_ms,
            original.landmarks, original.confidence, original.usable,
            original.aspect_ratio))
        with pytest.raises(SplitError, match="identical landmark data"):
            verify_split(split, dataset)

    def test_the_manifest_check_alone_catches_a_pilot_listed_as_held_out(self, made):
        """What training relies on: it never parses data, so the manifest must be self-checking."""
        _, split = made
        both = self.tampered(split, lambda s: s["assignments"]["validation"].append(PILOT))
        assert any("assigned to both" in p for p in check_manifest(both))
        assert any("pilot session is in validation or test" in p for p in check_manifest(both))

    def test_the_manifest_check_alone_catches_a_pilot_moved_into_validation(self, made):
        _, split = made

        def move(s):
            s["assignments"]["pilot_train"].remove(PILOT)
            s["assignments"]["validation"].append(PILOT)

        problems = check_manifest(self.tampered(split, move))
        assert any("too small to be a full session" in p for p in problems)

    def test_an_unknown_split_version(self, study, made):
        _, split = made
        bad = self.tampered(split, lambda s: s.update(split_version=99))
        assert "unsupported split_version" in check_manifest(bad)[0]


class TestFreezingAndReading:
    def test_writing_twice_leaves_the_frozen_file_alone(self, study, tmp_path, made):
        _, split = made
        runs = tmp_path / "runs"
        path, first = write_split(split, runs)
        before = path.read_bytes()
        again, second = write_split(split, runs)
        assert (first, second, again) == (True, False, path)
        assert path.read_bytes() == before

    def test_an_edited_frozen_file_is_not_overwritten(self, study, tmp_path, made):
        _, split = made
        path, _ = write_split(split, tmp_path / "runs")
        edited = json.loads(path.read_text(encoding="utf-8"))
        edited["assignments"]["test"] = ["s-full-a"]
        path.write_text(json.dumps(edited), encoding="utf-8")
        with pytest.raises(SplitError, match="not overwriting a frozen split"):
            write_split(split, tmp_path / "runs")

    def test_read_by_id_and_by_path(self, study, tmp_path, made):
        _, split = made
        runs = tmp_path / "runs"
        path, _ = write_split(split, runs)
        by_id, found = read_split(split["split_id"], runs)
        by_path, _ = read_split(str(path), runs)
        assert by_id == by_path == split and found == path

    def test_reading_a_missing_or_damaged_split(self, tmp_path):
        with pytest.raises(SplitError, match="not found"):
            read_split("split-nope", tmp_path)
        bad = tmp_path / "split.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(SplitError, match="cannot read"):
            read_split(str(bad), tmp_path)


class TestIsolation:
    def test_the_test_split_cannot_be_loaded_without_the_explicit_flag(self, study, made):
        _, split = made
        with pytest.raises(SplitError, match="reserved for evaluation"):
            load_split_items(split, study, ["test"])
        with pytest.raises(SplitError, match="reserved for evaluation"):
            load_split_items(split, study, ["train", "validation", "test"])

    def test_loading_train_and_validation_never_opens_test_or_stream_files(
        self, study, monkeypatch, made
    ):
        _, split = made
        opened = []
        real = splits_module.load_session
        monkeypatch.setattr(splits_module, "load_session",
                            lambda path, **kw: opened.append(path.stem) or real(path, **kw))
        load_split_items(split, study, ["train", "validation"])
        forbidden = {*split["assignments"]["test"], STREAM}
        assert opened and not forbidden & set(opened)

    def test_the_stream_is_only_returned_by_its_own_loader(self, study, made):
        _, split = made
        streams = load_stream(split, study)
        assert len(streams) == 1 and streams[0].kind == "stream"
        everything = load_split_items(split, study, ["train", "validation", "test"],
                                      allow_test=True)  # fmt: skip
        assert all(i.sequence.kind == "take" for i in everything)

    def test_the_stream_session_is_found_by_the_split_and_verified_by_hash(self, study_copy):
        _, split = make(study_copy)
        write_stream_session(study_copy, "s-stream-b")  # a new recording appears after the split
        with pytest.raises(SplitError, match="not covered|changed"):
            verify_manifest_and_files(split, study_copy)
