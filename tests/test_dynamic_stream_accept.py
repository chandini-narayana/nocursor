"""Freeze, one-shot acceptance, blind review, the CLI, and isolation from the desktop controls.

All data is SYNTHETIC geometry; these tests check software behaviour, not recognition accuracy.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dynamic_helpers import circle_path, make_sequence, swipe_path
from ml_fixtures import add_random_gru, build_study, freeze_split
from nocursor.dynamic.cli import main
from nocursor.dynamic.gate import GateConfig
from nocursor.dynamic.splits import verify_manifest_and_files
from nocursor.dynamic.stabilizer import StabilizerConfig
from nocursor.dynamic.stream_accept import (
    ITEMS_FILE,
    KEY_FILE,
    MARKS_FILE,
    RUN_FILE,
    STARTED_FILE,
    VERDICT_FILE,
    AcceptanceCriteria,
    AcceptanceDataError,
    AcceptanceError,
    FreezeError,
    ReviewIncompleteError,
    accept_finalize,
    accept_run,
    build_review,
    check_data_requirements,
    confirmed_false_counts,
    create_freeze,
    judge,
    load_final,
    load_freeze,
    source_hashes,
    verify_freeze,
)
from nocursor.dynamic.stream_eval import DecisionConfig
from nocursor.dynamic.train import TrainConfig, run_fit_baseline
from test_dynamic_stream_eval import build_stream, one_swipe, write_session

SRC = Path(__file__).resolve().parents[1] / "src"
RULE = DecisionConfig("rule_defaults", GateConfig(0.0), StabilizerConfig(cooldown_s=2.0))
RELAXED = AcceptanceCriteria(min_minutes=0.5, min_recordings=2, min_days=2, min_conditions=2,
                             min_protocol_sessions=1, min_takes_per_gesture=2)  # fmt: skip


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("stream")
    dataset_dir = build_study(root / "datasets", takes=10)
    runs_dir = root / "ml_runs"
    split, split_file = freeze_split(dataset_dir, runs_dir)
    primary = run_fit_baseline(split, split_file, dataset_dir, runs_dir, "primary",
                               TrainConfig(allow_synthetic=True), log=lambda _: None)  # fmt: skip
    add_random_gru(primary, split, seed=1)
    return {"root": root, "dataset": dataset_dir, "runs": runs_dir, "run": primary,
            "split": split}  # fmt: skip


@pytest.fixture
def work(prepared, tmp_path):
    """A private copy of the datasets and runs, plus a fake package root for source hashing."""
    shutil.copytree(prepared["dataset"], tmp_path / "datasets")
    shutil.copytree(prepared["runs"], tmp_path / "ml_runs")
    package = tmp_path / "pkg"
    (package / "dynamic").mkdir(parents=True)
    (package / "dynamic" / "a.py").write_text("A = 1\n", encoding="utf-8")
    (package / "scroll.py").write_text("S = 1\n", encoding="utf-8")
    (package / "models.py").write_text("M = 1\n", encoding="utf-8")
    return {"dataset": tmp_path / "datasets", "runs": tmp_path / "ml_runs",
            "run": tmp_path / "ml_runs" / "primary", "package": package,
            "final": tmp_path / "datasets" / "streaming" / "final"}  # fmt: skip


def freeze_for(work, decisions=(RULE,), primary="rule_defaults", criteria=RELAXED):
    return create_freeze(work["runs"], work["run"], list(decisions), primary, work["final"],
                         criteria=criteria, package_root=work["package"])  # fmt: skip


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------- freeze
class TestFreeze:
    def test_it_records_the_model_settings_criteria_and_source_hashes(self, work):
        freeze = load_freeze(freeze_for(work).parent)
        assert freeze["freeze_id"].startswith("freeze-")
        assert freeze["primary_system"] == "rule_defaults"
        entry = freeze["decisions"]["rule_defaults"]
        assert entry["config_hash"] == RULE.hash() and entry["model"]["kind"] == "rule"
        assert freeze["criteria"] == RELAXED.to_dict() and freeze["criteria_hash"] == RELAXED.hash()
        assert set(freeze["source_hashes"]) == {"dynamic/a.py", "scroll.py", "models.py"}
        assert freeze["grid_hash"] and freeze["feature_config"]["window_s"] == 1.6

    def test_the_real_package_hashes_cover_the_streaming_code_and_the_pose_measure(self):
        hashes = source_hashes()
        for name in ("dynamic/stream_eval.py", "dynamic/stabilizer.py", "dynamic/gate.py",
                     "dynamic/stream_accept.py", "scroll.py", "models.py"):  # fmt: skip
            assert name in hashes and len(hashes[name]) == 64

    def test_a_gru_decision_records_the_weights_hash(self, work):
        gru = DecisionConfig("gru", GateConfig(0.8), StabilizerConfig(tau=0.9))
        freeze = load_freeze(freeze_for(work, [gru], "gru").parent)
        weights = json.loads((work["run"] / "model.json").read_text("utf-8"))["weights_sha256"]
        assert freeze["decisions"]["gru"]["model"] == {"kind": "gru", "weights_sha256": weights}

    def test_it_refuses_once_final_recordings_exist(self, work):
        work["final"].mkdir(parents=True)
        (work["final"] / "s-early.jsonl").write_text("x", encoding="utf-8")
        with pytest.raises(FreezeError, match="before any final recording"):
            freeze_for(work)

    def test_an_empty_or_missing_final_folder_is_fine(self, work):
        work["final"].mkdir(parents=True)
        assert freeze_for(work).is_file()

    def test_a_freeze_is_never_overwritten(self, work):
        freeze_for(work)
        with pytest.raises(FreezeError, match="never overwritten"):
            freeze_for(work)

    def test_different_settings_give_a_different_freeze(self, work):
        other = DecisionConfig("rule_defaults", GateConfig(0.8), StabilizerConfig(cooldown_s=2.0))
        assert freeze_for(work).parent != freeze_for(work, [other]).parent

    def test_the_primary_must_be_frozen(self, work):
        with pytest.raises(FreezeError):
            freeze_for(work, primary="gru")

    def test_one_setting_per_system(self, work):
        with pytest.raises(FreezeError):
            freeze_for(work, [RULE, RULE])

    def test_an_edited_freeze_is_detected(self, work):
        path = freeze_for(work)
        data = json.loads(path.read_text("utf-8"))
        data["decisions"]["rule_defaults"]["stabilizer"]["cooldown_s"] = 0.0
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(FreezeError, match="edited"):
            load_freeze(path.parent)

    def test_a_missing_freeze_is_an_error(self, tmp_path):
        with pytest.raises(FreezeError):
            load_freeze(tmp_path)


class TestVerifyFreeze:
    def frozen(self, work):
        return load_freeze(freeze_for(work).parent)

    def test_unchanged_code_and_model_verify(self, work):
        verify_freeze(self.frozen(work), work["run"], work["package"])

    def test_a_changed_source_file_is_detected(self, work):
        freeze = self.frozen(work)
        (work["package"] / "dynamic" / "a.py").write_text("A = 2\n", encoding="utf-8")
        with pytest.raises(FreezeError, match=r"dynamic/a\.py changed"):
            verify_freeze(freeze, work["run"], work["package"])

    def test_the_pose_measure_is_part_of_the_freeze(self, work):
        freeze = self.frozen(work)
        (work["package"] / "scroll.py").write_text("S = 2\n", encoding="utf-8")
        with pytest.raises(FreezeError, match=r"scroll\.py changed"):
            verify_freeze(freeze, work["run"], work["package"])

    def test_an_added_source_file_is_detected(self, work):
        freeze = self.frozen(work)
        (work["package"] / "dynamic" / "b.py").write_text("B = 1\n", encoding="utf-8")
        with pytest.raises(FreezeError, match="added"):
            verify_freeze(freeze, work["run"], work["package"])

    def test_a_changed_baseline_is_detected(self, work):
        freeze = self.frozen(work)
        baseline = work["run"] / "baseline.json"
        baseline.write_text(baseline.read_text("utf-8") + " ", encoding="utf-8")
        with pytest.raises(FreezeError, match="rule_defaults model or baseline changed"):
            verify_freeze(freeze, work["run"], work["package"])

    def test_changed_gru_weights_are_detected(self, work):
        gru = DecisionConfig("gru", GateConfig(0.8), StabilizerConfig(tau=0.9))
        freeze = load_freeze(freeze_for(work, [gru], "gru").parent)
        meta_path = work["run"] / "model.json"
        meta = json.loads(meta_path.read_text("utf-8"))
        meta["weights_sha256"] = "0" * 64
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(FreezeError, match="gru model or baseline changed"):
            verify_freeze(freeze, work["run"], work["package"])


class TestCriteria:
    def test_the_defaults_are_the_approved_acceptance_criteria(self):
        c = AcceptanceCriteria()
        assert (c.min_minutes, c.min_recordings, c.min_days) == (40.0, 4, 2)
        assert c.max_confirmed_false == 0 and c.upper_bound_confidence == 0.95
        assert (c.min_pooled_recall, c.min_class_recall, c.max_cross_activations) == (0.85, 0.80, 2)
        assert c.min_protocol_sessions == 2 and c.min_takes_per_gesture == 30

    def test_the_hash_depends_on_every_criterion(self):
        base = AcceptanceCriteria()
        assert len({base.hash(), AcceptanceCriteria(min_minutes=30).hash(),
                    AcceptanceCriteria(max_confirmed_false=1).hash(),
                    AcceptanceCriteria(min_pooled_recall=0.8).hash()}) == 4  # fmt: skip


# ---------------------------------------------------------------------------- the review
def rows(*items):
    return [{"recording_id": r, "time_s": t, "label": label} for r, t, label in items]


class TestBuildReview:
    def test_the_same_movement_found_by_two_systems_is_one_blind_item(self):
        items, key = build_review(
            {"gru": rows(("r1", 10.0, "swipe_left")),
             "rule_tuned": rows(("r1", 10.4, "swipe_left"))}, 1.6, seed=1)  # fmt: skip
        assert len(items) == 1 and key[items[0]["item_id"]] == ["gru", "rule_tuned"]

    def test_items_do_not_name_any_system(self):
        items, _ = build_review({"gru": rows(("r1", 3.0, "swipe_left")),
                                 "rule_tuned": rows(("r1", 30.0, "circle_clockwise"))}, 1.6, 7)
        text = json.dumps(items)
        assert "gru" not in text and "rule" not in text

    def test_different_classes_recordings_or_times_are_separate_items(self):
        items, _ = build_review({"gru": rows(
            ("r1", 10.0, "swipe_left"), ("r1", 10.2, "swipe_right"),
            ("r2", 10.0, "swipe_left"), ("r1", 12.5, "swipe_left"))}, 1.6, 1)  # fmt: skip
        assert len(items) == 4

    def test_the_order_is_shuffled_deterministically_by_the_seed(self):
        data = {"gru": rows(*[("r1", 10.0 * i, "swipe_left") for i in range(1, 9)])}
        a, _ = build_review(data, 1.6, seed=3)
        b, _ = build_review(data, 1.6, seed=3)
        c, _ = build_review(data, 1.6, seed=4)
        assert a == b and a != c
        assert [i["item_id"] for i in a] == [f"c{n:03d}" for n in range(1, 9)]
        assert [i["from_s"] for i in a] != sorted(i["from_s"] for i in a)

    def test_the_replay_window_covers_the_candidate(self):
        (item,), _ = build_review({"gru": rows(("r1", 10.0, "swipe_left"))}, 1.6, 1)
        assert item["from_s"] == pytest.approx(8.4) and item["to_s"] == pytest.approx(10.5)
        (early,), _ = build_review({"gru": rows(("r1", 1.0, "swipe_left"))}, 1.6, 1)
        assert early["from_s"] == 0.0

    def test_no_candidates_means_no_items(self):
        assert build_review({"gru": []}, 1.6, 1) == ([], {})


class TestConfirmedCounts:
    items = [{"item_id": "c001", "recording_id": "r1"}, {"item_id": "c002", "recording_id": "r1"},
             {"item_id": "c003", "recording_id": "r2"}, {"item_id": "c004", "recording_id": "r2"}]
    key = {"c001": ["gru"], "c002": ["gru", "rule_tuned"], "c003": ["gru"], "c004": ["rule_tuned"]}

    def test_ambiguous_counts_as_false_and_is_also_reported_apart(self):
        marks = {"c001": "false", "c002": "ambiguous", "c003": "genuine", "c004": "false"}
        counts = confirmed_false_counts(self.key, self.items, marks, "gru")
        assert counts["false"] == 1 and counts["ambiguous"] == 1 and counts["genuine"] == 1
        assert counts["confirmed_false_for_acceptance"] == 2
        assert counts["per_recording"] == {"r1": {"false": 1, "ambiguous": 1, "genuine": 0},
                                           "r2": {"false": 0, "ambiguous": 0, "genuine": 1}}

    def test_only_items_found_by_that_system_count_for_it(self):
        marks = dict.fromkeys(self.key, "false")
        assert confirmed_false_counts(self.key, self.items, marks, "rule_tuned")[
            "confirmed_false_for_acceptance"] == 2  # fmt: skip

    def test_an_intended_gesture_is_not_a_false_activation(self):
        marks = dict.fromkeys(self.key, "genuine")
        assert confirmed_false_counts(self.key, self.items, marks, "gru")[
            "confirmed_false_for_acceptance"] == 0  # fmt: skip

    def test_a_missing_mark_blocks_the_verdict(self):
        with pytest.raises(ReviewIncompleteError, match="1 candidate"):
            confirmed_false_counts(self.key, self.items,
                                   {"c001": "false", "c002": "false", "c003": "false"}, "gru")

    def test_an_invalid_mark_blocks_the_verdict(self):
        marks = dict.fromkeys(self.key, "false") | {"c002": "maybe"}
        with pytest.raises(ReviewIncompleteError):
            confirmed_false_counts(self.key, self.items, marks, "gru")


# ------------------------------------------------------------------------------- judging
def recall(pooled=0.9, per_class=(0.9, 0.9, 0.9), cross=0, gestures=90):
    return {"gestures": gestures, "pooled_recall": pooled, "cross_activations": cross,
            "per_class": {c: {"recall": r} for c, r in zip(
                ("swipe_left", "swipe_right", "circle_clockwise"), per_class, strict=True)}}


FACTS = {"recordings": 4, "minutes": 41.0, "days": ["2099-01-01", "2099-01-02"],
         "conditions": ["a", "b"], "protocol_sessions": 2,
         "takes_per_gesture": {"swipe_left": 30, "swipe_right": 30, "circle_clockwise": 30}}


def failed(verdict):
    return [c["check"] for c in verdict["checks"] if not c["passed"]]


class TestJudge:
    def test_everything_passing_passes(self):
        verdict = judge(AcceptanceCriteria(), FACTS, 0, {"r": 0}, recall())
        assert verdict["passed"] and not failed(verdict)

    def test_one_confirmed_false_activation_fails_the_zero_tolerance_criterion(self):
        verdict = judge(AcceptanceCriteria(), FACTS, 1, {"r": 1}, recall())
        assert not verdict["passed"] and failed(verdict) == [
            "confirmed false activations (ambiguous counted as false)"]  # fmt: skip

    @pytest.mark.parametrize("kwargs,name", [
        ({"pooled": 0.84}, "pooled recall"),
        ({"per_class": (0.9, 0.79, 0.9)}, "recall swipe_right"),
        ({"per_class": (0.79, 0.9, 0.9)}, "recall swipe_left"),
        ({"per_class": (0.9, 0.9, 0.79)}, "recall circle_clockwise"),
        ({"cross": 3}, "cross activations"),
        ({"gestures": 60}, "gesture takes available"),
    ])  # fmt: skip
    def test_each_recall_requirement_can_fail_alone(self, kwargs, name):
        verdict = judge(AcceptanceCriteria(), FACTS, 0, {"r": 0}, recall(**kwargs))
        assert failed(verdict) == [name]

    def test_the_boundaries_pass(self):
        verdict = judge(AcceptanceCriteria(), FACTS, 0, {}, recall(0.85, (0.8, 0.8, 0.8), 2))
        assert verdict["passed"]

    def test_too_little_data_fails_the_data_requirement(self):
        facts = FACTS | {"minutes": 39.9}
        verdict = judge(AcceptanceCriteria(), facts, 0, {}, recall())
        assert "data requirements met" in failed(verdict)

    @pytest.mark.parametrize("change,fragment", [
        ({"recordings": 3}, "3 streams"), ({"minutes": 10.0}, "10.0 minutes"),
        ({"days": ["2099-01-01"]}, "1 recording days"), ({"conditions": ["a"]}, "1 distinct"),
        ({"protocol_sessions": 1}, "1 protocol sessions"),
        ({"takes_per_gesture": {"swipe_left": 29, "swipe_right": 30, "circle_clockwise": 30}},
         "29 swipe_left"),
    ])  # fmt: skip
    def test_each_data_requirement_is_checked(self, change, fragment):
        problems = check_data_requirements(AcceptanceCriteria(), FACTS | change)
        assert len(problems) == 1 and fragment in problems[0]


# ------------------------------------------------------------------------ final data
def write_final(final, *, created=("2099-01-01T10:00:00Z", "2099-01-02T10:00:00Z"),
                conditions=("lamp, 60 cm", "window, 80 cm"), takes_per_gesture=2):  # fmt: skip
    for i, (when, cond) in enumerate(zip(created, conditions, strict=True)):
        sid = f"s-fin-stream-{i}"
        write_session(final, sid, [build_stream(20, one_swipe, session_id=sid,
                                                sequence_id=f"{sid}-0001")],
                      created=when, conditions=cond)  # fmt: skip
    sid = "s-fin-takes"
    takes = []
    n = 0
    for _ in range(takes_per_gesture):
        for label, path in (("swipe_left", swipe_path(-1)), ("swipe_right", swipe_path(+1)),
                            ("circle_clockwise", circle_path(True))):  # fmt: skip
            n += 1
            takes.append(make_sequence(path, label=label, session_id=sid,
                                       sequence_id=f"{sid}-{n:04d}"))  # fmt: skip
    write_session(final, sid, takes, created="2099-01-03T10:00:00Z", conditions="lamp, 60 cm")


class TestLoadFinal:
    def test_it_summarises_the_data_the_criteria_are_checked_against(self, work):
        write_final(work["final"])
        data = load_final(work["final"], "2026-10-20T00:00:00Z")
        facts = data.facts
        assert facts["recordings"] == 2 and facts["minutes"] == pytest.approx(40 / 60)
        assert facts["days"] == ["2099-01-01", "2099-01-02"]
        assert facts["conditions"] == ["lamp, 60 cm", "window, 80 cm"]
        assert facts["protocol_sessions"] == 1
        assert facts["takes_per_gesture"] == {"swipe_left": 2, "swipe_right": 2,
                                              "circle_clockwise": 2}  # fmt: skip

    def test_sessions_created_at_or_before_the_freeze_are_not_final_data(self, work):
        write_final(work["final"], created=("2026-10-01T10:00:00Z", "2099-01-02T10:00:00Z"))
        with pytest.raises(AcceptanceDataError, match="s-fin-stream-0"):
            load_final(work["final"], "2026-10-20T00:00:00Z")
        with pytest.raises(AcceptanceDataError):
            load_final(work["final"], "2099-01-02T10:00:00Z")  # equal is not later

    def test_a_missing_final_folder_is_a_data_error(self, work):
        with pytest.raises(AcceptanceDataError):
            load_final(work["final"], "2026-10-20T00:00:00Z")


# ---------------------------------------------------------------------- the one attempt
class TestAcceptRun:
    def attempt(self, work):
        freeze_dir = freeze_for(work).parent
        write_final(work["final"])
        return freeze_dir, accept_run(freeze_dir, work["run"], work["final"],
                                      package_root=work["package"])  # fmt: skip

    def test_unmet_data_requirements_do_not_use_up_the_attempt(self, work):
        freeze_dir = freeze_for(work).parent
        write_final(work["final"], takes_per_gesture=1)
        with pytest.raises(AcceptanceDataError, match="not used up"):
            accept_run(freeze_dir, work["run"], work["final"], package_root=work["package"])
        assert not (freeze_dir / "acceptance" / STARTED_FILE).exists()
        write_final(work["final"].parent / "final2")  # a fixed dataset can still be accepted
        accept_run(freeze_dir, work["run"], work["final"].parent / "final2",
                   package_root=work["package"])  # fmt: skip

    def test_changed_code_stops_the_attempt_before_anything_is_opened(self, work):
        freeze_dir = freeze_for(work).parent
        write_final(work["final"])
        (work["package"] / "scroll.py").write_text("S = 3\n", encoding="utf-8")
        with pytest.raises(FreezeError):
            accept_run(freeze_dir, work["run"], work["final"], package_root=work["package"])
        assert not (freeze_dir / "acceptance").exists()

    def test_it_writes_the_candidates_and_a_blind_review(self, work):
        _, out = self.attempt(work)
        run = json.loads((out / RUN_FILE).read_text("utf-8"))
        items = json.loads((out / ITEMS_FILE).read_text("utf-8"))
        key = json.loads((out / KEY_FILE).read_text("utf-8"))
        result = run["systems"]["rule_defaults"]
        assert run["facts"]["recordings"] == 2 and result["total_candidates"] >= 2
        assert result["recall"]["gestures"] == 6 and run["review_items"] == len(items) >= 2
        assert all(key[i["item_id"]] == ["rule_defaults"] for i in items)
        assert "rule_defaults" not in json.dumps(items)
        assert (out / STARTED_FILE).is_file()

    def test_it_cannot_run_twice(self, work):
        freeze_dir, out = self.attempt(work)
        before = digest(out / RUN_FILE)
        with pytest.raises(AcceptanceError, match="already been run"):
            accept_run(freeze_dir, work["run"], work["final"], package_root=work["package"])
        assert digest(out / RUN_FILE) == before

    def test_the_verdict_needs_every_candidate_marked(self, work):
        freeze_dir, out = self.attempt(work)
        with pytest.raises(AcceptanceError, match="missing or unreadable"):
            accept_finalize(freeze_dir)
        items = json.loads((out / ITEMS_FILE).read_text("utf-8"))
        (out / MARKS_FILE).write_text(json.dumps({items[0]["item_id"]: "false"}), encoding="utf-8")
        with pytest.raises(ReviewIncompleteError):
            accept_finalize(freeze_dir)
        assert not (out / VERDICT_FILE).exists()

    def test_false_and_ambiguous_candidates_fail_a_zero_tolerance_acceptance(self, work):
        freeze_dir, out = self.attempt(work)
        items = json.loads((out / ITEMS_FILE).read_text("utf-8"))
        marks = {i["item_id"]: "ambiguous" if n == 0 else "false" for n, i in enumerate(items)}
        (out / MARKS_FILE).write_text(json.dumps(marks), encoding="utf-8")
        verdict = json.loads(accept_finalize(freeze_dir).read_text("utf-8"))
        system = verdict["systems"]["rule_defaults"]
        assert verdict["accepted"] is False and system["passed"] is False
        assert system["review"]["ambiguous"] == 1
        assert system["review"]["confirmed_false_for_acceptance"] == len(items)
        assert sum(system["per_recording_confirmed_false"].values()) == len(items)
        assert system["false_activation_rate"]["count"] == len(items)
        assert "not proof" in system["caution"]

    def test_intended_gestures_do_not_count_as_false_activations(self, work):
        freeze_dir, out = self.attempt(work)
        items = json.loads((out / ITEMS_FILE).read_text("utf-8"))
        (out / MARKS_FILE).write_text(json.dumps(dict.fromkeys((i["item_id"] for i in items),
                                                               "genuine")), encoding="utf-8")
        verdict = json.loads(accept_finalize(freeze_dir).read_text("utf-8"))
        system = verdict["systems"]["rule_defaults"]
        assert system["review"]["confirmed_false_for_acceptance"] == 0
        rate = system["false_activation_rate"]
        assert rate["per_minute"] == 0 and rate["upper95_one_sided_per_minute"] > 0
        by_check = {c["check"]: c["passed"] for c in system["checks"]}
        assert by_check["confirmed false activations (ambiguous counted as false)"] is True

    def test_marks_made_with_the_review_tool_are_accepted_by_the_verdict(self, work):
        from nocursor.dynamic.review import MarkStore

        freeze_dir, out = self.attempt(work)
        store = MarkStore.open(out / ITEMS_FILE)
        for item in store.items:
            store.set(item["item_id"], "genuine")
        verdict = json.loads(accept_finalize(freeze_dir).read_text("utf-8"))
        assert verdict["systems"]["rule_defaults"]["review"]["genuine"] == len(store.items)

    def test_marks_made_for_a_different_candidate_list_are_rejected(self, work):
        from nocursor.dynamic.review import MarkStore

        freeze_dir, out = self.attempt(work)
        store = MarkStore.open(out / ITEMS_FILE)
        for item in store.items:
            store.set(item["item_id"], "false")
        items = json.loads((out / ITEMS_FILE).read_text("utf-8"))
        items[0]["to_s"] += 1.0  # the candidate list changed after the marks were made
        (out / ITEMS_FILE).write_text(json.dumps(items), encoding="utf-8")
        with pytest.raises(ReviewIncompleteError, match="rejected"):
            accept_finalize(freeze_dir)
        assert not (out / VERDICT_FILE).exists()

    def test_a_tampered_mark_is_rejected_and_an_unreviewed_candidate_blocks_the_verdict(
        self, work
    ):
        from nocursor.dynamic.review import MarkStore

        freeze_dir, out = self.attempt(work)
        store = MarkStore.open(out / ITEMS_FILE)
        store.set(store.items[0]["item_id"], "false")
        if len(store.items) > 1:
            with pytest.raises(ReviewIncompleteError, match="no valid mark"):
                accept_finalize(freeze_dir)  # only one candidate has been reviewed
        data = json.loads((out / MARKS_FILE).read_text("utf-8"))
        data["marks"][store.items[0]["item_id"]]["recording_id"] = "another-recording"
        (out / MARKS_FILE).write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ReviewIncompleteError, match="rejected"):
            accept_finalize(freeze_dir)

    def test_the_verdict_is_written_once(self, work):
        freeze_dir, out = self.attempt(work)
        items = json.loads((out / ITEMS_FILE).read_text("utf-8"))
        (out / MARKS_FILE).write_text(json.dumps(dict.fromkeys((i["item_id"] for i in items),
                                                               "false")), encoding="utf-8")
        accept_finalize(freeze_dir)
        with pytest.raises(AcceptanceError, match="already written"):
            accept_finalize(freeze_dir)


# ------------------------------------------------------------------------------- the CLI
@pytest.fixture(scope="module")
def cli_run(prepared, tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    shutil.copytree(prepared["dataset"], root / "datasets")
    shutil.copytree(prepared["runs"], root / "ml_runs")
    return root


def tree_digest(directory: Path) -> dict[str, str]:
    return {p.relative_to(directory).as_posix(): digest(p) for p in sorted(directory.rglob("*"))
            if p.is_file()}  # fmt: skip


def cli_args(root, command, *extra):
    return [command, "--dataset-dir", str(root / "datasets"), "--runs-dir",
            str(root / "ml_runs"), "--run", "primary", *extra]  # fmt: skip


@pytest.fixture(scope="module")
def evaluated(cli_run):
    """One stream-eval over the synthetic study, with a small grid so it stays quick."""
    import nocursor.dynamic.stream_eval as module

    mp = pytest.MonkeyPatch()
    mp.setattr(module, "GATE_GRID", (0.0, 0.8))
    mp.setattr(module, "TAU_GRID", (0.5, 0.9))
    mp.setattr(module, "VOTE_GRID", ((1, 1), (3, 5)))
    mp.setattr(module, "COOLDOWN_GRID", (1.6,))
    mp.setattr(module, "QUIET_GRID", (0,))
    mp.setattr(module, "GUARD_GRID", (0.0,))
    before = tree_digest(cli_run / "ml_runs" / "primary")
    split_before = digest(next((cli_run / "ml_runs" / "splits").rglob("split.json")))
    code = main(cli_args(cli_run, "stream-eval"))
    mp.undo()
    out = sorted((cli_run / "ml_runs" / "streaming").glob("dev-eval-*"))
    return {"code": code, "out": out, "before": before, "split_before": split_before}


class TestCli:
    def test_the_evaluation_runs_and_writes_a_preliminary_report(self, evaluated, cli_run):
        assert evaluated["code"] == 0 and len(evaluated["out"]) == 1
        report = (evaluated["out"][0] / "report.md").read_text("utf-8")
        assert "PRELIMINARY" in report and "Development evidence only" in report
        assert "Fixed ladder" in report and "L5" in report

    def test_all_three_systems_and_the_fixed_ladder_are_reported(self, evaluated):
        metrics = json.loads((evaluated["out"][0] / "metrics.json").read_text("utf-8"))
        assert set(metrics["systems"]) == {"rule_defaults", "rule_tuned", "gru"}
        for system in metrics["systems"].values():
            assert len(system["ladder"]) == 5 and system["pareto_front"]
            assert set(system["legacy_pooled"]) == {"flagged_windows", "events_any",
                                                    "events_persistent"}  # fmt: skip
        assert metrics["grid_hash"] and metrics["recordings"][0]["file_sha256"]
        hashes = metrics["hashes"]
        assert len(hashes["gru_weights_sha256"]) == 64 and len(hashes["source_hash_combined"]) == 64
        assert hashes["split_id"] == metrics["split_id"] and hashes["acceptance_criteria_hash"]
        for system in metrics["systems"].values():
            assert all(len(h) == 64 for h in system["ladder_config_hashes"].values())
            assert "settings_meeting_dev_constraints" in system and "best_in_sample" in system
        assert any("DEVELOPMENT evidence" in n for n in metrics["notes"])

    def test_existing_runs_and_the_frozen_split_are_untouched(self, evaluated, cli_run):
        assert tree_digest(cli_run / "ml_runs" / "primary") == evaluated["before"]
        assert not (cli_run / "ml_runs" / "primary" / "eval_log.json").exists()
        assert digest(next((cli_run / "ml_runs" / "splits").rglob("split.json"))) == (
            evaluated["split_before"])  # fmt: skip

    def test_tuning_needs_two_recordings_and_changes_nothing(self, cli_run, capsys):
        before = sorted(p.name for p in (cli_run / "ml_runs" / "streaming").iterdir())
        code = main(cli_args(cli_run, "stream-tune"))
        assert code == 1 and "at least two" in capsys.readouterr().err
        assert sorted(p.name for p in (cli_run / "ml_runs" / "streaming").iterdir()) == before

    def test_the_sealed_final_folder_cannot_be_used_as_development_data(self, cli_run, capsys):
        final = cli_run / "datasets" / "streaming" / "final"
        write_final(final)
        code = main(cli_args(cli_run, "stream-eval", "--dev-path", str(final)))
        assert code == 1 and "sealed" in capsys.readouterr().err

    def test_freezing_refuses_when_final_recordings_exist(self, cli_run, capsys):
        report = cli_run / "tune.json"
        report.write_text(json.dumps({"system": "rule_defaults", "chosen": {
            "gate": {"min_palm_fraction": 0.0}, "stabilizer": StabilizerConfig().__dict__}}),
            encoding="utf-8")  # fmt: skip
        code = main(cli_args(cli_run, "stream-freeze", "--primary", "rule_defaults",
                              "--tuning-report", str(report)))  # fmt: skip
        assert code == 1 and "before any final recording" in capsys.readouterr().err

    def test_a_tuning_report_without_a_choice_cannot_be_frozen(self, cli_run, tmp_path, capsys):
        report = tmp_path / "tune.json"
        report.write_text(json.dumps({"system": "gru", "chosen": None}), encoding="utf-8")
        code = main(cli_args(cli_run, "stream-freeze", "--tuning-report", str(report)))
        assert code == 1 and "no chosen setting" in capsys.readouterr().err


class TestTuneAndFreezeEndToEnd:
    """tune on two synthetic dev recordings, then freeze the chosen setting (no final data yet)."""

    def test_the_whole_path_runs_and_the_freeze_matches_the_tuned_setting(
        self, prepared, tmp_path, monkeypatch
    ):
        import nocursor.dynamic.stream_eval as module

        shutil.copytree(prepared["dataset"], tmp_path / "datasets")
        shutil.copytree(prepared["runs"], tmp_path / "ml_runs")
        dev = tmp_path / "datasets" / "streaming" / "dev"
        for i in (1, 2):
            sid = f"s-dev-{i}"
            write_session(dev, sid, [build_stream(20, one_swipe, session_id=sid,
                                                  sequence_id=f"{sid}-0001")])  # fmt: skip
        for name, value in (("GATE_GRID", (0.0, 0.8)), ("TAU_GRID", (0.5,)),
                            ("VOTE_GRID", ((1, 1), (3, 5))), ("COOLDOWN_GRID", (2.5,)),
                            ("QUIET_GRID", (0, 3)), ("GUARD_GRID", (0.0,))):  # fmt: skip
            monkeypatch.setattr(module, name, value)
        base = ["--dataset-dir", str(tmp_path / "datasets"), "--runs-dir",
                str(tmp_path / "ml_runs"), "--run", "primary"]  # fmt: skip

        assert main(["stream-tune", *base, "--system", "rule_defaults", "--no-legacy-stream",
                     "--dev-path", str(dev)]) == 0  # fmt: skip
        (out,) = (tmp_path / "ml_runs" / "streaming").glob("tune-*")
        report = json.loads((out / "metrics.json").read_text("utf-8"))
        assert report["recordings"] == ["s-dev-1-0001", "s-dev-2-0001"]
        assert report["grid_size"] == 2 * 1 * 2 * 1 * 2 * 1 and report["chosen"]
        assert len(report["leave_one_recording_out"]["folds"]) == 2

        assert main(["stream-freeze", *base, "--primary", "rule_defaults", "--tuning-report",
                     str(out / "metrics.json")]) == 0  # fmt: skip
        (freeze_path,) = (tmp_path / "ml_runs" / "streaming").glob("freeze-*/freeze.json")
        freeze = load_freeze(freeze_path.parent)
        assert freeze["decisions"]["rule_defaults"]["config_hash"] == report["chosen"][
            "config_hash"]  # fmt: skip
        assert freeze["tuning_report_sha256"] is None or len(freeze["tuning_report_sha256"]) == 64
        assert not (tmp_path / "datasets" / "streaming" / "final").exists()


class TestHistoricalEvidenceStillVerifies:
    def test_new_streaming_folders_do_not_disturb_the_frozen_split(self, work, prepared):
        split = prepared["split"]
        verify_manifest_and_files(split, work["dataset"])  # baseline: passes
        (work["dataset"] / "streaming" / "dev").mkdir(parents=True)
        write_session(work["dataset"] / "streaming" / "dev", "s-new-dev", [build_stream(
            8, one_swipe, session_id="s-new-dev", sequence_id="s-new-dev-0001")])  # fmt: skip
        write_final(work["dataset"] / "streaming" / "final")
        verify_manifest_and_files(split, work["dataset"])  # still passes: top level untouched


# ------------------------------------------------------------- no link to desktop control
FORBIDDEN = {"pyautogui", "nocursor.controller", "nocursor.dispatch", "nocursor.app",
             "nocursor.interaction", "nocursor.processor", "nocursor.runtime",
             "nocursor.emergency"}  # fmt: skip


def imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


class TestOfflineOnly:
    def test_no_dynamic_module_imports_the_desktop_controls(self):
        for path in sorted((SRC / "nocursor" / "dynamic").glob("*.py")):
            leaked = {m for m in imported_modules(path)
                      if m in FORBIDDEN or m.split(".")[0] == "pyautogui"}  # fmt: skip
            assert not leaked, f"{path.name} imports {leaked}"

    def test_importing_the_streaming_tools_loads_neither_pyautogui_nor_torch(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]);"
            "import nocursor.dynamic.cli, nocursor.dynamic.stream_eval,"
            "nocursor.dynamic.stream_accept, nocursor.dynamic.cli_stream;"
            "bad = [m for m in ('pyautogui', 'torch', 'nocursor.controller', 'nocursor.app')"
            " if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)"
        )
        done = subprocess.run([sys.executable, "-c", code, str(SRC)], capture_output=True,
                              text=True, timeout=120)  # fmt: skip
        assert done.returncode == 0, done.stdout + done.stderr
