"""The controlled comparison of streaming variants: declarations, diagnostics, and the CLI.

All data is SYNTHETIC geometry; these tests check software behaviour, not recognition accuracy.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
from pathlib import Path

import pytest

from dynamic_helpers import make_sequence, swipe_path
from ml_fixtures import add_random_gru, build_study, freeze_split
from nocursor.dynamic import stream_compare as sc
from nocursor.dynamic.arming import ArmingConfig
from nocursor.dynamic.baseline import RuleBaseline, RuleParams
from nocursor.dynamic.cli import main
from nocursor.dynamic.features import FeatureConfig
from nocursor.dynamic.review import MarkStore, validate_items
from nocursor.dynamic.stream_eval import build_system_corpus
from nocursor.dynamic.train import TrainConfig, run_fit_baseline
from test_dynamic_stream_eval import build_stream, one_swipe, write_session

CONFIG = FeatureConfig()
NEWLINE = chr(10)  # the exporter ends its JSON files with one
HAS_TORCH = importlib.util.find_spec("torch") is not None


# ---------------------------------------------------------------------------- declarations
class TestDeclaredComparison:
    def test_the_grid_has_its_declared_size(self):
        assert len(sc.compare_grid("gru", True)) == 3 * 3 * 3 * 1 * 2 * 1 * 4 * 4 == 864
        assert len(sc.compare_grid("rule_defaults", False)) == 288

    def test_every_mechanism_appears_in_the_grid(self):
        mechanisms = {c.extras.mechanism() for c in sc.compare_grid("gru", True)}
        assert mechanisms == set(sc.MECHANISMS)

    def test_the_reference_setting_of_every_variant_is_in_the_grid(self):
        grid = {c.hash() for c in sc.compare_grid("gru", True)}
        for mechanism in sc.MECHANISMS:
            assert sc.reference_config("gru", True, mechanism).hash() in grid

    def test_the_reference_settings_are_the_declared_untuned_ones(self):
        base = sc.reference_config("gru", True, "base")
        assert base.gate.min_palm_fraction == 0.8 and base.stab.tau == 0.9
        assert (base.stab.k, base.stab.n) == (3, 5) and base.stab.cooldown_s == 2.5
        assert sc.reference_config("rule_defaults", False, "base").stab.tau == 0.5
        both = sc.reference_config("gru", True, "settle+arming")
        assert both.extras.settle_speed == 1.0 and both.extras.arming == ArmingConfig(0.3)

    def test_the_reference_arming_rule_is_always_computed(self):
        assert sc.REF_ARMING in sc.arming_configs()

    def test_the_two_protocols_differ_only_in_the_synthetic_lead_in(self):
        assert sc.PROTOCOLS["rest_padded"]["lead_s"] > 0
        assert sc.PROTOCOLS["own_lead_only"]["lead_s"] == 0
        assert sc.PROTOCOLS["rest_padded"]["tail_s"] == sc.PROTOCOLS["own_lead_only"]["tail_s"]

    def test_the_hash_is_stable_and_sensitive_to_every_declaration(self, monkeypatch):
        before = sc.compare_hash()
        assert before == sc.compare_hash()
        for name, value in (("REF_GATE", 0.9), ("SETTLE_SPEEDS", (None, 2.0)),
                            ("GATES", (0.0,)), ("REF_ARMING", ArmingConfig(0.4))):
            monkeypatch.setattr(sc, name, value)
            assert sc.compare_hash() != before
            monkeypatch.undo()
        assert sc.compare_hash() == before


# -------------------------------------------------------------------------- diagnostics
class TestDiagnostics:
    def test_a_cross_diagnosis_names_the_wrong_gesture_and_where_it_fired(self):
        outcomes = [
            {"outcome": "cross", "truth": "circle_clockwise", "wrong": [("swipe_right", 0.3)]},
            {"outcome": "cross", "truth": "circle_clockwise", "wrong": [("swipe_right", 1.0)]},
            {"outcome": "cross", "truth": "circle_clockwise", "wrong": [("swipe_left", 1.5)]},
            {"outcome": "cross", "truth": "swipe_left", "wrong": [("swipe_right", 2.5)]},
            {"outcome": "correct", "truth": "swipe_left", "wrong": []},
        ]  # fmt: skip
        diagnosis = sc.cross_diagnosis(outcomes)
        assert diagnosis["pairs"] == {"circle_clockwise read as swipe_left": 1,
                                      "circle_clockwise read as swipe_right": 2,
                                      "swipe_left read as swipe_right": 1}  # fmt: skip
        assert diagnosis["decided_in_part_of_take"] == {
            "first third": 1, "middle third": 1, "last third": 1, "after the take": 1}

    def test_latency_is_summarised_over_the_recalled_takes_only(self):
        rows = [{"latency_s": 0.2}, {"latency_s": 0.4}, {"latency_s": None}, {}]
        latency = sc.latency_summary(rows)
        assert latency["n"] == 2 and latency["p50_s"] == pytest.approx(0.3)
        assert sc.latency_summary([])["p50_s"] is None

    def test_the_shift_profile_shows_whole_and_partial_windows_differently(self):
        take = make_sequence(swipe_path(+1), label="swipe_right", sequence_id="t-0001")
        cases = sc.build_protocol_cases([take], CONFIG)["rest_padded"]
        corpus = build_system_corpus("rule_defaults", RuleBaseline(RuleParams(), "r"), [], cases)
        profile = sc.shift_profile(corpus, "swipe_right")
        assert profile and all(k.count("..") == 1 for k in profile)
        whole = profile["-0.4..+0.0"]
        far = profile["-1.2..-0.8"]
        assert whole.get("swipe_right", 0) >= 1 and far.get("swipe_right", 0) == 0
        assert far.get("no_gesture", 0) >= 1
        assert sc.shift_profile(corpus, "circle_clockwise") == {}  # no such takes


# ---------------------------------------------------------------------------------- CLI
@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("compare")
    dataset_dir = build_study(root / "datasets", takes=10)
    dev = dataset_dir / "streaming" / "dev"
    for i in (1, 2):
        sid = f"s-cmp-{i}"
        write_session(dev, sid, [build_stream(20, one_swipe, session_id=sid,
                                              sequence_id=f"{sid}-0001")])  # fmt: skip
    runs_dir = root / "ml_runs"
    split, split_file = freeze_split(dataset_dir, runs_dir)
    primary = run_fit_baseline(split, split_file, dataset_dir, runs_dir, "primary",
                               TrainConfig(allow_synthetic=True), log=lambda _: None)  # fmt: skip
    add_random_gru(primary, split, seed=1)
    return {"root": root, "dataset": dataset_dir, "runs": runs_dir, "dev": dev}


def tree_digest(directory: Path) -> dict[str, str]:
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}  # fmt: skip


@pytest.fixture
def tiny_grid(monkeypatch):
    monkeypatch.setattr(sc, "GATES", (0.0, 0.8))
    monkeypatch.setattr(sc, "TAUS", (0.5, 0.9))
    monkeypatch.setattr(sc, "VOTES", ((1, 1), (3, 5)))
    monkeypatch.setattr(sc, "QUIETS", (0,))
    monkeypatch.setattr(sc, "SETTLE_SPEEDS", (None, 1.0))
    monkeypatch.setattr(sc, "ARMINGS", (None, ArmingConfig(0.3)))


def run_compare(prepared, *extra):
    return main(["stream-compare", "--dataset-dir", str(prepared["dataset"]), "--runs-dir",
                 str(prepared["runs"]), "--run", "primary", "--dev-path", str(prepared["dev"]),
                 *extra])  # fmt: skip


@pytest.fixture(scope="module")
def first(prepared):
    # a class-wide run with a tiny grid, done once
    mp = pytest.MonkeyPatch()
    mp.setattr(sc, "GATES", (0.0, 0.8))
    mp.setattr(sc, "TAUS", (0.5, 0.9))
    mp.setattr(sc, "VOTES", ((1, 1), (3, 5)))
    mp.setattr(sc, "QUIETS", (0,))
    mp.setattr(sc, "SETTLE_SPEEDS", (None, 1.0))
    mp.setattr(sc, "ARMINGS", (None, ArmingConfig(0.3)))
    before = tree_digest(prepared["runs"] / "primary")
    code = run_compare(prepared)
    mp.undo()
    out = sorted((prepared["runs"] / "streaming").glob("compare-*"))
    return {"code": code, "out": out[-1], "before": before}


class TestCompareCli:
    def test_it_runs_and_writes_an_exploratory_report(self, first):
        assert first["code"] == 0
        report = (first["out"] / "report.md").read_text("utf-8")
        assert "EXPLORATORY" in report and "unreviewed" in report.lower()
        assert "Reference settings" in report and "How each window is read" in report

    def test_every_system_and_mechanism_is_reported_under_both_protocols(self, first):
        metrics = json.loads((first["out"] / "metrics.json").read_text("utf-8"))
        assert set(metrics["systems"]) == {"rule_defaults", "rule_tuned", "gru"}
        for blocks in metrics["systems"].values():
            assert set(blocks) == set(sc.MECHANISMS)
            for block in blocks.values():
                assert set(block["reference"]["protocols"]) == {"rest_padded", "own_lead_only"}
                assert set(block["screening"]) == {"rest_padded", "own_lead_only"}

    def test_provenance_and_hashes_are_recorded(self, first):
        metrics = json.loads((first["out"] / "metrics.json").read_text("utf-8"))
        hashes = metrics["hashes"]
        assert len(hashes["compare_grid_hash"]) == 64 and len(hashes["gru_weights_sha256"]) == 64
        assert len(hashes["source_hash_combined"]) == 64 and hashes["split_id"]
        assert all(len(r["file_sha256"]) == 64 for r in metrics["recordings"])
        assert metrics["protocols"]["rest_padded"] and metrics["protocols"]["own_lead_only"]

    def test_nothing_is_reviewed_at_first_and_unreviewed_is_never_counted_as_false(self, first):
        metrics = json.loads((first["out"] / "metrics.json").read_text("utf-8"))
        for blocks in metrics["systems"].values():
            for block in blocks.values():
                review = block["review"]
                assert review["false"] == review["genuine"] == review["ambiguous"] == 0
                assert review["unreviewed"] == review["candidates"] == block["reference"][
                    "candidates"]  # fmt: skip

    def test_the_exported_review_items_are_valid_and_blind(self, first):
        folder = first["out"] / "review"
        items = json.loads((folder / "review_items.json").read_text("utf-8"))
        key = json.loads((folder / "review_key.json").read_text("utf-8"))
        assert items and validate_items(items) == items
        assert "gru" not in json.dumps(items) and "rule" not in json.dumps(items)
        assert set(key) == {i["item_id"] for i in items}
        assert all("|" in variant for names in key.values() for variant in names)

    def test_existing_runs_are_untouched(self, first, prepared):
        assert tree_digest(prepared["runs"] / "primary") == first["before"]
        assert not (prepared["runs"] / "primary" / "eval_log.json").exists()

    def test_reviewed_candidates_are_counted_apart_from_unreviewed_ones(
        self, first, prepared, tiny_grid
    ):
        folder = first["out"] / "review"
        store = MarkStore.open(folder / "review_items.json")
        marked = store.items[: max(1, len(store.items) // 2)]
        for entry in marked:
            store.set(entry["item_id"], "false")
        assert run_compare(prepared, "--review-dir", str(folder)) == 0
        latest = sorted((prepared["runs"] / "streaming").glob("compare-*"))[-1]
        metrics = json.loads((latest / "metrics.json").read_text("utf-8"))
        assert latest != first["out"]
        totals = {"false": 0, "unreviewed": 0, "candidates": 0}
        for blocks in metrics["systems"].values():
            for block in blocks.values():
                for key in totals:
                    totals[key] += block["review"][key]
                review = block["review"]
                assert review["false"] + review["genuine"] + review["ambiguous"] + review[
                    "unreviewed"] == review["candidates"]  # fmt: skip
        assert totals["false"] > 0 and totals["unreviewed"] > 0  # half were marked: not all
        assert metrics["hashes"]["review_items_sha256"]

    def test_an_unknown_candidate_name_is_an_error(self, prepared, capsys, tiny_grid):
        assert run_compare(prepared, "--candidate", "does-not-exist") == 1
        assert "candidate not found" in capsys.readouterr().err

    def test_a_review_folder_with_the_wrong_marks_is_refused(self, first, prepared, tmp_path,
                                                             capsys, tiny_grid):  # fmt: skip
        folder = tmp_path / "review"
        shutil.copytree(first["out"] / "review", folder)
        store = MarkStore.open(folder / "review_items.json")
        store.set(store.items[0]["item_id"], "false")
        items = json.loads((folder / "review_items.json").read_text("utf-8"))
        items[0]["to_s"] += 1.0  # the list changed after the marks were made
        (folder / "review_items.json").write_text(json.dumps(items), encoding="utf-8")
        assert run_compare(prepared, "--review-dir", str(folder)) == 1
        assert "different candidate list" in capsys.readouterr().err


def _tiny_grid_patch() -> pytest.MonkeyPatch:
    mp = pytest.MonkeyPatch()
    mp.setattr(sc, "GATES", (0.0, 0.8))
    mp.setattr(sc, "TAUS", (0.5, 0.9))
    mp.setattr(sc, "VOTES", ((1, 1), (3, 5)))
    mp.setattr(sc, "QUIETS", (0,))
    mp.setattr(sc, "SETTLE_SPEEDS", (None, 1.0))
    mp.setattr(sc, "ARMINGS", (None, ArmingConfig(0.3)))
    return mp


@pytest.fixture(scope="module")
def review_variants(prepared, first, tmp_path_factory):
    """One comparison per kind of supplied review folder, all against the same candidates.

    ``same``: the folder this run's export would equal. ``differs``: one item's recorded candidate
    times differ, so the file hash differs. ``no_marks``: no marks file. ``missing_item``: one item
    (and its key entry) removed, so its candidates match no item.
    """
    base = tmp_path_factory.mktemp("review-variants")
    export = first["out"] / "review"
    items = json.loads((export / "review_items.json").read_text("utf-8"))
    key = json.loads((export / "review_key.json").read_text("utf-8"))

    def make(name, items_, key_, marks=True):
        folder = base / name
        folder.mkdir()
        # the exporter's exact byte format, so an unmodified list keeps the same hash
        text = {"review_items.json": items_, "review_key.json": key_}
        for filename, content in text.items():
            (folder / filename).write_text(json.dumps(content, indent=2) + NEWLINE,
                                           encoding="utf-8")
        if marks:
            store = MarkStore.open(folder / "review_items.json")
            for entry in store.items[: max(1, len(store.items) // 2)]:
                store.set(entry["item_id"], "false")
        return folder

    changed = [dict(item) for item in items]
    changed[0]["candidate_times_s"] = [*changed[0]["candidate_times_s"], 999.0]
    dropped = items[0]["item_id"]
    remaining = [item for item in items if item["item_id"] != dropped]
    folders = {
        "same": make("same", items, key),
        "differs": make("differs", changed, key),
        "no_marks": make("no_marks", items, key, marks=False),
        "missing_item": make("missing_item", remaining,
                             {k: v for k, v in key.items() if k != dropped}),
    }  # fmt: skip
    before = {name: tree_digest(folder) for name, folder in folders.items()}
    results: dict[str, dict] = {"changed_item": changed[0]["item_id"], "dropped_item": dropped}
    mp = _tiny_grid_patch()
    try:
        for name, folder in folders.items():
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = run_compare(prepared, "--review-dir", str(folder))
            out = sorted((prepared["runs"] / "streaming").glob("compare-*"))[-1]
            results[name] = {
                "code": code, "stderr": stderr.getvalue(), "out": out,
                "metrics": json.loads((out / "metrics.json").read_text("utf-8")),
                "report": (out / "report.md").read_text("utf-8"),
            }  # fmt: skip
    finally:
        mp.undo()
    results["folders_unchanged"] = all(tree_digest(f) == before[n] for n, f in folders.items())
    return results


def review_blocks(metrics):
    return {(s, m): block["review"] for s, blocks in metrics["systems"].items()
            for m, block in blocks.items()}  # fmt: skip


class TestReviewSourceTransparency:
    def test_every_run_succeeded_and_no_supplied_folder_was_modified(self, review_variants):
        for name in ("same", "differs", "no_marks", "missing_item"):
            assert review_variants[name]["code"] == 0
        assert review_variants["folders_unchanged"]

    def test_the_source_is_recorded_with_its_hashes_and_progress(self, review_variants):
        same = review_variants["same"]["metrics"]["review_source"]
        assert same["folder"].endswith("same")
        assert len(same["items_sha256"]) == 64 and len(same["marks_sha256"]) == 64
        assert same["marks_file_present"] is True
        assert same["items_total"] > 1
        assert same["items_marked"] == max(1, same["items_total"] // 2)
        assert same["current_export_items_sha256"] == review_variants["same"]["metrics"][
            "review_export"]["items_sha256"]  # fmt: skip

    def test_an_identical_list_is_reported_as_identical(self, review_variants):
        same = review_variants["same"]
        assert same["metrics"]["review_source"]["matches_current_export"] is True
        assert same["metrics"]["review_source"]["differing_items"] == []
        assert "identical to the list this run exports" in same["report"]
        assert "differs from the list this run exports" not in same["report"]

    def test_a_different_hash_is_explained_and_the_differing_item_is_named(self, review_variants):
        run = review_variants["differs"]
        source = run["metrics"]["review_source"]
        assert source["matches_current_export"] is False
        assert source["items_sha256"] != source["current_export_items_sha256"]
        (entry,) = source["differing_items"]
        assert entry["item_id"] == review_variants["changed_item"]
        assert 999.0 in entry["fields"]["candidate_times_s"]["source"]
        assert 999.0 not in entry["fields"]["candidate_times_s"]["current"]
        report = run["report"]
        assert "differs from the list this run exports" in report
        assert "by content" in report and "not by file hash or item id" in report
        assert f"item `{entry['item_id']}` differs in `candidate_times_s`" in report
        assert "999.0" in report
        assert "belong to the source list." + NEWLINE * 2 + "- item" in report  # a real list
        assert source["items_sha256"][:16] in report
        assert source["current_export_items_sha256"][:16] in report

    def test_content_based_matching_and_the_counts_are_unchanged_by_the_hash_difference(
        self, review_variants
    ):
        same = review_blocks(review_variants["same"]["metrics"])
        differs = review_blocks(review_variants["differs"]["metrics"])
        assert differs == same
        assert sum(r["false"] for r in same.values()) > 0
        assert sum(r["unreviewed"] for r in same.values()) > 0  # only half were marked
        assert all(r["not_in_review_set"] == 0 for r in differs.values())

    def test_candidates_matching_no_item_are_shown_when_there_are_any(self, review_variants):
        run = review_variants["missing_item"]
        blocks = review_blocks(run["metrics"])
        assert sum(r["not_in_review_set"] for r in blocks.values()) > 0
        for r in blocks.values():
            assert r["false"] + r["genuine"] + r["ambiguous"] + r["unreviewed"] == r["candidates"]
            assert r["not_in_review_set"] <= r["unreviewed"]
        source = run["metrics"]["review_source"]
        assert source["items_only_in_current"] == [review_variants["dropped_item"]]
        assert "match **no** item" in run["report"]
        assert f"only in this run's export: {review_variants['dropped_item']}" in run["report"]

    def test_the_no_match_notice_is_absent_when_every_candidate_matches(self, review_variants):
        assert "match **no** item" not in review_variants["same"]["report"]
        assert "match **no** item" not in review_variants["differs"]["report"]

    def test_a_missing_marks_file_is_warned_about_and_counts_nothing_as_reviewed(
        self, review_variants
    ):
        run = review_variants["no_marks"]
        source = run["metrics"]["review_source"]
        assert source["marks_file_present"] is False and source["marks_sha256"] is None
        assert "no review_marks.json" in run["stderr"]
        assert "Warning: the folder has no `review_marks.json`" in run["report"]
        for r in review_blocks(run["metrics"]).values():
            assert r["false"] == r["genuine"] == r["ambiguous"] == 0
            assert r["unreviewed"] == r["candidates"]

    def test_without_a_review_folder_the_report_says_so(self, first):
        metrics = json.loads((first["out"] / "metrics.json").read_text("utf-8"))
        assert metrics["review_source"] is None
        assert "No review folder was supplied" in (first["out"] / "report.md").read_text("utf-8")

    def test_the_existing_reports_were_not_rewritten(self, review_variants, first):
        # the runs above wrote new compare-* folders; the first one is exactly as it was written
        assert first["out"] not in {review_variants[n]["out"] for n in
                                    ("same", "differs", "no_marks", "missing_item")}  # fmt: skip


class TestFloorOverride:
    def test_the_floor_is_replaced_and_predictions_pass_through(self):
        class Inner:
            threshold = 0.999

            def predict_features(self, features):
                return "the inner prediction"

        view = sc.FloorOverride(Inner(), 0.5, "gru:x@floor0.50")
        assert view.threshold == 0.5 and view.name == "gru:x@floor0.50"
        assert view.predict_features(None) == "the inner prediction"

    def test_candidates_are_reported_as_saved_and_with_the_floor_replaced(
        self, prepared, tmp_path, tiny_grid
    ):
        shutil.copytree(prepared["dataset"], tmp_path / "datasets")
        shutil.copytree(prepared["runs"], tmp_path / "ml_runs")
        split = json.loads((tmp_path / "ml_runs" / "primary" / "split.json").read_text("utf-8"))
        add_random_gru(tmp_path / "ml_runs" / "streaming" / "candidates" / "fake", split,
                       threshold=0.99, seed=3)  # fmt: skip
        code = main(["stream-compare", "--dataset-dir", str(tmp_path / "datasets"), "--runs-dir",
                     str(tmp_path / "ml_runs"), "--run", "primary", "--dev-path",
                     str(tmp_path / "datasets" / "streaming" / "dev"), "--candidate", "fake"])
        assert code == 0
        out = sorted((tmp_path / "ml_runs" / "streaming").glob("compare-*"))[-1]  # the newest
        metrics = json.loads((out / "metrics.json").read_text("utf-8"))
        assert {"gru", "gru:fake", "gru:fake@floor0.60"} <= set(metrics["systems"])
        assert metrics["hashes"]["candidates"][0]["threshold"] == 0.99
        assert len(metrics["hashes"]["candidates"][0]["weights_sha256"]) == 64


@pytest.mark.skipif(HAS_TORCH, reason="PyTorch is installed here; that path is unreachable")
def test_without_pytorch_the_candidate_command_fails_clearly(prepared, capsys):
    code = main(["stream-candidate", "--dataset-dir", str(prepared["dataset"]), "--runs-dir",
                 str(prepared["runs"]), "--run", "primary", "--name", "never",
                 "--allow-synthetic"])  # fmt: skip
    assert code == 1 and "PyTorch" in capsys.readouterr().err
    assert not (prepared["runs"] / "streaming" / "candidates" / "never").exists()
