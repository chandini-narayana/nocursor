"""Command line for the streaming-robustness checkpoint (offline only).

``stream-eval``      score the three systems through gate + stabilizer on development evidence
``stream-tune``      choose a setting with leave-one-recording-out (needs >= 2 dev recordings)
``stream-freeze``    freeze the chosen settings, model and source hashes (before final recordings)
``stream-accept``    run the one-shot acceptance on the sealed final data, then finalize it

Nothing here is connected to the mouse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nocursor.dynamic.collect import EXIT_ERROR, EXIT_OK
from nocursor.dynamic.schema import DatasetError
from nocursor.paths import default_dataset_dir, default_ml_runs_dir

STREAM_COMMANDS = ("stream-eval", "stream-tune", "stream-freeze", "stream-accept",
                   "stream-review", "stream-candidate", "stream-compare")


def add_stream_commands(commands: Any) -> None:
    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--dataset-dir", type=Path, metavar="DIR")
        sub.add_argument("--runs-dir", type=Path, metavar="DIR")
        sub.add_argument("--run", required=True, metavar="DIR_OR_NAME",
                         help="the trained run whose systems are scored")  # fmt: skip

    def dev_data(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--dev-path", type=Path, action="append", default=[], metavar="PATH",
                         help="extra development session files or folders (never the sealed "
                              "final folder); repeatable")  # fmt: skip
        sub.add_argument("--no-legacy-stream", action="store_true",
                         help="do not include the original 120 s stream from the run's split")
        sub.add_argument("--no-validation-takes", action="store_true",
                         help="do not use the run's validation session takes for recall")

    ev = commands.add_parser("stream-eval", help="gate + stabilizer on development evidence",
                             description="Preliminary, development-only scoring of the rule "
                                         "baselines and the GRU through the open-palm gate and "
                                         "temporal stabilizer. Selects nothing.")  # fmt: skip
    common(ev)
    dev_data(ev)
    tune = commands.add_parser("stream-tune", help="choose settings with leave-one-recording-out",
                               description="Search the declared grid on development data only.")
    common(tune)
    dev_data(tune)
    tune.add_argument("--system", default="gru", choices=("gru", "rule_defaults", "rule_tuned"))

    freeze = commands.add_parser("stream-freeze", help="freeze settings before final recordings")
    common(freeze)
    freeze.add_argument("--tuning-report", required=True, type=Path, metavar="metrics.json")
    freeze.add_argument("--primary", default="gru", choices=("gru", "rule_defaults", "rule_tuned"))
    freeze.add_argument("--final-dir", type=Path, metavar="DIR",
                        help="where the sealed final recordings will go "
                             "(default: datasets/streaming/final)")  # fmt: skip

    review = commands.add_parser(
        "stream-review", help="review candidate activations by replaying the landmark skeleton",
        description="Replay recorded landmark sequences (no video, no webcam) and mark each "
                    "candidate false, genuine or ambiguous. Marks are saved after every keypress.",
    )  # fmt: skip
    review.add_argument("--items", required=True, type=Path, metavar="review_items.json")
    review.add_argument("--marks", type=Path, metavar="FILE",
                        help="where marks are saved (default: next to the items)")
    review.add_argument("--recordings", type=Path, action="append", default=[], metavar="PATH",
                        help="session files or folders holding the recordings; repeatable")
    review.add_argument("--run", metavar="DIR_OR_NAME",
                        help="also load the original 120 s stream recorded with this run's split")
    review.add_argument("--dataset-dir", type=Path, metavar="DIR")
    review.add_argument("--runs-dir", type=Path, metavar="DIR")

    cand = commands.add_parser(
        "stream-candidate", help="train a separately named partial-window candidate (PyTorch)",
        description="Train a candidate GRU on whole takes plus partial-window samples cut from the "
                    "training takes. Never touches the primary run, the test split or the stream.",
    )  # fmt: skip
    common(cand)
    cand.add_argument("--name", required=True, help="candidate name (never overwritten)")
    cand.add_argument("--seeds", type=int, default=5, help="seeds per setting, 0..N-1 (default 5)")
    cand.add_argument("--max-epochs", type=int, default=150)
    cand.add_argument("--hidden-sizes", type=int, nargs="+", default=[32, 64])
    cand.add_argument("--allow-synthetic", action="store_true",
                      help="software tests only: permit synthetic sessions")

    compare = commands.add_parser(
        "stream-compare", help="controlled comparison of the fallback variants (development)",
        description="Score the GRU, any candidates and the rule baselines, each with and without "
                    "the settle gate and the arming rule, on the same development evidence. "
                    "Exploratory: selects and freezes nothing.",
    )  # fmt: skip
    common(compare)
    dev_data(compare)
    compare.add_argument("--candidate", action="append", default=[], metavar="NAME_OR_DIR",
                         help="a candidate from stream-candidate; repeatable")
    compare.add_argument("--review-dir", type=Path, metavar="DIR",
                         help="a folder with review_items.json, review_key.json and "
                              "review_marks.json from an earlier comparison, to count reviewed "
                              "candidates separately from unreviewed ones")  # fmt: skip

    accept = commands.add_parser("stream-accept", help="one-shot acceptance on the final data")
    accept.add_argument("--runs-dir", type=Path, metavar="DIR")
    accept.add_argument("--dataset-dir", type=Path, metavar="DIR")
    accept.add_argument("--run", required=True, metavar="DIR_OR_NAME")
    accept.add_argument("--freeze", required=True, metavar="FREEZE_ID_OR_DIR")
    accept.add_argument("--final-dir", type=Path, metavar="DIR")
    accept.add_argument("--finalize", action="store_true",
                        help="after review: read review_marks.json and write the verdict")


def run_stream_command(args: argparse.Namespace) -> int:
    from nocursor.dynamic.review import ReviewError
    from nocursor.dynamic.stream_accept import AcceptanceError, FreezeError
    from nocursor.dynamic.stream_eval import SealedDataError
    from nocursor.dynamic.train import TorchUnavailableError

    try:
        if args.command == "stream-eval":
            return _eval(args)
        if args.command == "stream-tune":
            return _tune(args)
        if args.command == "stream-review":
            return _review(args)
        if args.command == "stream-candidate":
            return _candidate(args)
        if args.command == "stream-compare":
            return _compare(args)
        if args.command == "stream-freeze":
            return _freeze(args)
        return _accept(args)
    except (AcceptanceError, FreezeError, SealedDataError, ReviewError, DatasetError, ValueError,
            FileNotFoundError, TorchUnavailableError) as exc:  # fmt: skip
        print(f"nocursor.dynamic: {exc}", file=sys.stderr)
        return EXIT_ERROR


# ------------------------------------------------------------------------------- helpers
def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    dataset_dir: Path = getattr(args, "dataset_dir", None) or default_dataset_dir()
    runs_dir: Path = args.runs_dir or default_ml_runs_dir()
    candidate = Path(args.run)
    run_dir = candidate if candidate.is_dir() else runs_dir / args.run
    if not run_dir.is_dir():
        raise DatasetError(f"run not found: {args.run}")
    return dataset_dir, runs_dir, run_dir


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _next_dir(parent: Path, prefix: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    number = len(list(parent.glob(f"{prefix}-*"))) + 1
    out = parent / f"{prefix}-{number:03d}-{_now_tag()}"
    out.mkdir()
    return out


def _load_dev(args: argparse.Namespace, dataset_dir: Path, run_dir: Path):
    """Development streams, replay takes, the run's split and its feature config."""
    from nocursor.dynamic.features import FeatureConfig
    from nocursor.dynamic.splits import SPLIT_VALIDATION, load_split_items, load_stream
    from nocursor.dynamic.stream_eval import load_recordings

    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    config = FeatureConfig(**split["feature_config"])
    streams, takes = load_recordings(args.dev_path) if args.dev_path else ([], [])
    notes = []
    if not args.no_legacy_stream:
        from nocursor.dynamic.stream_eval import Recording

        for seq in load_stream(split, dataset_dir):
            row = next(r for r in split["sessions"] if r["session_id"] == seq.session_id)
            name = f"{seq.session_id}.jsonl"
            conditions = "legacy 120 s stream (ordinary desktop use)"
            streams.append(Recording(seq.sequence_id, seq.session_id, seq, name,
                                     row["file_sha256"], conditions, row["date"]))
        notes.append("the original 120 s stream is included as DEVELOPMENT evidence (it was "
                     "inspected before this checkpoint)")  # fmt: skip
    if not args.no_validation_takes:
        takes.extend(i.sequence for i in load_split_items(split, dataset_dir, [SPLIT_VALIDATION]))
        notes.append("validation-session takes are used for recall; the GRU's early stopping used "
                     "them, so recall is optimistic")  # fmt: skip
    if not streams:
        raise DatasetError("no development streams found")
    return streams, takes, split, config, notes


def _build_corpora(streams, takes, config, run_dir: Path):
    from nocursor.dynamic.evaluate import load_systems
    from nocursor.dynamic.stream_eval import (
        build_replay_cases,
        build_system_corpus,
        compute_windows,
    )

    windows = [compute_windows(r.sequence, config, recording_id=r.recording_id) for r in streams]
    gestures = [t for t in takes if t.label != "no_gesture"]
    negatives = [t for t in takes if t.label == "no_gesture"]
    cases = build_replay_cases(gestures + negatives, config)
    systems = load_systems(run_dir)
    corpora = {s.name: build_system_corpus(s.name, s.classifier, windows, cases) for s in systems}
    return windows, cases, systems, corpora


def _dump(out: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, indent=2, default=str) + "\n"
    (out / "metrics.json").write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def _hashes(run_dir: Path, split: dict[str, Any]) -> dict[str, Any]:
    from nocursor.dynamic.stream_accept import AcceptanceCriteria, source_hashes
    from nocursor.dynamic.stream_eval import grid_hash

    sources = source_hashes()
    meta = run_dir / "model.json"
    baseline = run_dir / "baseline.json"
    return {
        "gru_weights_sha256": json.loads(meta.read_text("utf-8"))["weights_sha256"]
        if meta.is_file() else None,
        "baseline_json_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest()
        if baseline.is_file() else None,
        "split_id": split["split_id"], "dataset_manifest_hash": split["dataset_manifest_hash"],
        "grid_hash": grid_hash(), "acceptance_criteria_hash": AcceptanceCriteria().hash(),
        "source_files": len(sources),
        "source_hash_combined": hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()).hexdigest(),
    }


# --------------------------------------------------------------------------------- eval
def _eval(args: argparse.Namespace) -> int:
    from nocursor.dynamic.stream_eval import (
        LADDER,
        DecisionConfig,
        evaluate_config,
        grid_configs,
        grid_hash,
        legacy_counts,
        meets_dev_constraints,
        pareto_summary,
        rate_summary,
        select_config,
    )
    from nocursor.dynamic.stream_report import render_eval

    dataset_dir, runs_dir, run_dir = _paths(args)
    streams, takes, split, config, notes = _load_dev(args, dataset_dir, run_dir)
    windows, cases, systems, corpora = _build_corpora(streams, takes, config, run_dir)
    print(f"{len(windows)} dev recording(s), {sum(w.minutes for w in windows):.1f} min; "
          f"{len({c.take_id for c in cases})} replayed takes")  # fmt: skip

    report: dict[str, Any] = {"hashes": _hashes(run_dir, split),
                              "kind": "stream-eval (development, preliminary)",
                              "created_utc": _now_tag(), "split_id": split["split_id"],
                              "run": run_dir.name, "grid_hash": grid_hash(), "notes": notes,
                              "recordings": [{"recording_id": r.recording_id,
                                              "file": r.file, "file_sha256": r.file_sha256,
                                              "conditions": r.conditions, "minutes": r.minutes,
                                              "windows": len(w.starts), "evaluated": w.evaluated}
                                             for r, w in zip(streams, windows, strict=True)],
                              "replay_takes": len({c.take_id for c in cases}),
                              "systems": {}}  # fmt: skip
    for system in systems:
        corpus = corpora[system.name]
        has_conf = system.name == "gru"
        legacy = [legacy_counts(w, o) for w, o in corpus.streams]
        minutes = sum(w.minutes for w in windows)
        ladder = []
        for label, gate, stab in LADDER:
            result = evaluate_config(corpus, DecisionConfig(system.name, gate, stab))
            ladder.append({"stage": label, **result.summary(),
                           "config_hash_full": result.config.hash(),
                           "rate": rate_summary(result.candidates, minutes),
                           "missed_by_reason": result.recall["missed_by_reason"],
                           "recall_detail": result.recall["per_class"]})  # fmt: skip
        results = [evaluate_config(corpus, c) for c in grid_configs(system.name, has_conf)]
        feasible = [r for r in results if meets_dev_constraints(r.recall)]
        best = select_config(results)
        report["systems"][system.name] = {
            "weights_or_params": system.description.get("weights_sha256")
            or system.description.get("parameters"),
            "legacy_window_counts": legacy,
            "legacy_pooled": {k: sum(x[k] for x in legacy) for k in
                              ("flagged_windows", "events_any", "events_persistent")},
            "ladder": ladder, "grid_size": len(results),
            "pareto_front": pareto_summary(results),
            "settings_meeting_dev_constraints": len(feasible),
            "best_in_sample": None if best is None else best.summary(),
            "ladder_config_hashes": {row["stage"]: row["config_hash_full"] for row in ladder},
        }  # fmt: skip
        print(f"  {system.name}: {len(results)} settings scored")
    out = _next_dir(runs_dir / "streaming", "dev-eval")
    digest = _dump(out, report)
    (out / "report.md").write_text(render_eval(report), encoding="utf-8")
    print(f"Report: {out / 'report.md'}\nmetrics.json sha256 {digest}")
    return EXIT_OK


# --------------------------------------------------------------------------------- tune
def _tune(args: argparse.Namespace) -> int:
    from nocursor.dynamic.stream_eval import (
        evaluate_config,
        grid_configs,
        grid_hash,
        leave_one_recording_out,
        pareto_summary,
        select_config,
    )

    dataset_dir, runs_dir, run_dir = _paths(args)
    streams, takes, split, config, notes = _load_dev(args, dataset_dir, run_dir)
    if len(streams) < 2:
        raise ValueError(f"tuning needs at least two development recordings, found {len(streams)}"
                         "; record more with `collect --stream-seconds` into "
                         "datasets/streaming/dev/ (nothing was tuned)")  # fmt: skip
    _, cases, _, corpora = _build_corpora(streams, takes, config, run_dir)
    corpus = corpora[args.system]
    results = [evaluate_config(corpus, c) for c in grid_configs(args.system, args.system == "gru")]
    chosen = select_config(results)
    report = {
        "kind": "stream-tune (development only)", "system": args.system, "grid_hash": grid_hash(),
        "grid_size": len(results), "notes": notes, "run": run_dir.name,
        "recordings": [r.recording_id for r in streams],
        "leave_one_recording_out": leave_one_recording_out(results),
        "chosen": None if chosen is None else {**chosen.summary(),
                                               "config_hash": chosen.config.hash()},
        "pareto_front": pareto_summary(results, 60),
    }  # fmt: skip
    out = _next_dir(runs_dir / "streaming", "tune")
    digest = _dump(out, report)
    print(f"chosen: {report['chosen'] and report['chosen']['config_hash'][:12]}\n"
          f"Report: {out / 'metrics.json'}  sha256 {digest}")  # fmt: skip
    return EXIT_OK


# ------------------------------------------------------------------------------- review
def _review(args: argparse.Namespace) -> int:
    from nocursor.dynamic.review import MarkStore
    from nocursor.dynamic.review_viewer import ReviewSession, run_viewer
    from nocursor.dynamic.splits import load_stream
    from nocursor.dynamic.stream_eval import load_recordings

    dataset_dir: Path = args.dataset_dir or default_dataset_dir()
    store = MarkStore.open(args.items, args.marks)
    # Only the candidates of an acceptance run may be reviewed against the sealed final data.
    acceptance = args.items.resolve().parent.name == "acceptance"
    paths = list(args.recordings) or ([dataset_dir / "streaming" / "final"] if acceptance else [])
    sequences = {}
    if paths:
        streams, _ = load_recordings(paths, allow_sealed=acceptance)
        sequences.update({r.recording_id: r.sequence for r in streams})
    if args.run:
        runs_dir: Path = args.runs_dir or default_ml_runs_dir()
        candidate = Path(args.run)
        run_dir = candidate if candidate.is_dir() else runs_dir / args.run
        split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
        sequences.update({s.sequence_id: s for s in load_stream(split, dataset_dir)})
    session = ReviewSession(store.items, store, sequences)
    run_viewer(session)
    reviewed, total = store.progress()
    print(f"Reviewed {reviewed} of {total} candidates; {total - reviewed} remain unreviewed. "
          f"Marks: {store.path}")  # fmt: skip
    return EXIT_OK


# ---------------------------------------------------------------------------- candidate
def _candidate(args: argparse.Namespace) -> int:
    from nocursor.dynamic.partial_window import run_partial_train
    from nocursor.dynamic.splits import read_split
    from nocursor.dynamic.train import TrainConfig

    if args.seeds < 1 or args.max_epochs < 1 or min(args.hidden_sizes) < 1:
        raise ValueError("--seeds, --max-epochs and --hidden-sizes must be >= 1")
    dataset_dir, runs_dir, run_dir = _paths(args)
    split, split_file = read_split(run_dir / "split.json", runs_dir)
    config = TrainConfig(hidden_sizes=tuple(args.hidden_sizes), seeds=tuple(range(args.seeds)),
                         max_epochs=args.max_epochs, allow_synthetic=args.allow_synthetic)
    target = run_partial_train(split, split_file, dataset_dir, runs_dir, args.name, config,
                               parent_run=run_dir)  # fmt: skip
    print(f"Candidate saved: {target}")
    return EXIT_OK


# ------------------------------------------------------------------------------ compare
def _resolve_candidate(reference: str, runs_dir: Path) -> Path:
    from nocursor.dynamic.partial_window import CANDIDATES_DIR

    candidate = Path(reference)
    path = candidate if candidate.is_dir() else runs_dir.joinpath(*CANDIDATES_DIR) / reference
    if not path.is_dir():
        raise DatasetError(f"candidate not found: {reference}")
    return path


def _load_review_dir(directory: Path) -> dict[str, Any]:
    from nocursor.dynamic.review import file_sha256, load_items, load_marks

    items_path = directory / "review_items.json"
    items = load_items(items_path)
    key = json.loads((directory / "review_key.json").read_text(encoding="utf-8"))
    marks_path = directory / "review_marks.json"
    marks = load_marks(marks_path, items, file_sha256(items_path)) if marks_path.is_file() else {}
    return {"items": items, "key": key, "marks": marks, "items_sha256": file_sha256(items_path),
            "folder": str(directory), "marks_present": marks_path.is_file(),
            "marks_sha256": file_sha256(marks_path) if marks_path.is_file() else None}


def _compare(args: argparse.Namespace) -> int:
    from nocursor.dynamic.evaluate import load_systems
    from nocursor.dynamic.model_io import load_classifier
    from nocursor.dynamic.review import describe_review_source, file_sha256
    from nocursor.dynamic.stream_accept import AcceptanceCriteria, source_hashes
    from nocursor.dynamic.stream_compare import (
        MECHANISMS,
        PROTOCOLS,
        FloorOverride,
        build_protocol_cases,
        build_stream_windows,
        candidate_rows,
        compare_hash,
        export_review,
        reference_result,
        score_system,
        shift_profile,
        summarize_variant,
        tally_for,
    )
    from nocursor.dynamic.stream_report import render_compare

    dataset_dir, runs_dir, run_dir = _paths(args)
    streams, takes, split, config, notes = _load_dev(args, dataset_dir, run_dir)
    review = _load_review_dir(args.review_dir) if args.review_dir else None
    if review is not None and not review["marks_present"]:
        print(f"nocursor.dynamic: warning: {args.review_dir} has no review_marks.json, so every "
              "candidate will be counted as unreviewed", file=sys.stderr)  # fmt: skip
    loaded = load_systems(run_dir)
    systems = [(s.name, s.classifier, s.description) for s in loaded]
    primary_floor = next((float(s.classifier.threshold) for s in loaded if s.name == "gru"), 0.5)
    candidate_info = []
    for reference in args.candidate:
        path = _resolve_candidate(reference, runs_dir)
        meta = json.loads((path / "model.json").read_text(encoding="utf-8"))
        classifier = load_classifier(path)
        systems.append((f"gru:{path.name}", classifier, {"candidate": meta.get("candidate")}))
        view = f"gru:{path.name}@floor{primary_floor:.2f}"
        systems.append((view, FloorOverride(classifier, primary_floor, view),
                        {"candidate": path.name, "floor_replaced_by": primary_floor}))
        candidate_info.append({"name": path.name, "weights_sha256": meta["weights_sha256"],
                               "model_json_sha256": file_sha256(path / "model.json"),
                               "threshold": meta["rejection"]["threshold"],
                               "disclosures": (meta.get("candidate") or {}).get(
                                   "disclosures")})  # fmt: skip
    windows = build_stream_windows(streams, config)
    cases = build_protocol_cases(takes, config)
    print(f"{len(windows)} dev recording(s), {sum(w.minutes for w in windows):.1f} min; "
          f"{len({c.take_id for c in cases['rest_padded']})} replayed takes; "
          f"{len(systems)} systems x {len(MECHANISMS)} mechanisms")  # fmt: skip

    report: dict[str, Any] = {
        "kind": "stream-compare (development, exploratory)", "created_utc": _now_tag(),
        "run": run_dir.name, "split_id": split["split_id"], "notes": notes,
        "recordings": [{"recording_id": r.recording_id, "file": r.file,
                        "file_sha256": r.file_sha256, "conditions": r.conditions,
                        "minutes": r.minutes} for r in streams],  # fmt: skip
        "replay_takes": {label: sum(1 for t in takes if t.label == label)
                         for label in sorted({t.label for t in takes})},  # fmt: skip
        "protocols": {k: v["meaning"] for k, v in PROTOCOLS.items()},
        "hashes": {**_hashes(run_dir, split), "compare_grid_hash": compare_hash(),
                   "candidates": candidate_info,
                   "acceptance_criteria_hash": AcceptanceCriteria().hash(),
                   "source_hash_combined": hashlib.sha256(
                       json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest(),
                   "review_items_sha256": review["items_sha256"] if review else None},
        "systems": {}, "shift_profiles": {},
    }  # fmt: skip
    reference_candidates: dict[str, list[dict[str, Any]]] = {}
    all_scores = {}
    for name, classifier, _ in systems:
        scores = score_system(name, classifier, hasattr(classifier, "threshold"), windows, cases)
        all_scores[name] = scores
        report["shift_profiles"][name] = {
            truth: shift_profile(scores.corpus, truth)
            for truth in ("circle_clockwise", "swipe_left", "swipe_right")}  # fmt: skip
        for mechanism in MECHANISMS:
            reference_candidates[f"{name}|{mechanism}"] = candidate_rows(
                reference_result(scores, mechanism)[0])  # fmt: skip
    for name, scores in all_scores.items():
        report["systems"][name] = {
            mechanism: summarize_variant(
                scores, mechanism,
                tally_for(f"{name}|{mechanism}", reference_candidates[f"{name}|{mechanism}"],
                          review))  # fmt: skip
            for mechanism in MECHANISMS
        }

    out = _next_dir(runs_dir / "streaming", "compare")
    items, key = export_review(reference_candidates, config.window_s)
    review_out = out / "review"
    review_out.mkdir()
    (review_out / "review_items.json").write_text(json.dumps(items, indent=2) + "\n",
                                                  encoding="utf-8")  # fmt: skip
    (review_out / "review_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    report["review_export"] = {"items": len(items), "folder": review_out.name,
                               "items_sha256": file_sha256(review_out / "review_items.json"),
                               "note": "blind items for the candidates of every reference "
                                       "setting; review them with stream-review"}  # fmt: skip
    report["review_source"] = (
        describe_review_source(review, items, report["review_export"]["items_sha256"])
        if review is not None else None)
    digest = _dump(out, report)
    (out / "report.md").write_text(render_compare(report), encoding="utf-8")
    print(f"Report: {out / 'report.md'}\nmetrics.json sha256 {digest}\n"
          f"{len(items)} candidates to review: {review_out}")  # fmt: skip
    return EXIT_OK


# ------------------------------------------------------------------------------- freeze
def _freeze(args: argparse.Namespace) -> int:
    from nocursor.dynamic.gate import GateConfig
    from nocursor.dynamic.stabilizer import StabilizerConfig
    from nocursor.dynamic.stream_accept import create_freeze
    from nocursor.dynamic.stream_eval import DecisionConfig

    dataset_dir, runs_dir, run_dir = _paths(args)
    report = json.loads(args.tuning_report.read_text(encoding="utf-8"))
    chosen = report.get("chosen")
    if not chosen:
        raise ValueError("the tuning report has no chosen setting; nothing to freeze")
    decision = DecisionConfig(report["system"], GateConfig(**chosen["gate"]),
                              StabilizerConfig(**chosen["stabilizer"]))  # fmt: skip
    if decision.system != args.primary:
        raise ValueError("this command freezes one tuned system; --primary must match it")
    final_dir = args.final_dir or dataset_dir / "streaming" / "final"
    path = create_freeze(runs_dir, run_dir, [decision], args.primary, final_dir,
                         tuning_report=args.tuning_report)  # fmt: skip
    print(f"Frozen: {path}\nconfig hash {decision.hash()}\nRecord the final data only now.")
    return EXIT_OK


# ------------------------------------------------------------------------------- accept
def _accept(args: argparse.Namespace) -> int:
    from nocursor.dynamic.stream_accept import accept_finalize, accept_run

    dataset_dir: Path = args.dataset_dir or default_dataset_dir()
    runs_dir: Path = args.runs_dir or default_ml_runs_dir()
    candidate = Path(args.freeze)
    freeze_dir = candidate if candidate.is_dir() else runs_dir / "streaming" / args.freeze
    if args.finalize:
        print(f"Verdict: {accept_finalize(freeze_dir)}")
        return EXIT_OK
    run = Path(args.run)
    run_dir = run if run.is_dir() else runs_dir / args.run
    final_dir = args.final_dir or dataset_dir / "streaming" / "final"
    out = accept_run(freeze_dir, run_dir, final_dir)
    print(f"Candidates written to {out}. Review them (mark every item in review_items.json as "
          "false, genuine or ambiguous in review_marks.json), then run with --finalize.")
    return EXIT_OK
