"""Command line for splitting, training and evaluating: ``split``, ``fit-baseline``, ``train``,
``evaluate``. Only ``train`` needs PyTorch, and it imports it only when it actually runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from nocursor.dynamic.collect import EXIT_ERROR, EXIT_OK, EXIT_USAGE
from nocursor.dynamic.schema import CLASS_NAMES, DatasetError
from nocursor.paths import default_dataset_dir, default_ml_runs_dir

ML_COMMANDS = ("split", "fit-baseline", "train", "evaluate")


def _common(sub: argparse.ArgumentParser, *, needs_dataset: bool = True) -> None:
    if needs_dataset:
        sub.add_argument("--dataset-dir", type=Path, metavar="DIR",
                         help=f"recorded sessions (default: {default_dataset_dir().name}/)")
    sub.add_argument("--runs-dir", type=Path, metavar="DIR",
                     help=f"splits, models and reports (default: {default_ml_runs_dir().name}/)")


def add_ml_commands(commands: Any) -> None:
    split = commands.add_parser(
        "split", help="freeze a session-grouped train/validation/test split",
        description="Assign whole sessions to train, validation and test with a seeded shuffle, "
                    "verify there is no leakage, and write the frozen split manifest.",
    )  # fmt: skip
    _common(split)
    split.add_argument("--seed", type=int, default=0, help="shuffle seed (default 0)")
    split.add_argument("--validation-sessions", type=int, default=1,
                       help="full sessions held out for validation (default 1)")
    split.add_argument("--test-sessions", type=int, default=1,
                       help="full sessions held out for test (default 1)")

    for name, text in (("fit-baseline", "tune the rule baseline (needs no PyTorch)"),
                       ("train", "tune the baseline and train the GRU (needs PyTorch)")):
        sub = commands.add_parser(name, help=text, description=text + " on the training and "
                                  "validation splits only; the test split and the stream are "
                                  "never opened.")  # fmt: skip
        _common(sub)
        sub.add_argument("--split", required=True, metavar="ID_OR_FILE",
                         help="a split id from `split`, or the path of its split.json")
        sub.add_argument("--run-name", metavar="NAME",
                         help="name of the run folder (default: run-<UTC time>)")
        sub.add_argument("--exclude-flagged", action="store_true",
                         help="sensitivity run: leave the flagged takes out of training and "
                              "validation (they are kept by default)")
        sub.add_argument("--allow-synthetic", action="store_true",
                         help="software tests only: permit synthetic sessions; outputs are "
                              "stamped SYNTHETIC")
        if name == "train":
            sub.add_argument("--seeds", type=int, default=5,
                             help="seeds per setting, 0..N-1 (default 5)")
            sub.add_argument("--max-epochs", type=int, default=150,
                             help="epoch limit per model (default 150)")
            sub.add_argument("--hidden-sizes", type=int, nargs="+", default=[32, 64],
                             help="GRU hidden sizes to choose between (default 32 64)")

    evaluate = commands.add_parser(
        "evaluate", help="score the baseline and the GRU on the test split and the stream",
        description="Score every system in a run on the same held-out test takes, then on the "
                    "reserved continuous stream. Writes metrics.json and report.md.",
    )  # fmt: skip
    _common(evaluate)
    evaluate.add_argument("--run", required=True, metavar="DIR_OR_NAME",
                          help="a run folder, or its name under the runs directory")
    evaluate.add_argument("--sensitivity-run", metavar="DIR_OR_NAME",
                          help="a run trained with --exclude-flagged on the same split, to "
                               "compare with")
    evaluate.add_argument("--no-stream", action="store_true",
                          help="skip the continuous-stream analysis")


def _resolve_run(reference: str, runs_dir: Path) -> Path:
    candidate = Path(reference)
    path = candidate if candidate.is_dir() else runs_dir / reference
    if not path.is_dir():
        raise DatasetError(f"run not found: {reference}")
    return path


def run_ml_command(args: argparse.Namespace) -> int:
    from nocursor.dynamic.evaluate import EvaluationError
    from nocursor.dynamic.model_io import ModelArtifactError
    from nocursor.dynamic.splits import SplitError
    from nocursor.dynamic.train import TorchUnavailableError, TrainingDataError

    dataset_dir: Path = args.dataset_dir or default_dataset_dir()
    runs_dir: Path = args.runs_dir or default_ml_runs_dir()
    try:
        if args.command == "split":
            return _split(args, dataset_dir, runs_dir)
        if args.command == "evaluate":
            return _evaluate(args, dataset_dir, runs_dir)
        return _fit(args, dataset_dir, runs_dir)
    except TorchUnavailableError as exc:
        print(f"nocursor.dynamic: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (SplitError, TrainingDataError, EvaluationError, ModelArtifactError, DatasetError,
            ValueError) as exc:  # fmt: skip
        print(f"nocursor.dynamic: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _split(args: argparse.Namespace, dataset_dir: Path, runs_dir: Path) -> int:
    from nocursor.dynamic.splits import (
        SplitConfig,
        create_split,
        load_for_split,
        verify_split,
        write_split,
    )

    config = SplitConfig(seed=args.seed, validation_sessions=args.validation_sessions,
                         test_sessions=args.test_sessions)  # fmt: skip
    dataset = load_for_split(dataset_dir)
    split = create_split(dataset, config)
    verify_split(split, dataset)  # leakage, coverage and a fresh recomputation
    path, new = write_split(split, runs_dir)
    print(f"Split {split['split_id']} (seed {split['seed']}) "
          f"{'written and frozen' if new else 'already frozen, unchanged'}: {path}")
    print("Verified: sessions and sequences disjoint, identical landmark data not shared, pilots "
          "and the stream kept out of validation and test, all four classes in every split.")
    for role, ids in split["assignments"].items():
        print(f"  {role:<16}{', '.join(ids) or '-'}")
    print("Feature-ready takes per class "
          f"({', '.join(CLASS_NAMES)}):")  # fmt: skip
    for name, counts in split["counts"].items():
        print(f"  {name:<12}{[counts[c] for c in CLASS_NAMES]}  dates {split['dates'][name]}")
    print(f"Excluded from model inputs: {len(split['excluded'])} takes (tracking not reliable "
          f"enough for features). Flagged for label review, kept: {len(split['flagged'])}.")
    for warning in split["warnings"]:
        print(f"  warning: {warning}")
    return EXIT_OK


def _fit(args: argparse.Namespace, dataset_dir: Path, runs_dir: Path) -> int:
    from nocursor.dynamic.splits import read_split
    from nocursor.dynamic.train import TrainConfig, run_fit_baseline, run_train

    split, split_file = read_split(args.split, runs_dir)
    extra: dict[str, Any] = {"exclude_flagged": args.exclude_flagged,
                             "allow_synthetic": args.allow_synthetic}  # fmt: skip
    if args.command == "train":
        if args.seeds < 1 or args.max_epochs < 1 or min(args.hidden_sizes) < 1:
            print("nocursor.dynamic: --seeds, --max-epochs and --hidden-sizes must be >= 1",
                  file=sys.stderr)  # fmt: skip
            return EXIT_USAGE
        config = TrainConfig(hidden_sizes=tuple(args.hidden_sizes), seeds=tuple(range(args.seeds)),
                             max_epochs=args.max_epochs, **extra)  # fmt: skip
        run_dir = run_train(split, split_file, dataset_dir, runs_dir, args.run_name, config)
    else:
        run_dir = run_fit_baseline(split, split_file, dataset_dir, runs_dir, args.run_name,
                                   TrainConfig(**extra))  # fmt: skip
    print(f"Run saved: {run_dir}")
    return EXIT_OK


def _evaluate(args: argparse.Namespace, dataset_dir: Path, runs_dir: Path) -> int:
    from nocursor.dynamic.evaluate import evaluate_run

    run_dir = _resolve_run(args.run, runs_dir)
    other = _resolve_run(args.sensitivity_run, runs_dir) if args.sensitivity_run else None
    out_dir = evaluate_run(run_dir, dataset_dir, sensitivity_run=other,
                           include_stream=not args.no_stream)  # fmt: skip
    print(f"Report: {out_dir / 'report.md'}")
    return EXIT_OK
