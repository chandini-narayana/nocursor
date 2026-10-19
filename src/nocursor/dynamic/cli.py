"""Command line for the dynamic-gesture tools: ``python -m nocursor.dynamic <command>``.

Data: ``collect`` (record landmark sequences), ``summarize`` and ``validate`` (inspect a dataset).
Modelling: ``split``, ``fit-baseline``, ``train`` and ``evaluate`` (see ``cli_ml``). Nothing here is
connected to the mouse.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from nocursor.dynamic.cli_ml import ML_COMMANDS, add_ml_commands, run_ml_command
from nocursor.dynamic.cli_stream import STREAM_COMMANDS, add_stream_commands, run_stream_command
from nocursor.dynamic.collect import (
    EXIT_ERROR,
    EXIT_INTERRUPTED,
    EXIT_OK,
    EXIT_USAGE,
    CollectOptions,
    run_collect,
)
from nocursor.dynamic.dataset import DatasetError, load_dataset
from nocursor.dynamic.quality import DEFAULT_MIN_PALM_FRACTION
from nocursor.dynamic.schema import HANDS, RECOMMENDED_NEGATIVE_TAKES_PER_SESSION
from nocursor.dynamic.schema import RECOMMENDED_TARGET_TAKES_PER_SESSION as _TARGET_PER_SESSION
from nocursor.dynamic.summary import format_summary, summarize
from nocursor.paths import default_dataset_dir

EPILOG = f"""\
examples:
  python -m nocursor.dynamic collect --participant p01 --hand right
      one recording session with the guided protocol
  python -m nocursor.dynamic collect --stream-seconds 120
      one continuous recording of ordinary use (negative examples)
  python -m nocursor.dynamic summarize        what has been recorded, and is it enough?
  python -m nocursor.dynamic validate         check every file; exit code 1 if any is damaged
  python -m nocursor.dynamic split            freeze the train/validation/test split by session
  python -m nocursor.dynamic evaluate --run NAME     score a trained run on the test split

Only landmark numbers are saved (never video). Files go to {default_dataset_dir().name}/, which
is excluded from version control. See docs/DYNAMIC_GESTURES.md.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nocursor.dynamic",
        description="Collect and inspect hand-landmark sequences for dynamic gesture research.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    collect = commands.add_parser(
        "collect", help="record labelled landmark sequences from the webcam",
        description="Record labelled landmark sequences. Landmarks only: no video is saved.",
    )  # fmt: skip
    collect.add_argument("--dataset-dir", type=Path, metavar="DIR",
                         help=f"where session files go (default: {default_dataset_dir().name}/)")
    collect.add_argument("--participant", default="p01", metavar="ID",
                         help="pseudonymous participant id, e.g. p01 (never a name; default p01)")
    collect.add_argument("--hand", choices=HANDS, default="unspecified",
                         help="the hand you record with (default unspecified)")
    collect.add_argument("--conditions", default="", metavar="TEXT",
                         help="one short line about the setup, e.g. 'daylight, 50 cm'. "
                              "Do not include personal information")
    collect.add_argument("--camera", type=int, default=0, dest="camera_index",
                         help="camera index (default 0)")
    collect.add_argument("--width", type=int, default=640, dest="frame_width",
                         help="capture width in pixels (default 640)")
    collect.add_argument("--height", type=int, default=480, dest="frame_height",
                         help="capture height in pixels (default 480)")
    collect.add_argument("--model", type=Path, dest="model_path", metavar="PATH",
                         help="path to hand_landmarker.task (default: models/ in the project)")
    collect.add_argument("--min-confidence", type=float, default=0.6,
                         help="minimum hand-detection confidence (default 0.6)")
    collect.add_argument("--countdown", type=float, default=2.0, dest="countdown_s",
                         help="seconds of countdown before each take (default 2, valid 0 to 10)")
    collect.add_argument("--per-class", type=int, default=_TARGET_PER_SESSION,
                         help="takes of each target gesture in the protocol "
                              f"(default {_TARGET_PER_SESSION}, valid 1 to 500)")
    collect.add_argument("--negatives", type=int, default=RECOMMENDED_NEGATIVE_TAKES_PER_SESSION,
                         help="no_gesture takes in the protocol (default "
                              f"{RECOMMENDED_NEGATIVE_TAKES_PER_SESSION}, valid 0 to 2000)")
    collect.add_argument("--seed", type=int, default=None,
                         help="seed for the order of the protocol (default: random, recorded in "
                              "the session file)")
    collect.add_argument("--free", action="store_true",
                         help="no protocol: choose each take's label yourself (keys 1-4, T)")
    collect.add_argument("--stream-seconds", type=float, default=None,
                         help="record ONE continuous recording of ordinary use instead of takes "
                              "(valid 5 to 600 seconds)")
    collect.add_argument("--min-palm-fraction", type=float, default=DEFAULT_MIN_PALM_FRACTION,
                         help="share of tracked frames that must be an open palm in a "
                              f"gesture take (default {DEFAULT_MIN_PALM_FRACTION:g}, valid "
                              "above 0 up to 1)")

    for name, text in (("summarize", "counts, coverage and label review for a dataset"),
                       ("validate", "check every dataset file and list problems")):  # fmt: skip
        sub = commands.add_parser(name, help=text, description=text)
        sub.add_argument("--dataset-dir", type=Path, metavar="DIR",
                         help=f"dataset folder (default: {default_dataset_dir().name}/)")
        if name == "summarize":
            sub.add_argument("--json", action="store_true", dest="as_json",
                             help="print the summary as JSON")
    add_ml_commands(commands)
    add_stream_commands(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.command in ML_COMMANDS:
            return run_ml_command(args)
        if args.command in STREAM_COMMANDS:
            return run_stream_command(args)
        directory: Path = args.dataset_dir or default_dataset_dir()
        if args.command == "collect":
            return _collect(args, directory)
        if args.command == "summarize":
            return _summarize(directory, args.as_json)
        return _validate(directory)
    except KeyboardInterrupt:
        print("\nnocursor.dynamic: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


def _collect(args: argparse.Namespace, directory: Path) -> int:
    options = CollectOptions(
        dataset_dir=directory, participant=args.participant, hand=args.hand,
        conditions=args.conditions, camera_index=args.camera_index,
        frame_width=args.frame_width, frame_height=args.frame_height,
        model_path=args.model_path, min_confidence=args.min_confidence,
        countdown_s=args.countdown_s, per_class=args.per_class, negatives=args.negatives,
        seed=args.seed, free=args.free, stream_seconds=args.stream_seconds,
        min_palm_fraction=args.min_palm_fraction,
    )  # fmt: skip
    try:
        return run_collect(options, out=_print)
    except DatasetError as exc:  # e.g. an invalid participant id
        print(f"nocursor.dynamic: {exc}", file=sys.stderr)
        return EXIT_USAGE


def _print(text: str) -> None:
    print(text, file=sys.stderr if text.startswith("nocursor.dynamic:") else sys.stdout)


def _summarize(directory: Path, as_json: bool) -> int:
    try:
        dataset = load_dataset(directory, strict=False)
    except DatasetError as exc:
        print(f"nocursor.dynamic: {_with_hint(exc)}", file=sys.stderr)
        return EXIT_ERROR
    summary = summarize(dataset)
    print(json.dumps(summary, indent=2) if as_json else format_summary(summary))
    for issue in dataset.errors:
        print(issue, file=sys.stderr)
    return EXIT_OK


def _validate(directory: Path) -> int:
    try:
        dataset = load_dataset(directory, strict=False)
    except DatasetError as exc:
        print(f"nocursor.dynamic: {_with_hint(exc)}", file=sys.stderr)
        return EXIT_ERROR
    for issue in dataset.issues:
        print(issue)
    sequences = sum(len(s.sequences) for s in dataset.sessions)
    print(f"{len(dataset.sessions)} session file(s), {sequences} valid sequence(s), "
          f"{len(dataset.errors)} error(s), {len(dataset.warnings)} warning(s)")  # fmt: skip
    return EXIT_ERROR if dataset.errors else EXIT_OK


def _with_hint(error: DatasetError) -> str:
    text = str(error)
    if text.startswith("dataset directory not found"):
        text += " (nothing has been recorded yet; start with: python -m nocursor.dynamic collect)"
    return text
