"""A SYNTHETIC study for the split, baseline, model, training and evaluation tests.

Everything here is generated geometry marked ``provenance="synthetic"``. It lets the software be
tested end to end (splits, training, evaluation, reports) and says nothing about how real hands,
cameras or gesture classifiers behave. Results computed from it must never be quoted as accuracy.

The layout mirrors the real recordings: several full sessions, a small pilot session and one
continuous-use stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dynamic_helpers import circle_path, make_sequence, still_path, swipe_path
from nocursor.dynamic.dataset import SessionWriter, load_dataset
from nocursor.dynamic.features import FEATURE_DIM, FeatureConfig
from nocursor.dynamic.model_io import MODEL_TYPE, save_artifact
from nocursor.dynamic.schema import CLASS_NAMES, SequenceRecord, SessionHeader
from nocursor.dynamic.splits import SplitConfig, create_split, write_split

FULL_SESSIONS = ("s-full-a", "s-full-b", "s-full-c", "s-full-d", "s-full-e", "s-full-f")
PILOT = "s-pilot-a"
STREAM = "s-stream-a"


def _header(session_id: str, day: int = 12) -> SessionHeader:
    return SessionHeader(session_id, "p01", f"2026-10-{day:02d}T10:00:00Z", hand="right",
                         provenance="synthetic")  # fmt: skip


def _negative(kind: int, k: int):
    """Three kinds of negative: still, a small sideways wiggle, a vertical sweep."""
    if kind == 0:
        return still_path(), "idle"
    if kind == 1:
        amplitude = 0.015 + 0.001 * k
        return (lambda t: (amplitude * np.sin(6 * np.pi * t), 0.0)), "pointing"
    return (lambda t: (0.0, 0.2 * (2 * t - 1))), "open_palm_other"


def write_session(
    directory: Path, session_id: str, index: int, *, takes: int = 15, negatives: int = 20,
    bad_negatives: int = 0, flagged_swipe: bool = False, day: int = 12,
) -> None:  # fmt: skip
    """One synthetic session; every take differs a little, so no two share landmark data."""
    writer = SessionWriter(directory, _header(session_id, day))
    n = 0

    def add(path, label, subtype=None, **kwargs):
        nonlocal n
        n += 1
        writer.append(make_sequence(
            path, label=label, subtype=subtype, sequence_id=f"{session_id}-{n:04d}",
            session_id=session_id, hand_size=0.2 - 0.002 * index - 0.0002 * n, **kwargs,
        ))  # fmt: skip

    for k in range(takes):
        amplitude = 0.16 + 0.002 * k + 0.0007 * index
        add(swipe_path(-1, amplitude), "swipe_left")
        add(swipe_path(+1, amplitude), "swipe_right")
        add(circle_path(True, 0.1 + 0.002 * k + 0.0005 * index), "circle_clockwise")
    for k in range(negatives):
        path, subtype = _negative(k % 3, k)
        add(path, "no_gesture", subtype)
    frames = make_sequence(still_path()).n_frames
    for _ in range(bad_negatives):  # tracking half lost: cannot become features
        add(still_path(), "no_gesture", "entering_leaving", usable=np.arange(frames) % 2 == 0)
    if flagged_swipe:  # a very small sweep labelled as a swipe: the label review flags it
        add(swipe_path(+1, 0.01), "swipe_right")
    writer.close()


def write_stream_session(directory: Path, session_id: str = STREAM, seconds: float = 8.0) -> None:
    writer = SessionWriter(directory, _header(session_id))
    base = make_sequence(still_path(), duration=seconds, sequence_id=f"{session_id}-0001",
                         session_id=session_id, label="no_gesture", hand_size=0.19)  # fmt: skip
    writer.append(SequenceRecord(
        sequence_id=base.sequence_id, session_id=session_id, label="no_gesture",
        subtype="continuous_use", kind="stream", t_ms=base.t_ms, landmarks=base.landmarks,
        confidence=base.confidence, usable=base.usable, aspect_ratio=base.aspect_ratio,
    ))  # fmt: skip
    writer.close()


def build_study(directory: Path, *, with_stream: bool = True, takes: int = 15) -> Path:
    """Six full sessions (one with excluded negatives, one flagged take), a pilot and a stream."""
    for i, sid in enumerate(FULL_SESSIONS):
        write_session(directory, sid, i, takes=takes, bad_negatives=2 if i == 1 else 0,
                      flagged_swipe=i == 2, day=12 + i // 3)  # fmt: skip
    write_session(directory, PILOT, 9, takes=2, negatives=4)
    if with_stream:
        write_stream_session(directory)
    return directory


def freeze_split(dataset_dir: Path, runs_dir: Path, seed: int = 0, **config: Any):
    """Create and freeze a split; returns (split dict, path)."""
    dataset = load_dataset(dataset_dir)
    split = create_split(dataset, SplitConfig(seed=seed, **config), FeatureConfig(),
                         created_utc="2026-10-13T00:00:00Z")  # fmt: skip
    path, _ = write_split(split, runs_dir)
    return split, path


def random_arrays(hidden: int = 6, seed: int = 0, scale: float = 0.3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "gru.weight_ih": rng.normal(0, scale, (3 * hidden, FEATURE_DIM)),
        "gru.weight_hh": rng.normal(0, scale, (3 * hidden, hidden)),
        "gru.bias_ih": rng.normal(0, scale, 3 * hidden),
        "gru.bias_hh": rng.normal(0, scale, 3 * hidden),
        "fc.weight": rng.normal(0, scale, (len(CLASS_NAMES), hidden)),
        "fc.bias": rng.normal(0, scale, len(CLASS_NAMES)),
        "norm.mean": np.zeros(FEATURE_DIM),
        "norm.std": np.ones(FEATURE_DIM),
    }  # fmt: skip


def model_metadata(hidden: int = 6, threshold: float = 0.6, split: dict | None = None) -> dict:
    config = FeatureConfig()
    return {
        "model_type": MODEL_TYPE, "class_names": list(CLASS_NAMES),
        "feature_schema_version": 1, "feature_schema_hash": config.schema_hash(),
        "feature_config": config.to_dict(),
        "architecture": {"input_dim": FEATURE_DIM, "hidden_size": hidden, "num_layers": 1,
                         "num_classes": len(CLASS_NAMES)},  # fmt: skip
        "rejection": {"threshold": threshold, "method": "test fixture",
                      "validation_negatives": 10,
                      "validation_false_activations_at_threshold": 0},  # fmt: skip
        "seed": 0, "training": {"selected_epoch": 1, "config": {}},
        "libraries": {"numpy": np.__version__}, "data_provenance": "SYNTHETIC (software test only)",
        "created_utc": "2026-10-13T00:00:00Z",
        "split_id": split["split_id"] if split else "split-test",
        "dataset_manifest_hash": split["dataset_manifest_hash"] if split else "0" * 64,
    }  # fmt: skip


def add_random_gru(run_dir: Path, split: dict, *, hidden: int = 6, seed: int = 0,
                   threshold: float = 0.6) -> None:  # fmt: skip
    """Give a run directory a NumPy-only model with random weights (no PyTorch needed)."""
    save_artifact(run_dir, random_arrays(hidden, seed), model_metadata(hidden, threshold, split))
