"""Partial-window training samples, and a separately named candidate model trained with them.

Why: the GRU was trained on whole 1.6 s takes that start when a gesture starts. In continuous use
the window slides, so most windows hold only *part* of a movement. A half circle looks like a
swipe, and the replay evaluation shows exactly that (circles read as swipes). This module builds
what the model never saw: windows that are shifted so that a gesture is cut off at one end.

* ``partial_negative``: the window is shifted by 0.5 s or 0.8 s either way, so at least about a
  third of the gesture is outside it. Labelled ``no_gesture``: an incomplete gesture must not fire.
* ``aligned_positive``: shifted by 0.1 s either way. Still the full gesture, labelled as such, so
  the model keeps tolerating small timing differences.

Shifted windows are cut from the take with its first and last frame held still on either side (the
same padding the replay uses). Only training and validation takes are used; the test split and
the continuous stream are never opened here, and nothing is mined from the 120 s stream, because
with one development recording it could not then be evaluated honestly.

The candidate is a new model in ``ml_runs/streaming/candidates/<name>/``. The ``primary`` run, its
weights and every historical artifact are left alone. Validation takes (and their shifted windows)
drive early stopping and the rejection threshold, as for the primary model, so results on those
takes are not independent evidence. Offline only: nothing is connected to the mouse.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nocursor.dynamic.features import (
    FeatureConfig,
    FeatureError,
    features_from_resampled,
    resample_view,
)
from nocursor.dynamic.schema import MIN_FRAMES, NEGATIVE_LABEL, TARGET_CLASSES
from nocursor.dynamic.splits import (
    SPLIT_FILE,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SplitError,
    SplitItem,
    verify_manifest_and_files,
)
from nocursor.dynamic.stream_eval import pad_take, slice_window

PARTIAL_NEGATIVE_SHIFTS_S = (-0.8, -0.5, 0.5, 0.8)
ALIGNED_POSITIVE_SHIFTS_S = (-0.1, 0.1)
HOLD_S = 1.0
KIND_NEGATIVE = "partial_negative"
KIND_POSITIVE = "aligned_positive"
CANDIDATES_DIR = ("streaming", "candidates")


@dataclass
class PartialSamples:
    pts: np.ndarray  # (N, steps, 21, 2) resampled landmarks, for augmentation
    features: np.ndarray  # (N, steps, FEATURE_DIM)
    labels: list[str]
    kinds: list[str]
    sources: list[str]
    shifts: list[float]
    skipped: int

    def __len__(self) -> int:
        return len(self.labels)


def shifted_window(take: Any, shift_s: float, config: FeatureConfig) -> Any | None:
    """The take's window moved by ``shift_s`` seconds against the gesture; None if too short."""
    padded, start, _ = pad_take(take, HOLD_S, HOLD_S)
    t = (padded.t_ms - padded.t_ms[0]) / 1000.0
    begin = start + shift_s
    lo = int(np.searchsorted(t, begin, side="left"))
    hi = int(np.searchsorted(t, begin + config.window_s, side="right"))
    if lo < 0 or hi - lo < MIN_FRAMES:
        return None
    return slice_window(padded, t, lo, hi, begin)


def build_partial_samples(
    items: Sequence[SplitItem], config: FeatureConfig,
    negative_shifts: Sequence[float] = PARTIAL_NEGATIVE_SHIFTS_S,
    positive_shifts: Sequence[float] = ALIGNED_POSITIVE_SHIFTS_S,
) -> PartialSamples:  # fmt: skip
    pts: list[np.ndarray] = []
    feats: list[np.ndarray] = []
    labels, kinds, sources, shifts = [], [], [], []
    skipped = 0
    plan = [(s, KIND_NEGATIVE) for s in negative_shifts] + [
        (s, KIND_POSITIVE) for s in positive_shifts]  # fmt: skip
    for item in items:
        take = item.sequence
        if take.label not in TARGET_CLASSES:
            continue  # only gestures are cut; ordinary negatives are used as recorded
        for shift, kind in plan:
            window = shifted_window(take, shift, config)
            if window is None:
                skipped += 1
                continue
            try:
                resampled = resample_view(window, config)
                feature = features_from_resampled(resampled, config)
            except FeatureError:
                skipped += 1
                continue
            pts.append(resampled)
            feats.append(feature)
            labels.append(NEGATIVE_LABEL if kind == KIND_NEGATIVE else take.label)
            kinds.append(kind)
            sources.append(take.sequence_id)
            shifts.append(shift)
    return PartialSamples(
        np.stack(pts) if pts else np.zeros((0, config.steps, 21, 2)),
        np.stack(feats) if feats else np.zeros((0, config.steps, 46)),
        labels, kinds, sources, shifts, skipped,
    )  # fmt: skip


def counts_by_kind(samples: PartialSamples) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for kind, label in zip(samples.kinds, samples.labels, strict=True):
        row = out.setdefault(kind, {})
        row[label] = row.get(label, 0) + 1
    return out


# ------------------------------------------------------------------------------ training
def run_partial_train(
    split: dict[str, Any], split_file: Path, dataset_dir: Path, runs_dir: Path, name: str,
    config: Any, *, parent_run: Path | None = None, log: Callable[[str], object] = print,
) -> Path:  # fmt: skip
    """Train a candidate on whole takes plus partial-window samples. Needs PyTorch (lazily)."""
    from nocursor.dynamic import train as T
    from nocursor.dynamic.model_io import load_classifier, save_artifact
    from nocursor.dynamic.splits import load_split_items

    torch = T._import_torch()
    started = time.perf_counter()
    feature_config = FeatureConfig(**split["feature_config"])
    verify_manifest_and_files(split, dataset_dir)
    provenance = T._check_provenance(split, config.allow_synthetic)
    # Only the train and validation splits are ever requested; the test split cannot be loaded
    # without a flag this code never passes.
    train_items = load_split_items(split, dataset_dir, [SPLIT_TRAIN],
                                   exclude_flagged=config.exclude_flagged)  # fmt: skip
    val_items = load_split_items(split, dataset_dir, [SPLIT_VALIDATION],
                                 exclude_flagged=config.exclude_flagged)  # fmt: skip
    target = runs_dir.joinpath(*CANDIDATES_DIR) / name
    if target.exists():
        raise SplitError(f"candidate {name} already exists; choose another --name")

    base_train = T._arrays(train_items, feature_config)
    base_val = T._arrays(val_items, feature_config)
    extra_train = build_partial_samples(train_items, feature_config)
    extra_val = build_partial_samples(val_items, feature_config)
    classes = T.CLASS_NAMES

    def merge(base: Any, extra: PartialSamples) -> Any:
        labels = np.array([classes.index(label) for label in extra.labels], dtype=int)
        return T._Arrays(np.concatenate([base.pts, extra.pts]),
                         np.concatenate([base.features, extra.features]),
                         np.concatenate([base.labels, labels]))  # fmt: skip

    train, val = merge(base_train, extra_train), merge(base_val, extra_val)
    flat = train.features.reshape(-1, train.features.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32).astype(np.float64)
    std = np.maximum(flat.std(axis=0), T.STD_FLOOR).astype(np.float32).astype(np.float64)
    counts = {
        "train_takes": T._count(train_items), "validation_takes": T._count(val_items),
        "train_partial_samples": counts_by_kind(extra_train),
        "validation_partial_samples": counts_by_kind(extra_val),
        "partial_samples_skipped": {"train": extra_train.skipped,
                                    "validation": extra_val.skipped},  # fmt: skip
        "training_samples_total": int(len(train.labels)),
        "validation_samples_total": int(len(val.labels)),
    }  # fmt: skip
    log(f"candidate {name}: {len(train_items)} training takes + {len(extra_train)} partial "
        f"samples; {len(val_items)} validation takes + {len(extra_val)} partial samples")

    results = []
    for hidden in config.hidden_sizes:
        for weight_decay in config.weight_decays:
            for seed in config.seeds:
                r = T._train_one(torch, hidden, weight_decay, seed, train, val, mean, std,
                                 config, feature_config)  # fmt: skip
                results.append(r)
                log(f"  hidden {hidden:>3} wd {weight_decay:g} seed {seed}: epoch "
                    f"{r['selected_epoch']:>3}/{r['epochs_run']:>3}  val loss "
                    f"{r['best_val_loss']:.3f}  val macro-F1 {r['val_macro_f1']:.3f}  "
                    f"tau {r['calibration']['threshold']:.2f}")  # fmt: skip

    groups: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for r in results:
        groups.setdefault((r["hidden"], r["weight_decay"]), []).append(r)
    ranked_groups = sorted(groups.items(), key=lambda kv: (
        -float(np.mean([r["val_macro_f1"] for r in kv[1]])),
        float(np.mean([r["best_val_loss"] for r in kv[1]])), kv[0]))  # fmt: skip
    (hidden, weight_decay), chosen_runs = ranked_groups[0]
    final = sorted(chosen_runs, key=lambda r: (-r["val_macro_f1"], r["best_val_loss"],
                                               r["seed"]))[0]  # fmt: skip

    metadata = T._metadata(final, split, feature_config, config, provenance, torch.__version__,
                           {"train": counts["train_takes"],
                            "validation": counts["validation_takes"]}, "candidate")  # fmt: skip
    parent_hash = None
    if parent_run is not None and (parent_run / "model.json").is_file():
        parent_hash = json.loads((parent_run / "model.json").read_text("utf-8"))["weights_sha256"]
    metadata["candidate"] = {
        "name": name, "kind": "partial-window training",
        "parent_run": None if parent_run is None else parent_run.name,
        "parent_weights_sha256": parent_hash,
        "partial_negative_shifts_s": list(PARTIAL_NEGATIVE_SHIFTS_S),
        "aligned_positive_shifts_s": list(ALIGNED_POSITIVE_SHIFTS_S), "hold_s": HOLD_S,
        "sample_counts": counts,
        "disclosures": [
            "validation takes and their shifted windows were used for early stopping, model "
            "selection and the rejection threshold, so results on them are not independent",
            "the held-out test session and the continuous stream were not opened",
            "trained on one participant's recordings only",
        ],
    }  # fmt: skip
    save_artifact(target, T._artifact_arrays(final["_state"], mean, std), metadata)
    classifier = load_classifier(target)
    parity = float(max(np.abs(classifier.logits(val.features[i]) - final["_val_logits"][i]).max()
                       for i in range(len(val.labels))))  # fmt: skip
    log(f"NumPy-versus-PyTorch max |logit difference| on validation samples: {parity:.2e}")
    if parity > 1e-3:
        shutil.rmtree(target)
        raise RuntimeError(f"NumPy and PyTorch disagree by {parity:.2e}; not keeping the model")
    shutil.copyfile(split_file, target / SPLIT_FILE)
    (target / T.TRAIN_LOG).write_text(json.dumps({
        "candidate": metadata["candidate"], "config": config.to_dict(),
        "selected": {"hidden": hidden, "weight_decay": weight_decay, "final_seed": final["seed"],
                     "selected_epoch": final["selected_epoch"],
                     "rejection": final["calibration"]},
        "selection": [{"hidden": h, "weight_decay": wd,
                       "mean_val_macro_f1": float(np.mean([r["val_macro_f1"] for r in rs]))}
                      for (h, wd), rs in ranked_groups],
        "numpy_vs_torch_max_abs_logit_difference_validation": parity,
        "libraries": T._libraries(torch.__version__),
        "seconds": round(time.perf_counter() - started, 1),
    }, indent=2) + "\n", encoding="utf-8")  # fmt: skip
    return target
