"""Fitting the baseline and training the GRU on the training split; selection on validation only.

PyTorch is optional and is imported lazily, only inside the functions that actually train. Importing
this module (or the rest of NoCursor) never loads it, and ``fit-baseline`` needs no PyTorch at all.

What each split is used for:

* **train** (full training sessions plus pilot sessions): gradient updates, and the feature
  standardisation (mean and standard deviation), and nothing else.
* **validation**: early stopping, choosing among the few hyper-parameter settings, and calibrating
  the rejection threshold.
* **test** and the **continuous stream**: never opened here. Loading the test split needs a flag
  that only the evaluation code passes, and the training code verifies the frozen manifest against
  the files by checksum without parsing them.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nocursor.dynamic.augment import AugmentConfig, augment_resampled
from nocursor.dynamic.baseline import save_baseline, tune
from nocursor.dynamic.features import (
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
    FeatureError,
    features_from_resampled,
    resample_view,
)
from nocursor.dynamic.model_io import MODEL_TYPE, load_classifier, save_artifact
from nocursor.dynamic.schema import CLASS_NAMES, NEGATIVE_LABEL
from nocursor.dynamic.scoring import macro_f1_of
from nocursor.dynamic.splits import (
    SPLIT_FILE,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SplitError,
    SplitItem,
    load_split_items,
    verify_manifest_and_files,
)

TRAIN_LOG = "train_log.json"
N_CLASSES = len(CLASS_NAMES)
NEGATIVE_INDEX = CLASS_NAMES.index(NEGATIVE_LABEL)
STD_FLOOR = 0.01  # hand sizes: features varying less than this are noise, not signal


class TorchUnavailableError(RuntimeError):
    """PyTorch is not installed in this environment (it is only needed for training)."""


class TrainingDataError(ValueError):
    """The data selected for training is not acceptable."""


@dataclass(frozen=True)
class TrainConfig:
    hidden_sizes: tuple[int, ...] = (32, 64)
    weight_decays: tuple[float, ...] = (0.0, 1e-3)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    dropout: float = 0.2
    learning_rate: float = 2e-3
    batch_size: int = 32
    max_epochs: int = 150
    patience: int = 20
    min_delta: float = 1e-4
    grad_clip: float = 1.0
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    # Rejection-threshold grid: the smallest value leaving no validation negative triggering.
    tau_grid: tuple[float, ...] = tuple(round(0.5 + 0.01 * i, 2) for i in range(50)) + (0.999,)
    exclude_flagged: bool = False
    allow_synthetic: bool = False

    def __post_init__(self) -> None:
        if not self.hidden_sizes or not self.weight_decays or not self.seeds:
            raise ValueError("hidden_sizes, weight_decays and seeds must not be empty")
        if min(self.hidden_sizes) < 1 or self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError("sizes, batch size and epochs must be >= 1")
        if not (0.0 <= self.dropout < 1.0) or self.learning_rate <= 0 or self.patience < 1:
            raise ValueError("dropout must be in [0, 1); learning rate > 0; patience >= 1")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["tau_grid"] = [self.tau_grid[0], "...", self.tau_grid[-2], self.tau_grid[-1]]
        return out


# ---------------------------------------------------------------------------- run folders
def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _libraries(torch_version: str | None) -> dict[str, str | None]:
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch_version,
            "platform": platform.platform()}  # fmt: skip


def _new_run_dir(runs_dir: Path, run_name: str | None) -> Path:
    path = runs_dir / (run_name or f"run-{utc_stamp()}")
    if path.exists():
        raise SplitError(f"{path.name} already exists; choose another --run-name")
    path.mkdir(parents=True)
    return path


def _check_provenance(split: dict[str, Any], allow_synthetic: bool) -> str:
    synthetic = [r["session_id"] for r in split["sessions"] if r.get("provenance") == "synthetic"]
    if synthetic and not allow_synthetic:
        raise TrainingDataError(
            "the split contains synthetic sessions; training on them is refused unless "
            "--allow-synthetic is given, and then every output is stamped SYNTHETIC"
        )
    return "SYNTHETIC (software test only)" if synthetic else "recorded"


def _count(items: list[SplitItem]) -> dict[str, int]:
    return {name: sum(1 for i in items if i.sequence.label == name) for name in CLASS_NAMES}


def _labels(items: list[SplitItem]) -> np.ndarray:
    return np.array([CLASS_NAMES.index(i.sequence.label) for i in items], dtype=int)


def _load_training_data(
    split: dict[str, Any], dataset_dir: Path, exclude_flagged: bool
) -> tuple[list[SplitItem], list[SplitItem]]:
    train = load_split_items(split, dataset_dir, [SPLIT_TRAIN], exclude_flagged=exclude_flagged)
    val = load_split_items(split, dataset_dir, [SPLIT_VALIDATION], exclude_flagged=exclude_flagged)
    for name, items in (("training", train), ("validation", val)):
        counts = _count(items)
        missing = [c for c, n in counts.items() if n == 0]
        if missing:
            raise TrainingDataError(f"the {name} data has no {', '.join(missing)} takes")
    return train, val


def _prepare_run(
    split: dict[str, Any], split_file: Path, dataset_dir: Path, runs_dir: Path,
    run_name: str | None, config: TrainConfig,
) -> tuple[Path, list[SplitItem], list[SplitItem], str, dict[str, Any]]:  # fmt: skip
    verify_manifest_and_files(split, dataset_dir)
    provenance = _check_provenance(split, config.allow_synthetic)
    train, val = _load_training_data(split, dataset_dir, config.exclude_flagged)
    run_dir = _new_run_dir(runs_dir, run_name)
    shutil.copyfile(split_file, run_dir / SPLIT_FILE)  # the run keeps the frozen split with it
    tuning = [*train, *val]
    result = tune([i.features for i in tuning], [i.sequence.label for i in tuning])
    save_baseline(run_dir, result, split_id=split["split_id"], tuning_counts=_count(tuning),
                  exclude_flagged=config.exclude_flagged)  # fmt: skip
    summary = {"rule_defaults_macro_f1_on_tuning_data": result.default_objective,
               "rule_tuned_macro_f1_on_tuning_data": result.objective}  # fmt: skip
    return run_dir, train, val, provenance, summary


def run_fit_baseline(
    split: dict[str, Any], split_file: Path, dataset_dir: Path, runs_dir: Path,
    run_name: str | None = None, config: TrainConfig | None = None,
    log: Callable[[str], object] = print,
) -> Path:  # fmt: skip
    """Tune the rule baseline on train + validation takes. Needs no PyTorch."""
    config = config or TrainConfig()
    run_dir, train, val, _, summary = _prepare_run(split, split_file, dataset_dir, runs_dir,
                                                   run_name, config)  # fmt: skip
    log(f"baseline tuned on {len(train)} training + {len(val)} validation takes: "
        f"macro-F1 {summary['rule_defaults_macro_f1_on_tuning_data']:.3f} (defaults) -> "
        f"{summary['rule_tuned_macro_f1_on_tuning_data']:.3f} (tuned), on that same data")
    return run_dir


# ------------------------------------------------------------------------------ GRU pieces
@dataclass
class _Arrays:
    pts: np.ndarray  # (N, steps, 21, 2) resampled landmarks, for augmentation
    features: np.ndarray  # (N, steps, FEATURE_DIM)
    labels: np.ndarray


def _arrays(items: list[SplitItem], config: FeatureConfig) -> _Arrays:
    return _Arrays(
        pts=np.stack([resample_view(i.sequence, config) for i in items]),
        features=np.stack([i.features for i in items]),
        labels=_labels(items),
    )


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise TorchUnavailableError(
            "PyTorch is not installed in this environment. Training needs it; install it in a "
            "separate environment (see docs/DYNAMIC_GESTURES.md) and run training from there."
        ) from exc
    return torch


def _build_model(torch: Any, hidden: int, dropout: float) -> Any:
    nn = torch.nn

    class GRUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(FEATURE_DIM, hidden, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden, N_CLASSES)

        def forward(self, x: Any) -> Any:
            _, state = self.gru(x)
            return self.fc(self.dropout(state[-1]))

    return GRUNet()


def calibrate_threshold(
    probabilities: np.ndarray, labels: np.ndarray, grid: tuple[float, ...]
) -> dict[str, Any]:
    """Smallest threshold on ``grid`` at which no validation *negative* triggers a gesture.

    A negative triggers when its top class is a gesture with probability at or above the
    threshold. If even the largest value leaves some, that value is used and a warning recorded.
    """
    top = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    negatives = labels == NEGATIVE_INDEX
    is_gesture = top != NEGATIVE_INDEX

    def triggers(tau: float) -> int:
        return int((negatives & is_gesture & (confidence >= tau)).sum())

    chosen = next((tau for tau in grid if triggers(tau) == 0), grid[-1])
    accepted = is_gesture & (confidence >= chosen)
    effective = np.where(is_gesture & ~accepted, NEGATIVE_INDEX, top)
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(matrix, (labels, effective), 1)
    gestures = ~negatives
    return {
        "threshold": float(chosen),
        "method": "smallest value on the grid at which no validation negative triggers a gesture",
        "validation_negatives": int(negatives.sum()),
        "validation_false_activations_at_threshold": triggers(chosen),
        "validation_gestures_lost_to_rejection": int((gestures & is_gesture & ~accepted).sum()),
        "validation_macro_f1_at_threshold": float(macro_f1_of(matrix)),
        "warning": None if triggers(chosen) == 0 else
        "no threshold on the grid removed every validation false activation",
    }


def _train_one(
    torch: Any, hidden: int, weight_decay: float, seed: int, train: _Arrays, val: _Arrays,
    mean: np.ndarray, std: np.ndarray, config: TrainConfig, feature_config: FeatureConfig,
) -> dict[str, Any]:  # fmt: skip
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = _build_model(torch, hidden, config.dropout)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=weight_decay)  # fmt: skip
    loss_fn = torch.nn.CrossEntropyLoss()
    x_val = torch.from_numpy(((val.features - mean) / std).astype(np.float32))
    y_val = torch.from_numpy(val.labels)
    n = len(train.labels)

    history: list[dict[str, float]] = []
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, config.batch_size):
            idx = order[start : start + config.batch_size]
            batch = []
            for i in idx:
                try:
                    batch.append(features_from_resampled(
                        augment_resampled(train.pts[i], rng, config.augment), feature_config))
                except FeatureError:
                    batch.append(train.features[i])
            x = torch.from_numpy(((np.stack(batch) - mean) / std).astype(np.float32))
            y = torch.from_numpy(train.labels[idx])
            optimiser.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            total += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            val_loss = float(loss_fn(logits, y_val).detach())
        predicted = logits.argmax(dim=1).numpy()
        matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        np.add.at(matrix, (val.labels, predicted), 1)
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
                        "val_macro_f1": float(macro_f1_of(matrix))})  # fmt: skip
        if val_loss < best_loss - config.min_delta:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x_val).numpy().astype(np.float64)
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = shifted / shifted.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(matrix, (val.labels, predicted), 1)
    state = {k: v.numpy().astype(np.float32) for k, v in model.state_dict().items()}
    return {
        "hidden": hidden, "weight_decay": weight_decay, "seed": seed,
        "selected_epoch": best_epoch, "epochs_run": len(history),
        "best_val_loss": best_loss, "val_macro_f1": float(macro_f1_of(matrix)),
        "history": history, "calibration": calibrate_threshold(probabilities, val.labels,
                                                                config.tau_grid),
        "_state": state, "_val_logits": logits,
    }  # fmt: skip


def _artifact_arrays(state: dict[str, np.ndarray], mean: np.ndarray, std: np.ndarray
                     ) -> dict[str, np.ndarray]:  # fmt: skip
    return {
        "gru.weight_ih": state["gru.weight_ih_l0"], "gru.weight_hh": state["gru.weight_hh_l0"],
        "gru.bias_ih": state["gru.bias_ih_l0"], "gru.bias_hh": state["gru.bias_hh_l0"],
        "fc.weight": state["fc.weight"], "fc.bias": state["fc.bias"],
        "norm.mean": mean.astype(np.float32), "norm.std": std.astype(np.float32),
    }  # fmt: skip


def _metadata(
    result: dict[str, Any], split: dict[str, Any], feature_config: FeatureConfig,
    config: TrainConfig, provenance: str, torch_version: str, counts: dict[str, Any],
    role: str,
) -> dict[str, Any]:  # fmt: skip
    return {
        "model_type": MODEL_TYPE,
        "class_names": list(CLASS_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": feature_config.schema_hash(),
        "feature_config": feature_config.to_dict(),
        "architecture": {"input_dim": FEATURE_DIM, "hidden_size": result["hidden"],
                         "num_layers": 1, "num_classes": N_CLASSES},  # fmt: skip
        "rejection": result["calibration"],
        "seed": result["seed"],
        "role_in_run": role,
        "training": {"config": config.to_dict(), "weight_decay": result["weight_decay"],
                     "selected_epoch": result["selected_epoch"],
                     "epochs_run": result["epochs_run"],
                     "best_validation_loss": result["best_val_loss"],
                     "validation_macro_f1_argmax": result["val_macro_f1"],
                     "class_counts": counts, "flagged_takes_excluded": config.exclude_flagged},
        "libraries": _libraries(torch_version),
        "dataset_manifest_hash": split["dataset_manifest_hash"],
        "split_id": split["split_id"],
        "split_seed": split["seed"],
        "data_provenance": provenance,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "A small single-participant dataset: see the evaluation report for what the "
                "results can and cannot show.",
    }


def run_train(
    split: dict[str, Any], split_file: Path, dataset_dir: Path, runs_dir: Path,
    run_name: str | None = None, config: TrainConfig | None = None,
    log: Callable[[str], object] = print,
) -> Path:  # fmt: skip
    """Tune the baseline, train and select the GRU, calibrate its threshold; return the run dir."""
    config = config or TrainConfig()
    torch = _import_torch()
    started = time.perf_counter()
    feature_config = FeatureConfig(**split["feature_config"])
    run_dir, train_items, val_items, provenance, baseline_summary = _prepare_run(
        split, split_file, dataset_dir, runs_dir, run_name, config
    )
    before = baseline_summary["rule_defaults_macro_f1_on_tuning_data"]
    after = baseline_summary["rule_tuned_macro_f1_on_tuning_data"]
    log(f"baseline tuned (macro-F1 on its tuning data {before:.3f} -> {after:.3f})")  # fmt: skip

    train, val = _arrays(train_items, feature_config), _arrays(val_items, feature_config)
    # Standardisation comes from the training takes only. The standard deviation is floored at
    # STD_FLOOR (1% of a hand size) so that a feature that barely varies, whose spread is tracker
    # noise, is not amplified into a signal. Both are rounded to float32 *before* training, because
    # that is what the saved model stores: PyTorch and NumPy then standardise identically.
    flat = train.features.reshape(-1, FEATURE_DIM)
    mean = flat.mean(axis=0).astype(np.float32).astype(np.float64)
    std = np.maximum(flat.std(axis=0), STD_FLOOR).astype(np.float32).astype(np.float64)

    counts = {"train": _count(train_items), "validation": _count(val_items)}
    warnings = list(split.get("warnings", []))
    if counts["validation"][NEGATIVE_LABEL] < 30:
        warnings.append(f"only {counts['validation'][NEGATIVE_LABEL]} validation negatives: the "
                        "rejection threshold is a coarse estimate")  # fmt: skip
    log(f"training takes {counts['train']}, validation takes {counts['validation']}")

    results: list[dict[str, Any]] = []
    for hidden in config.hidden_sizes:
        for weight_decay in config.weight_decays:
            for seed in config.seeds:
                result = _train_one(torch, hidden, weight_decay, seed, train, val, mean, std,
                                    config, feature_config)  # fmt: skip
                results.append(result)
                log(f"  hidden {hidden:>3} wd {weight_decay:g} seed {seed}: epoch "
                    f"{result['selected_epoch']:>3}/{result['epochs_run']:>3}  val loss "
                    f"{result['best_val_loss']:.3f}  val macro-F1 {result['val_macro_f1']:.3f}  "
                    f"tau {result['calibration']['threshold']:.2f}")  # fmt: skip

    groups: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for r in results:
        groups.setdefault((r["hidden"], r["weight_decay"]), []).append(r)

    def group_key(item: tuple[tuple[int, float], list[dict[str, Any]]]) -> tuple[float, ...]:
        (hidden, weight_decay), runs = item
        return (-float(np.mean([r["val_macro_f1"] for r in runs])),
                float(np.mean([r["best_val_loss"] for r in runs])), hidden, weight_decay)

    selection = sorted(groups.items(), key=group_key)
    (chosen_hidden, chosen_wd), chosen_runs = selection[0]
    ranked = sorted(chosen_runs, key=lambda r: (-r["val_macro_f1"], r["best_val_loss"], r["seed"]))
    final = ranked[0]
    log(f"selected hidden {chosen_hidden}, weight decay {chosen_wd:g} (mean val macro-F1 "
        f"{np.mean([r['val_macro_f1'] for r in chosen_runs]):.3f}); final model seed "
        f"{final['seed']}, epoch {final['selected_epoch']}, "
        f"tau {final['calibration']['threshold']:.2f}")

    torch_version = torch.__version__
    for r in chosen_runs:
        target = run_dir if r is final else run_dir / "seeds" / f"seed-{r['seed']}"
        save_artifact(
            target, _artifact_arrays(r["_state"], mean, std),
            _metadata(r, split, feature_config, config, provenance, torch_version, counts,
                      "final" if r is final else "stability"),
        )  # fmt: skip

    classifier = load_classifier(run_dir)  # the artifact must load and agree with PyTorch
    parity = float(max(np.abs(classifier.logits(val.features[i]) - final["_val_logits"][i]).max()
                       for i in range(len(val.labels))))  # fmt: skip
    log(f"NumPy-versus-PyTorch max |logit difference| on validation takes: {parity:.2e}")
    if parity > 1e-3:
        raise RuntimeError(f"NumPy and PyTorch disagree by {parity:.2e}; not keeping the model")

    def public(r: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in r.items() if not k.startswith("_")}

    log_payload = {
        "split_id": split["split_id"], "data_provenance": provenance,
        "config": config.to_dict(),
        "class_counts": counts,
        "excluded_takes": split["excluded"], "flagged_takes": split["flagged"],
        "flagged_takes_excluded_from_this_run": config.exclude_flagged,
        "warnings": warnings,
        "selection": [
            {"hidden": h, "weight_decay": wd,
             "mean_val_macro_f1": float(np.mean([r["val_macro_f1"] for r in runs])),
             "sd_val_macro_f1": float(np.std([r["val_macro_f1"] for r in runs])),
             "mean_best_val_loss": float(np.mean([r["best_val_loss"] for r in runs])),
             "chosen": (h, wd) == (chosen_hidden, chosen_wd)}
            for (h, wd), runs in selection
        ],  # fmt: skip
        "selected": {"hidden": chosen_hidden, "weight_decay": chosen_wd,
                     "final_seed": final["seed"], "selected_epoch": final["selected_epoch"],
                     "rejection": final["calibration"]},  # fmt: skip
        "numpy_vs_torch_max_abs_logit_difference_validation": parity,
        "runs": [public(r) for r in results],
        "libraries": _libraries(torch_version),
        "baseline": baseline_summary,
        "seconds": round(time.perf_counter() - started, 1),
        "python": sys.version.split()[0],
    }
    (run_dir / TRAIN_LOG).write_text(json.dumps(log_payload, indent=2) + "\n", encoding="utf-8")
    return run_dir
