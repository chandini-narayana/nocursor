"""Saving, validating and running a trained gesture model with NumPy only.

This module never imports PyTorch. A model is a directory with two files:

* ``model.npz``: the weights and the feature-standardisation arrays, loaded with
  ``allow_pickle=False`` so a tampered file cannot run code.
* ``model.json``: everything needed to use them correctly and to trust them: class names, the
  feature definition and its hash, the architecture, the rejection threshold, a checksum of the
  weights, and how the model was trained (seed, versions, dataset and split identifiers).

``load_classifier`` validates all of it and raises ``ModelArtifactError`` for anything missing,
corrupt, incompatible or not finite; there is no partial loading. ``GRUClassifier`` itself fails
closed: unusable input or output becomes a ``rejected`` prediction marked invalid, never a guess.

The forward pass reproduces PyTorch's single-layer GRU: with gates ``r`` (reset), ``z`` (update)::

    r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
    z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
    n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
    h = (1 - z) * n + z * h

with the three gates stacked in the order r, z, n, as PyTorch stores them.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from nocursor.dynamic.classifier import REJECTED, Prediction
from nocursor.dynamic.features import (
    FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
)
from nocursor.dynamic.schema import CLASS_NAMES, TARGET_CLASSES

ARTIFACT_SCHEMA_VERSION = 1
MODEL_FILE = "model.npz"
META_FILE = "model.json"
MODEL_TYPE = "gru"
TENSOR_NAMES = (
    "gru.weight_ih", "gru.weight_hh", "gru.bias_ih", "gru.bias_hh",
    "fc.weight", "fc.bias", "norm.mean", "norm.std",
)  # fmt: skip
REQUIRED_META = (
    "artifact_schema_version", "model_type", "class_names", "feature_schema_version",
    "feature_schema_hash", "feature_config", "architecture", "rejection", "weights_sha256",
)  # fmt: skip


class ModelArtifactError(ValueError):
    """The model files are missing, damaged, incompatible or not safe to use."""


def weights_checksum(arrays: dict[str, np.ndarray]) -> str:
    """SHA-256 over every tensor's name, dtype, shape and bytes, in a fixed order."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}|{array.dtype.str}|{array.shape}|".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _check(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    """Raise ``ModelArtifactError`` unless metadata and tensors are complete and consistent."""
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ModelArtifactError(f"model metadata is missing {', '.join(missing)}")
    if meta["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ModelArtifactError(
            f"unsupported artifact_schema_version {meta['artifact_schema_version']!r} "
            f"(this build reads {ARTIFACT_SCHEMA_VERSION})"
        )
    if meta["model_type"] != MODEL_TYPE:
        raise ModelArtifactError(f"unsupported model_type {meta['model_type']!r}")
    if list(meta["class_names"]) != list(CLASS_NAMES):
        raise ModelArtifactError(
            f"class names {meta['class_names']} do not match this build's {list(CLASS_NAMES)}"
        )
    if meta["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ModelArtifactError("the model was trained on a different feature schema version")
    try:
        config = FeatureConfig(**meta["feature_config"])
    except (TypeError, ValueError) as exc:
        raise ModelArtifactError(f"invalid feature_config: {exc}") from exc
    if config.schema_hash() != meta["feature_schema_hash"]:
        raise ModelArtifactError(
            "the feature definition recorded in the model does not match this build's "
            "(feature hash mismatch); the model cannot be used with these features"
        )

    arch = meta["architecture"]
    try:
        hidden, inputs, classes = int(arch["hidden_size"]), int(arch["input_dim"]), int(
            arch["num_classes"])  # fmt: skip
        layers = int(arch["num_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelArtifactError(f"invalid architecture: {exc}") from exc
    if layers != 1 or inputs != FEATURE_DIM or classes != len(CLASS_NAMES) or hidden < 1:
        raise ModelArtifactError(f"unsupported architecture {arch}")
    threshold = meta["rejection"].get("threshold") if isinstance(meta["rejection"], dict) else None
    if not isinstance(threshold, int | float) or not (0.0 < float(threshold) <= 1.0):
        raise ModelArtifactError("rejection threshold must be a number within (0, 1]")

    if sorted(arrays) != sorted(TENSOR_NAMES):
        raise ModelArtifactError(
            f"model tensors {sorted(arrays)} do not match the expected {sorted(TENSOR_NAMES)}"
        )
    expected = {
        "gru.weight_ih": (3 * hidden, inputs), "gru.weight_hh": (3 * hidden, hidden),
        "gru.bias_ih": (3 * hidden,), "gru.bias_hh": (3 * hidden,),
        "fc.weight": (classes, hidden), "fc.bias": (classes,),
        "norm.mean": (inputs,), "norm.std": (inputs,),
    }  # fmt: skip
    for name, shape in expected.items():
        array = arrays[name]
        if array.dtype != np.float32:
            raise ModelArtifactError(f"tensor {name} must be float32, found {array.dtype}")
        if array.shape != shape:
            raise ModelArtifactError(f"tensor {name} has shape {array.shape}, expected {shape}")
        if not np.all(np.isfinite(array)):
            raise ModelArtifactError(f"tensor {name} contains NaN or infinity")
    if np.any(arrays["norm.std"] <= 0):
        raise ModelArtifactError("norm.std must be positive")
    if weights_checksum(arrays) != meta["weights_sha256"]:
        raise ModelArtifactError("weights checksum mismatch: the model file is damaged or edited")


def save_artifact(directory: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    """Write ``model.npz`` and ``model.json``; refuses what ``load_classifier`` would refuse."""
    arrays = {name: np.ascontiguousarray(arrays[name], dtype=np.float32) for name in arrays}
    meta = dict(metadata)
    meta["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    meta["weights_sha256"] = weights_checksum(arrays)
    _check(meta, arrays)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / MODEL_FILE, **arrays)
    (directory / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")  # fmt: skip


def load_classifier(directory: Path) -> GRUClassifier:
    """Load and fully validate a model directory; raises ``ModelArtifactError``."""
    meta_path, weights_path = directory / META_FILE, directory / MODEL_FILE
    for path in (meta_path, weights_path):
        if not path.is_file():
            raise ModelArtifactError(f"model file missing: {path.name}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(f"{META_FILE} is not valid JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise ModelArtifactError(f"{META_FILE} must hold a JSON object")
    try:
        with np.load(weights_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
    except Exception as exc:  # corrupt zip, pickled objects, truncated file...
        raise ModelArtifactError(f"{MODEL_FILE} cannot be loaded safely: {exc}") from exc
    _check(meta, arrays)
    return GRUClassifier(arrays, meta)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class GRUClassifier:
    """Single-layer GRU plus linear head, evaluated in float64 from float32 weights."""

    def __init__(
        self, arrays: dict[str, np.ndarray], meta: dict[str, Any], name: str = "gru"
    ) -> None:
        self.name = name
        self.meta = meta
        self.hidden = int(meta["architecture"]["hidden_size"])
        self.threshold = float(meta["rejection"]["threshold"])
        self.config = FeatureConfig(**meta["feature_config"])
        w = {k: v.astype(np.float64) for k, v in arrays.items()}
        self._w_ih, self._w_hh = w["gru.weight_ih"], w["gru.weight_hh"]
        self._b_ih, self._b_hh = w["gru.bias_ih"], w["gru.bias_hh"]
        self._fc_w, self._fc_b = w["fc.weight"], w["fc.bias"]
        self._mean, self._std = w["norm.mean"], w["norm.std"]

    def logits(self, features: np.ndarray) -> np.ndarray:
        """Raw class scores for one ``(steps, FEATURE_DIM)`` feature matrix."""
        x = (np.asarray(features, dtype=np.float64) - self._mean) / self._std
        gates_in = x @ self._w_ih.T + self._b_ih
        h = np.zeros(self.hidden)
        n = self.hidden
        for t in range(x.shape[0]):
            gates_h = self._w_hh @ h + self._b_hh
            r = _sigmoid(gates_in[t, :n] + gates_h[:n])
            z = _sigmoid(gates_in[t, n : 2 * n] + gates_h[n : 2 * n])
            cand = np.tanh(gates_in[t, 2 * n :] + r * gates_h[2 * n :])
            h = (1.0 - z) * cand + z * h
        return self._fc_w @ h + self._fc_b

    def predict_features(self, features: np.ndarray) -> Prediction:
        shape = (self.config.steps, FEATURE_DIM)
        array = np.asarray(features, dtype=np.float64)
        if array.shape != shape or not np.all(np.isfinite(array)):
            return Prediction(REJECTED, note="invalid input features", valid=False)
        scores = self.logits(array)
        if scores.shape != (len(CLASS_NAMES),) or not np.all(np.isfinite(scores)):
            return Prediction(REJECTED, note="invalid model output", valid=False)
        shifted = np.exp(scores - scores.max())
        probabilities = shifted / shifted.sum()
        if not np.all(np.isfinite(probabilities)) or not math.isclose(
            float(probabilities.sum()), 1.0, abs_tol=1e-6
        ):
            return Prediction(REJECTED, note="invalid model output", valid=False)
        top = int(np.argmax(probabilities))
        label = CLASS_NAMES[top]
        probs = tuple(float(p) for p in probabilities)
        if label in TARGET_CLASSES and probs[top] < self.threshold:
            return Prediction(REJECTED, probs, note="below the rejection threshold")
        return Prediction(label, probs)
