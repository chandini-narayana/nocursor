"""Where NoCursor keeps its local runtime files (model, calibration, logs, benchmarks, datasets)."""

from __future__ import annotations

from pathlib import Path


def runtime_root() -> Path:
    """The project folder when running from a source checkout, else the working directory.

    A checkout is recognised by ``pyproject.toml`` two levels above this package
    (``<root>/src/nocursor/paths.py``). After a normal (non-editable) install there is no
    such folder, so runtime files go in the directory you run NoCursor from.
    """
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() else Path.cwd()


def default_model_dir() -> Path:
    return runtime_root() / "models"


def default_log_path() -> Path:
    return runtime_root() / "logs" / "nocursor.log"


def default_benchmark_dir() -> Path:
    return runtime_root() / "benchmarks"


def default_ml_runs_dir() -> Path:
    """Splits, trained models and evaluation reports (derived from personal data; git-ignored)."""
    return runtime_root() / "ml_runs"


def default_dataset_dir() -> Path:
    """Recorded gesture landmark sequences (personal data; excluded from version control)."""
    return runtime_root() / "datasets"
