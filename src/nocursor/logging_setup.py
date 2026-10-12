"""Logging configuration: concise console, optional bounded log file, safe formatting.

* Console: ``INFO`` and above, short format. ``-v`` adds ``DEBUG`` and logger names.
* File (opt-in via ``--log-file``): rotating, at most ``LOG_BYTES * (LOG_BACKUPS + 1)`` on disk.
* Gesture diagnostics are ``DEBUG`` records on the ``nocursor.diagnostics`` logger; they are
  shown only when ``--diagnostics`` or ``-v`` is given.
* Nothing that could identify the user or contain camera data is written: the formatter
  replaces the home directory with ``~``, summarises NumPy arrays / large byte strings /
  long sequences instead of printing them, and truncates very long messages. Neither
  frames nor landmark histories are ever passed to a logger by NoCursor.
* Third-party libraries (MediaPipe, absl, ...) still get a WARNING-level root handler so
  their real errors are not hidden.
"""

from __future__ import annotations

import copy
import logging
import logging.handlers
import os
import re
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

PACKAGE_LOGGER = "nocursor"
DIAGNOSTICS_LOGGER = "nocursor.diagnostics"
LOG_BYTES = 1_000_000
LOG_BACKUPS = 3
MAX_MESSAGE_CHARS = 2000
MAX_SEQUENCE_ITEMS = 20
MAX_BYTES_ITEMS = 64
_HANDLER_TAG = "_nocursor_handler"

CONSOLE_FORMAT = "%(asctime)s %(levelname)s %(message)s"
VERBOSE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def summarize_argument(value: Any) -> Any:
    """Replace values that must not be printed in full by a short description."""
    if isinstance(value, np.ndarray):
        return f"<ndarray shape={value.shape} dtype={value.dtype}>"
    if isinstance(value, bytes | bytearray) and len(value) > MAX_BYTES_ITEMS:
        return f"<{type(value).__name__} len={len(value)}>"
    if isinstance(value, list | tuple | set | frozenset) and len(value) > MAX_SEQUENCE_ITEMS:
        return f"<{type(value).__name__} len={len(value)}>"
    return value


class SafeFormatter(logging.Formatter):
    """Formats a record, then redacts the home directory and bounds the length."""

    def __init__(self, fmt: str, datefmt: str | None = None, home: str | None = None) -> None:
        super().__init__(fmt, datefmt)
        self._home_patterns = _home_patterns(home if home is not None else str(Path.home()))

    def format(self, record: logging.LogRecord) -> str:
        safe = copy.copy(record)  # never mutate the record other handlers will format
        if isinstance(safe.args, tuple):
            safe.args = tuple(summarize_argument(a) for a in safe.args)
        elif isinstance(safe.args, dict):
            safe.args = {k: summarize_argument(v) for k, v in safe.args.items()}
        text = super().format(safe)
        for pattern in self._home_patterns:
            text = pattern.sub("~", text)
        if len(text) > MAX_MESSAGE_CHARS:
            extra = len(text) - MAX_MESSAGE_CHARS
            text = f"{text[:MAX_MESSAGE_CHARS]}... [{extra} more characters truncated]"
        return text


def _home_patterns(home: str) -> list[re.Pattern[str]]:
    home = home.rstrip("\\/")
    if len(home) < 4:  # "/" or "C:" would redact far too much
        return []
    variants = {home, home.replace("\\", "/"), home.replace("/", "\\")}
    flags = re.IGNORECASE if os.name == "nt" else 0
    return [re.compile(re.escape(v), flags) for v in sorted(variants, key=len, reverse=True)]


class RecordFilter(logging.Filter):
    """Level policy for our handlers.

    Normal records need ``INFO``; ``-v`` lowers that to ``DEBUG``. Gesture diagnostics are
    ``DEBUG`` records that are allowed through only when diagnostics (or ``-v``) were asked
    for. Records marked ``file_only`` are skipped by the console.
    """

    def __init__(self, verbose: bool, diagnostics: bool, console: bool) -> None:
        super().__init__()
        self._verbose = verbose
        self._diagnostics = diagnostics
        self._console = console

    def filter(self, record: logging.LogRecord) -> bool:
        if self._console and getattr(record, "file_only", False):
            return False
        if record.name == DIAGNOSTICS_LOGGER or record.name.startswith(DIAGNOSTICS_LOGGER + "."):
            return self._diagnostics or self._verbose
        minimum = logging.DEBUG if self._verbose else logging.INFO
        return record.levelno >= minimum


@dataclass
class LoggingHandle:
    log_file: Path | None
    log_file_error: str | None = None

    def close(self) -> None:
        _remove_our_handlers()


def _remove_our_handlers() -> None:
    for logger in (logging.getLogger(PACKAGE_LOGGER), logging.getLogger()):
        for handler in list(logger.handlers):
            if getattr(handler, _HANDLER_TAG, False):
                logger.removeHandler(handler)
                handler.close()


def configure_logging(
    verbose: bool = False,
    diagnostics: bool = False,
    log_file: Path | None = None,
    stream: TextIO | None = None,
    home: str | None = None,
) -> LoggingHandle:
    """Install the console (and optional rotating file) handlers. Safe to call repeatedly."""
    _remove_our_handlers()
    package = logging.getLogger(PACKAGE_LOGGER)
    package.setLevel(logging.DEBUG)
    package.propagate = False  # we attach our own handlers; the root one is for third parties

    def tagged(handler: logging.Handler) -> logging.Handler:
        setattr(handler, _HANDLER_TAG, True)
        return handler

    console = tagged(logging.StreamHandler(stream or sys.stderr))
    console.setLevel(logging.DEBUG)
    console.addFilter(RecordFilter(verbose, diagnostics, console=True))
    console.setFormatter(
        SafeFormatter(VERBOSE_FORMAT if verbose else CONSOLE_FORMAT, "%H:%M:%S", home)
    )
    package.addHandler(console)

    third_party = tagged(logging.StreamHandler(stream or sys.stderr))
    third_party.setLevel(logging.INFO if verbose else logging.WARNING)
    third_party.setFormatter(SafeFormatter(VERBOSE_FORMAT, "%H:%M:%S", home))
    logging.getLogger().addHandler(third_party)

    handle = LoggingHandle(log_file=None)
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = tagged(
                logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=LOG_BYTES, backupCount=LOG_BACKUPS,
                    encoding="utf-8", delay=True,
                )
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.addFilter(RecordFilter(verbose, diagnostics, console=False))
            file_handler.setFormatter(SafeFormatter(FILE_FORMAT, None, home))
            package.addHandler(file_handler)
            handle.log_file = log_file
        except OSError as exc:
            handle.log_file_error = f"cannot write log file {log_file}: {exc}"
            package.warning("%s; continuing without a log file", handle.log_file_error)
    return handle


# ---------------------------------------------------------------- native output
@dataclass
class NativeCapture:
    """What C/C++ code (MediaPipe, TensorFlow Lite, glog) wrote to stderr in a block."""

    text: str = ""


@contextmanager
def capture_native_stderr() -> Iterator[NativeCapture]:
    """Capture file-descriptor-level stderr, which ordinary logging redirects cannot reach.

    The captured text is returned in ``NativeCapture.text`` after the block, so the caller
    decides what to do with it (see ``report_native_output``). If stderr cannot be
    redirected (no console, closed descriptor) the block just runs unchanged.
    """
    capture = NativeCapture()
    saved = scratch = None
    with suppress(OSError, ValueError, AttributeError):
        sys.stderr.flush()
    try:
        saved = os.dup(2)
        scratch = tempfile.TemporaryFile()  # noqa: SIM115 - closed explicitly below
        os.dup2(scratch.fileno(), 2)
    except OSError:
        if saved is not None:
            os.close(saved)
        if scratch is not None:
            scratch.close()
        saved = scratch = None
    try:
        yield capture
    finally:
        if saved is not None and scratch is not None:
            os.dup2(saved, 2)
            os.close(saved)
            scratch.seek(0)
            capture.text = scratch.read().decode("utf-8", errors="replace")
            scratch.close()


_SEVERE = re.compile(
    r"^[EF]\d{4}\b|\b(error|fatal|failed|exception|cannot|unable)\b", re.IGNORECASE
)


def report_native_output(
    text: str, source: str, failed: bool, logger: logging.Logger | None = None
) -> None:
    """Log captured native output without hiding anything that matters.

    * The operation failed: everything is logged at ERROR (it is the best clue to why).
    * Otherwise lines that look like errors are logged at WARNING and the routine rest
      (delegate creation, "feedback tensors" notices) at DEBUG, visible with ``-v``.
    """
    log = logger or logging.getLogger(__name__)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return
    if failed:
        log.error("%s reported:\n%s", source, "\n".join(lines))
        return
    for line in lines:
        log.log(logging.WARNING if _SEVERE.search(line) else logging.DEBUG, "%s: %s", source, line)
