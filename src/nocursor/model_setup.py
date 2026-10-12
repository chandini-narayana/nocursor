"""Obtaining Google's pretrained MediaPipe Hand Landmarker model.

This is the only network access NoCursor ever makes, and only when you ask for it
(``nocursor --download-model``). Webcam frames are never sent anywhere.

The download URL points at ``.../float16/latest/...``, a location Google may update, so the
file's SHA-256 is *reported* for your records but not enforced. The size and, when the server
sends it, the Content-Length are checked, and a truncated or error-page download is rejected
without touching any model file you already have.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nocursor.paths import default_model_dir

logger = logging.getLogger(__name__)

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MIN_EXPECTED_BYTES = 1_000_000  # the real model is several MB; guards against error pages
TIMEOUT_S = 30
CHUNK_BYTES = 64 * 1024


class ModelDownloadError(Exception):
    """The model could not be downloaded; the message says why and what to do next."""


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    size_bytes: int
    sha256: str
    already_present: bool


def default_model_path() -> Path:
    return default_model_dir() / MODEL_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    dest: Path | None = None,
    *,
    force: bool = False,
    url: str = MODEL_URL,
    opener: Callable[..., Any] | None = None,
    min_bytes: int = MIN_EXPECTED_BYTES,
) -> DownloadResult:
    """Download the model to ``dest`` (default: ``models/`` in the project / working directory).

    An existing plausible model is kept unless ``force`` is set; even then it is only replaced
    after a complete, verified download. Raises ``ModelDownloadError`` otherwise.
    """
    opener = opener or urllib.request.urlopen  # looked up now, not when the function was defined
    dest = dest or default_model_path()
    if not force and dest.is_file() and dest.stat().st_size >= min_bytes:
        return DownloadResult(dest, dest.stat().st_size, _sha256(dest), already_present=True)

    part = dest.with_name(dest.name + ".part")
    manual = f"You can also download it in a browser from {url} and save it as {dest}."
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelDownloadError(f"cannot create {dest.parent}: {exc}") from exc

    digest = hashlib.sha256()
    written = 0
    try:
        with opener(url, timeout=TIMEOUT_S) as response, part.open("wb") as out:
            expected = _content_length(response)
            while chunk := response.read(CHUNK_BYTES):
                out.write(chunk)
                digest.update(chunk)
                written += len(chunk)
        if expected is not None and written != expected:
            raise ModelDownloadError(
                f"the download was cut short ({written:,} of {expected:,} bytes). "
                "Check your connection and try again."
            )
        if written < min_bytes:
            raise ModelDownloadError(
                f"the downloaded file is only {written:,} bytes, which is too small to be the "
                f"model (an error page, perhaps). {manual}"
            )
        part.replace(dest)
    except ModelDownloadError:
        raise
    except urllib.error.HTTPError as exc:
        raise ModelDownloadError(
            f"the server answered HTTP {exc.code} ({exc.reason}). {manual}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ModelDownloadError(
            f"download failed: {getattr(exc, 'reason', exc)}. "
            f"Check your internet connection and try again. {manual}"
        ) from exc
    finally:
        part.unlink(missing_ok=True)  # never leave a partial file behind
    return DownloadResult(dest, written, digest.hexdigest(), already_present=False)


def _content_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    value = headers.get("Content-Length") if headers is not None else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def run_download(force: bool = False, dest: Path | None = None) -> int:
    """Command-line behaviour for ``--download-model`` / ``--redownload-model``."""
    import sys

    target = dest or default_model_path()
    if not force and target.is_file() and target.stat().st_size >= MIN_EXPECTED_BYTES:
        print(f"Model already present: {target}")
    else:
        print(f"Downloading the MediaPipe hand-landmarker model to {target} ...")
    try:
        result = download_model(dest, force=force)
    except ModelDownloadError as exc:
        print(f"nocursor: {exc}", file=sys.stderr)
        return 1
    if not result.already_present:
        print(f"Saved {result.size_bytes:,} bytes.")
    print(f"SHA-256: {result.sha256}  (for your records; not enforced, the source URL is 'latest')")
    return 0
