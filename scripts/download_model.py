"""Download Google's pretrained MediaPipe Hand Landmarker model into ./models.

This is the only network access NoCursor needs, and it happens only when you run
this script. Webcam frames are never sent anywhere.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEST = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
MIN_EXPECTED_BYTES = 1_000_000  # the real model is several MB; guards against error pages


def main() -> int:
    if DEST.is_file() and DEST.stat().st_size >= MIN_EXPECTED_BYTES:
        print(f"Model already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    partial = DEST.with_suffix(".part")
    print(f"Downloading {URL}")
    try:
        urllib.request.urlretrieve(URL, partial)  # noqa: S310 - fixed https URL
    except OSError as exc:
        partial.unlink(missing_ok=True)
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    if partial.stat().st_size < MIN_EXPECTED_BYTES:
        partial.unlink(missing_ok=True)
        print("Downloaded file is unexpectedly small; aborting.", file=sys.stderr)
        return 1
    partial.replace(DEST)
    print(f"Saved {DEST} ({DEST.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
