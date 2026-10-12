"""Download Google's pretrained MediaPipe Hand Landmarker model into ./models.

Equivalent to ``nocursor --download-model``; kept so the original command still works.
This is the only network access NoCursor needs, and it happens only when you run it.
Webcam frames are never sent anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from nocursor.model_setup import run_download
except ImportError:  # running from a checkout where the package is not installed
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from nocursor.model_setup import run_download

if __name__ == "__main__":
    sys.exit(run_download(force="--force" in sys.argv[1:]))
