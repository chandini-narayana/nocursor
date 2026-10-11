"""MediaPipe Hand Landmarker wrapper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from nocursor.models import NUM_LANDMARKS, HandObservation

logger = logging.getLogger(__name__)

MODEL_FILENAME = "hand_landmarker.task"
DOWNLOAD_HINT = "python scripts/download_model.py"


class ModelNotFoundError(FileNotFoundError):
    """The hand landmarker model file is missing."""


def resolve_model_path(explicit: Path | None = None) -> Path:
    """Find the model: an explicit path, else ./models, else the project's models/ folder."""
    if explicit is not None:
        if not explicit.is_file():
            raise ModelNotFoundError(f"model file not found: {explicit}")
        return explicit
    candidates = [
        Path.cwd() / "models" / MODEL_FILENAME,
        Path(__file__).resolve().parents[2] / "models" / MODEL_FILENAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ModelNotFoundError(
        f"{MODEL_FILENAME} not found. Download it with `{DOWNLOAD_HINT}` "
        "or pass --model PATH."
    )


class HandTracker:
    """Detects a single hand per frame using the pretrained MediaPipe Hand Landmarker."""

    def __init__(self, model_path: Path, min_confidence: float = 0.6) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            # A buffer (not a path) avoids Windows path-encoding problems in MediaPipe.
            base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker: Any = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandObservation | None:
        """Return the detected hand, or None. Timestamps must increase; we enforce it."""
        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return None
        points = result.hand_landmarks[0]
        if len(points) != NUM_LANDMARKS:
            logger.warning("unexpected landmark count: %d", len(points))
            return None
        landmarks = np.array([[p.x, p.y, p.z] for p in points], dtype=float)
        confidence = float(result.handedness[0][0].score) if result.handedness else 0.0
        return HandObservation(landmarks, confidence, aspect_ratio=width / height)

    def close(self) -> None:
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            landmarker.close()
