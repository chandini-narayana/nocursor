"""Checks that touch OpenCV drawing and the real MediaPipe model (no webcam, no window)."""

import numpy as np
import pytest

from helpers import make_hand
from nocursor.gestures import PinchState
from nocursor.hand_tracker import ModelNotFoundError, resolve_model_path
from nocursor.preview import OpenCVPreview, StatusInfo
from nocursor.processor import GestureFrame
from nocursor.tracking import TrackingState


@pytest.mark.parametrize("tracking,pinch", [
    (TrackingState.LOST, None),
    (TrackingState.UNCERTAIN, PinchState.OPEN),
    (TrackingState.TRACKING, PinchState.PINCHED),
])
def test_overlay_draws_without_error_and_changes_pixels(tracking, pinch):
    preview = OpenCVPreview(active_margin=0.15)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    gesture = GestureFrame(tracking=tracking, pinch=pinch, pinch_ratio=0.2)
    status = StatusInfo("STOPPED", 29.7, "ctrl+alt+q", "test reason")
    hand = make_hand() if tracking is not TrackingState.LOST else None
    if hand is not None:
        preview._draw_hand(canvas, hand, gesture)
    preview._draw_status(canvas, gesture, status)
    assert canvas.any()


def test_real_model_loads_and_finds_no_hand_in_blank_frame():
    try:
        model = resolve_model_path(None)
    except ModelNotFoundError:
        pytest.skip("hand_landmarker.task not downloaded (nocursor --download-model)")
    from nocursor.hand_tracker import HandTracker

    tracker = HandTracker(model, min_confidence=0.6)
    try:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        assert tracker.detect(blank, 1000) is None
        assert tracker.detect(blank, 1000) is None  # repeated timestamp must not raise
        assert tracker.detect(blank, 900) is None  # going backwards must not raise either
    finally:
        tracker.close()
