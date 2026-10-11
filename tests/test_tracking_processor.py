import numpy as np
import pytest

from helpers import make_hand
from nocursor.config import Config
from nocursor.gestures import PinchState
from nocursor.models import INDEX_TIP, HandObservation
from nocursor.processor import GestureProcessor
from nocursor.tracking import TrackingMonitor, TrackingState

SCREEN = (1920, 1080)


class TestTrackingMonitor:
    def monitor(self, **kw):
        return TrackingMonitor(**{"min_confidence": 0.6, "min_hand_size": 0.05,
                                  "stable_frames": 3, **kw})

    def test_no_hand_is_lost(self):
        assert self.monitor().update(None) is TrackingState.LOST

    def test_needs_stable_frames_before_tracking(self):
        m = self.monitor()
        states = [m.update(make_hand()) for _ in range(3)]
        assert states == [TrackingState.ACQUIRING, TrackingState.ACQUIRING, TrackingState.TRACKING]

    def test_low_confidence_is_uncertain(self):
        assert self.monitor().update(make_hand(confidence=0.3)) is TrackingState.UNCERTAIN

    def test_hand_partly_outside_frame_is_uncertain(self):
        assert self.monitor().update(make_hand(index_xy=(1.05, 0.5))) is TrackingState.UNCERTAIN
        assert self.monitor().update(make_hand(index_xy=(0.5, -0.01))) is TrackingState.UNCERTAIN

    def test_tiny_hand_is_uncertain(self):
        assert self.monitor().update(make_hand(hand_size=0.01)) is TrackingState.UNCERTAIN

    def test_nan_landmarks_are_uncertain(self):
        hand = make_hand()
        bad = hand.landmarks.copy()
        bad[INDEX_TIP, 0] = np.nan
        assert self.monitor().update(HandObservation(bad, 0.9, 1.0)) is TrackingState.UNCERTAIN

    def test_bad_frame_restarts_stability_count(self):
        m = self.monitor()
        for _ in range(5):
            m.update(make_hand())
        m.update(None)
        assert m.update(make_hand()) is TrackingState.ACQUIRING

    def test_zero_min_hand_size_does_not_allow_degenerate_hand(self):
        m = self.monitor(min_hand_size=0.0, stable_frames=1)
        assert m.update(make_hand(hand_size=0.0)) is TrackingState.UNCERTAIN


class TestHandObservation:
    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            HandObservation(np.zeros((5, 3)), 0.9, 1.0)

    def test_distance_is_aspect_corrected(self):
        hand = make_hand(pinch_ratio=1.0, aspect_ratio=16 / 9)
        assert hand.distance(4, 8) == pytest.approx(hand.hand_size)


def processor(**overrides):
    cfg = Config(**{"smoothing": 0.0, "stable_frames": 1, "pinch_confirm_frames": 1,
                    "click_cooldown_s": 0.0, "active_margin": 0.0, **overrides})
    return GestureProcessor(cfg, SCREEN)


class TestGestureProcessor:
    def test_no_hand_produces_no_actions(self):
        frame = processor().process(None, 0.0)
        assert frame.tracking is TrackingState.LOST
        assert frame.cursor is None and not frame.click

    def test_open_hand_moves_cursor_to_mapped_position(self):
        frame = processor().process(make_hand(index_xy=(0.5, 0.5), pinch_ratio=1.0), 0.0)
        assert abs(frame.cursor[0] - 959.5) <= 0.5
        assert abs(frame.cursor[1] - 539.5) <= 0.5
        assert not frame.click

    def test_cursor_follows_hand_with_mirroring(self):
        p = processor()
        left = p.process(make_hand(index_xy=(0.2, 0.5)), 0.0).cursor
        right = p.process(make_hand(index_xy=(0.8, 0.5)), 0.1).cursor
        assert left[0] > right[0]

    def test_pinch_clicks_once_and_freezes_cursor(self):
        p = processor()
        p.process(make_hand(pinch_ratio=1.0), 0.0)  # arm
        frame = p.process(make_hand(pinch_ratio=0.1), 0.1)
        assert frame.click and frame.pinch is PinchState.PINCHED
        assert frame.cursor is None
        assert not p.process(make_hand(pinch_ratio=0.1), 0.2).click

    def test_no_click_when_confidence_drops(self):
        p = processor()
        p.process(make_hand(pinch_ratio=1.0), 0.0)
        frame = p.process(make_hand(pinch_ratio=0.1, confidence=0.2), 0.1)
        assert frame.tracking is TrackingState.UNCERTAIN
        assert not frame.click and frame.cursor is None

    def test_no_click_when_hand_reappears_already_pinched(self):
        p = processor()
        p.process(make_hand(pinch_ratio=1.0), 0.0)
        p.process(None, 0.1)  # tracking lost
        assert not p.process(make_hand(pinch_ratio=0.1), 0.2).click
        assert not p.process(make_hand(pinch_ratio=0.1), 0.3).click

    def test_click_possible_after_reappearing_and_reopening(self):
        p = processor()
        p.process(None, 0.0)
        p.process(make_hand(pinch_ratio=1.0), 0.1)
        assert p.process(make_hand(pinch_ratio=0.1), 0.2).click

    def test_no_actions_while_acquiring(self):
        p = processor(stable_frames=3)
        frame = p.process(make_hand(), 0.0)
        assert frame.tracking is TrackingState.ACQUIRING
        assert frame.cursor is None and not frame.click

    def test_smoothing_applies_and_resets_after_loss(self):
        p = processor(smoothing=0.9)
        first = p.process(make_hand(index_xy=(0.5, 0.5)), 0.0).cursor
        second = p.process(make_hand(index_xy=(0.3, 0.5)), 0.1).cursor
        assert 0 < second[0] - first[0] < 100  # barely moved toward the target (heavy smoothing)
        p.process(None, 0.2)
        reacquired = p.process(make_hand(index_xy=(0.3, 0.5)), 0.3).cursor
        assert abs(reacquired[0] - first[0]) > 200  # history dropped: jumps straight to target
