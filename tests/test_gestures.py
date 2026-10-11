import pytest

from nocursor.gestures import PinchDetector, PinchState

OPEN = 1.0
PINCH = 0.1  # below default threshold 0.25
MID = 0.3  # between engage (0.25) and release (0.375)


def make(**kw):
    defaults = {"threshold": 0.25, "release_factor": 1.5, "confirm_frames": 2, "cooldown_s": 0.4}
    return PinchDetector(**{**defaults, **kw})


def armed(detector: PinchDetector, t: float = 0.0) -> PinchDetector:
    detector.update(OPEN, t)
    return detector


class TestClicking:
    def test_click_fires_once_after_confirmation_frames(self):
        d = armed(make())
        assert not d.update(PINCH, 0.1).clicked  # frame 1 of 2
        result = d.update(PINCH, 0.2)  # frame 2 of 2
        assert result.clicked and result.state is PinchState.PINCHED

    def test_holding_pinch_never_repeats_click(self):
        d = armed(make())
        clicks = [d.update(PINCH, 0.1 * i).clicked for i in range(1, 60)]
        assert sum(clicks) == 1

    def test_single_noisy_frame_does_not_click(self):
        d = armed(make())
        d.update(PINCH, 0.1)
        d.update(OPEN, 0.2)
        assert not d.update(PINCH, 0.3).clicked  # streak was broken

    def test_threshold_is_configurable(self):
        strict, lenient = armed(make(threshold=0.1)), armed(make(threshold=0.4))
        ratio = 0.2
        assert not any(strict.update(ratio, t).clicked for t in (0.1, 0.2, 0.3))
        assert any(lenient.update(ratio, t).clicked for t in (0.1, 0.2, 0.3))

    def test_ratio_exactly_at_threshold_counts(self):
        d = armed(make(confirm_frames=1))
        assert d.update(0.25, 0.1).clicked

    def test_confirm_frames_one_clicks_immediately(self):
        d = armed(make(confirm_frames=1))
        assert d.update(PINCH, 0.1).clicked


class TestHysteresisAndRepeat:
    def test_jitter_between_engage_and_release_keeps_pinch(self):
        d = armed(make())
        d.update(PINCH, 0.1)
        d.update(PINCH, 0.2)
        for t in (0.3, 0.4, 0.5):
            result = d.update(MID, t)
            assert result.state is PinchState.PINCHED and not result.clicked

    def test_second_click_requires_release_and_cooldown(self):
        d = armed(make(cooldown_s=0.4))
        d.update(PINCH, 0.1)
        assert d.update(PINCH, 0.2).clicked
        d.update(OPEN, 0.3)
        d.update(PINCH, 0.4)
        assert not d.update(PINCH, 0.5).clicked  # within 0.4 s of the click at 0.2
        d.update(OPEN, 0.6)
        d.update(PINCH, 0.7)
        assert d.update(PINCH, 0.8).clicked  # cooldown elapsed

    def test_pinch_consumed_by_cooldown_does_not_click_later_while_held(self):
        d = armed(make(cooldown_s=1.0))
        d.update(PINCH, 0.0)
        d.update(PINCH, 0.1)  # click
        d.update(OPEN, 0.2)
        d.update(PINCH, 0.3)
        d.update(PINCH, 0.4)  # consumed by cooldown
        assert not any(d.update(PINCH, 0.5 + 0.1 * i).clicked for i in range(30))

    def test_zero_cooldown_allows_back_to_back_pinches(self):
        d = armed(make(cooldown_s=0.0, confirm_frames=1))
        assert d.update(PINCH, 0.0).clicked
        d.update(OPEN, 0.01)
        assert d.update(PINCH, 0.02).clicked


class TestArming:
    def test_pinch_from_the_start_does_not_click_until_hand_opens(self):
        d = make(confirm_frames=1)
        results = [d.update(PINCH, 0.1 * i) for i in range(1, 10)]
        assert not any(r.clicked for r in results)
        assert all(r.state is PinchState.REOPEN for r in results)

    def test_reset_requires_reopen_before_next_click(self):
        d = armed(make(confirm_frames=1))
        d.reset()  # e.g. tracking lost
        assert not d.update(PINCH, 5.0).clicked
        d.update(OPEN, 5.1)
        assert d.update(PINCH, 5.2).clicked

    def test_reset_keeps_cooldown(self):
        d = armed(make(confirm_frames=1, cooldown_s=1.0))
        assert d.update(PINCH, 0.0).clicked
        d.reset()
        d.update(OPEN, 0.1)
        assert not d.update(PINCH, 0.2).clicked

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    def test_invalid_ratio_never_clicks_and_disarms(self, bad):
        d = armed(make(confirm_frames=1))
        result = d.update(bad, 1.0)
        assert not result.clicked and result.freeze_cursor
        assert not d.update(PINCH, 1.1).clicked  # disarmed until reopened


class TestCursorFreeze:
    def test_freeze_while_fingers_close_but_not_when_open(self):
        d = armed(make())
        assert not d.update(OPEN, 0.1).freeze_cursor
        assert d.update(MID, 0.2).freeze_cursor
        assert d.update(PINCH, 0.3).freeze_cursor

    def test_unfreezes_after_release(self):
        d = armed(make())
        d.update(PINCH, 0.1)
        d.update(PINCH, 0.2)
        assert not d.update(OPEN, 0.3).freeze_cursor


@pytest.mark.parametrize("kwargs", [
    {"threshold": 0}, {"threshold": -1}, {"release_factor": 1.0}, {"release_factor": 0.5},
    {"confirm_frames": 0}, {"cooldown_s": -0.1},
])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)
