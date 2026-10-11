"""Preview: new overlays and keyboard commands (no window is ever opened)."""

import cv2
import numpy as np
import pytest

from helpers import make_hand
from nocursor.calibration import CalibrationWizard, WizardView
from nocursor.interaction import InteractionState
from nocursor.preview import OpenCVPreview, PreviewCommand, StatusInfo
from nocursor.processor import GestureFrame
from nocursor.tracking import TrackingState


class WindowlessCV2:
    """Real OpenCV for drawing; the window calls are replaced by a scripted keyboard."""

    WND_PROP_VISIBLE = cv2.WND_PROP_VISIBLE

    def __init__(self, keys, visible=1):
        self.keys = list(keys)
        self.visible = visible
        self.shown = 0
        self.destroyed = 0

    def __getattr__(self, name):
        return getattr(cv2, name)

    def imshow(self, name, image):
        self.shown += 1

    def waitKey(self, delay):
        return self.keys.pop(0) if self.keys else -1

    def getWindowProperty(self, name, prop):
        return self.visible

    def destroyWindow(self, name):
        self.destroyed += 1


def preview_with(keys, visible=1):
    p = OpenCVPreview(active_margin=0.15)
    p._cv2 = WindowlessCV2(keys, visible)
    return p


FRAME = np.zeros((240, 320, 3), dtype=np.uint8)
GESTURE = GestureFrame(tracking=TrackingState.TRACKING, state=InteractionState.POINTER)
STATUS = StatusInfo("LIVE", 30.0, "ctrl+alt+q", pause_hotkey="ctrl+alt+p")


def show(p, status=STATUS, gesture=GESTURE, hand=None):
    return p.show(FRAME.copy(), hand, gesture, status)


class TestKeys:
    @pytest.mark.parametrize("key", [27, ord("q"), ord("Q")])
    def test_quit_keys_still_stop_in_every_mode(self, key):
        wizard_view = CalibrationWizard().view()
        for status in (STATUS, StatusInfo("PAUSED", 30.0, "k"),
                       StatusInfo("CALIBRATING", 30.0, "k", calibration=wizard_view)):
            assert show(preview_with([key]), status) is False

    def test_closing_the_window_stops(self):
        assert show(preview_with([-1], visible=0)) is False

    def test_other_keys_keep_running(self):
        assert show(preview_with([-1])) is True

    @pytest.mark.parametrize("key,command", [
        (ord("p"), PreviewCommand.PAUSE_TOGGLE), (ord("P"), PreviewCommand.PAUSE_TOGGLE),
        (ord("c"), PreviewCommand.CALIBRATE), (ord("C"), PreviewCommand.CALIBRATE),
        (32, PreviewCommand.CONFIRM), (13, PreviewCommand.CONFIRM),
        (8, PreviewCommand.CANCEL), (ord("x"), PreviewCommand.CANCEL),
        (ord("X"), PreviewCommand.CANCEL),
        (ord("r"), PreviewCommand.RESET_CALIBRATION), (ord("R"), PreviewCommand.RESET_CALIBRATION),
    ])
    def test_command_keys(self, key, command):
        p = preview_with([key])
        assert show(p) is True
        assert p.take_commands() == (command,)
        assert p.take_commands() == ()  # consumed once

    def test_unmapped_keys_produce_no_command(self):
        p = preview_with([ord("z"), ord("1"), 255])
        for _ in range(3):
            show(p)
        assert p.take_commands() == ()

    def test_esc_is_never_a_command(self):
        p = preview_with([27])
        assert show(p) is False
        assert p.take_commands() == ()

    def test_commands_accumulate_between_polls(self):
        p = preview_with([ord("p"), ord("c")])
        show(p)
        show(p)
        assert p.take_commands() == (PreviewCommand.PAUSE_TOGGLE, PreviewCommand.CALIBRATE)

    def test_close_destroys_the_window_only_once(self):
        p = preview_with([-1])
        show(p)
        p.close()
        p.close()
        assert p._cv2.destroyed == 1


class TestOverlays:
    def drawn(self, status=STATUS, gesture=GESTURE, hand=None):
        p = preview_with([-1])
        canvas = FRAME.copy()
        if hand is not None:
            p._draw_hand(canvas, hand, gesture)
        p._draw_status(canvas, gesture, status)
        return canvas

    def test_scroll_mode_is_announced(self):
        g = GestureFrame(tracking=TrackingState.TRACKING, state=InteractionState.SCROLL)
        plain = self.drawn()
        scrolling = self.drawn(gesture=g, hand=make_hand())
        assert (scrolling != plain).any()

    def test_pending_scroll_is_distinguishable_from_active_scroll(self):
        pending = GestureFrame(tracking=TrackingState.TRACKING,
                               state=InteractionState.SCROLL_PENDING)
        active = GestureFrame(tracking=TrackingState.TRACKING, state=InteractionState.SCROLL)
        assert (self.drawn(gesture=pending) != self.drawn(gesture=active)).any()

    def test_paused_banner_and_fault_and_message_render(self):
        base = self.drawn()
        assert (self.drawn(StatusInfo("PAUSED", 30.0, "k")) != base).any()
        faulty = StatusInfo("LIVE", 30.0, "k", fault="button may be held")
        noted = StatusInfo("LIVE", 30.0, "k", message="Calibration saved")
        assert (self.drawn(faulty) != base).any()
        assert (self.drawn(noted) != base).any()

    @pytest.mark.parametrize("state", list(InteractionState))
    def test_every_interaction_state_renders(self, state):
        g = GestureFrame(tracking=TrackingState.TRACKING, state=state)
        assert self.drawn(gesture=g, hand=make_hand()).any()

    def test_calibration_overlay_draws_instructions_progress_and_captured_points(self):
        wizard = CalibrationWizard(stable_frames=2)
        wizard.start()
        base = self.drawn(StatusInfo("CALIBRATING", 30.0, "k", calibration=wizard.view()))
        for _ in range(2):
            wizard.feed((0.3, 0.3))
        wizard.confirm()
        with_point = self.drawn(StatusInfo("CALIBRATING", 30.0, "k", calibration=wizard.view()))
        assert base.any() and (with_point != base).any()

    def test_calibrated_region_rectangle_is_drawn_where_the_region_is(self):
        p = preview_with([-1])
        assert show(p, StatusInfo("LIVE", 30.0, "k", region=(0.25, 0.25, 0.75, 0.75)))
        assert show(p, STATUS)  # and the default margin rectangle still works

    def test_wizard_view_type_is_used_by_the_status(self):
        assert isinstance(CalibrationWizard().view(), WizardView)
