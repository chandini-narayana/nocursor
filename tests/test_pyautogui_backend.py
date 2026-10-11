"""The real PyAutoGUI adapter, exercised against a fake ``pyautogui`` module.

Nothing here touches the real desktop: the module is replaced before the adapter imports it.
"""

import sys
import types

import pytest

from nocursor.controller import FailSafeTriggered, PyAutoGUIBackend, SafeController
from nocursor.emergency import EmergencyStop


class FakeFailSafe(Exception):
    pass


class FakePyAutoGUI(types.ModuleType):
    FailSafeException = FakeFailSafe

    def __init__(self):
        super().__init__("pyautogui")
        self.FAILSAFE = False
        self.PAUSE = 1.0
        self.corner = False  # pretend the pointer is in a screen corner
        self.calls: list[tuple] = []
        self.mouse_up_error: Exception | None = None

    def _check(self):
        if self.FAILSAFE and self.corner:
            raise FakeFailSafe("pointer in corner")

    def size(self):
        return types.SimpleNamespace(width=1920, height=1080)

    def moveTo(self, x, y, duration=0):
        self._check()
        self.calls.append(("moveTo", x, y, duration))

    def click(self):
        self._check()
        self.calls.append(("click",))

    def mouseDown(self, button="left"):
        self._check()
        self.calls.append(("mouseDown", button))

    def mouseUp(self, button="left"):
        self._check()  # the real function is failsafe-checked too
        if self.mouse_up_error:
            raise self.mouse_up_error
        self.calls.append(("mouseUp", button))

    def scroll(self, clicks):
        self._check()
        self.calls.append(("scroll", clicks))


@pytest.fixture
def pg(monkeypatch):
    fake = FakePyAutoGUI()
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    return fake


def test_construction_enables_the_failsafe_and_removes_the_built_in_pause(pg):
    PyAutoGUIBackend()
    assert pg.FAILSAFE is True and pg.PAUSE == 0


def test_screen_size_is_read_as_plain_ints(pg):
    assert PyAutoGUIBackend().screen_size() == (1920, 1080)


def test_actions_pass_through_with_the_left_button_and_signed_scroll(pg, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")  # notches pass through unchanged
    b = PyAutoGUIBackend()
    b.move_to(10, 20)
    b.click()
    b.mouse_down()
    b.mouse_up()
    b.scroll(3)
    b.scroll(-2)
    assert pg.calls == [
        ("moveTo", 10, 20, 0), ("click",), ("mouseDown", "left"), ("mouseUp", "left"),
        ("scroll", 3), ("scroll", -2),
    ]


def test_on_windows_scroll_notches_are_converted_to_wheel_units(pg, monkeypatch):
    """PyAutoGUI sends the number raw to Win32, where one notch is 120 units."""
    monkeypatch.setattr(sys, "platform", "win32")
    b = PyAutoGUIBackend()
    b.scroll(1)
    b.scroll(3)
    b.scroll(-2)
    assert pg.calls == [("scroll", 120), ("scroll", 360), ("scroll", -240)]


@pytest.mark.parametrize("call", [
    lambda b: b.move_to(1, 1), lambda b: b.click(), lambda b: b.mouse_down(),
    lambda b: b.scroll(1),
])
def test_failsafe_in_a_corner_becomes_failsafe_triggered(pg, call):
    b = PyAutoGUIBackend()
    pg.corner = True
    with pytest.raises(FailSafeTriggered):
        call(b)
    assert pg.calls == []


def test_mouse_up_bypasses_the_corner_failsafe_and_restores_it(pg):
    b = PyAutoGUIBackend()
    pg.corner = True
    b.mouse_up()  # must release even though a normal call would raise here
    assert pg.calls == [("mouseUp", "left")]
    assert pg.FAILSAFE is True  # and the safety net is back on afterwards


def test_failsafe_is_restored_even_if_the_release_itself_fails(pg):
    b = PyAutoGUIBackend()
    pg.mouse_up_error = OSError("blocked")
    with pytest.raises(OSError):
        b.mouse_up()
    assert pg.FAILSAFE is True


def test_corner_during_a_drag_still_ends_with_a_real_mouse_up(pg):
    stop = EmergencyStop()
    controller = SafeController(PyAutoGUIBackend(), stop)
    assert controller.mouse_down()
    pg.corner = True  # the physical mouse is slammed into a corner mid-drag
    assert not controller.move_to(0, 0)
    assert stop.is_set() and "fail-safe" in stop.reason
    assert ("mouseUp", "left") in pg.calls  # the button was released despite the corner
    assert not controller.button_down
