import socket
import sys
from pathlib import Path

import pytest

# Make `helpers` importable from test modules.
sys.path.insert(0, str(Path(__file__).parent))

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class ForbiddenInTests(AssertionError):
    """A unit test tried to reach the real world (network, camera or desktop input)."""


@pytest.fixture(autouse=True)
def no_real_world_access(monkeypatch):
    """Unit tests must never touch the network, a webcam, or the real mouse.

    * Outbound network connections (anything but loopback) raise.
    * ``cv2.VideoCapture`` raises, so no test can open a camera by accident.
    * The real PyAutoGUI input functions raise (tests that exercise the PyAutoGUI adapter
      install a fake ``pyautogui`` module instead).

    A test that needs one of these replaces it with its own fake via ``monkeypatch``.
    """
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def is_loopback(address) -> bool:
        return isinstance(address, tuple) and str(address[0]) in _LOOPBACK

    def guarded_connect(self, address):
        if not is_loopback(address):
            raise ForbiddenInTests(f"a test tried to open a network connection to {address!r}")
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if not is_loopback(address):
            raise ForbiddenInTests(f"a test tried to open a network connection to {address!r}")
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)

    try:
        import cv2

        def no_camera(*args, **kwargs):
            raise ForbiddenInTests("a test tried to open a camera / video device")

        monkeypatch.setattr(cv2, "VideoCapture", no_camera)
    except ImportError:
        pass

    try:
        import pyautogui

        def no_input(*args, **kwargs):
            raise ForbiddenInTests("a test tried to send real desktop input through PyAutoGUI")

        for name in ("moveTo", "move", "click", "mouseDown", "mouseUp", "scroll", "hscroll",
                     "vscroll", "drag", "dragTo", "press", "typewrite", "write", "hotkey"):
            if hasattr(pyautogui, name):
                monkeypatch.setattr(pyautogui, name, no_input)
    except ImportError:
        pass
