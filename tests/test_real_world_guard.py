"""The test-suite guard itself: tests cannot reach the network, a camera, or the real mouse."""

import socket

import pytest

from conftest import ForbiddenInTests


def test_outbound_network_connections_are_blocked():
    sock = socket.socket()
    try:
        with pytest.raises(ForbiddenInTests, match="network connection"):
            sock.connect(("example.com", 443))
        with pytest.raises(ForbiddenInTests):
            sock.connect(("93.184.216.34", 80))
        with pytest.raises(ForbiddenInTests):
            sock.connect_ex(("8.8.8.8", 53))
    finally:
        sock.close()


def test_loopback_connections_are_still_allowed():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.socket()
    try:
        client.connect(server.getsockname())  # no exception
    finally:
        client.close()
        server.close()


def test_a_camera_cannot_be_opened():
    import cv2

    with pytest.raises(ForbiddenInTests, match="camera"):
        cv2.VideoCapture(0)


@pytest.mark.parametrize("name,args", [
    ("click", ()), ("mouseDown", ()), ("mouseUp", ()), ("moveTo", (10, 10)), ("scroll", (3,)),
])
def test_real_desktop_input_is_blocked(name, args):
    import pyautogui

    with pytest.raises(ForbiddenInTests, match="real desktop input"):
        getattr(pyautogui, name)(*args)


def test_reading_the_screen_size_is_still_possible():
    import pyautogui

    size = pyautogui.size()
    assert size.width > 0 and size.height > 0
