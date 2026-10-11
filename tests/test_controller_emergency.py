import time

import pytest

from nocursor.controller import FailSafeTriggered, SafeController
from nocursor.emergency import EmergencyStop, HotkeyWatcher, parse_hotkey


class FakeBackend:
    def __init__(self, fail_with: Exception | None = None):
        self.moves: list[tuple[int, int]] = []
        self.clicks = 0
        self.fail_with = fail_with

    def screen_size(self):
        return (1280, 720)

    def move_to(self, x, y):
        if self.fail_with:
            raise self.fail_with
        self.moves.append((x, y))

    def click(self):
        if self.fail_with:
            raise self.fail_with
        self.clicks += 1


class TestSafeController:
    def test_actions_reach_backend_when_enabled(self):
        backend, stop = FakeBackend(), EmergencyStop()
        c = SafeController(backend, stop)
        assert c.move_to(10, 20) and c.click()
        assert backend.moves == [(10, 20)] and backend.clicks == 1
        assert (c.moves_sent, c.clicks_sent) == (1, 1)

    def test_disabled_controller_sends_nothing(self):
        backend = FakeBackend()
        c = SafeController(backend, EmergencyStop(), enabled=False)
        assert not c.move_to(1, 1) and not c.click()
        assert backend.moves == [] and backend.clicks == 0
        assert c.actions_suppressed == 2

    def test_can_be_toggled_at_runtime(self):
        backend = FakeBackend()
        c = SafeController(backend, EmergencyStop(), enabled=False)
        c.click()
        c.enabled = True
        c.click()
        assert backend.clicks == 1

    def test_screen_size_comes_from_backend_even_when_disabled(self):
        assert SafeController(FakeBackend(), EmergencyStop(), enabled=False).screen_size() == (
            1280, 720)

    def test_nothing_is_sent_after_emergency_stop(self):
        backend, stop = FakeBackend(), EmergencyStop()
        c = SafeController(backend, stop)
        stop.trigger("test")
        assert not c.move_to(1, 1) and not c.click()
        assert backend.moves == [] and backend.clicks == 0

    def test_failsafe_trips_emergency_stop(self):
        stop = EmergencyStop()
        c = SafeController(FakeBackend(FailSafeTriggered("corner")), stop)
        assert not c.move_to(0, 0)
        assert stop.is_set() and "fail-safe" in stop.reason
        assert not c.click()

    def test_backend_error_trips_emergency_stop_instead_of_retrying(self):
        backend, stop = FakeBackend(RuntimeError("access denied")), EmergencyStop()
        c = SafeController(backend, stop)
        assert not c.click()
        assert stop.is_set()
        assert (c.moves_sent, c.clicks_sent) == (0, 0)


class TestEmergencyStop:
    def test_latches_with_first_reason(self):
        stop = EmergencyStop()
        assert not stop.is_set()
        stop.trigger("first")
        stop.trigger("second")
        assert stop.is_set() and stop.reason == "first"


class TestParseHotkey:
    def test_modifiers_and_letter(self):
        assert parse_hotkey("ctrl+alt+q") == (0x11, 0x12, ord("Q"))

    def test_case_and_spacing_insensitive(self):
        assert parse_hotkey(" Ctrl + SHIFT + F12 ") == (0x11, 0x10, 0x7B)

    def test_single_named_key(self):
        assert parse_hotkey("esc") == (0x1B,)

    @pytest.mark.parametrize("bad", ["", "  ", "ctrl+alt", "ctrl+nonsense", "ctrl+ctrl+q", "+"])
    def test_invalid_hotkeys_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_hotkey(bad)


class TestHotkeyWatcher:
    def wait_for(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.005)
        return predicate()

    def test_trips_stop_when_all_keys_down(self):
        stop, down = EmergencyStop(), {0x11, 0x12, ord("Q")}
        watcher = HotkeyWatcher(stop, (0x11, 0x12, ord("Q")), down.__contains__, 0.001)
        watcher.start()
        try:
            assert self.wait_for(stop.is_set)
        finally:
            watcher.close()

    def test_does_not_trip_on_partial_combo(self):
        stop = EmergencyStop()
        watcher = HotkeyWatcher(stop, (0x11, 0x12, ord("Q")), {0x11, 0x12}.__contains__, 0.001)
        watcher.start()
        time.sleep(0.05)
        watcher.close()
        assert not stop.is_set()

    def test_trips_later_when_key_pressed_while_main_thread_is_busy(self):
        stop, down = EmergencyStop(), set()
        watcher = HotkeyWatcher(stop, (1,), down.__contains__, 0.001)
        watcher.start()
        try:
            time.sleep(0.02)
            assert not stop.is_set()
            down.add(1)
            assert self.wait_for(stop.is_set)
        finally:
            watcher.close()

    def test_polling_failure_stops_for_safety(self):
        def boom(_vk):
            raise OSError("no access")

        stop = EmergencyStop()
        watcher = HotkeyWatcher(stop, (1,), boom, 0.001)
        watcher.start()
        try:
            assert self.wait_for(stop.is_set)
        finally:
            watcher.close()
        assert "watcher failure" in stop.reason

    def test_close_stops_thread(self):
        watcher = HotkeyWatcher(EmergencyStop(), (1,), lambda _vk: False, 0.001)
        watcher.start()
        watcher.close()
        assert not watcher._thread.is_alive()
