"""SafeController button/scroll safety, dry-run recording, and stop/pause plumbing."""

import threading
import time

from nocursor.controller import FailSafeTriggered, SafeController
from nocursor.emergency import EmergencyStop, HotkeyAction, PauseControl


class Backend:
    def __init__(self):
        self.events: list[tuple] = []
        self.fail: dict[str, Exception] = {}

    def _do(self, name, *args):
        if name in self.fail:
            raise self.fail[name]
        self.events.append((name, *args))

    def screen_size(self):
        return (1280, 720)

    def move_to(self, x, y):
        self._do("move", x, y)

    def click(self):
        self._do("click")

    def mouse_down(self):
        self._do("down")

    def mouse_up(self):
        self._do("up")

    def scroll(self, clicks):
        self._do("scroll", clicks)

    def names(self):
        return [e[0] for e in self.events]


def make(enabled=True):
    backend, stop = Backend(), EmergencyStop()
    return SafeController(backend, stop, enabled=enabled), backend, stop


class TestButtonInvariants:
    def test_down_then_up_reaches_the_backend_once_each(self):
        c, b, _ = make()
        assert c.mouse_down() and c.button_down
        assert c.mouse_up() and not c.button_down
        assert b.names() == ["down", "up"]

    def test_second_mouse_down_while_held_is_refused(self):
        c, b, _ = make()
        c.mouse_down()
        assert not c.mouse_down()
        assert b.names() == ["down"]
        assert c.refused_button_events == 1

    def test_mouse_up_without_a_press_is_ignored(self):
        c, b, _ = make()
        assert not c.mouse_up()
        assert b.events == []

    def test_repeated_mouse_up_sends_only_one(self):
        c, b, _ = make()
        c.mouse_down()
        for _ in range(10):
            c.mouse_up()
        assert b.names().count("up") == 1

    def test_click_and_scroll_are_refused_while_a_press_is_outstanding(self):
        c, b, _ = make()
        c.mouse_down()
        assert not c.click() and not c.scroll(3)
        assert b.names() == ["down"]

    def test_moves_are_still_allowed_during_a_press_so_dragging_works(self):
        c, b, _ = make()
        c.mouse_down()
        assert c.move_to(5, 6)
        assert b.names() == ["down", "move"]

    def test_release_button_is_idempotent(self):
        c, b, _ = make()
        c.mouse_down()
        assert c.release_button("test") and c.release_button("test")
        assert b.names().count("up") == 1

    def test_release_with_nothing_held_sends_nothing(self):
        c, b, _ = make()
        assert c.release_button("test")
        assert b.events == []


class TestStopAndInhibit:
    def test_emergency_stop_releases_a_held_button_immediately(self):
        c, b, stop = make()
        c.mouse_down()
        stop.trigger("hotkey")
        assert b.names() == ["down", "up"]
        assert not c.button_down

    def test_release_works_from_another_thread_while_the_main_loop_is_blocked(self):
        c, b, stop = make()
        c.mouse_down()
        t = threading.Thread(target=stop.trigger, args=("hotkey thread",))
        t.start()
        t.join(timeout=2)
        assert b.names() == ["down", "up"]

    def test_nothing_new_is_sent_after_a_stop(self):
        c, b, stop = make()
        stop.trigger("test")
        assert not c.mouse_down() and not c.move_to(1, 1) and not c.scroll(2) and not c.click()
        assert b.events == []

    def test_mouse_up_is_still_allowed_after_a_stop_if_something_is_held(self):
        c, b, stop = make()
        c.mouse_down()
        stop.trigger("test")
        assert b.names().count("up") == 1  # released by the stop listener
        assert not c.mouse_up()  # a second request is ignored, not repeated
        assert b.names().count("up") == 1

    def test_controller_created_after_a_stop_is_already_blocked(self):
        stop = EmergencyStop()
        stop.trigger("earlier")
        b = Backend()
        c = SafeController(b, stop)
        assert not c.mouse_down()
        assert b.events == []

    def test_inhibited_blocks_new_actions_but_not_release(self):
        c, b, _ = make()
        c.mouse_down()
        c.inhibited = True  # e.g. paused / calibrating
        assert b.names() == ["down", "up"]  # inhibiting released the press
        assert not c.mouse_down() and not c.move_to(1, 1) and not c.scroll(1)
        c.inhibited = False
        assert c.move_to(1, 1)

    def test_disabling_the_controller_at_runtime_releases_a_held_button(self):
        c, b, _ = make()
        c.mouse_down()
        c.enabled = False
        assert b.names() == ["down", "up"]


class TestFailures:
    def test_failed_mouse_down_trips_stop_and_leaves_nothing_held(self):
        c, b, stop = make()
        b.fail["down"] = RuntimeError("denied")
        assert not c.mouse_down()
        assert stop.is_set() and not c.button_down
        assert c.mouse_downs_sent == 0

    def test_failsafe_during_a_drag_stops_and_releases_despite_the_corner(self):
        c, b, stop = make()
        c.mouse_down()
        b.fail["move"] = FailSafeTriggered("corner")
        assert not c.move_to(0, 0)
        assert stop.is_set() and "fail-safe" in stop.reason
        assert b.names() == ["down", "up"]  # release is attempted, and it succeeded

    def test_failed_release_is_recorded_blocks_everything_and_is_reported(self):
        c, b, stop = make()
        c.mouse_down()
        b.fail["up"] = OSError("input desktop unavailable")
        assert not c.mouse_up()
        assert stop.is_set()
        assert "may still be held" in c.fault_message
        assert "input desktop unavailable" in c.fault_message
        assert c.button_down  # unknown state is not claimed to be released
        assert not c.move_to(1, 1) and not c.mouse_down()

    def test_a_later_successful_release_attempt_clears_the_held_state(self):
        c, b, _ = make()
        c.mouse_down()
        b.fail["up"] = OSError("busy")
        c.mouse_up()
        del b.fail["up"]
        assert c.release_button("shutdown retry")
        assert not c.button_down

    def test_failure_on_move_trips_the_stop(self):
        c, b, stop = make()
        b.fail["move"] = RuntimeError("boom")
        assert not c.move_to(1, 1)
        assert stop.is_set()

    def test_listener_exceptions_do_not_break_the_stop(self):
        stop = EmergencyStop()

        def boom(_reason):
            raise RuntimeError("listener bug")

        stop.add_listener(boom)
        stop.trigger("test")
        assert stop.is_set()


class TestDryRun:
    def test_dry_run_never_touches_the_backend(self):
        c, b, _ = make(enabled=False)
        c.move_to(1, 1)
        c.mouse_down()
        c.move_to(2, 2)
        c.scroll(3)
        c.mouse_up(drag=True)
        c.click()
        c.release_button("x")
        assert b.events == []

    def test_dry_run_records_intended_actions(self):
        c, _, _ = make(enabled=False)
        c.mouse_down()
        c.mouse_up(drag=False)
        c.mouse_down()
        c.mouse_up(drag=True)
        c.scroll(-2)
        kinds = [(r.kind, r.sent) for r in c.records]
        assert ("mouse_down", False) in kinds and ("mouse_up", False) in kinds
        assert ("click", False) in kinds and ("drag", False) in kinds
        assert ("scroll", False) in kinds
        assert not any(sent for _, sent in kinds)

    def test_dry_run_keeps_the_same_button_invariants(self):
        c, _, _ = make(enabled=False)
        c.mouse_down()
        assert c.button_down
        c.mouse_down()
        assert c.refused_button_events == 1
        c.mouse_up()
        assert not c.button_down
        c.mouse_up()
        assert c.refused_button_events == 2

    def test_dry_run_return_values_say_nothing_was_sent(self):
        c, _, _ = make(enabled=False)
        assert c.mouse_down() is False and c.mouse_up() is False
        assert c.scroll(1) is False and c.move_to(1, 1) is False

    def test_dry_run_stop_releases_the_simulated_button(self):
        c, b, stop = make(enabled=False)
        c.mouse_down()
        stop.trigger("test")
        assert not c.button_down and b.events == []

    def test_record_log_is_bounded(self):
        backend, stop = Backend(), EmergencyStop()
        c = SafeController(backend, stop, enabled=False, record_limit=10)
        for _ in range(100):
            c.mouse_down()
            c.mouse_up()
        assert len(c.records) == 10

    def test_enabled_controller_records_sent_actions(self):
        c, _, _ = make()
        c.mouse_down()
        c.mouse_up(drag=True)
        c.scroll(2)
        assert [(r.kind, r.sent) for r in c.records] == [
            ("mouse_down", True), ("mouse_up", True), ("drag", True), ("scroll", True),
        ]


class TestScroll:
    def test_scroll_is_sent_with_its_sign(self):
        c, b, _ = make()
        c.scroll(3)
        c.scroll(-2)
        assert b.events == [("scroll", 3), ("scroll", -2)]

    def test_zero_scroll_is_a_no_op(self):
        c, b, _ = make()
        assert not c.scroll(0)
        assert b.events == []


class TestPauseControlAndHotkeyAction:
    def test_pause_control_toggles(self):
        p = PauseControl()
        assert not p.paused
        assert p.toggle() is True and p.paused
        assert p.toggle() is False and not p.paused
        p.set_paused(True)
        assert p.paused

    def wait_for(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not predicate():
            time.sleep(0.005)
        return predicate()

    class PolledKeys:
        """Key state that counts how often the watcher thread has read it."""

        def __init__(self):
            self.down: set[int] = set()
            self.polls = 0

        def __call__(self, vk):
            self.polls += 1
            return vk in self.down

    def let_the_watcher_see_the_keys_up(self, keys):
        """Wait until the polling thread has demonstrably read the released state.

        A fixed sleep is not enough: on a loaded machine the thread can be starved for longer
        than any sleep, never observe the keys up, and so never see the next press as a new one.
        """
        target = keys.polls + 8
        assert self.wait_for(lambda: keys.polls >= target)

    def test_hotkey_action_fires_once_per_press(self):
        keys = self.PolledKeys()
        calls: list[int] = []
        action = HotkeyAction((1, 2), keys, lambda: calls.append(1), 0.001)
        action.start()
        try:
            keys.down.update({1, 2})
            assert self.wait_for(lambda: len(calls) == 1)
            held_until = keys.polls + 20  # keys still held across many polls: must not repeat
            assert self.wait_for(lambda: keys.polls >= held_until)
            assert len(calls) == 1
            keys.down.clear()
            self.let_the_watcher_see_the_keys_up(keys)
            keys.down.update({1, 2})
            assert self.wait_for(lambda: len(calls) == 2)
        finally:
            action.close()

    def test_hotkey_action_ignores_partial_combos(self):
        calls: list[int] = []
        action = HotkeyAction((1, 2), {1}.__contains__, lambda: calls.append(1), 0.001)
        action.start()
        time.sleep(0.05)
        action.close()
        assert calls == []

    def test_hotkey_action_survives_an_action_exception(self):
        keys = self.PolledKeys()
        calls: list[int] = []

        def flaky():
            calls.append(1)
            raise RuntimeError("bug")

        action = HotkeyAction((1,), keys, flaky, 0.001)
        action.start()
        try:
            keys.down.add(1)
            assert self.wait_for(lambda: len(calls) == 1)
            keys.down.clear()
            self.let_the_watcher_see_the_keys_up(keys)
            keys.down.add(1)
            assert self.wait_for(lambda: len(calls) == 2)
        finally:
            action.close()

    def test_emergency_listener_registered_late_fires_immediately(self):
        stop = EmergencyStop()
        stop.trigger("early")
        seen = []
        stop.add_listener(seen.append)
        assert seen == ["early"]

    def test_listeners_run_exactly_once(self):
        stop = EmergencyStop()
        seen = []
        stop.add_listener(seen.append)
        stop.trigger("a")
        stop.trigger("b")
        assert seen == ["a"]
