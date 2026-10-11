"""Emergency stop: a thread-safe latch plus a global hotkey watcher."""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)

_MODIFIER_KEYS = {"ctrl": 0x11, "alt": 0x12, "shift": 0x10}
_NAMED_KEYS = {"esc": 0x1B, "pause": 0x13, "end": 0x23, "home": 0x24, "delete": 0x2E}
_F_KEYS = {f"f{n}": 0x6F + n for n in range(1, 13)}


class EmergencyStop:
    """A latch that, once triggered, stays triggered. Safe to use from any thread."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None

    def trigger(self, reason: str) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()
        logger.warning("EMERGENCY STOP: %s", reason)

    def is_set(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason


def parse_hotkey(spec: str) -> tuple[int, ...]:
    """Parse e.g. ``"ctrl+alt+q"`` into Windows virtual-key codes.

    Raises ``ValueError`` for empty/unknown parts or a combo with no non-modifier key.
    """
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError("hotkey is empty")
    codes: list[int] = []
    has_main_key = False
    for part in parts:
        if part in _MODIFIER_KEYS:
            codes.append(_MODIFIER_KEYS[part])
            continue
        if part in _NAMED_KEYS:
            codes.append(_NAMED_KEYS[part])
        elif part in _F_KEYS:
            codes.append(_F_KEYS[part])
        elif len(part) == 1 and part.isascii() and part.isalnum():
            codes.append(ord(part.upper()))
        else:
            raise ValueError(f"unknown hotkey part: {part!r}")
        has_main_key = True
    if not has_main_key:
        raise ValueError("hotkey needs a non-modifier key")
    if len(set(codes)) != len(codes):
        raise ValueError("hotkey has duplicate keys")
    return tuple(codes)


def system_key_state() -> Callable[[int], bool] | None:
    """Return a ``is_key_down(vk)`` function for the whole desktop, or None if unsupported."""
    if sys.platform != "win32":
        return None
    import ctypes

    get_async_key_state = ctypes.windll.user32.GetAsyncKeyState  # type: ignore[attr-defined]
    return lambda vk: bool(get_async_key_state(vk) & 0x8000)


class HotkeyWatcher:
    """Polls a key combination on a background thread and trips the emergency stop.

    It runs independently of the video loop, so the stop works even if capture
    or inference is stalled, and regardless of which window has focus.
    """

    def __init__(
        self,
        stop: EmergencyStop,
        keys: tuple[int, ...],
        is_key_down: Callable[[int], bool],
        poll_interval_s: float = 0.01,
    ) -> None:
        self._stop = stop
        self._keys = keys
        self._is_key_down = is_key_down
        self._poll_interval_s = poll_interval_s
        self._quit = threading.Event()
        self._thread = threading.Thread(target=self._run, name="nocursor-hotkey", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._quit.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._quit.is_set() and not self._stop.is_set():
            try:
                pressed = all(self._is_key_down(vk) for vk in self._keys)
            except Exception:  # never let the watcher die silently without stopping
                logger.exception("hotkey polling failed; stopping for safety")
                self._stop.trigger("hotkey watcher failure")
                return
            if pressed:
                self._stop.trigger("emergency hotkey pressed")
                return
            self._quit.wait(self._poll_interval_s)
