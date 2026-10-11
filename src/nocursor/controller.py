"""Desktop input, gated by a safety layer.

Gesture code never calls PyAutoGUI. It produces intents, and the app passes them
to ``SafeController``, which refuses to act when disabled (dry-run) or after an
emergency stop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from nocursor.emergency import EmergencyStop

logger = logging.getLogger(__name__)


class FailSafeTriggered(Exception):
    """The backend's own fail-safe (e.g. mouse pushed into a screen corner) fired."""


class InputBackend(Protocol):
    def screen_size(self) -> tuple[int, int]: ...
    def move_to(self, x: int, y: int) -> None: ...
    def click(self) -> None: ...


class PyAutoGUIBackend:
    """Thin PyAutoGUI adapter. PyAutoGUI is imported lazily so tests don't need it."""

    def __init__(self) -> None:
        import pyautogui

        pyautogui.FAILSAFE = True  # slamming the mouse into a corner raises FailSafeException
        pyautogui.PAUSE = 0  # no built-in sleep after each call; we run in real time
        self._pg = pyautogui

    def screen_size(self) -> tuple[int, int]:
        size = self._pg.size()
        return int(size.width), int(size.height)

    def move_to(self, x: int, y: int) -> None:
        try:
            self._pg.moveTo(x, y, duration=0)
        except self._pg.FailSafeException as exc:
            raise FailSafeTriggered(str(exc)) from exc

    def click(self) -> None:
        try:
            self._pg.click()
        except self._pg.FailSafeException as exc:
            raise FailSafeTriggered(str(exc)) from exc


class SafeController:
    """Executes cursor moves and left clicks only when it is safe to do so.

    * ``enabled=False`` turns it into a dry-run: actions are counted and logged
      but never sent to the backend (screen size is still read).
    * Once the emergency stop is set, nothing is ever sent again.
    * Any backend error trips the emergency stop rather than being retried.
    """

    def __init__(self, backend: InputBackend, stop: EmergencyStop, enabled: bool = True) -> None:
        self._backend = backend
        self._stop = stop
        self.enabled = enabled
        self.moves_sent = 0
        self.clicks_sent = 0
        self.actions_suppressed = 0

    def screen_size(self) -> tuple[int, int]:
        return self._backend.screen_size()

    def move_to(self, x: int, y: int) -> bool:
        """Move the cursor. Returns True only if input was actually sent."""
        if not self._may_act():
            self.actions_suppressed += 1
            return False
        if not self._send(lambda: self._backend.move_to(x, y)):
            return False
        self.moves_sent += 1
        return True

    def click(self) -> bool:
        """Left-click at the current cursor position. Returns True only if input was sent."""
        if not self._may_act():
            self.actions_suppressed += 1
            if not self._stop.is_set():
                logger.info("dry-run: click suppressed")
            return False
        if not self._send(self._backend.click):
            return False
        self.clicks_sent += 1
        return True

    def _may_act(self) -> bool:
        return self.enabled and not self._stop.is_set()

    def _send(self, action: Callable[[], None]) -> bool:
        try:
            action()
        except FailSafeTriggered:
            self._stop.trigger("PyAutoGUI fail-safe (cursor in screen corner)")
            return False
        except Exception:
            logger.exception("desktop input failed")
            self._stop.trigger("desktop input error")
            return False
        return True
