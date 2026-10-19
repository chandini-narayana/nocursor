"""Desktop input, gated by a safety layer.

Gesture code never calls PyAutoGUI. It produces intents, and the app passes them
to ``SafeController``, which refuses to act when disabled (dry-run), inhibited
(paused / calibrating) or after an emergency stop.

Mouse-button invariants enforced here, independently of the gesture logic:

* At most one left-button press is outstanding.
* A second mouse-down while one is outstanding, or a mouse-up with none
  outstanding, is refused (so repeated frames can never repeat button events).
* Releasing is never blocked by the emergency stop, pause or inhibition, and
  the emergency stop itself triggers a release attempt (even from the hotkey
  thread), so a drag cannot be left stuck.
* A failed action trips the emergency stop; a failed release is recorded in
  ``fault_message`` so the user can be told the button may still be held.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from nocursor.emergency import EmergencyStop

logger = logging.getLogger(__name__)

WINDOWS_WHEEL_DELTA = 120  # Win32 wheel units per notch


class FailSafeTriggered(Exception):
    """The backend's own fail-safe (e.g. mouse pushed into a screen corner) fired."""


class InputBackend(Protocol):
    def screen_size(self) -> tuple[int, int]: ...
    def move_to(self, x: int, y: int) -> None: ...
    def click(self) -> None: ...
    def mouse_down(self) -> None: ...
    def mouse_up(self) -> None: ...
    def scroll(self, clicks: int) -> None: ...
    def right_click(self) -> None: ...


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
        self._guarded(lambda: self._pg.moveTo(x, y, duration=0))

    def click(self) -> None:
        self._guarded(self._pg.click)

    def mouse_down(self) -> None:
        self._guarded(lambda: self._pg.mouseDown(button="left"))

    def mouse_up(self) -> None:
        """Release the left button. The corner fail-safe is bypassed for this call only.

        PyAutoGUI would otherwise raise instead of releasing when the cursor sits
        in a screen corner, which is exactly when a stuck button is most likely.
        """
        previous = self._pg.FAILSAFE
        self._pg.FAILSAFE = False
        try:
            self._pg.mouseUp(button="left")
        finally:
            self._pg.FAILSAFE = previous

    def scroll(self, clicks: int) -> None:
        """Scroll ``clicks`` wheel notches (positive = up).

        PyAutoGUI hands the number straight to Win32's wheel event, where one notch is
        WHEEL_DELTA (120) units; elsewhere it already counts notches.
        """
        amount = clicks * WINDOWS_WHEEL_DELTA if sys.platform == "win32" else clicks
        self._guarded(lambda: self._pg.scroll(amount))

    def right_click(self) -> None:
        """One complete right click (press and release together; the button is never held)."""
        self._guarded(self._pg.rightClick)

    def _guarded(self, action: Callable[[], object]) -> None:
        try:
            action()
        except self._pg.FailSafeException as exc:
            raise FailSafeTriggered(str(exc)) from exc


class InputBlocked(RuntimeError):
    """An input method was called on a backend that is not allowed to send input."""


class ScreenSizeOnlyBackend:
    """A backend that can report a screen size and nothing else.

    Benchmarks use it so that sending desktop input is impossible by construction, not merely
    switched off: every input method raises ``InputBlocked`` (which the controller treats as a
    fault and answers with the emergency stop). ``screen`` fixes the size for reproducibility;
    without it the size is read from ``source`` (a real backend), which is read-only.
    """

    def __init__(
        self, screen: tuple[int, int] | None = None, source: InputBackend | None = None
    ) -> None:
        if screen is None and source is None:
            raise ValueError("give a fixed screen size or a source backend to read it from")
        self._screen = screen
        self._source = source

    def screen_size(self) -> tuple[int, int]:
        if self._screen is not None:
            return self._screen
        assert self._source is not None
        return self._source.screen_size()

    def _blocked(self, what: str) -> None:
        raise InputBlocked(f"{what} is not allowed: this backend can only report the screen size")

    def move_to(self, x: int, y: int) -> None:
        self._blocked("move_to")

    def click(self) -> None:
        self._blocked("click")

    def mouse_down(self) -> None:
        self._blocked("mouse_down")

    def mouse_up(self) -> None:
        self._blocked("mouse_up")

    def scroll(self, clicks: int) -> None:
        self._blocked("scroll")

    def right_click(self) -> None:
        self._blocked("right_click")


@dataclass(frozen=True)
class ActionRecord:
    """One discrete input event, kept so dry-run (and tests) can inspect intent."""

    kind: str  # mouse_down, mouse_up, click, drag, scroll, right_click, release
    detail: str = ""
    sent: bool = False  # True only if it was really sent to the desktop backend


class _Gate(Enum):
    SEND = "send"
    SIMULATE = "simulate"  # dry-run: log and track state, never touch the backend
    REFUSE = "refuse"  # stopped or inhibited


class SafeController:
    """Executes cursor moves, button presses and scrolls only when it is safe to do so.

    * ``enabled=False`` is a dry-run: actions are logged in ``records`` but never
      sent to the backend (screen size is still read). The simulated button
      state is internal and never changes the real desktop's button state.
    * Once the emergency stop is set, nothing new is ever sent again.
    * ``inhibited`` (set while paused or calibrating) blocks new actions too.
    * ``quiet=True`` moves the dry-run narration from INFO to DEBUG (used by benchmarks).
    * Any backend error trips the emergency stop rather than being retried.
    """

    def __init__(
        self,
        backend: InputBackend,
        stop: EmergencyStop,
        enabled: bool = True,
        record_limit: int = 500,
        quiet: bool = False,
    ) -> None:
        self._backend = backend
        self._stop = stop
        self._lock = threading.RLock()  # re-entrant: stop listeners fire inside _send
        self._enabled = enabled
        self._quiet = quiet  # benchmarks: no console I/O inside the timed dispatch stage
        self._inhibited = False
        self._button_down = False
        self.fault_message: str | None = None
        self.records: deque[ActionRecord] = deque(maxlen=record_limit)
        self.moves_sent = 0
        self.clicks_sent = 0
        self.mouse_downs_sent = 0
        self.mouse_ups_sent = 0
        self.scrolls_sent = 0
        self.right_clicks_sent = 0
        self.actions_suppressed = 0
        self.refused_button_events = 0
        self.events_accepted = 0  # discrete events sent or simulated: press, release, scroll
        stop.add_listener(self._on_stop)

    # ------------------------------------------------------------------ state
    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        with self._lock:
            if self._enabled and not value:
                self.release_button("controller disabled")
            self._enabled = value

    @property
    def inhibited(self) -> bool:
        return self._inhibited

    @inhibited.setter
    def inhibited(self, value: bool) -> None:
        with self._lock:
            if value and not self._inhibited:
                self.release_button("input inhibited")
            self._inhibited = value

    @property
    def button_down(self) -> bool:
        """True while a press is outstanding (real, or simulated in dry-run)."""
        return self._button_down

    def screen_size(self) -> tuple[int, int]:
        return self._backend.screen_size()

    # ---------------------------------------------------------------- actions
    def move_to(self, x: int, y: int) -> bool:
        """Move the cursor. Returns True only if input was actually sent."""
        with self._lock:
            if self._gate() is not _Gate.SEND:
                self.actions_suppressed += 1
                return False
            if not self._send(lambda: self._backend.move_to(x, y)):
                return False
            self.moves_sent += 1
            return True

    def click(self) -> bool:
        """Left-click at the current cursor position. Returns True only if input was sent."""
        with self._lock:
            gate = self._gate()
            if gate is not _Gate.SEND:
                self.actions_suppressed += 1
                if gate is _Gate.SIMULATE:
                    self._say("dry-run: click suppressed")
                    self._log(ActionRecord("click", "single"))
                return False
            if self._button_down:
                return self._refuse("click while a button press is outstanding")
            if not self._send(self._backend.click):
                return False
            self.clicks_sent += 1
            self._log(ActionRecord("click", "single", sent=True))
            return True

    def mouse_down(self) -> bool:
        """Press and hold the left button (once). Returns True only if input was sent."""
        with self._lock:
            gate = self._gate()
            if gate is _Gate.REFUSE:
                self.actions_suppressed += 1
                return False
            if self._button_down:
                return self._refuse("mouse_down while a press is already outstanding")
            if gate is _Gate.SIMULATE:
                self._button_down = True
                self._log(ActionRecord("mouse_down"))
                self._say("dry-run: mouse_down (simulated, not sent)")
                return False
            self._button_down = True  # assume pressed until proven otherwise
            if not self._send(self._backend.mouse_down):
                return False  # _send tripped the stop; its listener releases the button
            self.mouse_downs_sent += 1
            self._log(ActionRecord("mouse_down", sent=True))
            return True

    def mouse_up(self, drag: bool = False) -> bool:
        """Release the left button. Never blocked by stop/pause; refused if none is held.

        ``drag`` only labels the dry-run log (click vs drag); it changes no behaviour.
        """
        with self._lock:
            if not self._button_down:
                self.refused_button_events += 1
                logger.debug("ignored mouse_up: no press outstanding")
                return False
            summary = "drag" if drag else "click"
            if not self._enabled:
                self._button_down = False
                self._log(ActionRecord("mouse_up"))
                self._log(ActionRecord(summary, "mouse_down+mouse_up"))
                self._say("dry-run: mouse_up (simulated) -> would be a %s", summary)
                return False
            if not self._release_backend("mouse_up"):
                return False
            self._log(ActionRecord("mouse_up", sent=True))
            self._log(ActionRecord(summary, "mouse_down+mouse_up", sent=True))
            return True

    def release_button(self, reason: str) -> bool:
        """Safety release: idempotent, always allowed. True if nothing is held afterwards."""
        with self._lock:
            if not self._button_down:
                return True
            if not self._enabled:
                self._button_down = False
                self._log(ActionRecord("release", reason))
                self._say("dry-run: simulated button released (%s)", reason)
                return True
            if not self._release_backend(reason):
                return False
            self._log(ActionRecord("release", reason, sent=True))
            logger.warning("released the left mouse button (%s)", reason)
            return True

    def scroll(self, clicks: int) -> bool:
        """Scroll the wheel (positive = up). Returns True only if input was sent."""
        if clicks == 0:
            return False
        with self._lock:
            gate = self._gate()
            if gate is not _Gate.SEND:
                self.actions_suppressed += 1
                if gate is _Gate.SIMULATE:
                    self._log(ActionRecord("scroll", str(clicks)))
                    self._say("dry-run: scroll %+d suppressed", clicks)
                return False
            if self._button_down:
                return self._refuse("scroll while a button press is outstanding")
            if not self._send(lambda: self._backend.scroll(clicks)):
                return False
            self.scrolls_sent += 1
            self._log(ActionRecord("scroll", str(clicks), sent=True))
            return True

    def right_click(self) -> bool:
        """Right-click at the current cursor position. Returns True only if input was sent.

        A complete click, never a held button. Refused while a left press is outstanding (real or
        simulated), so a drag can never be turned into, or interrupted by, a right click.
        """
        with self._lock:
            gate = self._gate()
            if gate is _Gate.REFUSE:
                self.actions_suppressed += 1
                return False
            if self._button_down:
                return self._refuse("right click while a button press is outstanding")
            if gate is _Gate.SIMULATE:
                self.actions_suppressed += 1
                self._log(ActionRecord("right_click"))
                self._say("dry-run: right click suppressed")
                return False
            if not self._send(self._backend.right_click):
                return False
            self.right_clicks_sent += 1
            self._log(ActionRecord("right_click", sent=True))
            return True

    # --------------------------------------------------------------- internals
    def _say(self, message: str, *args: object) -> None:
        """Dry-run narration: INFO normally, DEBUG when ``quiet`` (so timing is not perturbed)."""
        logger.log(logging.DEBUG if self._quiet else logging.INFO, message, *args)

    def _log(self, record: ActionRecord) -> None:
        self.records.append(record)
        discrete = record.kind in ("mouse_down", "mouse_up", "scroll", "right_click") or (
            record.kind == "click" and record.detail == "single"
        )
        if discrete:
            self.events_accepted += 1

    def _gate(self) -> _Gate:
        if self._stop.is_set() or self._inhibited:
            return _Gate.REFUSE
        return _Gate.SEND if self._enabled else _Gate.SIMULATE

    def _refuse(self, why: str) -> bool:
        self.refused_button_events += 1
        logger.warning("refused: %s", why)
        return False

    def _release_backend(self, reason: str) -> bool:
        """Send a real mouse-up. On failure: record why, block everything, keep state 'held'."""
        try:
            self._backend.mouse_up()
        except Exception as exc:
            self.fault_message = (
                f"Could not release the left mouse button ({reason}): {exc}. "
                "It may still be held down: click the physical mouse button to clear it."
            )
            logger.error(self.fault_message)
            self._stop.trigger("mouse button release failed")
            return False
        self._button_down = False
        self.mouse_ups_sent += 1
        return True

    def _on_stop(self, reason: str) -> None:
        self.release_button(f"emergency stop: {reason}")

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
