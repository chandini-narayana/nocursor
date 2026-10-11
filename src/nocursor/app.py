"""The real-time loop that wires camera -> tracker -> gestures -> desktop."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from nocursor.calibration import CalibrationError, CalibrationStore, CalibrationWizard, WizardState
from nocursor.camera import Camera, CameraError
from nocursor.controller import SafeController
from nocursor.diagnostics import DiagnosticsReporter
from nocursor.emergency import EmergencyStop, PauseControl
from nocursor.interaction import MouseDown, MouseUp, MoveTo, Scroll
from nocursor.mapping import ViewRegion
from nocursor.models import HandObservation
from nocursor.preview import Preview, PreviewCommand, StatusInfo
from nocursor.processor import GestureFrame, GestureProcessor
from nocursor.tracking import TrackingState

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_EMERGENCY = 2


class Tracker(Protocol):
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandObservation | None: ...
    def close(self) -> None: ...


class Closable(Protocol):
    def start(self) -> None: ...
    def close(self) -> None: ...


class NoCursorApp:
    """Runs the control loop; every collaborator is injected so the loop is testable.

    The loop never decides what to do with a gesture: ``GestureProcessor`` returns
    typed actions and ``SafeController`` executes (or refuses) them. While paused or
    calibrating the controller is inhibited, so no new desktop input can happen.

    ``run()`` always releases a held mouse button first, then the hotkey watchers,
    preview, tracker and camera, whatever caused the loop to end.
    """

    def __init__(
        self,
        camera: Camera,
        tracker: Tracker,
        processor: GestureProcessor,
        controller: SafeController,
        preview: Preview,
        stop: EmergencyStop,
        hotkey_label: str,
        watcher: Closable | None = None,
        clock: Callable[[], float] = time.monotonic,
        *,
        pause: PauseControl | None = None,
        wizard: CalibrationWizard | None = None,
        store: CalibrationStore | None = None,
        extra_watchers: Sequence[Closable] = (),
        camera_index: int = 0,
        pause_hotkey_label: str = "",
        start_in_calibration: bool = False,
        message_seconds: float = 4.0,
        diagnostics: DiagnosticsReporter | None = None,
    ) -> None:
        self._camera = camera
        self._tracker = tracker
        self._processor = processor
        self._controller = controller
        self._preview = preview
        self._stop = stop
        self._hotkey_label = hotkey_label
        self._watchers = ([watcher] if watcher is not None else []) + list(extra_watchers)
        self._clock = clock
        self._pause = pause if pause is not None else PauseControl()
        self._wizard = wizard
        self._store = store
        self._camera_index = camera_index
        self._pause_hotkey_label = pause_hotkey_label
        self._start_in_calibration = start_in_calibration
        self._message_seconds = message_seconds
        self._message: tuple[str, float] | None = None
        self._diagnostics = diagnostics
        self._diagnostic_lines: tuple[str, ...] = ()
        self._aspect = 4 / 3
        self._fps = 0.0

    def run(self) -> int:
        try:
            for watcher in self._watchers:
                watcher.start()
            self._camera.open()
            self._loop()
        except CameraError as exc:
            logger.error("camera error: %s", exc)
            return EXIT_ERROR
        except KeyboardInterrupt:
            logger.info("interrupted by Ctrl+C")
            return EXIT_OK
        except Exception:
            self._stop.trigger("unexpected error")
            logger.exception("unexpected error in main loop")
            return EXIT_ERROR
        finally:
            self._shutdown()
        return EXIT_EMERGENCY if self._stop.is_set() else EXIT_OK

    # ------------------------------------------------------------------ loop
    @property
    def _calibrating(self) -> bool:
        return self._wizard is not None and self._wizard.active

    def _loop(self) -> None:
        previous = self._clock()
        if self._start_in_calibration:
            self._begin_calibration()
        while not self._stop.is_set():
            frame = self._camera.read()
            now = self._clock()
            self._update_fps(now - previous)
            previous = now

            if frame is None:  # dropped frame: treat as tracking lost, take no action
                observation = None
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                self._aspect = frame.shape[1] / frame.shape[0]
                observation = self._tracker.detect(frame, int(now * 1000))

            calibrating = self._calibrating
            self._controller.inhibited = calibrating or self._pause.paused
            if calibrating:
                gesture = self._processor.idle(observation, now)
                self._feed_wizard(gesture, observation)
            else:
                gesture = self._processor.process(observation, now, paused=self._pause.paused)

            self._report_diagnostics(now)  # read-only: shows/logs, never acts

            if not self._stop.is_set():  # the hotkey may have fired during inference
                self._reconcile(gesture)  # never act on top of a press the machine does not know
                self._apply(gesture)
            self._reconcile()

            keep_going = self._preview.show(frame, observation, gesture, self._status(gesture))
            if not keep_going:
                self._stop.trigger("quit requested from preview window")
            for command in self._preview.take_commands():
                self._handle_command(command)

    def _report_diagnostics(self, now: float) -> None:
        self._diagnostic_lines = ()
        diag = self._processor.diagnostics
        if self._diagnostics is None or diag is None:
            return
        self._diagnostics.observe(diag, now)
        self._diagnostic_lines = self._diagnostics.overlay_lines(diag)
        line = self._diagnostics.log_line(diag, now)  # at most one per interval
        if line is not None:
            logger.info(line)

    def _apply(self, gesture: GestureFrame) -> None:
        for action in gesture.actions:
            if isinstance(action, MoveTo):
                self._controller.move_to(action.x, action.y)
            elif isinstance(action, MouseDown):
                self._controller.mouse_down()
                if not self._controller.button_down:  # refused: never pretend it is pressed
                    self._processor.abort_interaction()
            elif isinstance(action, MouseUp):
                self._controller.mouse_up(drag=action.drag)
            elif isinstance(action, Scroll):
                self._controller.scroll(action.clicks)

    def _reconcile(self, pending: GestureFrame | None = None) -> None:
        """Invariant check: the controller never holds a button the state machine dropped.

        Before a frame is applied, a held button is expected to differ from the machine
        exactly when that frame carries the machine's own MouseUp, which is left to run.
        """
        if not self._controller.button_down or self._processor.holding_button:
            return
        if pending is not None and any(isinstance(a, MouseUp) for a in pending.actions):
            return
        self._controller.release_button("state machine and controller disagreed")

    # -------------------------------------------------------------- commands
    def _handle_command(self, command: PreviewCommand) -> None:
        if command is PreviewCommand.PAUSE_TOGGLE:
            if self._calibrating:
                self._notify("Finish or cancel calibration before pausing.")
            elif self._pause.toggle():
                self._notify("Paused. Press P again to resume.")
            else:
                self._notify("Resuming: hold an open hand steady until tracking is stable.")
        elif command is PreviewCommand.CALIBRATE:
            self._begin_calibration()
        elif command is PreviewCommand.CONFIRM:
            if self._wizard is not None and self._calibrating:
                self._wizard.confirm()
                self._after_wizard()
        elif command is PreviewCommand.CANCEL:
            if self._wizard is not None and self._calibrating:
                self._wizard.cancel()
                self._after_wizard()
        elif command is PreviewCommand.RESET_CALIBRATION:
            self._reset_calibration()

    def _begin_calibration(self) -> None:
        if self._wizard is None:
            self._notify("Calibration is not available in this session.")
        elif not self._calibrating:
            self._wizard.start()  # next frame: controller inhibited, state machine releases

    def _feed_wizard(self, gesture: GestureFrame, observation: HandObservation | None) -> None:
        assert self._wizard is not None
        tip = None
        if observation is not None and gesture.tracking is TrackingState.TRACKING:
            x, y = observation.index_tip
            tip = (1.0 - x, y)  # mirrored view coordinates, as shown in the preview
        self._wizard.feed(tip)

    def _after_wizard(self) -> None:
        wizard = self._wizard
        if wizard is None or wizard.state in (WizardState.IDLE, WizardState.COLLECTING):
            return
        if wizard.state is WizardState.SUCCEEDED and wizard.region is not None:
            self._apply_calibration(wizard.region)
        elif wizard.notice:
            self._notify(wizard.notice)
        wizard.acknowledge()

    def _apply_calibration(self, region: ViewRegion) -> None:
        self._processor.apply_calibration(region)
        if self._store is None:
            self._notify("Calibration applied for this session.")
            return
        try:
            self._store.save(region, self._camera_index, self._aspect)
        except (OSError, CalibrationError) as exc:
            logger.warning("could not save calibration: %s", exc)
            self._notify(f"Calibration applied for this session but NOT saved: {exc}")
        else:
            self._notify("Calibration saved. Show an open hand to resume.")

    def _reset_calibration(self) -> None:
        if self._calibrating:
            self._notify("Cancel calibration first (Backspace or X).")
            return
        self._processor.apply_calibration(None)
        if self._store is not None:
            try:
                self._store.delete()
            except OSError as exc:
                logger.warning("could not delete calibration file: %s", exc)
        self._notify("Calibration reset to the default mapping.")

    def _notify(self, text: str) -> None:
        logger.info(text)
        self._message = (text, self._clock() + self._message_seconds)

    # ---------------------------------------------------------------- status
    def _status(self, gesture: GestureFrame) -> StatusInfo:
        if self._stop.is_set():
            control = "STOPPED"
        elif self._calibrating:
            control = "CALIBRATING"
        elif self._pause.paused:
            control = "PAUSED"
        else:
            control = "LIVE" if self._controller.enabled else "DRY-RUN"
        message = None
        if self._message is not None and self._clock() < self._message[1]:
            message = self._message[0]
        region = self._processor.region
        return StatusInfo(
            control,
            self._fps,
            self._hotkey_label,
            self._stop.reason,
            message=message,
            fault=self._controller.fault_message,
            region=(region.left, region.top, region.right, region.bottom) if region else None,
            calibration=self._wizard.view() if self._calibrating and self._wizard else None,
            pause_hotkey=self._pause_hotkey_label,
            diagnostics_lines=self._diagnostic_lines,
        )

    def _update_fps(self, dt: float) -> None:
        if dt > 0:
            instant = 1.0 / dt
            self._fps = instant if self._fps == 0.0 else 0.9 * self._fps + 0.1 * instant

    # -------------------------------------------------------------- shutdown
    def _release_input(self, reason: str) -> None:
        """Halt the state machine and make sure no mouse button is left held."""
        for action in self._processor.halt(reason):
            if isinstance(action, MouseUp):
                self._controller.mouse_up(drag=action.drag)
        self._controller.release_button(reason)

    def _shutdown(self) -> None:
        # Each step is isolated so one failure cannot skip the rest (esp. the button
        # release and the camera).
        steps: list[tuple[str, Callable[[], object]]] = [
            ("mouse button release", lambda: self._release_input("shutdown")),
        ]
        steps += [("hotkey watcher", watcher.close) for watcher in self._watchers]
        steps += [
            ("preview", self._preview.close),
            ("tracker", self._tracker.close),
            ("camera", self._camera.release),
        ]
        for name, step in steps:
            try:
                step()
            except Exception:
                logger.exception("error closing %s", name)
        if self._controller.fault_message:
            logger.error(self._controller.fault_message)
