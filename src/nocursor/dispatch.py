"""Applying a frame's gesture actions to the safety-gated controller.

This is the one place that turns ``Action`` objects into controller calls, shared by the
live loop and the benchmarks so that what is measured is what runs.
"""

from __future__ import annotations

from collections.abc import Iterable

from nocursor.controller import SafeController
from nocursor.interaction import Action, MouseDown, MouseUp, MoveTo, RightClick, Scroll
from nocursor.processor import GestureProcessor

EVENT_PRESS = "press"
EVENT_CLICK = "click"
EVENT_DRAG_END = "drag_end"
EVENT_RIGHT_CLICK = "right_click"


def apply_actions(
    controller: SafeController, processor: GestureProcessor, actions: Iterable[Action]
) -> tuple[str, ...]:
    """Execute ``actions`` through the controller; return the discrete events it accepted.

    An event is reported only if the controller actually sent it (or, in dry-run, simulated
    it), never merely because the gesture logic asked for it. That is what lets the UI say
    "pressed" only after a press really happened.
    """
    events: list[str] = []
    for action in actions:
        before = controller.events_accepted
        if isinstance(action, MoveTo):
            controller.move_to(action.x, action.y)
        elif isinstance(action, MouseDown):
            controller.mouse_down()
            if not controller.button_down:  # refused: never pretend it is pressed
                processor.abort_interaction()
            elif controller.events_accepted > before:
                events.append(EVENT_PRESS)
        elif isinstance(action, MouseUp):
            controller.mouse_up(drag=action.drag)
            if controller.events_accepted > before:
                events.append(EVENT_DRAG_END if action.drag else EVENT_CLICK)
        elif isinstance(action, Scroll):
            controller.scroll(action.clicks)
            if controller.events_accepted > before:
                events.append(f"scroll {action.clicks:+d}")
        elif isinstance(action, RightClick):
            controller.right_click()
            if controller.events_accepted > before:
                events.append(EVENT_RIGHT_CLICK)
    return tuple(events)
