"""The collection tool's OpenCV window: a mirrored preview, the hand skeleton and the HUD text."""

from __future__ import annotations

import numpy as np

from nocursor.dynamic.collect import CollectCommand, CollectorHud
from nocursor.hud import Tone
from nocursor.models import HAND_CONNECTIONS, HandObservation

WINDOW_NAME = "NoCursor - gesture data collection"
_QUIT_KEYS = {27, ord("q"), ord("Q")}  # Esc, q

_COMMAND_KEYS: dict[int, CollectCommand] = {
    32: CollectCommand.START,  # Space
    13: CollectCommand.START,  # Enter
    8: CollectCommand.UNDO,  # Backspace
    ord("n"): CollectCommand.SKIP,
    ord("N"): CollectCommand.SKIP,
    ord("t"): CollectCommand.SUBTYPE,
    ord("T"): CollectCommand.SUBTYPE,
    ord("1"): CollectCommand.LABEL_1,
    ord("2"): CollectCommand.LABEL_2,
    ord("3"): CollectCommand.LABEL_3,
    ord("4"): CollectCommand.LABEL_4,
}

_TONE_COLORS: dict[Tone, tuple[int, int, int]] = {  # BGR
    Tone.OK: (0, 220, 0),
    Tone.INFO: (255, 255, 255),
    Tone.PENDING: (0, 165, 255),
    Tone.ACTIVE: (60, 60, 255),
    Tone.SCROLL: (255, 255, 0),
    Tone.WARN: (0, 215, 255),
    Tone.ERROR: (60, 60, 255),
}


def wrap_text(text: str, width: int) -> list[str]:
    """Break ``text`` into lines of at most ``width`` characters (on spaces)."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_overlay(view: np.ndarray, observation: HandObservation | None, hud: CollectorHud) -> None:
    """Draw the skeleton and HUD onto an already mirrored image, in place.

    Status goes in a compact bar at the top and the instructions at the bottom, so the middle of
    the frame, where the hand is while a gesture is performed, stays clear.
    """
    import cv2

    height, width = view.shape[:2]
    if observation is not None:
        pts = [(int((1.0 - x) * width), int(y * height)) for x, y, _ in observation.landmarks]
        color = (60, 60, 255) if hud.recording else (0, 220, 0)
        for a, b in HAND_CONNECTIONS:
            cv2.line(view, pts[a], pts[b], color, 2)
        for p in pts:
            cv2.circle(view, p, 3, (255, 255, 255), -1)

    def darken(top: int, bottom: int) -> None:
        roi = view[max(0, top):min(height, bottom), :]
        if roi.size:
            view[max(0, top):min(height, bottom), :] = cv2.addWeighted(
                roi, 0.4, np.zeros_like(roi), 0.6, 0
            )

    def put(text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
        # Outline from four offset copies at the same thickness as the text: a thicker outline is
        # drawn wider than the text itself and leaves dark tails past the end of each line.
        x, y = origin
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            cv2.putText(view, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (0, 0, 0), 1)  # fmt: skip
        cv2.putText(view, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

    chars = max(30, width // 9)
    top_bottom = 36 + 19 * len(hud.lines) + 4
    darken(0, top_bottom)
    put(hud.headline, (10, 26), 0.7, _TONE_COLORS[hud.tone])
    y = 52
    for text, tone in hud.lines:
        put(text, (10, y), 0.45, _TONE_COLORS[tone])
        y += 19

    instruction_rows = [row for text in hud.instructions for row in wrap_text(text, chars)]
    bottom_top = height - 30 - 19 * (len(instruction_rows) + 1) - 6
    darken(bottom_top, height)
    y = bottom_top + 20
    for row in instruction_rows:
        put(row, (10, y), 0.5, (0, 255, 255))
        y += 19
    put(hud.privacy, (10, y), 0.45, (255, 255, 255))
    put(hud.footer, (10, height - 9), 0.4, (200, 200, 200))

    if hud.big_text:
        put(hud.big_text, (width // 2 - 25, height // 2 + 30), 3.0, (0, 165, 255))
    if hud.progress is not None:
        cv2.circle(view, (width - 24, 24), 9, (60, 60, 255), -1)  # the REC dot
        cv2.rectangle(view, (10, top_bottom + 4), (width - 10, top_bottom + 12), (255, 255, 255), 1)
        cv2.rectangle(
            view, (10, top_bottom + 4), (10 + int((width - 20) * hud.progress), top_bottom + 12),
            (60, 60, 255), -1,
        )  # fmt: skip


class OpenCVCollectorView:
    """Mirrored window. Esc/Q or closing the window ends the session."""

    def __init__(self) -> None:
        import cv2

        self._cv2 = cv2
        self._opened = False
        self._commands: list[CollectCommand] = []

    def show(
        self, frame_bgr: np.ndarray, observation: HandObservation | None, hud: CollectorHud
    ) -> bool:
        cv2 = self._cv2
        view = cv2.flip(frame_bgr, 1)  # mirrored, so on-screen motion matches the user's
        draw_overlay(view, observation, hud)
        cv2.imshow(WINDOW_NAME, view)
        self._opened = True
        key = cv2.waitKey(1) & 0xFF
        if key in _QUIT_KEYS:
            return False
        command = _COMMAND_KEYS.get(key)
        if command is not None:
            self._commands.append(command)
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1

    def take_commands(self) -> tuple[CollectCommand, ...]:
        commands, self._commands = tuple(self._commands), []
        return commands

    def close(self) -> None:
        if self._opened:
            self._cv2.destroyWindow(WINDOW_NAME)
            self._cv2.waitKey(1)
            self._opened = False
