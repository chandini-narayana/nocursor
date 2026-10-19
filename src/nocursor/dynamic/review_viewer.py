"""Skeleton replay viewer for reviewing candidate activations.

It replays the *recorded landmark sequences* only: a hand skeleton drawn from numbers on a plain
canvas, in the same mirrored view as the preview, with the palm's path as a faint trail. There is
no webcam, no video, no controller and no desktop input anywhere in this module.

What the reviewer sees is deliberately limited, to keep the judgement blind: the candidate's
anonymous id, the gesture the system reported, and the recording and time range. It never shows
which system produced the candidate or how confident it was. The reviewer answers one question,
"was the person actually performing that gesture?":

* ``F`` false: no gesture was intended (the system fired by mistake)
* ``G`` genuine: the person did perform it
* ``A`` ambiguous: cannot tell
* ``U`` un-mark (back to unreviewed), ``S``/Right next, ``P``/Left previous, ``N`` next unreviewed
* Space pause, ``R`` restart the clip, ``Q``/Esc quit

Nothing is ever marked except by one of those keys. Moving between candidates, looping, pausing and
quitting leave unreviewed candidates unreviewed.

``ReviewSession`` holds all the logic and is tested without a display; ``render_frame`` and
``run_viewer`` are the thin OpenCV layer.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from nocursor.dynamic.features import PALM_POINTS, to_view_coordinates
from nocursor.dynamic.review import MARKS, MarkStore, ReviewError
from nocursor.dynamic.schema import SequenceRecord
from nocursor.models import HAND_CONNECTIONS

WINDOW_NAME = "NoCursor - candidate review"
CANVAS = (480, 640)  # height, width
GESTURE_HELP = {
    "swipe_left": "one sweep to your LEFT, then stop",
    "swipe_right": "one sweep to your RIGHT, then stop",
    "circle_clockwise": "one full CLOCKWISE circle (as seen on the mirrored preview)",
}
MARK_KEYS = {"false": "F", "genuine": "G", "ambiguous": "A"}
ACTIONS = ("false", "genuine", "ambiguous", "clear", "next", "prev", "next_unreviewed", "pause",
           "restart", "quit")  # fmt: skip


class ReviewSession:
    """Which candidate is on screen, where playback is, and what has been marked."""

    def __init__(
        self, items: Sequence[dict[str, Any]], store: MarkStore,
        sequences: dict[str, SequenceRecord], *, start_at: int | None = None,
    ) -> None:  # fmt: skip
        if not items:
            raise ReviewError("there are no candidates to review")
        missing = sorted({i["recording_id"] for i in items} - set(sequences))
        if missing:
            raise ReviewError(f"recording(s) not found: {', '.join(missing)}")
        self.items = list(items)
        self.store = store
        self.sequences = sequences
        unreviewed = store.unreviewed()
        if start_at is None:
            start_at = next((n for n, i in enumerate(self.items) if i["item_id"] in unreviewed), 0)
        self.index = start_at
        self.paused = False
        self.finished = False
        self.t = self.item["from_s"]
        self._times: dict[str, np.ndarray] = {}

    # ----------------------------------------------------------------------- navigation
    @property
    def item(self) -> dict[str, Any]:
        return self.items[self.index]

    def _go(self, index: int) -> None:
        self.index = max(0, min(len(self.items) - 1, index))
        self.t = self.item["from_s"]

    def apply(self, action: str) -> str:
        """Carry out one action. Only the three mark actions and ``clear`` change the marks."""
        if action not in ACTIONS:
            raise ReviewError(f"unknown action {action!r}")
        item_id = self.item["item_id"]
        if action in MARKS:
            self.store.set(item_id, action)
            self._advance_after_mark()
        elif action == "clear":
            self.store.clear(item_id)
        elif action == "next":
            self._go(self.index + 1)
        elif action == "prev":
            self._go(self.index - 1)
        elif action == "next_unreviewed":
            self._go_to_next_unreviewed()
        elif action == "pause":
            self.paused = not self.paused
        elif action == "restart":
            self.t = self.item["from_s"]
        elif action == "quit":
            self.finished = True
        return action

    def _go_to_next_unreviewed(self) -> None:
        pending = set(self.store.unreviewed())
        order = [*range(self.index + 1, len(self.items)), *range(0, self.index)]
        target = next((n for n in order if self.items[n]["item_id"] in pending), None)
        if target is not None:
            self._go(target)

    def _advance_after_mark(self) -> None:
        self._go_to_next_unreviewed()

    # ------------------------------------------------------------------------- playback
    def tick(self, dt: float) -> None:
        """Advance playback; the clip loops. Looping never marks anything."""
        if self.paused:
            return
        start, end = self.item["from_s"], self.item["to_s"]
        self.t += dt
        if self.t > end:
            self.t = start + (self.t - end) % max(end - start, 1e-6)

    def _frame_times(self, recording_id: str) -> np.ndarray:
        if recording_id not in self._times:
            seq = self.sequences[recording_id]
            self._times[recording_id] = (seq.t_ms - seq.t_ms[0]) / 1000.0
        return self._times[recording_id]

    def hand_at(self, t: float) -> np.ndarray | None:
        """The 21 landmarks in mirrored view coordinates at time ``t``; None if not tracked."""
        seq = self.sequences[self.item["recording_id"]]
        index = int(np.searchsorted(self._frame_times(self.item["recording_id"]), t,
                                    side="right")) - 1  # fmt: skip
        if index < 0 or not seq.usable[index]:
            return None
        return to_view_coordinates(seq.landmarks[index], seq.aspect_ratio)

    def trail(self) -> np.ndarray:
        """Palm-centre positions (view coordinates) of the trusted frames in the clip."""
        recording_id = self.item["recording_id"]
        seq = self.sequences[recording_id]
        times = self._frame_times(recording_id)
        keep = np.flatnonzero((times >= self.item["from_s"]) & (times <= self.item["to_s"])
                              & seq.usable)  # fmt: skip
        if keep.size == 0:
            return np.zeros((0, 2))
        view = to_view_coordinates(seq.landmarks[keep], seq.aspect_ratio)
        return view[:, list(PALM_POINTS), :].mean(axis=1)

    # ------------------------------------------------------------------------ the screen
    def hud_lines(self) -> list[str]:
        """Text shown to the reviewer. It carries no system name and no confidence."""
        item = self.item
        reviewed, total = self.store.progress()
        mark = self.store.mark_of(item["item_id"])
        predicted = item["predicted"]
        return [
            f"Candidate {item['item_id']}  ({self.index + 1} of {len(self.items)})",
            f"The system reported: {predicted}",
            f"Meaning: {GESTURE_HELP[predicted]}",
            f"Recording {item['recording_id']}, {item['from_s']:.1f} to {item['to_s']:.1f} s "
            f"(now {self.t:.1f} s)",
            f"Your mark: {mark or 'NOT REVIEWED'}",
            f"Reviewed {reviewed} of {total}",
            "Was the person actually performing this gesture?",
            "F false   G genuine   A ambiguous   U un-mark",
            "Space pause  R restart  Left/Right or P/S move  N next unreviewed  Q quit",
        ]


# ------------------------------------------------------------------------------ drawing
def _to_pixels(points: np.ndarray, aspect: float, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    return np.stack([points[..., 0] / aspect * (width - 1), points[..., 1] * (height - 1)],
                    axis=-1)  # fmt: skip


def render_frame(session: ReviewSession, size: tuple[int, int] = CANVAS) -> np.ndarray:
    """One BGR image of the current moment. Needs OpenCV (only for drawing)."""
    try:
        import cv2
    except ImportError as exc:  # the lean ML environment has no OpenCV
        raise ReviewError("the review viewer needs OpenCV for drawing; use the main "
                          "environment") from exc  # fmt: skip
    height, width = size
    image = np.full((height, width, 3), 24, dtype=np.uint8)
    seq = session.sequences[session.item["recording_id"]]
    aspect = seq.aspect_ratio

    trail = session.trail()
    if len(trail) > 1:
        pixels = _to_pixels(trail, aspect, size).astype(np.int32)
        cv2.polylines(image, [pixels.reshape(-1, 1, 2)], False, (90, 90, 90), 1, cv2.LINE_AA)
    hand = session.hand_at(session.t)
    if hand is None:
        cv2.putText(image, "no hand tracked", (width // 2 - 80, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 200), 1, cv2.LINE_AA)
    else:
        pixels = _to_pixels(hand, aspect, size).astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, tuple(pixels[a]), tuple(pixels[b]), (200, 200, 200), 2, cv2.LINE_AA)
        for point in pixels:
            cv2.circle(image, tuple(point), 3, (60, 200, 255), -1, cv2.LINE_AA)

    start, end = session.item["from_s"], session.item["to_s"]
    fraction = (session.t - start) / max(end - start, 1e-6)
    cv2.rectangle(image, (10, height - 14), (width - 10, height - 8), (60, 60, 60), -1)
    cv2.rectangle(image, (10, height - 14), (10 + int((width - 20) * fraction), height - 8),
                  (60, 200, 255), -1)  # fmt: skip

    mark = session.store.mark_of(session.item["item_id"])
    colour = {"false": (80, 80, 230), "genuine": (80, 200, 80), "ambiguous": (60, 200, 230)}.get(
        mark or "", (200, 200, 200))  # fmt: skip
    for n, line in enumerate(session.hud_lines()):
        cv2.putText(image, line, (12, 22 + 20 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colour if line.startswith("Your mark") else (230, 230, 230), 1, cv2.LINE_AA)
    return image


_KEY_ACTIONS: dict[int, str] = {
    ord("f"): "false", ord("F"): "false", ord("g"): "genuine", ord("G"): "genuine",
    ord("a"): "ambiguous", ord("A"): "ambiguous", ord("u"): "clear", ord("U"): "clear",
    ord("s"): "next", ord("S"): "next", 2555904: "next", ord("p"): "prev", ord("P"): "prev",
    2424832: "prev", ord("n"): "next_unreviewed", ord("N"): "next_unreviewed",
    32: "pause", ord("r"): "restart", ord("R"): "restart", 27: "quit", ord("q"): "quit",
    ord("Q"): "quit",
}  # fmt: skip


def key_to_action(code: int) -> str | None:
    return _KEY_ACTIONS.get(code)


def run_viewer(
    session: ReviewSession, *, fps: float = 30.0,
    clock: Callable[[], float] = time.perf_counter,
) -> None:  # fmt: skip
    """The interactive loop; Q, Esc or closing the window ends it. Marks are saved as made."""
    try:
        import cv2
    except ImportError as exc:
        raise ReviewError("the review viewer needs OpenCV; use the main environment") from exc
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    last = clock()
    try:
        while not session.finished:
            cv2.imshow(WINDOW_NAME, render_frame(session))
            code = cv2.waitKeyEx(max(1, int(1000 / fps)))
            action = key_to_action(code) if code != -1 else None
            if action:
                session.apply(action)
            now = clock()
            session.tick(now - last)
            last = now
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
