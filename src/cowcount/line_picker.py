"""Interactive selection of the counting line (and its IN side) on the first video frame."""
from __future__ import annotations

import cv2
import numpy as np

from .draw import GREEN, RED, draw_line, draw_text

Point = tuple[int, int]
WINDOW = "Counting line: 2 clicks = line, 3rd click = IN side | ENTER confirm | r reset | ESC quit"
MAX_DISPLAY_WIDTH = 1280


def _cross(p1: Point, p2: Point, pt: Point) -> int:
    """> 0 if pt is on the positive side of p1 -> p2 (the side LineCounter counts as IN)."""
    return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])


def _orient(p1: Point, p2: Point, in_pt: Point) -> tuple[Point, Point]:
    """Order the endpoints so the 3rd clicked point lies on the positive (IN) side."""
    return (p1, p2) if _cross(p1, p2, in_pt) > 0 else (p2, p1)


def _window_closed() -> bool:
    return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1


def pick_line(frame: np.ndarray) -> tuple[Point, Point]:
    """Let the user click 2 line points and then a 3rd point on the IN side.

    Returns (p1, p2) in original pixel coordinates, ordered so the chosen side is IN.
    """
    # Large frames are shown downscaled; clicks are mapped back with `scale`.
    h, w = frame.shape[:2]
    scale = min(1.0, MAX_DISPLAY_WIDTH / w)
    display = cv2.resize(frame, (round(w * scale), round(h * scale))) if scale < 1 else frame
    points: list[Point] = []

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(points) == 0 or (len(points) == 1 and (x, y) != points[0]):
            points.append((x, y))
        elif len(points) == 2 and _cross(*points, (x, y)) != 0:  # a click on the line picks no side
            points.append((x, y))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)

    while True:
        # Redraw the current state of the selection.
        canvas = display.copy()
        for p in points[:2]:
            cv2.circle(canvas, p, 6, RED, -1)
        if len(points) == 2:
            cv2.line(canvas, points[0], points[1], RED, 2)
        if len(points) == 3:
            cv2.circle(canvas, points[2], 6, GREEN, -1)
            draw_line(canvas, *_orient(*points))
        msg = ("ENTER to confirm, r to reset", "Click on the side that counts as IN",
               "Click line point 2 of 2", "Click line point 1 of 2")[3 - len(points)]
        draw_text(canvas, msg, (10, 30))
        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key in (13, 10) and len(points) == 3:  # Enter
            break
        elif key == 27 or _window_closed():  # Esc or window closed with the mouse
            cv2.destroyAllWindows()
            raise SystemExit("Line selection cancelled.")

    cv2.destroyWindow(WINDOW)
    cv2.waitKey(1)
    p1, p2 = _orient(*points)
    return tuple(round(c / scale) for c in p1), tuple(round(c / scale) for c in p2)
