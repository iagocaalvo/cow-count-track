"""Drawing helpers shared by the line picker and the video output."""
from __future__ import annotations

import cv2
import numpy as np

RED, GREEN, WHITE, BLACK = (0, 0, 255), (0, 255, 0), (255, 255, 255), (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_line(img: np.ndarray, p1, p2, thick: int = 2) -> None:
    """Counting line plus a green arrow pointing to its IN (positive) side."""
    cv2.line(img, p1, p2, RED, thick + 1)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    norm = max(1e-6, float(np.hypot(dx, dy)))
    nx, ny = -dy / norm, dx / norm  # unit normal towards the positive side (see LineCounter.side_of)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    length = 30 * thick
    tip = (int(mid[0] + nx * length), int(mid[1] + ny * length))
    cv2.arrowedLine(img, mid, tip, GREEN, thick, tipLength=0.35)
    # Label placed just beyond the arrow tip so it does not cover the arrow.
    label = (int(tip[0] + nx * 10 * thick) - 10 * thick, int(tip[1] + ny * 10 * thick) + 5 * thick)
    cv2.putText(img, "IN", label, FONT, 0.4 * thick, GREEN, max(1, thick // 2 + 1))


def draw_text(img: np.ndarray, text: str, org, scale: float = 0.8, thick: int = 2) -> None:
    """White text with a black outline, readable on any background."""
    cv2.putText(img, text, org, FONT, scale, BLACK, thick + 2)
    cv2.putText(img, text, org, FONT, scale, WHITE, thick)


def draw_counts(img: np.ndarray, counts: dict[str, int], thick: int = 2) -> None:
    """IN / OUT / TOTAL panel in the top-left corner."""
    scale, line_h = 0.4 * thick, 13 * thick
    rows = [f"{k}: {v}" for k, v in counts.items()]
    cv2.rectangle(img, (10, 10), (10 + 80 * thick, 15 + line_h * len(rows) + 2 * thick), BLACK, -1)
    for i, text in enumerate(rows, start=1):
        cv2.putText(img, text, (10 + 4 * thick, 10 + line_h * i), FONT, scale, WHITE, max(1, thick - 1))
