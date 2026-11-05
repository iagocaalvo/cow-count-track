"""Directional line-crossing counter for tracked objects, with occlusion handover."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

Point = tuple[float, float]


@dataclass(frozen=True)
class CrossingEvent:
    frame: int
    track_id: int
    direction: str  # "IN" or "OUT"
    handover: bool = False  # True if the crossing happened while the cow was hidden


@dataclass
class _Track:
    side: int                       # confirmed side of the line (+1 / -1)
    x: float                        # last anchor position
    y: float
    width: float                    # last box width (used as a size-relative distance scale)
    last_frame: int
    vx: float = 0.0                 # smoothed velocity in px/frame (for predicting hidden cows)
    vy: float = 0.0
    pending: int = 0                # frames seen on the opposite side, not yet confirmed
    counted: set[str] = field(default_factory=set)
    handover: tuple[_Track, CrossingEvent] | None = None  # set on a lost track whose crossing was
                                                           # credited to a new track (see update)


class LineCounter:
    """Counts tracks whose anchor point crosses the segment p1 -> p2.

    - Side of a point: sign of the cross product (p2 - p1) x (pt - p1); positive side = IN.
    - A crossing is confirmed only after `min_frames` frames on the new side (filters jitter).
    - Each track ID is counted at most once per direction.
    - Occlusion handover: if a cow is lost close to the line (e.g. hidden behind another cow) and
      a NEW track appears on the other side, near where the lost cow was predicted to be, within
      `handover_frames`, the cow crossed while hidden: the crossing is counted once.
      If ByteTrack later re-finds the old ID on its ORIGINAL side, the guess was wrong and the
      handover count is undone.
    """

    MIN_LOST_FRAMES = 5  # brief detection flickers are not occlusions

    def __init__(self, p1: Point, p2: Point, min_frames: int = 3, handover_frames: int = 90):
        if tuple(p1) == tuple(p2):
            raise ValueError("Line points must be different.")
        self.p1, self.p2 = p1, p2
        self.min_frames = max(1, min_frames)
        self.handover_frames = handover_frames
        self._dx, self._dy = p2[0] - p1[0], p2[1] - p1[1]
        self._len = math.hypot(self._dx, self._dy)
        self.counts = {"IN": 0, "OUT": 0}
        self.events: list[CrossingEvent] = []
        self._tracks: dict[int, _Track] = {}

    @property
    def in_count(self) -> int:
        return self.counts["IN"]

    @property
    def out_count(self) -> int:
        return self.counts["OUT"]

    @property
    def total(self) -> int:
        return self.in_count + self.out_count

    # ----------------------------------------------------------------------------- geometry
    def _cross(self, x: float, y: float) -> float:
        return self._dx * (y - self.p1[1]) - self._dy * (x - self.p1[0])

    def side_of(self, x: float, y: float) -> int:
        """+1 / -1 depending on the side, 0 if on the line or beyond the segment's ends."""
        # Projection onto the segment: only points "in front of" the segment can cross it.
        t = (self._dx * (x - self.p1[0]) + self._dy * (y - self.p1[1])) / self._len ** 2
        if not 0.0 <= t <= 1.0:
            return 0
        c = self._cross(x, y)
        return (c > 0) - (c < 0)

    def _dist_to_line(self, x: float, y: float) -> float:
        return abs(self._cross(x, y)) / self._len

    # ----------------------------------------------------------------------------- counting
    def _count(self, frame_idx: int, tid: int, direction: str, handover: bool = False) -> CrossingEvent:
        self.counts[direction] += 1
        event = CrossingEvent(frame_idx, tid, direction, handover)
        self.events.append(event)
        return event

    def _find_hidden_crosser(self, frame_idx: int, x: float, y: float, side: int) -> _Track | None:
        """Lost track that was near the line on the other side and is predicted to be at (x, y)."""
        best, best_d = None, math.inf
        for tr in self._tracks.values():
            dt = frame_idx - tr.last_frame
            if tr.handover or tr.side != -side or not self.MIN_LOST_FRAMES <= dt <= self.handover_frames:
                continue
            if self._dist_to_line(tr.x, tr.y) > tr.width:  # was not about to cross
                continue
            # Constant-velocity prediction of where the hidden cow should be now. The new track must
            # be within half a body width of it: looser values matched re-detected cows that had
            # crossed earlier (false counts in video 7).
            d = math.hypot(tr.x + tr.vx * dt - x, tr.y + tr.vy * dt - y)
            if d < 0.5 * tr.width and d < best_d:
                best, best_d = tr, d
        return best

    def update(self, frame_idx: int, track_ids, anchors, widths=None) -> list[CrossingEvent]:
        """Feed this frame's track IDs, anchor points and box widths; returns new crossings."""
        new_events = []
        if widths is None:
            widths = [0.0] * len(track_ids)
        for tid, (x, y), w in zip(map(int, track_ids), anchors, widths):
            x, y, w = float(x), float(y), float(w)
            s = self.side_of(x, y)
            if s == 0:
                continue
            tr = self._tracks.get(tid)

            # --- New track: record its side, or inherit a crossing hidden by an occlusion.
            if tr is None:
                tr = self._tracks[tid] = _Track(side=s, x=x, y=y, width=w, last_frame=frame_idx)
                lost = self._find_hidden_crosser(frame_idx, x, y, s)
                direction = "IN" if s > 0 else "OUT"
                if lost is not None and direction not in lost.counted:
                    # The hidden cow crossed: count it once and mark both IDs so neither counts
                    # this direction again (ByteTrack may later re-find the old ID on this side).
                    event = self._count(frame_idx, tid, direction, handover=True)
                    new_events.append(event)
                    lost.counted.add(direction)
                    tr.counted.add(direction)
                    lost.handover = (tr, event)
                continue

            # --- Old ID re-found after a handover: confirm it, or undo it if it never crossed.
            if tr.handover is not None:
                new_tr, event = tr.handover
                tr.handover = None
                if s == tr.side:  # still on its original side -> the new track was another cow
                    self.counts[event.direction] -= 1
                    self.events.remove(event)
                    tr.counted.discard(event.direction)
                    new_tr.counted.discard(event.direction)
                    print(f"  frame {frame_idx:5d}  undo occlusion handover of cow #{event.track_id}")
                else:
                    tr.side, tr.pending = s, 0

            # --- Known track: update its motion model (exponentially smoothed velocity).
            dt = max(1, frame_idx - tr.last_frame)
            tr.vx = 0.8 * tr.vx + 0.2 * (x - tr.x) / dt
            tr.vy = 0.8 * tr.vy + 0.2 * (y - tr.y) / dt
            tr.x, tr.y, tr.width, tr.last_frame = x, y, w, frame_idx

            if s == tr.side:
                tr.pending = 0  # back on its side -> cancel any pending crossing
                continue

            # --- On the opposite side: wait `min_frames` frames before confirming.
            tr.pending += 1
            if tr.pending < self.min_frames:
                continue
            tr.side, tr.pending = s, 0
            direction = "IN" if s > 0 else "OUT"
            if direction not in tr.counted:
                tr.counted.add(direction)
                new_events.append(self._count(frame_idx, tid, direction))
        return new_events
