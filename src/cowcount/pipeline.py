"""Main loop: YOLO detection -> ByteTrack tracking -> line counting -> display + save."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from .counter import LineCounter
from .draw import draw_counts, draw_line

COW_CLASS_ID = 19  # "cow" in the COCO dataset
ANCHORS = {"bottom": sv.Position.BOTTOM_CENTER, "center": sv.Position.CENTER}
DEFAULT_TRACKER = Path(__file__).resolve().parents[2] / "configs" / "bytetrack_cows.yaml"
DISPLAY_WINDOW = "Cow tracking + counting (q to stop)"
MAX_DISPLAY_WIDTH = 1280


class DisplayIds:
    """Maps raw ByteTrack IDs to clean sequential IDs (1, 2, 3...).

    ByteTrack consumes an internal ID for every tentative track, even those discarded before being
    confirmed, so raw IDs grow much faster than the real number of animals. A track only gets a
    display ID once seen for `min_frames` frames; shorter tracks (partial cows at the border,
    fragments during occlusions) get -1 and are hidden and never counted.
    """

    def __init__(self, min_frames: int):
        self.min_frames = min_frames
        self._age: dict[int, int] = {}
        self._map: dict[int, int] = {}

    def __call__(self, raw_ids: np.ndarray) -> np.ndarray:
        out = np.full(len(raw_ids), -1, dtype=int)
        for i, rid in enumerate(map(int, raw_ids)):
            self._age[rid] = age = self._age.get(rid, 0) + 1
            if rid not in self._map and age >= self.min_frames:
                self._map[rid] = len(self._map) + 1
            out[i] = self._map.get(rid, -1)
        return out


def run(
    video_path: str | Path,
    line: tuple[tuple[int, int], tuple[int, int]],
    output_path: str | Path,
    model_name: str = "yolo11s.pt",
    tracker_cfg: str | Path = DEFAULT_TRACKER,
    conf: float = 0.1,
    nms_iou: float = 0.7,
    imgsz: int = 1280,
    device: str | None = None,
    anchor: str = "center",
    min_frames: int = 3,
    min_track_frames: int = 15,
    handover_frames: int = 90,
    show: bool = True,
) -> LineCounter:
    """Process the whole video and return the LineCounter with the final counts and events."""
    # --- Input video and output writer ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    # --- Model, tracker settings and counter ---
    model = YOLO(model_name)
    # Never upscale small videos: inference size is capped at the frame's longest side
    # (rounded up to a multiple of 32, as YOLO requires).
    imgsz = min(imgsz, -(-max(w, h) // 32) * 32)
    # nms_iou: boxes overlapping a stronger box by more than this IoU are removed as duplicates.
    # Higher values keep more boxes for overlapping cows, but also more duplicate boxes.
    track_kwargs = dict(persist=True, tracker=str(tracker_cfg), classes=[COW_CLASS_ID], conf=conf,
                        iou=nms_iou, imgsz=imgsz, device=device, verbose=False)
    counter = LineCounter(*line, min_frames=min_frames, handover_frames=handover_frames)
    display_ids = DisplayIds(min_track_frames)
    anchor_pos = ANCHORS[anchor]

    # --- Annotators (sizes scale with the video resolution) ---
    thick = max(2, w // 640)
    lookup = sv.ColorLookup.TRACK
    box_ann = sv.BoxAnnotator(thickness=thick, color_lookup=lookup)
    label_ann = sv.LabelAnnotator(text_scale=0.25 * thick, text_thickness=1, text_padding=3 * thick // 2,
                                  color_lookup=lookup)
    trace_ann = sv.TraceAnnotator(thickness=thick, trace_length=int(fps * 2), position=anchor_pos,
                                  color_lookup=lookup)
    disp_scale = min(1.0, MAX_DISPLAY_WIDTH / w)

    frame_idx, t0 = 0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 1) Detect cows and update ByteTrack (Kalman prediction + 2-stage association).
            result = model.track(frame, **track_kwargs)[0]
            dets = sv.Detections.from_ultralytics(result)

            # 2) Keep only confirmed tracks, relabelled with sequential display IDs.
            if dets.tracker_id is None:  # no active tracks in this frame
                dets = sv.Detections.empty()
            else:
                dets.tracker_id = display_ids(dets.tracker_id)
                dets = dets[dets.tracker_id > 0]

            # 3) Count line crossings using each track's anchor point (box width scales the
            #    distances used to match a cow that crossed while hidden).
            if len(dets):
                anchors = dets.get_anchors_coordinates(anchor_pos)
                widths = dets.xyxy[:, 2] - dets.xyxy[:, 0]
                for ev in counter.update(frame_idx, dets.tracker_id, anchors, widths):
                    note = "  (crossed while occluded)" if ev.handover else ""
                    print(f"  frame {ev.frame:5d} ({ev.frame / fps:6.2f}s)  cow #{ev.track_id} -> {ev.direction}{note}")

            # 4) Draw on the frame in place (it is not needed afterwards) and save it.
            labels = [f"#{tid} {c:.2f}" for tid, c in zip(dets.tracker_id, dets.confidence)] if len(dets) else []
            trace_ann.annotate(frame, dets)
            box_ann.annotate(frame, dets)
            label_ann.annotate(frame, dets, labels=labels)
            draw_line(frame, counter.p1, counter.p2, thick)
            draw_counts(frame, {**counter.counts, "TOTAL": counter.total}, thick)
            writer.write(frame)

            # 5) Live preview; 'q' or closing the window stops early (video saved so far is kept).
            if show:
                cv2.imshow(DISPLAY_WINDOW, cv2.resize(frame, None, fx=disp_scale, fy=disp_scale)
                           if disp_scale < 1 else frame)
                if (cv2.waitKey(1) & 0xFF == ord("q")
                        or cv2.getWindowProperty(DISPLAY_WINDOW, cv2.WND_PROP_VISIBLE) < 1):
                    print("Stopped by user.")
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  processed {frame_idx}/{n_frames} frames ({frame_idx / (time.time() - t0):.1f} FPS)")
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    return counter
