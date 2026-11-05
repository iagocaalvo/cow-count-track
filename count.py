#!/usr/bin/env python
"""Count cows crossing a user-defined line using YOLO + ByteTrack.

Usage:
    python count.py --video vacas-uruguay/Videos/6.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cowcount.line_picker import pick_line  # noqa: E402
from cowcount.pipeline import run  # noqa: E402


def default_device() -> str:
    """Prefer NVIDIA GPU, then Apple GPU (Ultralytics does not pick MPS on its own), else CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Path to the input video")
    ap.add_argument("--model", default="yolo11s.pt", help="YOLO weights (yolo11n/s/m/l/x.pt, auto-downloaded)")
    ap.add_argument("--tracker", default=str(ROOT / "configs" / "bytetrack_cows.yaml"), help="ByteTrack config")
    ap.add_argument("--conf", type=float, default=0.1, help="Min detection confidence passed to the tracker")
    ap.add_argument("--nms-iou", type=float, default=0.7,
                    help="NMS overlap threshold: higher keeps boxes of overlapping cows (and more duplicates)")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    ap.add_argument("--anchor", choices=["center", "bottom"], default="center",
                    help="Box point used for the trail and the crossing test (same point the Kalman filter tracks)")
    ap.add_argument("--min-frames", type=int, default=3,
                    help="Frames an animal must stay on the new side before counting")
    ap.add_argument("--min-track-frames", type=int, default=15,
                    help="Frames a track must exist before it gets an ID and can be counted")
    ap.add_argument("--handover-frames", type=int, default=90,
                    help="Max frames a cow can stay hidden (occluded) near the line and still be counted "
                         "if it reappears on the other side with a new ID (0 disables)")
    ap.add_argument("--device", default=None, help="cpu / mps / cuda (auto if omitted)")
    ap.add_argument("--output-dir", default=str(ROOT / "results"), help="Where the annotated video is saved")
    args = ap.parse_args()

    # 1) Read the first frame and let the user define the line and its IN side.
    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Cannot read first frame of {video}")

    print("On the first frame: click 2 points for the line, then a 3rd point on the IN side...")
    line = pick_line(first)
    print(f"Line: {line[0]} -> {line[1]}")

    # 2) Track, count and save the annotated video.
    output = Path(args.output_dir) / f"{video.stem}_tracked.mp4"
    device = args.device or default_device()
    print(f"Processing {video} on {device} with {args.model} ...")
    counter = run(video, line, output, model_name=args.model, tracker_cfg=args.tracker, conf=args.conf, nms_iou=args.nms_iou,
                  imgsz=args.imgsz, device=device, anchor=args.anchor, min_frames=args.min_frames,
                  min_track_frames=args.min_track_frames, handover_frames=args.handover_frames)

    print(f"\n=== Result ===\nIN:    {counter.in_count}\nOUT:   {counter.out_count}\nTOTAL: {counter.total}")
    print(f"Annotated video saved to: {output}")


if __name__ == "__main__":
    main()
