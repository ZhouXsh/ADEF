#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FID evaluation script that accepts VIDEO FILES as input (not just frame folders).

This script wraps evaluate_fid.py: when given a video path, it extracts frames into
a temporary directory using OpenCV, computes the FID, then cleans up.

Supported input types for --path1 / --path2:
  * a video file (.mp4 .avi .mov .mkv .webm ...) — frames are extracted on the fly
  * a directory of frames — used directly as before
  * a precomputed .npz stats file — used directly as before

Usage examples:
    # Two videos
    python evaluate_fid_video.py --path1 real.mp4 --path2 fake.mp4

    # Video vs frame directory
    python evaluate_fid_video.py --path1 real.mp4 --path2 /frames/fake

    # Video vs cached stats
    python evaluate_fid_video.py --path1 real.mp4 --path2 real_stats.npz

    # Custom frame sampling (every 5th frame, resize to 256x256)
    python evaluate_fid_video.py --path1 a.mp4 --path2 b.mp4 \\
        --frame-stride 5 --resize 256

    # Limit max frames per video
    python evaluate_fid_video.py --path1 a.mp4 --path2 b.mp4 --max-frames 500
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

import torch

from pytorch_fid.fid_score import calculate_fid_given_paths
from pytorch_fid.inception import InceptionV3


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg"}


def is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def extract_video_frames(
    video_path: str,
    out_dir: str,
    frame_stride: int = 1,
    max_frames: int | None = None,
    resize: int | None = None,
) -> int:
    """Extract frames from a video into out_dir using OpenCV.

    Args:
        video_path: source video file
        out_dir:    directory where PNG frames will be written
        frame_stride: keep every Nth frame (default 1 = keep all)
        max_frames: cap on number of frames extracted (None = no cap)
        resize: if set, resize shorter side to this value, keeping aspect ratio
                (then center-crop to a square of this size). None = keep original.

    Returns:
        Number of frames written.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  [video] {video_path}")
    print(f"  [video]   {w}x{h}, {fps:.2f} fps, {n_frames_video} frames total")

    os.makedirs(out_dir, exist_ok=True)
    written = 0
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_stride == 0:
            if resize is not None:
                frame = resize_and_crop(frame, resize)
            out_path = os.path.join(out_dir, f"frame_{saved:06d}.png")
            cv2.imwrite(out_path, frame)
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        idx += 1
    cap.release()

    kept = saved
    expected_kept = n_frames_video // frame_stride
    if max_frames is not None:
        expected_kept = min(expected_kept, max_frames)
    print(f"  [video]   extracted {kept} frames (stride={frame_stride}, "
          f"expected~{expected_kept})")
    return kept


def resize_and_crop(img: "np.ndarray", size: int) -> "np.ndarray":
    """Resize shorter side to `size`, then center-crop to (size, size)."""
    import numpy as np
    h, w = img.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # center crop
    x0 = (new_w - size) // 2
    y0 = (new_h - size) // 2
    img = img[y0:y0 + size, x0:x0 + size]
    return img


class TempFrameDir:
    """Context manager that creates a temp dir and always cleans it up."""

    def __init__(self, keep: bool = False):
        self.keep = keep
        self.path: str | None = None

    def __enter__(self) -> str:
        self.path = tempfile.mkdtemp(prefix="fid_video_frames_")
        return self.path

    def __exit__(self, exc_type, exc, tb):
        if self.path and os.path.isdir(self.path) and not self.keep:
            shutil.rmtree(self.path, ignore_errors=True)
        return False


def resolve_input(
    path: str,
    frame_stride: int,
    max_frames: int | None,
    resize: int | None,
    temp_dirs: list,
) -> str:
    """If path is a video, extract frames into a temp dir and return that dir.
    Otherwise return the path unchanged.
    `temp_dirs` collects dirs created so caller can clean them up later.
    """
    if not is_video(path):
        return path  # already a directory or .npz file
    tmp = tempfile.mkdtemp(prefix="fid_video_frames_")
    temp_dirs.append(tmp)
    extract_video_frames(
        path, tmp, frame_stride=frame_stride, max_frames=max_frames, resize=resize
    )
    return tmp


def parse_args():
    parser = argparse.ArgumentParser(
        description="FID evaluation with VIDEO file support.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--path1", type=str, required=True,
                        help="Video file, frame directory, or .npz stats file.")
    parser.add_argument("--path2", type=str, required=True,
                        help="Video file, frame directory, or .npz stats file.")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cuda:0 / cpu. Auto-detected if not set.")
    parser.add_argument("--dims", type=int, default=2048,
                        choices=[64, 192, 768, 2048])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Keep every Nth frame from input video(s).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap on frames extracted per video.")
    parser.add_argument("--resize", type=int, default=None,
                        help="If set, resize shorter side to N and center-crop to NxN.")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep extracted frame directories after FID is computed.")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[config] device={device}, dims={args.dims}, batch_size={args.batch_size}")
    print(f"[config] frame_stride={args.frame_stride}, max_frames={args.max_frames}, "
          f"resize={args.resize}")

    temp_dirs: list[str] = []
    try:
        print(f"[input] path1={args.path1}")
        resolved_1 = resolve_input(
            args.path1, args.frame_stride, args.max_frames, args.resize, temp_dirs
        )
        print(f"[input] path2={args.path2}")
        resolved_2 = resolve_input(
            args.path2, args.frame_stride, args.max_frames, args.resize, temp_dirs
        )

        # sanity check
        for p in (resolved_1, resolved_2):
            if not os.path.exists(p):
                raise RuntimeError(f"Invalid path: {p}")

        # load Inception once
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
        model = InceptionV3([block_idx]).to(device)
        model.eval()

        t0 = time.time()
        fid = float(calculate_fid_given_paths(
            [resolved_1, resolved_2],
            args.batch_size,
            device,
            args.dims,
            args.num_workers,
        ))
        elapsed = time.time() - t0

        print(f"\n[result] FID = {fid:.6f}  (took {elapsed:.2f}s)")

        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump({
                    "config": {
                        "device": device,
                        "dims": args.dims,
                        "batch_size": args.batch_size,
                        "num_workers": args.num_workers,
                        "frame_stride": args.frame_stride,
                        "max_frames": args.max_frames,
                        "resize": args.resize,
                    },
                    "inputs": {
                        "path1": {"raw": args.path1, "resolved": resolved_1,
                                  "is_video": is_video(args.path1)},
                        "path2": {"raw": args.path2, "resolved": resolved_2,
                                  "is_video": is_video(args.path2)},
                    },
                    "fid": fid,
                    "elapsed_sec": elapsed,
                }, f, indent=2, ensure_ascii=False)
            print(f"[result] Saved JSON summary to {args.output_json}")
    finally:
        if not args.keep_temp:
            for d in temp_dirs:
                shutil.rmtree(d, ignore_errors=True)
                print(f"[cleanup] removed temp dir {d}")
        else:
            for d in temp_dirs:
                print(f"[cleanup] kept temp dir {d}")


if __name__ == "__main__":
    main()