#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paper-grade FID for talking-head videos.

This wrapper follows the Wav2Lip evaluation protocol: dump *all selected
frames from the complete real set and complete generated set* and run the
official ``pytorch-fid`` implementation once.  It deliberately does not
compute one FID per video and average those numbers, because Fréchet distance
is a distribution-level statistic and that averaging is not equivalent.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import cv2
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))
from pytorch_fid.fid_score import calculate_fid_given_paths  # noqa: E402

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _read_list(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    out = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            out.append(s)
    if not out:
        raise ValueError(f"empty video list: {p}")
    return out


def _collect_videos(path: str) -> list[str]:
    p = Path(path)
    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
        return [str(p)]
    if p.is_dir():
        return [str(x) for x in sorted(p.rglob("*")) if x.is_file() and x.suffix.lower() in VIDEO_EXTENSIONS]
    return []


def _is_frame_dir(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and any(x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS for x in p.iterdir())


def _extract_video_frames(video_path: str, out_dir: Path, prefix: str,
                          frame_stride: int = 1, max_frames: int | None = None,
                          resize: int | None = None) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    source_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if source_idx % frame_stride == 0:
            if resize is not None:
                h, w = frame.shape[:2]
                scale = resize / min(h, w)
                nw, nh = int(round(w * scale)), int(round(h * scale))
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
                x0, y0 = (nw - resize) // 2, (nh - resize) // 2
                frame = frame[y0:y0 + resize, x0:x0 + resize]
            target = out_dir / f"{prefix}_{saved:06d}.png"
            if not cv2.imwrite(str(target), frame):
                cap.release()
                raise RuntimeError(f"failed to write extracted frame: {target}")
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        source_idx += 1
    cap.release()
    if saved == 0:
        raise RuntimeError(f"no frames extracted from {video_path}")
    return saved


def _materialize_video_set(videos: Iterable[str], out_dir: Path, frame_stride: int,
                           max_frames: int | None, resize: int | None) -> tuple[int, int]:
    n_videos = 0
    n_frames = 0
    for i, video in enumerate(videos):
        if not Path(video).is_file():
            raise FileNotFoundError(video)
        n_frames += _extract_video_frames(video, out_dir, f"v{i:06d}", frame_stride, max_frames, resize)
        n_videos += 1
    return n_videos, n_frames


def _resolve_side(path: str | None, list_file: str | None, temp_root: Path,
                  side: str, frame_stride: int, max_frames: int | None,
                  resize: int | None) -> tuple[str, dict]:
    listed = _read_list(list_file)
    if listed:
        out_dir = temp_root / side
        n_videos, n_frames = _materialize_video_set(listed, out_dir, frame_stride, max_frames, resize)
        return str(out_dir), {"mode": "video_list", "n_videos": n_videos, "n_frames": n_frames}

    if not path:
        raise ValueError(f"{side}: provide path or list file")
    p = Path(path)
    if p.suffix.lower() == ".npz" and p.is_file():
        return str(p), {"mode": "stats", "n_videos": None, "n_frames": None}
    if _is_frame_dir(path):
        n_frames = sum(1 for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS)
        return str(p), {"mode": "frame_dir", "n_videos": None, "n_frames": n_frames}

    videos = _collect_videos(path)
    if not videos:
        raise ValueError(f"{side}: no videos or frames found at {path}")
    out_dir = temp_root / side
    n_videos, n_frames = _materialize_video_set(videos, out_dir, frame_stride, max_frames, resize)
    return str(out_dir), {"mode": "video_set", "n_videos": n_videos, "n_frames": n_frames}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--path1", help="Real video/file directory, frame directory, or .npz stats")
    p.add_argument("--path2", help="Fake video/file directory, frame directory, or .npz stats")
    p.add_argument("--list1", help="Text file containing one real video path per line")
    p.add_argument("--list2", help="Text file containing one fake video path per line")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--frame-stride", type=int, default=1,
                   help="Use every Nth frame. Paper default is 1 (all frames).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Optional per-video cap; leave unset for the paper protocol.")
    p.add_argument("--resize", type=int, default=None,
                   help="Optional square resize/crop; leave unset for pytorch-fid's own preprocessing.")
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument("--output-json")
    return p.parse_args()


def main():
    args = parse_args()
    if bool(args.list1) != bool(args.list2):
        raise SystemExit("--list1 and --list2 must be supplied together")
    if not args.list1 and (not args.path1 or not args.path2):
        raise SystemExit("provide --path1/--path2 or --list1/--list2")
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    temp_root = Path(tempfile.mkdtemp(prefix="adef_fid_dataset_"))
    try:
        real_path, real_meta = _resolve_side(args.path1, args.list1, temp_root, "real",
                                             args.frame_stride, args.max_frames, args.resize)
        fake_path, fake_meta = _resolve_side(args.path2, args.list2, temp_root, "fake",
                                             args.frame_stride, args.max_frames, args.resize)
        if real_meta.get("n_frames") is not None and real_meta["n_frames"] < 2:
            raise RuntimeError("FID needs at least two real frames")
        if fake_meta.get("n_frames") is not None and fake_meta["n_frames"] < 2:
            raise RuntimeError("FID needs at least two fake frames")

        t0 = time.time()
        fid = float(calculate_fid_given_paths(
            [real_path, fake_path], args.batch_size, device, args.dims, args.num_workers
        ))
        elapsed = time.time() - t0
        result = {
            "protocol": "pytorch-fid pooled-frame dataset FID",
            "fid": fid,
            "real": real_meta,
            "fake": fake_meta,
            "config": {
                "dims": args.dims, "batch_size": args.batch_size,
                "frame_stride": args.frame_stride, "max_frames": args.max_frames,
                "resize": args.resize, "device": device,
            },
            "elapsed_sec": elapsed,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output_json:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result
    finally:
        if args.keep_temp:
            print(f"[FID] kept temporary frames: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
