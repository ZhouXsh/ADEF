# coding: utf-8
"""Video, audio, JSON and manifest helpers for evaluation scripts."""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import cv2
import numpy as np


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(obj, path: str | Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_manifest(path: str | Path) -> List[Dict[str, str]]:
    """Read a txt list or csv manifest.

    Txt format: one video path per line.
    Csv format: columns may include generated, reference, audio, label.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8-sig") as f:
            return [dict(row) for row in csv.DictReader(f)]
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append({"generated": line})
    return rows


def iter_video_paths(video: Optional[str] = None, manifest: Optional[str] = None) -> Iterator[Dict[str, str]]:
    if video:
        yield {"generated": video}
    if manifest:
        yield from read_manifest(manifest)


def video_info(video_path: str | Path) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": fps, "n_frames": n_frames, "width": width, "height": height, "duration": n_frames / max(fps, 1e-6)}


def read_video_frames(video_path: str | Path, max_frames: int = 0, stride: int = 1, rgb: bool = True) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, stride) == 0:
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def sample_frames(video_path: str | Path, num_frames: int = 32, rgb: bool = True) -> List[np.ndarray]:
    info = video_info(video_path)
    n = max(1, int(info["n_frames"]))
    indices = set(np.linspace(0, n - 1, num=min(num_frames, n), dtype=int).tolist())
    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in indices:
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def extract_audio(video_path: str | Path, out_wav: Optional[str | Path] = None, sr: int = 16000) -> str:
    """Extract mono wav using ffmpeg and return the output path."""
    if out_wav is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_wav = tmp.name
        tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr), str(out_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return str(out_wav)


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "count": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }
