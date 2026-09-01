#!/usr/bin/env python3
"""Wav2Lip-compatible LSE-D / LSE-C evaluation.

The published Wav2Lip protocol first runs joonson/syncnet_python's
``run_pipeline.py`` (25-fps conversion, S3FD face detection/tracking and
224x224 face crop), then scores the resulting face track with SyncNet-v2.
This wrapper performs exactly that sequence and produces structured JSON/CSV.

For short talking-head clips we lower only the upstream track-duration gate
(``--min-track``; default 5 detected frames).  The detector, tracker, crop,
25-fps conversion, audio resampling, SyncNet weights and score definitions are
otherwise the official pipeline.  Use ``--min-track 100`` to reproduce the
original Wav2Lip real-video shell default literally.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = SCRIPT_DIR.parent.parent
SYNCNET_DIR = EVAL_ROOT / "syncnet_python"
sys.path.insert(0, str(SYNCNET_DIR))
from SyncNetInstance import SyncNetInstance  # noqa: E402

DEFAULT_WEIGHTS = SYNCNET_DIR / "data" / "syncnet_v2.model"
RUN_PIPELINE = SYNCNET_DIR / "run_pipeline.py"
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".m4v")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lse_official")


@dataclass
class Result:
    video: str
    lse_d: Optional[float]
    lse_c: Optional[float]
    av_offset: Optional[int]
    min_dist_raw: Optional[float]
    n_frames: int
    n_tracks: int
    selected_track: Optional[str]
    duration_s: float
    elapsed_s: float
    error: Optional[str] = None


class _Opt:
    pass


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return n


def _select_track(track_files: list[Path]) -> Path:
    if not track_files:
        raise RuntimeError("official SyncNet face pipeline produced no face track")
    # Talking-head clips should contain one track.  If a detector creates more,
    # choose the longest one deterministically rather than averaging identities.
    return max(track_files, key=lambda p: (_video_frame_count(p), p.name))


class Evaluator:
    def __init__(self, weights: Path, device: str, vshift: int, batch_size: int,
                 min_track: int, face_scale: float, crop_scale: float,
                 pipeline_python: str):
        if not weights.is_file():
            raise FileNotFoundError(f"SyncNet-v2 weights not found: {weights}")
        if not RUN_PIPELINE.is_file():
            raise FileNotFoundError(f"official face pipeline not found: {RUN_PIPELINE}")
        self.model = SyncNetInstance(device=device)
        self.model.loadParameters(str(weights))
        self.model.eval()
        self.device = device
        self.vshift = vshift
        self.batch_size = batch_size
        self.min_track = min_track
        self.face_scale = face_scale
        self.crop_scale = crop_scale
        self.pipeline_python = pipeline_python

    def _score_crop(self, crop: Path, tmp_root: Path):
        opt = _Opt()
        opt.tmp_dir = str(tmp_root / "score_tmp")
        opt.reference = "score"
        opt.vshift = self.vshift
        opt.batch_size = self.batch_size
        offset, conf, dists = self.model.evaluate(opt, str(crop))
        dists = np.asarray(dists)
        if dists.ndim != 2 or dists.shape[0] == 0:
            raise RuntimeError(f"invalid SyncNet distance matrix: {dists.shape}")
        mdist = dists.mean(axis=0)
        # This is Wav2Lip's published `minval` / LSE-D.
        lse_d = float(mdist.min())
        return int(np.asarray(offset).item()), float(np.asarray(conf).item()), lse_d, float(dists.min()), int(dists.shape[0])

    def score(self, video: Path) -> Result:
        t0 = time.time()
        root = Path(tempfile.mkdtemp(prefix="adef_syncnet_official_"))
        try:
            reference = "clip"
            data_dir = root / "pipeline"
            cmd = [
                self.pipeline_python, str(RUN_PIPELINE),
                "--videofile", str(video),
                "--reference", reference,
                "--data_dir", str(data_dir),
                "--min_track", str(self.min_track),
                "--facedet_scale", str(self.face_scale),
                "--crop_scale", str(self.crop_scale),
                "--frame_rate", "25",
            ]
            proc = subprocess.run(cmd, cwd=str(SYNCNET_DIR), capture_output=True, text=True)
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "").splitlines()[-3:]
                raise RuntimeError("run_pipeline failed: " + " | ".join(tail))
            tracks = sorted((data_dir / "pycrop" / reference).glob("0*.avi"))
            crop = _select_track(tracks)
            offset, conf, lse_d, min_raw, n_frames = self._score_crop(crop, root)
            return Result(
                video=str(video), lse_d=lse_d, lse_c=conf, av_offset=offset,
                min_dist_raw=min_raw, n_frames=n_frames, n_tracks=len(tracks),
                selected_track=crop.name, duration_s=n_frames / 25.0, elapsed_s=time.time() - t0,
            )
        except Exception as exc:
            return Result(
                video=str(video), lse_d=None, lse_c=None, av_offset=None,
                min_dist_raw=None, n_frames=0, n_tracks=0, selected_track=None,
                duration_s=0.0, elapsed_s=time.time() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


def expand_inputs(args) -> list[str]:
    out: list[str] = []
    if args.video:
        out.append(args.video)
    if args.videos:
        out.extend(args.videos)
    if args.video_dir:
        for ext in VIDEO_EXTS:
            out.extend(glob(os.path.join(args.video_dir, "**", "*" + ext), recursive=True))
    if args.filelist:
        out.extend(x.strip() for x in Path(args.filelist).read_text().splitlines() if x.strip() and not x.lstrip().startswith("#"))
    seen = set()
    result = []
    for raw in out:
        p = str(Path(raw).expanduser().resolve())
        if p not in seen:
            seen.add(p); result.append(p)
    return result


def _stats(vals):
    xs = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    mean = sum(xs) / len(xs)
    return {"n": len(xs), "mean": mean,
            "std": math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video")
    g.add_argument("--videos", nargs="+")
    g.add_argument("--video_dir")
    g.add_argument("--filelist")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vshift", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--min-track", type=int, default=5,
                   help="Upstream face-track duration gate; 5 supports short MEAD clips, 100 is literal Wav2Lip shell default.")
    p.add_argument("--facedet-scale", type=float, default=0.25)
    p.add_argument("--crop-scale", type=float, default=0.40)
    default_pipeline_python = SYNCNET_DIR / "syncnet_venv" / "bin" / "python"
    p.add_argument("--pipeline-python", default=str(default_pipeline_python if default_pipeline_python.is_file() else Path(sys.executable)),
                   help="Interpreter used for official syncnet_python/run_pipeline.py")
    p.add_argument("--output_json")
    p.add_argument("--output_csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    videos = expand_inputs(args)
    if not videos:
        print("ERROR: no input videos", file=sys.stderr)
        return 2
    missing = [v for v in videos if not Path(v).is_file()]
    if missing:
        print(f"ERROR: {len(missing)} input video(s) missing; first={missing[0]}", file=sys.stderr)
        return 2
    ev = Evaluator(Path(args.weights), args.device, args.vshift, args.batch_size,
                   args.min_track, args.facedet_scale, args.crop_scale, args.pipeline_python)
    results = []
    for i, v in enumerate(videos, 1):
        log.info("[%d/%d] %s", i, len(videos), v)
        r = ev.score(Path(v))
        results.append(r)
        if r.error:
            log.error("%s", r.error)
        else:
            log.info("LSE-D %.4f | LSE-C %.4f | offset %+d", r.lse_d, r.lse_c, r.av_offset)

    payload = {
        "protocol": "Wav2Lip official LSE: syncnet_python run_pipeline -> SyncNet-v2",
        "protocol_source": "https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation/scores_LSE",
        "config": {
            "fps": 25, "audio_hz": 16000, "vshift": args.vshift,
            "min_track": args.min_track, "facedet_scale": args.facedet_scale,
            "crop_scale": args.crop_scale, "weights": str(Path(args.weights).resolve()),
        },
        "n_total": len(results),
        "n_success": sum(r.error is None for r in results),
        "aggregate": {
            "lse_d": _stats([r.lse_d for r in results]),
            "lse_c": _stats([r.lse_c for r in results]),
        },
        "results": [asdict(r) for r in results],
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_csv:
        path = Path(args.output_csv); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            w.writeheader(); w.writerows(asdict(r) for r in results)
    return 0 if payload["n_success"] == payload["n_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
