#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lip-Sync Evaluation using SyncNet (LSE-D / LSE-C metrics).

Computes per-video lip-synchronization quality for already-generated videos
(typically the output of Wav2Lip, SadTalker, MuseTalk, etc.) using the
SyncNet audio-visual sync model from https://github.com/joonson/syncnet_python.

Two metrics are reported per video (lower LSE-D / higher LSE-C = better sync).
The definitions match the Wav2Lip paper and the upstream syncnet_python code
(see SyncNetInstance.py:138-150):

  * LSE-D : ``min over offsets ( mean over frames of pairwise distance )``
            i.e. for each candidate offset (vshift × 2 + 1 values), compute
            the average audio-visual distance across all frames, then take
            the best offset.  Lower means audio and mouth shapes are closer
            in the model's embedding space at the optimal alignment.
  * LSE-C : ``median(mdist) - min(mdist)`` where ``mdist`` is the per-offset
            vector above.  Higher means the best alignment is clearly better
            than the median (the model is confident the audio and video are
            actually synced).
  * AV offset : the audio-visual time offset, in 25-fps frames, that maximises
            sync.  0 is perfect, positive means audio leads, negative means
            video leads.

NOTE: do NOT confuse ``LSE-D`` here with the absolute minimum of the (T,
2vshift+1) distance matrix — those are different quantities.  See
``paper_aligned_metrics`` in the source for the exact formula.

Usage examples
--------------
Single video:
    python eval_lipsync.py --video path/to/video.mp4

Directory of videos:
    python eval_lipsync.py --video_dir path/to/videos/

Multiple specific videos (glob-friendly shell expansion):
    python eval_lipsync.py --videos v1.mp4 v2.mp4 v3.mp4

Save results and aggregate stats:
    python eval_lipsync.py --video_dir results/ \
                           --output_csv results/lipsync_scores.csv \
                           --output_json results/lipsync_scores.json

Save per-frame confidence traces for offline plotting:
    python eval_lipsync.py --video_dir results/ \
                           --save_frame_conf frame_conf/

Notes
-----
* SyncNet assumes 25 fps video and 16 kHz mono audio.  ffmpeg resamples to
  these rates internally so any input that ffmpeg can decode is supported.
* The SyncNet model file must be downloaded separately to
  syncnet_python/data/syncnet_v2.model (see its README).
* This script uses the same SyncNet weights and inference code as the
  reference syncnet_python repository, so its scores are directly comparable
  to the LSE-D / LSE-C values reported in the Wav2Lip paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, asdict
from glob import glob
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup so we can import the upstream SyncNetInstance from syncnet_python
# regardless of where this script is invoked from.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent            # .../Wav2Lip
REPO_ROOT = PROJECT_ROOT.parent             # .../ADEF_remake/eval
SYNCNET_DIR = REPO_ROOT / "syncnet_python"
if not SYNCNET_DIR.is_dir():
    raise RuntimeError(
        f"Could not find syncnet_python at {SYNCNET_DIR}. "
        "Make sure the repository has been cloned alongside Wav2Lip."
    )
sys.path.insert(0, str(SYNCNET_DIR))

from SyncNetInstance import SyncNetInstance  # noqa: E402

DEFAULT_SYNCNET_WEIGHTS = SYNCNET_DIR / "data" / "syncnet_v2.model"
DEFAULT_FACE_WEIGHTS = SYNCNET_DIR / "detectors" / "s3fd" / "weights" / "sfd_face.pth"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_lipsync")


# ---------------------------------------------------------------------------
# Result dataclass — kept tiny so JSON output stays human-readable.
# ---------------------------------------------------------------------------
@dataclass
class LipSyncResult:
    video: str
    lse_d: float                 # paper-aligned: min over offsets of mean over frames
    lse_c: float                 # median(mdist) - min(mdist), higher is better
    av_offset: int               # frames, +ve = audio leads
    min_dist_raw: float          # raw global min of dists (kept for diagnostics, NOT the paper value)
    n_frames: int
    duration_s: float
    elapsed_s: float
    error: Optional[str] = None

    @classmethod
    def from_syncnet(
        cls,
        video: str,
        offset: int,
        conf: float,
        lse_d: float,
        min_dist_raw: float,
        n_frames: int,
        duration_s: float,
        elapsed_s: float,
    ) -> "LipSyncResult":
        return cls(
            video=video,
            lse_d=float(lse_d),
            lse_c=float(conf),
            av_offset=int(offset),
            min_dist_raw=float(min_dist_raw),
            n_frames=int(n_frames),
            duration_s=float(duration_s),
            elapsed_s=float(elapsed_s),
        )


# ---------------------------------------------------------------------------
# SyncNet wrapper.
# ---------------------------------------------------------------------------
class LipSyncEvaluator:
    """Loads SyncNet once and exposes a `.score(video)` method."""

    def __init__(
        self,
        weights_path: Path = DEFAULT_SYNCNET_WEIGHTS,
        vshift: int = 15,
        batch_size: int = 20,
        device: Optional[str] = None,
    ) -> None:
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"SyncNet weights not found at {weights_path}. "
                "Run syncnet_python/download_model.sh first."
            )

        self.vshift = vshift
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        log.info("Loading SyncNet from %s on %s", weights_path, self.device)
        t0 = time.time()
        self.model = SyncNetInstance(device=self.device)
        self.model.loadParameters(str(weights_path))
        self.model.eval()
        log.info("SyncNet ready (%.1fs)", time.time() - t0)

        # SyncNetInstance.evaluate reads a handful of attributes off `opt`
        # and uses opt.tmp_dir / opt.reference for its working directory.
        class _Opt:
            pass

        self._opt = _Opt()
        self._opt.tmp_dir = tempfile.mkdtemp(prefix="syncnet_eval_")
        self._opt.reference = "work"
        self._opt.vshift = vshift
        self._opt.batch_size = batch_size

    def score(self, video_path: str) -> LipSyncResult:
        """Run SyncNet on a single video.  Returns a LipSyncResult."""
        if not os.path.isfile(video_path):
            return LipSyncResult(
                video=video_path,
                lse_d=float("nan"),
                lse_c=float("nan"),
                av_offset=0,
                min_dist_raw=float("nan"),
                n_frames=0,
                duration_s=0.0,
                elapsed_s=0.0,
                error="file_not_found",
            )

        t0 = time.time()
        try:
            offset, conf, dists = self.model.evaluate(self._opt, video_path)
        except Exception as exc:  # noqa: BLE001
            log.error("SyncNet failed on %s: %s", video_path, exc)
            log.debug(traceback.format_exc())
            return LipSyncResult(
                video=video_path,
                lse_d=float("nan"),
                lse_c=float("nan"),
                av_offset=0,
                min_dist_raw=float("nan"),
                n_frames=0,
                duration_s=0.0,
                elapsed_s=time.time() - t0,
                error=f"{type(exc).__name__}: {exc}",
            )

        # Paper-aligned metrics.
        # SyncNetInstance.evaluate returns:
        #   offset       : int   (frames, +ve = audio leads)
        #   conf         : float = median(mdist) - min(mdist)  -> LSE-C directly
        #   dists        : ndarray shape (T, 2*vshift+1) of pairwise distances
        #                  where dists[t, k] = || vis_t - aud_{t + k - vshift} ||_2
        # The Wav2Lip / SyncNet papers define LSE-D as:
        #   LSE-D = min over offsets ( mean over frames of dists[:, k] )
        # i.e. compute the mean distance across all frames for each candidate
        # offset, then take the offset with the smallest mean.  This is NOT
        # simply min(dists), which is a different, non-paper quantity.
        mdist_per_offset = dists.mean(axis=0)              # (2*vshift+1,)
        lse_d_paper = float(mdist_per_offset.min())         # scalar
        min_dist_raw = float(dists.min())                   # raw global min, kept for diagnostics

        n_frames = int(dists.shape[0])
        # The "duration" of useful inference is roughly n_frames / 25 because
        # SyncNet consumes 5-frame stacks.
        duration_s = n_frames / 25.0
        elapsed = time.time() - t0

        return LipSyncResult.from_syncnet(
            video=video_path,
            offset=int(offset),
            conf=float(conf),
            lse_d=lse_d_paper,
            min_dist_raw=min_dist_raw,
            n_frames=n_frames,
            duration_s=duration_s,
            elapsed_s=elapsed,
        )

    def close(self) -> None:
        import shutil

        try:
            shutil.rmtree(self._opt.tmp_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".m4v")


def expand_inputs(args: argparse.Namespace) -> List[str]:
    """Resolve --video / --videos / --video_dir into a flat list of files."""
    out: List[str] = []
    if args.video:
        out.append(args.video)
    if args.videos:
        out.extend(args.videos)
    if args.video_dir:
        for ext in VIDEO_EXTS:
            out.extend(sorted(glob(os.path.join(args.video_dir, f"*{ext}"))))
            out.extend(sorted(glob(os.path.join(args.video_dir, f"**/*{ext}"), recursive=True)))
    if args.filelist:
        with open(args.filelist) as f:
            out.extend(line.strip() for line in f if line.strip())
    # de-dup but keep order
    seen, dedup = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def save_frame_confidence(
    evaluator: LipSyncEvaluator,
    video_path: str,
    out_dir: Path,
) -> Optional[Path]:
    """Optional: re-run inference and persist per-frame confidence to .npy."""
    try:
        offset, conf, dists = evaluator.model.evaluate(evaluator._opt, video_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Frame-level confidence failed for %s: %s", video_path, exc)
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    npy_path = out_dir / f"{stem}_conf.npy"
    np.save(
        npy_path,
        np.stack(
            [
                np.arange(dists.shape[0]),                # frame index
                np.median(dists, axis=1) - dists.min(1),  # per-frame confidence
                dists.min(axis=1),                        # per-frame min distance
            ],
            axis=1,
        ),
    )
    return npy_path


def write_csv(results: List[LipSyncResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else [
        "video", "lse_d", "lse_c", "av_offset", "min_dist_raw",
        "n_frames", "duration_s", "elapsed_s", "error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def write_json(results: List[LipSyncResult], path: Path, aggregate: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(
            {
                "results": [asdict(r) for r in results],
                "aggregate": aggregate,
            },
            f,
            indent=2,
        )


def aggregate(results: List[LipSyncResult]) -> dict:
    """Mean/std of LSE-D and LSE-C across videos that succeeded."""
    ok = [r for r in results if r.error is None]
    if not ok:
        return {"n_total": len(results), "n_success": 0}
    lse_d = np.array([r.lse_d for r in ok])
    lse_c = np.array([r.lse_c for r in ok])
    return {
        "n_total": len(results),
        "n_success": len(ok),
        "n_failed": len(results) - len(ok),
        "lse_d_mean": float(lse_d.mean()),
        "lse_d_std": float(lse_d.std()),
        "lse_c_mean": float(lse_c.mean()),
        "lse_c_std": float(lse_c.std()),
        "lse_c_min": float(lse_c.min()),
        "lse_c_max": float(lse_c.max()),
        "av_offset_mean": float(np.mean([r.av_offset for r in ok])),
    }


# ---------------------------------------------------------------------------
# Pretty console summary.
# ---------------------------------------------------------------------------
def print_summary(results: List[LipSyncResult], agg: dict) -> None:
    bar = "=" * 78
    print(bar)
    print(f"{'video':<48} {'LSE-D':>8} {'LSE-C':>8} {'offset':>7} {'frames':>7}")
    print("-" * 78)
    for r in results:
        if r.error:
            print(f"{Path(r.video).name:<48} ERROR  {r.error[:30]}")
            continue
        print(
            f"{Path(r.video).name:<48} "
            f"{r.lse_d:>8.3f} {r.lse_c:>8.3f} {r.av_offset:>7d} {r.n_frames:>7d}"
        )
    print("-" * 78)
    if agg.get("n_success", 0):
        print(
            f"AGGREGATE  n={agg['n_success']}/{agg['n_total']}  "
            f"LSE-D = {agg['lse_d_mean']:.3f} ± {agg['lse_d_std']:.3f}  "
            f"LSE-C = {agg['lse_c_mean']:.3f} ± {agg['lse_c_std']:.3f}  "
            f"AV offset ≈ {agg['av_offset_mean']:.2f} frames"
        )
    print(bar)


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lip-sync evaluation using SyncNet (LSE-D / LSE-C).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inp = parser.add_argument_group("input (choose at least one)")
    inp.add_argument("--video", help="path to a single video file")
    inp.add_argument("--videos", nargs="+", help="list of video files")
    inp.add_argument("--video_dir", help="directory of videos (recurses one level)")
    inp.add_argument("--filelist", help="plain-text file with one video path per line")

    out = parser.add_argument_group("output")
    out.add_argument("--output_csv", help="write per-video results to this CSV")
    out.add_argument("--output_json", help="write per-video + aggregate to this JSON")
    out.add_argument(
        "--save_frame_conf",
        help="directory to dump per-frame confidence .npy traces (one per video)",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--syncnet_weights", default=str(DEFAULT_SYNCNET_WEIGHTS))
    model.add_argument("--vshift", type=int, default=15,
                       help="audio-visual search window in frames (each side)")
    model.add_argument("--batch_size", type=int, default=20)
    model.add_argument("--device", default=None, help="cuda / cuda:0 / cpu")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if not (args.video or args.videos or args.video_dir or args.filelist):
        parser.error("Provide --video, --videos, --video_dir, or --filelist.")

    videos = expand_inputs(args)
    if not videos:
        log.error("No videos found.")
        return 1

    log.info("Found %d video(s).", len(videos))

    evaluator = LipSyncEvaluator(
        weights_path=Path(args.syncnet_weights),
        vshift=args.vshift,
        batch_size=args.batch_size,
        device=args.device,
    )

    results: List[LipSyncResult] = []
    try:
        for v in tqdm(videos, desc="Evaluating", unit="video"):
            log.info("Scoring %s", v)
            r = evaluator.score(v)
            results.append(r)
            if args.save_frame_conf:
                npy = save_frame_confidence(evaluator, v, Path(args.save_frame_conf))
                if npy:
                    log.info("  → frame confidences saved to %s", npy)
    finally:
        evaluator.close()

    agg = aggregate(results)
    print_summary(results, agg)

    if args.output_csv:
        write_csv(results, Path(args.output_csv))
        log.info("Wrote CSV → %s", args.output_csv)
    if args.output_json:
        write_json(results, Path(args.output_json), agg)
        log.info("Wrote JSON → %s", args.output_json)

    return 0 if agg.get("n_success", 0) == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())