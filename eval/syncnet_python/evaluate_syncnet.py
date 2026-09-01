#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SyncNet 评估脚本
================

提供两种评估模式:
1. 单视频评估 - 对单个已裁剪的人脸视频计算 AV offset / confidence / min distance
2. 批量评估 - 对一个目录下所有视频进行评估,并输出 JSON/CSV 报告
3. 完整管线评估 - 对原始视频运行完整管线 (face detect -> crop -> sync -> 可选可视化)

环境要求:
    - Python 3.10+ (在 3.13 上测试通过)
    - PyTorch 2.5+ (CUDA 12.4)
    - opencv-contrib-python, scenedetect, python_speech_features, numpy, scipy, tqdm, ffmpeg

使用示例:
    # 单个视频评估
    python evaluate_syncnet.py --videofile /path/to/video.mp4 --mode single

    # 批量评估
    python evaluate_syncnet.py --video_dir /path/to/videos --output report.csv --mode batch

    # 完整管线
    python evaluate_syncnet.py --videofile /path/to/video.mp4 --data_dir ./output --mode pipeline
"""

import argparse
import csv
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("SyncNetEval")

# Make the local module files importable when invoked from anywhere
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from SyncNetInstance import SyncNetInstance  # noqa: E402


# ==================== Default args shared by all sub-scripts ====================
DEFAULTS = {
    "initial_model": str(SCRIPT_DIR / "data" / "syncnet_v2.model"),
    "batch_size": 20,
    "vshift": 15,
    "data_dir": str(SCRIPT_DIR / "data" / "work"),
    "tmp_dir": str(SCRIPT_DIR / "data" / "work" / "pytmp"),
    "reference": "demo",
    "facedet_scale": 0.25,
    "crop_scale": 0.40,
    "min_track": 50,
    "frame_rate": 25,
    "num_failed_det": 25,
    "min_face_size": 100,
}


# ==================== Helpers ====================
def make_opt(args, **overrides):
    """Build a namespace compatible with SyncNetInstance.evaluate / run_pipeline."""
    import argparse as _ap

    opt = _ap.Namespace(**{**DEFAULTS, **vars(args)})
    for k, v in overrides.items():
        setattr(opt, k, v)
    opt.avi_dir = os.path.join(opt.data_dir, "pyavi")
    opt.work_dir = os.path.join(opt.data_dir, "pywork")
    opt.crop_dir = os.path.join(opt.data_dir, "pycrop")
    opt.frames_dir = os.path.join(opt.data_dir, "pyframes")
    opt.tmp_dir = os.path.join(opt.data_dir, "pytmp")
    return opt


def load_model(model_path: str) -> SyncNetInstance:
    """Load the SyncNet model from disk, log device info."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"SyncNet model not found at {model_path}. "
            f"Run `bash download_model.sh` first."
        )
    s = SyncNetInstance()
    s.loadParameters(model_path)
    logger.info("SyncNet model loaded from %s", model_path)
    return s


def parse_demo_result(offset, conf, dist):
    """Normalize a single (offset, conf, dist) tuple to a Python dict.

    The SyncNet training script reports two distance numbers:
    - `Min dist` — the smallest entry of the mean cross-distance curve
      (one value per shift). `evaluate()` calls this `minval` and logs it.
    - `Confidence` — median(mdist) - minval.

    `dist` is the full (T, 2*vshift+1) cross-distance matrix; the curve-level
    minimum is `min(mean(dist, axis=0))`, matching `minval`.
    """
    dist = np.asarray(dist)
    if dist.ndim == 2:
        mean_curve = np.mean(dist, axis=0)
        min_dist = float(mean_curve.min())
    else:
        min_dist = float(dist.min())
    return {
        "offset": int(offset) if hasattr(offset, "item") else int(offset),
        "confidence": float(conf) if hasattr(conf, "item") else float(conf),
        "min_distance": min_dist,
    }


# ==================== Mode 1: single video (cropped face track) ====================
def eval_single(args):
    """Evaluate a single cropped face video (must be the output of run_pipeline.py)."""
    model = load_model(args.initial_model)
    opt = make_opt(args)
    os.makedirs(os.path.join(opt.tmp_dir, opt.reference), exist_ok=True)

    offset, conf, dist = model.evaluate(opt, videofile=args.videofile)
    result = parse_demo_result(offset, conf, dist)
    logger.info(
        "AV offset: %d  Min dist: %.3f  Confidence: %.3f",
        result["offset"],
        result["min_distance"],
        result["confidence"],
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"videofile": args.videofile, **result}, f, indent=2)
        logger.info("Saved single-video result to %s", out_path)

    return result


# ==================== Mode 2: batch ====================
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wav"}


def iter_videos(path):
    p = Path(path)
    if p.is_file():
        yield p
        return
    for ext in VIDEO_EXTS:
        for f in sorted(p.glob(f"*{ext}")):
            yield f
        for f in sorted(p.glob(f"*{ext.upper()}")):
            yield f


def eval_batch(args):
    """Run sync evaluation on every video in a directory."""
    videos = list(iter_videos(args.video_dir))
    if not videos:
        raise RuntimeError(f"No videos found in {args.video_dir}")

    logger.info("Found %d videos to evaluate", len(videos))
    model = load_model(args.initial_model)

    results = []
    for v in videos:
        ref = v.stem
        opt = make_opt(args, reference=ref)
        os.makedirs(os.path.join(opt.tmp_dir, ref), exist_ok=True)
        logger.info("=== Evaluating %s ===", v.name)
        t0 = time.time()
        try:
            offset, conf, dist = model.evaluate(opt, videofile=str(v))
            r = parse_demo_result(offset, conf, dist)
            r["videofile"] = str(v)
            r["elapsed_sec"] = round(time.time() - t0, 2)
            logger.info(
                "AV offset: %d  Min dist: %.3f  Confidence: %.3f  (%.1fs)",
                r["offset"], r["min_distance"], r["confidence"], r["elapsed_sec"],
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed on %s: %s", v, e)
            r = {"videofile": str(v), "error": str(e)}
        results.append(r)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".csv":
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list({k for r in results for k in r.keys()}))
                w.writeheader()
                for r in results:
                    w.writerow(r)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Saved batch result to %s", out_path)
    return results


# ==================== Mode 3: full pipeline ====================
def _import_run_pipeline():
    """Import run_pipeline while shielding sys.argv.

    run_pipeline.py calls `parser.parse_args()` at module top level, which
    consumes whatever is currently in sys.argv. We swap in a harmless argv
    before importing, then restore the original so the parent script is
    unaffected. This lets us reuse run_pipeline's functions without it
    choking on the CLI flags passed to evaluate_syncnet.py.
    """
    import run_pipeline  # noqa: WPS433
    return run_pipeline


def eval_pipeline(args):
    """Run the full SyncNet pipeline: face detection + crop + sync + visualisation."""
    # Guard sys.argv so the eager parser.parse_args() in run_pipeline does not
    # see evaluate_syncnet.py's own CLI flags.
    saved_argv = sys.argv
    sys.argv = ["run_pipeline.py"]
    try:
        run_pipeline = _import_run_pipeline()
        # run_pipeline.py also runs its "delete existing dirs" check at module
        # import time using its own opt (which has no CLI args). Force
        # --overwrite there if the user requested it.
        if getattr(args, "overwrite", False):
            run_pipeline.opt.overwrite = True
        inference_video = run_pipeline.inference_video
        scene_detect = run_pipeline.scene_detect
        track_shot = run_pipeline.track_shot
        crop_video = run_pipeline.crop_video
    finally:
        sys.argv = saved_argv

    from scenedetect import open_video, SceneManager, ContentDetector  # noqa: F401

    from shutil import rmtree

    opt = make_opt(args, reference=args.reference)
    for d in [opt.work_dir, opt.crop_dir, opt.avi_dir, opt.frames_dir, opt.tmp_dir]:
        path = os.path.join(d, opt.reference)
        if os.path.exists(path):
            if not args.overwrite:
                raise FileExistsError(f"Output exists: {path}. Use --overwrite to overwrite.")
            rmtree(path)
        os.makedirs(path, exist_ok=True)

    # ---- Convert & extract frames/audio at 25 fps ----
    logger.info("Converting video to 25fps: %s", args.videofile)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", args.videofile,
            "-qscale:v", "2", "-async", "1", "-r", "25",
            os.path.join(opt.avi_dir, opt.reference, "video.avi"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i",
            os.path.join(opt.avi_dir, opt.reference, "video.avi"),
            "-qscale:v", "2", "-threads", "1", "-f", "image2",
            os.path.join(opt.frames_dir, opt.reference, "%06d.jpg"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i",
            os.path.join(opt.avi_dir, opt.reference, "video.avi"),
            "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
            os.path.join(opt.avi_dir, opt.reference, "audio.wav"),
        ],
        check=True,
    )

    faces = inference_video(opt)
    scene = scene_detect(opt)

    all_tracks = []
    for shot in scene:
        # scenedetect >= 0.7 deprecated FrameTimecode.get_frames() in favour of frame_num
        start_frame = getattr(shot[0], "frame_num", None)
        if start_frame is None:
            start_frame = shot[0].get_frames()
        end_frame = getattr(shot[1], "frame_num", None)
        if end_frame is None:
            end_frame = shot[1].get_frames()

        if end_frame - start_frame >= opt.min_track:
            all_tracks.extend(track_shot(opt, faces[start_frame:end_frame]))

    vid_tracks = []
    for i, track in enumerate(all_tracks):
        vid_tracks.append(
            crop_video(opt, track, os.path.join(opt.crop_dir, opt.reference, f"{i:05d}"))
        )

    with open(os.path.join(opt.work_dir, opt.reference, "tracks.pckl"), "wb") as fil:
        pickle.dump(vid_tracks, fil)
    rmtree(os.path.join(opt.tmp_dir, opt.reference))

    # ---- SyncNet scoring on each track ----
    import glob as _g
    flist = sorted(_g.glob(os.path.join(opt.crop_dir, opt.reference, "0*.avi")))
    if not flist:
        raise RuntimeError("Pipeline produced no face tracks; check --facedet_scale / --min_track")

    model = load_model(args.initial_model)
    dists = []
    for fname in flist:
        offset, conf, dist = model.evaluate(opt, videofile=fname)
        dists.append(dist)
        logger.info(
            "Track %s  AV offset: %d  Confidence: %.3f",
            os.path.basename(fname), int(offset) if hasattr(offset, "item") else int(offset),
            float(conf) if hasattr(conf, "item") else float(conf),
        )

    with open(os.path.join(opt.work_dir, opt.reference, "activesd.pckl"), "wb") as fil:
        pickle.dump(dists, fil)

    # ---- Optional visualisation ----
    if args.visualise:
        # run_visualise runs as a top-level script. Re-implement the core step here so
        # that we can call it programmatically with the same options.
        from scipy import signal as _sig  # noqa: WPS433

        with open(os.path.join(opt.work_dir, opt.reference, "tracks.pckl"), "rb") as fil:
            tracks = pickle.load(fil, encoding="latin1")
        with open(os.path.join(opt.work_dir, opt.reference, "activesd.pckl"), "rb") as fil:
            dists_pkl = pickle.load(fil, encoding="latin1")

        flist_frames = sorted(_g.glob(os.path.join(opt.frames_dir, opt.reference, "*.jpg")))
        faces_per_frame = [[] for _ in range(len(flist_frames))]
        for tidx, track in enumerate(tracks):
            mean_d = np.mean(np.stack(dists_pkl[tidx], 1), 1)
            minidx = np.argmin(mean_d, 0)
            fdist = np.stack([d[minidx] for d in dists_pkl[tidx]])
            fdist = np.pad(fdist, (3, 3), "constant", constant_values=10)
            fconf = np.median(mean_d) - fdist
            fconfm = _sig.medfilt(fconf, kernel_size=9)
            track_frames = track["track"]["frame"].tolist()
            n = len(track_frames)
            # For short tracks the per-track distance array can be shorter than
            # the number of frames in the track (SyncNet produces one entry per
            # 5-frame window, not per frame). Re-pad fconfm to match n.
            if len(fconfm) < n:
                fconfm = np.pad(fconfm, (0, n - len(fconfm)), "edge")
            for fidx, frame in enumerate(track_frames):
                conf_idx = min(fidx, len(fconfm) - 1)
                faces_per_frame[frame].append(
                    {
                        "track": tidx,
                        "conf": float(fconfm[conf_idx]),
                        "s": float(track["proc_track"]["s"][fidx]),
                        "x": float(track["proc_track"]["x"][fidx]),
                        "y": float(track["proc_track"]["y"][fidx]),
                    }
                )

        first = cv2.imread(flist_frames[0])
        fw, fh = first.shape[1], first.shape[0]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        vout = cv2.VideoWriter(
            os.path.join(opt.avi_dir, opt.reference, "video_only.avi"),
            fourcc, opt.frame_rate, (fw, fh),
        )
        from tqdm import tqdm  # noqa: WPS433
        for fidx, fname in tqdm(enumerate(flist_frames), total=len(flist_frames), desc="Rendering"):
            img = cv2.imread(fname)
            for face in faces_per_frame[fidx]:
                clr = int(max(min(face["conf"] * 25, 255), 0))
                cv2.rectangle(
                    img,
                    (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                    (int(face["x"] + face["s"]), int(face["y"] + face["s"])),
                    (0, clr, 255 - clr), 3,
                )
                cv2.putText(
                    img,
                    f"Track {face['track']}, Conf {face['conf']:.3f}",
                    (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                )
            vout.write(img)
        vout.release()

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", os.path.join(opt.avi_dir, opt.reference, "video_only.avi"),
                "-i", os.path.join(opt.avi_dir, opt.reference, "audio.wav"),
                "-c:v", "copy", "-c:a", "copy",
                os.path.join(opt.avi_dir, opt.reference, "video_out.avi"),
            ],
            check=True,
        )
        logger.info("Wrote %s", os.path.join(opt.avi_dir, opt.reference, "video_out.avi"))

    return {"reference": opt.reference, "n_tracks": len(vid_tracks)}


# ==================== CLI ====================
def build_parser():
    p = argparse.ArgumentParser(
        description="SyncNet evaluation toolkit (single / batch / full-pipeline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["single", "batch", "pipeline"], default="single",
                   help="Evaluation mode")
    p.add_argument("--initial_model", default=DEFAULTS["initial_model"],
                   help="Path to the pretrained SyncNet model")
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--vshift", type=int, default=DEFAULTS["vshift"],
                   help="Maximum allowed AV shift (in frames)")

    # single / batch
    p.add_argument("--videofile", type=str, help="Path to a single (cropped) video file")
    p.add_argument("--video_dir", type=str, help="Directory of (cropped) video files (batch mode)")
    p.add_argument("--output", type=str, help="Output path (json or csv) for the report")

    # pipeline
    p.add_argument("--data_dir", default=DEFAULTS["data_dir"], help="Pipeline output directory")
    p.add_argument("--reference", default=DEFAULTS["reference"], help="Reference name for the video")
    p.add_argument("--facedet_scale", type=float, default=DEFAULTS["facedet_scale"])
    p.add_argument("--crop_scale", type=float, default=DEFAULTS["crop_scale"])
    p.add_argument("--min_track", type=int, default=DEFAULTS["min_track"])
    p.add_argument("--frame_rate", type=int, default=DEFAULTS["frame_rate"])
    p.add_argument("--num_failed_det", type=int, default=DEFAULTS["num_failed_det"])
    p.add_argument("--min_face_size", type=int, default=DEFAULTS["min_face_size"])
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing pipeline outputs")
    p.add_argument("--visualise", action="store_true",
                   help="(pipeline mode) render the bounding-box visualisation video")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "single":
        if not args.videofile:
            parser.error("--videofile is required in single mode")
        eval_single(args)
    elif args.mode == "batch":
        if not args.video_dir:
            parser.error("--video_dir is required in batch mode")
        eval_batch(args)
    elif args.mode == "pipeline":
        if not args.videofile:
            parser.error("--videofile is required in pipeline mode")
        eval_pipeline(args)


if __name__ == "__main__":
    main()
