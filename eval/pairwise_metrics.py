#!/usr/bin/env python3
"""Paired PSNR/SSIM/LPIPS/LMD evaluation with resumable per-video aggregation.

PSNR/SSIM use the vendored EAT ``utils_crop_psnr.crop_and_align`` path.
LMD uses EAT ``utils_crop.crop_and_align`` followed by the official dlib-68
landmark definition. LPIPS is evaluated on the same aligned pairs as
PSNR/SSIM.

For large evaluation sets, every metric is first averaged within each video.
The final dataset value is then the equal-weight mean over videos that produced
a valid value for that metric. A checkpoint and per-video CSV are written after
each processed video so interrupted/timeout runs can resume without discarding
completed work.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
EAT_CODE = THIS_DIR / "evaluation_eat" / "code"
EAT_CHECKPOINTS = THIS_DIR / "evaluation_eat" / "checkpoints"
sys.path.insert(0, str(THIS_DIR))
from paper_protocol import (  # noqa: E402
    PROTOCOL_VERSION,
    manifest_fingerprint,
    read_manifest,
    summarize,
)

PAIRWISE_AGGREGATION_VERSION = "per-video-resumable-v1"

warnings.filterwarnings("ignore", message=r"`estimate` is deprecated.*", category=FutureWarning)


def _read_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {path}")
    return frames


def _temporal_pairs(fake: list[np.ndarray], gt: list[np.ndarray]):
    length = min(len(fake), len(gt))
    if length <= 0:
        return []
    fi = np.linspace(0, len(fake), length, endpoint=False).astype(np.int32)
    gi = np.linspace(0, len(gt), length, endpoint=False).astype(np.int32)
    return [(fake[int(a)], gt[int(b)]) for a, b in zip(fi, gi)]


def _resolve_lmd_predictor(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    candidates = [
        EAT_CODE / "shape_predictor_68_face_landmarks.dat",
        EAT_CHECKPOINTS / "shape_predictor_68_face_landmarks.dat",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


def _load_eat_cropper(filename: str, module_name: str):
    path = EAT_CODE / filename
    if not path.is_file():
        raise FileNotFoundError(f"EAT crop/alignment helper is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import EAT cropper: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EAT_CODE))
    old_cwd = os.getcwd()
    try:
        os.chdir(EAT_CODE)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(EAT_CODE))
        except ValueError:
            pass

    def official_crop(image):
        cwd = os.getcwd()
        try:
            os.chdir(EAT_CODE)
            return mod.crop_and_align(image)
        finally:
            os.chdir(cwd)

    return official_crop


def _aligned_pair(fake_rgb: np.ndarray, gt_rgb: np.ndarray, cropper):
    gt_aligned, gt_ok = cropper(gt_rgb)
    if not gt_ok:
        return None
    fake_aligned, fake_ok = cropper(fake_rgb)
    if not fake_ok:
        return None
    if fake_aligned.shape != gt_aligned.shape:
        fake_aligned = cv2.resize(
            fake_aligned, (gt_aligned.shape[1], gt_aligned.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return fake_aligned, gt_aligned


def _psnr_ssim(fake: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    psnr = float(peak_signal_noise_ratio(gt, fake, data_range=255))
    try:
        ssim = float(structural_similarity(gt, fake, data_range=255, channel_axis=-1))
    except TypeError:
        ssim = float(structural_similarity(gt, fake, data_range=255, multichannel=True))
    return psnr, ssim


class LandmarkMetric:
    def __init__(self, predictor_path: Path):
        try:
            import dlib
            from imutils import face_utils
        except ImportError as exc:
            raise RuntimeError("dlib and imutils are required for EAT LMD") from exc
        if not predictor_path.is_file():
            raise FileNotFoundError(
                f"EAT landmark predictor not found: {predictor_path}. Place "
                "shape_predictor_68_face_landmarks.dat under evaluation_eat/code/ "
                "or evaluation_eat/checkpoints/."
            )
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor_path))
        self.face_utils = face_utils
        self.mouth_start, self.mouth_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    def landmarks(self, img: np.ndarray):
        # Keep the same channel convention as the vendored EAT evaluation code.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        if not rects:
            return None, None
        shape = self.predictor(gray, rects[0])
        shape = self.face_utils.shape_to_np(shape)
        return shape[self.mouth_start:self.mouth_end], shape

    @staticmethod
    def distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
        if a is None or b is None or len(a) != len(b):
            return None
        a = a.astype(np.float64) - a.astype(np.float64).mean(axis=0, keepdims=True)
        b = b.astype(np.float64) - b.astype(np.float64).mean(axis=0, keepdims=True)
        return float(np.linalg.norm(a - b, axis=1).mean())


def _load_lpips(device: str, net: str):
    try:
        import lpips
        import torch
    except ImportError as exc:
        raise RuntimeError("official `lpips` package is required for LPIPS") from exc
    resolved = torch.device("cpu" if device.startswith("cuda") and not torch.cuda.is_available() else device)
    model = lpips.LPIPS(net=net).to(resolved).eval()
    return lpips, torch, model, resolved


def _lpips_score(lpips_mod, torch_mod, model, device, fake: np.ndarray, gt: np.ndarray) -> float:
    a = lpips_mod.im2tensor(fake).to(device)
    b = lpips_mod.im2tensor(gt).to(device)
    with torch_mod.no_grad():
        return float(model(a, b).reshape(-1)[0].item())


def _mean_keep_posinf(values: list[float]) -> Optional[float]:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return None
    if any(math.isinf(v) and v > 0 for v in vals):
        return float("inf")
    finite = [v for v in vals if math.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _summarize_psnr(values: list[float]) -> dict:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None}
    if any(math.isinf(v) and v > 0 for v in vals):
        return {"n": len(vals), "mean": float("inf"), "std": None}
    return summarize(vals)


def _aggregate(per_video: list[dict[str, Any]], metrics: set[str]) -> tuple[dict, dict]:
    psnr_values = [r["psnr"] for r in per_video if r.get("psnr_ok") and r.get("psnr") is not None]
    ssim_values = [r["ssim"] for r in per_video if r.get("ssim_ok") and r.get("ssim") is not None]
    lpips_values = [r["lpips"] for r in per_video if r.get("lpips_ok") and r.get("lpips") is not None]
    mouth_values = [r["mouth_lmd"] for r in per_video if r.get("lmd_ok") and r.get("mouth_lmd") is not None]
    face_values = [r["face_lmd"] for r in per_video if r.get("lmd_ok") and r.get("face_lmd") is not None]

    aggregate = {
        "psnr": _summarize_psnr(psnr_values) if "psnr" in metrics else None,
        "ssim": summarize(ssim_values) if "ssim" in metrics else None,
        "lpips": summarize(lpips_values) if "lpips" in metrics else None,
        "mouth_lmd": summarize(mouth_values) if "lmd" in metrics else None,
        "face_lmd": summarize(face_values) if "lmd" in metrics else None,
    }
    coverage = {
        "psnr": len(psnr_values) if "psnr" in metrics else None,
        "ssim": len(ssim_values) if "ssim" in metrics else None,
        "lpips": len(lpips_values) if "lpips" in metrics else None,
        "lmd": len(mouth_values) if "lmd" in metrics else None,
    }
    return aggregate, coverage


def _input_signature(samples) -> list[dict[str, Any]]:
    """File-state signature prevents reuse after a video is regenerated in place."""
    rows = []
    for sample in samples:
        item: dict[str, Any] = {"name": sample.name, "fake": sample.fake, "gt": sample.gt}
        for label, raw in (("fake", sample.fake), ("gt", sample.gt)):
            p = Path(raw)
            try:
                st = p.stat()
                item[f"{label}_size"] = st.st_size
                item[f"{label}_mtime_ns"] = st.st_mtime_ns
            except OSError:
                item[f"{label}_size"] = None
                item[f"{label}_mtime_ns"] = None
        rows.append(item)
    return rows


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _write_per_video_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "name", "fake", "gt",
        "frames_fake", "frames_gt", "frames_paired",
        "frames_pixel_aligned", "frames_lmd_aligned", "frames_lmd_valid",
        "psnr", "ssim", "lpips", "mouth_lmd", "face_lmd",
        "psnr_ok", "ssim_ok", "lpips_ok", "lmd_ok", "error",
    ]
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _build_payload(*, fingerprint: str, samples, metrics: set[str], predictor_path: Path,
                   args, per_video: list[dict[str, Any]], failures: list[dict[str, Any]],
                   started_at: float, complete: bool) -> dict:
    aggregate, coverage = _aggregate(per_video, metrics)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "pairwise_aggregation_version": PAIRWISE_AGGREGATION_VERSION,
        "fingerprint": fingerprint,
        "protocol": {
            "psnr_ssim": (
                "EAT test_psnr_ssim temporal pairing + utils_crop_psnr; "
                "per-video valid-frame mean, then equal-weight successful-video mean"
            ),
            "lmd": (
                "EAT utils_crop preprocessing + test_lmd dlib-68; "
                "per-video valid-frame mean, then equal-weight successful-video mean"
            ),
            "lpips": (
                f"official lpips net={args.lpips_net} on EAT pixel-aligned pairs; "
                "per-video valid-frame mean, then equal-weight successful-video mean"
            ),
            "alignment": "none (diagnostic)" if args.no_align else {
                "pixel": "evaluation_eat/code/utils_crop_psnr.py",
                "lmd": "evaluation_eat/code/utils_crop.py",
            },
            "lmd_predictor": str(predictor_path) if "lmd" in metrics else None,
        },
        "n_samples": len(samples),
        "processed_n": len(per_video),
        "complete": complete,
        "coverage": coverage,
        "failures": failures,
        "aggregate": aggregate,
        "per_video": per_video,
        "elapsed_sec": time.time() - started_at,
    }


def _load_checkpoint(path: Path, fingerprint: str):
    if not path.is_file():
        return [], []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[pairwise] ignoring unreadable checkpoint: {type(exc).__name__}: {exc}", file=sys.stderr)
        return [], []
    if payload.get("pairwise_aggregation_version") != PAIRWISE_AGGREGATION_VERSION:
        print("[pairwise] ignoring checkpoint from an older aggregation protocol", file=sys.stderr)
        return [], []
    if payload.get("fingerprint") != fingerprint:
        print("[pairwise] ignoring checkpoint because manifest/config/input files changed", file=sys.stderr)
        return [], []
    rows = payload.get("per_video", [])
    failures = payload.get("failures", [])
    if not isinstance(rows, list) or not isinstance(failures, list):
        return [], []
    print(f"[pairwise] resuming checkpoint: {len(rows)} video(s) already processed")
    return rows, failures


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--manifest", required=True, help="CSV/TSV: name,fake,gt[,emotion]")
    p.add_argument("--output", required=True)
    p.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips", "lmd"],
                   choices=["psnr", "ssim", "lpips", "lmd"])
    p.add_argument("--no-align", action="store_true",
                   help="Diagnostic only. Paper protocol uses the corresponding EAT croppers.")
    p.add_argument("--lmd-predictor", default=None)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--restart", action="store_true",
                   help="Ignore and replace the resumable per-video checkpoint.")
    p.add_argument("--allow-partial", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    samples = read_manifest(args.manifest, require_files=True)
    metrics = set(args.metrics)
    t0 = time.time()

    need_pixel = bool(metrics & {"psnr", "ssim", "lpips"})
    need_lmd = "lmd" in metrics
    psnr_cropper = None
    lmd_cropper = None
    if not args.no_align:
        if need_pixel:
            psnr_cropper = _load_eat_cropper("utils_crop_psnr.py", "adef_eat_utils_crop_psnr")
        if need_lmd:
            lmd_cropper = _load_eat_cropper("utils_crop.py", "adef_eat_utils_crop_lmd")

    predictor_path = _resolve_lmd_predictor(args.lmd_predictor)
    landmark = LandmarkMetric(predictor_path) if need_lmd else None
    lpips_ctx = _load_lpips(args.device, args.lpips_net) if "lpips" in metrics else None

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_name(output.name + ".checkpoint")
    per_video_csv = output.with_name("pairwise_per_video.csv")

    fingerprint = manifest_fingerprint(
        samples,
        metrics,
        context={
            "pairwise_aggregation_version": PAIRWISE_AGGREGATION_VERSION,
            "no_align": bool(args.no_align),
            "lpips_net": args.lpips_net,
            "lmd_predictor": str(predictor_path) if need_lmd else None,
            "input_signature": _input_signature(samples),
        },
    )

    if args.restart:
        checkpoint.unlink(missing_ok=True)
        per_video_csv.unlink(missing_ok=True)

    per_video, failures = _load_checkpoint(checkpoint, fingerprint)
    valid_names = {s.name for s in samples}
    per_video = [r for r in per_video if isinstance(r, dict) and r.get("name") in valid_names]
    processed_names = {str(r.get("name")) for r in per_video}

    for sample_index, sample in enumerate(samples, start=1):
        if sample.name in processed_names:
            continue

        video_t0 = time.time()
        rec: dict[str, Any] = {"name": sample.name, "fake": sample.fake, "gt": sample.gt}
        try:
            fake_frames = _read_frames(sample.fake)
            gt_frames = _read_frames(sample.gt)
            pairs = _temporal_pairs(fake_frames, gt_frames)
            psnr_v: list[float] = []
            ssim_v: list[float] = []
            lpips_v: list[float] = []
            mouth_v: list[float] = []
            face_v: list[float] = []
            pixel_aligned_valid = 0
            lmd_aligned_valid = 0
            lmd_valid = 0

            for fake_rgb, gt_rgb in pairs:
                if need_pixel:
                    aligned = (fake_rgb, gt_rgb) if args.no_align else _aligned_pair(fake_rgb, gt_rgb, psnr_cropper)
                    if aligned is not None:
                        af, ag = aligned
                        pixel_aligned_valid += 1
                        if "psnr" in metrics or "ssim" in metrics:
                            p, s = _psnr_ssim(af, ag)
                            if "psnr" in metrics and not math.isnan(p):
                                psnr_v.append(p)
                            if "ssim" in metrics and math.isfinite(s):
                                ssim_v.append(s)
                        if "lpips" in metrics and lpips_ctx is not None:
                            v = _lpips_score(*lpips_ctx, af, ag)
                            if math.isfinite(v):
                                lpips_v.append(v)

                if landmark is not None:
                    aligned_lmd = (fake_rgb, gt_rgb) if args.no_align else _aligned_pair(fake_rgb, gt_rgb, lmd_cropper)
                    if aligned_lmd is not None:
                        lmd_aligned_valid += 1
                        lf, lg = aligned_lmd
                        m1, f1 = landmark.landmarks(lf)
                        m2, f2 = landmark.landmarks(lg)
                        md = landmark.distance(m1, m2)
                        fd = landmark.distance(f1, f2)
                        if md is not None and fd is not None:
                            mouth_v.append(md)
                            face_v.append(fd)
                            lmd_valid += 1

            rec.update({
                "frames_fake": len(fake_frames),
                "frames_gt": len(gt_frames),
                "frames_paired": len(pairs),
                "frames_pixel_aligned": pixel_aligned_valid,
                "frames_lmd_aligned": lmd_aligned_valid,
                "frames_lmd_valid": lmd_valid,
                "psnr": _mean_keep_posinf(psnr_v) if psnr_v else None,
                "ssim": float(np.mean(ssim_v)) if ssim_v else None,
                "lpips": float(np.mean(lpips_v)) if lpips_v else None,
                "mouth_lmd": float(np.mean(mouth_v)) if mouth_v else None,
                "face_lmd": float(np.mean(face_v)) if face_v else None,
            })

            metric_checks = {
                "psnr": ("psnr" not in metrics) or rec["psnr"] is not None,
                "ssim": ("ssim" not in metrics) or rec["ssim"] is not None,
                "lpips": ("lpips" not in metrics) or rec["lpips"] is not None,
                "lmd": ("lmd" not in metrics) or (rec["mouth_lmd"] is not None and rec["face_lmd"] is not None),
            }
            rec.update({f"{k}_ok": bool(v) for k, v in metric_checks.items()})
            for metric, ok in metric_checks.items():
                if metric not in metrics or ok:
                    continue
                if metric in {"psnr", "ssim", "lpips"}:
                    reason = (
                        "no valid EAT pixel-aligned frames "
                        f"(paired={len(pairs)}, aligned={pixel_aligned_valid})"
                    )
                else:
                    reason = (
                        "no valid dlib landmark frames after EAT alignment "
                        f"(paired={len(pairs)}, aligned={lmd_aligned_valid}, landmarks={lmd_valid})"
                    )
                failures.append({
                    "metric": metric,
                    "name": sample.name,
                    "fake": sample.fake,
                    "gt": sample.gt,
                    "error": reason,
                })
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            for metric in metrics:
                rec[f"{metric}_ok"] = False
                failures.append({
                    "metric": metric,
                    "name": sample.name,
                    "fake": sample.fake,
                    "gt": sample.gt,
                    "error": rec["error"],
                })

        per_video.append(rec)
        processed_names.add(sample.name)

        payload = _build_payload(
            fingerprint=fingerprint,
            samples=samples,
            metrics=metrics,
            predictor_path=predictor_path,
            args=args,
            per_video=per_video,
            failures=failures,
            started_at=t0,
            complete=False,
        )
        _atomic_json(checkpoint, payload)
        _write_per_video_csv(per_video_csv, per_video)

        values = []
        for label, key in (("PSNR", "psnr"), ("SSIM", "ssim"), ("LPIPS", "lpips"),
                           ("M-LMD", "mouth_lmd"), ("F-LMD", "face_lmd")):
            if rec.get(key) is not None:
                values.append(f"{label}={rec[key]:.6g}")
        suffix = " ".join(values) if values else "no usable metric"
        print(
            f"[pairwise] [{sample_index}/{len(samples)}] {sample.name}: {suffix} "
            f"({time.time() - video_t0:.1f}s)",
            flush=True,
        )

    payload = _build_payload(
        fingerprint=fingerprint,
        samples=samples,
        metrics=metrics,
        predictor_path=predictor_path,
        args=args,
        per_video=per_video,
        failures=failures,
        started_at=t0,
        complete=True,
    )
    _atomic_json(checkpoint, payload)
    _atomic_json(output, payload)
    _write_per_video_csv(per_video_csv, per_video)

    coverage = payload["coverage"]
    aggregate = payload["aggregate"]

    if failures:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for failure in failures:
            grouped.setdefault(str(failure["metric"]), []).append(failure)
        for metric, rows in grouped.items():
            print(
                f"[pairwise] {metric}: {coverage.get(metric, 0)}/{len(samples)} sample(s) usable",
                file=sys.stderr,
            )
            for row in rows:
                print(f"  FAIL [{metric}] {row['name']}: {row['error']}", file=sys.stderr)

    print("[pairwise] final aggregation: equal-weight mean over successful per-video values")
    for key, label in (("psnr", "PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS"),
                       ("mouth_lmd", "M-LMD"), ("face_lmd", "F-LMD")):
        stats = aggregate.get(key)
        if isinstance(stats, dict):
            print(f"[pairwise] {label}: mean={stats.get('mean')} n={stats.get('n')}")

    unusable = []
    for metric in metrics:
        if metric == "lmd":
            ok = coverage.get("lmd", 0) > 0 and aggregate["mouth_lmd"]["mean"] is not None
        else:
            ok = coverage.get(metric, 0) > 0 and aggregate[metric]["mean"] is not None
        if not ok:
            unusable.append(metric)
    if unusable:
        print(f"[pairwise] no usable result for: {', '.join(sorted(unusable))}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
