#!/usr/bin/env python3
"""Paired PSNR/SSIM/LPIPS/LMD evaluation with per-metric partial coverage.

PSNR/SSIM use the vendored EAT ``utils_crop_psnr.crop_and_align`` path.
LMD uses EAT ``utils_crop.crop_and_align`` followed by the official dlib-68
landmark definition. LPIPS is evaluated on the same aligned pairs as
PSNR/SSIM. Failed samples are excluded only from the metric(s) they failed;
all successful samples still contribute to the aggregate.
"""
from __future__ import annotations

import argparse
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
from paper_protocol import PROTOCOL_VERSION, read_manifest, summarize  # noqa: E402

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


def _summarize_psnr(values: list[float]) -> dict:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "std": None}
    if any(math.isinf(v) and v > 0 for v in vals):
        return {"n": len(vals), "mean": float("inf"), "std": None}
    return summarize(vals)


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

    all_psnr: list[float] = []
    all_ssim: list[float] = []
    all_lpips: list[float] = []
    video_mouth_lmd: list[float] = []
    video_face_lmd: list[float] = []
    per_video: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for sample in samples:
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
                                all_psnr.append(p)
                            if "ssim" in metrics and math.isfinite(s):
                                ssim_v.append(s)
                                all_ssim.append(s)
                        if "lpips" in metrics and lpips_ctx is not None:
                            v = _lpips_score(*lpips_ctx, af, ag)
                            if math.isfinite(v):
                                lpips_v.append(v)
                                all_lpips.append(v)

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
                "psnr": float(np.mean(psnr_v)) if psnr_v else None,
                "ssim": float(np.mean(ssim_v)) if ssim_v else None,
                "lpips": float(np.mean(lpips_v)) if lpips_v else None,
                "mouth_lmd": float(np.mean(mouth_v)) if mouth_v else None,
                "face_lmd": float(np.mean(face_v)) if face_v else None,
            })

            if mouth_v:
                video_mouth_lmd.append(rec["mouth_lmd"])
                video_face_lmd.append(rec["face_lmd"])

            metric_checks = {
                "psnr": ("psnr" not in metrics) or rec["psnr"] is not None,
                "ssim": ("ssim" not in metrics) or rec["ssim"] is not None,
                "lpips": ("lpips" not in metrics) or rec["lpips"] is not None,
                "lmd": ("lmd" not in metrics) or (rec["mouth_lmd"] is not None and rec["face_lmd"] is not None),
            }
            rec.update({f"{k}_ok": bool(v) for k, v in metric_checks.items()})
            for metric, ok in metric_checks.items():
                if metric in metrics and not ok:
                    if metric in {"psnr", "ssim", "lpips"}:
                        reason = f"no valid EAT pixel-aligned frames (paired={len(pairs)}, aligned={pixel_aligned_valid})"
                    else:
                        reason = (
                            "no valid dlib landmark frames after EAT alignment "
                            f"(paired={len(pairs)}, aligned={lmd_aligned_valid}, landmarks={lmd_valid})"
                        )
                    failures.append({"metric": metric, "name": sample.name, "fake": sample.fake,
                                     "gt": sample.gt, "error": reason})
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            for metric in metrics:
                rec[f"{metric}_ok"] = False
                failures.append({"metric": metric, "name": sample.name, "fake": sample.fake,
                                 "gt": sample.gt, "error": rec["error"]})
        per_video.append(rec)

    aggregate = {
        "psnr": _summarize_psnr(all_psnr) if "psnr" in metrics else None,
        "ssim": summarize(all_ssim) if "ssim" in metrics else None,
        "lpips": summarize(all_lpips) if "lpips" in metrics else None,
        "mouth_lmd": summarize(video_mouth_lmd) if need_lmd else None,
        "face_lmd": summarize(video_face_lmd) if need_lmd else None,
    }
    coverage = {
        "psnr": sum(bool(r.get("psnr_ok")) for r in per_video) if "psnr" in metrics else None,
        "ssim": sum(bool(r.get("ssim_ok")) for r in per_video) if "ssim" in metrics else None,
        "lpips": sum(bool(r.get("lpips_ok")) for r in per_video) if "lpips" in metrics else None,
        "lmd": sum(bool(r.get("lmd_ok")) for r in per_video) if "lmd" in metrics else None,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": {
            "psnr_ssim": "EAT test_psnr_ssim temporal pairing + utils_crop_psnr; global valid-frame mean",
            "lmd": "EAT utils_crop preprocessing + test_lmd dlib-68; per-video mean then dataset mean",
            "lpips": f"official lpips net={args.lpips_net} on EAT pixel-aligned pairs",
            "alignment": "none (diagnostic)" if args.no_align else {
                "pixel": "evaluation_eat/code/utils_crop_psnr.py",
                "lmd": "evaluation_eat/code/utils_crop.py",
            },
            "lmd_predictor": str(predictor_path) if need_lmd else None,
        },
        "n_samples": len(samples),
        "coverage": coverage,
        "failures": failures,
        "aggregate": aggregate,
        "per_video": per_video,
        "elapsed_sec": time.time() - t0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if failures:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for failure in failures:
            grouped.setdefault(str(failure["metric"]), []).append(failure)
        for metric, rows in grouped.items():
            print(f"[pairwise] {metric}: {coverage.get(metric, 0)}/{len(samples)} sample(s) usable", file=sys.stderr)
            for row in rows:
                print(f"  FAIL [{metric}] {row['name']}: {row['error']}", file=sys.stderr)

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
