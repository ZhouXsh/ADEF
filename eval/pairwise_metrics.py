#!/usr/bin/env python3
"""Paper-grade paired image/geometry metrics for talking-head videos.

PSNR/SSIM and landmark distance follow the definitions and temporal matching
used by the official EAT evaluation repository (yuangan/evaluation_eat).
Unlike the upstream scripts, this wrapper accepts an explicit manifest, so
correctness does not depend on method-specific file-name slicing or hard-coded
MEAD directories.

LPIPS uses the official ``lpips`` package (AlexNet backbone by default) on the
same aligned and temporally paired frames.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
EAT_CODE = THIS_DIR / "evaluation_eat" / "code"
EAT_CHECKPOINTS = THIS_DIR / "evaluation_eat" / "checkpoints"
sys.path.insert(0, str(THIS_DIR))
from paper_protocol import PROTOCOL_VERSION, read_manifest, summarize  # noqa: E402


def _read_frames(path: str) -> list[np.ndarray]:
    """Decode video as RGB uint8 frames, matching imageio's EAT convention."""
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
    """EAT temporal equalisation: linspace both clips to min length."""
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


def _load_eat_cropper():
    path = EAT_CODE / "utils_crop.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"EAT crop/alignment helper is missing: {path}. "
            "Restore eval/evaluation_eat/code/utils_crop.py from yuangan/evaluation_eat."
        )
    spec = importlib.util.spec_from_file_location("adef_eat_utils_crop_psnr", path)
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
        # Keep the EAT call convention while allowing the vendored helper to
        # resolve its resources relative to its own file.
        cwd = os.getcwd()
        try:
            os.chdir(EAT_CODE)
            return mod.crop_and_align(image)
        finally:
            os.chdir(cwd)
    return official_crop


def _aligned_pair(fake_rgb: np.ndarray, gt_rgb: np.ndarray, cropper):
    # EAT test_psnr_ssim.py calls crop_and_align on imageio RGB frames.
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
    except TypeError:  # old scikit-image used by the original EAT environment
        ssim = float(structural_similarity(gt, fake, data_range=255, multichannel=True))
    return psnr, ssim


class LandmarkMetric:
    """Official EAT 68-point / 20-mouth landmark distance definition."""

    def __init__(self, predictor_path: Path):
        try:
            import dlib
            from imutils import face_utils
        except ImportError as exc:
            raise RuntimeError("dlib and imutils are required for EAT LMD") from exc
        if not predictor_path.is_file():
            raise FileNotFoundError(
                f"EAT landmark predictor not found: {predictor_path}. "
                "Place shape_predictor_68_face_landmarks.dat under "
                "eval/evaluation_eat/code/ or eval/evaluation_eat/checkpoints/."
            )
        self.dlib = dlib
        self.face_utils = face_utils
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(predictor_path))
        self.mouth_start, self.mouth_end = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    def landmarks(self, img: np.ndarray):
        # This intentionally matches EAT test_lmd.py. The upstream code decodes
        # with imageio then uses COLOR_BGR2GRAY; preserving that convention is
        # more important for comparability than changing its channel order.
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
        # EAT's ld() sums point distances then divides by landmark count.
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
    # lpips.im2tensor expects HWC RGB uint8 and maps [0,255] -> [-1,1].
    a = lpips_mod.im2tensor(fake).to(device)
    b = lpips_mod.im2tensor(gt).to(device)
    with torch_mod.no_grad():
        return float(model(a, b).reshape(-1)[0].item())


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--manifest", required=True, help="CSV/TSV: name,fake,gt[,emotion]")
    p.add_argument("--output", required=True)
    p.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips", "lmd"],
                   choices=["psnr", "ssim", "lpips", "lmd"])
    p.add_argument("--no-align", action="store_true",
                   help="Diagnostic only. Paper protocol aligns both videos with EAT cropper.")
    p.add_argument(
        "--lmd-predictor", default=None,
        help="Optional dlib predictor path. By default code/ and checkpoints/ are searched.",
    )
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-partial", action="store_true",
                   help="Allow samples/frames to fail. Default is strict for paper use.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    samples = read_manifest(args.manifest, require_files=True)
    metrics = set(args.metrics)
    t0 = time.time()

    cropper = None if args.no_align else _load_eat_cropper()
    predictor_path = _resolve_lmd_predictor(args.lmd_predictor)
    landmark = LandmarkMetric(predictor_path) if "lmd" in metrics else None
    lpips_ctx = _load_lpips(args.device, args.lpips_net) if "lpips" in metrics else None

    all_psnr: list[float] = []
    all_ssim: list[float] = []
    all_lpips: list[float] = []
    video_mouth_lmd: list[float] = []
    video_face_lmd: list[float] = []
    per_video: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

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
            aligned_valid = 0
            lmd_valid = 0

            for fake_rgb, gt_rgb in pairs:
                aligned = (fake_rgb, gt_rgb) if cropper is None else _aligned_pair(fake_rgb, gt_rgb, cropper)
                if aligned is not None:
                    af, ag = aligned
                    aligned_valid += 1
                    if "psnr" in metrics or "ssim" in metrics:
                        p, s = _psnr_ssim(af, ag)
                        if "psnr" in metrics and math.isfinite(p):
                            psnr_v.append(p); all_psnr.append(p)
                        if "ssim" in metrics and math.isfinite(s):
                            ssim_v.append(s); all_ssim.append(s)
                    if "lpips" in metrics and lpips_ctx is not None:
                        v = _lpips_score(*lpips_ctx, af, ag)
                        if math.isfinite(v):
                            lpips_v.append(v); all_lpips.append(v)

                if landmark is not None and aligned is not None:
                    # Official EAT test_lmd.py operates on the same pre-aligned
                    # 128x128 videos produced by preprocess.py.
                    lf, lg = aligned
                    m1, f1 = landmark.landmarks(lf)
                    m2, f2 = landmark.landmarks(lg)
                    md = landmark.distance(m1, m2)
                    fd = landmark.distance(f1, f2)
                    if md is not None and fd is not None:
                        mouth_v.append(md); face_v.append(fd); lmd_valid += 1

            rec.update({
                "frames_fake": len(fake_frames), "frames_gt": len(gt_frames),
                "frames_paired": len(pairs), "frames_aligned": aligned_valid,
                "frames_lmd_valid": lmd_valid,
                "psnr": float(np.mean(psnr_v)) if psnr_v else None,
                "ssim": float(np.mean(ssim_v)) if ssim_v else None,
                "lpips": float(np.mean(lpips_v)) if lpips_v else None,
                "mouth_lmd": float(np.mean(mouth_v)) if mouth_v else None,
                "face_lmd": float(np.mean(face_v)) if face_v else None,
            })
            # EAT test_lmd.py averages per-video LMD, not all frames globally.
            if mouth_v:
                video_mouth_lmd.append(float(np.mean(mouth_v)))
            if face_v:
                video_face_lmd.append(float(np.mean(face_v)))

            required_ok = True
            if ("psnr" in metrics or "ssim" in metrics or "lpips" in metrics) and aligned_valid == 0:
                required_ok = False
            if "lmd" in metrics and lmd_valid == 0:
                required_ok = False
            rec["ok"] = required_ok
            if not required_ok:
                failures.append({"name": sample.name, "error": "no valid frames for one or more requested metrics"})
        except Exception as exc:  # keep a complete audit trail
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
            failures.append({"name": sample.name, "error": rec["error"]})
        per_video.append(rec)

    aggregate = {
        "psnr": summarize(all_psnr) if "psnr" in metrics else None,
        "ssim": summarize(all_ssim) if "ssim" in metrics else None,
        "lpips": summarize(all_lpips) if "lpips" in metrics else None,
        "mouth_lmd": summarize(video_mouth_lmd) if "lmd" in metrics else None,
        "face_lmd": summarize(video_face_lmd) if "lmd" in metrics else None,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": {
            "psnr_ssim": "EAT official preprocess crop_and_align + temporal pairing; global valid-frame mean",
            "lmd": "EAT official dlib-68 definition; per-video mean then dataset mean",
            "lpips": f"official lpips package, net={args.lpips_net}, same EAT-aligned frame pairs",
            "alignment": "none (diagnostic)" if args.no_align else "evaluation_eat/code/utils_crop.py",
            "lmd_predictor": str(predictor_path) if "lmd" in metrics else None,
        },
        "n_samples": len(samples),
        "n_success": sum(bool(x.get("ok")) for x in per_video),
        "failures": failures,
        "aggregate": aggregate,
        "per_video": per_video,
        "elapsed_sec": time.time() - t0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures and not args.allow_partial:
        print(f"[pairwise] strict failure: {len(failures)}/{len(samples)} sample(s) incomplete", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
