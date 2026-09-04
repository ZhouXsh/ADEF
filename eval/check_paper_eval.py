#!/usr/bin/env python3
"""Preflight checker for ADEF paper evaluation dependencies and assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIT_B32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

DEFAULT_EVAL_PY = Path("/home/Zhouxishi/miniconda3/envs/eval/bin/python")
DEFAULT_FVD_PY = Path("/home/Zhouxishi/miniconda3/envs/fvd/bin/python")
DEFAULT_LSE_PY = ROOT / "Wav2Lip" / "evaluation" / "venv" / "bin" / "python"
DEFAULT_PAIRWISE_PY = ROOT / "evaluation_eat" / "venv" / "bin" / "python"
DEFAULT_SYNCNET_PIPELINE_PY = ROOT / "syncnet_python" / "syncnet_venv" / "bin" / "python"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def _python(preferred: Path) -> Path:
    return preferred if preferred.is_file() else Path(sys.executable)


def _check_imports(python: Path, modules: list[str]) -> tuple[bool, str]:
    code = "; ".join(f"import {m}" for m in modules)
    try:
        proc = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=60)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode == 0:
        return True, ""
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, " | ".join(tail[-3:])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", dest="json_out")
    p.add_argument("--deep-hash", action="store_true", help="verify the OpenAI ViT-B/32 SHA256")
    p.add_argument("--with-emonet", action="store_true")
    args = p.parse_args()

    eat_predictor = _first_existing(
        ROOT / "evaluation_eat" / "code" / "shape_predictor_68_face_landmarks.dat",
        ROOT / "evaluation_eat" / "checkpoints" / "shape_predictor_68_face_landmarks.dat",
    )
    checks = {
        "Wav2Lip SyncNet-v2 weights": ROOT / "syncnet_python" / "data" / "syncnet_v2.model",
        "SyncNet S3FD weights": ROOT / "syncnet_python" / "detectors" / "s3fd" / "weights" / "sfd_face.pth",
        "EAT preprocess crop helper": ROOT / "evaluation_eat" / "code" / "utils_crop.py",
        "EAT PSNR/SSIM crop helper": ROOT / "evaluation_eat" / "code" / "utils_crop_psnr.py",
        "EAT base_68.npy": ROOT / "evaluation_eat" / "code" / "base_68.npy",
        "EAT base_68_close.npy": ROOT / "evaluation_eat" / "code" / "base_68_close.npy",
        "EAT dlib 68 predictor": eat_predictor,
        "DFER-CLIP source": ROOT / "New_Emo" / "DFER-CLIP" / "models" / "Generate_Model.py",
        "DFER-CLIP fold-1 weights": ROOT / "New_Emo" / "weights" / "DFEW_fold1.pth",
        "OpenAI CLIP ViT-B/32": ROOT / "New_Emo" / "weights" / "ViT-B-32.pt",
        "Vendored EAT preprocess": ROOT / "evaluation_eat" / "code" / "preprocess.py",
        "Vendored EAT PSNR/SSIM reference": ROOT / "evaluation_eat" / "code" / "test_psnr_ssim.py",
        "Vendored EAT LMD reference": ROOT / "evaluation_eat" / "code" / "test_lmd.py",
        "Vendored EmoNet source": ROOT / "emonet" / "emonet" / "models" / "__init__.py",
    }
    if args.with_emonet:
        checks["EmoNet 8-class weights"] = ROOT / "emonet" / "pretrained" / "emonet_8.pth"

    rows = []
    ok = True
    for name, path in checks.items():
        exists = path.is_file()
        rows.append({"name": name, "path": str(path), "ok": exists})
        ok &= exists

    if args.deep_hash and checks["OpenAI CLIP ViT-B/32"].is_file():
        got = sha256(checks["OpenAI CLIP ViT-B/32"])
        match = got == VIT_B32_SHA256
        rows.append({"name": "ViT-B/32 SHA256", "expected": VIT_B32_SHA256, "actual": got, "ok": match})
        ok &= match

    eval_py = _python(DEFAULT_EVAL_PY)
    fvd_py = _python(DEFAULT_FVD_PY)
    lse_py = _python(DEFAULT_LSE_PY)
    # Match paper_evaluator.py exactly: pairwise falls back to eval Python,
    # while SyncNet run_pipeline falls back to the LSE evaluator interpreter.
    pairwise_py = DEFAULT_PAIRWISE_PY if DEFAULT_PAIRWISE_PY.is_file() else eval_py
    syncnet_pipeline_py = DEFAULT_SYNCNET_PIPELINE_PY if DEFAULT_SYNCNET_PIPELINE_PY.is_file() else lse_py

    env_checks = [
        ("Pairwise Python imports", pairwise_py,
         ["cv2", "numpy", "skimage", "dlib", "imutils", "lpips", "torch"]),
        ("Eval Python imports", eval_py,
         ["cv2", "numpy", "torch", "PIL", "torchvision", "facenet_pytorch", "emotiefflib"]),
        ("FVD Python imports", fvd_py,
         ["numpy", "tensorflow", "tensorflow_hub"]),
        ("LSE Python imports", lse_py, ["cv2", "numpy", "torch"]),
        ("SyncNet pipeline Python imports", syncnet_pipeline_py,
         ["cv2", "numpy", "torch"]),
    ]
    for name, interpreter, modules in env_checks:
        import_ok, detail = _check_imports(interpreter, modules)
        rows.append({"name": name, "path": str(interpreter), "modules": modules,
                     "detail": detail, "ok": import_ok})
        ok &= import_ok

    payload = {"ok": bool(ok), "checks": rows}
    for row in rows:
        suffix = f" ({row['detail']})" if row.get("detail") else ""
        print(f"[{'OK' if row['ok'] else 'MISSING'}] {row['name']}: {row.get('path', '')}{suffix}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
