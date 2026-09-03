#!/usr/bin/env python3
"""Preflight checker for ADEF paper evaluation dependencies and assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

VIT_B32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", dest="json_out")
    p.add_argument("--deep-hash", action="store_true", help="verify the OpenAI ViT-B/32 SHA256")
    p.add_argument(
        "--with-emonet", action="store_true",
        help="also require the optional EmoNet 8-class checkpoint",
    )
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

    # evaluation_eat and emonet are intentionally vendored as normal directories
    # in this repository. Check the actual files used by the wrappers instead of
    # requiring submodule metadata.
    vendored_checks = {
        "Vendored EAT preprocess": ROOT / "evaluation_eat" / "code" / "preprocess.py",
        "Vendored EAT PSNR/SSIM reference": ROOT / "evaluation_eat" / "code" / "test_psnr_ssim.py",
        "Vendored EAT LMD reference": ROOT / "evaluation_eat" / "code" / "test_lmd.py",
        "Vendored EmoNet source": ROOT / "emonet" / "emonet" / "models" / "__init__.py",
    }
    for name, path in vendored_checks.items():
        exists = path.is_file()
        rows.append({"name": name, "path": str(path), "ok": exists})
        ok &= exists

    payload = {"ok": bool(ok), "checks": rows}
    for row in rows:
        print(f"[{'OK' if row['ok'] else 'MISSING'}] {row['name']}: {row.get('path', '')}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
