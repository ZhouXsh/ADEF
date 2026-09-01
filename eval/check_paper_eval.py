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
CHECKS = {
    "Wav2Lip SyncNet-v2 weights": ROOT / "syncnet_python" / "data" / "syncnet_v2.model",
    "SyncNet S3FD weights": ROOT / "syncnet_python" / "detectors" / "s3fd" / "weights" / "sfd_face.pth",
    "EAT official utils_crop.py": ROOT / "evaluation_eat" / "code" / "utils_crop.py",
    "EAT dlib 68 predictor": ROOT / "evaluation_eat" / "code" / "shape_predictor_68_face_landmarks.dat",
    "DFER-CLIP source": ROOT / "New_Emo" / "DFER-CLIP" / "models" / "Generate_Model.py",
    "DFER-CLIP fold-1 weights": ROOT / "New_Emo" / "weights" / "DFEW_fold1.pth",
    "OpenAI CLIP ViT-B/32": ROOT / "New_Emo" / "weights" / "ViT-B-32.pt",
}
VIT_B32_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", dest="json_out")
    p.add_argument("--deep-hash", action="store_true", help="verify the OpenAI ViT-B/32 SHA256")
    args = p.parse_args()
    rows = []
    ok = True
    for name, path in CHECKS.items():
        exists = path.is_file()
        rows.append({"name": name, "path": str(path), "ok": exists})
        ok &= exists
    if args.deep_hash and CHECKS["OpenAI CLIP ViT-B/32"].is_file():
        got = sha256(CHECKS["OpenAI CLIP ViT-B/32"])
        match = got == VIT_B32_SHA256
        rows.append({"name": "ViT-B/32 SHA256", "expected": VIT_B32_SHA256, "actual": got, "ok": match})
        ok &= match
    # Git submodule metadata itself should be present so fresh clones are reproducible.
    gm = ROOT.parent / ".gitmodules"
    gm_ok = gm.is_file() and "yuangan/evaluation_eat" in gm.read_text(errors="ignore") and "face-analysis/emonet" in gm.read_text(errors="ignore")
    rows.append({"name": ".gitmodules official sources", "path": str(gm), "ok": gm_ok})
    ok &= gm_ok
    payload = {"ok": bool(ok), "checks": rows}
    for row in rows:
        print(f"[{'OK' if row['ok'] else 'MISSING'}] {row['name']}: {row.get('path', '')}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
