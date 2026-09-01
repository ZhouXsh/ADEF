#!/usr/bin/env python3
"""DFER-CLIP video emotion evaluation using the official BMVC'23 model.

Important protocol note: DFEW/DFER-CLIP has seven classes and does not contain
MEAD's ``contempt`` category.  Unsupported target labels are therefore marked
``label_supported=false`` and excluded from the accuracy denominator instead
of being silently counted as errors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
DFER_DIR = THIS_DIR / "DFER-CLIP"
if not DFER_DIR.is_dir():
    raise SystemExit(f"DFER-CLIP repo not found: {DFER_DIR}")
for p in (DFER_DIR, DFER_DIR / "models"):
    sys.path.insert(0, str(p))
from clip import clip  # noqa: E402
from models.Generate_Model import GenerateModel  # noqa: E402
from dataloader.video_transform import GroupResize, Stack, ToTorchFormatTensor  # noqa: E402

DFEW_CLASSES = ["happiness", "sadness", "neutral", "anger", "surprise", "disgust", "fear"]
DFEW_DESCRIPTORS = [
    "a smiling mouth, raised cheeks, wrinkled eyes, and arched eyebrows.",
    "tears, a downward turned mouth, drooping upper eyelids, and a wrinkled forehead.",
    "relaxed facial muscles, a straight mouth, a smooth forehead, and unremarkable eyebrows.",
    "furrowed eyebrows, narrow eyes, tightened lips, and flared nostrils.",
    "widened eyes, an open mouth, raised eyebrows, and a frozen expression.",
    "a wrinkled nose, lowered eyebrows, a tightened mouth, and narrow eyes.",
    "raised eyebrows, parted lips, a furrowed brow, and a retracted chin.",
]
ALIASES = {
    "happy": "happiness", "happiness": "happiness",
    "sad": "sadness", "sadness": "sadness",
    "angry": "anger", "anger": "anger",
    "disgusted": "disgust", "disgust": "disgust",
    "surprised": "surprise", "surprise": "surprise",
    "fear": "fear", "neutral": "neutral", "calm": "neutral",
    "contempt": "contempt",
}


def canonical_label(label: str | None) -> str | None:
    if not label:
        return None
    s = label.strip().lower()
    return ALIASES.get(s, s)


def list_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def load_label_map(path: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path:
        return out
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split(maxsplit=1)
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def sample_frames(path: Path, nseg: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        raw = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            raw.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not raw:
            raise RuntimeError(f"empty video: {path}")
        ids = np.linspace(0, len(raw) - 1, nseg).astype(int)
        return [Image.fromarray(raw[i]) for i in ids]
    tick = n / float(nseg)
    if n >= nseg:
        ids = [min(n - 1, int(tick / 2.0 + tick * x)) for x in range(nseg)]
    else:
        ids = list(range(n)) + [n - 1] * (nseg - n)
    frames = []
    for idx in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"failed to decode frame {idx}: {path}")
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def transform_frames(frames: list[Image.Image], size: int = 224) -> torch.Tensor:
    # Use the bundled upstream transforms verbatim.  In particular,
    # torchvision Resize(int) semantics are preserved instead of replacing it
    # with an ad-hoc square resize.
    resized = GroupResize(size)(frames)
    stacked = Stack()(resized)
    tensor = ToTorchFormatTensor()(stacked)
    try:
        return torch.reshape(tensor, (-1, 3, size, size))
    except RuntimeError as exc:
        raise RuntimeError(
            "DFER-CLIP expects the same square face-frame geometry as its official DFEW loader; "
            f"post-resize tensor shape is {tuple(tensor.shape)}"
        ) from exc


class ModelArgs:
    contexts_number = 8
    class_token_position = "end"
    class_specific_contexts = "True"
    load_and_tune_prompt_learner = "False"
    temporal_layers = 1


def load_model(clip_weights: str, dfer_weights: str, device: torch.device):
    if not Path(clip_weights).is_file():
        raise FileNotFoundError(f"CLIP ViT-B/32 weights not found: {clip_weights}")
    if not Path(dfer_weights).is_file():
        raise FileNotFoundError(f"DFER-CLIP checkpoint not found: {dfer_weights}")
    clip_model, _ = clip.load(clip_weights, device="cpu", download_root=str(Path(clip_weights).parent))
    model = GenerateModel(input_text=DFEW_DESCRIPTORS, clip_model=clip_model, args=ModelArgs()).to(device)
    ckpt = torch.load(dfer_weights, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    # Official main.py loads the saved model state strictly.  Training wraps
    # GenerateModel in DataParallel, so only the `module.` prefix is removed.
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict(model, path: Path, nseg: int, device: torch.device) -> dict[str, Any]:
    x = transform_frames(sample_frames(path, nseg)).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
    idx = int(np.argmax(probs))
    return {
        "prediction": DFEW_CLASSES[idx],
        "probs": {c: float(p) for c, p in zip(DFEW_CLASSES, probs)},
    }


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--video_dir")
    p.add_argument("--label")
    p.add_argument("--label_file")
    p.add_argument("--clip_weights", default=str(THIS_DIR / "weights" / "ViT-B-32.pt"))
    p.add_argument("--dfer_weights", default=str(THIS_DIR / "weights" / "DFEW_fold1.pth"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_segments", type=int, default=16)
    p.add_argument("--output")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = [Path(args.video)] if args.video else list_videos(Path(args.video_dir))
    if not paths:
        print("ERROR: no videos found", file=sys.stderr); return 2
    labels = load_label_map(args.label_file)
    if args.video and args.label:
        labels[paths[0].stem] = args.label
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(args.clip_weights, args.dfer_weights, device)

    results = []
    supported_labelled = 0
    correct = 0
    for i, vp in enumerate(paths, 1):
        if not args.quiet:
            print(f"[DFER-CLIP] [{i}/{len(paths)}] {vp}")
        label = canonical_label(labels.get(vp.stem))
        supported = label is None or label in DFEW_CLASSES
        try:
            pred = predict(model, vp, args.num_segments, device)
            row = {"video": str(vp), "label": label, "label_supported": supported, **pred}
            if label is not None:
                if supported:
                    row["correct"] = pred["prediction"] == label
                    row["target_probability"] = pred["probs"].get(label)
                    supported_labelled += 1
                    correct += int(row["correct"])
                else:
                    row["correct"] = None
                    row["target_probability"] = None
                    row["unsupported_reason"] = f"{label!r} is not a DFEW/DFER-CLIP class"
            results.append(row)
        except Exception as exc:
            results.append({"video": str(vp), "label": label, "label_supported": supported,
                            "prediction": None, "probs": {}, "correct": None,
                            "error": f"{type(exc).__name__}: {exc}"})

    payload = {
        "model": "DFER-CLIP DFEW fold-1",
        "classes": DFEW_CLASSES,
        "num_segments": args.num_segments,
        "n_videos": len(results),
        "n_labelled_supported": supported_labelled,
        "n_correct": correct,
        "accuracy": (correct / supported_labelled) if supported_labelled else None,
        "unsupported_labels": sorted({r["label"] for r in results if r.get("label") and not r.get("label_supported")}),
        "results": results,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    if not args.quiet:
        print(text)
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
