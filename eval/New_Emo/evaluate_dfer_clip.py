#!/usr/bin/env python3
"""
DFER-CLIP dynamic facial expression recognition for generated videos.

Wraps the official DFER-CLIP repo (zengqunzhao/DFER-CLIP, BMVC'23) so it can
ingest a single .mp4 file or a directory of videos. Internally:

  1. samples exactly `num_segments` frames uniformly from each video
  2. resizes each frame to 224x224 (the test transform used in main.py)
  3. feeds them through the CLIP-visual-encoder + temporal-transformer
  4. returns per-video softmax over the 7 DFEW classes

The repo expects pretrained weights for both ViT-B/32 (CLIP backbone) and the
DFEW fold-1 checkpoint. Set --clip_weights and --dfer_weights to local files.

Usage:
    # single video
    python evaluate_dfer_clip.py --video /path/to/video.mp4

    # batch
    python evaluate_dfer_clip.py --video_dir /path/to/videos/ --label_file labels.txt

    # choose fold + clip weights paths
    python evaluate_dfer_clip.py --video a.mp4 \
        --clip_weights /path/to/ViT-B-32.pt \
        --dfer_weights /path/to/DFEW_fold1.pth

    # pick a non-default GPU
    python evaluate_dfer_clip.py --video a.mp4 --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Make sure the bundled clip/ submodule is importable.
THIS_DIR = Path(__file__).resolve().parent
DFER_DIR = THIS_DIR / "DFER-CLIP"
if not DFER_DIR.is_dir():
    sys.stderr.write(f"ERROR: DFER-CLIP repo not found at {DFER_DIR}\n")
    sys.exit(2)

for p in [str(DFER_DIR), str(DFER_DIR / "models")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Suppress matplotlib "non-interactive" warnings etc.
import warnings
warnings.filterwarnings("ignore")

from clip import clip  # bundled OpenAI-style CLIP under DFER-CLIP/models/clip
from models.Generate_Model import GenerateModel


# 7 DFEW classes used by the trained checkpoint. Order matters: this is the
# order printed to the JSON output and must match the prompt order.
DFEW_CLASSES = [
    "happiness",
    "sadness",
    "neutral",
    "anger",
    "surprise",
    "disgust",
    "fear",
]

# The released fold-1 checkpoint was trained with `text-type=class_descriptor`
# (see DFER-CLIP/train_DFEW.sh), so we feed these descriptions into
# PromptLearner — not the bare class names — otherwise logits come out
# uninitialised.
DFEW_DESCRIPTORS = [
    "a smiling mouth, raised cheeks, wrinkled eyes, and arched eyebrows.",
    "tears, a downward turned mouth, drooping upper eyelids, and a wrinkled forehead.",
    "relaxed facial muscles, a straight mouth, a smooth forehead, and unremarkable eyebrows.",
    "furrowed eyebrows, narrow eyes, tightened lips, and flared nostrils.",
    "widened eyes, an open mouth, raised eyebrows, and a frozen expression.",
    "a wrinkled nose, lowered eyebrows, a tightened mouth, and narrow eyes.",
    "raised eyebrows, parted lips, a furrowed brow, and a retracted chin.",
]


def list_videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in exts and p.is_file())


def load_label_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    out: dict[str, str] = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            out[parts[0]] = " ".join(parts[1:])
    return out


def sample_frames(video_path: Path, num_segments: int) -> list[Image.Image]:
    """Uniformly sample `num_segments` PIL frames from a video."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total <= 0:
        # Fallback: read until EOF, counting
        cap.release()
        return _read_all_frames(video_path, num_segments)
    if n_total >= num_segments:
        tick = (n_total) / float(num_segments)
        indices = [int(tick / 2.0 + tick * x) for x in range(num_segments)]
    else:
        indices = list(range(n_total)) + [n_total - 1] * (num_segments - n_total)
    frames: list[Image.Image] = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if not ok:
            bgr = np.zeros((224, 224, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


def _read_all_frames(video_path: Path, num_segments: int) -> list[Image.Image]:
    """Last-resort: read every frame and pick evenly spaced ones."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    all_frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        all_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    n = len(all_frames)
    if n == 0:
        return [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))] * num_segments
    indices = np.linspace(0, n - 1, num_segments).astype(int).tolist()
    return [Image.fromarray(all_frames[i]) for i in indices]


def transform_frames(frames: list[Image.Image], image_size: int = 224) -> torch.Tensor:
    """Apply the same GroupResize -> Stack -> ToTorchFormatTensor as test_data_loader."""
    resized = [f.resize((image_size, image_size), Image.BILINEAR) for f in frames]
    arr = np.concatenate(resized, axis=2)  # H, W, T*3
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # T*3, H, W
    tensor = tensor.to(torch.float32).div_(255.0)
    tensor = tensor.view(-1, 3, image_size, image_size)  # T, 3, H, W
    return tensor


def build_args(args) -> Any:
    """Construct the argparse Namespace expected by GenerateModel / PromptLearner."""
    class Args:
        contexts_number = 8
        class_token_position = "end"
        class_specific_contexts = "True"
        load_and_tune_prompt_learner = "False"
        temporal_layers = 1
    return Args()


def load_model(clip_weights: str, dfer_weights: str, device: torch.device) -> GenerateModel:
    """Initialise DFER-CLIP model and load the fold-1 checkpoint."""
    if not Path(clip_weights).is_file():
        raise FileNotFoundError(f"CLIP weights not found: {clip_weights}")
    if not Path(dfer_weights).is_file():
        raise FileNotFoundError(f"DFER-CLIP checkpoint not found: {dfer_weights}")

    # `clip.load` will look for `clip_weights` either as a path or download from OpenAI.
    clip_model, _ = clip.load(clip_weights, device="cpu", download_root=os.path.dirname(clip_weights))

    class_names = DFEW_DESCRIPTORS
    args = build_args(None)
    model = GenerateModel(input_text=class_names, clip_model=clip_model, args=args).to(device)

    ckpt = torch.load(dfer_weights, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # strip the "module." prefix from DataParallel if present in the ckpt
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if unexpected:
        sys.stderr.write(f"[warn] unexpected keys when loading DFER-CLIP ckpt: "
                         f"{unexpected[:5]} (n_unexpected={len(unexpected)})\n")
    if missing:
        sys.stderr.write(f"[warn] missing keys when loading DFER-CLIP ckpt: "
                         f"{missing[:5]} (n_missing={len(missing)})\n")

    # Wrap with DataParallel AFTER weights are loaded — matches training-time
    # behaviour, and is required for the checkpoint's `module.*` keys to land
    # in the right place.
    model = torch.nn.DataParallel(model)
    model.eval()
    return model


@torch.no_grad()
def predict_video(model, video_path: Path, num_segments: int, device: torch.device) -> dict[str, Any]:
    """Run DFER-CLIP on a single video, return per-class probabilities + argmax."""
    frames = sample_frames(video_path, num_segments)
    tensor = transform_frames(frames).unsqueeze(0).to(device)  # 1, T, 3, 224, 224
    tensor = tensor.to(torch.float32)

    # DFER-CLIP casts to clip dtype internally; mimic by moving with model's dtype.
    # The model expects image features through CLIP's visual encoder.
    logits = model(tensor)  # 1, num_classes
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return {
        "pred_idx": pred_idx,
        "pred_emotion": DFEW_CLASSES[pred_idx],
        "probs": {c: round(float(p), 4) for c, p in zip(DFEW_CLASSES, probs)},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=str, help="Path to a single video file")
    p.add_argument("--video_dir", type=str, help="Path to a directory of videos")
    p.add_argument("--label_file", type=str,
                   help="Optional `<stem> <label>` per line for batch eval")
    p.add_argument("--label", type=str, help="Optional GT label for --video mode")
    p.add_argument("--clip_weights", type=str,
                   default=str(THIS_DIR / "weights" / "ViT-B-32.pt"),
                   help="Path to OpenAI CLIP ViT-B/32 weights (default: ./weights/ViT-B-32.pt)")
    p.add_argument("--dfer_weights", type=str,
                   default=str(THIS_DIR / "weights" / "DFEW_fold1.pth"),
                   help="Path to DFER-CLIP DFEW fold-1 checkpoint")
    p.add_argument("--device", type=str, default="cuda",
                   help="Torch device, e.g. cuda:0, cpu (default: cuda)")
    p.add_argument("--num_segments", type=int, default=16,
                   help="Frames sampled per video (must match checkpoint training; default: 16)")
    p.add_argument("--output", type=str, default=None,
                   help="Optional path to write JSON results")
    p.add_argument("--quiet", action="store_true", help="Suppress per-video progress output")
    args = p.parse_args()

    if not args.video and not args.video_dir:
        p.error("Provide --video <path> or --video_dir <dir>")

    if args.video:
        video_paths = [Path(args.video)]
        label_map = {video_paths[0].stem: args.label} if args.label else {}
    else:
        video_paths = list_videos(Path(args.video_dir))
        label_map = load_label_map(args.label_file)
    if not video_paths:
        sys.stderr.write("ERROR: no videos found\n")
        sys.exit(2)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.stderr.write("WARNING: cuda requested but not available, falling back to cpu\n")
        device = torch.device("cpu")

    print(f"[DFER-CLIP] loading model on {device} (num_segments={args.num_segments})", flush=True)
    model = load_model(args.clip_weights, args.dfer_weights, device)

    results: list[dict] = []
    n_correct = 0
    n_labelled = 0
    for i, vp in enumerate(video_paths, 1):
        lbl = label_map.get(vp.stem)
        try:
            r = predict_video(model, vp, args.num_segments, device)
        except Exception as exc:
            print(f"  !! failed on {vp}: {exc}", file=sys.stderr)
            r = {"pred_emotion": None, "probs": {}, "error": str(exc)}
        r_out = {
            "video": str(vp),
            "label": lbl,
            "prediction": r.get("pred_emotion"),
            "probs": r.get("probs", {}),
        }
        if lbl is not None and r.get("pred_emotion") is not None:
            r_out["correct"] = (lbl.lower() == r["pred_emotion"].lower())
            if r_out["correct"]:
                n_correct += 1
            n_labelled += 1
        if not args.quiet:
            print(f"[{i}/{len(video_paths)}] {vp.name} -> {r.get('pred_emotion')!r} "
                  f"probs={r.get('probs', {})} correct={r_out.get('correct')}", flush=True)
        results.append(r_out)

    overall = {
        "model": "DFER-CLIP/DFEW_fold1",
        "device": str(device),
        "num_segments": args.num_segments,
        "n_videos": len(results),
        "n_labelled": n_labelled,
        "n_correct": n_correct,
        "accuracy": round(n_correct / n_labelled, 4) if n_labelled else None,
        "results": results,
    }
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(overall, fh, indent=2)
        print(f"[DFER-CLIP] wrote {args.output}")
    else:
        print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
