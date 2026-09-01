#!/usr/bin/env python3
"""Strict Emotion-FAN inference for *pre-aligned face frames*.

The official Emotion-FAN repository publishes a FER+ backbone but not a
ready-to-use AFEW FAN checkpoint.  A FAN attention model without an AFEW
fine-tuned checkpoint has randomly initialised alpha/beta/classifier layers and
must not be used as an evaluation metric.  This wrapper therefore requires an
explicit trained FAN checkpoint and follows the official AFEW validation
aggregation exactly.

Input must be a directory of face-aligned image frames produced by the official
Emotion-FAN preprocessing.  Raw videos are intentionally rejected rather than
silently resized as whole frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from basic_code import networks  # noqa: E402

ID2NAME = {0: "Happy", 1: "Angry", 2: "Disgust", 3: "Fear", 4: "Sad", 5: "Neutral", 6: "Surprise"}
TRANSFORM = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])


def _state(path: Path):
    obj = torch.load(path, map_location="cpu")
    sd = obj.get("state_dict", obj)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def load_model(pretrain: Path, checkpoint: Path, at_type: int, device: torch.device):
    if not pretrain.is_file():
        raise FileNotFoundError(f"FER+ backbone not found: {pretrain}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"trained AFEW Emotion-FAN checkpoint is REQUIRED: {checkpoint}. "
            "Do not evaluate at_type=0/1 with FER+ backbone only."
        )
    name = ["self-attention", "self_relation-attention"][at_type]
    model = networks.resnet18_at(at_type=name)
    # Official load.model_parameters loads FER+ features but excludes its fc head.
    base = _state(pretrain)
    own = model.state_dict()
    for k, v in base.items():
        if k in ("fc.weight", "fc.bias"):
            continue
        if k in own and own[k].shape == v.shape:
            own[k] = v
    model.load_state_dict(own, strict=True)
    # The FAN checkpoint must define attention + prediction heads; strict load
    # catches accidental backbone-only or wrong-architecture checkpoints.
    model.load_state_dict(_state(checkpoint), strict=True)
    return model.to(device).eval()


def load_frames(folder: Path) -> torch.Tensor:
    if not folder.is_dir():
        raise ValueError(
            "Emotion-FAN paper protocol requires a directory of face-aligned frames. "
            "Run the official data/face_alignment_code preprocessing first."
        )
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not files:
        raise RuntimeError(f"no aligned face frames: {folder}")
    return torch.stack([TRANSFORM(Image.open(p).convert("RGB")) for p in files])


@torch.no_grad()
def predict(model, frames: torch.Tensor, at_type: int, device: torch.device, batch_size: int):
    feats, alphas = [], []
    for i in range(0, len(frames), batch_size):
        f, a = model(frames[i:i+batch_size].to(device), phrase="eval")
        feats.append(f); alphas.append(a)
    f = torch.cat(feats, 0); a = torch.cat(alphas, 0)
    index = torch.ones(1, len(f), device=device)
    weighted = f.mul(a)
    vm = index.mm(weighted).div(index.mm(a) + 1e-12)
    if at_type == 0:
        logits = model(vm=vm, phrase="eval", AT_level="pred")
    else:
        logits = model(vectors=f, vm=vm, alphas_from1=a, index_matrix=index,
                       phrase="eval", AT_level="second_level")
    probs = F.softmax(logits.squeeze(0), dim=0)
    idx = int(probs.argmax().item())
    return {
        "emotion_id": idx, "emotion": ID2NAME[idx], "confidence": float(probs[idx]),
        "probabilities": {ID2NAME[i]: float(probs[i]) for i in range(7)},
        "n_frames_used": int(len(frames)),
    }


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="directory of official face-aligned frames")
    p.add_argument("--pretrain_fer", default=str(THIS_DIR / "pretrain_model" / "Resnet18_FER+_pytorch.pth.tar"))
    p.add_argument("--checkpoint", required=True, help="trained AFEW FAN checkpoint")
    p.add_argument("--at_type", type=int, default=1, choices=[0, 1])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.pretrain_fer), Path(args.checkpoint), args.at_type, device)
    result = predict(model, load_frames(Path(args.input)), args.at_type, device, args.batch_size)
    payload = {str(Path(args.input)): result}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out_json:
        Path(args.out_json).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
