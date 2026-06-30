# coding: utf-8
from __future__ import annotations

import argparse

import numpy as np
import torch
from torchvision import transforms

from eval.common.io import iter_video_paths, read_video_frames, summarize, write_json


def load_lpips(device: str):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("Please install lpips: pip install lpips") from exc
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def to_tensor(frame_rgb, device):
    tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tr(frame_rgb).unsqueeze(0).to(device)


@torch.no_grad()
def evaluate_video(model, video_path: str, device: str, stride: int = 1, max_frames: int = 64):
    frames = read_video_frames(video_path, max_frames=max_frames, stride=stride, rgb=True)
    scores = []
    for a, b in zip(frames[:-1], frames[1:]):
        aa = to_tensor(a, device)
        bb = to_tensor(b, device)
        scores.append(float(model(aa, bb).item()))
    return {
        "video": video_path,
        "sampled_frames": len(frames),
        "temporal_lpips": summarize(scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    model = load_lpips(args.device)
    rows = [evaluate_video(model, (row.get("generated") or row.get("video")), args.device, args.stride, args.max_frames) for row in iter_video_paths(args.video or None, args.manifest or None)]
    summary = {
        "temporal_lpips_mean": summarize(r["temporal_lpips"]["mean"] for r in rows),
        "temporal_lpips_std": summarize(r["temporal_lpips"]["std"] for r in rows),
    }
    write_json({"summary": summary, "items": rows}, args.out)


if __name__ == "__main__":
    main()
