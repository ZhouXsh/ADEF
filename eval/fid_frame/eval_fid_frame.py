# coding: utf-8
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from eval.common.io import write_json

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderDataset(Dataset):
    def __init__(self, root: str):
        self.paths = sorted([p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS])
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def get_features(root: str, batch_size: int, device: str):
    dataset = ImageFolderDataset(root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    feats = []
    for x in loader:
        x = x.to(device)
        y = model(x)
        feats.append(y.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(feats1: np.ndarray, feats2: np.ndarray):
    mu1, mu2 = feats1.mean(axis=0), feats2.mean(axis=0)
    sigma1, sigma2 = np.cov(feats1, rowvar=False), np.cov(feats2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        covmean = linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(sigma2 + np.eye(sigma2.shape[0]) * eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def run_pytorch_fid(real_dir: str, gen_dir: str):
    proc = subprocess.run(["python", "-m", "pytorch_fid", real_dir, gen_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    fid = None
    for token in proc.stdout.replace(":", " ").split():
        try:
            fid = float(token)
        except ValueError:
            pass
    return {"fid": fid, "stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_pytorch_fid", action="store_true")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    if args.use_pytorch_fid:
        result = run_pytorch_fid(args.real_dir, args.gen_dir)
    else:
        f1 = get_features(args.real_dir, args.batch_size, args.device)
        f2 = get_features(args.gen_dir, args.batch_size, args.device)
        result = {
            "fid": frechet_distance(f1, f2),
            "real_count": int(f1.shape[0]),
            "gen_count": int(f2.shape[0]),
            "feature_dim": int(f1.shape[1]),
        }
    write_json(result, args.out)


if __name__ == "__main__":
    main()
