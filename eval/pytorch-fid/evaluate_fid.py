#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FID (Fréchet Inception Distance) evaluation script for image generation models.

This script computes the FID score between two sets of images. It is designed
to work with the local pytorch-fid package deployed in this directory.

Usage examples:
    # Compare two folders of images
    python evaluate_fid.py --path1 /path/to/real --path2 /path/to/fake

    # Save statistics from a folder for reuse
    python evaluate_fid.py --path1 /path/to/real --path2 stats.npz --save-stats

    # Use GPU and adjust batch size
    python evaluate_fid.py --path1 /path/to/real --path2 /path/to/fake \\
        --device cuda:0 --batch-size 64

    # Use a different feature dimension (e.g. 192, 768, or 2048)
    python evaluate_fid.py --path1 /path/to/real --path2 /path/to/fake --dims 2048

    # One folder against many (multi-comparison)
    python evaluate_fid.py --path1 /path/to/real --multi /path/to/fake_a /path/to/fake_b
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Ensure local pytorch_fid package is importable when running this script directly
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from pytorch_fid.fid_score import (
    calculate_fid_given_paths,
    compute_statistics_of_path,
    calculate_activation_statistics,
)
from pytorch_fid.inception import InceptionV3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID score between two image folders (or .npz stats).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path1",
        type=str,
        required=True,
        help="Path to the first image folder OR a precomputed .npz stats file.",
    )
    parser.add_argument(
        "--path2",
        type=str,
        required=True,
        help="Path to the second image folder OR a precomputed .npz stats file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size used when extracting Inception features.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu). Defaults to cuda if available else cpu.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=2048,
        choices=[64, 192, 768, 2048],
        help="Dimensionality of Inception features. Default 2048 (final pooling).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader worker processes.",
    )
    parser.add_argument(
        "--save-stats",
        action="store_true",
        help="If set, will compute & save .npz statistics for path1 (use path2 as output).",
    )
    parser.add_argument(
        "--save-stats1",
        type=str,
        default=None,
        help="If given, save path1 statistics to this .npz file.",
    )
    parser.add_argument(
        "--save-stats2",
        type=str,
        default=None,
        help="If given, save path2 statistics to this .npz file.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write the FID result as JSON.",
    )
    parser.add_argument(
        "--multi",
        type=str,
        nargs="+",
        default=None,
        help="Compare path1 against multiple folders and report each FID.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> str:
    if device_arg is not None:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_inception(device: str, dims: int) -> InceptionV3:
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    return model


def save_stats_from_path(path: str, model: InceptionV3, args, out_file: str) -> None:
    """Compute activation stats for a folder of images and save as .npz."""
    if not os.path.isdir(path):
        raise RuntimeError(f"--save-stats expects a directory, got: {path}")
    print(f"[save-stats] Computing statistics for: {path}")
    t0 = time.time()
    m, s = compute_statistics_of_path(
        path, model, args.batch_size, args.dims, args.device, args.num_workers
    )
    print(f"[save-stats] Done in {time.time() - t0:.2f}s. Saving to {out_file}")
    np = __import__("numpy")
    np.savez(out_file, mu=m, sigma=s)


def compute_fid(path1: str, path2: str, model: InceptionV3, args) -> float:
    """Compute FID between two paths (folders or .npz files)."""
    fid_value = calculate_fid_given_paths(
        [path1, path2],
        args.batch_size,
        args.device,
        args.dims,
        args.num_workers,
    )
    return float(fid_value)


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    print(f"[config] device={args.device}, dims={args.dims}, batch_size={args.batch_size}")
    print(f"[config] path1={args.path1}")
    print(f"[config] path2={args.path2}")

    # --save-stats mode: compute statistics for path1, save to path2 as .npz
    if args.save_stats:
        if not os.path.isdir(args.path1):
            raise RuntimeError(f"--save-stats expects a directory for path1, got: {args.path1}")
        model = load_inception(args.device, args.dims)
        save_stats_from_path(args.path1, model, args, args.path2)
        return

    # Validate input paths (both must exist for normal FID computation)
    for p in (args.path1, args.path2):
        if not os.path.exists(p):
            raise RuntimeError(f"Invalid path: {p}")

    # Load the inception model once and reuse for all comparisons
    model = load_inception(args.device, args.dims)

    results = []

    # Optional: cache statistics so repeated FID calls don't re-extract features
    if args.save_stats1:
        save_stats_from_path(args.path1, model, args, args.save_stats1)
    if args.save_stats2:
        save_stats_from_path(args.path2, model, args, args.save_stats2)

    t0 = time.time()
    fid = compute_fid(args.path1, args.path2, model, args)
    elapsed = time.time() - t0
    print(f"\n[result] FID({args.path1} vs {args.path2}) = {fid:.6f}  (took {elapsed:.2f}s)")
    results.append({"path1": args.path1, "path2": args.path2, "fid": fid, "elapsed_sec": elapsed})

    # Multi-comparison mode: same reference (path1) against many candidates
    if args.multi:
        for cand in args.multi:
            if not os.path.exists(cand):
                print(f"[warn] Skipping missing path: {cand}")
                continue
            t0 = time.time()
            fid_i = compute_fid(args.path1, cand, model, args)
            elapsed_i = time.time() - t0
            print(f"[result] FID({args.path1} vs {cand}) = {fid_i:.6f}  (took {elapsed_i:.2f}s)")
            results.append({"path1": args.path1, "path2": cand, "fid": fid_i, "elapsed_sec": elapsed_i})

    # Persist JSON summary if requested
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": {
                        "device": args.device,
                        "dims": args.dims,
                        "batch_size": args.batch_size,
                        "num_workers": args.num_workers,
                    },
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[result] Saved JSON summary to {args.output_json}")


if __name__ == "__main__":
    main()
