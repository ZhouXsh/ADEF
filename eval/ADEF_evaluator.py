#!/usr/bin/env python3
"""Evaluate one ADEF experiment directory with the paper-grade protocol.

Pairs file format: ``fake_filename,gt_fullpath[,emotion]``.  The fake filename
is resolved below ``<father-dir>/<exam_name>``.  A method-level manifest is
built and passed to ``paper_evaluator.py`` once, so all experiments use the
same sample set and dataset-level Fréchet metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from paper_protocol import Sample, canonical_emotion, infer_emotion, write_manifest  # noqa: E402

PAPER_EVALUATOR = THIS_DIR / "paper_evaluator.py"
DEFAULT_FATHER = Path("/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake")
DEFAULT_SUMMARY = DEFAULT_FATHER / "summary.csv"


def read_pairs(path: Path, fake_root: Path) -> list[Sample]:
    if not path.is_file():
        raise FileNotFoundError(path)
    out = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = [x.strip() for x in s.split(",")]
        if len(parts) not in (2, 3):
            raise ValueError(f"{path}:{lineno}: expected fake,gt[,emotion]")
        fake, gt = fake_root / parts[0], Path(parts[1])
        if not fake.is_file():
            raise FileNotFoundError(f"{path}:{lineno}: fake not found: {fake}")
        if not gt.is_file():
            raise FileNotFoundError(f"{path}:{lineno}: gt not found: {gt}")
        emo = canonical_emotion(parts[2]) if len(parts) == 3 and parts[2] else infer_emotion(gt)
        out.append(Sample(name=Path(parts[0]).stem, fake=str(fake), gt=str(gt), emotion=emo))
    if not out:
        raise ValueError(f"no usable pairs in {path}")
    return out


def update_summary(summary: Path, row: dict):
    rows = []
    fieldnames = list(row.keys())
    if summary.is_file():
        try:
            with summary.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # Old pre-v2 summary schemas are intentionally not merged: they
                # used per-video FID/FVD and are not paper-comparable.
                if reader.fieldnames and "Protocol" in reader.fieldnames and "Status" in reader.fieldnames:
                    rows = [r for r in reader if r.get("Method") != row.get("Method")]
                    fieldnames = list(dict.fromkeys([*reader.fieldnames, *fieldnames]))
        except Exception:
            rows = []
    rows.append(row)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("exam_name")
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--father-dir", default=str(DEFAULT_FATHER))
    p.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY))
    p.add_argument("--metrics", nargs="+", default=None,
                   choices=["lse", "fid", "fvd", "pairwise", "emotiefflib", "dfer_clip"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--allow-partial", action="store_true")
    # Kept for CLI compatibility; dataset-level evaluator owns cache validity.
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eat-device", default=None, help=argparse.SUPPRESS)
    p.add_argument("--reports-dirname", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fake_root = Path(args.father_dir) / args.exam_name
    if not fake_root.is_dir():
        print(f"ERROR: experiment directory not found: {fake_root}", file=sys.stderr); return 2
    try:
        samples = read_pairs(Path(args.pairs_file), fake_root)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2
    eval_dir = fake_root / "paper_eval"
    manifest = write_manifest(eval_dir / "manifest.csv", samples)
    cmd = [sys.executable, str(PAPER_EVALUATOR), "--manifest", str(manifest),
           "--method", args.exam_name, "--output-dir", str(eval_dir),
           "--device", args.device, "--timeout", str(args.timeout)]
    if args.metrics:
        cmd += ["--metrics", *args.metrics]
    if args.allow_partial:
        cmd.append("--allow-partial")
    rc = subprocess.call(cmd, cwd=str(THIS_DIR))
    table = eval_dir / "paper_table.csv"
    if table.is_file():
        with table.open(newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f), None)
        if row:
            update_summary(Path(args.summary_csv), row)
            print(f"[ADEF-eval] summary updated: {args.summary_csv}")
    if rc != 0:
        print("[ADEF-eval] incomplete evaluation; summary row (if any) is marked incomplete", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
