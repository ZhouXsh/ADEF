#!/usr/bin/env python3
"""Evaluate one ADEF experiment directory with paper protocol v3.

Pairs file: ``fake_filename,gt_fullpath[,emotion]``. Missing individual files
are recorded as upstream sample failures and the remaining valid samples are
evaluated; they no longer abort the entire experiment.
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


def read_pairs(path: Path, fake_root: Path) -> tuple[list[Sample], list[dict], int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    samples: list[Sample] = []
    failures: list[dict] = []
    expected = 0
    seen_names = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        expected += 1
        parts = [x.strip() for x in text.split(",")]
        if len(parts) not in (2, 3):
            raise ValueError(f"{path}:{lineno}: expected fake,gt[,emotion]")
        fake = fake_root / parts[0]
        gt = Path(parts[1])
        name = Path(parts[0]).stem
        if name in seen_names:
            raise ValueError(f"{path}:{lineno}: duplicate fake/sample name: {name}")
        seen_names.add(name)
        emo = canonical_emotion(parts[2]) if len(parts) == 3 and parts[2] else infer_emotion(gt)
        missing = []
        if not fake.is_file():
            missing.append(f"fake not found: {fake}")
        if not gt.is_file():
            missing.append(f"gt not found: {gt}")
        if missing:
            failures.append({
                "stage": "input", "name": name, "fake": str(fake), "gt": str(gt),
                "error": "; ".join(missing),
            })
            continue
        samples.append(Sample(name=name, fake=str(fake), gt=str(gt), emotion=emo))
    if expected == 0:
        raise ValueError(f"no usable rows in {path}")
    return samples, failures, expected


def update_summary(summary: Path, row: dict):
    rows = []
    fieldnames = list(row.keys())
    if summary.is_file():
        try:
            with summary.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
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


def _clear_outputs(eval_dir: Path) -> None:
    for name in ("paper_table.csv", "paper_metrics.json", "per_video.csv", "failed_samples.csv"):
        p = eval_dir / name
        if p.is_file() or p.is_symlink():
            p.unlink()


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
    p.add_argument("--allow-partial", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--eat-device", default=None, help=argparse.SUPPRESS)
    p.add_argument("--reports-dirname", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fake_root = Path(args.father_dir) / args.exam_name
    if not fake_root.is_dir():
        print(f"ERROR: experiment directory not found: {fake_root}", file=sys.stderr)
        return 2

    eval_dir = fake_root / "paper_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _clear_outputs(eval_dir)
    try:
        samples, upstream_failures, expected_n = read_pairs(Path(args.pairs_file), fake_root)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    upstream_path = eval_dir / "upstream_failures.json"
    upstream_path.write_text(json.dumps({"failures": upstream_failures}, indent=2, ensure_ascii=False), encoding="utf-8")
    if not samples:
        print(f"ERROR: all {expected_n} requested samples are unavailable; see {upstream_path}", file=sys.stderr)
        return 2

    manifest = write_manifest(eval_dir / "manifest.csv", samples)
    cmd = [sys.executable, str(PAPER_EVALUATOR), "--manifest", str(manifest),
           "--method", args.exam_name, "--output-dir", str(eval_dir),
           "--device", args.device, "--timeout", str(args.timeout),
           "--expected-n", str(expected_n), "--upstream-failures", str(upstream_path)]
    if args.metrics:
        cmd += ["--metrics", *args.metrics]
    rc = subprocess.call(cmd, cwd=str(THIS_DIR))

    table = eval_dir / "paper_table.csv"
    row = None
    if table.is_file():
        with table.open(newline="", encoding="utf-8") as f:
            row = next(csv.DictReader(f), None)
        if row:
            update_summary(Path(args.summary_csv), row)
            print(f"[ADEF-eval] summary updated: {args.summary_csv}")
            print(f"[ADEF-eval] status={row.get('Status')} evaluated={row.get('Evaluated-N')}/{row.get('N')}")
    failed = eval_dir / "failed_samples.csv"
    if failed.is_file() and failed.stat().st_size > 0:
        print(f"[ADEF-eval] failed sample report: {failed}", file=sys.stderr)
    if rc != 0:
        print("[ADEF-eval] evaluation failed: at least one requested metric has no usable aggregate", file=sys.stderr)
    elif row and row.get("Status") == "partial":
        print("[ADEF-eval] partial evaluation: table values use successful samples only", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
