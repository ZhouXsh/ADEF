#!/usr/bin/env python3
"""Compatibility single-pair evaluator.

For publication runs use ``paper_evaluator.py``.  FID and FVD are
*dataset-level* Fréchet metrics and are deliberately rejected here; computing
them for one pair and averaging later is not mathematically equivalent to the
standard metric.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from paper_protocol import Sample, infer_emotion, write_manifest  # noqa: E402

PAPER = THIS_DIR / "paper_evaluator.py"
SAFE_SINGLE = {"lse", "pairwise", "emotiefflib", "dfer_clip"}
DISTRIBUTION = {"fid", "fvd"}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--fake", required=True)
    p.add_argument("--gt")
    p.add_argument("--name", default="single_pair")
    p.add_argument("--output", required=True)
    p.add_argument("--metrics", nargs="+", default=["all"])
    p.add_argument("--skip", nargs="+", default=[])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timeout", type=int, default=7200)
    # Compatibility with the historical CLI.
    p.add_argument("--eat-device", default=None, help=argparse.SUPPRESS)
    p.add_argument("--workdir", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    requested = set(SAFE_SINGLE | DISTRIBUTION) if "all" in args.metrics else set(args.metrics)
    requested -= set(args.skip)
    invalid = requested & DISTRIBUTION
    if invalid:
        msg = ("FID/FVD are dataset-level metrics and cannot be produced by the single-pair "
               "unified_evaluator. Build a manifest and run paper_evaluator.py instead. "
               f"Rejected: {sorted(invalid)}")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"ok": False, "error": msg}, indent=2), encoding="utf-8")
        print("ERROR: " + msg, file=sys.stderr)
        return 2
    metrics = sorted(requested & SAFE_SINGLE)
    if not metrics:
        print("ERROR: no supported single-pair metrics requested", file=sys.stderr); return 2
    if any(x in metrics for x in ("pairwise",)) and not args.gt:
        print("ERROR: pairwise metrics require --gt", file=sys.stderr); return 2
    gt = args.gt or args.fake
    with tempfile.TemporaryDirectory(prefix="adef_unified_compat_") as td:
        root = Path(td)
        manifest = write_manifest(root / "manifest.csv", [Sample(
            name=args.name, fake=args.fake, gt=gt, emotion=infer_emotion(args.gt or args.fake)
        )])
        cmd = [sys.executable, str(PAPER), "--manifest", str(manifest), "--method", args.name,
               "--output-dir", str(root / "out"), "--metrics", *metrics,
               "--device", args.device, "--timeout", str(args.timeout)]
        rc = subprocess.call(cmd, cwd=str(THIS_DIR))
        report_path = root / "out" / "paper_metrics.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {
            "status": "incomplete", "errors": [f"paper_evaluator returned {rc}"]
        }
        out = {
            "compatibility_wrapper": True,
            "publication_entrypoint": "paper_evaluator.py",
            "fake": args.fake, "gt": args.gt, "name": args.name,
            "status": report.get("status"), "errors": report.get("errors", []),
            "table_row": report.get("table_row"), "details": report.get("details", {}),
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
