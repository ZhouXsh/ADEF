#!/usr/bin/env python3
"""Run paper-grade ADEF evaluation over all experiment subdirectories.

Only rows with ``Status=complete`` and the current paper protocol are treated
as finished.  Failed/incomplete experiments are retried on the next run.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ADEF_EVALUATOR = THIS_DIR / "ADEF_evaluator.py"
DEFAULT_FATHER = Path("/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake")
DEFAULT_SUMMARY = DEFAULT_FATHER / "summary.csv"


def completed(summary: Path) -> set[str]:
    if not summary.is_file():
        return set()
    try:
        with summary.open(newline="", encoding="utf-8") as f:
            return {r.get("Method", "") for r in csv.DictReader(f)
                    if r.get("Status") == "complete" and r.get("Protocol", "").startswith("ADEF-paper-eval-v2")}
    except Exception:
        return set()


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--father-dir", default=str(DEFAULT_FATHER))
    p.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY))
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--metrics", nargs="+", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--limit", type=int)
    p.add_argument("--include-done", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--eat-device", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    father = Path(args.father_dir)
    if not father.is_dir():
        print(f"ERROR: father directory not found: {father}", file=sys.stderr); return 2
    # paper_eval itself is an output folder inside each experiment, never an exam.
    exams = sorted(p.name for p in father.iterdir() if p.is_dir() and not p.name.startswith("."))
    done = completed(Path(args.summary_csv))
    todo = exams if args.include_done else [e for e in exams if e not in done]
    if args.limit is not None:
        todo = todo[:max(0, args.limit)]
    print(f"[ADEF-all] exams={len(exams)} complete={len(done)} todo={len(todo)}")
    rc_total = 0
    for i, exam in enumerate(todo, 1):
        cmd = [sys.executable, str(ADEF_EVALUATOR), exam,
               "--father-dir", str(father), "--summary-csv", args.summary_csv]
        if args.pairs_file: cmd += ["--pairs-file", args.pairs_file]
        if args.metrics: cmd += ["--metrics", *args.metrics]
        if args.device: cmd += ["--device", args.device]
        if args.timeout: cmd += ["--timeout", str(args.timeout)]
        if args.allow_partial: cmd += ["--allow-partial"]
        print(f"[{i}/{len(todo)}] {exam}")
        if args.dry_run:
            print("  $ " + " ".join(cmd)); continue
        rc = subprocess.call(cmd, cwd=str(THIS_DIR))
        rc_total = rc_total or rc
        print(f"  -> {'OK' if rc == 0 else f'FAIL rc={rc}'}")
    return rc_total


if __name__ == "__main__":
    raise SystemExit(main())
