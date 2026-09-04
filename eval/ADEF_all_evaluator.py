#!/usr/bin/env python3
"""Run paper protocol v3 over all ADEF experiment directories.

Both ``complete`` and ``partial`` rows are considered finished because partial
rows already contain aggregates over successful samples plus an explicit
failure report. Only ``failed`` rows are automatically retried next run.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from paper_protocol import PROTOCOL_VERSION  # noqa: E402

ADEF_EVALUATOR = THIS_DIR / "ADEF_evaluator.py"
DEFAULT_FATHER = Path("/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake")
DEFAULT_SUMMARY = DEFAULT_FATHER / "summary.csv"


def completed(summary: Path) -> set[str]:
    if not summary.is_file():
        return set()
    try:
        with summary.open(newline="", encoding="utf-8") as f:
            return {
                r.get("Method", "") for r in csv.DictReader(f)
                if r.get("Status") in {"complete", "partial"}
                and r.get("Protocol") == PROTOCOL_VERSION
            }
    except Exception:
        return set()


def _read_status(exam_dir: Path) -> tuple[str | None, int]:
    table = exam_dir / "paper_eval" / "paper_table.csv"
    failures = exam_dir / "paper_eval" / "failed_samples.csv"
    status = None
    if table.is_file():
        try:
            with table.open(newline="", encoding="utf-8") as f:
                row = next(csv.DictReader(f), None)
            status = row.get("Status") if row else None
        except Exception:
            status = None
    n_fail = 0
    if failures.is_file():
        try:
            with failures.open(newline="", encoding="utf-8") as f:
                n_fail = sum(1 for _ in csv.DictReader(f))
        except Exception:
            n_fail = 0
    return status, n_fail


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--father-dir", default=str(DEFAULT_FATHER))
    p.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY))
    p.add_argument("--pairs-file", required=True)
    p.add_argument("--metrics", nargs="+", default=None,
                   choices=["lse", "fid", "fvd", "pairwise", "emotiefflib", "dfer_clip"])
    p.add_argument("--device", default=None)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--limit", type=int)
    p.add_argument("--include-done", action="store_true")
    p.add_argument("--allow-partial", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--eat-device", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    father = Path(args.father_dir)
    if not father.is_dir():
        print(f"ERROR: father directory not found: {father}", file=sys.stderr)
        return 2
    exams = sorted(p.name for p in father.iterdir() if p.is_dir() and not p.name.startswith("."))
    done = completed(Path(args.summary_csv))
    todo = exams if args.include_done else [e for e in exams if e not in done]
    if args.limit is not None:
        todo = todo[:max(0, args.limit)]
    print(f"[ADEF-all] protocol={PROTOCOL_VERSION} exams={len(exams)} done={len(done)} todo={len(todo)}")

    rc_total = 0
    n_complete = n_partial = n_failed = 0
    for i, exam in enumerate(todo, 1):
        cmd = [sys.executable, str(ADEF_EVALUATOR), exam,
               "--father-dir", str(father), "--summary-csv", args.summary_csv,
               "--pairs-file", args.pairs_file]
        if args.metrics:
            cmd += ["--metrics", *args.metrics]
        if args.device:
            cmd += ["--device", args.device]
        if args.timeout:
            cmd += ["--timeout", str(args.timeout)]
        print(f"[{i}/{len(todo)}] {exam}")
        if args.dry_run:
            print("  $ " + " ".join(cmd))
            continue
        rc = subprocess.call(cmd, cwd=str(THIS_DIR))
        status, n_fail = _read_status(father / exam)
        if rc != 0 or status == "failed" or status is None:
            n_failed += 1
            rc_total = rc_total or (rc or 2)
            print(f"  -> FAILED rc={rc} failures={n_fail}")
        elif status == "partial":
            n_partial += 1
            print(f"  -> PARTIAL failures={n_fail} (successful samples retained)")
        else:
            n_complete += 1
            print("  -> COMPLETE")
    if not args.dry_run:
        print(f"[ADEF-all] complete={n_complete} partial={n_partial} failed={n_failed}")
    return rc_total


if __name__ == "__main__":
    raise SystemExit(main())
